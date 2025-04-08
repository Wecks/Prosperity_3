from datamodel import OrderDepth, UserId, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import List, Dict, Any
import statistics
import math
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        # Initialize trader state
        self.product_state = {}

        # Track open positions with their entry prices for profit targeting
        self.open_positions = {}

        # Parameters for envelope strategy
        self.history_len = 150  # Longer price history for better trend detection
        self.price_update_weight = 0.15  # Slightly more responsive to recent price changes
        self.last_timestamp = 0

        # Envelope configuration
        self.envelopes = [
            {"level": 0.5, "weight": 1.0, "profit_target": 0.003},    # 0.5 std dev, 0.3% profit target
            {"level": 1.0, "weight": 2.0, "profit_target": 0.006},    # 1.0 std dev, 0.6% profit target
            {"level": 1.5, "weight": 3.0, "profit_target": 0.009},    # 1.5 std dev, 0.9% profit target
            {"level": 2.0, "weight": 5.0, "profit_target": 0.015}     # 2.0 std dev, 1.5% profit target
        ]

        # Risk management
        self.default_max_position = 25  # Base position size

    def calculate_mid_price(self, buy_orders, sell_orders):
        """Calculate the mid price from the order book"""
        if len(buy_orders) > 0 and len(sell_orders) > 0:
            best_bid = max(buy_orders.keys())
            best_ask = min(sell_orders.keys())
            return (best_bid + best_ask) / 2
        elif len(buy_orders) > 0:
            return max(buy_orders.keys())
        elif len(sell_orders) > 0:
            return min(sell_orders.keys())
        else:
            return None

    def initialize_product(self, product, mid_price):
        """Initialize product state if we haven't seen this product before"""
        if mid_price is not None:
            self.product_state[product] = {
                "fair_price": mid_price,           # Start with current mid price
                "std_dev": mid_price * 0.005,      # Initial volatility estimate (0.5% of price)
                "max_position": self.default_max_position,
                "price_history": [mid_price],      # Initialize price history
                "volume_history": [],              # Track trading volumes
                "moving_average": mid_price        # Initialize MA with current price
            }

            # Initialize position tracking for this product
            self.open_positions[product] = {
                "long_positions": [],              # List of (price, size) tuples for long positions
                "short_positions": [],             # List of (price, size) tuples for short positions
                "total_position": 0                # Total current position
            }
        else:
            # Fallback if no mid price available
            self.product_state[product] = {
                "fair_price": 0,
                "std_dev": 1,
                "max_position": self.default_max_position,
                "price_history": [],
                "volume_history": [],
                "moving_average": 0
            }
            self.open_positions[product] = {
                "long_positions": [],
                "short_positions": [],
                "total_position": 0
            }

    def update_product_state(self, product, mid_price, order_depth, position):
        """Update our understanding of a product based on new market data"""
        # Initialize product state if we haven't seen it before
        if product not in self.product_state:
            self.initialize_product(product, mid_price)
            return

        # Update position tracking
        self.open_positions[product]["total_position"] = position

        # Update price history if we have a new mid price
        if mid_price is not None:
            self.product_state[product]["price_history"].append(mid_price)

            # Keep history at a manageable size
            if len(self.product_state[product]["price_history"]) > self.history_len:
                self.product_state[product]["price_history"] = self.product_state[product]["price_history"][-self.history_len:]

            # Calculate trading volumes
            buy_volume = sum(abs(vol) for vol in order_depth.buy_orders.values())
            sell_volume = sum(abs(vol) for vol in order_depth.sell_orders.values())
            self.product_state[product]["volume_history"].append((buy_volume, sell_volume))

            # Trim volume history to match price history
            if len(self.product_state[product]["volume_history"]) > self.history_len:
                self.product_state[product]["volume_history"] = self.product_state[product]["volume_history"][-self.history_len:]

            # Update moving average - weighted more toward recent prices
            history = self.product_state[product]["price_history"]
            if len(history) > 10:
                # Exponential moving average
                weights = [0.9 ** i for i in range(min(30, len(history)))]
                weighted_sum = sum(p * w for p, w in zip(reversed(history[-30:]), weights))
                self.product_state[product]["moving_average"] = weighted_sum / sum(weights)

            # Update volatility estimate based on recent price changes
            if len(self.product_state[product]["price_history"]) > 10:
                recent_prices = self.product_state[product]["price_history"][-20:]
                if len(recent_prices) > 1:  # Need at least 2 prices for standard deviation
                    self.product_state[product]["std_dev"] = max(
                        mid_price * 0.001,  # Minimum volatility of 0.1%
                        statistics.stdev(recent_prices)  # Use full volatility for envelope calculation
                    )

            # Adjust max position based on observed volumes and volatility
            if len(self.product_state[product]["volume_history"]) > 10:
                avg_vol = sum(b + s for b, s in self.product_state[product]["volume_history"][-10:]) / 10
                # Scale max position with market volume, but cap between 15 and 120
                volume_based_max = min(120, max(15, int(avg_vol * 0.35)))

                # Also consider price and volatility for position sizing
                price_scale = 100000 / max(1, mid_price)  # Lower priced assets can have larger positions
                vol_scale = 0.01 / max(0.001, self.product_state[product]["std_dev"] / mid_price)  # Lower volatility allows larger positions

                adjusted_max = int(volume_based_max * min(2.0, price_scale * vol_scale))
                self.product_state[product]["max_position"] = min(150, max(15, adjusted_max))

    def check_take_profit_orders(self, product, mid_price):
        """Check if any open positions should be closed based on profit targets"""
        orders = []
        if product not in self.open_positions:
            return orders

        # Check long positions for take profit
        new_long_positions = []
        for entry_price, size in self.open_positions[product]["long_positions"]:
            # Determine which envelope this position belongs to
            price_diff = abs(entry_price - self.product_state[product]["moving_average"])
            std_dev = self.product_state[product]["std_dev"]

            # Find the appropriate envelope level and profit target
            profit_target = 0.003  # Default profit target
            for env in self.envelopes:
                if price_diff >= env["level"] * std_dev:
                    profit_target = env["profit_target"]
                else:
                    break

            # Check if profit target reached
            if mid_price >= entry_price * (1 + profit_target):
                # Take profit on this position
                logger.print(f"TAKE PROFIT LONG {product}: {size}x at {mid_price} (Entry: {entry_price}, Target: {profit_target*100}%)")
                orders.append(Order(product, int(mid_price), -size))
            else:
                # Keep position open
                new_long_positions.append((entry_price, size))

        # Update long positions list
        self.open_positions[product]["long_positions"] = new_long_positions

        # Check short positions for take profit
        new_short_positions = []
        for entry_price, size in self.open_positions[product]["short_positions"]:
            # Determine which envelope this position belongs to
            price_diff = abs(entry_price - self.product_state[product]["moving_average"])
            std_dev = self.product_state[product]["std_dev"]

            # Find the appropriate envelope level and profit target
            profit_target = 0.003  # Default profit target
            for env in self.envelopes:
                if price_diff >= env["level"] * std_dev:
                    profit_target = env["profit_target"]
                else:
                    break

            # Check if profit target reached
            if mid_price <= entry_price * (1 - profit_target):
                # Take profit on this position
                logger.print(f"TAKE PROFIT SHORT {product}: {size}x at {mid_price} (Entry: {entry_price}, Target: {profit_target*100}%)")
                orders.append(Order(product, int(mid_price), size))
            else:
                # Keep position open
                new_short_positions.append((entry_price, size))

        # Update short positions list
        self.open_positions[product]["short_positions"] = new_short_positions

        return orders

    def run(self, state: TradingState):
        result = {}
        trader_data = {}

        # Check if this is a new timestamp
        timestamp = state.timestamp
        new_interval = timestamp != self.last_timestamp
        self.last_timestamp = timestamp

        for product in state.order_depths:
            # Initialize orders list for this product
            orders: List[Order] = []

            # Get order depth
            order_depth: OrderDepth = state.order_depths[product]

            # Get current position or initialize to zero
            position = 0
            if product in state.position:
                position = state.position[product]

            # Update position tracking
            if product in self.open_positions:
                self.open_positions[product]["total_position"] = position
            else:
                self.open_positions[product] = {
                    "long_positions": [],
                    "short_positions": [],
                    "total_position": position
                }

            # Calculate the current mid price
            mid_price = self.calculate_mid_price(order_depth.buy_orders, order_depth.sell_orders)
            if mid_price is None:
                continue  # Skip if no valid price data

            # Update our understanding of the product
            self.update_product_state(product, mid_price, order_depth, position)

            # Check for take profit opportunities
            take_profit_orders = self.check_take_profit_orders(product, mid_price)
            orders.extend(take_profit_orders)

            # Get moving average and standard deviation for envelope calculation
            moving_average = self.product_state[product]["moving_average"]
            std_dev = self.product_state[product]["std_dev"]
            max_position = self.product_state[product]["max_position"]

            # Log state information using logger
            logger.print(f"{product} - Position: {position}/{max_position}, MA: {moving_average}, Mid: {mid_price}, StdDev: {std_dev}")

            # Calculate available position capacity (how many more units we can buy/sell)
            remaining_buy_capacity = max_position - position
            remaining_sell_capacity = max_position + position

            # Process sell orders (potential buying opportunities)
            if len(order_depth.sell_orders) > 0 and remaining_buy_capacity > 0:
                # Sort sell orders by price (ascending)
                ask_prices = sorted(order_depth.sell_orders.keys())

                for ask_price in ask_prices:
                    ask_volume = order_depth.sell_orders[ask_price]

                    # Calculate how deep into the envelope we are
                    price_deviation = moving_average - ask_price
                    deviation_in_std = price_deviation / std_dev if std_dev > 0 else 0

                    # Skip if price is above moving average (we only buy below MA)
                    if deviation_in_std <= 0:
                        continue

                    # Determine which envelope level we're at
                    selected_envelope = None
                    for env in reversed(self.envelopes):  # Check deepest envelopes first
                        if deviation_in_std >= env["level"]:
                            selected_envelope = env
                            break

                    # If we found an envelope match, place a buy order
                    if selected_envelope:
                        # Calculate trade size based on envelope weight
                        base_size = min(5, max(1, int(max_position / 10)))  # Base position size
                        trade_size = int(base_size * selected_envelope["weight"])

                        # Don't exceed ask volume or remaining capacity
                        buy_volume = min(abs(ask_volume), trade_size, remaining_buy_capacity)

                        if buy_volume > 0:
                            logger.print(f"BUY {product}: {buy_volume}x {int(ask_price)} (MA: {moving_average}, Env: {selected_envelope['level']}σ)")
                            orders.append(Order(product, int(ask_price), buy_volume))

                            # Update position tracking
                            self.open_positions[product]["long_positions"].append((int(ask_price), buy_volume))
                            remaining_buy_capacity -= buy_volume

                            # Stop if we've used our capacity
                            if remaining_buy_capacity <= 0:
                                break

            # Process buy orders (potential selling opportunities)
            if len(order_depth.buy_orders) > 0 and remaining_sell_capacity > 0:
                # Sort buy orders by price (descending)
                bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)

                for bid_price in bid_prices:
                    bid_volume = order_depth.buy_orders[bid_price]

                    # Calculate how deep into the envelope we are
                    price_deviation = bid_price - moving_average
                    deviation_in_std = price_deviation / std_dev if std_dev > 0 else 0

                    # Skip if price is below moving average (we only sell above MA)
                    if deviation_in_std <= 0:
                        continue

                    # Determine which envelope level we're at
                    selected_envelope = None
                    for env in reversed(self.envelopes):  # Check deepest envelopes first
                        if deviation_in_std >= env["level"]:
                            selected_envelope = env
                            break

                    # If we found an envelope match, place a sell order
                    if selected_envelope:
                        # Calculate trade size based on envelope weight
                        base_size = min(5, max(1, int(max_position / 10)))  # Base position size
                        trade_size = int(base_size * selected_envelope["weight"])

                        # Don't exceed bid volume or remaining sell capacity
                        sell_volume = min(bid_volume, trade_size, remaining_sell_capacity)

                        if sell_volume > 0:
                            logger.print(f"SELL {product}: {sell_volume}x {int(bid_price)} (MA: {moving_average}, Env: {selected_envelope['level']}σ)")
                            orders.append(Order(product, int(bid_price), -sell_volume))

                            # Update position tracking
                            self.open_positions[product]["short_positions"].append((int(bid_price), sell_volume))
                            remaining_sell_capacity -= sell_volume

                            # Stop if we've used our capacity
                            if remaining_sell_capacity <= 0:
                                break

            # Add orders to result
            result[product] = orders

        # Store open positions and other state data
        trader_data_str = json.dumps({
            "open_positions": {k: {
                "long_count": len(v["long_positions"]),
                "short_count": len(v["short_positions"])
            } for k, v in self.open_positions.items()}
        })

        # Use logger to flush the output
        logger.flush(state, result, 0, trader_data_str)
        return result, 0, trader_data_str