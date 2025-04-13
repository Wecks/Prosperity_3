from datamodel import OrderDepth, UserId, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
import json
from typing import Dict, List, Any, Tuple
import statistics
import math

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

class KelpStrategy:
    def __init__(self):
        # Initialize strategy parameters
        self.mean_price = 2034.5  # Base mean price of KELP observed from logs
        self.buy_threshold = 2033.5  # Buy when price is at or below this
        self.sell_threshold = 2035.5  # Sell when price is at or above this
        self.instrument = "KELP"
        self.base_position_size = 10  # Base size for each trade
        self.max_position = 100  # Maximum position size (positive or negative)
        self.max_trade_size = 25  # Maximum size per individual trade
        self.pyramid_step = 0.5  # Price steps between pyramid levels
        self.pyramid_levels = 5  # Maximum number of pyramid levels
        self.pyramid_multiplier = [1.0, 1.5, 2.0, 2.5, 3.0]  # Multiplier for each pyramid level
        self.position_scaling = 1.5  # Scaling factor for pyramiding
        self.reversal_multiplier = 1.5  # Multiplier for position reversal

        # Position tracking
        self.positions = {
            "longs": [],  # [(price, size), ...]
            "shorts": [],  # [(price, size), ...]
            "total": 0,
            "avg_long_price": 0,
            "avg_short_price": 0,
            "long_total_size": 0,
            "short_total_size": 0
        }

        # Performance metrics
        self.trades_executed = 0
        self.profitable_trades = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0

        # Dynamic threshold parameters
        self.volatility = 0.5  # Estimated volatility
        self.min_threshold_distance = 0.5  # Minimum distance between mean and thresholds
        self.adaptive_factor = 0.05  # How quickly to adjust to market changes
        self.recent_prices = []  # Store recent prices to calculate volatility
        self.price_history = []  # Longer price history for trend analysis
        self.max_history_length = 100

    def update_parameters(self, mid_price):
        """
        Dynamically update strategy parameters based on market conditions
        """
        # Store recent prices for volatility calculation
        self.recent_prices.append(mid_price)
        if len(self.recent_prices) > 10:
            self.recent_prices.pop(0)  # Keep only last 10 prices

        # Calculate short-term volatility
        if len(self.recent_prices) >= 3:
            price_range = max(self.recent_prices) - min(self.recent_prices)
            self.volatility = max(0.5, price_range * 0.5)  # Scale and apply minimum

        # Adjust mean price more aggressively toward current mid price
        self.mean_price = 0.90 * self.mean_price + 0.10 * mid_price

        # Calculate dynamic threshold distance based on volatility
        threshold_distance = max(self.min_threshold_distance, self.volatility * 0.6)

        # Set thresholds closer to mean for more trades
        self.buy_threshold = self.mean_price - threshold_distance
        self.sell_threshold = self.mean_price + threshold_distance

    def update_position_stats(self):
        """Update average position prices and total sizes"""
        # Calculate average long position price and total size
        if self.positions["longs"]:
            total_size = sum(size for _, size in self.positions["longs"])
            if total_size > 0:
                avg_price = sum(price * size for price, size in self.positions["longs"]) / total_size
                self.positions["avg_long_price"] = avg_price
                self.positions["long_total_size"] = total_size
            else:
                self.positions["avg_long_price"] = 0
                self.positions["long_total_size"] = 0
        else:
            self.positions["avg_long_price"] = 0
            self.positions["long_total_size"] = 0

        # Calculate average short position price and total size
        if self.positions["shorts"]:
            total_size = sum(size for _, size in self.positions["shorts"])
            if total_size > 0:
                avg_price = sum(price * size for price, size in self.positions["shorts"]) / total_size
                self.positions["avg_short_price"] = avg_price
                self.positions["short_total_size"] = total_size
            else:
                self.positions["avg_short_price"] = 0
                self.positions["short_total_size"] = 0
        else:
            self.positions["avg_short_price"] = 0
            self.positions["short_total_size"] = 0

    def determine_pyramid_level(self, price, is_buy):
        """Determine which pyramid level we're at based on price deviation from threshold"""
        if is_buy:
            # For buys, we measure how far below buy_threshold we are
            price_deviation = self.buy_threshold - price
        else:
            # For sells, we measure how far above sell_threshold we are
            price_deviation = price - self.sell_threshold

        # Calculate pyramid level based on price deviation
        level = min(int(price_deviation / self.pyramid_step), self.pyramid_levels - 1)
        return max(0, level)  # Ensure level is at least 0

    def determine_position_size(self, price, current_position, is_buy):
        """Calculate the appropriate position size based on pyramid level and current position"""
        # Determine pyramid level
        pyramid_level = self.determine_pyramid_level(price, is_buy)

        # Base size scaled by pyramid multiplier
        size = int(self.base_position_size * self.pyramid_multiplier[pyramid_level])

        # Scale up position size for reversals
        if is_buy and self.positions["short_total_size"] > 0:
            # If we have shorts and we're buying, scale up to close shorts plus take new position
            reversal_size = int(self.positions["short_total_size"] * self.reversal_multiplier)
            size = max(size, reversal_size)
        elif not is_buy and self.positions["long_total_size"] > 0:
            # If we have longs and we're selling, scale up to close longs plus take new position
            reversal_size = int(self.positions["long_total_size"] * self.reversal_multiplier)
            size = max(size, reversal_size)

        # Respect maximum position limit
        available_capacity = self.max_position - abs(current_position)

        return min(size, available_capacity)

    def run(self, state: TradingState) -> List[Order]:
        """
        Main strategy execution method with rainforest-style continuous trading and dynamic thresholds
        """
        # Initialize orders list
        orders = []

        # Skip if our instrument is not in the order depths
        if self.instrument not in state.order_depths:
            return orders

        # Get order depth and current position
        order_depth = state.order_depths[self.instrument]
        position = state.position.get(self.instrument, 0)
        self.positions["total"] = position

        # Calculate mid price
        mid_price = self.calculate_mid_price(order_depth)
        if mid_price is None:
            return orders

        # Update price history and parameters with dynamic thresholds
        self.price_history.append(mid_price)
        if len(self.price_history) > self.max_history_length:
            self.price_history.pop(0)

        self.update_parameters(mid_price)
        self.update_position_stats()

        # Log current state
        logger.print(f"{self.instrument} - Position: {position}/{self.max_position}, Mean: {self.mean_price:.2f}, Mid: {mid_price}")
        logger.print(f"Buy threshold: {self.buy_threshold}, Sell threshold: {self.sell_threshold}")
        if self.positions["long_total_size"] > 0:
            logger.print(f"Long positions: {self.positions['long_total_size']}x @ avg {self.positions['avg_long_price']:.2f}")
        if self.positions["short_total_size"] > 0:
            logger.print(f"Short positions: {self.positions['short_total_size']}x @ avg {self.positions['avg_short_price']:.2f}")

        # Calculate remaining position capacity
        remaining_buy_capacity = self.max_position - position
        remaining_sell_capacity = self.max_position + position

        # Process sell orders in the book (opportunities to buy) - RAINFOREST STYLE
        if len(order_depth.sell_orders) > 0 and remaining_buy_capacity > 0:
            # Sort sell orders by price (ascending)
            ask_prices = sorted(order_depth.sell_orders.keys())

            for ask_price in ask_prices:
                # Only buy if price is at or below our threshold - CONTINUOUS BUYING
                if ask_price <= self.buy_threshold:
                    ask_volume = abs(order_depth.sell_orders[ask_price])

                    # Calculate position size based on pyramid level
                    buy_size = self.determine_position_size(ask_price, position, True)
                    buy_size = min(buy_size, ask_volume, remaining_buy_capacity)

                    if buy_size > 0:
                        # Check if we're closing shorts or opening new longs
                        if self.positions["short_total_size"] > 0:
                            logger.print(f"REVERSAL BUY: {buy_size}x at {ask_price} (Closing shorts + new long)")
                        else:
                            pyramid_level = self.determine_pyramid_level(ask_price, True)
                            logger.print(f"BUY: {buy_size}x at {ask_price} (Pyramid level: {pyramid_level})")

                        # Place the order
                        orders.append(Order(self.instrument, int(ask_price), buy_size))

                        # Track position
                        self.positions["longs"].append((ask_price, buy_size))
                        remaining_buy_capacity -= buy_size
                        position += buy_size
                        self.trades_executed += 1

                        # Stop if we've reached capacity
                        if remaining_buy_capacity <= 0:
                            break

        # Process buy orders in the book (opportunities to sell) - RAINFOREST STYLE
        if len(order_depth.buy_orders) > 0 and remaining_sell_capacity > 0:
            # Sort buy orders by price (descending)
            bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)

            for bid_price in bid_prices:
                # Only sell if price is at or above our threshold - CONTINUOUS SELLING
                if bid_price >= self.sell_threshold:
                    bid_volume = order_depth.buy_orders[bid_price]

                    # Calculate position size based on pyramid level
                    sell_size = self.determine_position_size(bid_price, position, False)
                    sell_size = min(sell_size, abs(bid_volume), remaining_sell_capacity)

                    if sell_size > 0:
                        # Check if we're closing longs or opening new shorts
                        if self.positions["long_total_size"] > 0:
                            logger.print(f"REVERSAL SELL: {sell_size}x at {bid_price} (Closing longs + new short)")
                        else:
                            pyramid_level = self.determine_pyramid_level(bid_price, False)
                            logger.print(f"SELL: {sell_size}x at {bid_price} (Pyramid level: {pyramid_level})")

                        # Place the order
                        orders.append(Order(self.instrument, int(bid_price), -sell_size))

                        # Track position
                        self.positions["shorts"].append((bid_price, sell_size))
                        remaining_sell_capacity -= sell_size
                        position -= sell_size
                        self.trades_executed += 1

                        # Stop if we've reached capacity
                        if remaining_sell_capacity <= 0:
                            break

        # Update position stats after trades
        self.update_position_stats()

        return orders

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """
        Calculate the mid price from the order book
        """
        if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        elif len(order_depth.sell_orders) > 0:
            return min(order_depth.sell_orders.keys())
        elif len(order_depth.buy_orders) > 0:
            return max(order_depth.buy_orders.keys())
        return None

    def get_best_bid(self, order_depth: OrderDepth):
        """
        Return the best bid price and volume
        """
        if len(order_depth.buy_orders) == 0:
            return None, 0

        best_bid = max(order_depth.buy_orders.keys())
        return best_bid, order_depth.buy_orders[best_bid]

    def get_best_ask(self, order_depth: OrderDepth):
        """
        Return the best ask price and volume
        """
        if len(order_depth.sell_orders) == 0:
            return None, 0

        best_ask = min(order_depth.sell_orders.keys())
        return best_ask, order_depth.sell_orders[best_ask]

    def serialize_state(self):
        """
        Serialize the current strategy state to JSON
        """
        strategy_state = {
            "mean_price": self.mean_price,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "trades_executed": self.trades_executed,
            "profitable_trades": self.profitable_trades,
            "realized_pnl": self.realized_pnl,
            "positions": self.positions
        }
        return strategy_state


class RainforestStrategy:
    def __init__(self):
        # Target product
        self.target_product = "RAINFOREST_RESIN"

        # Parameters for mean reversion strategy
        self.mean_price = 10000  # Mean price for RAINFOREST_RESIN
        self.buy_threshold = 9999  # Buy when price is at or below this
        self.sell_threshold = 10001  # Sell when price is at or above this

        # Order sizing and risk management
        self.base_position_size = 15
        self.max_position = 100
        self.position_scaling = 2.0  # Increased scaling factor for pyramiding
        self.reversal_multiplier = 2.0  # Multiplier for position reversal

        # Pyramiding parameters
        self.pyramid_step = 2  # Price steps between pyramid levels
        self.pyramid_levels = 5  # Maximum number of pyramid levels
        self.pyramid_multiplier = [1.0, 1.5, 2.0, 3.0, 4.0]  # Multiplier for each pyramid level

        # Position tracking
        self.positions = {
            "longs": [],  # [(price, size), ...]
            "shorts": [],  # [(price, size), ...]
            "total": 0,
            "avg_long_price": 0,
            "avg_short_price": 0,
            "long_total_size": 0,
            "short_total_size": 0
        }

        # Trading statistics and state tracking
        self.trades_executed = 0
        self.profitable_trades = 0
        self.price_history = []
        self.max_history_length = 100
        self.last_timestamp = 0

        # Performance metrics
        self.realized_pnl = 0
        self.unrealized_pnl = 0

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
            return self.mean_price  # Default to mean if no orders

    def update_price_history(self, mid_price):
        """Update price history with new data"""
        if mid_price is not None:
            self.price_history.append(mid_price)

            # Keep history at manageable size
            if len(self.price_history) > self.max_history_length:
                self.price_history = self.price_history[-self.max_history_length:]

            # Calculate updated mean from most recent prices
            if len(self.price_history) >= 30:
                # Use exponentially weighted moving average
                weights = [0.95 ** i for i in range(min(30, len(self.price_history)))]
                weighted_sum = sum(p * w for p, w in zip(reversed(self.price_history[-30:]), weights))
                recent_mean = weighted_sum / sum(weights)

                # Slowly adjust our mean price based on recent data
                adjustment_rate = 0.02
                self.mean_price = (1 - adjustment_rate) * self.mean_price + adjustment_rate * recent_mean

                # Set buy and sell thresholds exactly at 9999 and 10001
                self.buy_threshold = 9999
                self.sell_threshold = 10001

    def update_position_stats(self):
        """Update average position prices and total sizes"""
        # Calculate average long position price and total size
        if self.positions["longs"]:
            total_size = sum(size for _, size in self.positions["longs"])
            if total_size > 0:
                avg_price = sum(price * size for price, size in self.positions["longs"]) / total_size
                self.positions["avg_long_price"] = avg_price
                self.positions["long_total_size"] = total_size
            else:
                self.positions["avg_long_price"] = 0
                self.positions["long_total_size"] = 0
        else:
            self.positions["avg_long_price"] = 0
            self.positions["long_total_size"] = 0

        # Calculate average short position price and total size
        if self.positions["shorts"]:
            total_size = sum(size for _, size in self.positions["shorts"])
            if total_size > 0:
                avg_price = sum(price * size for price, size in self.positions["shorts"]) / total_size
                self.positions["avg_short_price"] = avg_price
                self.positions["short_total_size"] = total_size
            else:
                self.positions["avg_short_price"] = 0
                self.positions["short_total_size"] = 0
        else:
            self.positions["avg_short_price"] = 0
            self.positions["short_total_size"] = 0

    def determine_pyramid_level(self, price, is_buy):
        """Determine which pyramid level we're at based on price deviation from threshold"""
        if is_buy:
            # For buys, we measure how far below buy_threshold we are
            price_deviation = self.buy_threshold - price
        else:
            # For sells, we measure how far above sell_threshold we are
            price_deviation = price - self.sell_threshold

        # Calculate pyramid level based on price deviation
        level = min(int(price_deviation / self.pyramid_step), self.pyramid_levels - 1)
        return max(0, level)  # Ensure level is at least 0

    def determine_position_size(self, price, current_position, is_buy):
        """Calculate the appropriate position size based on pyramid level and current position"""
        # Determine pyramid level
        pyramid_level = self.determine_pyramid_level(price, is_buy)

        # Base size scaled by pyramid multiplier
        size = int(self.base_position_size * self.pyramid_multiplier[pyramid_level])

        # Scale up position size for reversals
        if is_buy and self.positions["short_total_size"] > 0:
            # If we have shorts and we're buying, scale up to close shorts plus take new position
            reversal_size = int(self.positions["short_total_size"] * self.reversal_multiplier)
            size = max(size, reversal_size)
        elif not is_buy and self.positions["long_total_size"] > 0:
            # If we have longs and we're selling, scale up to close longs plus take new position
            reversal_size = int(self.positions["long_total_size"] * self.reversal_multiplier)
            size = max(size, reversal_size)

        # Respect maximum position limit
        available_capacity = self.max_position - abs(current_position)

        return min(size, available_capacity)

    def run(self, state: TradingState) -> List[Order]:
        orders = []

        # Only process if RAINFOREST_RESIN is in the order book
        if self.target_product not in state.order_depths:
            return orders

        # Get order depth for our target product
        order_depth = state.order_depths[self.target_product]

        # Get current position
        current_position = 0
        if self.target_product in state.position:
            current_position = state.position[self.target_product]
        self.positions["total"] = current_position

        # Calculate mid price
        mid_price = self.calculate_mid_price(order_depth.buy_orders, order_depth.sell_orders)

        # Update price history and statistical measures
        self.update_price_history(mid_price)

        # Update position stats
        self.update_position_stats()

        # Log current state
        logger.print(f"{self.target_product} - Position: {current_position}/{self.max_position}, Mean: {self.mean_price:.2f}, Mid: {mid_price}")
        logger.print(f"Buy threshold: {self.buy_threshold}, Sell threshold: {self.sell_threshold}")
        if self.positions["long_total_size"] > 0:
            logger.print(f"Long positions: {self.positions['long_total_size']}x @ avg {self.positions['avg_long_price']:.2f}")
        if self.positions["short_total_size"] > 0:
            logger.print(f"Short positions: {self.positions['short_total_size']}x @ avg {self.positions['avg_short_price']:.2f}")

        # Calculate remaining position capacity
        remaining_buy_capacity = self.max_position - current_position
        remaining_sell_capacity = self.max_position + current_position

        # Process sell orders in the book (opportunities to buy)
        if len(order_depth.sell_orders) > 0 and remaining_buy_capacity > 0:
            # Sort sell orders by price (ascending)
            ask_prices = sorted(order_depth.sell_orders.keys())

            for ask_price in ask_prices:
                # Only buy if price is at or below our threshold
                if ask_price <= self.buy_threshold:
                    ask_volume = abs(order_depth.sell_orders[ask_price])

                    # Calculate position size based on pyramid level and reversal potential
                    buy_size = self.determine_position_size(ask_price, current_position, True)
                    buy_size = min(ask_volume, buy_size, remaining_buy_capacity)

                    if buy_size > 0:
                        # Check if we're closing shorts or opening new longs
                        if self.positions["short_total_size"] > 0:
                            logger.print(f"REVERSAL BUY: {buy_size}x at {ask_price} (Closing shorts + new long)")
                        else:
                            pyramid_level = self.determine_pyramid_level(ask_price, True)
                            logger.print(f"BUY: {buy_size}x at {ask_price} (Pyramid level: {pyramid_level})")

                        orders.append(Order(self.target_product, int(ask_price), buy_size))

                        # Track position
                        self.positions["longs"].append((ask_price, buy_size))
                        remaining_buy_capacity -= buy_size
                        current_position += buy_size
                        self.trades_executed += 1
                        self.update_position_stats()

                        # Stop if we've reached capacity
                        if remaining_buy_capacity <= 0:
                            break

        # Process buy orders in the book (opportunities to sell)
        if len(order_depth.buy_orders) > 0 and remaining_sell_capacity > 0:
            # Sort buy orders by price (descending)
            bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)

            for bid_price in bid_prices:
                # Only sell if price is at or above our threshold
                if bid_price >= self.sell_threshold:
                    bid_volume = order_depth.buy_orders[bid_price]

                    # Calculate position size based on pyramid level and reversal potential
                    sell_size = self.determine_position_size(bid_price, current_position, False)
                    sell_size = min(bid_volume, sell_size, remaining_sell_capacity)

                    if sell_size > 0:
                        # Check if we're closing longs or opening new shorts
                        if self.positions["long_total_size"] > 0:
                            logger.print(f"REVERSAL SELL: {sell_size}x at {bid_price} (Closing longs + new short)")
                        else:
                            pyramid_level = self.determine_pyramid_level(bid_price, False)
                            logger.print(f"SELL: {sell_size}x at {bid_price} (Pyramid level: {pyramid_level})")

                        orders.append(Order(self.target_product, int(bid_price), -sell_size))

                        # Track position
                        self.positions["shorts"].append((bid_price, sell_size))
                        remaining_sell_capacity -= sell_size
                        current_position -= sell_size
                        self.trades_executed += 1
                        self.update_position_stats()

                        # Stop if we've reached capacity
                        if remaining_sell_capacity <= 0:
                            break

        return orders

    def serialize_state(self):
        """
        Serialize the current strategy state to JSON
        """
        trader_data = {
            "mean_price": self.mean_price,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "trades_executed": self.trades_executed,
            "profitable_trades": self.profitable_trades,
            "realized_pnl": self.realized_pnl,
            "position": {
                "longs": len(self.positions["longs"]),
                "long_total": self.positions["long_total_size"],
                "shorts": len(self.positions["shorts"]),
                "short_total": self.positions["short_total_size"],
                "total": self.positions["total"]
            }
        }
        return trader_data


class SquidInkStrategy:
    def __init__(self):
        self.symbol = "SQUID_INK"
        self.price_history = []
        self.window = 30
        self.max_position = 100
        self.base_step = 2
        self.volatility_threshold = 1.5
        self.hold_threshold = 0.5

    def run(self, state: TradingState) -> List[Order]:
        orders = []

        order_depth = state.order_depths.get(self.symbol, None)
        if not order_depth:
            return orders

        position = state.position.get(self.symbol, 0)

        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
        elif order_depth.buy_orders:
            mid_price = max(order_depth.buy_orders.keys())
        elif order_depth.sell_orders:
            mid_price = min(order_depth.sell_orders.keys())
        else:
            return orders

        self.price_history.append(mid_price)
        if len(self.price_history) > self.window:
            self.price_history.pop(0)

        if len(self.price_history) < self.window:
            return orders

        log_prices = [math.log(p) for p in self.price_history if p > 0]
        mean_log_price = statistics.mean(log_prices)
        std_log_price = statistics.stdev(log_prices)
        current_log_price = math.log(mid_price)

        z_score = (current_log_price - mean_log_price) / std_log_price
        allow_trading = abs(z_score) < self.volatility_threshold

        base_price = statistics.mean(self.price_history)
        buy_threshold = base_price - std_log_price
        sell_threshold = base_price + std_log_price

        # reduce exposure if z_score is very close to 0
        if abs(z_score) < self.hold_threshold and position != 0:
            side = 1 if position < 0 else -1
            orders.append(Order(self.symbol, int(mid_price), side * min(abs(position), self.base_step)))
            logger.print(f"REDUCING position {side * min(abs(position), self.base_step)} SQUID_INK @ {mid_price}")

        elif allow_trading:
            # light pyramid when signal is mild
            if abs(z_score) < 1.0:
                step = self.base_step
            else:
                step = self.base_step * 2

            if mid_price < buy_threshold and position < self.max_position:
                volume = min(step, self.max_position - position)
                orders.append(Order(self.symbol, int(mid_price), volume))
                logger.print(f"BUY {volume} SQUID_INK @ {mid_price}")

            elif mid_price > sell_threshold and position > -self.max_position:
                volume = min(step, self.max_position + position)
                orders.append(Order(self.symbol, int(mid_price), -volume))
                logger.print(f"SELL {volume} SQUID_INK @ {mid_price}")

        return orders

    def serialize_state(self, mid_price, z_score, buy_threshold, sell_threshold, position, allow_trading):
        """
        Serialize the current strategy state to JSON
        """
        trader_data = {
            "mid_price": mid_price,
            "log_price": math.log(mid_price) if mid_price > 0 else 0,
            "z_score": z_score,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "position": position,
            "allow_trading": allow_trading
        }
        return trader_data


class Trader:
    def __init__(self):
        # Initialize the individual strategy instances
        self.kelp_strategy = KelpStrategy()
        self.rainforest_strategy = RainforestStrategy()
        self.squid_ink_strategy = SquidInkStrategy()

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        # Initialize result dictionary for all symbols
        result = {
            "KELP": [],
            "RAINFOREST_RESIN": [],
            "SQUID_INK": []
        }
        conversions = 0

        # Execute KELP strategy if we have data for it
        if "KELP" in state.order_depths:
            result["KELP"] = self.kelp_strategy.run(state)

        # Execute RAINFOREST_RESIN strategy if we have data for it
        if "RAINFOREST_RESIN" in state.order_depths:
            result["RAINFOREST_RESIN"] = self.rainforest_strategy.run(state)

        # Execute SQUID_INK strategy if we have data for it
        if "SQUID_INK" in state.order_depths:
            result["SQUID_INK"] = self.squid_ink_strategy.run(state)

        # Clean up empty result entries
        result = {k: v for k, v in result.items() if v}

        # Combine trader data from all strategies
        trader_data = {
            "kelp": self.kelp_strategy.serialize_state(),
            "rainforest": self.rainforest_strategy.serialize_state(),
            "squid_ink": {},  # We'll populate this only if we have data
        }

        # Add squid ink data if available
        if "SQUID_INK" in state.order_depths:
            order_depth = state.order_depths["SQUID_INK"]
            position = state.position.get("SQUID_INK", 0)

            # Calculate mid price and other metrics if we have order data
            if (order_depth.buy_orders or order_depth.sell_orders) and len(self.squid_ink_strategy.price_history) >= self.squid_ink_strategy.window:
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
                elif order_depth.buy_orders:
                    mid_price = max(order_depth.buy_orders.keys())
                else:
                    mid_price = min(order_depth.sell_orders.keys())

                log_prices = [math.log(p) for p in self.squid_ink_strategy.price_history if p > 0]
                mean_log_price = statistics.mean(log_prices)
                std_log_price = statistics.stdev(log_prices)
                current_log_price = math.log(mid_price)
                z_score = (current_log_price - mean_log_price) / std_log_price
                allow_trading = abs(z_score) < self.squid_ink_strategy.volatility_threshold
                base_price = statistics.mean(self.squid_ink_strategy.price_history)
                buy_threshold = base_price - std_log_price
                sell_threshold = base_price + std_log_price

                # Add Squid Ink data to trader_data
                trader_data["squid_ink"] = self.squid_ink_strategy.serialize_state(
                    mid_price, z_score, buy_threshold, sell_threshold, position, allow_trading
                )

        # Serialize trader_data to JSON
        trader_data_str = json.dumps(trader_data)

        # Flush logs and return
        logger.flush(state, result, conversions, trader_data_str)
        return result, conversions, trader_data_str