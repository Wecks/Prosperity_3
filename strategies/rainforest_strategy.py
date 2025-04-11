from datamodel import OrderDepth, UserId, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import List, Dict, Any
import statistics
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

    def calculate_average_price(self, buy_orders, sell_orders):
        """Calculate the average price from all orders in the book"""
        all_prices = []

        # Add all bid prices (weighted by volume)
        for bid_price, bid_volume in buy_orders.items():
            all_prices.extend([bid_price] * bid_volume)

        # Add all ask prices (weighted by volume)
        for ask_price, ask_volume in sell_orders.items():
            all_prices.extend([ask_price] * abs(ask_volume))

        if all_prices:
            return sum(all_prices) / len(all_prices)
        else:
            return self.mean_price  # Default to mean if no orders

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

    def run(self, state: TradingState):
        result = {}
        orders = []

        # Only process if RAINFOREST_RESIN is in the order book
        if self.target_product not in state.order_depths:
            logger.flush(state, {self.target_product: []}, 0, "{}")
            return {self.target_product: []}, 0, "{}"

        # Get order depth for our target product
        order_depth = state.order_depths[self.target_product]

        # Get current position
        current_position = 0
        if self.target_product in state.position:
            current_position = state.position[self.target_product]
        self.positions["total"] = current_position

        # Calculate mid price
        mid_price = self.calculate_mid_price(order_depth.buy_orders, order_depth.sell_orders)

        # Calculate the average price of all bids and asks - this will be our threshold
        average_price = self.calculate_average_price(order_depth.buy_orders, order_depth.sell_orders)

        # Update thresholds to use dynamic average price instead of static values
        self.buy_threshold = average_price - 1
        self.sell_threshold = average_price + 1

        # Update price history and statistical measures
        self.update_price_history(mid_price)

        # Update position stats
        self.update_position_stats()

        # Log current state
        logger.print(f"{self.target_product} - Position: {current_position}/{self.max_position}, Mean: {self.mean_price:.2f}, Mid: {mid_price}")
        logger.print(f"Average price (threshold): {average_price:.2f}")
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

        # Update trader state data
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
        trader_data_str = json.dumps(trader_data)

        # Add orders to result and return
        result[self.target_product] = orders
        logger.flush(state, result, 0, trader_data_str)

        return result, 0, trader_data_str