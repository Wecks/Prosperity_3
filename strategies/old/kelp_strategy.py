from datamodel import OrderDepth, UserId, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
import json
from typing import Dict, List, Any, Tuple

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
        # Initialize strategy parameters
        self.mean_price = 2034.5  # Base mean price of KELP observed from logs

        # Instrument to track
        self.instrument = "KELP"

        # Order sizing and position management
        self.base_position_size = 10  # Base size for each trade
        self.max_position = 100  # Maximum position size (positive or negative)
        self.max_trade_size = 25  # Maximum size per individual trade

        # Pyramiding parameters (like rainforest strategy)
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

        # Price history and moving averages
        self.price_history = []  # Store price history for calculating moving averages
        self.max_history_length = 100
        self.long_term_ma_period = 20  # Long-term moving average period
        self.long_term_ma = self.mean_price  # Initial long-term MA
        self.volatility = 0.5  # Estimated volatility

        # Define thresholds that will be dynamically updated
        self.buy_threshold = 2033.5  # Initial value
        self.sell_threshold = 2035.5  # Initial value

        # Minimum threshold distance
        self.min_threshold_distance = 0.5

    def calculate_moving_averages(self, new_price):
        """Calculate moving averages when a new price is available"""
        # Add the new price to history
        self.price_history.append(new_price)

        # Trim history if needed
        if len(self.price_history) > self.max_history_length:
            self.price_history = self.price_history[-self.max_history_length:]

        # Calculate long-term MA if we have enough data
        if len(self.price_history) >= self.long_term_ma_period:
            self.long_term_ma = sum(self.price_history[-self.long_term_ma_period:]) / self.long_term_ma_period
        else:
            # Not enough data yet, use exponential smoothing instead
            self.long_term_ma = 0.95 * self.long_term_ma + 0.05 * new_price

    def calculate_bid_ask_average(self, order_depth):
        """Calculate the average of best bid and best ask from the order book"""
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        elif len(order_depth.buy_orders) > 0:
            return max(order_depth.buy_orders.keys())
        elif len(order_depth.sell_orders) > 0:
            return min(order_depth.sell_orders.keys())
        else:
            return self.mean_price  # Default to mean if no orders

    def update_parameters(self, mid_price):
        """
        Dynamically update strategy parameters based on market conditions and moving averages
        """
        # Update moving averages
        self.calculate_moving_averages(mid_price)

        # Store recent prices for volatility calculation
        self.recent_prices = self.price_history[-10:] if len(self.price_history) >= 10 else self.price_history

        # Calculate short-term volatility
        if len(self.recent_prices) >= 3:
            # Simple volatility calculation using range
            price_range = max(self.recent_prices) - min(self.recent_prices)
            self.volatility = max(0.5, price_range * 0.5)  # Scale and apply minimum

        # Adjust mean price more aggressively toward current mid price
        self.mean_price = 0.90 * self.mean_price + 0.10 * mid_price

        # Calculate dynamic threshold distance based on volatility
        # Lower volatility = tighter thresholds = more trades
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

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """
        Main strategy execution method with the new optimized approach combining:
        1. Moving average trend detection (long when price > MA, short when price < MA)
        2. Bid/ask average for optimized entry points
        """
        # Initialize result dict with empty lists for each symbol
        result = {self.instrument: []}
        orders = []
        conversions = 0

        # Skip if our instrument is not in the order depths
        if self.instrument not in state.order_depths:
            trader_data = self.serialize_state()
            logger.flush(state, result, conversions, trader_data)
            return result, conversions, trader_data

        # Get order depth and current position
        order_depth = state.order_depths[self.instrument]
        position = state.position.get(self.instrument, 0)
        self.positions["total"] = position

        # Calculate mid price and bid/ask average
        mid_price = self.calculate_mid_price(order_depth)
        if mid_price is None:
            trader_data = self.serialize_state()
            logger.flush(state, result, conversions, trader_data)
            return result, conversions, trader_data

        # Calculate bid/ask average for entry point optimization
        bid_ask_average = self.calculate_bid_ask_average(order_depth)

        # Update parameters including moving averages
        self.update_parameters(mid_price)
        self.update_position_stats()

        # Determine trend direction based on price vs long-term MA
        is_uptrend = mid_price > self.long_term_ma
        is_downtrend = mid_price < self.long_term_ma

        # Log current state with MA information
        logger.print(f"{self.instrument} - Position: {position}/{self.max_position}, Mean: {self.mean_price:.2f}, Mid: {mid_price}")
        logger.print(f"Long-term MA: {self.long_term_ma:.2f}, Bid/Ask Avg: {bid_ask_average:.2f}")
        logger.print(f"Trend: {'UP' if is_uptrend else ('DOWN' if is_downtrend else 'NEUTRAL')}")

        if self.positions["long_total_size"] > 0:
            logger.print(f"Long positions: {self.positions['long_total_size']}x @ avg {self.positions['avg_long_price']:.2f}")
        if self.positions["short_total_size"] > 0:
            logger.print(f"Short positions: {self.positions['short_total_size']}x @ avg {self.positions['avg_short_price']:.2f}")

        # Calculate remaining position capacity
        remaining_buy_capacity = self.max_position - position
        remaining_sell_capacity = self.max_position + position

        # Process sell orders in the book (opportunities to buy)
        if len(order_depth.sell_orders) > 0 and remaining_buy_capacity > 0:
            # Sort sell orders by price (ascending)
            ask_prices = sorted(order_depth.sell_orders.keys())

            for ask_price in ask_prices:
                # Buy if:
                # 1. It's an uptrend (price > long-term MA) AND
                # 2. The ask price is below the bid/ask average (good entry point)
                if (is_uptrend and ask_price <= bid_ask_average) or (is_downtrend and ask_price <= self.buy_threshold):
                    ask_volume = abs(order_depth.sell_orders[ask_price])

                    # Calculate position size
                    buy_size = self.determine_position_size(ask_price, position, True)
                    buy_size = min(buy_size, ask_volume, remaining_buy_capacity)

                    if buy_size > 0:
                        if is_uptrend:
                            logger.print(f"TREND BUY: {buy_size}x at {ask_price} (Uptrend + below Bid/Ask Avg: {bid_ask_average:.2f})")
                        else:
                            logger.print(f"MEAN REVERSION BUY: {buy_size}x at {ask_price} (Below buy threshold: {self.buy_threshold:.2f})")

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

        # Process buy orders in the book (opportunities to sell)
        if len(order_depth.buy_orders) > 0 and remaining_sell_capacity > 0:
            # Sort buy orders by price (descending)
            bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)

            for bid_price in bid_prices:
                # Sell if:
                # 1. It's a downtrend (price < long-term MA) AND
                # 2. The bid price is above the bid/ask average (good entry point)
                if (is_downtrend and bid_price >= bid_ask_average) or (is_uptrend and bid_price >= self.sell_threshold):
                    bid_volume = order_depth.buy_orders[bid_price]

                    # Calculate position size
                    sell_size = self.determine_position_size(bid_price, position, False)
                    sell_size = min(sell_size, abs(bid_volume), remaining_sell_capacity)

                    if sell_size > 0:
                        if is_downtrend:
                            logger.print(f"TREND SELL: {sell_size}x at {bid_price} (Downtrend + above Bid/Ask Avg: {bid_ask_average:.2f})")
                        else:
                            logger.print(f"MEAN REVERSION SELL: {sell_size}x at {bid_price} (Above sell threshold: {self.sell_threshold:.2f})")

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

        # Add orders to result
        result[self.instrument] = orders

        # Prepare trader data
        trader_data = self.serialize_state()
        trader_data_obj = json.loads(trader_data)
        trader_data_obj["long_term_ma"] = self.long_term_ma
        trader_data_obj["bid_ask_average"] = bid_ask_average
        trader_data_obj["is_uptrend"] = is_uptrend
        trader_data_str = json.dumps(trader_data_obj)

        # Let logger handle the flush
        logger.flush(state, result, conversions, trader_data_str)
        return result, conversions, trader_data_str

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

    def log_state(self, mid_price):
        """
        Log current strategy state
        """
        log_message = f"{self.instrument} - Position: {self.positions['total']}/{self.max_position}, Mean: {self.mean_price:.2f}, Mid: {mid_price}\n"
        log_message += f"Buy threshold: {self.buy_threshold}, Sell threshold: {self.sell_threshold}"

        # Add position details if we have positions
        if self.positions["long_total_size"] > 0:
            log_message += f"\nLong positions: {self.positions['long_total_size']}x @ avg {self.buy_threshold:.2f}"
        if self.positions["short_total_size"] > 0:
            log_message += f"\nShort positions: {self.positions['short_total_size']}x @ avg {self.sell_threshold:.2f}"

        logger.print(log_message)

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
        return json.dumps(strategy_state)
