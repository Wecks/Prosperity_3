from datamodel import OrderDepth, TradingState, Order, Symbol, ProsperityEncoder
from typing import Dict, List, Tuple, Any
import statistics
import json
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

    def compress_listings(self, listings: dict[Symbol, Any]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [od.buy_orders, od.sell_orders] for symbol, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Any]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for ts in trades.values() for t in ts]

    def compress_observations(self, observations: Any) -> list[Any]:
        conversion_obs = {
            p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
            for p, o in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conversion_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for os in orders.values() for o in os]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        # Target product
        self.symbol = "SQUID_INK"

        # Price analysis parameters
        self.price_history = []
        self.max_history_length = 100
        self.window_short = 15    # Short-term window for volatility
        self.window_medium = 30   # Medium-term window for trend analysis
        self.window_long = 60     # Long-term window for mean reversion

        # Trading thresholds and parameters
        self.mean_price = 10000   # Initial assumption, will be adjusted
        self.max_position = 100   # Maximum position size in each direction
        self.base_position_size = 10
        self.vol_lookback = 30    # Lookback period for volatility calculation
        self.aggressive_factor = 0.75  # How aggressively to trade anomalies

        # Position tracking
        self.positions = {
            "longs": [],          # [(price, size), ...]
            "shorts": [],         # [(price, size), ...]
            "total": 0,
            "avg_long_price": 0,
            "avg_short_price": 0,
            "long_total_size": 0,
            "short_total_size": 0
        }

        # Pyramiding parameters
        self.pyramid_levels = 5
        self.pyramid_step = 3     # Price steps between pyramid levels
        self.pyramid_multiplier = [1.0, 1.5, 2.0, 2.5, 3.0]

        # Dynamic thresholds based on volatility
        self.vol_multiplier = 1.5  # Volatility threshold multiplier
        self.buy_threshold = 9997  # Initial value, will be adjusted
        self.sell_threshold = 10003 # Initial value, will be adjusted

        # Historical volatility tracking
        self.volatility = None
        self.vol_history = []
        self.max_vol_history = 50

        # Performance tracking
        self.trades_executed = 0
        self.profitable_trades = 0
        self.realized_pnl = 0

    def calculate_mid_price(self, buy_orders, sell_orders):
        """Calculate the mid price from the order book"""
        if buy_orders and sell_orders:
            best_bid = max(buy_orders.keys())
            best_ask = min(sell_orders.keys())
            return (best_bid + best_ask) / 2
        elif buy_orders:
            return max(buy_orders.keys())
        elif sell_orders:
            return min(sell_orders.keys())
        else:
            return self.mean_price

    def update_price_history(self, mid_price):
        """Update price history and derived metrics"""
        if mid_price is not None:
            # Add price to history and maintain history length
            self.price_history.append(mid_price)
            if len(self.price_history) > self.max_history_length:
                self.price_history = self.price_history[-self.max_history_length:]

            # Calculate multiple time window averages if we have enough data
            if len(self.price_history) >= self.window_long:
                # Update mean price with exponential weighting
                recent_prices = self.price_history[-self.window_medium:]
                weights = [0.95 ** i for i in range(len(recent_prices))]
                weighted_sum = sum(p * w for p, w in zip(reversed(recent_prices), weights))
                recent_mean = weighted_sum / sum(weights)

                # Gradually adjust the mean price
                adjustment_rate = 0.05
                self.mean_price = (1 - adjustment_rate) * self.mean_price + adjustment_rate * recent_mean

                # Calculate volatility
                log_returns = [math.log(self.price_history[i] / self.price_history[i-1])
                              for i in range(-self.vol_lookback, 0)]
                self.volatility = statistics.stdev(log_returns) * math.sqrt(252) * mid_price

                # Add to volatility history
                self.vol_history.append(self.volatility)
                if len(self.vol_history) > self.max_vol_history:
                    self.vol_history = self.vol_history[-self.max_vol_history:]

                # Dynamic thresholds based on current volatility
                # Use more aggressive thresholds if volatility is low
                vol_percentile = self.get_volatility_percentile()

                if vol_percentile < 0.3:  # Low volatility - use tighter thresholds
                    vol_factor = 0.8
                    self.buy_threshold = self.mean_price - int(vol_factor * self.volatility)
                    self.sell_threshold = self.mean_price + int(vol_factor * self.volatility)
                elif vol_percentile > 0.7:  # High volatility - use wider thresholds
                    vol_factor = 1.5
                    self.buy_threshold = self.mean_price - int(vol_factor * self.volatility)
                    self.sell_threshold = self.mean_price + int(vol_factor * self.volatility)
                else:  # Medium volatility - use standard thresholds
                    vol_factor = 1.0
                    self.buy_threshold = self.mean_price - int(vol_factor * self.volatility)
                    self.sell_threshold = self.mean_price + int(vol_factor * self.volatility)

    def get_volatility_percentile(self):
        """Calculate the current volatility percentile"""
        if not self.vol_history or self.volatility is None:
            return 0.5  # Default mid percentile

        sorted_vols = sorted(self.vol_history)
        index = sorted_vols.index(min([v for v in sorted_vols if v >= self.volatility], default=self.volatility))
        return index / len(sorted_vols)

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

    def calculate_position_size(self, price, current_position, is_buy):
        """Calculate the appropriate position size based on pyramid level and other factors"""
        # Determine pyramid level
        pyramid_level = self.determine_pyramid_level(price, is_buy)

        # Base size scaled by pyramid multiplier
        size = int(self.base_position_size * self.pyramid_multiplier[pyramid_level])

        # Scale based on price distance from mean (more aggressive when further from mean)
        price_deviation_pct = abs(price - self.mean_price) / self.mean_price
        distance_factor = 1.0 + (price_deviation_pct * 10)  # Amplify small percentage differences
        size = int(size * min(distance_factor, 3.0))  # Cap multiplier at 3x

        # Account for existing positions (reduce size when building on same side)
        # Increase size when reversing positions
        if is_buy:
            # If buying, check existing long positions
            if self.positions["long_total_size"] > 40:
                # Reduce size when we already have significant longs
                size = max(1, int(size * 0.7))
            # Increase size if we're closing shorts
            if self.positions["short_total_size"] > 0:
                size = max(size, int(self.positions["short_total_size"] * 1.5))
        else:
            # If selling, check existing short positions
            if self.positions["short_total_size"] > 40:
                # Reduce size when we already have significant shorts
                size = max(1, int(size * 0.7))
            # Increase size if we're closing longs
            if self.positions["long_total_size"] > 0:
                size = max(size, int(self.positions["long_total_size"] * 1.5))

        # Respect maximum position limit
        available_capacity = self.max_position - abs(current_position)
        return min(size, available_capacity)

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result = {self.symbol: []}
        conversions = 0
        orders = []

        # Get order depth for our target product
        order_depth = state.order_depths.get(self.symbol, None)
        if not order_depth:
            logger.flush(state, result, conversions, "{}")
            return result, conversions, "{}"

        # Get current position
        current_position = state.position.get(self.symbol, 0)
        self.positions["total"] = current_position

        # Calculate mid price and update price history
        mid_price = self.calculate_mid_price(order_depth.buy_orders, order_depth.sell_orders)
        self.update_price_history(mid_price)

        # Update position stats
        self.update_position_stats()

        # Log current state
        logger.print(f"{self.symbol} - Position: {current_position}/{self.max_position}, Mean: {self.mean_price:.2f}, Mid: {mid_price}")
        logger.print(f"Buy threshold: {self.buy_threshold}, Sell threshold: {self.sell_threshold}")
        if self.volatility:
            logger.print(f"Current volatility: {self.volatility:.2f}, Percentile: {self.get_volatility_percentile():.2f}")

        if self.positions["long_total_size"] > 0:
            logger.print(f"Long positions: {self.positions['long_total_size']}x @ avg {self.positions['avg_long_price']:.2f}")
        if self.positions["short_total_size"] > 0:
            logger.print(f"Short positions: {self.positions['short_total_size']}x @ avg {self.positions['avg_short_price']:.2f}")

        # Calculate remaining position capacity
        remaining_buy_capacity = self.max_position - current_position
        remaining_sell_capacity = self.max_position + current_position

        # Process exit opportunities first (lock in profits / cut losses)

        # Exit shorts if price is approaching buy threshold
        if (self.positions["short_total_size"] > 0 and
            mid_price <= self.buy_threshold + int(0.5 * self.volatility if self.volatility else 5)):
            # Calculate how much of our short position to close
            exit_size = max(1, int(self.positions["short_total_size"] * 0.8))  # Close most of the position
            exit_size = min(exit_size, remaining_buy_capacity)

            if exit_size > 0 and order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                ask_volume = abs(order_depth.sell_orders[best_ask])
                buy_size = min(exit_size, ask_volume)

                if buy_size > 0:
                    logger.print(f"EXIT SHORT: Buying {buy_size}x at {best_ask} (covering shorts)")
                    orders.append(Order(self.symbol, int(best_ask), buy_size))

                    # Update position tracking
                    self.positions["longs"].append((best_ask, buy_size))
                    remaining_buy_capacity -= buy_size
                    current_position += buy_size
                    self.trades_executed += 1

                    # Calculate profit/loss
                    avg_short_price = self.positions["avg_short_price"]
                    trade_pnl = (avg_short_price - best_ask) * buy_size
                    self.realized_pnl += trade_pnl
                    if trade_pnl > 0:
                        self.profitable_trades += 1

                    self.update_position_stats()

        # Exit longs if price is approaching sell threshold
        if (self.positions["long_total_size"] > 0 and
            mid_price >= self.sell_threshold - int(0.5 * self.volatility if self.volatility else 5)):
            # Calculate how much of our long position to close
            exit_size = max(1, int(self.positions["long_total_size"] * 0.8))  # Close most of the position
            exit_size = min(exit_size, remaining_sell_capacity)

            if exit_size > 0 and order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                bid_volume = order_depth.buy_orders[best_bid]
                sell_size = min(exit_size, bid_volume)

                if sell_size > 0:
                    logger.print(f"EXIT LONG: Selling {sell_size}x at {best_bid} (closing longs)")
                    orders.append(Order(self.symbol, int(best_bid), -sell_size))

                    # Update position tracking
                    self.positions["shorts"].append((best_bid, sell_size))
                    remaining_sell_capacity -= sell_size
                    current_position -= sell_size
                    self.trades_executed += 1

                    # Calculate profit/loss
                    avg_long_price = self.positions["avg_long_price"]
                    trade_pnl = (best_bid - avg_long_price) * sell_size
                    self.realized_pnl += trade_pnl
                    if trade_pnl > 0:
                        self.profitable_trades += 1

                    self.update_position_stats()

        # Process sell orders in the book (opportunities to buy)
        if len(order_depth.sell_orders) > 0 and remaining_buy_capacity > 0:
            # Sort sell orders by price (ascending)
            ask_prices = sorted(order_depth.sell_orders.keys())

            for ask_price in ask_prices:
                # Only buy if price is at or below our threshold
                if ask_price <= self.buy_threshold:
                    ask_volume = abs(order_depth.sell_orders[ask_price])

                    # Calculate position size
                    buy_size = self.calculate_position_size(ask_price, current_position, True)
                    buy_size = min(ask_volume, buy_size, remaining_buy_capacity)

                    if buy_size > 0:
                        # Check if we're closing shorts or opening new longs
                        if self.positions["short_total_size"] > 0:
                            logger.print(f"REVERSAL BUY: {buy_size}x at {ask_price} (Closing shorts + new long)")
                        else:
                            pyramid_level = self.determine_pyramid_level(ask_price, True)
                            logger.print(f"BUY: {buy_size}x at {ask_price} (Pyramid level: {pyramid_level})")

                        orders.append(Order(self.symbol, int(ask_price), buy_size))

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

                    # Calculate position size
                    sell_size = self.calculate_position_size(bid_price, current_position, False)
                    sell_size = min(bid_volume, sell_size, remaining_sell_capacity)

                    if sell_size > 0:
                        # Check if we're closing longs or opening new shorts
                        if self.positions["long_total_size"] > 0:
                            logger.print(f"REVERSAL SELL: {sell_size}x at {bid_price} (Closing longs + new short)")
                        else:
                            pyramid_level = self.determine_pyramid_level(bid_price, False)
                            logger.print(f"SELL: {sell_size}x at {bid_price} (Pyramid level: {pyramid_level})")

                        orders.append(Order(self.symbol, int(bid_price), -sell_size))

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
            "volatility": self.volatility,
            "vol_percentile": self.get_volatility_percentile() if self.vol_history else 0.5,
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
        result[self.symbol] = orders
        logger.flush(state, result, conversions, trader_data_str)

        return result, conversions, trader_data_str