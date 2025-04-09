from datamodel import OrderDepth, TradingState, Order, Symbol, ProsperityEncoder
from typing import Dict, List, Tuple, Any
import statistics
import json
import math
from collections import deque

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
        return json.dumps(value)

    def truncate(self, string: str, length: int) -> str:
        if len(string) <= length:
            return string
        return string[:length - 3] + "..."

class SquidInkStrategy:
    def __init__(self):
        # Constants for the strategy
        self.product = "SQUID_INK"
        self.symbol = "SQUID_INK"
        self.position_limit = 50
        self.price_history = deque(maxlen=100)  # Store up to 100 historical mid prices
        self.volume_history = deque(maxlen=20)  # Store recent trading volumes
        self.volatility_window = 20  # Window size for volatility calculation

        # Dynamic parameters - FINAL REVISION
        self.mean_reversion_threshold = 1.2  # Further reduced to be even less aggressive
        self.trend_threshold = 0.5  # Further reduced for more trend selectivity
        self.max_spread = 2  # Maximum spread to offer
        self.min_spread = 1  # Minimum spread to offer

        # Risk management parameters - FINAL REVISION
        self.max_position_per_trade = 8  # Further reduced for more conservative position sizing
        self.risk_factor = 0.5  # Further reduced for even more conservative approach
        self.edge_requirement = 0.75  # Minimum price difference from fair value to trade

        # State variables
        self.current_position = 0
        self.market_regime = "UNKNOWN"
        self.last_mid_price = None
        self.trade_count = 0
        self.position_history = []
        self.profit_loss = 0
        self.recent_trades = deque(maxlen=20)  # Track recent trades for better position management
        self.baseline_fair_value = 1950  # Starting approximation based on data analysis

    def calculate_fair_value(self, order_depth, timestamp) -> float:
        """Calculate fair value price based on order book and historical data"""
        # Get best bid and ask if available
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        # Calculate mid price if both bid and ask available
        if best_bid and best_ask:
            # Calculate weighted mid price based on order volumes
            bid_volume = sum(abs(v) for v in order_depth.buy_orders.values())
            ask_volume = sum(abs(v) for v in order_depth.sell_orders.values())
            total_volume = bid_volume + ask_volume

            if total_volume > 0:
                # Weight the price by volume on each side
                mid_price = (best_bid * ask_volume + best_ask * bid_volume) / total_volume
            else:
                mid_price = (best_bid + best_ask) / 2
        elif best_bid:
            mid_price = best_bid
        elif best_ask:
            mid_price = best_ask
        elif len(self.price_history) > 0:
            # Use last price if no orders
            mid_price = self.price_history[-1]
        else:
            # Default if no history available
            mid_price = self.baseline_fair_value

        # Store the mid price in history
        self.price_history.append(mid_price)

        # Calculate market regime and adjust fair value
        if len(self.price_history) >= self.volatility_window:
            # Get a window of recent prices
            recent_prices = list(self.price_history)[-self.volatility_window:]

            # Calculate volatility (standard deviation)
            volatility = statistics.stdev(recent_prices)

            # Calculate recent price trend (using exponential moving average for more responsiveness)
            alpha_short = 0.25  # Lower weight from 0.3
            alpha_long = 0.05   # Lower weight from 0.1

            # Calculate EMAs
            short_ema = recent_prices[-1]
            long_ema = recent_prices[-1]

            for i in range(2, min(6, len(recent_prices) + 1)):
                short_ema = alpha_short * recent_prices[-i] + (1 - alpha_short) * short_ema

            for i in range(2, min(21, len(recent_prices) + 1)):
                long_ema = alpha_long * recent_prices[-i] + (1 - alpha_long) * long_ema

            # Calculate mean of recent prices for mean reversion
            mean_price = sum(recent_prices) / len(recent_prices)

            # Detect trending or mean-reverting market
            if short_ema > long_ema + volatility * 0.25:
                self.market_regime = "TRENDING_UP"
                # In uptrend, fair value is slightly higher but less aggressive
                fair_value = mid_price + volatility * 0.1
            elif short_ema < long_ema - volatility * 0.25:
                self.market_regime = "TRENDING_DOWN"
                # In downtrend, fair value is slightly lower but less aggressive
                fair_value = mid_price - volatility * 0.1
            else:
                # Calculate z-score for mean reversion
                z_score = (mid_price - mean_price) / volatility if volatility > 0 else 0

                if abs(z_score) > self.mean_reversion_threshold:
                    self.market_regime = "MEAN_REVERTING"
                    # If price is far from mean, it should revert - but even less aggressively
                    fair_value = mean_price + (z_score / abs(z_score)) * volatility * 0.15
                else:
                    self.market_regime = "NEUTRAL"
                    # In neutral market, use weighted average of mid price and mean price
                    fair_value = mid_price * 0.8 + mean_price * 0.2
        else:
            # Not enough data points yet
            fair_value = mid_price
            self.market_regime = "UNKNOWN"

        self.last_mid_price = mid_price
        return fair_value

    def calculate_order_volumes(self, fair_value, order_depth, position) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Calculate order volumes based on current position and market state"""
        # Initialize dictionaries for buy and sell orders
        buy_orders = {}
        sell_orders = {}

        # Available position capacity
        buy_capacity = self.position_limit - position
        sell_capacity = self.position_limit + position

        # Scale capacity based on current position - be even more conservative when position is large
        position_scale_factor = max(0.2, 1.0 - (abs(position) / self.position_limit) * 0.7)

        if buy_capacity <= 0 or sell_capacity <= 0:
            # We're at position limit, can only trade in one direction
            if buy_capacity <= 0:  # At max long position
                sell_capacity = min(sell_capacity, self.max_position_per_trade)
            else:  # At max short position
                buy_capacity = min(buy_capacity, self.max_position_per_trade)
        else:
            # We have capacity in both directions, but limit per trade
            # Scale down as we approach position limits
            buy_capacity = min(buy_capacity, int(self.max_position_per_trade * position_scale_factor))
            sell_capacity = min(sell_capacity, int(self.max_position_per_trade * position_scale_factor))

        # Calculate position sizing based on market regime
        if self.market_regime == "TRENDING_UP":
            # In uptrend, be marginally more aggressive with buys
            buy_capacity = int(buy_capacity * self.risk_factor * 1.05)
            sell_capacity = int(sell_capacity * self.risk_factor * 0.95)
        elif self.market_regime == "TRENDING_DOWN":
            # In downtrend, be marginally more aggressive with sells
            buy_capacity = int(buy_capacity * self.risk_factor * 0.95)
            sell_capacity = int(sell_capacity * self.risk_factor * 1.05)
        elif self.market_regime == "MEAN_REVERTING":
            # In mean reversion, be balanced
            buy_capacity = int(buy_capacity * self.risk_factor)
            sell_capacity = int(sell_capacity * self.risk_factor)
        else:
            # Default position sizing - more conservative
            buy_capacity = int(buy_capacity * self.risk_factor * 0.8)
            sell_capacity = int(sell_capacity * self.risk_factor * 0.8)

        # IMPROVED: Calculate total bid and ask volumes for the depth analysis
        total_bid_volume = sum(abs(v) for v in order_depth.buy_orders.values())
        total_ask_volume = sum(abs(v) for v in order_depth.sell_orders.values())

        # Adjust sizing based on imbalance
        if total_bid_volume > 0 and total_ask_volume > 0:
            volume_ratio = total_bid_volume / total_ask_volume
            # If more buying pressure (high bid volume), be more aggressive on sells
            if volume_ratio > 1.3:
                sell_capacity = int(sell_capacity * min(1.2, volume_ratio / 1.3))
            # If more selling pressure (high ask volume), be more aggressive on buys
            elif volume_ratio < 0.7:
                buy_capacity = int(buy_capacity * min(1.2, (1 / volume_ratio) / 1.3))

        # Find profitable buy opportunities - sort by price to prioritize best prices
        sorted_sell_prices = sorted(order_depth.sell_orders.keys())
        for price in sorted_sell_prices:
            volume = order_depth.sell_orders[price]
            # Only trade if price is sufficiently below our fair value (minimum edge)
            if price <= fair_value - self.edge_requirement:
                # Calculate volume to buy, considering position limit
                volume_to_buy = min(abs(volume), buy_capacity)
                if volume_to_buy > 0:
                    buy_orders[price] = volume_to_buy
                    buy_capacity -= volume_to_buy

                    # Stop if we've reached capacity
                    if buy_capacity <= 0:
                        break

        # Find profitable sell opportunities - sort by price to prioritize best prices
        sorted_buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
        for price in sorted_buy_prices:
            volume = order_depth.buy_orders[price]
            # Only trade if price is sufficiently above our fair value (minimum edge)
            if price >= fair_value + self.edge_requirement:
                # Calculate volume to sell, considering position limit
                volume_to_sell = min(volume, sell_capacity)
                if volume_to_sell > 0:
                    sell_orders[price] = -volume_to_sell
                    sell_capacity -= volume_to_sell

                    # Stop if we've reached capacity
                    if sell_capacity <= 0:
                        break

        # IMPROVED: Position unwinding logic for risk management
        # If we have a significant position, try to unwind when appropriate
        if position > 20 and self.market_regime != "TRENDING_UP":
            # We have a large long position, look for opportunities to reduce it
            for price in sorted_buy_prices:
                if price >= fair_value:  # Not strict about edge requirement when unwinding
                    volume_to_reduce = min(order_depth.buy_orders[price], position - 15)
                    if volume_to_reduce > 0:
                        if price in sell_orders:
                            sell_orders[price] -= volume_to_reduce
                        else:
                            sell_orders[price] = -volume_to_reduce
                        break

        elif position < -20 and self.market_regime != "TRENDING_DOWN":
            # We have a large short position, look for opportunities to reduce it
            for price in sorted_sell_prices:
                if price <= fair_value:  # Not strict about edge requirement when unwinding
                    volume_to_reduce = min(abs(order_depth.sell_orders[price]), abs(position) - 15)
                    if volume_to_reduce > 0:
                        if price in buy_orders:
                            buy_orders[price] += volume_to_reduce
                        else:
                            buy_orders[price] = volume_to_reduce
                        break

        # If we haven't placed all orders and there's market depth, place limit orders
        # Only place limit orders when we have enough price history and sufficient volatility
        if len(self.price_history) >= self.volatility_window and (buy_capacity > 0 or sell_capacity > 0):
            recent_prices = list(self.price_history)[-self.volatility_window:]
            volatility = statistics.stdev(recent_prices)

            # Only place limit orders if volatility is significant enough
            if volatility > 0.8:
                # Adjust spread based on volatility - but more conservatively
                dynamic_spread = max(self.min_spread, min(self.max_spread, int(volatility * 0.4)))

                # Ensure we have a last mid price
                if self.last_mid_price:
                    # Only place limit orders if they're not too close to the fair value
                    # Place limit buy orders below fair value
                    if buy_capacity > 0 and position < 30:  # Don't place buy limits if position is too large
                        limit_buy_price = math.floor(fair_value) - dynamic_spread
                        # Use smaller sizes for limit orders
                        remaining_volume = min(2, buy_capacity)
                        buy_orders[limit_buy_price] = remaining_volume

                    # Place limit sell orders above fair value
                    if sell_capacity > 0 and position > -30:  # Don't place sell limits if position is too negative
                        limit_sell_price = math.ceil(fair_value) + dynamic_spread
                        # Use smaller sizes for limit orders
                        remaining_volume = min(2, sell_capacity)
                        sell_orders[limit_sell_price] = -remaining_volume

        return buy_orders, sell_orders

    def calculate_market_volume(self, market_trades):
        """Calculate recent trading volume to gauge market activity"""
        total_volume = 0
        if self.symbol in market_trades and market_trades[self.symbol]:
            for trade in market_trades[self.symbol]:
                total_volume += abs(trade.quantity)
                self.recent_trades.append(trade)

        self.volume_history.append(total_volume)

        # Return average recent volume
        if len(self.volume_history) > 0:
            return sum(self.volume_history) / len(self.volume_history)
        return 0

    def update_position_history(self, position):
        """Track position changes over time"""
        self.position_history.append(position)
        # Keep only the most recent 100 position records
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-100:]

    def run(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        """
        Main method called by the exchange. Takes market data and returns orders.
        """
        # Initialize the result dict
        result = {}

        # Initialize the logger
        logger = Logger()

        # Store current timestamp
        timestamp = state.timestamp

        # Check if our product is in the order depths
        if self.symbol in state.order_depths:
            # Get the order depth for our product
            order_depth = state.order_depths[self.symbol]

            # Get current position
            position = state.position.get(self.product, 0)
            self.current_position = position
            self.update_position_history(position)

            # Calculate market volume
            market_volume = self.calculate_market_volume(state.market_trades)

            # Calculate fair value
            fair_value = self.calculate_fair_value(order_depth, timestamp)

            # Calculate buy and sell order volumes
            buy_orders, sell_orders = self.calculate_order_volumes(fair_value, order_depth, position)

            # Create and add orders to result
            orders = []

            # Add buy orders
            for price, volume in buy_orders.items():
                orders.append(Order(self.symbol, price, volume))

            # Add sell orders
            for price, volume in sell_orders.items():
                orders.append(Order(self.symbol, price, volume))

            # Add all orders to the result
            if orders:
                result[self.symbol] = orders

            # Calculate estimated profit/loss
            # Track our own trades if available
            if self.symbol in state.own_trades and state.own_trades[self.symbol]:
                for trade in state.own_trades[self.symbol]:
                    trade_value = trade.price * trade.quantity
                    if trade.buyer == 'SUBMISSION':
                        # We bought, so we spent money
                        self.profit_loss -= trade_value
                    elif trade.seller == 'SUBMISSION':
                        # We sold, so we gained money
                        self.profit_loss += trade_value

            # Log key information
            logger.print(f"Timestamp: {timestamp}")
            logger.print(f"Market regime: {self.market_regime}")
            logger.print(f"Fair value: {fair_value}")
            logger.print(f"Current position: {position}")
            logger.print(f"Market volume: {market_volume}")
            logger.print(f"Profit/Loss estimate: {self.profit_loss}")
            logger.print(f"Orders: {orders}")

        # Convert traderData to string
        trader_data = json.dumps({
            'price_history': list(self.price_history),
            'market_regime': self.market_regime,
            'position_history': self.position_history,
            'trade_count': self.trade_count,
            'profit_loss': self.profit_loss
        })

        # Increment trade count
        self.trade_count += 1

        # Return result and log
        logger.flush(state, result, 0, trader_data)
        return result


# Create a Trader class that will be used by the backtester
class Trader:
    def __init__(self):
        # Initialize our strategy implementation
        self.strategy = SquidInkStrategy()

    def run(self, state: TradingState):
        """
        Main method called by the backtester. Delegates to our strategy implementation.
        The backtester expects (orders, conversions, trader_data) to be returned.
        """
        # Get orders from our strategy
        orders = self.strategy.run(state)

        # Return orders, conversions (0), and trader_data (empty string)
        # as expected by the backtester
        return orders, 0, ""