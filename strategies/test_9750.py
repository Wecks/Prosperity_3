from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Tuple, Any, Optional
import numpy as np # type: ignore
from statistics import NormalDist
from collections import deque
import string
import math
from math import log, sqrt
import jsonpickle
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

class Product: 
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000  = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250  = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500  = "VOLCANIC_ROCK_VOUCHER_10500"

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            (log(spot) - log(strike)) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            (log(spot) - log(strike)) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-15
    ):
        """
        A binary-search approach to implied vol.
        We'll exit once we get close to the observed call_price,
        or we run out of iterations.
        """
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            if diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

class Status:
    def __init__(self, product: str, state: TradingState, strike=None) -> None:
        self.product = product
        self._state = state
        self.ma_window = 10  # For the moving average of implied vol
        self.alpha = 0.3
        self.volatility = 0.16
        self.initial_time_to_expiry = 7  # 4 trading days remaining at Round 4 (day 4)
        self.strike = strike
        self.price_history = []  # Track price history for trend analysis
        self.volatility_history = []  # Track volatility history
        self.last_trade_timestamp = 0  # Track when we last traded
        self.trade_count = 0  # Count trades to adjust aggressiveness
        self.profit_history = []  # Track profit/loss history


    @property
    def order_depth(self) -> OrderDepth:
        return self._state.order_depths[self.product]

    @property
    def bids(self) -> List[Tuple[int, int]]:
        return list(self._state.order_depths[self.product].buy_orders.items())

    @property
    def asks(self) -> List[Tuple[int, int]]:
        return list(self._state.order_depths[self.product].sell_orders.items())

    @property
    def position(self) -> int:
        return self._state.position.get(self.product, 0)

    @property
    def possible_buy_amt(self) -> int:
        return 50 - self.position

    @property
    def possible_sell_amt(self) -> int:
        return 50 + self.position

    @property
    def jam_possible_buy_amt(self) -> int:
        return 350 - self.position

    @property
    def jam_possible_sell_amt(self) -> int:
        return 350 + self.position

    @property
    def best_bid(self) -> int:
        bids = self._state.order_depths[self.product].buy_orders
        return max(bids.keys()) if bids else 0

    @property
    def best_ask(self) -> int:
        asks = self._state.order_depths[self.product].sell_orders
        return min(asks.keys()) if asks else float('inf')

    @property
    def maxamt_midprc(self) -> float:
        buy_orders = self._state.order_depths[self.product].buy_orders
        sell_orders = self._state.order_depths[self.product].sell_orders
        if not buy_orders or not sell_orders:
            return (self.best_bid + self.best_ask) / 2.0
        max_bv = 0
        max_bv_price = self.best_bid
        for p, v in buy_orders.items():
            if v > max_bv:
                max_bv = v
                max_bv_price = p
        max_sv = 0
        max_sv_price = self.best_ask
        for p, v in sell_orders.items():
            if -v > max_sv:
                max_sv = -v
                max_sv_price = p
        return (max_bv_price + max_sv_price) / 2

    @property
    def vwap(self) -> float:
        """
        Calculate Volume Weighted Average Price (VWAP) for the product.
        Combines bid and ask data.
        """
        buy_orders = self._state.order_depths[self.product].buy_orders
        sell_orders = self._state.order_depths[self.product].sell_orders

        total_value = 0  # Total (price * volume)
        total_volume = 0  # Total volume

        # Aggregate bid data
        for price, volume in buy_orders.items():
            total_value += price * volume
            total_volume += volume

        # Aggregate ask data
        for price, volume in sell_orders.items():
            total_value += price * abs(volume)
            total_volume += abs(volume)

        # Prevent division by zero
        if total_volume == 0:
            return (self.best_bid + self.best_ask) / 2.0  # Default to mid-price

        return total_value / total_volume

    @property
    def timestamp(self) -> int:
        return self._state.timestamp

    @property
    def order_depth(self) -> OrderDepth:
        return self._state.order_depths[self.product]

    @property
    def bids(self) -> List[Tuple[int, int]]:
        return list(self._state.order_depths[self.product].buy_orders.items())

    @property
    def asks(self) -> List[Tuple[int, int]]:
        return list(self._state.order_depths[self.product].sell_orders.items())

    @property
    def position(self) -> int:
        return self._state.position.get(self.product, 0)

    @property
    def possible_buy_amt(self) -> int:
        """
        The position limit is different for each product.
        We keep the logic that KELP is +/-50, vouchers are +/-200,
        and VOLCANIC_ROCK is +/-400.
        """
        if self.product == "KELP":
            return 50 - self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_9500":
            return 200 - self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_9750":
            return 200 - self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_10000":
            return 200 - self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_10250":
            return 200 - self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_10500":
            return 200 - self.position
        elif self.product == "VOLCANIC_ROCK":
            return 400 - self.position

    @property
    def possible_sell_amt(self) -> int:
        if self.product == "KELP":
            return 50 + self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_9500":
            return 200 + self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_9750":
            return 200 + self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_10000":
            return 200 + self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_10250":
            return 200 + self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_10500":
            return 200 + self.position
        elif self.product == "VOLCANIC_ROCK":
            return 400 + self.position

    @property
    def best_bid(self) -> int:
        bids = self._state.order_depths[self.product].buy_orders
        return max(bids.keys()) if bids else 0

    @property
    def best_ask(self) -> int:
        asks = self._state.order_depths[self.product].sell_orders
        return min(asks.keys()) if asks else float("inf")

    @property
    def vwap(self) -> float:
        """
        Compute the VWAP, combining all current bid/ask levels.
        This is a safer reference point than the naive mid-price.
        """
        buy_orders = self._state.order_depths[self.product].buy_orders
        sell_orders = self._state.order_depths[self.product].sell_orders

        total_value = 0
        total_volume = 0

        for price, volume in buy_orders.items():
            total_value += price * volume
            total_volume += volume

        for price, volume in sell_orders.items():
            total_value += price * abs(volume)
            total_volume += abs(volume)

        if total_volume == 0:
            return (self.best_bid + self.best_ask) / 2.0
        return total_value / total_volume

    def update_price_history(self, price: float) -> None:
        """Update price history with current price"""
        self.price_history.append(price)
        # Keep only the last 20 prices
        if len(self.price_history) > 20:
            self.price_history = self.price_history[-20:]

    def update_volatility_history(self, volatility: float) -> None:
        """Update volatility history"""
        self.volatility_history.append(volatility)
        # Keep only the last 10 volatility readings
        if len(self.volatility_history) > 10:
            self.volatility_history = self.volatility_history[-10:]

    def update_profit_history(self, profit: float) -> None:
        """Update profit history"""
        self.profit_history.append(profit)
        # Keep only the last 5 profit readings
        if len(self.profit_history) > 5:
            self.profit_history = self.profit_history[-5:]

    def get_price_trend(self) -> float:
        """Calculate price trend based on recent history"""
        if len(self.price_history) < 2:
            return 0

        # Simple linear regression for trend
        x = list(range(len(self.price_history)))
        y = self.price_history

        # Calculate means
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)

        # Calculate slope
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return 0

        slope = numerator / denominator
        return slope

    def get_volatility_trend(self) -> float:
        """Calculate volatility trend"""
        if len(self.volatility_history) < 2:
            return 0

        # Simple linear regression for trend
        x = list(range(len(self.volatility_history)))
        y = self.volatility_history

        # Calculate means
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)

        # Calculate slope
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return 0

        slope = numerator / denominator
        return slope

    def get_recent_profit_trend(self) -> float:
        """Calculate recent profit trend"""
        if len(self.profit_history) < 2:
            return 0

        # Simple linear regression for trend
        x = list(range(len(self.profit_history)))
        y = self.profit_history

        # Calculate means
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)

        # Calculate slope
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return 0

        slope = numerator / denominator
        return slope

    def update_IV_history(self, underlying_price) -> None:
        """Refresh the stored implied vol reading."""
        temp_history = IV_history.get_IV_history(self.product)
        temp_history.append(self.IV(underlying_price))
        IV_history.set_IV_history(self.product, temp_history[-self.ma_window :])

        # Also update our volatility history
        self.update_volatility_history(self.IV(underlying_price))

    def IV(self, underlying_price) -> float:
        return BlackScholes.implied_volatility(
            call_price=self.vwap,
            spot=underlying_price,
            strike=self.strike,
            time_to_expiry=self.tte,
        )

    def moving_average(self, underlying_price: int) -> float:
        """
        Simple average of the last few implied vol readings.
        If we have no stored history yet, just seed from current IV.
        """
        hist = IV_history.get_IV_history(self.product)
        if not hist:
            return self.IV(underlying_price)
        return sum(hist) / len(hist)

    @property
    def tte(self) -> float:
        """
        We have 6 days left to expiry. Each "day" is effectively chunked
        as you proceed in your simulation. We divide by 250 so that
        each day is treated as 1 "trading day" in annualized terms.
        """
        # The environment's timestamp goes up to ~1,000,000 per day,
        # so we do (6 - dayProgress) / 250.
        return (self.initial_time_to_expiry - (self.timestamp / 1_000_000)) / 252.0

class IV_history:
    def __init__(self):
        self.v_9500_IV_history = []
        self.v_9750_IV_history = []
        self.v_10000_IV_history = []
        self.v_10250_IV_history = []
        self.v_10500_IV_history = []

    def get_IV_history(self, product: str) -> List[float]:
        if product == "VOLCANIC_ROCK_VOUCHER_9500":
            return self.v_9500_IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_9750":
            return self.v_9750_IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_10000":
            return self.v_10000_IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_10250":
            return self.v_10250_IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_10500":
            return self.v_10500_IV_history
        return []

    def set_IV_history(self, product: str, IV_history: List[float]) -> None:
        if product == "VOLCANIC_ROCK_VOUCHER_9500":
            self.v_9500_IV_history = IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_9750":
            self.v_9750_IV_history = IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_10000":
            self.v_10000_IV_history = IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_10250":
            self.v_10250_IV_history = IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_10500":
            self.v_10500_IV_history = IV_history

IV_history = IV_history()

class Trade:
    """Trading strategies for different products."""

    @staticmethod
    def volcanic_rock(state: TradingState) -> List[Order]:
        """Generate trading orders for VOLCANIC_ROCK using a conservative mean-reversion strategy."""
        orders: List[Order] = []

        # Get current market data
        product = "VOLCANIC_ROCK"
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)

        # Calculate basic metrics
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = (
            min(order_depth.sell_orders.keys())
            if order_depth.sell_orders
            else float("inf")
        )
        mid_price = (
            (best_bid + best_ask) / 2 if best_bid and best_ask != float("inf") else None
        )

        if not mid_price:
            return orders

        # Calculate VWAP for mean reversion
        vwap = (
            sum(price * abs(qty) for price, qty in order_depth.buy_orders.items())
            / sum(abs(qty) for qty in order_depth.buy_orders.values())
            if order_depth.buy_orders
            else mid_price
        )

        # Conservative position limits
        max_position = 20  # Reduced from previous value
        min_position = -20

        # Calculate price deviation from VWAP
        price_deviation = (mid_price - vwap) / vwap

        # Trading thresholds
        entry_threshold = 0.002  # 0.2% deviation for entry
        exit_threshold = 0.001  # 0.1% deviation for exit

        # Position sizing based on deviation
        base_quantity = 5  # Reduced base quantity
        quantity = min(base_quantity, max_position - position)

        # Trading logic
        if price_deviation > entry_threshold and position < max_position:
            # Price is above VWAP - consider selling
            if best_ask < float("inf"):
                orders.append(Order(product, best_ask, -quantity))

        elif price_deviation < -entry_threshold and position > min_position:
            # Price is below VWAP - consider buying
            if best_bid > 0:
                orders.append(Order(product, best_bid, quantity))

        # Exit logic - more aggressive
        if position > 0 and price_deviation > exit_threshold:
            # Exit long position if price is above VWAP
            if best_ask < float("inf"):
                orders.append(Order(product, best_ask, -position))

        elif position < 0 and price_deviation < -exit_threshold:
            # Exit short position if price is below VWAP
            if best_bid > 0:
                orders.append(Order(product, best_bid, -position))

        return orders

    @staticmethod
    def voucher(state: Status, underlying_state: Status, strike: int) -> List[Order]:
        """
        Simplified volatility trading for 10000
        """
        orders = []

        # dont trade all strikes 
        if strike not in [9750]:
            return orders

        # Update price and volatility history
        state.update_IV_history(underlying_state.vwap)
        state.update_price_history(state.vwap)

        # Calculate current and previous implied volatility
        current_IV = BlackScholes.implied_volatility(
            call_price=state.vwap,
            spot=underlying_state.vwap,
            strike=strike,
            time_to_expiry=state.tte,
        )
        prev_IV = state.moving_average(underlying_state.vwap)

        # Get market trends
        price_trend = state.get_price_trend()
        vol_trend = state.get_volatility_trend()
        profit_trend = state.get_recent_profit_trend()

        # Base parameters
        base_threshold = 0.001
        max_position = 200

        # Adjust threshold based on volatility trend
        threshold = base_threshold
        if vol_trend > 0.01:
            threshold *= 1.2
        elif vol_trend < -0.01:
            threshold *= 0.8

        # Selling volatility (when current IV > previous IV + threshold)
        if current_IV > prev_IV + threshold:
            if state.bids and state.position > -max_position:
                # Calculate position room and base quantity
                position_room = max_position + state.position
                base_quantity = min(state.possible_sell_amt, state.bids[0][1])
                quantity = min(base_quantity, position_room)

                # Adjust quantity based on market conditions
                if abs(price_trend) > 0.1:
                    quantity = int(quantity * 0.7)
                if vol_trend > 0.01:
                    quantity = int(quantity * 0.8)
                if profit_trend < -1000:
                    quantity = int(quantity * 0.6)

                # Place sell order if quantity is positive
                if quantity > 0:
                    orders.append(Order(state.product, state.best_bid, -quantity))
                    state.last_trade_timestamp = state.timestamp
                    state.trade_count += 1

        # Buying volatility (when current IV < previous IV - threshold)
        elif current_IV < prev_IV - threshold:
            if state.asks and state.position < max_position:
                # Calculate position room and base quantity
                position_room = max_position - state.position
                base_quantity = min(state.possible_buy_amt, abs(state.asks[0][1]))
                quantity = min(base_quantity, position_room)

                # Adjust quantity based on market conditions
                if abs(price_trend) > 0.1:
                    quantity = int(quantity * 0.7)
                if vol_trend > 0.01:
                    quantity = int(quantity * 0.8)
                if profit_trend < -1000:
                    quantity = int(quantity * 0.6)

                # Place buy order if quantity is positive
                if quantity > 0:
                    orders.append(Order(state.product, state.best_ask, quantity))
                    state.last_trade_timestamp = state.timestamp
                    state.trade_count += 1

        return orders


class Trader:
    def __init__(self, params=None):
        # Rock
        self.volcanic_prices: List[float] = []

       
    def run(self, state: TradingState):

        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}    
        self.volcanic_prices: List[float] = []

        if Product.VOLCANIC_ROCK in state.order_depths:
            # underlying
            result[Product.VOLCANIC_ROCK] = Trade.volcanic_rock(state)
            underlying = Status(Product.VOLCANIC_ROCK, state)

            # voucher for 10000
            for strike in (0, 9750):
                key = f"{Product.VOLCANIC_ROCK}_VOUCHER_{strike}"
                if key in state.order_depths:
                    st = Status(key, state, strike)
                    result[key] = Trade.voucher(st, underlying, strike)
        else:
            logger.print("VOLCANIC_ROCK not found")




        traderData = jsonpickle.encode(traderObject)
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData