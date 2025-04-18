import json
import math
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from enum import IntEnum
from statistics import NormalDist
from typing import Any, TypeAlias, Deque

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

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
            # Access Listing object as an object with attributes instead of dict-like
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
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

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

        return value[:max_length - 3] + "..."


logger = Logger()

class TechnicalIndicators:
    @staticmethod
    def simple_moving_average(prices: list[float], window: int) -> list[float]:
        """Calculate Simple Moving Average (SMA) for a list of prices.

        Args:
            prices: List of historical prices
            window: The window size for the moving average

        Returns:
            A list of SMA values (same length as prices, with initial values containing None)
        """
        if len(prices) < window:
            return [None] * len(prices)

        result = [None] * (window - 1)
        for i in range(len(prices) - window + 1):
            result.append(sum(prices[i:i+window]) / window)

        return result

    @staticmethod
    def exponential_moving_average(prices: list[float], window: int) -> list[float]:
        """Calculate Exponential Moving Average (EMA) for a list of prices.

        Args:
            prices: List of historical prices
            window: The window size for the EMA

        Returns:
            A list of EMA values
        """
        if len(prices) < window:
            return [None] * len(prices)

        ema = [None] * (window - 1)
        # Start with SMA
        ema.append(sum(prices[:window]) / window)

        # EMA formula: EMA = Price(t) * k + EMA(y) * (1 – k)
        # where k = 2/(window + 1)
        k = 2 / (window + 1)

        for i in range(window, len(prices)):
            ema.append(prices[i] * k + ema[-1] * (1 - k))

        return ema

    @staticmethod
    def macd(prices: list[float], fast_window: int = 12, slow_window: int = 26, signal_window: int = 9) -> tuple[list[float], list[float], list[float]]:
        """Calculate the MACD (Moving Average Convergence Divergence) for a list of prices.

        Args:
            prices: List of historical prices
            fast_window: Window for the fast EMA (default: 12)
            slow_window: Window for the slow EMA (default: 26)
            signal_window: Window for the signal line (default: 9)

        Returns:
            tuple containing (macd_line, signal_line, histogram)
        """
        fast_ema = TechnicalIndicators.exponential_moving_average(prices, fast_window)
        slow_ema = TechnicalIndicators.exponential_moving_average(prices, slow_window)

        # Calculate MACD line
        macd_line = [None] * (slow_window - 1)
        for i in range(slow_window - 1, len(prices)):
            macd_line.append(fast_ema[i] - slow_ema[i])

        # Calculate signal line using EMA of MACD line
        valid_macd = [x for x in macd_line if x is not None]
        signal_line = [None] * (len(prices) - len(valid_macd))
        signal_line.extend(TechnicalIndicators.exponential_moving_average(valid_macd, signal_window))

        # Calculate histogram (MACD line - signal line)
        histogram = [None] * max(len(prices) - len(valid_macd) + signal_window - 1, slow_window - 1)
        for i in range(len(histogram), len(prices)):
            if macd_line[i] is not None and signal_line[i] is not None:
                histogram.append(macd_line[i] - signal_line[i])
            else:
                histogram.append(None)

        return macd_line, signal_line, histogram

    @staticmethod
    def relative_strength_index(prices: list[float], window: int = 14) -> list[float]:
        """Calculate the Relative Strength Index (RSI) for a list of prices.

        Args:
            prices: List of historical prices
            window: The window size for the RSI (default: 14)

        Returns:
            A list of RSI values (0-100)
        """
        if len(prices) <= window:
            return [None] * len(prices)

        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]

        # Calculate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [abs(delta) if delta < 0 else 0 for delta in deltas]

        # Prepare result array
        rsi = [None] * window

        # Calculate average gain and average loss for the first window
        avg_gain = sum(gains[:window]) / window
        avg_loss = sum(losses[:window]) / window

        # Calculate RSI
        for i in range(window, len(deltas)):
            # Update average gain and loss using the formula:
            # avgGain = ((previous avgGain) * (window - 1) + currentGain) / window
            avg_gain = (avg_gain * (window - 1) + gains[i]) / window
            avg_loss = (avg_loss * (window - 1) + losses[i]) / window

            if avg_loss == 0:  # Avoid division by zero
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))

        return rsi

    @staticmethod
    def bollinger_bands(prices: list[float], window: int = 20, num_std: float = 2.0) -> tuple[list[float], list[float], list[float]]:
        """Calculate Bollinger Bands for a list of prices.

        Args:
            prices: List of historical prices
            window: The window size for the moving average (default: 20)
            num_std: Number of standard deviations for the bands (default: 2.0)

        Returns:
            tuple containing (upper_band, middle_band, lower_band)
        """
        if len(prices) < window:
            return [None] * len(prices), [None] * len(prices), [None] * len(prices)

        # Calculate SMA (middle band)
        middle_band = TechnicalIndicators.simple_moving_average(prices, window)

        upper_band = [None] * (window - 1)
        lower_band = [None] * (window - 1)

        for i in range(window - 1, len(prices)):
            # Calculate standard deviation
            std_dev = math.sqrt(sum((prices[j] - middle_band[i])**2 for j in range(i-window+1, i+1)) / window)

            upper_band.append(middle_band[i] + num_std * std_dev)
            lower_band.append(middle_band[i] - num_std * std_dev)

        return upper_band, middle_band, lower_band

    @staticmethod
    def stochastic_oscillator(highs: list[float], lows: list[float], closes: list[float],
                             k_window: int = 14, d_window: int = 3) -> tuple[list[float], list[float]]:
        """Calculate Stochastic Oscillator for price data.

        Args:
            highs: List of high prices
            lows: List of low prix
            closes: List of closing prices
            k_window: Window size for %K (default: 14)
            d_window: Window size for %D (default: 3)

        Returns:
            tuple contenant (%K, %D)
        """
        if len(closes) < k_window:
            return [None] * len(closes), [None] * len(closes)

        k_values = [None] * (k_window - 1)

        for i in range(k_window - 1, len(closes)):
            window_low = min(lows[i-k_window+1:i+1])
            window_high = max(highs[i-k_window+1:i+1])

            if window_high - window_low == 0:
                k_values.append(50)  # Neutral value if no range
            else:
                # %K = 100 * (C - L14) / (H14 - L14)
                k_values.append(100 * (closes[i] - window_low) / (window_high - window_low))

        # Calculate %D by taking SMA of %K
        d_values = TechnicalIndicators.simple_moving_average(k_values, d_window)

        return k_values, d_values

    @staticmethod
    def average_true_range(highs: list[float], lows: list[float], closes: list[float], window: int = 14) -> list[float]:
        """Calculate Average True Range (ATR) for volatility measurement.

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            window: Window size for ATR (default: 14)

        Returns:
            List of ATR values
        """
        if len(closes) <= 1:
            return [None] * len(closes)

        # Calculate True Range series
        tr_values = [highs[0] - lows[0]]  # First TR is just the first day's range

        for i in range(1, len(closes)):
            # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)

        # Calculate ATR using Wilder's smoothing method
        atr_values = [None] * (window - 1)
        atr_values.append(sum(tr_values[:window]) / window)

        for i in range(window, len(tr_values)):
            # Wilder's smoothing: ATR = ((n-1) * prev_ATR + TR) / n
            atr_values.append((atr_values[-1] * (window - 1) + tr_values[i]) / window)

        return atr_values

    @staticmethod
    def on_balance_volume(closes: list[float], volumes: list[float]) -> list[float]:
        """Calculate On-Balance Volume (OBV) indicator.

        Args:
            closes: List of closing prices
            volumes: List of volume data

        Returns:
            List of OBV values
        """
        if len(closes) <= 1 or len(closes) != len(volumes):
            return [0] * len(closes)

        obv = [0]  # Initial OBV value

        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i-1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])

        return obv

    @staticmethod
    def money_flow_index(highs: list[float], lows: list[float], closes: list[float],
                        volumes: list[float], window: int = 14) -> list[float]:
        """Calculate Money Flow Index (MFI) indicator.

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            volumes: List of volume data
            window: Window size for MFI calculation (default: 14)

        Returns:
            List of MFI values (0-100)
        """
        if len(closes) < window or len(highs) != len(lows) or len(highs) != len(closes) or len(highs) != len(volumes):
            return [None] * len(closes)

        # Calculate typical price
        typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]

        # Calculate money flow
        money_flows = [typical_prices[i] * volumes[i] for i in range(len(typical_prices))]

        mfi_values = [None] * (window)

        for i in range(window, len(typical_prices)):
            positive_flow = 0
            negative_flow = 0

            # Determine positive and negative flows
            for j in range(i-window+1, i+1):
                if j > 0 and typical_prices[j] > typical_prices[j-1]:
                    positive_flow += money_flows[j]
                elif j > 0 and typical_prices[j] < typical_prices[j-1]:
                    negative_flow += money_flows[j]

            if negative_flow == 0:
                mfi_values.append(100)
            else:
                money_ratio = positive_flow / negative_flow
                mfi_values.append(100 - (100 / (1 + money_ratio)))

        return mfi_values

    @staticmethod
    def price_rate_of_change(prices: list[float], window: int = 14) -> list[float]:
        """Calculate Price Rate of Change (ROC) indicator.

        Args:
            prices: List of prices
            window: Window size for ROC calculation (default: 14)

        Returns:
            List of ROC values
        """
        if len(prices) <= window:
            return [None] * len(prices)

        roc_values = [None] * window

        for i in range(window, len(prices)):
            # ROC = ((Current Price / Price n periods ago) - 1) * 100
            roc_values.append(((prices[i] / prices[i-window]) - 1) * 100)

        return roc_values

    @staticmethod
    def get_vwap(state: TradingState, symbol: str, lookback: int = 0) -> float:
        """Calculate Volume Weighted Average Price (VWAP).

        Args:
            state: The current TradingState
            symbol: The symbol to calculate VWAP for
            lookback: How many timestamps to look back (0 means all available)

        Returns:
            VWAP value or None if insufficient data
        """
        trades = state.market_trades.get(symbol, [])
        if lookback > 0:
            current_timestamp = state.timestamp
            trades = [t for t in trades if current_timestamp - t.timestamp <= lookback]

        if not trades:
            return None

        total_value = sum(t.price * abs(t.quantity) for t in trades)
        total_volume = sum(abs(t.quantity) for t in trades)

        return total_value / total_volume if total_volume > 0 else None

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0

        self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class Signal(IntEnum):
    NEUTRAL = 0
    SHORT = 1
    LONG = 2

class SignalStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.signal = Signal.NEUTRAL

    @abstractmethod
    def get_signal(self, state: TradingState) -> Signal | None:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        new_signal = self.get_signal(state)
        if new_signal is not None:
            self.signal = new_signal

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        if self.signal == Signal.NEUTRAL:
            if position < 0:
                self.buy(self.get_buy_price(order_depth), -position)
            elif position > 0:
                self.sell(self.get_sell_price(order_depth), position)
        elif self.signal == Signal.SHORT:
            self.sell(self.get_sell_price(order_depth), self.limit + position)
        elif self.signal == Signal.LONG:
            self.buy(self.get_buy_price(order_depth), self.limit - position)

    def get_buy_price(self, order_depth: OrderDepth) -> int:
        return min(order_depth.sell_orders.keys())

    def get_sell_price(self, order_depth: OrderDepth) -> int:
        return max(order_depth.buy_orders.keys())

    def save(self) -> JSON:
        return self.signal.value

    def load(self, data: JSON) -> None:
        self.signal = Signal(data)

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data)

#TODO Finish that if usefull.
class MAStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.prices = deque(maxlen=100)
        self.sma_length = 10

    def act(self, state: TradingState) -> None:
        # Get the current mid price
        current_mid = self.get_mid_price(state, self.symbol)

        # Add new prices to our historical record
        self.prices.append(current_mid)

        sma = simple_moving_average(list(self.prices), self.sma_length)[-1]

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        #TODO : implement the logic.
        # max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        # min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        # for price, volume in sell_orders:
        #     if to_buy > 0 and price <= max_buy_price:
        #         quantity = min(to_buy, -volume)
        #         self.buy(price, quantity)
        #         to_buy -= quantity

        # if to_buy > 0:
        #     popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        #     price = min(max_buy_price, popular_buy_price + 1)
        #     self.buy(price, to_buy)

        # for price, volume in buy_orders:
        #     if to_sell > 0 and price >= min_sell_price:
        #         quantity = min(to_sell, volume)
        #         self.sell(price, quantity)
        #         to_sell -= quantity

        # if to_sell > 0:
        #     popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        #     price = max(min_sell_price, popular_sell_price - 1)
        #     self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.prices)

    def load(self, data: JSON) -> None:
        self.prices = deque(data, maxlen=100)


class RainforestStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        # return 10000
        return round(self.get_mid_price(state, self.symbol)) #More Flexible but a bit less performant in backtest - Same performance in live round2

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return round(self.get_mid_price(state, self.symbol))

class SquidinkJamsStrategy(Strategy):
    # Define default parameters - will be overridden by specific implementations
    DEFAULT_PARAMS = {
        "take_width": 5,
        "clear_width": 0.1,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.15,
        "min_edge": 2.0
    }

    # Symbol-specific parameters
    SYMBOL_PARAMS = {
        "SQUID_INK": {
            "take_width": 5,
            "clear_width": 0.1,
            "prevent_adverse": True,
            "adverse_volume": 15,
            "reversion_beta": -0.129,
            "min_edge": 2.5
        },
        "JAMS": {
            "take_width": 5,
            "clear_width": 0.1,
            "prevent_adverse": True,
            "adverse_volume": 15,
            "reversion_beta": -0.200,
            "min_edge": 2.0
        }
    }

    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.last_price = None  # To track the last average price

        # Set parameters based on the symbol
        if symbol in self.SYMBOL_PARAMS:
            self.params = self.SYMBOL_PARAMS[symbol]
        else:
            # For any other symbol, use default parameters
            self.params = self.DEFAULT_PARAMS.copy()

    def compute_fair_value(self, order_depth: OrderDepth) -> float:
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            adv_vol = self.params["adverse_volume"]
            filtered_ask = [price for price, qty in order_depth.sell_orders.items() if abs(qty) >= adv_vol]
            filtered_bid = [price for price, qty in order_depth.buy_orders.items() if abs(qty) >= adv_vol]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                if self.last_price is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = self.last_price
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            if self.last_price is not None:
                last_returns = (mmmid_price - self.last_price) / self.last_price
                pred_returns = last_returns * self.params["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            self.last_price = mmmid_price
            return fair
        return None

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            return
        order_depth = state.order_depths[self.symbol]
        current_position = state.position.get(self.symbol, 0)
        fair = self.compute_fair_value(order_depth)
        if fair is None:
            return
        take_width = self.params["take_width"]
        clear_width = self.params["clear_width"]
        buy_order_volume = 0
        sell_order_volume = 0

        # Take orders (buy side)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            ask_qty = abs(order_depth.sell_orders[best_ask])
            if best_ask <= fair - take_width:
                quantity = min(ask_qty, self.limit - current_position)
                if quantity > 0:
                    self.buy(best_ask, quantity)
                    buy_order_volume += quantity

        # Take orders (sell side)
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            bid_qty = order_depth.buy_orders[best_bid]
            if best_bid >= fair + take_width:
                quantity = min(bid_qty, self.limit + current_position)
                if quantity > 0:
                    self.sell(best_bid, quantity)
                    sell_order_volume += quantity

        position_after_take = current_position + buy_order_volume - sell_order_volume
        fair_bid = round(fair - clear_width)
        fair_ask = round(fair + clear_width)

        # Position management
        if position_after_take > 0:
            clear_quantity = sum(qty for price, qty in order_depth.buy_orders.items() if price >= fair_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            if clear_quantity > 0:
                self.sell(fair_ask, clear_quantity)
                sell_order_volume += clear_quantity

        if position_after_take < 0:
            clear_quantity = sum(abs(qty) for price, qty in order_depth.sell_orders.items() if price <= fair_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            if clear_quantity > 0:
                self.buy(fair_bid, clear_quantity)
                buy_order_volume += clear_quantity

        # Market making orders based on min_edge
        min_edge = self.params["min_edge"]
        aaf = [price for price in order_depth.sell_orders.keys() if price >= round(fair + min_edge)]
        bbf = [price for price in order_depth.buy_orders.keys() if price <= round(fair - min_edge)]
        baaf = min(aaf) if aaf else round(fair + min_edge)
        bbbf = max(bbf) if bbf else round(fair - min_edge)
        mm_buy_price = bbbf + 1
        mm_sell_price = baaf - 1
        remaining_buy = self.limit - (current_position + buy_order_volume)
        remaining_sell = self.limit + (current_position - sell_order_volume)

        if remaining_buy > 0:
            self.buy(mm_buy_price, remaining_buy)
        if remaining_sell > 0:
            self.sell(mm_sell_price, remaining_sell)

class PicnicBasketStrategy(SignalStrategy):
     # Class-level default thresholds
     DEFAULT_THRESHOLDS = {
        # BEST PERFORM IN REAL (100K TIMESTAMPS)
        #  "CROISSANTS": {"long": -100, "short": 150},
        #  "JAMS": {"long": 0, "short": 150},
        #  "DJEMBES": {"long": -30, "short": 250},
        #  "PICNIC_BASKET1": {"long": -60, "short": 110},
        #  "PICNIC_BASKET2": {"long": -60, "short": 120}

        # BEST PERFORM IN BACKTEST (6M TIMESTAMPS)
         "CROISSANTS": {"long": -180, "short": 150},
         "JAMS": {"long": 0, "short": 150},
         "DJEMBES": {"long": -150, "short": 250},
         "PICNIC_BASKET1": {"long": -60, "short": 130},
         "PICNIC_BASKET2": {"long": -100, "short": 140}
     }

     # Override class-level thresholds with values from command line
     THRESHOLDS = DEFAULT_THRESHOLDS.copy()

     def __init__(self, symbol: Symbol, limit: int) -> None:
         super().__init__(symbol, limit)
         # Each instance will use the class-level thresholds

     def get_signal(self, state: TradingState) -> Signal | None:
         if any(symbol not in state.order_depths for symbol in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]):
             return

         croissants = self.get_mid_price(state, "CROISSANTS")
         jams = self.get_mid_price(state, "JAMS")
         djembes = self.get_mid_price(state, "DJEMBES")
         picnic_basket1 = self.get_mid_price(state, "PICNIC_BASKET1")
         picnic_basket2 = self.get_mid_price(state, "PICNIC_BASKET2")

         diff = {
             "PICNIC_BASKET1": picnic_basket1 - 6 * croissants - 3 * jams - djembes,
             "PICNIC_BASKET2": picnic_basket2 - 4 * croissants - 2 * jams,
             "CROISSANTS": picnic_basket1 - 6 * croissants - 3 * jams - djembes,  # Using basket1 for CROISSANTS
             "JAMS": picnic_basket1 - 6 * croissants - 3 * jams - djembes,        # Using basket1 for JAMS
             "DJEMBES": picnic_basket1 - 6 * croissants - 3 * jams - djembes,     # Using basket1 for DJEMBES
         }[self.symbol]

         # Get thresholds for this symbol
         long_threshold = PicnicBasketStrategy.THRESHOLDS[self.symbol]["long"]
         short_threshold = PicnicBasketStrategy.THRESHOLDS[self.symbol]["short"]

         if diff < long_threshold:
             return Signal.LONG
         elif diff > short_threshold:
             return Signal.SHORT

         return None

class VolcanicRockVoucherStrategy(SignalStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.cdf = NormalDist().cdf

        # Extract strike price from symbol name (e.g., VOLCANIC_ROCK_VOUCHER_10000 -> 10000)
        self.strike_price = float(symbol.split('_')[-1])

        # Set volatility parameter (may need tuning based on market behavior)
        # Sigma is set so that the Black-Scholes value matches the initial coupon price at day 1 timestamp 0
        # TODO: Check what value matches the initial coupon price at day 1 timestamp 0.
        self.volatility = 0.2

        # Track days to expiration - starts at 7 days and decreases each round
        # 7 days is at beginning of round 1. At the end of round5 it will be 2 days left.
        self.days_to_expiration = 7-3  # Starting value - 3 because we are at round 4.
        self.last_timestamp = None

    def get_signal(self, state: TradingState) -> Signal | None:
        # Update days to expiration based on timestamp (one round = one day)
        if self.last_timestamp is not None and state.timestamp > self.last_timestamp:
            timestamp_diff = (state.timestamp - self.last_timestamp) / 100
            if timestamp_diff >= 1:  # If at least one round has passed
                self.days_to_expiration -= int(timestamp_diff)

        self.last_timestamp = state.timestamp

        # If expired or no market data, don't trade
        if self.days_to_expiration <= 0 or "VOLCANIC_ROCK" not in state.order_depths:
            return

        # Make sure we have market data for the underlying and the voucher
        if len(state.order_depths["VOLCANIC_ROCK"].buy_orders) == 0 or len(state.order_depths["VOLCANIC_ROCK"].sell_orders) == 0:
            return

        if len(state.order_depths[self.symbol].buy_orders) == 0 or len(state.order_depths[self.symbol].sell_orders) == 0:
            return

        # Get current prices
        volcanic_rock_price = self.get_mid_price(state, "VOLCANIC_ROCK")
        voucher_price = self.get_mid_price(state, self.symbol)
        # Convert days to years for Black-Scholes
        expiration_time = self.days_to_expiration / 365
        risk_free_rate = 0

        # Calculate expected price using Black-Scholes
        expected_price = self.black_scholes(
            volcanic_rock_price,  # Current price of VOLCANIC_ROCK (underlying)
            self.strike_price,    # Strike price from the voucher
            expiration_time,      # Time to expiration in years
            risk_free_rate,       # Risk-free rate
            self.volatility       # Volatility parameter
        )

        # Trading threshold - adjust these values based on market behavior
        threshold = 0.02  # Threshold for trading signal

        # Generate signals based on pricing difference
        if voucher_price > expected_price + threshold:
            return Signal.SHORT  # Voucher is overpriced
        elif voucher_price < expected_price - threshold:
            return Signal.LONG   # Voucher is underpriced

    def black_scholes(
        self,
        asset_price: float,
        strike_price: float,
        expiration_time: float,
        risk_free_rate: float,
        volatility: float,
    ) -> float:
        """Calculate the Black-Scholes price for an option."""
        d1 = (math.log(asset_price / strike_price) + (risk_free_rate + volatility ** 2 / 2) * expiration_time) / (volatility * math.sqrt(expiration_time))
        d2 = d1 - volatility * math.sqrt(expiration_time)
        return asset_price * self.cdf(d1) - strike_price * math.exp(-risk_free_rate * expiration_time) * self.cdf(d2)

class MagnificentMacaronsStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        # record recent sunlight readings
        self.sun_history: deque[float] = deque(maxlen=1000)
        # static base CSI chosen from backtest
        self.base_csi: float = 40.0
        self.threshold: float = 3
        self.persistent_length: int = 300     # consecutive ticks under effective CSI
        self.per_trade_size: int = 5        # max units per conversion

    def act(self, state: TradingState) -> None:
        obs = state.observations.conversionObservations.get(self.symbol)
        if not obs:
            return

        sun = obs.sunlightIndex
        self.sun_history.append(sun)

        # wait until enough history
        if len(self.sun_history) < self.persistent_length:
            return

        pos = state.position.get(self.symbol, 0)
        self.convert(-pos)
        self.per_trade_size = int(1 + (abs(sun - 10)/1))


        # 1) Persistent high sunlight: convert and sell on market
        if len(self.sun_history) >= self.persistent_length:
            last_slice = list(self.sun_history)[-self.persistent_length:]
            if all(s > self.base_csi + self.threshold for s in last_slice):
                buy_qty = min(self.per_trade_size, self.limit - pos)
                if buy_qty > 0:
                    market_price = int(obs.bidPrice)
                    self.buy(market_price, buy_qty)
                return

        # 2) Persistent low sunlight: convert negative and buy on market
        if len(self.sun_history) >= self.persistent_length:
            last_slice = list(self.sun_history)[-self.persistent_length:]
            if all(s < self.base_csi - self.threshold for s in last_slice):
                sell_qty = min(self.per_trade_size, self.limit + pos)
                obs = state.observations.conversionObservations.get(self.symbol)
                if sell_qty > 0 and obs:
                    market_price = int(obs.askPrice + obs.transportFees + obs.importTariff)
                    self.sell(market_price, sell_qty)
                return

class Trader:
    def __init__(self) -> None:
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "JAMS": 350,
            "CROISSANTS": 250,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200, #Expiration 7 days (=round) for all voucher,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS": 75,
        }

        self.strategies: dict[Symbol, Strategy] = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "RAINFOREST_RESIN": RainforestStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK": SquidinkJamsStrategy,
            "JAMS": SquidinkJamsStrategy,
            "CROISSANTS": PicnicBasketStrategy,
            "DJEMBES": PicnicBasketStrategy,
            "PICNIC_BASKET1": PicnicBasketStrategy,
            "PICNIC_BASKET2": PicnicBasketStrategy,
            "VOLCANIC_ROCK_VOUCHER_9500": VolcanicRockVoucherStrategy,
            "VOLCANIC_ROCK_VOUCHER_9750": VolcanicRockVoucherStrategy,
            "VOLCANIC_ROCK_VOUCHER_10000": VolcanicRockVoucherStrategy,
            "VOLCANIC_ROCK_VOUCHER_10250": VolcanicRockVoucherStrategy,
            "VOLCANIC_ROCK_VOUCHER_10500": VolcanicRockVoucherStrategy,
            "MAGNIFICENT_MACARONS":MagnificentMacaronsStrategy
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            if symbol in state.order_depths and len(state.order_depths[symbol].buy_orders) > 0 and len(state.order_depths[symbol].sell_orders) > 0:
                strategy_orders, strategy_conversions = strategy.run(state)
                orders[symbol] = strategy_orders
                conversions += strategy_conversions

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
