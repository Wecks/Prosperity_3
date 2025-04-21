import json
import math
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from enum import IntEnum
from statistics import NormalDist
from typing import Any, TypeAlias, Deque
from math import log, sqrt
from typing import Optional


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

class GiftBasketStrategy(Strategy):
    # Seuils long / short centralisés pour chaque produit
    THRESHOLDS = {
        "JAMS":           (230, 355),
        "CROISSANTS":     (195, 485),
        "DJEMBES":        (325, 370),
        "PICNIC_BASKET1": (290, 355),
        "PICNIC_BASKET2": ( 50, 100),
    }

    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

    def act(self, state: TradingState) -> None:
        required = ["JAMS", "CROISSANTS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]
        if any(prod not in state.order_depths for prod in required):
            return

        # Calcul des mid-prices
        p_jams            = self.get_mid_price(state, "JAMS")
        p_croissants      = self.get_mid_price(state, "CROISSANTS")
        p_djembes         = self.get_mid_price(state, "DJEMBES")
        p_basket1         = self.get_mid_price(state, "PICNIC_BASKET1")
        p_basket2         = self.get_mid_price(state, "PICNIC_BASKET2")

        # Différentiel selon le produit que l’on trade
        if self.symbol == "PICNIC_BASKET2":
            diff = p_basket2 - 4 * p_croissants - 2 * p_jams
        else:
            diff = p_basket1 - 4 * p_jams - 6 * p_croissants - p_djembes

        # Récupération des seuils
        long_th, short_th = GiftBasketStrategy.THRESHOLDS[self.symbol]

        # Passage d’ordre
        if diff < long_th:
            self.go_long(state)
        elif diff > short_th:
            self.go_short(state)

    def go_long(self, state: TradingState) -> None:
        od = state.order_depths[self.symbol]
        if not od.sell_orders:
            return
        best_ask = min(od.sell_orders.keys())
        pos      = state.position.get(self.symbol, 0)
        qty      = self.limit - pos
        if qty > 0:
            self.buy(best_ask, qty)

    def go_short(self, state: TradingState) -> None:
        od = state.order_depths[self.symbol]
        if not od.buy_orders:
            return
        best_bid = max(od.buy_orders.keys())
        pos      = state.position.get(self.symbol, 0)
        qty      = self.limit + pos
        if qty > 0:
            self.sell(best_bid, qty)


class VolatilityManager:
    """Singleton class that manages volatility across all voucher strikes."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VolatilityManager, cls).__new__(cls)
            # Initialize master data structures
            cls._instance.initialized = False
            cls._instance.strike_implied_vols = {}  # K → σₖ mapping
            cls._instance.atm_strike = None         # At-the-money strike
            cls._instance.master_vol = None         # Master volatility (ATM or median)
            cls._instance.lambda_ = 0.94            # EWMA decay factor
            cls._instance.ewma_var = 0.0            # Current variance estimate
            cls._instance.last_price = None         # Last underlying price
        return cls._instance

    def update_implied_vol(self, strike: float, implied_vol: float) -> None:
        """Update implied vol for a specific strike."""
        self.strike_implied_vols[strike] = implied_vol
        if not self.initialized and len(self.strike_implied_vols) >= 3:  # Wait for at least 3 strikes
            self._update_master_vol()

    def _update_master_vol(self) -> None:
        """Update master volatility based on all available implied vols."""
        # Method 1: Use median of all implied vols
        all_vols = list(self.strike_implied_vols.values())
        self.master_vol = sorted(all_vols)[len(all_vols)//2]  # median

        # Method 2: If we know underlying price, use closest strike
        if self.atm_strike is not None:
            self.master_vol = self.strike_implied_vols.get(
                self.atm_strike, self.master_vol)

        # Initialize EWMA with this volatility
        self.ewma_var = (self.master_vol ** 2) / 252.0  # Daily variance
        self.initialized = True

    def set_atm_strike(self, underlying_price: float) -> None:
        """Update which strike is closest to at-the-money."""
        if not self.strike_implied_vols:
            return

        # Find closest strike to current underlying price
        strikes = list(self.strike_implied_vols.keys())
        self.atm_strike = min(strikes, key=lambda k: abs(k - underlying_price))

    def ewma_update(self, current_price: float) -> float:
        """Update EWMA variance estimate and return current volatility."""
        if not self.initialized:
            return self.master_vol if self.master_vol else 0.2  # Default

        if self.last_price is not None:
            ret = math.log(current_price / self.last_price)
            # Update variance estimate
            self.ewma_var = self.lambda_ * self.ewma_var + (1 - self.lambda_) * ret * ret

        self.last_price = current_price
        # Return annualized volatility
        return math.sqrt(self.ewma_var * 252.0)

class VolcanicRockVoucherStrategy(SignalStrategy):
    """Strategy for trading Volcanic Rock Vouchers using Black-Scholes and implied volatility sharing."""
    vol_manager = VolatilityManager()

    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.cdf = NormalDist().cdf
        self.strike_price = float(symbol.split('_')[-1])
        self.days_to_expiration = 7 - 4  # Round 5: 2 trading days left (7-5), adjust as needed
        self.last_timestamp = None
        self.initialized = False

    def get_signal(self, state: TradingState) -> Signal | None:
        vr_price = self.get_mid_price(state, "VOLCANIC_ROCK")
        voucher_pr = self.get_mid_price(state, self.symbol)
        T = self.days_to_expiration / 365
        r = 0.0
        self.vol_manager.set_atm_strike(vr_price)

        # Compute implied vol on first tick for this voucher
        if not self.initialized:
            imp_vol = self.implied_volatility(
                market_price=voucher_pr,
                S=vr_price,
                K=self.strike_price,
                T=T,
                r=r
            )
            self.vol_manager.update_implied_vol(self.strike_price, imp_vol)
            self.initialized = True
            logger.print(f"Strike {self.strike_price} implied vol: {imp_vol:.2%}")

        # Use implied vol of closest strike
        if self.vol_manager.strike_implied_vols:
            closest_strike = min(self.vol_manager.strike_implied_vols.keys(), key=lambda k: abs(k - vr_price))
            self.volatility = self.vol_manager.strike_implied_vols[closest_strike]
            logger.print(f"Using volatility: {self.volatility:.2%} for strike {self.strike_price} (closest strike: {closest_strike})")
        else:
            self.volatility = 0.2
            logger.print(f"Using fallback volatility: {self.volatility:.2%} for strike {self.strike_price}")

        fair = self.black_scholes(vr_price, self.strike_price, T, r, self.volatility)
        logger.print(f"Strike {self.strike_price} fair: {fair:.2f}, market: {voucher_pr:.2f}")

        threshold = 0.001
        if voucher_pr > fair * (1 + threshold):
            return Signal.SHORT
        if voucher_pr < fair * (1 - threshold):
            return Signal.LONG
        return None

    def black_scholes(self, S, K, T, r, sigma) -> float:
        """Black-Scholes option pricing formula for call options."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        d1 = (math.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.cdf(d1) - K * math.exp(-r * T) * self.cdf(d2)

    def bs_vega(self, S, K, T, r, sigma) -> float:
        """Vega = ∂Price/∂σ under Black–Scholes."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
        phi = math.exp(-0.5 * d1*d1) / math.sqrt(2*math.pi)
        return S * phi * math.sqrt(T)

    def implied_volatility(self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float = 0.0,
        initial_vol: float = 0.2,
        tol: float = 1e-6,
        max_iter: int = 50
    ) -> float:
        """Newton–Raphson inversion of BS to find implied σ."""
        sigma = initial_vol
        for _ in range(max_iter):
            price = self.black_scholes(S, K, T, r, sigma)
            vega  = self.bs_vega(     S, K, T, r, sigma)
            diff  = price - market_price
            if abs(diff) < tol:
                break
            sigma -= diff / vega
            if sigma <= 0:
                sigma = tol
        return sigma


class MagnificentMacaronsStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        # record recent sunlight readings
        self.sun_history: deque[float] = deque(maxlen=1000)
        # static base CSI chosen from backtest
        self.base_csi: float = 50.0
        self.threshold: float = 3
        self.persistent_length: int = 1000     # consecutive ticks under effective CSI
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
        self.per_trade_size = int(1 + (abs(sun - 50)/5))


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
            if all(s < self.base_csi - self.threshold - 5 for s in last_slice):
                sell_qty = min(self.per_trade_size, self.limit + pos)
                if sell_qty > 0:
                    market_price = int(obs.askPrice + obs.transportFees + obs.importTariff)
                    self.sell(market_price, sell_qty)
                return

class VolcanicRockStrategy(Strategy):
    VOLCANIC_SMA_WINDOW = 50 #50
    VOLCANIC_THRESHOLD = 2 # 5
    VOLCANIC_ORDER_SIZE = 10 # 10
    VOLCANIC_STOP_LOSS_THRESHOLD = 10000000  # effectively disables stop-loss
    PRICE_HISTORY_MAXLEN = VOLCANIC_SMA_WINDOW + 10

    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.price_history: Deque[float] = deque(maxlen=self.PRICE_HISTORY_MAXLEN)

    def act(self, state: TradingState) -> None:
        od = state.order_depths.get(self.symbol)
        if not od:
            logger.print("VOLCANIC_ROCK not found")
            return

        ts = state.timestamp
        logger.print(f"\n--- Volcanic Rock @ {ts} ---")

        price = self.get_micro_price(od) or self.get_estimate_price(od)
        if price is None:
            logger.print("No price available for Volcanic Rock; skipping.")
            return

        logger.print(f"Price: {price:.2f}")
        self.price_history.append(price)

        if len(self.price_history) < self.VOLCANIC_SMA_WINDOW:
            return  # Not enough data yet

        sma = sum(list(self.price_history)[-self.VOLCANIC_SMA_WINDOW:]) / self.VOLCANIC_SMA_WINDOW
        pos = state.position.get(self.symbol, 0)
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        logger.print(f"SMA({self.VOLCANIC_SMA_WINDOW}): {sma:.2f}, Pos={pos}")

        target = pos
        stop = False

        if pos > 0 and price < sma - self.VOLCANIC_STOP_LOSS_THRESHOLD:
            logger.print("*** STOP‑LOSS LONG ***")
            target, stop = 0, True
        elif pos < 0 and price > sma + self.VOLCANIC_STOP_LOSS_THRESHOLD:
            logger.print("*** STOP‑LOSS SHORT ***")
            target, stop = 0, True

        if not stop:
            if price < sma - self.VOLCANIC_THRESHOLD:
                logger.print("→ SIGNAL LONG")
                target = +self.limit
            elif price > sma + self.VOLCANIC_THRESHOLD:
                logger.print("→ SIGNAL SHORT")
                target = -self.limit
            else:
                logger.print("→ HOLD")

        delta = target - pos
        if delta > 0 and best_ask is not None:
            qty = min(delta, self.VOLCANIC_ORDER_SIZE, self.limit - pos)
            if qty > 0:
                logger.print(f"BUY {qty}@{best_ask}")
                self.buy(best_ask, qty)
        elif delta < 0 and best_bid is not None:
            qty = min(-delta, self.VOLCANIC_ORDER_SIZE, self.limit + pos)
            if qty > 0:
                logger.print(f"SELL {qty}@{best_bid}")
                self.sell(best_bid, qty)

    def get_micro_price(self, od: OrderDepth) -> Optional[float]:
        if not od.buy_orders or not od.sell_orders:
            return None
        bid = max(od.buy_orders.keys())
        ask = min(od.sell_orders.keys())
        if bid >= ask:
            return (bid + ask) / 2
        bid_vol = od.buy_orders[bid]
        ask_vol = abs(od.sell_orders[ask])
        return (bid * ask_vol + ask * bid_vol) / (bid_vol + ask_vol)

    def get_estimate_price(self, od: OrderDepth) -> Optional[float]:
        bids = od.buy_orders
        asks = od.sell_orders
        if bids and asks:
            return (max(bids) + min(asks)) / 2
        elif bids:
            return float(max(bids))
        elif asks:
            return float(min(asks))
        return None

    def save(self) -> JSON:
        return list(self.price_history)

    def load(self, data: JSON) -> None:
        if data:
            self.price_history = deque(data, maxlen=self.PRICE_HISTORY_MAXLEN)


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
    def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-15):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            if diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class VolcanicRockVoucher10000Strategy(Strategy):
    """Trading IV‑driven pour VOLCANIC_ROCK_VOUCHER_10000."""
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.strike = 10000
        self.iv_history: deque[float] = deque(maxlen=10)

    def act(self, state: TradingState) -> None:
        od_v = state.order_depths.get(self.symbol)
        od_u = state.order_depths.get("VOLCANIC_ROCK")
        if not od_v or not od_u or not od_v.buy_orders or not od_v.sell_orders:
            return

        # mid prix sous-jacent
        bb_u, ba_u = max(od_u.buy_orders), min(od_u.sell_orders)
        spot = (bb_u + ba_u) / 2
        # mid prix voucher
        bb_v, ba_v = max(od_v.buy_orders), min(od_v.sell_orders)
        price = (bb_v + ba_v) / 2

        # T en années : ~3 jours de trading
        T = 3/252
        iv = BlackScholes.implied_volatility(price, spot, self.strike, T)
        prev_iv = sum(self.iv_history)/len(self.iv_history) if self.iv_history else iv
        self.iv_history.append(iv)

        pos = state.position.get(self.symbol, 0)
        thr = 0.002

        # vendre si IV en hausse
        if iv > prev_iv + thr and pos > -self.limit:
            qty = min(self.limit + pos, od_v.buy_orders[bb_v])
            if qty>0: self.sell(bb_v, qty)

        # acheter si IV en baisse
        elif iv < prev_iv - thr and pos < self.limit:
            qty = min(self.limit - pos, abs(od_v.sell_orders[ba_v]))
            if qty>0: self.buy(ba_v, qty)


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
            "VOLCANIC_ROCK_VOUCHER_10250": 0,
            "VOLCANIC_ROCK_VOUCHER_10500": 0,
            "MAGNIFICENT_MACARONS": 75,
        }

        self.strategies: dict[Symbol, Strategy] = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "VOLCANIC_ROCK": VolcanicRockStrategy,
            "RAINFOREST_RESIN": RainforestStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK": SquidinkJamsStrategy,
            "JAMS": SquidinkJamsStrategy,
            "CROISSANTS": PicnicBasketStrategy,
            "DJEMBES": PicnicBasketStrategy,
            "PICNIC_BASKET1": PicnicBasketStrategy,
            "PICNIC_BASKET2": GiftBasketStrategy,
            "VOLCANIC_ROCK_VOUCHER_9500": VolcanicRockVoucherStrategy,
            "VOLCANIC_ROCK_VOUCHER_9750": VolcanicRockVoucherStrategy,
            "VOLCANIC_ROCK_VOUCHER_10000": VolcanicRockVoucher10000Strategy,
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
