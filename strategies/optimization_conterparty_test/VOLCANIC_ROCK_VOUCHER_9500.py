import json
import math
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from enum import IntEnum
from statistics import NormalDist
from typing import Any, TypeAlias, Deque, Optional

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

# --- Counterparty Manager ---
class CounterpartyManager:
    def __init__(self, me: str):
        # your own trader ID string
        self.me = me
        # stats[symbol][party] = {count, volume, price_diff_sum, aggressions}
        self.stats: dict[Symbol, dict[str, dict[str, float]]] = {}

    def get_counterparty(self, trade: Trade) -> Optional[str]:
        # derive the other side of the trade
        if trade.buyer == self.me:
            return trade.seller
        if trade.seller == self.me:
            return trade.buyer
        # fallback
        return trade.buyer or trade.seller

    def update(self, symbol: Symbol, trade: Trade, mid_price: float) -> None:
        party = self.get_counterparty(trade)
        if not party:
            return

        sym_stats = self.stats.setdefault(symbol, {})
        s = sym_stats.setdefault(party, {
            "count": 0.0,
            "volume": 0.0,
            "price_diff_sum": 0.0,
            "aggressions": 0.0
        })

        s["count"] += 1
        s["volume"] += abs(trade.quantity)
        s["price_diff_sum"] += (trade.price - mid_price)
        # aggression = lifting ask or hitting bid
        is_aggressive = (
            (trade.buyer == party and trade.price >= mid_price) or
            (trade.seller == party and trade.price <= mid_price)
        )
        s["aggressions"] += 1 if is_aggressive else 0

    def get_profile(self, symbol: Symbol, party: str) -> Optional[dict[str, float]]:
        return self.stats.get(symbol, {}).get(party)

    def get_aggressiveness(self, symbol: Symbol, party: str) -> Optional[float]:
        p = self.get_profile(symbol, party)
        if not p or p["count"] < 5:
            return None
        return p["aggressions"] / p["count"]


# --- Strategy base class with cp_manager setter ---
class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.cp_manager: CounterpartyManager = None  # injected

    def set_counterparty_manager(self, manager: CounterpartyManager) -> None:
        self.cp_manager = manager

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

    def get_mid_price(self, state, symbol) -> float:
        od = state.order_depths[symbol]
        if not od.buy_orders or not od.sell_orders:
            # no market right now, return last known mid or 0
            return getattr(self, "_last_mid", 0.0)
        best_bid = max(od.buy_orders, key=od.buy_orders.get)
        best_ask = min(od.sell_orders, key=od.sell_orders.get)
        mid = (best_bid + best_ask) / 2
        self._last_mid = mid
        return mid

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

# --- Spread Multiplier Parameters (for optimizer) ---
SPREAD_MULT_UPPER = 0.5
SPREAD_MULT_LOWER = 0.3
SPREAD_MULT_HIGH = 1.0
SPREAD_MULT_LOW = 0.7

# --- MarketMakingStrategy with cp adjustment ---
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
        mid = true_value
        # compute counterparty aggressiveness
        last_trades = state.own_trades.get(self.symbol, [])
        if last_trades and self.cp_manager:
            last_cp = self.cp_manager.get_counterparty(last_trades[-1])
            agg = self.cp_manager.get_aggressiveness(self.symbol, last_cp)
        else:
            agg = None

        # determine spread multiplier (parameterized for optimizer)
        mult = 1.0
        if agg is not None:
            if agg > SPREAD_MULT_UPPER:
                mult = SPREAD_MULT_HIGH
            elif agg < SPREAD_MULT_LOWER:
                mult = SPREAD_MULT_LOW

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # ... liquidate logic unchanged ...

        # example adjusted top-of-book quoting
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        buy_price = min(true_value - 0, popular_buy_price + 1)  # base logic
        sell_price = max(true_value + 0, popular_sell_price - 1)
        buy_price = round((buy_price + mid) / 2 * mult)
        sell_price = round((sell_price + mid) / 2 * mult)

        if to_buy > 0:
            self.buy(buy_price, to_buy)
        if to_sell > 0:
            self.sell(sell_price, to_sell)

class RainforestStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        # return 10000
        return round(self.get_mid_price(state, self.symbol)) #More Flexible but a bit less performant in backtest - Same performance in live round2

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return round(self.get_mid_price(state, self.symbol))

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
            # "VOLCANIC_ROCK": KelpStrategy,
            # "RAINFOREST_RESIN": RainforestStrategy,
            # "KELP": KelpStrategy,
            # "SQUID_INK": KelpStrategy,
            # "JAMS": KelpStrategy,
            # "CROISSANTS": KelpStrategy,
            # "DJEMBES": KelpStrategy,
            # "PICNIC_BASKET1": KelpStrategy,
            # "PICNIC_BASKET2": KelpStrategy,
            "VOLCANIC_ROCK_VOUCHER_9500": KelpStrategy,
            # "VOLCANIC_ROCK_VOUCHER_9750": KelpStrategy,
            # "VOLCANIC_ROCK_VOUCHER_10000": KelpStrategy,
            # "VOLCANIC_ROCK_VOUCHER_10250": KelpStrategy,
            # "VOLCANIC_ROCK_VOUCHER_10500": KelpStrategy,
            # "MAGNIFICENT_MACARONS":KelpStrategy
        }.items()}

        self.cp_manager = CounterpartyManager("SUBMISSION")
        for strat in self.strategies.values():
            strat.set_counterparty_manager(self.cp_manager)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # 1) update counterparty stats before strategies
        for symbol, trades in state.own_trades.items():
            # skip if no order book or one side is empty
            od = state.order_depths.get(symbol)
            if not od or not od.buy_orders or not od.sell_orders:
                continue

            # safe to compute mid now
            mid = self.strategies[symbol].get_mid_price(state, symbol)
            for t in trades:
                self.cp_manager.update(symbol, t, mid)

        # 2) existing save/load and strategy execution
        old_trader_data = json.loads(state.traderData) if state.traderData else {}
        new_trader_data = {}
        orders = {}
        conversions = 0
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])
            if symbol in state.order_depths and state.order_depths[symbol].buy_orders and state.order_depths[symbol].sell_orders:
                strategy_orders, strategy_conversions = strategy.run(state)
                orders[symbol] = strategy_orders
                conversions += strategy_conversions
            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",",":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
