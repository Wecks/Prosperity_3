import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self,
            state: TradingState,
            orders: dict[Symbol, list[Order]],
            conversions: int,
            trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3

        message = self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ])

        # CORRECTIONS ICI
        logger.print(message)  # stocke dans self.logs
        print(message)         # nécessaire pour le visualizer

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
        return [
            [listing.symbol, listing.product, listing.denomination]
            for listing in listings.values()
        ]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {
            sym: [od.buy_orders, od.sell_orders]
            for sym, od in order_depths.items()
        }

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        result: list[list[Any]] = []
        for arr in trades.values():
            for t in arr:
                result.append([
                    t.symbol,
                    t.price,
                    t.quantity,
                    t.buyer,
                    t.seller,
                    t.timestamp,
                ])
        return result

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv: dict[Symbol, list[Any]] = {}
        for prod, obs in observations.conversionObservations.items():
            conv[prod] = [
                obs.bidPrice,
                obs.askPrice,
                obs.transportFees,
                obs.exportTariff,
                obs.importTariff,
                obs.sugarPrice,       # Correction ici
                obs.sunlightIndex,    # et ici :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
            ]
        return [observations.plainValueObservations, conv]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        result: list[list[Any]] = []
        for arr in orders.values():
            for o in arr:
                result.append([o.symbol, o.price, o.quantity])
        return result

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "."

logger = Logger()


class Strategy:
    def __init__(self, symbol: Symbol, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders: list[Order] = []
        self.conversions = 0
        self.act(state)
        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        ...

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)
        od = state.order_depths[self.symbol]
        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft = (len(self.window) == self.window_size and sum(self.window) >= self.window_size/2 and self.window[-1])
        hard = (len(self.window) == self.window_size and all(self.window))

        max_buy = true_value - 1 if position > self.limit*0.5 else true_value
        min_sell = true_value + 1 if position < -self.limit*0.5 else true_value

        # Achat
        for price, vol in sell_orders:
            if to_buy > 0 and price <= max_buy:
                q = min(to_buy, -vol)
                self.buy(price, q)
                to_buy -= q
        if to_buy > 0 and hard:
            self.buy(true_value, to_buy//2)
        if to_buy > 0 and soft:
            self.buy(true_value-2, to_buy//2)
        if to_buy > 0 and buy_orders:
            pb = max(buy_orders, key=lambda t: t[1])[0]
            price = min(max_buy, pb+1)
            self.buy(price, to_buy)

        # Vente
        for price, vol in buy_orders:
            if to_sell > 0 and price >= min_sell:
                q = min(to_sell, vol)
                self.sell(price, q)
                to_sell -= q
        if to_sell > 0 and hard:
            self.sell(true_value, to_sell//2)
        if to_sell > 0 and soft:
            self.sell(true_value+2, to_sell//2)
        if to_sell > 0 and sell_orders:
            ps = min(sell_orders, key=lambda t: t[1])[0]
            price = max(min_sell, ps-1)
            self.sell(price, to_sell)

class AmethystsStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10_000

class StarfruitStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        od = state.order_depths[self.symbol]
        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())
        if buy_orders and sell_orders:
            pb = max(buy_orders, key=lambda t: t[1])[0]
            ps = min(sell_orders, key=lambda t: t[1])[0]
            return round((pb + ps) / 2)
        return 0

class MAGNIFICENT_MACARONSStrategy(Strategy):
    def act(self, state: TradingState) -> None:
        pos = state.position.get(self.symbol, 0)
        self.convert(-pos)
        obs = state.observations.conversionObservations.get(self.symbol)
        if not obs:
            return
        buy_price = obs.askPrice + obs.transportFees + obs.importTariff
        self.sell(max(int(obs.bidPrice - 0.1), int(buy_price + 0.80)), self.limit)

class Trader:
    def __init__(self) -> None:
        limits = {
            "AMETHYSTS": 20,
            "STARFRUIT": 20,
            "MAGNIFICENT_MACARONS": 75,
        }
        strat_map = {
            "AMETHYSTS": AmethystsStrategy,
            "STARFRUIT": StarfruitStrategy,
            "MAGNIFICENT_MACARONS": MAGNIFICENT_MACARONSStrategy,
        }
        self.strategies = {sym: cls(sym, limits[sym]) for sym, cls in strat_map.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders: dict[Symbol, list[Order]] = {}
        conversions = 0
        old = json.loads(state.traderData) if state.traderData else {}
        new_data: dict[Symbol, JSON] = {}

        for sym, strat in self.strategies.items():
            if sym in old:
                strat.load(old[sym])
            if sym in state.order_depths:
                o, c = strat.run(state)
                orders[sym] = o
                conversions += c
            new_data[sym] = strat.save()

        trader_data = json.dumps(new_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        print(logger.logs)  # <--- cette ligne est ajoutée
        return orders, conversions, trader_data

