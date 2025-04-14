from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle

# Logger officiel du visualizer
import json
from typing import Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [od.buy_orders, od.sell_orders] for symbol, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv_obs = {
            p: [
                o.bidPrice,
                o.askPrice,
                o.transportFees,
                o.exportTariff,
                o.importTariff,
                o.sugarPrice,
                o.sunlightIndex,
            ]
            for p, o in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conv_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

# --------------------------- Code de trading ---------------------------

class Trader:
    def __init__(self):
        self.LIMIT = {"JAMS": 50}
        self.params = {
            "JAMS": {"take_orders_with_min_volume": 0, "take_orders_with_max_volume": 35}
        }

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)

        orders = {}
        conversions = 0
        trader_data = ""

        product = "JAMS"

        if product in self.params and product in state.order_depths:
            params = self.params[product]
            limit = self.LIMIT[product]
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]

            best_bid = max(order_depth.buy_orders.keys(), default=None)
            best_ask = min(order_depth.sell_orders.keys(), default=None)

            order_list = []

            if best_bid is not None:
                best_bid_volume = order_depth.buy_orders[best_bid]
                if params["take_orders_with_min_volume"] <= best_bid_volume <= params["take_orders_with_max_volume"]:
                    sell_volume = min(best_bid_volume, limit + position)
                    if sell_volume > 0:
                        logger.print(f"Vente de {sell_volume} {product} à {best_bid}")
                        order_list.append(Order(product, best_bid, -sell_volume))

            if best_ask is not None:
                best_ask_volume = abs(order_depth.sell_orders[best_ask])
                if params["take_orders_with_min_volume"] <= best_ask_volume <= params["take_orders_with_max_volume"]:
                    buy_volume = min(best_ask_volume, limit - position)
                    if buy_volume > 0:
                        logger.print(f"Achat de {buy_volume} {product} à {best_ask}")
                        order_list.append(Order(product, best_ask, buy_volume))

            if order_list:
                orders[product] = order_list

        trader_data = jsonpickle.encode(traderObject)
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
