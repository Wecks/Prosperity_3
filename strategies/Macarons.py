from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from datamodel import Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import List, Any, Dict
import jsonpickle
import numpy as np
import json


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base = [
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]
        base_length = len(self.to_json(base))
        max_item_length = (self.max_log_length - base_length) // 3

        payload = [
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]
        print(self.to_json(payload))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
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

    def compress_listings(
        self, listings: Dict[Symbol, Listing]
    ) -> List[List[Any]]:
        result = []
        for lst in listings.values():
            result.append([lst.symbol, lst.product, lst.denomination])
        return result

    def compress_order_depths(
        self, order_depths: Dict[Symbol, OrderDepth]
    ) -> Dict[Symbol, List[Any]]:
        result = {}
        for sym, od in order_depths.items():
            result[sym] = [od.buy_orders, od.sell_orders]
        return result

    def compress_trades(
        self, trades: Dict[Symbol, List[Trade]]
    ) -> List[List[Any]]:
        result = []
        for arr in trades.values():
            for tr in arr:
                result.append([
                    tr.symbol,
                    tr.price,
                    tr.quantity,
                    tr.buyer,
                    tr.seller,
                    tr.timestamp,
                ])
        return result

    def compress_observations(
        self, obs: Observation
    ) -> List[Any]:
        conv = {}
        for prod, o in obs.conversionObservations.items():
            conv[prod] = [
                o.bidPrice,
                o.askPrice,
                o.transportFees,
                o.exportTariff,
                o.importTariff,
                o.sugarPrice,
                o.sunlightIndex,
            ]
        return [obs.plainValueObservations, conv]

    def compress_orders(
        self, orders: Dict[Symbol, List[Order]]
    ) -> List[List[Any]]:
        result = []
        for arr in orders.values():
            for o in arr:
                result.append([o.symbol, o.price, o.quantity])
        return result

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."


logger = Logger()


class Product:
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

# Enhanced parameters for larger swings
PARAMS = {
    Product.MAGNIFICENT_MACARONS: {
        "make_edge": 1,
        "make_min_edge": 0.5,
        "make_probability": 0.566,
        "init_make_edge": 1,
        "min_edge": 0.3,
        "volume_avg_timestamp": 2,
        "volume_bar": 10,
        "dec_edge_discount": 0.8,
        "step_size": 0.5,
        # CSI settings
        "CSI": 0.7,
        "sunlight_window": 2,
        # Panic multipliers
        "panic_edge_mult": 3.0,
        "panic_size_mult": 2.0,
    }
}


class Trader:
    def __init__(self, params=None):
        self.params = params or PARAMS
        self.LIMIT = {Product.MAGNIFICENT_MACARONS: 75}
        self.CONVERSION_LIMIT = 10

    def macarons_implied_bid_ask(self, obs: ConversionObservation) -> (float, float):
        return (
            obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1,
            obs.askPrice + obs.importTariff + obs.transportFees,
        )

    def macarons_adap_edge(
        self,
        timestamp: int,
        curr_edge: float,
        position: int,
        state_obj: Dict,
    ) -> float:
        prod = Product.MAGNIFICENT_MACARONS
        if timestamp == 0:
            state_obj[prod]["curr_edge"] = self.params[prod]["init_make_edge"]
            return self.params[prod]["init_make_edge"]
        hist = state_obj[prod]["volume_history"]
        hist.append(abs(position))
        if len(hist) > self.params[prod]["volume_avg_timestamp"]:
            hist.pop(0)
        if len(hist) < self.params[prod]["volume_avg_timestamp"]:
            return curr_edge
        if not state_obj[prod]["optimized"]:
            avg = np.mean(hist)
            if avg >= self.params[prod]["volume_bar"]:
                state_obj[prod]["volume_history"] = []
                state_obj[prod]["curr_edge"] = curr_edge + self.params[prod]["step_size"]
                return curr_edge + self.params[prod]["step_size"]
            discount = (
                self.params[prod]["dec_edge_discount"] * self.params[prod]["volume_bar"] *
                (curr_edge - self.params[prod]["step_size"]) > avg * curr_edge
            )
            if discount:
                new_edge = max(curr_edge - self.params[prod]["step_size"], self.params[prod]["min_edge"])
                state_obj[prod]["volume_history"] = []
                state_obj[prod]["curr_edge"] = new_edge
                state_obj[prod]["optimized"] = True
                return new_edge
        state_obj[prod]["curr_edge"] = curr_edge
        return curr_edge

    def macarons_arb_take(
        self,
        od: OrderDepth,
        obs: ConversionObservation,
        edge: float,
        position: int,
    ) -> (List[Order], int, int):
        prod = Product.MAGNIFICENT_MACARONS
        orders, buy_v, sell_v = [], 0, 0
        ib, ia = self.macarons_implied_bid_ask(obs)
        buy_cap = self.LIMIT[prod] - position
        sell_cap = self.LIMIT[prod] + position
        for p in sorted(od.sell_orders):
            if p >= ib - edge:
                break
            q = min(abs(od.sell_orders[p]), buy_cap)
            if q > 0:
                orders.append(Order(prod, round(p), q))
                buy_v += q
        for p in sorted(od.buy_orders, reverse=True):
            if p <= ia + edge:
                break
            q = min(abs(od.buy_orders[p]), sell_cap)
            if q > 0:
                orders.append(Order(prod, round(p), -q))
                sell_v += q
        return orders, buy_v, sell_v

    def macarons_arb_make(
        self,
        od: OrderDepth,
        obs: ConversionObservation,
        position: int,
        edge: float,
        buy_v: int,
        sell_v: int,
        panic: bool = False,
    ) -> (List[Order], int, int):
        prod = Product.MAGNIFICENT_MACARONS
        orders = []
        ib, ia = self.macarons_implied_bid_ask(obs)
        bid, ask = ib - edge, ia + edge
        if panic:
            bid -= edge * (self.params[prod]["panic_edge_mult"] - 1)
            ask += edge * (self.params[prod]["panic_edge_mult"] - 1)
        size_mult = self.params[prod]["panic_size_mult"] if panic else 1.0
        bq = int((self.LIMIT[prod] - (position + buy_v)) * size_mult)
        sq = int((self.LIMIT[prod] + (position - sell_v)) * size_mult)
        if bq > 0:
            orders.append(Order(prod, round(bid), bq))
        if sq > 0:
            orders.append(Order(prod, round(ask), -sq))
        return orders, buy_v, sell_v

    def run(self, state: TradingState):
        prod = Product.MAGNIFICENT_MACARONS
        tobj = jsonpickle.decode(state.traderData) if state.traderData else {}
        if prod not in tobj:
            tobj[prod] = {
                "curr_edge": self.params[prod]["init_make_edge"],
                "volume_history": [],
                "optimized": False,
                "sunlight_history": [],
                "panic_mode": False,
            }
        pos = state.position.get(prod, 0)
        conv = max(min(-pos, self.CONVERSION_LIMIT), -self.CONVERSION_LIMIT)
        pos = 0
        obs = state.observations.conversionObservations.get(prod)
        panic = False
        if obs:
            sh = tobj[prod]["sunlight_history"]
            sh.append(obs.sunlightIndex)
            if len(sh) > self.params[prod]["sunlight_window"]:
                sh.pop(0)
            panic = (len(sh) == self.params[prod]["sunlight_window"] and
                     np.mean(sh) < self.params[prod]["CSI"])
            tobj[prod]["panic_mode"] = panic
        edge = self.macarons_adap_edge(state.timestamp, tobj[prod]["curr_edge"], pos, tobj)
        if panic:
            edge *= self.params[prod]["panic_edge_mult"]
        result = {}
        if obs:
            tks, bv, sv = self.macarons_arb_take(
                state.order_depths.get(prod, OrderDepth()), obs, edge, pos
            )
            mks, _, _ = self.macarons_arb_make(
                state.order_depths.get(prod, OrderDepth()), obs, pos, edge, bv, sv, panic
            )
            result[prod] = tks + mks
        td = jsonpickle.encode(tobj)
        logger.flush(state, result, conv, td)
        return result, conv, td