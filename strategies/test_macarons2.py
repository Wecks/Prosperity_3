from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List
import string
import jsonpickle
import numpy as np
import math


class Product:
    AMETHYSTS = "AMETHYSTS"
    STARFRUIT = "STARFRUIT"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


PARAMS = {
    Product.AMETHYSTS: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 0,
    },
    Product.STARFRUIT: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "starfruit_min_edge": 2,
    },
    Product.MAGNIFICENT_MACARONS:{
        "make_edge": 0.2,      # base value 2
        "make_min_edge": 0.3,  # base value 1
        "make_probability": 0.1,  # base value 0.566
        "init_make_edge": 0.2,    # base value 2
        "min_edge": 0.2,          # base value 0.5
        "volume_avg_timestamp": 5,# base value 5
        "volume_bar": 35,         # base value 75
        "dec_edge_discount": 0.8, # base value 0.8
        "step_size": 0.5          # base value 0.5
    }
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.AMETHYSTS: 20,
            Product.STARFRUIT: 20,
            Product.MAGNIFICENT_MACARONS: 100
        }

        self.MAGNIFICENT_MACARONS_data = {
            "curr_edge": self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"],
            "volume_history": [],
            # "optimized": False  # Désactivé
        }

    # Returns buy_order_volume, sell_order_volume
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amt = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                qty = min(best_ask_amt, position_limit - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_order_volume += qty
                    order_depth.sell_orders[best_ask] += qty
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amt = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                qty = min(best_bid_amt, position_limit + position)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_order_volume += qty
                    order_depth.buy_orders[best_bid] -= qty
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amt = -order_depth.sell_orders[best_ask]
            if abs(best_ask_amt) <= adverse_volume and best_ask <= fair_value - take_width:
                qty = min(best_ask_amt, position_limit - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_order_volume += qty
                    order_depth.sell_orders[best_ask] += qty
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amt = order_depth.buy_orders[best_bid]
            if abs(best_bid_amt) <= adverse_volume and best_bid >= fair_value + take_width:
                qty = min(best_bid_amt, position_limit + position)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_order_volume += qty
                    order_depth.buy_orders[best_bid] -= qty
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))

        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(
                v for p, v in order_depth.buy_orders.items() if p >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent = min(sell_quantity, clear_quantity)
            if sent > 0:
                orders.append(Order(product, fair_for_ask, -sent))
                sell_order_volume += sent

        if position_after_take < 0:
            clear_quantity = sum(
                -v for p, v in order_depth.sell_orders.items() if p <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, -position_after_take)
            sent = min(buy_quantity, clear_quantity)
            if sent > 0:
                orders.append(Order(product, fair_for_bid, sent))
                buy_order_volume += sent

        return buy_order_volume, sell_order_volume

    def starfruit_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                p for p, v in order_depth.sell_orders.items()
                if abs(v) >= self.params[Product.STARFRUIT]["adverse_volume"]
            ]
            filtered_bid = [
                p for p, v in order_depth.buy_orders.items()
                if abs(v) >= self.params[Product.STARFRUIT]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None

            if mm_ask is None or mm_bid is None:
                mmmid = ((best_ask + best_bid) / 2
                         if "starfruit_last_price" not in traderObject
                         else traderObject["starfruit_last_price"])
            else:
                mmmid = (mm_ask + mm_bid) / 2

            if "starfruit_last_price" in traderObject:
                last = traderObject["starfruit_last_price"]
                ret = (mmmid - last) / last
                pred = ret * self.params[Product.STARFRUIT]["reversion_beta"]
                fair = mmmid + mmmid * pred
            else:
                fair = mmmid

            traderObject["starfruit_last_price"] = mmmid
            return fair
        return None

    def make_amethyst_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        volume_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        baaf = min(
            [p for p in order_depth.sell_orders.keys() if p > fair_value + 1],
            default=fair_value + 3
        )
        bbbf = max(
            [p for p in order_depth.buy_orders.keys() if p < fair_value - 1],
            default=fair_value - 3
        )
        return self.market_make(
            Product.AMETHYSTS,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_vol, sell_vol = 0, 0
        if prevent_adverse:
            buy_vol, sell_vol = self.take_best_orders_with_adverse(
                product, fair_value, take_width,
                orders, order_depth,
                position, buy_vol, sell_vol,
                adverse_volume
            )
        else:
            buy_vol, sell_vol = self.take_best_orders(
                product, fair_value, take_width,
                orders, order_depth,
                position, buy_vol, sell_vol
            )
        return orders, buy_vol, sell_vol

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width,
            orders, order_depth,
            position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def make_starfruit_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        aaf = [
            p for p in order_depth.sell_orders.keys()
            if p >= round(fair_value + min_edge)
        ]
        bbf = [
            p for p in order_depth.buy_orders.keys()
            if p <= round(fair_value - min_edge)
        ]
        baaf = min(aaf) if aaf else round(fair_value + min_edge)
        bbbf = max(bbf) if bbf else round(fair_value - min_edge)
        return self.market_make(
            Product.STARFRUIT,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )

    def MAGNIFICENT_MACARONS_implied_bid_ask(
        self,
        observation: ConversionObservation,
    ) -> (float, float):
        return (
            observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1,
            observation.askPrice + observation.importTariff + observation.transportFees
        )

    def MAGNIFICENT_MACARONS_adap_edge(
        self,
        timestamp: int,
        observation: ConversionObservation,
        position: int
    ) -> float:
        if timestamp == 0:
            self.MAGNIFICENT_MACARONS_data["curr_edge"] = self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"]
            return self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"]
        curr_edge = self.MAGNIFICENT_MACARONS_data["curr_edge"]
        implied_bid, implied_ask = self.MAGNIFICENT_MACARONS_implied_bid_ask(observation)
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1

        if aggressive_ask > implied_ask:
            return self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"]

        self.MAGNIFICENT_MACARONS_data["volume_history"].append(abs(position))
        if len(self.MAGNIFICENT_MACARONS_data["volume_history"]) > self.params[Product.MAGNIFICENT_MACARONS]["volume_avg_timestamp"]:
            self.MAGNIFICENT_MACARONS_data["volume_history"].pop(0)
        if len(self.MAGNIFICENT_MACARONS_data["volume_history"]) < self.params[Product.MAGNIFICENT_MACARONS]["volume_avg_timestamp"]:
            return curr_edge

        volume_avg = np.mean(self.MAGNIFICENT_MACARONS_data["volume_history"])
        if volume_avg >= self.params[Product.MAGNIFICENT_MACARONS]["volume_bar"]:
            self.MAGNIFICENT_MACARONS_data["volume_history"] = []
            self.MAGNIFICENT_MACARONS_data["curr_edge"] = curr_edge + self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
            return self.MAGNIFICENT_MACARONS_data["curr_edge"]

        dec_cond = (
            self.params[Product.MAGNIFICENT_MACARONS]["dec_edge_discount"]
            * self.params[Product.MAGNIFICENT_MACARONS]["volume_bar"]
            * (curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"])
        )
        if dec_cond > volume_avg * curr_edge:
            new_edge = max(curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"],
                           self.params[Product.MAGNIFICENT_MACARONS]["min_edge"])
            self.MAGNIFICENT_MACARONS_data["volume_history"] = []
            self.MAGNIFICENT_MACARONS_data["curr_edge"] = new_edge
            return new_edge

        return curr_edge

    def MAGNIFICENT_MACARONS_arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        adap_edge: float,
        position: int
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        implied_bid, implied_ask = self.MAGNIFICENT_MACARONS_implied_bid_ask(observation)
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
        buy_vol = 0
        sell_vol = 0

        buy_cap = position_limit - position
        sell_cap = position_limit + position

        ask = implied_ask + adap_edge
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        if foreign_mid - 1 > implied_ask:
            ask = foreign_mid - 1

        edge = (ask - implied_ask) * self.params[Product.MAGNIFICENT_MACARONS]["make_probability"]

        for price in sorted(order_depth.sell_orders.keys()):
            if price > implied_bid - edge:
                break
            qty = min(-order_depth.sell_orders[price], buy_cap)
            if qty > 0:
                orders.append(Order(Product.MAGNIFICENT_MACARONS, round(price), qty))
                buy_vol += qty

        for price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if price < implied_ask + edge:
                break
            qty = min(order_depth.buy_orders[price], sell_cap)
            if qty > 0:
                orders.append(Order(Product.MAGNIFICENT_MACARONS, round(price), -qty))
                sell_vol += qty

        return orders, buy_vol, sell_vol

    def MAGNIFICENT_MACARONS_arb_make(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        edge: float,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        implied_bid, implied_ask = self.MAGNIFICENT_MACARONS_implied_bid_ask(observation)
        bid = implied_bid - edge
        ask = implied_ask + edge

        # if foreign mid suggests a more aggressive ask
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        if foreign_mid - 1 > implied_ask:
            ask = foreign_mid - 1

        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
        buy_qty = position_limit - (position + buy_order_volume)
        if buy_qty > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(bid), buy_qty))

        sell_qty = position_limit + (position - sell_order_volume)
        # only place ask if it's above implied_ask (guaranteed profit)
        if sell_qty > 0 and ask > implied_ask:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(ask), -sell_qty))

        return orders, buy_order_volume, sell_order_volume

    def MAGNIFICENT_MACARONS_arb_clear(self, position: int) -> int:
        return -position

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if (Product.MAGNIFICENT_MACARONS in state.order_depths
            and Product.MAGNIFICENT_MACARONS in state.observations.conversionObservations):

            pos = state.position.get(Product.MAGNIFICENT_MACARONS, 0)
            # conversions = self.MAGNIFICENT_MACARONS_arb_clear(pos)
            conversions = 0

            obs = state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS]
            edge = self.MAGNIFICENT_MACARONS_adap_edge(state.timestamp, obs, pos)

            od = state.order_depths[Product.MAGNIFICENT_MACARONS]
            orders_take, buy_v, sell_v = self.MAGNIFICENT_MACARONS_arb_take(od, obs, edge, pos)
            orders_make, _, _  = self.MAGNIFICENT_MACARONS_arb_make(od, obs, pos, edge, buy_v, sell_v)

            result[Product.MAGNIFICENT_MACARONS] = orders_take + orders_make

        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData
