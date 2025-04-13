from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List
import jsonpickle
import numpy as np
import math


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP" 
    JAMS = "JAMS" 
    PICNIC_BASKET1 = "PICNIC_BASKET1" 
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    JAMS = "JAMS" 
    JAMS = "JAMS"




PARAMS = {
    Product.JAMS: {"take_orders_with_min_volume": 0, "take_orders_with_max_volume": 35}
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.JAMS: 60}

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if Product.JAMS in self.params and Product.JAMS in state.order_depths:
            JAMS_params = self.params[Product.JAMS]
            JAMS_limit = self.LIMIT[Product.JAMS]

            JAMS_position = (
                state.position[Product.JAMS] if Product.JAMS in state.position else 0
            )

            JAMS_order_depth = state.order_depths[Product.JAMS]

            best_bid = max(JAMS_order_depth.buy_orders.keys())
            best_bid_volume = JAMS_order_depth.buy_orders[best_bid]

            best_ask = min(JAMS_order_depth.sell_orders.keys())
            best_ask_volume = abs(JAMS_order_depth.sell_orders[best_ask])

            JAMS_orders = []
            if (
                best_bid_volume >= JAMS_params["take_orders_with_min_volume"]
                and best_bid_volume <= JAMS_params["take_orders_with_max_volume"]
            ):
                # sell to best bid
                trade_volume = min(best_bid_volume, JAMS_limit + JAMS_position)
                if trade_volume > 0:
                    JAMS_orders.append(Order(Product.JAMS, best_bid, -1* trade_volume))
            elif (
                best_ask_volume >= JAMS_params["take_orders_with_min_volume"]
                and best_ask_volume <= JAMS_params["take_orders_with_max_volume"]
            ):
                # buy from best ask
                trade_volume = min(best_ask_volume, JAMS_limit - JAMS_position)
                if trade_volume > 0: 
                    JAMS_orders.append(Order(Product.JAMS, best_ask, trade_volume))

            result[Product.JAMS] = JAMS_orders

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
