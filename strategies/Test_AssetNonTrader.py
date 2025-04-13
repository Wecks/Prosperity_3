from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List
import jsonpickle
import numpy as np
import math


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP" 
    SQUID_INK = "SQUID_INK" 
    PICNIC_BASKET1 = "PICNIC_BASKET1" 
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS" 
    JAMS = "JAMS"




PARAMS = {
    Product.SQUID_INK: {"take_orders_with_min_volume": 0, "take_orders_with_max_volume": 35}
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.SQUID_INK: 60}

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            SQUID_INK_params = self.params[Product.SQUID_INK]
            SQUID_INK_limit = self.LIMIT[Product.SQUID_INK]

            SQUID_INK_position = (
                state.position[Product.SQUID_INK] if Product.SQUID_INK in state.position else 0
            )

            SQUID_INK_order_depth = state.order_depths[Product.SQUID_INK]

            best_bid = max(SQUID_INK_order_depth.buy_orders.keys())
            best_bid_volume = SQUID_INK_order_depth.buy_orders[best_bid]

            best_ask = min(SQUID_INK_order_depth.sell_orders.keys())
            best_ask_volume = abs(SQUID_INK_order_depth.sell_orders[best_ask])

            SQUID_INK_orders = []
            if (
                best_bid_volume >= SQUID_INK_params["take_orders_with_min_volume"]
                and best_bid_volume <= SQUID_INK_params["take_orders_with_max_volume"]
            ):
                # sell to best bid
                trade_volume = min(best_bid_volume, SQUID_INK_limit + SQUID_INK_position)
                if trade_volume > 0:
                    SQUID_INK_orders.append(Order(Product.SQUID_INK, best_bid, -1* trade_volume))
            elif (
                best_ask_volume >= SQUID_INK_params["take_orders_with_min_volume"]
                and best_ask_volume <= SQUID_INK_params["take_orders_with_max_volume"]
            ):
                # buy from best ask
                trade_volume = min(best_ask_volume, SQUID_INK_limit - SQUID_INK_position)
                if trade_volume > 0: 
                    SQUID_INK_orders.append(Order(Product.SQUID_INK, best_ask, trade_volume))

            result[Product.SQUID_INK] = SQUID_INK_orders

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
