from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List
import jsonpickle
import numpy as np
import math


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP" 
    CROISSANTS = "CROISSANTS" 
    PICNIC_BASKET1 = "PICNIC_BASKET1" 
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS" 
    CROISSANTS = "CROISSANTS"




PARAMS = {
    Product.CROISSANTS: {"take_orders_with_min_volume": 0, "take_orders_with_max_volume": 35}
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.CROISSANTS: 60}

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if Product.CROISSANTS in self.params and Product.CROISSANTS in state.order_depths:
            CROISSANTS_params = self.params[Product.CROISSANTS]
            CROISSANTS_limit = self.LIMIT[Product.CROISSANTS]

            CROISSANTS_position = (
                state.position[Product.CROISSANTS] if Product.CROISSANTS in state.position else 0
            )

            CROISSANTS_order_depth = state.order_depths[Product.CROISSANTS]

            best_bid = max(CROISSANTS_order_depth.buy_orders.keys())
            best_bid_volume = CROISSANTS_order_depth.buy_orders[best_bid]

            best_ask = min(CROISSANTS_order_depth.sell_orders.keys())
            best_ask_volume = abs(CROISSANTS_order_depth.sell_orders[best_ask])

            CROISSANTS_orders = []
            if (
                best_bid_volume >= CROISSANTS_params["take_orders_with_min_volume"]
                and best_bid_volume <= CROISSANTS_params["take_orders_with_max_volume"]
            ):
                # sell to best bid
                trade_volume = min(best_bid_volume, CROISSANTS_limit + CROISSANTS_position)
                if trade_volume > 0:
                    CROISSANTS_orders.append(Order(Product.CROISSANTS, best_bid, -1* trade_volume))
            elif (
                best_ask_volume >= CROISSANTS_params["take_orders_with_min_volume"]
                and best_ask_volume <= CROISSANTS_params["take_orders_with_max_volume"]
            ):
                # buy from best ask
                trade_volume = min(best_ask_volume, CROISSANTS_limit - CROISSANTS_position)
                if trade_volume > 0: 
                    CROISSANTS_orders.append(Order(Product.CROISSANTS, best_ask, trade_volume))

            result[Product.CROISSANTS] = CROISSANTS_orders

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
