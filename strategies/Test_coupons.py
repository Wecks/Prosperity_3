from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List
import jsonpickle
import numpy as np
import math


class Product:
    AMETHYSTS = "AMETHYSTS"
    STARFRUIT = "STARFRUIT"
    ORCHIDS = "ORCHIDS"
    GIFT_BASKET = "GIFT_BASKET"
    CHOCOLATE = "CHOCOLATE"
    STRAWBERRIES = "STRAWBERRIES"
    VOLCANIK_ROCK = "VOLCANIK_ROCK"


PARAMS = {
    Product.VOLCANIK_ROCK: {"take_orders_with_min_volume": 0, "take_orders_with_max_volume": 35}
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.VOLCANIK_ROCK: 60}

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if Product.VOLCANIK_ROCK in self.params and Product.VOLCANIK_ROCK in state.order_depths:
            VOLCANIK_ROCK_params = self.params[Product.VOLCANIK_ROCK]
            VOLCANIK_ROCK_limit = self.LIMIT[Product.VOLCANIK_ROCK]

            VOLCANIK_ROCK_position = (
                state.position[Product.VOLCANIK_ROCK] if Product.VOLCANIK_ROCK in state.position else 0
            )

            VOLCANIK_ROCK_order_depth = state.order_depths[Product.VOLCANIK_ROCK]

            best_bid = max(VOLCANIK_ROCK_order_depth.buy_orders.keys())
            best_bid_volume = VOLCANIK_ROCK_order_depth.buy_orders[best_bid]

            best_ask = min(VOLCANIK_ROCK_order_depth.sell_orders.keys())
            best_ask_volume = abs(VOLCANIK_ROCK_order_depth.sell_orders[best_ask])

            VOLCANIK_ROCK_orders = []
            if (
                best_bid_volume >= VOLCANIK_ROCK_params["take_orders_with_min_volume"]
                and best_bid_volume <= VOLCANIK_ROCK_params["take_orders_with_max_volume"]
            ):
                # sell to best bid
                trade_volume = min(best_bid_volume, VOLCANIK_ROCK_limit + VOLCANIK_ROCK_position)
                if trade_volume > 0:
                    VOLCANIK_ROCK_orders.append(Order(Product.VOLCANIK_ROCK, best_bid, -1* trade_volume))
            elif (
                best_ask_volume >= VOLCANIK_ROCK_params["take_orders_with_min_volume"]
                and best_ask_volume <= VOLCANIK_ROCK_params["take_orders_with_max_volume"]
            ):
                # buy from best ask
                trade_volume = min(best_ask_volume, VOLCANIK_ROCK_limit - VOLCANIK_ROCK_position)
                if trade_volume > 0: 
                    VOLCANIK_ROCK_orders.append(Order(Product.VOLCANIK_ROCK, best_ask, trade_volume))

            result[Product.VOLCANIK_ROCK] = VOLCANIK_ROCK_orders

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData