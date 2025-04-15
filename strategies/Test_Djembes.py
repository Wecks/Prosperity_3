from typing import List
import jsonpickle
from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
import numpy as np
import math


class Product:
    AMETHYSTS = "AMETHYSTS"
    STARFRUIT = "STARFRUIT"
    ORCHIDS = "ORCHIDS"
    DJEMBES = "DJEMBES"
    CHOCOLATE = "CHOCOLATE"
    STRAWBERRIES = "STRAWBERRIES"
    ROSES = "ROSES"


PARAMS = {Product.DJEMBES: {"min_width": 1, "max_width": 8, "mm_min_volume": 10}}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.DJEMBES: 60}

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if (
            Product.DJEMBES in self.params
            and Product.DJEMBES in state.order_depths
        ):
            basket_params = self.params[Product.DJEMBES]
            basket_limit = self.LIMIT[Product.DJEMBES]

            basket_position = (
                state.position[Product.DJEMBES]
                if Product.DJEMBES in state.position
                else 0
            )

            basket_order_depth = state.order_depths[Product.DJEMBES]

            mm_bids = [
                level
                for level in basket_order_depth.buy_orders.keys()
                if abs(basket_order_depth.buy_orders[level]) >= basket_params["mm_min_volume"]
            ]
            mm_asks = [
                level
                for level in basket_order_depth.sell_orders.keys()
                if abs(basket_order_depth.sell_orders[level]) >= basket_params["mm_min_volume"]
            ]
            if len(mm_bids) > 0 and len(mm_asks) > 0:
                best_mm_bid = max(mm_bids)
                best_mm_ask = min(mm_asks)
                # On calcule le milieu en tronquant au nombre entier
                mm_mid = int((best_mm_bid + best_mm_ask) / 2)

                num_levels = basket_params['max_width'] - basket_params['min_width'] + 1
                num_buy_levels = min(num_levels, basket_limit - basket_position)
                num_sell_levels = min(num_levels, basket_limit + basket_position)

                basket_orders = []
                for level in range(1, num_buy_levels + 1):
                    basket_orders.append(Order(Product.DJEMBES, mm_mid - level, 1))
                for level in range(1, num_sell_levels + 1):
                    basket_orders.append(Order(Product.DJEMBES, mm_mid + level, -1))
                
                result[Product.DJEMBES] = basket_orders
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData


# --- Correction de la définition de "days" ---
# Le framework (fichier __main__.py) attend une liste d'arguments dont les éléments
# devant être convertis en entier contiennent uniquement des chiffres.
# Ici, on remplace '3prosperity3bt' par '3' (par exemple) :

days = ['3', './strategies/Djembes.py', '3']

if __name__ == '__main__':
    # Exemple d'utilisation : on affiche les paramètres corrigés.
    print("Les paramètres 'days' sont corrigés :", days)
    # Ici, le Trader serait lancé en passant ces paramètres au framework.
