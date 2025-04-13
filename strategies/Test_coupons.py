from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any
import jsonpickle
import numpy as np
import math
from math import log, sqrt
from statistics import NormalDist

# Définition des produits (uniquement ceux qui nous intéressent)
class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    ALL_COUPONS = [
        VOLCANIC_ROCK_VOUCHER_9500,
        VOLCANIC_ROCK_VOUCHER_9750,
        VOLCANIC_ROCK_VOUCHER_10000,
        VOLCANIC_ROCK_VOUCHER_10250,
        VOLCANIC_ROCK_VOUCHER_10500,
    ]

# Paramètres pour chaque Volcanic_Rock voucher
PARAMS = {
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.16,
        "threshold": 0.00163,
        "strike": 9500,
        "starting_time_to_expiry": 0.98,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.16,
        "threshold": 0.00163,
        "strike": 9750,
        "starting_time_to_expiry": 0.98,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.1596,
        "threshold": 0.00163,
        "strike": 10000,
        "starting_time_to_expiry": 0.98,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.16,
        "threshold": 0.00163,
        "strike": 10250,
        "starting_time_to_expiry": 0.98,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.16,
        "threshold": 0.00163,
        "strike": 10500,
        "starting_time_to_expiry": 0.98,
        "std_window": 6,
        "zscore_threshold": 21,
    },
}

# Black-Scholes avec méthode implied_volatility corrigée
class BlackScholes:
    @staticmethod
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        return spot * BlackScholes.norm_cdf(d1) - strike * BlackScholes.norm_cdf(d2)
    
    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)
    
    @staticmethod
    def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=100, tolerance=1e-5):
        low_vol = 0.01
        high_vol = 3.0
        for i in range(max_iterations):
            mid_vol = (low_vol + high_vol) / 2.0
            price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, mid_vol)
            diff = price - call_price
            if abs(diff) < tolerance:
                return mid_vol
            if diff > 0:
                high_vol = mid_vol
            else:
                low_vol = mid_vol
        return mid_vol

# Trader se concentrant uniquement sur Volcanic_Rock vouchers et leur hedging
class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {
            Product.VOLCANIC_ROCK: 300,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 600,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 600,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 600,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 600,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 600,
        }

    # Gestion d'un voucher : calcule le mid price, la vol implicite, le delta, envoie un ordre sur le voucher 
    # si le z-score de vol dépasse le seuil, et calcule l'ordre hedge correspondant pour l'underlying.
    def handle_coupon(self, underlying_product: str, coupon_product: str, state: TradingState, traderObject: Dict[str, Any]):
        params = self.params[coupon_product]
        coupon_position = state.position.get(coupon_product, 0)
        underlying_position = state.position.get(underlying_product, 0)
        underlying_od = state.order_depths[underlying_product]
        coupon_od = state.order_depths[coupon_product]

        # Calcul du mid price de l'underlying
        if underlying_od.buy_orders and underlying_od.sell_orders:
            under_best_bid = max(underlying_od.buy_orders.keys())
            under_best_ask = min(underlying_od.sell_orders.keys())
            underlying_mid = (under_best_bid + under_best_ask) / 2
        else:
            underlying_mid = traderObject.get(f"{underlying_product}_prev_mid", 10000)
        traderObject[f"{underlying_product}_prev_mid"] = underlying_mid

        # Calcul du mid price du voucher
        if coupon_od.buy_orders and coupon_od.sell_orders:
            coupon_best_bid = max(coupon_od.buy_orders.keys())
            coupon_best_ask = min(coupon_od.sell_orders.keys())
            coupon_mid = (coupon_best_bid + coupon_best_ask) / 2
        else:
            coupon_mid = traderObject.get(f"{coupon_product}_prev_price", 0)
        traderObject[f"{coupon_product}_prev_price"] = coupon_mid

        # Calcul du temps restant
        tte = params["starting_time_to_expiry"] - (state.timestamp / 1e6 / 250)
        # On utilise le prix mid du voucher comme call_price pour calculer la vol implicite
        vol = BlackScholes.implied_volatility(coupon_mid, underlying_mid, params["strike"], tte)
        delta = BlackScholes.delta(underlying_mid, params["strike"], tte, vol)

        if f"{coupon_product}_past_vol" not in traderObject:
            traderObject[f"{coupon_product}_past_vol"] = []
        traderObject[f"{coupon_product}_past_vol"].append(vol)
        if len(traderObject[f"{coupon_product}_past_vol"]) > params["std_window"]:
            traderObject[f"{coupon_product}_past_vol"].pop(0)
        if len(traderObject[f"{coupon_product}_past_vol"]) < params["std_window"]:
            return [], []

        std_vol = np.std(traderObject[f"{coupon_product}_past_vol"])
        vol_z_score = (vol - params["mean_volatility"]) / (std_vol if std_vol != 0 else 1e-9)

        orders_coupon: List[Order] = []
        orders_hedge: List[Order] = []

        # Stratégie de market making sur le voucher (similaire au code de base) :
        if vol_z_score >= params["zscore_threshold"]:
            if coupon_od.buy_orders:
                best_bid = max(coupon_od.buy_orders.keys())
                quantity = self.LIMIT[coupon_product] - abs(coupon_position)
                if quantity > 0:
                    orders_coupon.append(Order(coupon_product, best_bid, -quantity))
        elif vol_z_score <= -params["zscore_threshold"]:
            if coupon_od.sell_orders:
                best_ask = min(coupon_od.sell_orders.keys())
                quantity = self.LIMIT[coupon_product] - abs(coupon_position)
                if quantity > 0:
                    orders_coupon.append(Order(coupon_product, best_ask, quantity))

        final_coupon_position = coupon_position + sum(o.quantity for o in orders_coupon)
        target_underlying = -int(delta * final_coupon_position)
        diff = target_underlying - underlying_position

        if diff > 0:
            if underlying_od.sell_orders:
                best_ask = min(underlying_od.sell_orders.keys())
                quant = min(diff, -underlying_od.sell_orders[best_ask])
                quant = min(quant, self.LIMIT[underlying_product] - underlying_position)
                if quant > 0:
                    orders_hedge.append(Order(underlying_product, best_ask, quant))
        elif diff < 0:
            if underlying_od.buy_orders:
                best_bid = max(underlying_od.buy_orders.keys())
                quant = min(abs(diff), underlying_od.buy_orders[best_bid])
                quant = min(quant, self.LIMIT[underlying_product] + underlying_position)
                if quant > 0:
                    orders_hedge.append(Order(underlying_product, best_bid, -quant))

        return orders_coupon, orders_hedge

    # Fonction run se limitant au traitement des Volcanic_Rock vouchers et du hedging global en agrégeant les ordres hedge individuels.
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}
        conversions = 0

        underlying_hedge_orders = []
        for coupon in Product.ALL_COUPONS:
            if coupon in state.order_depths:
                orders_coupon, orders_hedge = self.handle_coupon(Product.VOLCANIC_ROCK, coupon, state, traderObject)
                if orders_coupon:
                    if coupon not in result:
                        result[coupon] = []
                    result[coupon].extend(orders_coupon)
                if orders_hedge:
                    underlying_hedge_orders.extend(orders_hedge)
        if underlying_hedge_orders:
            result[Product.VOLCANIC_ROCK] = underlying_hedge_orders

        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData
