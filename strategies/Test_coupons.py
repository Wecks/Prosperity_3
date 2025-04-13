from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any
import jsonpickle
import numpy as np
import math
from math import log, sqrt
from statistics import NormalDist

# Logger obligatoire pour l'environnement
class Logger:
    def __init__(self):
        self.logs = []

    def print(self, *args, **kwargs):
        message = " ".join(str(a) for a in args)
        self.logs.append(message)

    def flush(self):
        print("LOGGER:" + "\n".join(self.logs))

logger = Logger()

# Produits 
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

# Paramètres
PARAMS = {p: {
    "mean_volatility": 0.16,
    "threshold": 0.00163,
    "strike": int(p[-4:]),
    "starting_time_to_expiry": 0.98,
    "std_window": 6,
    "zscore_threshold": 21,
    "exit_zscore": 5,
} for p in Product.ALL_COUPONS}
PARAMS[Product.VOLCANIC_ROCK_VOUCHER_10000]["mean_volatility"] = 0.1596

# Black-Scholes simplifié
class BlackScholes:
    @staticmethod
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    @staticmethod
    def black_scholes_call(spot, strike, tte, vol):
        if spot <= 0 or strike <= 0 or tte <= 0 or vol <= 0:
            return 0
        d1 = (log(spot / strike) + 0.5 * vol**2 * tte) / (vol * sqrt(tte))
        d2 = d1 - vol * sqrt(tte)
        return spot * BlackScholes.norm_cdf(d1) - strike * BlackScholes.norm_cdf(d2)

    @staticmethod
    def delta(spot, strike, tte, vol):
        if spot <= 0 or strike <= 0 or tte <= 0 or vol <= 0:
            return 0
        d1 = (log(spot / strike) + 0.5 * vol**2 * tte) / (vol * sqrt(tte))
        return NormalDist().cdf(d1)

    @staticmethod
    def implied_volatility(call_price, spot, strike, tte, tol=1e-5, max_iter=100):
        if spot <= 0 or strike <= 0 or tte <= 0:
            return 0.16
        low, high = 0.01, 3.0
        for _ in range(max_iter):
            mid = (low + high) / 2
            price = BlackScholes.black_scholes_call(spot, strike, tte, mid)
            if abs(price - call_price) < tol:
                return mid
            if price > call_price:
                high = mid
            else:
                low = mid
        return mid

class Trader:
    def __init__(self):
        self.params = PARAMS
        self.LIMIT = {
            Product.VOLCANIC_ROCK: 300,
            **{p: 600 for p in Product.ALL_COUPONS},
        }

    def get_mid_price(self, order_depth: OrderDepth, fallback: float) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            bid = max(order_depth.buy_orders.keys())
            ask = min(order_depth.sell_orders.keys())
            return (bid + ask) / 2
        return fallback

    def run(self, state: TradingState):
        traderData = jsonpickle.decode(state.traderData) if state.traderData else {}
        result: Dict[str, List[Order]] = {}
        conversions = 0

        underlying_product = Product.VOLCANIC_ROCK
        underlying_od = state.order_depths.get(underlying_product)
        underlying_mid = self.get_mid_price(underlying_od, traderData.get("under_mid", 10000))
        traderData["under_mid"] = underlying_mid

        total_delta = 0
        total_coupon_position = 0

        for coupon in Product.ALL_COUPONS:
            if coupon not in state.order_depths:
                continue

            od = state.order_depths[coupon]
            pos = state.position.get(coupon, 0)
            fallback_price = traderData.get(f"{coupon}_prev_price", 100)
            coupon_mid = self.get_mid_price(od, fallback_price)
            traderData[f"{coupon}_prev_price"] = coupon_mid

            params = self.params[coupon]
            tte = params["starting_time_to_expiry"] - (state.timestamp / 1e6 / 250)
            if tte <= 0:
                continue

            vol = BlackScholes.implied_volatility(coupon_mid, underlying_mid, params["strike"], tte)
            if vol <= 0:
                continue

            delta = BlackScholes.delta(underlying_mid, params["strike"], tte, vol)
            vols = traderData.setdefault(f"{coupon}_vols", [])
            vols.append(vol)
            if len(vols) > params["std_window"]:
                vols.pop(0)
            if len(vols) < params["std_window"]:
                continue

            vol_std = np.std(vols)
            if vol_std == 0:
                continue

            z = (vol - params["mean_volatility"]) / vol_std

            orders = []
            if z >= params["zscore_threshold"]:
                if od.buy_orders:
                    price = max(od.buy_orders)
                    qty = self.LIMIT[coupon] - abs(pos)
                    if qty > 0:
                        orders.append(Order(coupon, price, -qty))
            elif z <= -params["zscore_threshold"]:
                if od.sell_orders:
                    price = min(od.sell_orders)
                    qty = self.LIMIT[coupon] - abs(pos)
                    if qty > 0:
                        orders.append(Order(coupon, price, qty))
            elif abs(z) <= params.get("exit_zscore", 5) and pos != 0:
                # Prise de profit
                if pos > 0 and od.buy_orders:
                    price = max(od.buy_orders)
                    orders.append(Order(coupon, price, -pos))
                elif pos < 0 and od.sell_orders:
                    price = min(od.sell_orders)
                    orders.append(Order(coupon, price, -pos))

            if orders:
                result[coupon] = orders

            total_coupon_position += pos
            total_delta += delta * pos

        underlying_pos = state.position.get(underlying_product, 0)
        hedge_target = -int(total_delta)
        diff = hedge_target - underlying_pos
        hedge_orders = []

        if diff > 0 and underlying_od and underlying_od.sell_orders:
            ask = min(underlying_od.sell_orders)
            vol = min(diff, -underlying_od.sell_orders[ask], self.LIMIT[underlying_product] - underlying_pos)
            if vol > 0:
                hedge_orders.append(Order(underlying_product, ask, vol))
        elif diff < 0 and underlying_od and underlying_od.buy_orders:
            bid = max(underlying_od.buy_orders)
            vol = min(abs(diff), underlying_od.buy_orders[bid], self.LIMIT[underlying_product] + underlying_pos)
            if vol > 0:
                hedge_orders.append(Order(underlying_product, bid, -vol))

        if hedge_orders:
            result[underlying_product] = hedge_orders

        logger.print("Underlying mid:", underlying_mid)
        logger.print("Delta net:", total_delta)
        logger.print("Hedge target:", hedge_target, "Underlying pos:", underlying_pos)
        logger.flush()
        return result, conversions, jsonpickle.encode(traderData)
