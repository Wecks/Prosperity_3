from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any
import jsonpickle
import numpy as np
import math
from math import log, sqrt
from statistics import NormalDist

# Logger utilisé par le simulateur
class Logger:
    def __init__(self):
        self.logs = []

    def print(self, *args, **kwargs):
        message = " ".join(str(a) for a in args)
        self.logs.append(message)

    def flush(self):
        print("LOGGER:" + "\n".join(self.logs))

logger = Logger()

# Définition des produits
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

# Paramètres tirés du code de base
PARAMS = {
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.16,
        "strike": 9500,
        "starting_time_to_expiry": 0.98,
        "std_window": 6,
        "zscore_threshold": 3.0,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.16,
        "strike": 9750,
        "starting_time_to_expiry": 0.98,
        "std_window": 6,
        "zscore_threshold": 3.0,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.1596,
        "strike": 10000,
        "starting_time_to_expiry": 0.98,
        "std_window": 6,
        "zscore_threshold": 3.0,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.16,
        "strike": 10250,
        "starting_time_to_expiry": 0.98,
        "std_window": 6,
        "zscore_threshold": 3.0,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.16,
        "strike": 10500,
        "starting_time_to_expiry": 0.98,
        "std_window": 6,
        "zscore_threshold": 3.0,
    },
}

class BlackScholes:
    @staticmethod
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    @staticmethod
    def black_scholes_call(spot, strike, tte, vol):
        if vol <= 0 or tte <= 0 or spot <= 0 or strike <= 0:
            return 0
        d1 = (log(spot / strike) + 0.5 * vol**2 * tte) / (vol * sqrt(tte))
        d2 = d1 - vol * sqrt(tte)
        return spot * BlackScholes.norm_cdf(d1) - strike * BlackScholes.norm_cdf(d2)

    @staticmethod
    def delta(spot, strike, tte, vol):
        if vol <= 0 or tte <= 0 or spot <= 0 or strike <= 0:
            return 0
        d1 = (log(spot / strike) + 0.5 * vol**2 * tte) / (vol * sqrt(tte))
        return NormalDist().cdf(d1)

    @staticmethod
    def implied_volatility(call_price, spot, strike, tte, tol=1e-5, max_iter=100):
        if call_price <= 0 or spot <= 0 or strike <= 0 or tte <= 0:
            return 0.2
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
        if order_depth and order_depth.buy_orders and order_depth.sell_orders:
            bid = max(order_depth.buy_orders.keys())
            ask = min(order_depth.sell_orders.keys())
            return (bid + ask) / 2
        return fallback

    def run(self, state: TradingState):
        traderData = jsonpickle.decode(state.traderData) if state.traderData else {}
        result: Dict[str, List[Order]] = {}
        conversions = 0

        underlying = Product.VOLCANIC_ROCK
        under_od = state.order_depths.get(underlying)
        under_mid = self.get_mid_price(under_od, traderData.get("under_mid", 10000))
        traderData["under_mid"] = under_mid

        total_delta = 0
        under_pos = state.position.get(underlying, 0)

        for coupon in Product.ALL_COUPONS:
            if coupon not in state.order_depths:
                continue

            od = state.order_depths[coupon]
            pos = state.position.get(coupon, 0)
            fallback = traderData.get(f"{coupon}_prev_price", 100)
            mid = self.get_mid_price(od, fallback)
            traderData[f"{coupon}_prev_price"] = mid

            params = self.params[coupon]
            tte = params["starting_time_to_expiry"] - (state.timestamp / 1e6 / 250)
            if tte <= 0 or under_mid <= 0 or mid <= 0:
                logger.print(f"⛔ Skip {coupon}: bad tte/mid")
                continue

            vol = BlackScholes.implied_volatility(mid, under_mid, params["strike"], tte)
            delta = BlackScholes.delta(under_mid, params["strike"], tte, vol)
            traderData.setdefault(f"{coupon}_vols", []).append(vol)
            if len(traderData[f"{coupon}_vols"]) > params["std_window"]:
                traderData[f"{coupon}_vols"].pop(0)
            if len(traderData[f"{coupon}_vols"]) < params["std_window"]:
                continue

            vol_std = np.std(traderData[f"{coupon}_vols"])
            z = (vol - params["mean_volatility"]) / (vol_std or 1e-9)

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
            elif abs(z) < 1 and pos != 0:
                # Sortie de position si le z-score s'est normalisé
                price = min(od.sell_orders) if pos > 0 else max(od.buy_orders)
                orders.append(Order(coupon, price, -pos))

            if orders:
                result[coupon] = orders

            total_delta += delta * pos

        hedge_target = -int(total_delta)
        diff = hedge_target - under_pos
        hedge_orders = []

        if diff > 0 and under_od.sell_orders:
            ask = min(under_od.sell_orders)
            vol = min(diff, -under_od.sell_orders[ask], self.LIMIT[underlying] - under_pos)
            if vol > 0:
                hedge_orders.append(Order(underlying, ask, vol))
        elif diff < 0 and under_od.buy_orders:
            bid = max(under_od.buy_orders)
            vol = min(abs(diff), under_od.buy_orders[bid], self.LIMIT[underlying] + under_pos)
            if vol > 0:
                hedge_orders.append(Order(underlying, bid, -vol))

        if hedge_orders:
            result[underlying] = hedge_orders

        logger.print("✅ RUN")
        logger.print("Delta total:", total_delta)
        logger.print("Target hedge:", hedge_target, "Current:", under_pos)
        logger.flush()

        return result, conversions, jsonpickle.encode(traderData)
