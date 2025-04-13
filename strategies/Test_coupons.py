from datamodel import OrderDepth, TradingState, Order, Symbol
from typing import List, Dict, Any
import jsonpickle
import numpy as np
from math import log, sqrt
from statistics import NormalDist
import json

# Logger officiel du visualizer
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {symbol: [od.buy_orders, od.sell_orders] for symbol, od in state.order_depths.items()},
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in state.own_trades.values() for t in arr],
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in state.market_trades.values() for t in arr],
            state.position,
            [state.observations.plainValueObservations, {
                p: [
                    o.bidPrice,
                    o.askPrice,
                    o.transportFees,
                    o.exportTariff,
                    o.importTariff,
                    o.sugarPrice,
                    o.sunlightIndex,
                ] for p, o in state.observations.conversionObservations.items()
            }],
        ]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        from datamodel import ProsperityEncoder
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

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

PARAMS = {p: {
    "mean_volatility": 0.16,
    "strike": int(p[-4:]),
    "starting_time_to_expiry": 247 / 250,
    "std_window": 30,
    "zscore_threshold": 5.1,
} for p in Product.ALL_COUPONS}
PARAMS[Product.VOLCANIC_ROCK_VOUCHER_10000]["mean_volatility"] = 0.1596

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, tte, vol):
        if vol < 1e-6 or tte < 1e-6 or spot <= 0 or strike <= 0:
            return 0.0
        d1 = (log(spot / strike) + 0.5 * vol ** 2 * tte) / (vol * sqrt(tte))
        d2 = d1 - vol * sqrt(tte)
        return spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)

    @staticmethod
    def implied_volatility(call_price, spot, strike, tte, max_iter=100, tol=1e-8):
        if call_price <= 0 or spot <= 0 or strike <= 0 or tte <= 1e-6:
            return 0.2
        low, high = 0.01, 1.0
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

    @staticmethod
    def delta(spot, strike, tte, vol):
        if vol < 1e-6 or tte < 1e-6 or spot <= 0 or strike <= 0:
            return 0.0
        d1 = (log(spot / strike) + 0.5 * vol ** 2 * tte) / (vol * sqrt(tte))
        return NormalDist().cdf(d1)

class Trader:
    def __init__(self):
        self.params = PARAMS
        self.LIMIT = {p: 600 for p in Product.ALL_COUPONS}
        self.LIMIT[Product.VOLCANIC_ROCK] = 300

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
        underlying_od = state.order_depths.get(underlying)
        under_mid = self.get_mid_price(underlying_od, traderData.get("under_mid", 10000))
        traderData["under_mid"] = under_mid

        best_coupon = None
        best_zscore = 0
        best_data = {}

        for coupon in Product.ALL_COUPONS:
            if coupon not in state.order_depths:
                continue

            od = state.order_depths[coupon]
            fallback = traderData.get(f"{coupon}_prev", 100)
            mid_price = self.get_mid_price(od, fallback)
            traderData[f"{coupon}_prev"] = mid_price

            params = self.params[coupon]
            tte = params["starting_time_to_expiry"] - (state.timestamp / 1e6 / 250)
            if tte <= 0:
                continue

            vol = BlackScholes.implied_volatility(mid_price, under_mid, params["strike"], tte)
            if vol <= 0.0101:
                continue

            vols = traderData.setdefault(f"{coupon}_vols", [])
            vols.append(vol)
            if len(vols) > params["std_window"]:
                vols.pop(0)
            if len(vols) < params["std_window"]:
                continue

            vol_std = np.std(vols)
            zscore = (vol - params["mean_volatility"]) / (vol_std if vol_std > 1e-6 else 1e-6)
            zscore = max(min(zscore, 10), -10)

            logger.print(f"{coupon} | vol: {vol:.4f} | z-score: {zscore:.2f} | mid: {mid_price:.2f}")

            if abs(zscore) > abs(best_zscore) and abs(zscore) > params["zscore_threshold"]:
                best_coupon = coupon
                best_zscore = zscore
                best_data = {
                    "od": od,
                    "pos": state.position.get(coupon, 0),
                    "vol": vol,
                    "mid": mid_price,
                    "params": params,
                    "tte": tte
                }

        if best_coupon:
            od = best_data["od"]
            pos = best_data["pos"]
            params = best_data["params"]
            vol = best_data["vol"]
            tte = best_data["tte"]

            logger.print(f"[Z-SELECT] Best coupon: {best_coupon} | Z: {best_zscore:.2f} | Vol: {vol:.4f} | Pos: {pos}")

            orders = []
            if best_zscore > 0:
                if od.buy_orders:
                    price = max(od.buy_orders)
                    qty = min(self.LIMIT[best_coupon] - abs(pos), od.buy_orders[price])
                    if qty > 0:
                        orders.append(Order(best_coupon, price, -qty))
            else:
                if od.sell_orders:
                    price = min(od.sell_orders)
                    qty = min(self.LIMIT[best_coupon] - abs(pos), -od.sell_orders[price])
                    if qty > 0:
                        orders.append(Order(best_coupon, price, qty))

            if orders:
                result[best_coupon] = orders

            delta = BlackScholes.delta(under_mid, params["strike"], tte, vol)
            total_delta = delta * (pos + (orders[0].quantity if orders else 0))
            hedge_target = -int(total_delta)
            under_pos = state.position.get(underlying, 0)
            diff = hedge_target - under_pos
            hedge_orders = []

            if diff > 0 and underlying_od and underlying_od.sell_orders:
                ask = min(underlying_od.sell_orders)
                qty = min(diff, -underlying_od.sell_orders[ask], self.LIMIT[underlying] - under_pos)
                if qty > 0:
                    hedge_orders.append(Order(underlying, ask, qty))
            elif diff < 0 and underlying_od and underlying_od.buy_orders:
                bid = max(underlying_od.buy_orders)
                qty = min(-diff, underlying_od.buy_orders[bid], self.LIMIT[underlying] + under_pos)
                if qty > 0:
                    hedge_orders.append(Order(underlying, bid, -qty))

            if hedge_orders:
                result[underlying] = hedge_orders

        trader_data = jsonpickle.encode(traderData)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
