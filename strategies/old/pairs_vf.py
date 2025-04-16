# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 22:44:30 2025

@author: mathi
"""

from typing import Dict, List, Tuple, Any
from datamodel import Order, OrderDepth, TradingState, Symbol, Listing, Trade, Observation, ProsperityEncoder
import json
import numpy as np

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
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
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, List[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for trades_list in trades.values() for t in trades_list]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv_obs = {
            p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
            for p, o in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conv_obs]

    def compress_orders(self, orders: dict[Symbol, List[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for ol in orders.values() for o in ol]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        self.asset1 = "CROISSANTS"
        self.asset2 = "PICNIC_BASKET2"
        self.history1 = []
        self.history2 = []
        self.window = 60
        self.beta = 1.0
        self.z_entry = 1.0
        self.z_exit = 0.2

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2
        elif order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        elif order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        return 0.0

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        orders = {self.asset1: [], self.asset2: []}
        conversions = 0
        trader_data = {}

        d1 = state.order_depths.get(self.asset1)
        d2 = state.order_depths.get(self.asset2)
        if not d1 or not d2:
            logger.flush(state, orders, conversions, json.dumps(trader_data))
            return orders, conversions, json.dumps(trader_data)

        p1 = self.get_mid_price(d1)
        p2 = self.get_mid_price(d2)
        self.history1.append(p1)
        self.history2.append(p2)

        if len(self.history1) > self.window:
            self.history1.pop(0)
            self.history2.pop(0)

            spread = np.array(self.history1) - self.beta * np.array(self.history2)
            mean = spread.mean()
            std = spread.std()
            z = (spread[-1] - mean) / std if std else 0

            pos1 = state.position.get(self.asset1, 0)
            pos2 = state.position.get(self.asset2, 0)

            logger.print(f"Z-score: {z:.2f} | spread: {spread[-1]:.2f} | mean: {mean:.2f} | std: {std:.2f}")
            logger.print(f"Position1: {pos1} | Position2: {pos2}")

            if z > 2.3 * self.z_entry and pos1 > -30:
                orders[self.asset1].append(Order(self.asset1, max(d1.buy_orders), -10))
                orders[self.asset2].append(Order(self.asset2, min(d2.sell_orders), 10))
                logger.print("Short CROISSANTS, Long PICNIC_BASKET2")

            elif z < -2.3 * self.z_entry and pos1 < 30:
                orders[self.asset1].append(Order(self.asset1, min(d1.sell_orders), 10))
                orders[self.asset2].append(Order(self.asset2, max(d2.buy_orders), -10))
                logger.print("Long CROISSANTS, Short PICNIC_BASKET2")

            elif abs(z) < 0.15 * self.z_exit:
                if pos1 != 0:
                    price = max(d1.buy_orders) if pos1 > 0 else min(d1.sell_orders)
                    orders[self.asset1].append(Order(self.asset1, price, -pos1))
                if pos2 != 0:
                    price = max(d2.buy_orders) if pos2 > 0 else min(d2.sell_orders)
                    orders[self.asset2].append(Order(self.asset2, price, -pos2))
                logger.print("EXIT POSITIONS")

            trader_data = {
                "z_score": round(z, 3),
                "spread": round(spread[-1], 2),
                "mean": round(mean, 2),
                "std": round(std, 2),
                "pos1": pos1,
                "pos2": pos2,
            }

        logger.flush(state, orders, conversions, json.dumps(trader_data))
        return orders, conversions, json.dumps(trader_data)
