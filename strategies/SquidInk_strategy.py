from datamodel import OrderDepth, TradingState, Order, Symbol, ProsperityEncoder
from typing import Dict, List, Tuple, Any
import statistics
import json
import math

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

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

    def compress_listings(self, listings: dict[Symbol, Any]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [od.buy_orders, od.sell_orders] for symbol, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Any]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for ts in trades.values() for t in ts]

    def compress_observations(self, observations: Any) -> list[Any]:
        conversion_obs = {
            p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
            for p, o in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conversion_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for os in orders.values() for o in os]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        self.symbol = "SQUID_INK"
        self.price_history: List[float] = []
        self.window = 30
        self.max_position = 100
        self.base_step = 2
        self.volatility_threshold = 1.5
        self.hold_threshold = 0.5

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result = {self.symbol: []}
        conversions = 0

        order_depth: OrderDepth = state.order_depths.get(self.symbol, None)
        if not order_depth:
            logger.flush(state, result, conversions, "")
            return result, conversions, ""

        position = state.position.get(self.symbol, 0)

        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
        elif order_depth.buy_orders:
            mid_price = max(order_depth.buy_orders.keys())
        elif order_depth.sell_orders:
            mid_price = min(order_depth.sell_orders.keys())
        else:
            logger.flush(state, result, conversions, "")
            return result, conversions, ""

        self.price_history.append(mid_price)
        if len(self.price_history) > self.window:
            self.price_history.pop(0)

        if len(self.price_history) < self.window:
            logger.flush(state, result, conversions, "")
            return result, conversions, ""

        log_prices = [math.log(p) for p in self.price_history if p > 0]
        mean_log_price = statistics.mean(log_prices)
        std_log_price = statistics.stdev(log_prices)
        current_log_price = math.log(mid_price)

        z_score = (current_log_price - mean_log_price) / std_log_price
        allow_trading = abs(z_score) < self.volatility_threshold

        base_price = statistics.mean(self.price_history)
        buy_threshold = base_price - std_log_price
        sell_threshold = base_price + std_log_price

        # reduce exposure if z_score is very close to 0
        if abs(z_score) < self.hold_threshold and position != 0:
            side = 1 if position < 0 else -1
            result[self.symbol].append(Order(self.symbol, int(mid_price), side * min(abs(position), self.base_step)))
            logger.print(f"REDUCING position {side * min(abs(position), self.base_step)} SQUID_INK @ {mid_price}")

        elif allow_trading:
            # light pyramid when signal is mild
            if abs(z_score) < 1.0:
                step = self.base_step
            else:
                step = self.base_step * 2

            if mid_price < buy_threshold and position < self.max_position:
                volume = min(step, self.max_position - position)
                result[self.symbol].append(Order(self.symbol, int(mid_price), volume))
                logger.print(f"BUY {volume} SQUID_INK @ {mid_price}")

            elif mid_price > sell_threshold and position > -self.max_position:
                volume = min(step, self.max_position + position)
                result[self.symbol].append(Order(self.symbol, int(mid_price), -volume))
                logger.print(f"SELL {volume} SQUID_INK @ {mid_price}")

        trader_data = json.dumps({
            "mid_price": mid_price,
            "log_price": current_log_price,
            "z_score": z_score,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "position": position,
            "allow_trading": allow_trading
        })

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
