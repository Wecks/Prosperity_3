from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle

# Logger officiel du visualizer
import json
from typing import Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [od.buy_orders, od.sell_orders] for symbol, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv_obs = {
            p: [
                o.bidPrice,
                o.askPrice,
                o.transportFees,
                o.exportTariff,
                o.importTariff,
                o.sugarPrice,
                o.sunlightIndex,
            ]
            for p, o in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conv_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."

logger = Logger()

# --------------------------- Code de trading ---------------------------

from datamodel import OrderDepth, TradingState, Order
import jsonpickle

class Trader:
    def __init__(self):
        self.LIMIT = {"JAMS": 50}
        self.params = {
            "JAMS": {
                "take_orders_with_min_volume": 15,
                "reversion_beta": -0.5,
                "clear_width": 0.25,  # Réduit
                "min_volume_to_liquidate": 5  # Réduit
            }
        }

    def fair_value(self, order_depth: OrderDepth, traderObject: dict, product: str) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            best_bid_vol = abs(order_depth.buy_orders[best_bid])
            best_ask_vol = abs(order_depth.sell_orders[best_ask])

            mid = (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

            last_price_key = f"{product}_last_price"
            if last_price_key in traderObject:
                last = traderObject[last_price_key]
                ret = (mid - last) / last
                pred_ret = ret * self.params[product]["reversion_beta"]
                fair = mid + mid * pred_ret
            else:
                fair = mid

            traderObject[last_price_key] = mid
            return fair
        return None

    def generate_clear_position_orders(self, product, position, fair, order_depth, params):
        orders = []
        clear_width = params["clear_width"]
        min_vol = params["min_volume_to_liquidate"]

        if position > 0:
            for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                volume = order_depth.buy_orders[price]
                if price >= fair + clear_width and volume >= min_vol:
                    qty = min(position, volume)
                    orders.append(Order(product, price, -qty))
                    logger.print(f"[{product}] Sortie partielle: Vente {qty} à {price} (fair={fair})")
                    position -= qty
                    if position <= 0:
                        break

        elif position < 0:
            for price in sorted(order_depth.sell_orders.keys()):
                volume = abs(order_depth.sell_orders[price])
                if price <= fair - clear_width and volume >= min_vol:
                    qty = min(-position, volume)
                    orders.append(Order(product, price, qty))
                    logger.print(f"[{product}] Sortie partielle: Achat {qty} à {price} (fair={fair})")
                    position += qty
                    if position >= 0:
                        break

        return orders

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)

        orders = {}
        conversions = 0
        trader_data = ""
        product = "JAMS"

        if product in self.params and product in state.order_depths:
            params = self.params[product]
            limit = self.LIMIT[product]
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]

            fair = self.fair_value(order_depth, traderObject, product)
            logger.print(f"[{product}] Fair value estimée : {fair}")

            best_bid = max(order_depth.buy_orders.keys(), default=None)
            best_ask = min(order_depth.sell_orders.keys(), default=None)

            order_list = []

            if best_bid is not None:
                volume = order_depth.buy_orders[best_bid]
                if best_bid > fair and volume >= params["take_orders_with_min_volume"]:
                    qty = min(volume, limit + position)
                    if qty > 0:
                        logger.print(f"[{product}] Vente de {qty} à {best_bid} (fair={fair})")
                        order_list.append(Order(product, best_bid, -qty))

            if best_ask is not None:
                volume = abs(order_depth.sell_orders[best_ask])
                if best_ask < fair and volume >= params["take_orders_with_min_volume"]:
                    qty = min(volume, limit - position)
                    if qty > 0:
                        logger.print(f"[{product}] Achat de {qty} à {best_ask} (fair={fair})")
                        order_list.append(Order(product, best_ask, qty))

            # Ajout : sortie de position intelligente
            clear_orders = self.generate_clear_position_orders(product, position, fair, order_depth, params)
            order_list += clear_orders

            if order_list:
                orders[product] = order_list

        trader_data = jsonpickle.encode(traderObject)
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
