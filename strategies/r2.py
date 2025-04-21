from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Dict, List

class GiftBasketStrategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: List[Order] = []

    def act(self, state: TradingState) -> None:
        required_symbols = ["JAMS", "CROISSANTS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]
        if any(symbol not in state.order_depths for symbol in required_symbols):
            return

        chocolate = self.get_mid_price(state, "JAMS")
        strawberries = self.get_mid_price(state, "CROISSANTS")
        roses = self.get_mid_price(state, "DJEMBES")
        gift_basket1 = self.get_mid_price(state, "PICNIC_BASKET1")
        gift_basket2 = self.get_mid_price(state, "PICNIC_BASKET2")

        if self.symbol == "PICNIC_BASKET2":
            diff = gift_basket2 - 4 * strawberries - 2 * chocolate
        else:
            diff = gift_basket1 - 4 * chocolate - 6 * strawberries - roses

        long_threshold, short_threshold = {
            "JAMS": (230, 355),
            "CROISSANTS": (195, 485),
            "DJEMBES": (325, 370),
            "PICNIC_BASKET1": (290, 355),
            "PICNIC_BASKET2": (50, 100),
        }[self.symbol]

        if diff < long_threshold:
            self.go_long(state)
        elif diff > short_threshold:
            self.go_short(state)

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths.get(symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return 0.0
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        return (popular_buy_price + popular_sell_price) / 2

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        if not order_depth.sell_orders:
            return
        price = min(order_depth.sell_orders.keys())
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        if to_buy > 0:
            self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders:
            return
        price = max(order_depth.buy_orders.keys())
        position = state.position.get(self.symbol, 0)
        to_sell = self.limit + position
        if to_sell > 0:
            self.sell(price, to_sell)

class Trader:
    def __init__(self) -> None:
        self.strategies: Dict[str, GiftBasketStrategy] = {
            "JAMS": GiftBasketStrategy("JAMS", 350),
            "CROISSANTS": GiftBasketStrategy("CROISSANTS", 250),
            "DJEMBES": GiftBasketStrategy("DJEMBES", 60),
            "PICNIC_BASKET1": GiftBasketStrategy("PICNIC_BASKET1", 60),
            "PICNIC_BASKET2": GiftBasketStrategy("PICNIC_BASKET2", 100),
        }

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        orders = {}
        for symbol, strategy in self.strategies.items():
            strategy.orders = []
            strategy.act(state)
            orders[symbol] = strategy.orders
        return orders, 0, ""
