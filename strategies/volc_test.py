from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Tuple, Any, Optional
import numpy as np # type: ignore
from statistics import NormalDist
from collections import deque
import string
import math
from math import log, sqrt
import jsonpickle
import json

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
        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class Product: 
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000  = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250  = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500  = "VOLCANIC_ROCK_VOUCHER_10500"

# ---------------------------- Constants ------------------------------------
VOLCANIC_ROCK = "VOLCANIC_ROCK"
POSITION_LIMITS = {VOLCANIC_ROCK: 400} # Position limit for Volcanic Rock

# Mean‑reversion parameters for Volcanic Rock
VOLCANIC_SMA_WINDOW = 50  # Lookback period for Simple Moving Average
VOLCANIC_PRICE_DEQUE_LEN = VOLCANIC_SMA_WINDOW + 10 # Max length for price history list
VOLCANIC_THRESHOLD = 5    # Price deviation from SMA to trigger entry/exit signal
VOLCANIC_ORDER_SIZE = 10  # Max size of individual orders placed

# Stop-Loss parameter for Volcanic Rock
# This threshold defines a significant adverse move relative to the SMA.
# It MUST be larger than VOLCANIC_THRESHOLD to allow normal trading.
# *** THIS VALUE REQUIRES TUNING BASED ON OBSERVED VOLATILITY ***
VOLCANIC_STOP_LOSS_THRESHOLD = 10000000 #got rid of stop loss

# ---------------------------- Helper funcs ----------------------------------

def calculate_micro_price(od: OrderDepth) -> Optional[float]:
    """
    Calculates the micro-price from the order depth.
    Micro-price is a volume-weighted average of the best bid and ask.
    Returns None if order depth is insufficient.
    """
    if not od or not od.buy_orders or not od.sell_orders:
        # logger.print("MicroPrice: Insufficient order depth data.") # Optional logging
        return None
    try:
        # Get best bid and ask prices
        best_bid_price = max(od.buy_orders.keys())
        best_ask_price = min(od.sell_orders.keys())

        # Handle crossed book (best bid >= best ask) - simply return mid-price
        if best_bid_price >= best_ask_price:
            # logger.print(f"MicroPrice: Crossed book detected (Bid {best_bid_price} >= Ask {best_ask_price}). Using mid-price.") # Optional
            return (best_bid_price + best_ask_price) / 2.0

        # Get volumes at best bid and ask
        bid_volume = od.buy_orders[best_bid_price]
        ask_volume = abs(od.sell_orders[best_ask_price]) # Sell orders have negative quantity

        # Calculate weighted average if total volume is significant
        total_volume = bid_volume + ask_volume
        if total_volume > 1e-9: # Use tolerance for float comparison
            micro_price = (best_bid_price * ask_volume + best_ask_price * bid_volume) / total_volume
            # logger.print(f"MicroPrice calculated: {micro_price:.2f}") # Optional
            return micro_price
        else:
            # logger.print("MicroPrice: Zero or negligible volume at best bid/ask. Using mid-price.") # Optional
            # Fallback to mid-price if volumes are zero or tiny
            return (best_bid_price + best_ask_price) / 2.0

    except (ValueError, KeyError, IndexError) as e:
        # Catch potential errors if keys don't exist (should be rare with initial checks)
        logger.print(f"MicroPrice: Error calculating micro-price: {e}")
        return None # Return None on error

def get_price_estimate(od: OrderDepth) -> Optional[float]:
    """
    Estimates the price from order depth.
    Uses mid-price if both bid and ask exist, otherwise best bid or best ask.
    Returns None if no price information is available.
    """
    if not od:
        # logger.print("PriceEstimate: No order depth provided.") # Optional
        return None

    best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
    best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

    if best_bid is not None and best_ask is not None:
        estimate = (best_bid + best_ask) / 2.0
        # logger.print(f"PriceEstimate: Using mid-price: {estimate:.2f}") # Optional
        return estimate
    elif best_bid is not None:
        # logger.print(f"PriceEstimate: Using best bid: {float(best_bid):.2f}") # Optional
        return float(best_bid) # Cast to float for consistency
    elif best_ask is not None:
        # logger.print(f"PriceEstimate: Using best ask: {float(best_ask):.2f}") # Optional
        return float(best_ask) # Cast to float for consistency
    else:
        # logger.print("PriceEstimate: No bids or asks found in order depth.") # Optional
        return None # No price information available

def get_best_bid_ask(od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """
    Gets the best bid and ask prices (as integers) from the order depth.
    Returns (None, None) if either is missing.
    """
    best_bid = max(od.buy_orders.keys()) if od and od.buy_orders else None
    best_ask = min(od.sell_orders.keys()) if od and od.sell_orders else None
    # logger.print(f"Best Bid: {best_bid}, Best Ask: {best_ask}") # Optional detailed logging
    return best_bid, best_ask


class Trader:
    def __init__(self, params=None):
        # Rock
        self.volcanic_prices: List[float] = []
        self.volcanic_max_len: int = VOLCANIC_PRICE_DEQUE_LEN
        logger.print("Trader initialized (Volcanic Rock strategy with Stop-Loss).")
        logger.print(f"VOLCANIC PARAMS: SMA_Window={VOLCANIC_SMA_WINDOW}, Threshold={VOLCANIC_THRESHOLD}, StopThreshold={VOLCANIC_STOP_LOSS_THRESHOLD}, OrderSize={VOLCANIC_ORDER_SIZE}")

       
    def run(self, state: TradingState):

        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}    
        self.volcanic_prices: List[float] = []
        self.volcanic_max_len = VOLCANIC_PRICE_DEQUE_LEN

        # ─── Volcanic Rock (mean‑reversion + stop‑loss) ─────────────────
        if Product.VOLCANIC_ROCK in state.order_depths:
            ts = state.timestamp
            logger.print(f"\n--- Volcanic Rock @ {ts} ---")

            # restore history from traderObject
            hist: List[float] = traderObject.get("volcanic_prices", [])
            hist = hist[-self.volcanic_max_len :]

            od = state.order_depths[Product.VOLCANIC_ROCK]
            # current price via micro‑price or mid‑price
            v_price = calculate_micro_price(od) or get_price_estimate(od)
            orders: List[Order] = []

            if v_price is not None:
                logger.print(f"Price: {v_price:.2f}")
                # update history
                hist.append(v_price)
                if len(hist) > self.volcanic_max_len:
                    hist.pop(0)

                pos   = state.position.get(Product.VOLCANIC_ROCK, 0)
                limit = POSITION_LIMITS[Product.VOLCANIC_ROCK]
                best_bid, best_ask = get_best_bid_ask(od)

                # default: hold
                target = pos
                stop   = False

                # only compute SMA once we have enough data
                if len(hist) >= VOLCANIC_SMA_WINDOW:
                    sma = sum(hist[-VOLCANIC_SMA_WINDOW:]) / VOLCANIC_SMA_WINDOW
                    logger.print(f"SMA({VOLCANIC_SMA_WINDOW}): {sma:.2f}, Pos={pos}")

                    # stop‑loss
                    if pos > 0 and v_price < sma - VOLCANIC_STOP_LOSS_THRESHOLD:
                        logger.print("*** STOP‑LOSS LONG ***")
                        target, stop = 0, True
                    elif pos < 0 and v_price > sma + VOLCANIC_STOP_LOSS_THRESHOLD:
                        logger.print("*** STOP‑LOSS SHORT ***")
                        target, stop = 0, True

                    # mean‑reversion entry/exit
                    if not stop:
                        if v_price < sma - VOLCANIC_THRESHOLD:
                            logger.print("→ SIGNAL LONG")
                            target = +limit
                        elif v_price > sma + VOLCANIC_THRESHOLD:
                            logger.print("→ SIGNAL SHORT")
                            target = -limit
                        else:
                            logger.print("→ HOLD")

                # build a single order up to VOLCANIC_ORDER_SIZE toward target
                delta = target - pos
                if delta > 0 and best_ask is not None:
                    qty = min(delta, VOLCANIC_ORDER_SIZE, limit - pos)
                    if qty > 0:
                        logger.print(f"BUY {qty}@{best_ask}")
                        orders.append(Order(Product.VOLCANIC_ROCK, best_ask, qty))

                elif delta < 0 and best_bid is not None:
                    qty = min(-delta, VOLCANIC_ORDER_SIZE, limit + pos)
                    if qty > 0:
                        logger.print(f"SELL {qty}@{best_bid}")
                        orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -qty))

            else:
                logger.print("No price available for Volcanic Rock; skipping.")

            # write back
            result[Product.VOLCANIC_ROCK] = orders
            traderObject["volcanic_prices"] = hist

        else:
            logger.print("VOLCANIC_ROCK not found")




        traderData = jsonpickle.encode(traderObject)
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData