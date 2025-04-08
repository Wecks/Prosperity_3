from datamodel import OrderDepth, TradingState, Order
from personnal_example_program import Trader  # Use the personalized example program
# from example_program import Trader

# Create a sample TradingState object for testing
def create_sample_state():
    order_depths = {
        "PRODUCT1": OrderDepth(),
        "PRODUCT2": OrderDepth()
    }

    # Manually set buy_orders and sell_orders for each product
    order_depths["PRODUCT1"].buy_orders = {10: 5, 9: 3}
    order_depths["PRODUCT1"].sell_orders = {11: -4, 12: -6}

    order_depths["PRODUCT2"].buy_orders = {20: 2, 19: 1}
    order_depths["PRODUCT2"].sell_orders = {21: -3, 22: -5}

    state = TradingState(
        traderData="",
        timestamp=1000,
        listings={},
        order_depths=order_depths,
        own_trades={},
        market_trades={},
        position={},
        observations={}
    )
    return state

# Additional complex test cases
def create_complex_state():
    order_depths = {
        "PRODUCT1": OrderDepth(),
        "PRODUCT2": OrderDepth(),
        "PRODUCT3": OrderDepth()
    }

    # PRODUCT1: Overlapping buy and sell prices
    order_depths["PRODUCT1"].buy_orders = {10: 5, 11: 3}
    order_depths["PRODUCT1"].sell_orders = {11: -4, 12: -6}

    # PRODUCT2: No buy orders, only sell orders
    order_depths["PRODUCT2"].buy_orders = {}
    order_depths["PRODUCT2"].sell_orders = {20: -2, 21: -3}

    # PRODUCT3: No sell orders, only buy orders
    order_depths["PRODUCT3"].buy_orders = {30: 4, 29: 2}
    order_depths["PRODUCT3"].sell_orders = {}

    state = TradingState(
        traderData="",
        timestamp=2000,
        listings={},
        order_depths=order_depths,
        own_trades={},
        market_trades={},
        position={"PRODUCT1": 5, "PRODUCT2": -3, "PRODUCT3": 0},
        observations={}
    )
    return state

# Adjusted complex test cases to trigger the envelope strategy
def create_adjusted_complex_state():
    order_depths = {
        "PRODUCT1": OrderDepth(),
        "PRODUCT2": OrderDepth(),
        "PRODUCT3": OrderDepth()
    }

    # PRODUCT1: Best ask below lower envelope
    order_depths["PRODUCT1"].buy_orders = {10: 5, 9: 3}
    order_depths["PRODUCT1"].sell_orders = {8: -4, 12: -6}

    # PRODUCT2: Best bid above upper envelope
    order_depths["PRODUCT2"].buy_orders = {22: 2, 21: 1}
    order_depths["PRODUCT2"].sell_orders = {20: -3, 19: -5}

    # PRODUCT3: Prices within the envelope (no trades expected)
    order_depths["PRODUCT3"].buy_orders = {30: 4, 29: 2}
    order_depths["PRODUCT3"].sell_orders = {31: -3, 32: -5}

    state = TradingState(
        traderData="",
        timestamp=3000,
        listings={},
        order_depths=order_depths,
        own_trades={},
        market_trades={},
        position={"PRODUCT1": 0, "PRODUCT2": 0, "PRODUCT3": 0},
        observations={}
    )
    return state

# Instantiate the Trader class
trader = Trader()

# Create a sample state
state = create_sample_state()

# Call the run method and print the result
result, conversions, traderData = trader.run(state)
print("Result:", result)
print("Conversions:", conversions)
print("Trader Data:", traderData)

# Test the Trader with the complex state
complex_state = create_complex_state()
result, conversions, traderData = trader.run(complex_state)
print("Complex Test Result:", result)
print("Complex Test Conversions:", conversions)
print("Complex Test Trader Data:", traderData)

# Test the Trader with the adjusted complex state
adjusted_complex_state = create_adjusted_complex_state()
result, conversions, traderData = trader.run(adjusted_complex_state)
print("Adjusted Complex Test Result:", result)
print("Adjusted Complex Test Conversions:", conversions)
print("Adjusted Complex Test Trader Data:", traderData)