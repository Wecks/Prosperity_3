# Trading Strategies Documentation

This document outlines the key trading strategies implemented in the IMC Prosperity 3 competition.

## Table of Contents
- [Trading Strategies Documentation](#trading-strategies-documentation)
  - [Table of Contents](#table-of-contents)
  - [Market Making Strategies](#market-making-strategies)
    - [RainforestStrategy](#rainforeststrategy)
    - [KelpStrategy](#kelpstrategy)
  - [Statistical Arbitrage Strategies](#statistical-arbitrage-strategies)
    - [SquidinkJamsStrategy](#squidinkjamsstrategy)
    - [PicnicBasketStrategy](#picnicbasketstrategy)
  - [Options Strategies](#options-strategies)
    - [VolcanicRockVoucherStrategy](#volcanicrockvoucherstrategy)
  - [Conversion Strategies](#conversion-strategies)
    - [MagnificentMacaronsStrategy](#magnificentmacaronsstrategy)
  - [Strategy Optimization Results](#strategy-optimization-results)
    - [PicnicBasket Optimal Thresholds](#picnicbasket-optimal-thresholds)
    - [Alternative Real-World Thresholds](#alternative-real-world-thresholds)

## Market Making Strategies

### RainforestStrategy
A market making strategy for RAINFOREST_RESIN that determines fair value based on the mid-price of the order book.

- **Implementation**: Inherits from MarketMakingStrategy
- **Fair Value Calculation**: Based on current mid-price
- **Trade Logic**: Buys below and sells above calculated fair value
- **Position Management**: Includes soft and hard liquidation logic when positions reach limits

### KelpStrategy
Similar to RainforestStrategy but optimized for KELP's specific market characteristics.

- **Implementation**: Inherits from MarketMakingStrategy
- **Fair Value Calculation**: Uses mid-price as the true value
- **Market Characteristics**: Adapts to KELP's lower volatility profile

## Statistical Arbitrage Strategies

### SquidinkJamsStrategy
A sophisticated trading strategy for SQUID_INK and JAMS that uses advanced market microstructure principles.

- **Parameters**:
  - `take_width`: Threshold for taking aggressive orders
  - `clear_width`: Width around fair value for position clearing
  - `reversion_beta`: Mean reversion coefficient
  - `min_edge`: Minimum edge for passive orders

- **SQUID_INK Parameters**:
  - `reversion_beta`: -0.129
  - `min_edge`: 2.5

- **JAMS Parameters**:
  - `reversion_beta`: -0.200
  - `min_edge`: 2.0

- **Strategy Logic**:
  - Calculates fair value based on microstructure indicators
  - Takes aggressive orders when prices deviate significantly
  - Manages positions with mean reversion assumptions
  - Places passive orders at favorable prices

### PicnicBasketStrategy
An arbitrage strategy that exploits price differences between PICNIC_BASKET1, PICNIC_BASKET2, and their constituent products (CROISSANTS, JAMS, DJEMBES).

- **Basket Compositions**:
  - PICNIC_BASKET1 = 6 CROISSANTS + 3 JAMS + 1 DJEMBES
  - PICNIC_BASKET2 = 4 CROISSANTS + 2 JAMS

- **Optimized Thresholds (Backtest)**:
  - CROISSANTS: long=-180, short=150
  - JAMS: long=0, short=150
  - DJEMBES: long=-150, short=250
  - PICNIC_BASKET1: long=-60, short=130
  - PICNIC_BASKET2: long=-100, short=140

- **Strategy Logic**:
  - Calculates theoretical price difference between baskets and components
  - Goes long when difference < long threshold (underpriced)
  - Goes short when difference > short threshold (overpriced)

## Options Strategies

### VolcanicRockVoucherStrategy
A Black-Scholes based strategy for trading VOLCANIC_ROCK_VOUCHER variants against the underlying VOLCANIC_ROCK.

- **Implementation**: Uses Black-Scholes model to price option value
- **Parameters**:
  - `volatility`: Set to 0.2 (tunable parameter)
  - `days_to_expiration`: Tracks time until voucher expiration
  - `threshold`: Trading signal threshold (0.02)

- **Strategy Logic**:
  - Calculates theoretical option price based on Black-Scholes
  - Sells vouchers when they're overpriced relative to model
  - Buys vouchers when they're underpriced relative to model
  - Updates expiration timeline based on timestamp progression

## Conversion Strategies

### MagnificentMacaronsStrategy
A conversion strategy that leverages the MAGNIFICENT_MACARONS' unique conversion mechanics based on sunlight index.

- **Parameters**:
  - `base_csi`: 50.0 (baseline sunlight index)
  - `threshold`: 3 (deviation trigger)
  - `persistent_length`: 1000 (consecutive ticks needed)
  - `per_trade_size`: Dynamic based on sunlight deviation

- **Strategy Logic**:
  - Tracks sunlight index readings
  - Liquidates existing positions to neutral at the beginning
  - When sunlight persistently high (>53), buys and converts
  - When sunlight persistently low (<42), sells and converts negative
  - Trade size scales with sunlight deviation from baseline

## Strategy Optimization Results

Based on backtesting and optimization, the following parameter configurations have shown strong performance:

### PicnicBasket Optimal Thresholds

| Product | Long Threshold | Short Threshold | Notes |
|---------|---------------|----------------|-------|
| PICNIC_BASKET1 | -60 | 130 | Best backtest performance |
| PICNIC_BASKET2 | -100 | 140 | Best backtest performance |
| CROISSANTS | -180 | 150 | Best backtest performance |
| JAMS | 0 | 150 | Consistent across tests |
| DJEMBES | -150 | 250 | Best backtest performance |

### Alternative Real-World Thresholds

For shorter timeframes (100K timestamps), these thresholds may perform better:

| Product | Long Threshold | Short Threshold |
|---------|---------------|----------------|
| CROISSANTS | -100 | 150 |
| JAMS | 0 | 150 |
| DJEMBES | -30 | 250 |
| PICNIC_BASKET1 | -60 | 110 |
| PICNIC_BASKET2 | -60 | 120 |
