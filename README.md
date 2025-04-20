# IMC Prosperity 3 Trading Algorithms

Trading algorithms developed for the IMC Prosperity 3 Global Trading Challenge 2025.

## Overview

This repository contains algorithmic trading strategies developed for the IMC Prosperity 3 competition, a 15-day global trading challenge where participants combine algorithmic and manual trading skills to maximize profit in a virtual market.

## Competition Details

**IMC Prosperity 3** is a 15-day trading simulation divided into 5 rounds of 72 hours each. Participants develop Python programs to implement trading strategies with the goal of securing maximum profit in the form of "SeaShells".

The competition involves:
- Developing automated trading strategies in Python
- Participating in manual trading exercises for bonus profits
- Competing against teams from around the world

## Repository Structure

```
IMC-Prosperity/
├── strategies/           # Trading strategy implementations
│   ├── manual/           # Some tables for manual trading decisions
│   ├── old/              # Old strategies files
│   ├── datamodel.py      # Data model classes provided by the competition
│   └── round5.py         # Current trading strategy with all asset types
├── logs/                 # Strategy execution logs and results
├── backtester/           # Backtesting and optimization tools
│   └── optimizers/       # Parameter optimization scripts
└── templates/            # Template files provided by the competition
```

## Implemented Strategies

The repository contains these trading strategies:

### RainforestStrategy and KelpStrategy
Market making strategies that determine true value based on mid-price and place buy/sell orders accordingly.

### SquidinkJamsStrategy
A sophisticated trading strategy for SQUID_INK and JAMS assets that uses market microstructure and mean reversion principles.

### PicnicBasketStrategy
Arbitrage trading strategy for PICNIC_BASKET1, PICNIC_BASKET2, CROISSANTS, JAMS, and DJEMBES based on the price difference between baskets and their constituent parts.

### VolcanicRockVoucherStrategy
Option pricing strategy for VOLCANIC_ROCK_VOUCHER assets using Black-Scholes model with volatility parameter.

### MagnificentMacaronsStrategy
Conversion strategy for MAGNIFICENT_MACARONS based on sunlight index readings.

## Getting Started

To run these strategies in the backtesting environment:

1. Install the backtester:
   ```bash
   pip install -U prosperity3bt
   ```

2. Run a backtest:
   ```bash
   ./backtester/backtest.sh round4 1
   ```

3. Visualize results:
   ```bash
   prosperity3bt strategies/round4.py 1 --vis
   ```

See [BACKTESTER.md](BACKTESTER.md) for detailed backtesting instructions.

## Optimization

The repository includes optimization scripts for fine-tuning strategy parameters:

```bash
# Optimize parameters for a specific product
./backtester/optimizers/optimize_jams.sh
```

Optimization results are saved to the `backtester/optimization_results/` directory.

## Results and Performance

Performance metrics are captured in log files that can be found in the `logs/` directory after algorithm execution.

For enhanced visualization of results, use the [IMC Prosperity 3 Visualizer](https://jmerle.github.io/imc-prosperity-3-visualizer/?/visualizer).

## License

MIT License

## Acknowledgments

- [IMC Trading](https://www.imc.com/) for organizing the Prosperity Trading Challenge
- [jmerle](https://github.com/jmerle) for the excellent backtester and visualization tools