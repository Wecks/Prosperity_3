# IMC Prosperity 3 Trading Algorithms

## Competition Summary

Prosperity 3 was a three-week “holiday” in a tropical paradise hosted by IMC Trading, split across 5 intense rounds. Our team focused on building, testing, and refining algorithmic trading models through constant experimentation. We explored a range of strategies, with volatility trading at the core, complemented by market making, pairs trading, fair value estimation, mean reversion, and options arbitrage using the Black-Scholes model.

Each round also brought unique manual challenges, from probability modeling to game theory, making the experience both technical and creative.

The competition was a rollercoaster of highs and lows, and it definitely wrecked our sleep schedule!

**Round Results:**

| Round    | Placement | Notes                                      |
|----------|----------|---------------------------------------------|
| Round 1  | #1512    | A rough start, all about discovery          |
| Round 2  | #744     | A slight recovery                           |
| Round 3  | #214     | A comeback!                                 |
| Round 4  | #98      | Pushing towards the top 50?                 |
| Round 5  | #90      | Our algorithm didn’t quite hit the mark     |

In the end, we placed **8th in France** and **90th globally**, out of over 12,000 teams from more than 90 countries. While there were moments we wished we had done even better, we’re proud to be among the top 1% worldwide.

Overall, it was an incredibly rewarding and fun learning experience. We feel very fortunate to have worked together as such an amazing team: **Alexis Lemaire**, **Mathis Fourreau**, and **Pedro Azevedo**.

Many thanks to IMC Trading for organizing one of the most engaging trading competitions.

Let’s do this again next year!

## Overview

This repository contains algorithmic trading strategies developed for the IMC Prosperity 3 competition, a 15-day global trading challenge where participants combine algorithmic and manual trading skills to maximize profit in a virtual market.

## Competition Details

**IMC Prosperity 3** is a 15-day trading simulation divided into 5 rounds of 72 hours each. Participants develop Python programs to implement trading strategies with the goal of securing maximum profit in the form of "SeaShells".

The competition involves:

- Developing automated trading strategies in Python
- Participating in manual trading exercises for bonus profits
- Competing against teams from around the world

## Repository Structure

```text
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
   prosperity3bt strategies/round4.py 1
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

Performance metrics are captured in log files that can be found in the `logs/` directory after algorithm execution. The `logs` directory is excluded from version control via `.gitignore` due to its large file size.

For enhanced visualization of results, use the [IMC Prosperity 3 Visualizer](https://jmerle.github.io/imc-prosperity-3-visualizer/?/visualizer).

## License

[MIT License](LICENCE)

Portions of this codebase are adapted from [jmerle/imc-prosperity-2](https://github.com/jmerle/imc-prosperity-2), used under the MIT License.

## Acknowledgments

- [IMC Trading](https://www.imc.com/) for organizing the Prosperity Trading Challenge
- [jmerle](https://github.com/jmerle) for the excellent backtester and visualization tools
- Alexis Lemaire, Mathis Fourreau, and Pedro Azevedo