# IMC Prosperity 3 Trading Algorithms

Trading algorithms developed for the IMC Prosperity 3 Global Trading Challenge - 2025.

## Overview

This repository contains algorithmic trading strategies developed for the IMC Prosperity 3 competition, a 15-day global trading challenge where participants combine algorithmic and manual trading skills to maximize profit in a virtual market.

## Competition Details

**IMC Prosperity 3** is a 15-day trading simulation divided into 5 rounds of 72 hours each. Participants develop Python programs to implement trading strategies with the goal of securing maximum profit in the form of "SeaShells".

The competition involves:
- Developing successful automated trading strategies in Python
- Participating in manual trading exercises for bonus profits
- Competing against teams from around the world

## Repository Structure

```
IMC-Prosperity/
├── strategies/           # Trading strategy implementations
│   ├── datamodel.py      # Data model classes provided by the competition
│   ├── kelp_strategy.py  # Trading strategy for the KELP asset
│   ├── envelope_strategy.py # Trading strategy using envelope methodology
│   ├── rainforest_strategy.py # Strategy for the RAINFOREST asset
│   └── tests/            # Strategy tests
├── logs/                 # Strategy execution logs
└── templates/            # Template files provided by the competition
```

## Strategies

The repository contains several trading strategies developed and refined over the competition rounds:

### Kelp Strategy
A mean-reversion based strategy for trading the KELP asset with dynamic threshold adjustment and position pyramiding.

### Envelope Strategy
Trading strategy using envelope methodology that identifies overbought and oversold conditions.

### Rainforest Strategy
A specialized strategy for trading the RAINFOREST asset with pattern recognition approach that identifies key support and resistance zones. (mean-reversion as well I think)

## Getting Started

To run these strategies in the IMC Prosperity competition environment:

1. Clone this repository
2. Pull the latest changes to ensure you have the most up-to-date code (git pull)
3. Explore the strategy implementations in the `strategies/` directory
   - Each strategy is optimized for specific assets
   - The most effective strategies can be combined for maximum profit (in a single file - main_strategy.py)
4. Submit your chosen strategy through the competition Dashboard

## Results and Performance

The strategies in this repository were developed and refined over the competition rounds, with performance metrics captured in the log files.
These logs are available in your dashboard after algorithm submission, along with CSV result files.

For enhanced visualization of results, I use the tool created by [jmerle](https://github.com/jmerle):
- [IMC Prosperity 3 Visualizer](https://jmerle.github.io/imc-prosperity-3-visualizer/?/visualizer)

## License

[Specify your license here]

## Acknowledgments

- [IMC Trading](https://www.imc.com/) for organizing the Prosperity Trading Challenge
- [jmerle](https://github.com/jmerle) for the excellent visualization tool