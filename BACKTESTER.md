# IMC Prosperity 3 Backtester

This document provides instructions on how to backtest your trading strategies using the IMC Prosperity 3 backtester.

## Setup

1. Install the backtester using pip:
   ```bash
   pip install -U prosperity3bt
   ```

2. Make sure your strategy files are ready in the `strategies/` folder.

## Running Backtests

### Basic Usage

Run a backtest on an algorithm using all data from round 0:
```bash
prosperity3bt strategies/your_strategy.py 0
```

### More Examples

```bash
# Backtest on all days from round 1
prosperity3bt strategies/your_strategy.py 1

# Backtest on round 1 day 0
prosperity3bt strategies/your_strategy.py 1-0

# Backtest on round 1 day -1 and round 1 day 0
prosperity3bt strategies/your_strategy.py 1--1 1-0

# Backtest on all days from rounds 1 and 2
prosperity3bt strategies/your_strategy.py 1 2

# Merge profit and loss across days
prosperity3bt strategies/your_strategy.py 1 --merge-pnl

# Write algorithm output to custom file
prosperity3bt strategies/your_strategy.py 1 --out custom-output.log

# Skip saving the output log to a file
prosperity3bt strategies/your_strategy.py 1 --no-out

# Print trader's output to stdout while running
# This may be helpful when debugging a broken trader
prosperity3bt strategies/your_strategy.py 1 --print
```

## Environment Variables

During backtests, two environment variables are set:
- `PROSPERITY3BT_ROUND`: contains the round number
- `PROSPERITY3BT_DAY`: contains the day number

Note that these environment variables do not exist in the official submission environment, so make sure your submitted code doesn't require them.

## Using the Helper Scripts

We've included several (only one for now) helper scripts in the `backtester/` folder to make backtesting easier:

### backtest.sh

This script allows you to quickly backtest a strategy on a specific round:

```bash
./backtester/backtest.sh kelp_strategy 1
```

This will run the backtest on the `kelp_strategy.py` file for round 1 and save the output to the `logs/backtest_results` folder.

## Visualizing Results

The backtester can be used alongside the [IMC Prosperity 3 Visualizer](https://github.com/jmerle/imc-prosperity-3-visualizer) to visualize your trading results.

To automatically open results in the visualizer:
```bash
prosperity3bt strategies/your_strategy.py 1 --visualizer
```

## Available Data

The backtester includes data for the following rounds:

- **Round 0**: Prices and anonymized trades data on RAINFOREST_RESIN and KELP that was used during tutorial submission runs.
- **Round 1**: Prices and anonymized trades data on RAINFOREST_RESIN and KELP.
- **Round 6**: Prices and anonymized trades data that was used during submission runs. Round 6 day X represents the submission data of round X.

## Additional Resources

- [IMC Prosperity 3 Backtester GitHub Repository](https://github.com/jmerle/imc-prosperity-3-backtester)
- [IMC Prosperity 3 Visualizer](https://github.com/jmerle/imc-prosperity-3-visualizer)