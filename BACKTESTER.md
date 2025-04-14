# IMC Prosperity 3 Backtester Guide

This document provides comprehensive instructions on how to backtest your trading strategies using the IMC Prosperity 3 backtester and our custom helper scripts.

## Table of Contents

- [IMC Prosperity 3 Backtester Guide](#imc-prosperity-3-backtester-guide)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
  - [Helper Scripts](#helper-scripts)
    - [backtest.sh - For Specific Rounds](#backtestsh---for-specific-rounds)
    - [backtest\_all.sh - For All Rounds](#backtest_allsh---for-all-rounds)
  - [Manual Backtesting](#manual-backtesting)
    - [Basic Usage](#basic-usage)
    - [More Examples](#more-examples)
  - [Advanced Options](#advanced-options)
  - [Environment Variables](#environment-variables)
  - [Available Data](#available-data)
  - [Visualizing Results](#visualizing-results)
  - [Optimization](#optimization)
    - [Available Optimization Scripts](#available-optimization-scripts)
    - [Usage](#usage)
    - [Important Note](#important-note)
  - [Additional Resources](#additional-resources)

## Setup

1. Install the backtester using pip:
   ```bash
   pip install -U prosperity3bt
   ```

2. Make sure your strategy files are in the `strategies/` folder.

3. Ensure the helper scripts are executable:
   ```bash
   chmod +x backtester/backtest.sh
   chmod +x backtester/backtest_all.sh
   ```

## Helper Scripts

We provide two convenient scripts to make backtesting easier and more efficient.

### backtest.sh - For Specific Rounds

This script allows you to quickly test a strategy on specific rounds or days with a clean, simple interface.

**Usage:**
```bash
./backtester/backtest.sh <strategy_name> <round_spec...> [options]
```

**Examples:**

```bash
# Test a strategy on all days in round 1
./backtester/backtest.sh kelp_strategy 1

# Test on multiple rounds (all days in each)
./backtester/backtest.sh kelp_strategy 1 2

# Test on specific days if needed
./backtester/backtest.sh kelp_strategy 1-0 1--1 1--2

# Mix different specifications
./backtester/backtest.sh kelp_strategy 1-0 2 3--1

# Add backtester options
./backtester/backtest.sh kelp_strategy 1 "--merge-pnl --print"
```

All results will be saved to the `logs/backtest_results/` folder with a filename that reflects the tested rounds.

### backtest_all.sh - For All Rounds

This script runs your strategy on all available rounds (0, 1, 6, and 7) in a single command.

**Usage:**
```bash
./backtester/backtest_all.sh <strategy_name> [options]
```

**Examples:**

```bash
# Test on all available rounds
./backtester/backtest_all.sh kelp_strategy

# Test on all rounds with options
./backtester/backtest_all.sh kelp_strategy "--merge-pnl --print"
```

Results will be saved to `logs/backtest_results/<strategy>_all_rounds.log`.

## Manual Backtesting

If you need more control, you can use the prosperity3bt command directly:

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
prosperity3bt strategies/your_strategy.py 1 --print
```

## Advanced Options

The `prosperity3bt` command supports several advanced options:

| Option | Description |
|--------|-------------|
| `--merge-pnl` | Merge profit and loss across days |
| `--vis` | Automatically open results in the visualizer |
| `--print` | Print trader's output to stdout while running |
| `--out FILE` | Write algorithm output to a custom file |
| `--no-out` | Skip saving the output log to a file |
| `--match-trades MODE` | Configure how orders are matched against market trades (all/worse/none) |
| `--data PATH` | Specify path to custom backtest data |

## Environment Variables

During backtests, two environment variables are set:
- `PROSPERITY3BT_ROUND`: contains the round number
- `PROSPERITY3BT_DAY`: contains the day number

Note that these environment variables do not exist in the official submission environment, so make sure your submitted code doesn't require them.

## Available Data

The backtester includes data for the following rounds:

- **Round 0**: Prices and anonymized trades data on RAINFOREST_RESIN and KELP that was used during tutorial submission runs.
- **Round 1**: Prices and anonymized trades data on RAINFOREST_RESIN, KELP, and SQUID_INK.
- **Round 2**: Prices and anonymized trades data on RAINFOREST_RESIN, KELP, SQUID_INK, CROISSANTS, JAMS, DJEMBES, PICNIC_BASKET1, and PICNIC_BASKET2.
- **Round 3**: Prices and anonymized trades data on RAINFOREST_RESIN, KELP, SQUID_INK, CROISSANTS, JAMS, DJEMBES, PICNIC_BASKET1, PICNIC_BASKET2, VOLCANIC_ROCK, VOLCANIC_ROCK_VOUCHER_9500, VOLCANIC_ROCK_VOUCHER_9750, VOLCANIC_ROCK_VOUCHER_10000, VOLCANIC_ROCK_VOUCHER_10250, and VOLCANIC_ROCK_VOUCHER_10500.
- **Round 6**: Prices and anonymized trades data from submission runs. Round 6 day X represents the submission data of round X, where X = 0 means the tutorial round and X = 6 means the submission data of round 2 before it was updated.
- **Round 7**: Prices data used during end-of-round runs. Round 7 day X represents the end-of-round data of round X. For round 1, its old end-of-round data is in round 7 day 0 and its new data is in round 7 day 1.
- **Round 8**: The version of round 2 data before it was updated (earlier version of RAINFOREST_RESIN data).

## Visualizing Results

The backtester can be used alongside the [IMC Prosperity 3 Visualizer](https://github.com/jmerle/imc-prosperity-3-visualizer) to visualize your trading results.

To automatically open results in the visualizer:
```bash
prosperity3bt strategies/your_strategy.py 1 --vis
```

## Optimization

Our workspace includes several optimization scripts that can help tune parameters for better trading performance. These scripts are located in the `backtester/` folder and are designed to test multiple parameter combinations to find optimal settings.

### Available Optimization Scripts

- **optimize_croissants.sh**: Optimizes long/short thresholds for CROISSANTS
- **optimize_djembes.sh**: Optimizes long/short thresholds for DJEMBES
- **optimize_jams.sh**: Optimizes long/short thresholds for JAMS
- **optimize_picnic_basket1.sh**: Optimizes long/short thresholds for PICNIC_BASKET1
- **optimize_picnic_basket2.sh**: Optimizes long/short thresholds for PICNIC_BASKET2
- **optimize_thresholds.sh**: General threshold optimization script

### Usage

```bash
# Run optimization for a specific product
./backtester/optimize_croissants.sh

# Check results in the optimization_results directory
```

### Important Note

**These optimization scripts require modifying your strategy code to work properly.** The current implementation assumes that your strategy code:

1. Contains configurable thresholds that can be modified by the scripts
2. Uses a specific format for these thresholds (e.g., `"CROISSANTS": {"long": 50, "short": -50}`)

Before using these scripts, you may need to adapt your `PicnicBasketStrategy` or other strategies to match the expected format. The scripts create temporary modified copies of your strategy files for testing and extract PnL results to find optimal values.

All optimization results are saved to CSV files in the `optimization_results/` directory, with timestamps for easy tracking.

## Additional Resources

- [IMC Prosperity 3 Backtester GitHub Repository](https://github.com/jmerle/imc-prosperity-3-backtester)
- [IMC Prosperity 3 Visualizer](https://github.com/jmerle/imc-prosperity-3-visualizer)