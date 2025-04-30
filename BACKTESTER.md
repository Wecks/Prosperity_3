# IMC Prosperity 3 Backtester Guide

This guide explains how to use the IMC Prosperity 3 backtester to test and optimize your trading strategies.

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
  - [Available Data](#available-data)
  - [Visualizing Results](#visualizing-results)
  - [Optimization](#optimization)
    - [Available Optimization Scripts](#available-optimization-scripts)
    - [Usage](#usage)
  - [Additional Resources](#additional-resources)

## Setup

1. Install the backtester package:
   ```bash
   pip install -U prosperity3bt
   ```

2. Ensure your strategy files are in the `strategies/` folder

3. Make the helper scripts executable:
   ```bash
   chmod +x backtester/backtest.sh
   chmod +x backtester/backtest_all.sh
   ```

## Helper Scripts

> **Deprecated:**
> The scripts `backtest.sh` and `backtest_all.sh` are now **deprecated**. Please use the `prosperity3bt` command and library for all backtesting. See below for details and usage. ([prosperity3bt GitHub](https://github.com/jmerle/imc-prosperity-3-backtester))

We provide convenient scripts that simplify the backtesting process, but these are no longer maintained and may not support all new features.

### backtest.sh - For Specific Rounds

> **Deprecated:** Use `prosperity3bt` directly instead.

Test a strategy on specific rounds or days with a clean interface.

**Usage:**
```bash
./backtester/backtest.sh <strategy_name> <round_spec...> [options]
```

**Examples:**

```bash
# Test on all days in round 1
./backtester/backtest.sh round4 1

# Test on multiple rounds
./backtester/backtest.sh round4 1 2

# Test on specific days
./backtester/backtest.sh round4 1-0 1--1 1--2

# Mix different specifications
./backtester/backtest.sh round4 1-0 2 3--1

# Add backtester options
./backtester/backtest.sh round4 1 "--merge-pnl --print"
```

Results are saved to `logs/backtest_results/` with filenames reflecting the tested rounds.

### backtest_all.sh - For All Rounds

> **Deprecated:** Use `prosperity3bt` directly instead.

Run your strategy on all available rounds in one command.

**Usage:**
```bash
./backtester/backtest_all.sh <strategy_name> [options]
```

**Examples:**

```bash
# Test on all rounds
./backtester/backtest_all.sh round4

# With options
./backtester/backtest_all.sh round4 "--merge-pnl --print"
```

Results are saved to `logs/backtest_results/<strategy>_all_rounds.log`.

## Manual Backtesting

**Recommended:** Use the `prosperity3bt` command and library for all backtesting. This is the official and most up-to-date method. See the [prosperity3bt GitHub repository](https://github.com/jmerle/imc-prosperity-3-backtester) for documentation and updates.

For more control, use the `prosperity3bt` command directly:

### Basic Usage

```bash
# Backtest on all data from round 0
prosperity3bt strategies/round4.py 0
```

### More Examples

```bash
# Backtest on all days from round 1
prosperity3bt strategies/round4.py 1

# Backtest on round 1 day 0
prosperity3bt strategies/round4.py 1-0

# Backtest on multiple specific days
prosperity3bt strategies/round4.py 1--1 1-0

# Backtest on multiple rounds
prosperity3bt strategies/round4.py 1 2

# Merge profit and loss across days
prosperity3bt strategies/round4.py 1 --merge-pnl

# Use custom output file
prosperity3bt strategies/round4.py 1 --out custom-output.log

# Skip saving output log
prosperity3bt strategies/round4.py 1 --no-out

# Print trader's output to stdout
prosperity3bt strategies/round4.py 1 --print
```

## Advanced Options

The backtester supports several advanced options:

| Option | Description |
|--------|-------------|
| `--merge-pnl` | Merge profit and loss across days |
| `--vis` | Automatically open results in the visualizer |
| `--print` | Print trader's output while running |
| `--out FILE` | Write output to a custom file |
| `--no-out` | Skip saving the output log |
| `--match-trades MODE` | Configure order matching (all/worse/none) |
| `--data PATH` | Use custom backtest data |

## Available Data

The backtester includes data for these rounds:

- **Round 0**: RAINFOREST_RESIN and KELP data from tutorial submission runs
- **Round 1**: RAINFOREST_RESIN, KELP, and SQUID_INK
- **Round 2**: Adds CROISSANTS, JAMS, DJEMBES, PICNIC_BASKET1, and PICNIC_BASKET2
- **Round 3**: Adds VOLCANIC_ROCK and all VOLCANIC_ROCK_VOUCHER variants (9500, 9750, 10000, 10250, and 10500)
- **Round 4**: Adds MAGNIFICENT_MACARONS
- **Round 6**: Submission run data (day X = data from round X)
- **Round 7**: End-of-round data (day X = data from round X)
- **Round 8**: Earlier version of round 2 data

## Visualizing Results

Use the [IMC Prosperity 3 Visualizer](https://jmerle.github.io/imc-prosperity-3-visualizer) to visualize results:

```bash
# Automatically open results in the visualizer
prosperity3bt strategies/round4.py 1 --vis
```

## Optimization

The repository includes optimization scripts to find optimal parameters for your strategies.

> **Note:** Some optimizer scripts require your strategy file to have a specific name (e.g., `DJEMBES.py`, `CROISSANTS.py`, etc.). You may need to rename or adjust your strategy files before running these scripts for them to work properly.

### Available Optimization Scripts

- **optimize_croissants.sh**: Optimizes CROISSANTS thresholds
- **optimize_djembes.sh**: Optimizes DJEMBES thresholds
- **optimize_jams.sh**: Optimizes JAMS thresholds
- **optimize_picnic_basket1.sh**: Optimizes PICNIC_BASKET1 thresholds
- **optimize_picnic_basket2.sh**: Optimizes PICNIC_BASKET2 thresholds
- **optimize_<assetname>_spread_mult**: Used for counterparty optimization. See [strategies/optimization_conterparty_test](./strategies/optimization_conterparty_test/)

### Usage

```bash
# Run optimization for a specific product
./backtester/optimizers/optimize_jams.sh

# Check results in optimization_results directory
```

> **Tip:** If you encounter errors, ensure your strategy file matches the expected name for the optimizer script you are using.

## Additional Resources

- [IMC Prosperity 3 Backtester GitHub Repository](https://github.com/jmerle/imc-prosperity-3-backtester)
- [IMC Prosperity 3 Visualizer](https://github.com/jmerle/imc-prosperity-3-visualizer)
- [IMC Prosperity 3 Visualizer Web App](https://jmerle.github.io/imc-prosperity-3-visualizer)