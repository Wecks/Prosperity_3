#!/bin/bash

# Check if both strategy name and round are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <strategy_name> <round> [day] [options]"
    echo "Example: $0 kelp_strategy 1"
    echo "Example with specific day: $0 kelp_strategy 1-0"
    echo "Example with options: $0 kelp_strategy 1 \"--merge-pnl --print\""
    exit 1
fi

STRATEGY=$1
ROUND=$2
OPTIONS="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/backtest_results"
STRATEGY_PATH="$PROJECT_DIR/strategies/${STRATEGY}.py"
OUTPUT_PATH="$LOG_DIR/${STRATEGY}_${ROUND}.log"

# Check if strategy file exists
if [ ! -f "$STRATEGY_PATH" ]; then
    echo "Error: Strategy file $STRATEGY_PATH does not exist"
    exit 1
fi

# Create backtest_results directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Run the backtest
echo "Running backtest for strategy $STRATEGY on round $ROUND..."
echo "Output will be saved to $OUTPUT_PATH"
prosperity3bt "$STRATEGY_PATH" "$ROUND" --out "$OUTPUT_PATH" $OPTIONS

# Check if the backtest was successful
if [ $? -eq 0 ]; then
    echo "Backtest completed successfully"
    echo "Log file saved to $OUTPUT_PATH"
else
    echo "Backtest failed"
fi