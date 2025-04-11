#!/bin/bash

# Check if strategy name is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <strategy_name> [options]"
    echo "This script runs backtests on all available rounds (0, 1, 6, 7)"
    echo "Examples:"
    echo "  - All rounds:                $0 kelp_strategy"
    echo "  - With options:              $0 kelp_strategy \"--merge-pnl --print\""
    exit 1
fi

STRATEGY=$1
OPTIONS="${@:2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/backtest_results"
STRATEGY_PATH="$PROJECT_DIR/strategies/${STRATEGY}.py"

# Available rounds
ROUNDS=(0 1 6 7)

# Check if strategy file exists
if [ ! -f "$STRATEGY_PATH" ]; then
    echo "Error: Strategy file $STRATEGY_PATH does not exist"
    exit 1
fi

# Create backtest_results directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Generate a unique output path
OUTPUT_PATH="$LOG_DIR/${STRATEGY}_all_rounds.log"

# Build the command with all rounds
SPECS_STR=""
for round in "${ROUNDS[@]}"; do
    SPECS_STR="$SPECS_STR \"$round\""
done

# Build the command
COMMAND="prosperity3bt \"$STRATEGY_PATH\" $SPECS_STR --out \"$OUTPUT_PATH\" $OPTIONS"

# Run the backtest
echo "Running backtest for strategy $STRATEGY on all available rounds: ${ROUNDS[@]}"
echo "Output will be saved to $OUTPUT_PATH"
echo "Executing: $COMMAND"
eval $COMMAND

# Check if the backtest was successful
if [ $? -eq 0 ]; then
    echo "Backtest completed successfully"
    echo "Log file saved to $OUTPUT_PATH"
else
    echo "Backtest failed"
fi
