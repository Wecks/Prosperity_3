#!/bin/bash

# Check if both strategy names and round are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <strategy1_name> <strategy2_name> <round> [options]"
    echo "Example: $0 kelp_strategy envelope_strategy 1"
    echo "Example with options: $0 kelp_strategy envelope_strategy 1 \"--merge-pnl\""
    exit 1
fi

STRATEGY1=$1
STRATEGY2=$2
ROUND=$3
OPTIONS="${4:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKTEST_SCRIPT="$SCRIPT_DIR/backtest.sh"

# Check if backtest script exists and is executable
if [ ! -x "$BACKTEST_SCRIPT" ]; then
    echo "Making backtest.sh executable..."
    chmod +x "$BACKTEST_SCRIPT"
fi

# Run backtests for both strategies
echo "Running backtests for comparison..."
"$BACKTEST_SCRIPT" "$STRATEGY1" "$ROUND" "$OPTIONS"
STATUS1=$?
"$BACKTEST_SCRIPT" "$STRATEGY2" "$ROUND" "$OPTIONS"
STATUS2=$?

# Check if both backtests were successful
if [ $STATUS1 -eq 0 ] && [ $STATUS2 -eq 0 ]; then
    echo "-----------------------------------------------------"
    echo "Both backtests completed successfully."
    echo "You can now compare the logs for $STRATEGY1 and $STRATEGY2 on round $ROUND."
    echo "Consider using the IMC Prosperity 3 Visualizer to compare the results visually."
    echo "To visualize a specific result:"
    echo "prosperity3bt strategies/${STRATEGY1}.py $ROUND --visualizer"
    echo "prosperity3bt strategies/${STRATEGY2}.py $ROUND --visualizer"
else
    echo "One or both backtests failed."
fi