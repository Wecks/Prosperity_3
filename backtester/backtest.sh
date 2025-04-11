#!/bin/bash

# Check if at least strategy name is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <strategy_name> <round_numbers...> [options]"
    echo "Examples:"
    echo "  - Specific round (all days):  $0 kelp_strategy 1"
    echo "  - Multiple rounds (all days): $0 kelp_strategy 1 2"
    echo "  - With options:              $0 kelp_strategy 1 \"--merge-pnl --print\""
    echo ""
    echo "For specific days, use:"
    echo "  - Specific day:              $0 kelp_strategy 1-0"
    echo "  - Multiple specific days:    $0 kelp_strategy 1-0 1--1 1--2"
    exit 1
fi

STRATEGY=$1
ROUND_SPECS=()
OPTIONS=""

# Parse arguments to separate round specifications from options
for arg in "${@:2}"; do
    if [[ "$arg" == --* ]]; then
        # If it starts with --, it's an option
        OPTIONS="$OPTIONS $arg"
    else
        # If it's just a round number without day specification,
        # we'll expand it to include all available days
        if [[ "$arg" =~ ^[0-9]+$ ]]; then
            # This is just a round number, let prosperity3bt handle it
            # It will automatically test all days in the round
            ROUND_SPECS+=("$arg")
        else
            # This is a specific day specification
            ROUND_SPECS+=("$arg")
        fi
    fi
done

# If no round specs were found, check if the last argument might be options string
if [ ${#ROUND_SPECS[@]} -eq 0 ] && [[ "${@: -1}" != --* ]]; then
    echo "Error: No round specifications provided"
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/backtest_results"
STRATEGY_PATH="$PROJECT_DIR/strategies/${STRATEGY}.py"

# Generate a unique output path based on the round specs
ROUND_SPEC_LABEL=$(echo "${ROUND_SPECS[@]}" | tr ' ' '_' | tr -d '-')
OUTPUT_PATH="$LOG_DIR/${STRATEGY}_multiday_${ROUND_SPEC_LABEL}.log"

# Check if strategy file exists
if [ ! -f "$STRATEGY_PATH" ]; then
    echo "Error: Strategy file $STRATEGY_PATH does not exist"
    exit 1
fi

# Create backtest_results directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Build the command with all round specs
# Convert array to space-separated string with proper quoting
SPECS_STR=""
for spec in "${ROUND_SPECS[@]}"; do
    SPECS_STR="$SPECS_STR \"$spec\""
done
COMMAND="prosperity3bt \"$STRATEGY_PATH\" $SPECS_STR --out \"$OUTPUT_PATH\" $OPTIONS"

# Run the backtest
echo "Running backtest for strategy $STRATEGY on specified rounds/days: ${ROUND_SPECS[@]}"
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