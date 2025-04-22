#!/bin/bash

# Optimization script for spread multiplier thresholds and multipliers in VOLCANIC_ROCK_VOUCHER_9500.py
DIR=$(dirname "$0")
RESULTS_DIR="$DIR/../optimization_results_spread_mult"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_FILE="$RESULTS_DIR/optimization_volcanic_rock_voucher_9500_spread_mult_${TIMESTAMP}.csv"

echo "upper,lower,mult_high,mult_low,pnl" > "$RESULTS_FILE"

run_backtest() {
    local upper=$1
    local lower=$2
    local mult_high=$3
    local mult_low=$4
    local tmp_name="VOLCANIC_ROCK_VOUCHER_9500_spreadmult_${RANDOM}"
    local base_dir="$(cd "$DIR/../.." && pwd)"
    local strategies_dir="$base_dir/strategies"
    local tmp_file="$strategies_dir/${tmp_name}.py"

    if [ ! -f "$strategies_dir/VOLCANIC_ROCK_VOUCHER_9500.py" ]; then
        echo "Error: Source strategy file not found at $strategies_dir/VOLCANIC_ROCK_VOUCHER_9500.py"
        return 1
    fi

    cp "$strategies_dir/VOLCANIC_ROCK_VOUCHER_9500.py" "$tmp_file"

    sed -i "s/SPREAD_MULT_UPPER = [0-9.]*$/SPREAD_MULT_UPPER = $upper/" "$tmp_file"
    sed -i "s/SPREAD_MULT_LOWER = [0-9.]*$/SPREAD_MULT_LOWER = $lower/" "$tmp_file"
    sed -i "s/SPREAD_MULT_HIGH = [0-9.]*$/SPREAD_MULT_HIGH = $mult_high/" "$tmp_file"
    sed -i "s/SPREAD_MULT_LOW = [0-9.]*$/SPREAD_MULT_LOW = $mult_low/" "$tmp_file"

    echo "Running test with upper=$upper, lower=$lower, mult_high=$mult_high, mult_low=$mult_low..."
    cd "$base_dir"
    local output_file="$strategies_dir/${tmp_name}.output"
    prosperity3bt "strategies/${tmp_name}.py" "--no-out" 5 > "$output_file" 2>&1
    local exit_code=$?
    local output=$(cat "$output_file")

    rm "$tmp_file"
    rm "$output_file"

    if [ $exit_code -ne 0 ]; then
        echo "Backtest failed with exit code $exit_code"
        echo "$output"
        pnl="ERROR"
        return 1
    fi

    local profit_line=$(echo "$output" | grep "Total profit:" | tail -1)
    local pnl=$(echo "$profit_line" | grep -o '[0-9,\-]*' | tr -d ',')
    if [ -z "$pnl" ]; then
        pnl=$(echo "$output" | grep -o 'PnL: [0-9,\-]*' | grep -o '[0-9,\-]*' | tr -d ',' | tail -1)
    fi
    if [ -z "$pnl" ]; then
        local summary_line=$(echo "$output" | grep -A1 "Profit summary:" | grep "Total profit:")
        pnl=$(echo "$summary_line" | grep -o '[0-9,\-]*' | tr -d ',')
    fi
    if [ -z "$pnl" ]; then
        pnl="0"
    fi

    echo "$upper,$lower,$mult_high,$mult_low,$pnl" >> "$RESULTS_FILE"
    echo "Tested: upper=$upper, lower=$lower, mult_high=$mult_high, mult_low=$mult_low → PnL: $pnl"
}

for upper in 0.5 0.6 0.7 0.8 0.9 1; do
    for lower in 0 0.1 0.2 0.3 0.4 0.5; do
        for mult_high in 0.9 1.1 1.3 1.5 1.9 2.5; do
            for mult_low in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
                run_backtest $upper $lower $mult_high $mult_low
            done
        done
    done
done

echo "Optimization completed. Results saved to $RESULTS_FILE"
echo "Best result:"
sort -t ',' -k5 -nr "$RESULTS_FILE" | head -2
