#!/bin/bash

# Optimization script specifically for VOLCANIC_ROCK_VOUCHER thresholds
# Get the directory of this script
DIR=$(dirname "$0")

# Create results directory
RESULTS_DIR="$DIR/../optimization_results"
mkdir -p "$RESULTS_DIR"

# Timestamp for this optimization run
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_FILE="$RESULTS_DIR/optimization_volcanic_vouchers_${TIMESTAMP}.csv"

# Write header
echo "voucher_9500,voucher_9750,voucher_10000,voucher_10250,voucher_10500,pnl" > "$RESULTS_FILE"

# Function to run backtest with given parameters
run_backtest() {
    local v9500=$1
    local v9750=$2
    local v10000=$3
    local v10250=$4
    local v10500=$5

    # Create a temporary modified strategy file in the strategies directory
    local tmp_name="optimizer_round3_optimized_${RANDOM}"
    local base_dir="$(cd "$DIR/../.." && pwd)"
    local strategies_dir="$base_dir/strategies"
    local tmp_file="$strategies_dir/${tmp_name}.py"

    # Check if the strategy file exists
    if [ ! -f "$strategies_dir/optimizer_round3.py" ]; then
        echo "Error: Source strategy file not found at $strategies_dir/optimizer_round3.py"
        return 1
    fi

    # Copy the strategy file
    cp "$strategies_dir/optimizer_round3.py" "$tmp_file"

    # Update the thresholds in the temporary file - make sure to match exact pattern
    sed -i "s/\"VOLCANIC_ROCK_VOUCHER_9500\": [0-9][0-9]*/\"VOLCANIC_ROCK_VOUCHER_9500\": $v9500/g" "$tmp_file"
    sed -i "s/\"VOLCANIC_ROCK_VOUCHER_9750\": [0-9][0-9]*/\"VOLCANIC_ROCK_VOUCHER_9750\": $v9750/g" "$tmp_file"
    sed -i "s/\"VOLCANIC_ROCK_VOUCHER_10000\": [0-9][0-9]*/\"VOLCANIC_ROCK_VOUCHER_10000\": $v10000/g" "$tmp_file"
    sed -i "s/\"VOLCANIC_ROCK_VOUCHER_10250\": [0-9][0-9]*/\"VOLCANIC_ROCK_VOUCHER_10250\": $v10250/g" "$tmp_file"
    sed -i "s/\"VOLCANIC_ROCK_VOUCHER_10500\": [0-9][0-9]*/\"VOLCANIC_ROCK_VOUCHER_10500\": $v10500/g" "$tmp_file"

    # Run the backtest with the modified file and wait for it to complete
    echo "Running test with Vouchers($v9500,$v9750,$v10000,$v10250,$v10500)..."
    cd "$base_dir"
    local output_file="$strategies_dir/${tmp_name}.output"
    prosperity3bt "strategies/${tmp_name}.py" 3 > "$output_file" 2>&1
    local exit_code=$?
    local output=$(cat "$output_file")

    # Clean up temporary files
    rm "$tmp_file"
    rm "$output_file"

    # Check if the backtest completed successfully
    if [ $exit_code -ne 0 ]; then
        echo "Backtest failed with exit code $exit_code"
        echo "$output"
        pnl="ERROR"
        return 1
    fi

    # Extract PnL using pattern matching - handle commas in the number
    local profit_line=$(echo "$output" | grep "Total profit:" | tail -1)
    local pnl=$(echo "$profit_line" | grep -o '[0-9,\-]*' | tr -d ',')

    # If PnL extraction failed, try other patterns
    if [ -z "$pnl" ]; then
        pnl=$(echo "$output" | grep -o 'PnL: [0-9,\-]*' | grep -o '[0-9,\-]*' | tr -d ',' | tail -1)
    fi

    # If still no result, check for the profit summary
    if [ -z "$pnl" ]; then
        local summary_line=$(echo "$output" | grep -A1 "Profit summary:" | grep "Total profit:")
        pnl=$(echo "$summary_line" | grep -o '[0-9,\-]*' | tr -d ',')
    fi

    # If still no value found, set to 0
    if [ -z "$pnl" ]; then
        pnl="0"
    fi

    # Append result to CSV
    echo "$v9500,$v9750,$v10000,$v10250,$v10500,$pnl" >> "$RESULTS_FILE"

    echo "Tested: Vouchers($v9500,$v9750,$v10000,$v10250,$v10500) → PnL: $pnl"
}

# Test each voucher threshold independently
# Default threshold value for all vouchers (from optimizer_round3.py)
DEFAULT=5

# Test VOLCANIC_ROCK_VOUCHER_9500 threshold
echo "Testing VOLCANIC_ROCK_VOUCHER_9500 threshold independently..."
for v9500 in $(seq 1 3 10); do
    run_backtest $v9500 $DEFAULT $DEFAULT $DEFAULT $DEFAULT
done

# Test VOLCANIC_ROCK_VOUCHER_9750 threshold
echo "Testing VOLCANIC_ROCK_VOUCHER_9750 threshold independently..."
for v9750 in $(seq 1 3 10); do
    run_backtest $DEFAULT $v9750 $DEFAULT $DEFAULT $DEFAULT
done

# Test VOLCANIC_ROCK_VOUCHER_10000 threshold
echo "Testing VOLCANIC_ROCK_VOUCHER_10000 threshold independently..."
for v10000 in $(seq 1 3 10); do
    run_backtest $DEFAULT $DEFAULT $v10000 $DEFAULT $DEFAULT
done

# Test VOLCANIC_ROCK_VOUCHER_10250 threshold
echo "Testing VOLCANIC_ROCK_VOUCHER_10250 threshold independently..."
for v10250 in $(seq 1 3 10); do
    run_backtest $DEFAULT $DEFAULT $DEFAULT $v10250 $DEFAULT
done

# Test VOLCANIC_ROCK_VOUCHER_10500 threshold
echo "Testing VOLCANIC_ROCK_VOUCHER_10500 threshold independently..."
for v10500 in $(seq 1 3 10); do
    run_backtest $DEFAULT $DEFAULT $DEFAULT $DEFAULT $v10500
done

echo "Optimization completed. Results saved to $RESULTS_FILE"

# Find and display the best result
echo "Best result:"
sort -t ',' -k6 -nr "$RESULTS_FILE" | head -2

# Create a visualization if gnuplot is available
if command -v gnuplot &> /dev/null; then
    PLOT_FILE="$RESULTS_DIR/volcanic_vouchers_plot_${TIMESTAMP}.png"
    gnuplot <<- EOF
    set terminal png size 1200,800
    set output "$PLOT_FILE"
    set title "Volcanic Rock Voucher Optimization Results"
    set datafile separator ","
    set xlabel "Threshold Value"
    set ylabel "PnL"
    plot "$RESULTS_FILE" using 1:6 with linespoints title "9500", \
         "$RESULTS_FILE" using 2:6 with linespoints title "9750", \
         "$RESULTS_FILE" using 3:6 with linespoints title "10000", \
         "$RESULTS_FILE" using 4:6 with linespoints title "10250", \
         "$RESULTS_FILE" using 5:6 with linespoints title "10500"
EOF
    echo "Plot saved to $PLOT_FILE"
fi

# Optional: Add more advanced tests here focusing on specific voucher thresholds
# Example: Test different thresholds for 9500 voucher while keeping others constant
# for v9500 in $(seq 1 1 10); do
#     run_backtest $v9500 5 5 5 5
# done
