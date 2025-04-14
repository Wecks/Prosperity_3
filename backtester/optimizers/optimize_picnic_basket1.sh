#!/bin/bash

# Optimization script specifically for PICNIC_BASKET1 thresholds
# Get the directory of this script
DIR=$(dirname "$0")

# Create results directory
RESULTS_DIR="$DIR/../optimization_results"
mkdir -p "$RESULTS_DIR"

# Timestamp for this optimization run
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_FILE="$RESULTS_DIR/optimization_picnic_basket1_${TIMESTAMP}.csv"

# Write header
echo "pb1_long,pb1_short,pnl" > "$RESULTS_FILE"

# Function to run backtest with given parameters
run_backtest() {
    local pb1_long=$1
    local pb1_short=$2

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

    # Update the thresholds in the temporary file
    sed -i "s/\"PICNIC_BASKET1\": {\"long\": [0-9]*, \"short\": [0-9]*}/\"PICNIC_BASKET1\": {\"long\": $pb1_long, \"short\": $pb1_short}/g" "$tmp_file"

    # Run the backtest with the modified file and wait for it to complete
    echo "Running test with PB1($pb1_long,$pb1_short)..."
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
    echo "$pb1_long,$pb1_short,$pnl" >> "$RESULTS_FILE"

    echo "Tested: PB1($pb1_long,$pb1_short) → PnL: $pnl"
}

# Test a range of PICNIC_BASKET1 thresholds
# More granular search around promising values
for pb1_long in $(seq 60 10 90); do
    for pb1_short in $(seq 110 10 150); do
        run_backtest $pb1_long $pb1_short
    done
done

echo "Optimization completed. Results saved to $RESULTS_FILE"

# Find and display the best result
echo "Best result:"
sort -t ',' -k3 -nr "$RESULTS_FILE" | head -2

# Create a plot if gnuplot is available
if command -v gnuplot &> /dev/null; then
    PLOT_FILE="$RESULTS_DIR/picnic_basket1_plot_${TIMESTAMP}.png"
    gnuplot <<- EOF
    set terminal png size 1200,800
    set output "$PLOT_FILE"
    set title "PICNIC_BASKET1 Optimization Results"
    set datafile separator ","
    set xlabel "Long Threshold"
    set ylabel "Short Threshold"
    set zlabel "PnL" rotate
    set dgrid3d 30,30
    set view 45,45
    splot "$RESULTS_FILE" using 1:2:3 with pm3d title "PnL"
EOF
    echo "Plot saved to $PLOT_FILE"
fi
