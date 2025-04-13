#!/bin/bash

# Optimization script specifically for DJEMBES thresholds
# Get the directory of this script
DIR=$(dirname "$0")

# Create results directory
RESULTS_DIR="$DIR/../optimization_results"
mkdir -p "$RESULTS_DIR"

# Timestamp for this optimization run
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_FILE="$RESULTS_DIR/optimization_djembes_${TIMESTAMP}.csv"

# Write header
echo "djembes_long,djembes_short,pnl" > "$RESULTS_FILE"

# Function to run backtest with given parameters
run_backtest() {
    local dj_long=$1
    local dj_short=$2

    # Create a temporary modified strategy file in the strategies directory
    local tmp_name="round2_optimized_${RANDOM}"
    local tmp_file="$DIR/../strategies/${tmp_name}.py"
    cp "$DIR/../strategies/round2.py" "$tmp_file"

    # Update the thresholds in the temporary file
    sed -i "s/\"DJEMBES\": {\"long\": [0-9]*, \"short\": [0-9]*}/\"DJEMBES\": {\"long\": $dj_long, \"short\": $dj_short}/g" "$tmp_file"

    # Run the backtest with the modified file and wait for it to complete
    echo "Running test with DJEMBES($dj_long,$dj_short)..."
    cd "$DIR/.."
    local output_file="${tmp_file}.output"
    prosperity3bt "./strategies/${tmp_name}.py" 2 > "$output_file" 2>&1
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
    echo "$dj_long,$dj_short,$pnl" >> "$RESULTS_FILE"

    echo "Tested: DJEMBES($dj_long,$dj_short) → PnL: $pnl"
}

# Test a range of DJEMBES thresholds
# Starting with values around the defaults (325, 370)
for dj_long in $(seq 120 10 160); do
    for dj_short in $(seq 190 10 230); do
        run_backtest $dj_long $dj_short
    done
done

echo "Optimization completed. Results saved to $RESULTS_FILE"

# Find and display the best result
echo "Best result:"
sort -t ',' -k3 -nr "$RESULTS_FILE" | head -2

# Create a plot if gnuplot is available
if command -v gnuplot &> /dev/null; then
    PLOT_FILE="$RESULTS_DIR/djembes_plot_${TIMESTAMP}.png"
    gnuplot <<- EOF
    set terminal png size 1200,800
    set output "$PLOT_FILE"
    set title "DJEMBES Optimization Results"
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
