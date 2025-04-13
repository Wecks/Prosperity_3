#!/bin/bash

# Get the directory of this script
DIR=$(dirname "$0")

# Create results directory
RESULTS_DIR="$DIR/../optimization_results"
mkdir -p "$RESULTS_DIR"

# Timestamp for this optimization run
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_FILE="$RESULTS_DIR/optimization_${TIMESTAMP}.csv"

# Write header
echo "pb1_long,pb1_short,pb2_long,pb2_short,croissants_long,croissants_short,jams_long,jams_short,djembes_long,djembes_short,pnl" > "$RESULTS_FILE"

# Function to run backtest with given parameters
run_backtest() {
    local pb1_long=$1
    local pb1_short=$2
    local pb2_long=$3
    local pb2_short=$4
    local cr_long=$5
    local cr_short=$6
    local jm_long=$7
    local jm_short=$8
    local dj_long=$9
    local dj_short=${10}

    # Create a temporary modified strategy file in the strategies directory
    local tmp_name="round2_optimized_${RANDOM}"
    local tmp_file="$DIR/../strategies/${tmp_name}.py"
    cp "$DIR/../strategies/round2.py" "$tmp_file"

    # Update the thresholds in the temporary file
    sed -i "s/\"PICNIC_BASKET1\": {\"long\": [0-9]*, \"short\": [0-9]*}/\"PICNIC_BASKET1\": {\"long\": $pb1_long, \"short\": $pb1_short}/g" "$tmp_file"
    sed -i "s/\"PICNIC_BASKET2\": {\"long\": [0-9]*, \"short\": [0-9]*}/\"PICNIC_BASKET2\": {\"long\": $pb2_long, \"short\": $pb2_short}/g" "$tmp_file"
    sed -i "s/\"CROISSANTS\": {\"long\": [0-9]*, \"short\": [0-9]*}/\"CROISSANTS\": {\"long\": $cr_long, \"short\": $cr_short}/g" "$tmp_file"
    sed -i "s/\"JAMS\": {\"long\": [0-9]*, \"short\": [0-9]*}/\"JAMS\": {\"long\": $jm_long, \"short\": $jm_short}/g" "$tmp_file"
    sed -i "s/\"DJEMBES\": {\"long\": [0-9]*, \"short\": [0-9]*}/\"DJEMBES\": {\"long\": $dj_long, \"short\": $dj_short}/g" "$tmp_file"

    # Run the backtest with the modified file and wait for it to complete
    echo "Running test with PB1($pb1_long,$pb1_short) PB2($pb2_long,$pb2_short)..."
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
    echo "$pb1_long,$pb1_short,$pb2_long,$pb2_short,$cr_long,$cr_short,$jm_long,$jm_short,$dj_long,$dj_short,$pnl" >> "$RESULTS_FILE"

    echo "Tested: PB1($pb1_long,$pb1_short) PB2($pb2_long,$pb2_short) CR($cr_long,$cr_short) JM($jm_long,$jm_short) DJ($dj_long,$dj_short) → PnL: $pnl"
}

# Loop over ranges of parameters to find optimal values
# Here we're using the default values as the base and exploring around them

# PICNIC_BASKET1 optimization
for pb1_long in $(seq 30 10 70); do
    for pb1_short in $(seq 90 10 130); do
        # Use defaults for other parameters
        run_backtest $pb1_long $pb1_short 50 100 230 355 195 485 325 370
    done
done

# PICNIC_BASKET2 optimization (using best PB1 values, which we'd need to fill in)
# Replace PB1_BEST_LONG and PB1_BEST_SHORT with actual values from first round
PB1_BEST_LONG=50  # Replace with best value found
PB1_BEST_SHORT=100  # Replace with best value found

for pb2_long in $(seq 30 10 70); do
    for pb2_short in $(seq 90 10 130); do
        # Use defaults for other parameters
        run_backtest $PB1_BEST_LONG $PB1_BEST_SHORT $pb2_long $pb2_short 230 355 195 485 325 370
    done
done

# Finding best combinations for all parameters would require more testing
# You can extend the loops to test other parameters as well

echo "Optimization completed. Results saved to $RESULTS_FILE"

# Find and display the best result
echo "Best result:"
sort -t ',' -k11 -nr "$RESULTS_FILE" | head -2

# Create a plot if gnuplot is available
if command -v gnuplot &> /dev/null; then
    PLOT_FILE="$RESULTS_DIR/optimization_plot_${TIMESTAMP}.png"
    gnuplot <<- EOF
    set terminal png size 800,600
    set output "$PLOT_FILE"
    set title "Optimization Results"
    set datafile separator ","
    set xlabel "PICNIC_BASKET1 long threshold"
    set ylabel "PnL"
    plot "$RESULTS_FILE" using 1:11 with points pt 7 title "PnL"
EOF
    echo "Plot saved to $PLOT_FILE"
fi
