#!/bin/bash

# Bash script to run benchmarks for colleague's reduction kernel

# Parse command line arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <data_type> [executable_name]"
    echo "  data_type: int, float, double, long"
    echo "  executable_name: (optional) default is ./colleague_reduce"
    exit 1
fi

DATA_TYPE=$1
EXECUTABLE=${2:-"./colleague_reduce"}

# Validate data type
if [[ ! "$DATA_TYPE" =~ ^(int|float|double|long)$ ]]; then
    echo "Error: Invalid data type '$DATA_TYPE'"
    echo "Supported types: int, float, double, long"
    exit 1
fi

# Output CSV file with data type in name
OUTPUT_CSV="colleague_results_${DATA_TYPE}.csv"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable $EXECUTABLE not found!"
    echo "Please compile the code first with:"
    echo "  icpx -fsycl colleague_reduce.cpp -o colleague_reduce"
    exit 1
fi

# Create CSV header
echo "VECTOR_SIZE,DATA_TYPE,AVG_TIME_MS" > "$OUTPUT_CSV"

# Array for vector sizes (powers of 10)
VECTOR_SIZES=(1 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000)

# Total number of experiments
TOTAL_EXPERIMENTS=${#VECTOR_SIZES[@]}
CURRENT_EXPERIMENT=0

echo "Starting Benchmark for Colleague's Reduction Kernel..."
echo "Data Type: $DATA_TYPE"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Output file: $OUTPUT_CSV"
echo ""

# Loop through all vector sizes
for VSIZE in "${VECTOR_SIZES[@]}"; do
    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
    
    echo "[$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] Running: VECTOR_SIZE=$VSIZE, TYPE=$DATA_TYPE"
    
    # Run the executable and capture output
    OUTPUT=$($EXECUTABLE $VSIZE $DATA_TYPE 2>&1)
    EXIT_CODE=$?
    
    # Check if execution was successful
    if [ $EXIT_CODE -ne 0 ]; then
        echo "  ERROR: Execution failed!"
        echo "  Output: $OUTPUT"
        echo "$VSIZE,$DATA_TYPE,ERROR" >> "$OUTPUT_CSV"
        continue
    fi
    
    # The output should be just the time value
    TIME=$(echo "$OUTPUT" | tail -n 1 | tr -d '[:space:]')
    
    if [ -z "$TIME" ]; then
        echo "  WARNING: Could not extract timing from output"
        echo "  Output: $OUTPUT"
        echo "$VSIZE,$DATA_TYPE,PARSE_ERROR" >> "$OUTPUT_CSV"
    else
        echo "  Time: ${TIME}ms"
        echo "$VSIZE,$DATA_TYPE,$TIME" >> "$OUTPUT_CSV"
    fi
done

echo ""
echo "Experiments complete!"
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary statistics:"
wc -l "$OUTPUT_CSV"
echo ""
echo "All results:"
cat "$OUTPUT_CSV"
echo ""
echo "To run experiments for other data types, use:"
echo "  $0 int"
echo "  $0 float"
echo "  $0 double"
echo "  $0 long"