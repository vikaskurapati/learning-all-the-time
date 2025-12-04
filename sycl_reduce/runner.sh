#!/bin/bash

# Bash script to run Design of Experiments for SYCL reduction kernel

# Parse command line arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <data_type> [executable_name]"
    echo "  data_type: int, float, double, long"
    echo "  executable_name: (optional) default is ./sycl_reduce"
    exit 1
fi

DATA_TYPE=$1
EXECUTABLE=${2:-"./sycl_reduce"}

# Validate data type
if [[ ! "$DATA_TYPE" =~ ^(int|float|double|long)$ ]]; then
    echo "Error: Invalid data type '$DATA_TYPE'"
    echo "Supported types: int, float, double, long"
    exit 1
fi

# Output CSV file with data type in name
OUTPUT_CSV="results_${DATA_TYPE}.csv"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable $EXECUTABLE not found!"
    echo "Please compile the code first."
    exit 1
fi

# Create CSV header
echo "VECTOR_SIZE,WORK_GROUP_SIZE,ITEMS_PER_WORK_ITEM,DATA_TYPE,AVG_TIME_MS" > "$OUTPUT_CSV"

# Arrays for parameters
VECTOR_SIZES=(1 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000)
WORK_GROUP_SIZES=(2 4 8 16 32 64 128 256 512 1024)
ITEMS_PER_WORK_ITEM=(1 2 4 8 16)

# Total number of experiments
TOTAL_EXPERIMENTS=$((${#VECTOR_SIZES[@]} * ${#WORK_GROUP_SIZES[@]} * ${#ITEMS_PER_WORK_ITEM[@]}))
CURRENT_EXPERIMENT=0

echo "Starting Design of Experiments..."
echo "Data Type: $DATA_TYPE"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Output file: $OUTPUT_CSV"
echo ""

# Loop through all combinations
for VSIZE in "${VECTOR_SIZES[@]}"; do
    for WGSIZE in "${WORK_GROUP_SIZES[@]}"; do
        for ITEMS in "${ITEMS_PER_WORK_ITEM[@]}"; do
            CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
            
            echo "[$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] Running: VECTOR_SIZE=$VSIZE, WORK_GROUP_SIZE=$WGSIZE, ITEMS=$ITEMS, TYPE=$DATA_TYPE"
            
            # Run the executable and capture output
            OUTPUT=$($EXECUTABLE $VSIZE $WGSIZE $ITEMS $DATA_TYPE 2>&1)
            EXIT_CODE=$?
            
            # Check if execution was successful
            if [ $EXIT_CODE -ne 0 ]; then
                echo "  ERROR: Execution failed!"
                echo "  Output: $OUTPUT"
                echo "$VSIZE,$WGSIZE,$ITEMS,$DATA_TYPE,ERROR" >> "$OUTPUT_CSV"
                continue
            fi
            
            # The output should now be just the time value
            TIME=$(echo "$OUTPUT" | tail -n 1 | tr -d '[:space:]')
            
            if [ -z "$TIME" ]; then
                echo "  WARNING: Could not extract timing from output"
                echo "  Output: $OUTPUT"
                echo "$VSIZE,$WGSIZE,$ITEMS,$DATA_TYPE,PARSE_ERROR" >> "$OUTPUT_CSV"
            else
                echo "  Time: ${TIME}ms"
                echo "$VSIZE,$WGSIZE,$ITEMS,$DATA_TYPE,$TIME" >> "$OUTPUT_CSV"
            fi
        done
    done
done

echo ""
echo "Experiments complete!"
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary statistics:"
wc -l "$OUTPUT_CSV"
echo ""
echo "First few results:"
head -n 6 "$OUTPUT_CSV"
echo ""
echo "To run experiments for other data types, use:"
echo "  $0 int"
echo "  $0 float"
echo "  $0 double"
echo "  $0 long"