#!/bin/bash

# Bash script to run Design of Experiments for SYCL reduction kernel

# Output CSV file
OUTPUT_CSV="results.csv"

# Name of the compiled executable
EXECUTABLE="./sycl_reduce"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable $EXECUTABLE not found!"
    echo "Please compile the code first."
    exit 1
fi

# Create CSV header
echo "VECTOR_SIZE,WORK_GROUP_SIZE,ITEMS_PER_WORK_ITEM,AVG_TIME_MS" > "$OUTPUT_CSV"

# Arrays for parameters
VECTOR_SIZES=(1 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000)
WORK_GROUP_SIZES=(2 4 8 16 32 64 128 256 512 1024)
ITEMS_PER_WORK_ITEM=(1 2 4 8 16)

# Total number of experiments
TOTAL_EXPERIMENTS=$((${#VECTOR_SIZES[@]} * ${#WORK_GROUP_SIZES[@]} * ${#ITEMS_PER_WORK_ITEM[@]}))
CURRENT_EXPERIMENT=0

echo "Starting Design of Experiments..."
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo ""

# Loop through all combinations
for VSIZE in "${VECTOR_SIZES[@]}"; do
    for WGSIZE in "${WORK_GROUP_SIZES[@]}"; do
        for ITEMS in "${ITEMS_PER_WORK_ITEM[@]}"; do
            CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
            
            echo "[$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] Running: VECTOR_SIZE=$VSIZE, WORK_GROUP_SIZE=$WGSIZE, ITEMS_PER_WORK_ITEM=$ITEMS"
            
            # Run the executable and capture output
            OUTPUT=$($EXECUTABLE $VSIZE $WGSIZE $ITEMS 2>&1)
            
            # Check if execution was successful
            if [ $? -ne 0 ]; then
                echo "  ERROR: Execution failed!"
                echo "  Output: $OUTPUT"
                echo "$VSIZE,$WGSIZE,$ITEMS,ERROR" >> "$OUTPUT_CSV"
                continue
            fi
            
            # Extract the timing from output (looking for "GPU code took XXXms")
            TIME=$(echo "$OUTPUT" | grep "GPU code took" | awk '{print $4}' | sed 's/ms//')
            
            if [ -z "$TIME" ]; then
                echo "  WARNING: Could not extract timing from output"
                echo "  Output: $OUTPUT"
                echo "$VSIZE,$WGSIZE,$ITEMS,PARSE_ERROR" >> "$OUTPUT_CSV"
            else
                echo "  Time: ${TIME}ms"
                echo "$VSIZE,$WGSIZE,$ITEMS,$TIME" >> "$OUTPUT_CSV"
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