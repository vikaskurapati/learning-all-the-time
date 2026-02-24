#!/bin/bash

# 1. Input Validation
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <data_type>"
    exit 1
fi

DTYPE=$1
OUTPUT_FILE="benchmark_results_new_"$1".csv"

# 2. CSV Setup (Added WG_Size and Items headers)
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "VectorSize,WG_Size,Items,WarmupTime_ms,WarmupBandwidth_GBs,AvgTime_ms,AvgBandwidth_GBs" > $OUTPUT_FILE
fi

# 3. Define Grids for Tuning
SIZES=(1 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000)
WG_SIZES=(128 256 512 1024)
ITEMS_ARR=(4 8 16)

echo "Starting automated tuning for type: $DTYPE"
echo "------------------------------------------------"

# Nested loop to test combinations
for wg in "${WG_SIZES[@]}"; do
    for items in "${ITEMS_ARR[@]}"; do
        
        echo "================================================"
        echo "Compiling for WG_SIZE=$wg, ITEMS=$items..."
        
        # Inject the macros via compiler flags
        nvcc -O3 -arch=sm_90 \
             -DTUNING_WG_SIZE=$wg \
             -DTUNING_ITEMS=$items \
             reduce.cu -o reduce
             
        if [ $? -ne 0 ]; then
            echo "Compilation failed for WG=$wg, Items=$items. Skipping..."
            continue
        fi
        
        for size in "${SIZES[@]}"; do
            
            echo -n "  Running size $size... "
            OUTPUT=$(./reduce $size $DTYPE 2>&1)
            
            # Extract metrics
            WARM_TIME=$(echo "$OUTPUT" | grep "Warmup Time:" | awk '{print $3}')
            WARM_BW=$(echo "$OUTPUT" | grep "Warump Bandwidth:" | awk '{print $3}')
            AVG_TIME=$(echo "$OUTPUT" | grep "Average Time:" | awk '{print $3}')
            AVG_BW=$(echo "$OUTPUT" | grep "^Bandwidth:" | awk '{print $2}')
            
            # Handle Errors
            if [ -z "$AVG_TIME" ]; then
                WARM_TIME="ERR"
                WARM_BW="ERR"
                AVG_TIME="ERR"
                AVG_BW="ERR"
            fi

            # Append data row including the configuration
            echo "$size,$wg,$items,$WARM_TIME,$WARM_BW,$AVG_TIME,$AVG_BW" >> $OUTPUT_FILE
            echo "Done."
            
        done
    done
done

echo "------------------------------------------------"
echo "Tuning complete! Results saved to $OUTPUT_FILE"