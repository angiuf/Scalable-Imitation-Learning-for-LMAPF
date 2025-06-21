#!/bin/bash
# Evaluation script for SILLM on custom warehouse environments
# Similar to eval_ltf.sh but for warehouse datasets

# Configuration
MODEL_PATH="pretrained_models/ltf_reeval/v3/IL/best"  # Update this path as needed
NUM_TESTS=10  # Reduced for testing
ROLLOUT_LENGTH=256
OUTPUT_FOLDER="exp_custom_warehouse"

# Test different WPPL modes
for WPPL_mode in PIBT PIBT-RL PIBT-IL; do
    echo "Evaluating with WPPL mode: $WPPL_mode"
    
    # Run the evaluation
    python evaluate_warehouse.py \
        --model_path $MODEL_PATH \
        --WPPL_mode $WPPL_mode \
        --rollout_length $ROLLOUT_LENGTH \
        --output_folder $OUTPUT_FOLDER \
        --num_processes 1 \
        --num_devices 1
done

echo "Warehouse evaluation completed!"
