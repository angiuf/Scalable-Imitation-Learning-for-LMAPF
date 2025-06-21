#!/bin/bash
# Evaluation script for SILLM on custom warehouse environments
# Similar to eval_ltf.sh but for warehouse datasets

# Configuration
MODEL_PATH="pretrained_models/ltf_reeval/v3/IL/best"  # Update this path as needed
ROLLOUT_LENGTH=256
OUTPUT_FOLDER="exp_custom_warehouse"
# With PIBT-RL, the agents oscillates around the goal
WPPL_MODE="PIBT-RL"  # Default WPPL mode, can be changed in the loop

# Test different WPPL modes

# Run the evaluation
python evaluate_warehouse.py \
    --model_path $MODEL_PATH \
    --WPPL_mode $WPPL_MODE \
    --rollout_length $ROLLOUT_LENGTH \
    --output_folder $OUTPUT_FOLDER \
    --num_processes 1 \
    --num_devices 1

echo "Warehouse evaluation completed!"
