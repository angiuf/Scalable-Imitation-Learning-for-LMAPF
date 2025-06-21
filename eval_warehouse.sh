# the root folder for output
DATASET_DIR=/home/giuffrida/code/master_thesis_MAPF_DRL/baselines/Dataset/
# the root folder for pretrained weights
# IL is trained with our Scalable Imitation Algorithm
# RL is trained with MAPPO
MODEL_FOLDER=pretrained_models/dynamic_guidance/v4/IL 
# PIBT, PIBT-RL, PIBT-LNS, PIBT-RL-LNS. 
# PIBT-RL will load pretrained weights, PIBT will just call the original PIBT.
# The LNS version will call LNS after PIBT initialization, used for training.
WPPL_mode=PIBT-RL 
# NUM_DEVICE=1

ROLLOUT_LENGTH=256

######### small maps for training ##########

## NOTE: there is another sortation_small without uniform. 
## It means that the start and goal locations are not generated uniformaly but 
## by rules in the League of Robot Runner 2023 Competition.
python evaluate_warehouse.py \
 --dataset_dir ${DATASET_DIR} \
 --model_path ${MODEL_FOLDER}/sortation/best \
 --WPPL_mode ${WPPL_mode} \
 --rollout_length ${ROLLOUT_LENGTH}