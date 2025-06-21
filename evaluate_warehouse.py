#!/usr/bin/env python3
"""
Custom warehouse environment evaluation script for SILLM on MAPF tasks.
Adapted from evaluate_copy.py to work with Dataset folder structure similar to other baseline models.
"""

import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from torch.distributions import Categorical
import time
import os
import argparse
import json
import csv
import datetime
import tempfile
import traceback
import shutil
from pathlib import Path
from tqdm import tqdm
from default_configs import default_configs
import sys

# SILLM imports
from light_malib.envs.LMAPF.env import MultiLMAPFEnv
from light_malib.utils.logger import Logger
from light_malib.utils.cfg import load_cfg
from light_malib.rollout.rollout_func_LMAPF import rollout_func, rollout_func_for_WPPL
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.algorithm.mappo.policy import MAPPO
from light_malib.envs.LMAPF.WPPL import WPPL
from light_malib.utils.timer import global_timer
from torch.multiprocessing import Pool
from easydict import EasyDict

# https://github.com/pytorch/pytorch/issues/82843
multiprocessing.set_start_method("spawn", force=True)

# Global variables for multiprocessing
policy = None
device = None

# Parse arguments
def parse_arguments():
    arg_parser = argparse.ArgumentParser(description="Evaluate SILLM on custom warehouse environments (MAPF mode).")
    arg_parser.add_argument("--dataset_dir", type=str, default="/home/giuffrida/code/master_thesis_MAPF_DRL/baselines/Dataset", help="Path to dataset directory")
    arg_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    arg_parser.add_argument("--map_weights_path", type=str, default="NONE", help="Path to the map weights file.")
    arg_parser.add_argument("--WPPL_mode", type=str, default="PIBT-RL", choices=["PIBT","PIBT-RL","PIBT-IL","PIBT-LNS","PIBT-RL-LNS"], help="WPPL mode to use.")
    arg_parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use for evaluation.")
    arg_parser.add_argument("--num_devices", type=int, default=1, help="Number of GPU devices to use for evaluation.")
    arg_parser.add_argument("--rollout_length", type=int, default=256, help="Maximum episode length.")
    arg_parser.add_argument("--output_folder", type=str, default="exp_custom_warehouse", help="Output folder to save the results.")
    return arg_parser.parse_args()

# Map configurations to test (similar to other baseline models)
def get_map_configurations():
    return [
        {
            "map_name": "15_15_simple_warehouse",
            "size": 15,
            "n_tests": 200,
            "list_num_agents": [4, 8, 12, 16, 20, 22]
        },
        {
            "map_name": "50_55_simple_warehouse",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4,8,16,32,64,128,256]
        },
        {
            "map_name": "50_55_long_shelves",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4,8,16,32,64,128,256]
        },
        {
            "map_name": "50_55_open_space_warehouse_bottom",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4,8,16,32,64,128,256]
        }
    ]

def load_map_and_scenario(dataset_dir, map_name, num_agents, test_id):
    """Load map data and scenario from dataset files."""
    dataset_path = Path(dataset_dir)
    
    # Load map
    map_file_path = dataset_path / map_name / "input" / "map" / f"{map_name}.npy"
    if not map_file_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_file_path}")
    
    map_data = np.load(map_file_path)
    
    # Load scenario (start and goal positions)
    case_file_path = dataset_path / map_name / "input" / "start_and_goal" / f"{num_agents}_agents" / f"{map_name}_{num_agents}_agents_ID_{str(test_id).zfill(3)}.npy"
    if not case_file_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {case_file_path}")
    
    positions = np.load(case_file_path, allow_pickle=True)
    start_positions = positions[:, 0]  # Shape: (num_agents, 2)
    goal_positions = positions[:, 1]   # Shape: (num_agents, 2)
    
    return map_data, start_positions, goal_positions

def count_collisions(positions, obstacle_map, map_width):
    """Count agent-agent and obstacle collisions from LMAPF position data.
    
    Args:
        positions: 2D array where each row represents a timestep, each column an agent's position (as flattened index)
        obstacle_map: 2D numpy array representing the map (1 = obstacle, 0 = free)
        map_width: Width of the map (number of columns)
    """
    agent_agent_collisions = 0
    obstacle_collisions = 0
    
    if len(positions) == 0:
        return agent_agent_collisions, obstacle_collisions
    
    # Convert to numpy array if not already
    positions = np.array(positions)
    
    # positions shape: (timesteps, num_agents)
    timesteps, num_agents = positions.shape
    
    # Convert flattened indices to (row, col) coordinates and count collisions
    for timestep in range(timesteps):
        timestep_positions = positions[timestep]
        agent_coords = []
        
        for agent_idx in range(num_agents):
            flat_pos = timestep_positions[agent_idx]
            
            # Convert flattened position to (row, col)
            row = flat_pos // map_width
            col = flat_pos % map_width
            
            # Check obstacle collision
            if (0 <= row < obstacle_map.shape[0] and 
                0 <= col < obstacle_map.shape[1] and 
                obstacle_map[row, col] == 1):  # 1 = obstacle in numpy format
                obstacle_collisions += 1
            
            agent_coords.append((row, col))
        
        # Check agent-agent collisions
        for i in range(len(agent_coords)):
            for j in range(i + 1, len(agent_coords)):
                if agent_coords[i] == agent_coords[j]:
                    agent_agent_collisions += 1
    
    return agent_agent_collisions, obstacle_collisions


def compute_solution_from_positions(positions, goals, map_height, map_width):
    """Convert LMAPF position data to solution format similar to test_custom_env.py.
    
    Args:
        positions: 2D array where each row represents a timestep, each column an agent's position (as flattened index)
        goals: 1D array of goal positions (as flattened indices)
        map_width: Width of the map
        
    Returns:
        List of agent paths in (row, col) format
    """
    if len(positions) == 0:
        return []
    
    positions = np.array(positions)
    goals = np.array(goals)
    
    timesteps, num_agents = positions.shape
    solution = [[] for _ in range(num_agents)]

    done = [False] * num_agents

    for timestep in range(timesteps):
        for agent_idx in range(num_agents):
            if done[agent_idx]:
                continue
            flat_pos = positions[timestep][agent_idx]
            row = flat_pos // map_height
            col = flat_pos % map_width
            solution[agent_idx].append((row, col))
            if positions[timestep][agent_idx] == goals[agent_idx]:
                done[agent_idx] = True

    return solution


def check_agents_reached_goals(positions, goals, map_width):
    """Check if all agents reached their goals.
    
    Args:
        positions: 2D array where each row represents a timestep, each column an agent's position
        goals: 1D array of goal positions
        map_width: Width of the map
        
    Returns:
        Boolean indicating if all agents reached their goals
    """
    if len(positions) == 0 or len(goals) == 0:
        return False
    
    positions = np.array(positions)
    goals = np.array(goals)
    
    done = [False] * len(goals)

    for timestep_position in positions:
        for agent_idx in range(len(timestep_position)):
            flat_pos = timestep_position[agent_idx]
            if flat_pos == goals[agent_idx]:
                done[agent_idx] = True
    
    return all(done)

def get_csv_logger(model_dir, default_model_name):
    """Create CSV logger for results."""
    model_dir_path = Path(model_dir)
    csv_path = model_dir_path / f"log-{default_model_name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

def create_temporary_map_file(map_data, temp_dir):
    """Create a temporary .map file from numpy map data."""
    height, width = map_data.shape
    
    # Create temporary map file
    map_file = temp_dir / "temp_map.map"
    with open(map_file, 'w') as f:
        f.write("type octile\n")
        f.write(f"height {height}\n")
        f.write(f"width {width}\n")
        f.write("map\n")
        
        for row in range(height):
            line = ""
            for col in range(width):
                if map_data[row, col] == 1:  # Obstacle
                    line += "@"
                else:  # Free space
                    line += "."
            f.write(line + "\n")
    
    return str(map_file)

def create_temporary_start_goal_files(start_positions, goal_positions, temp_dir, map_width):
    """Create temporary start and goal files for LMAPF."""
    num_agents = len(start_positions)
    
    # Create start file (.agents format used by LMAPF)
    start_file = temp_dir / "temp_start.agents"
    with open(start_file, 'w') as f:
        f.write(f"{num_agents}\n")
        for i, pos in enumerate(start_positions):
            # LMAPF expects flattened index: row*width + col
            flattened_idx = pos[0] * map_width + pos[1]
            f.write(f"{flattened_idx}\n")
    
    # Create goal file (.tasks format used by LMAPF) 
    goal_file = temp_dir / "temp_goal.tasks"
    with open(goal_file, 'w') as f:
        f.write(f"{num_agents}\n")
        for i, pos in enumerate(goal_positions):
            # LMAPF expects flattened index: row*width + col
            flattened_idx = pos[0] * map_width + pos[1]
            f.write(f"{flattened_idx}\n")
    
    return str(start_file), str(goal_file)

def create_mapf_environment_config(map_file, num_agents, rollout_length, 
                                  WPPL_mode):
    """Create environment configuration for LMAPF that works like MAPF."""
    config = {
        'rollout_length': rollout_length,
        'device': 'cpu',
        'gae_gamma': 1.0,
        'gae_lambda': 0.95,
        'mappo_reward': False,
        'instances': [{"map_path": map_file, "agent_bins": [num_agents]}],
        # 'map_path': map_file,  # Use the temporary map file created earlier,
        # 'num_robots': num_agents,  # Number of agents for this map
        'device': 'cpu',  # Use CPU for evaluation
        # 'map_path': map_file,
        # 'num_robots': num_agents,
        'map_weights_path': '',
        'use_rank_feats': False,  # Additional field that might be needed
        'use_guiding_path': False,  # Enable guiding path to match original behavior
        'WPPL': {
            'mode': WPPL_mode,
            'verbose': True,
            'max_iterations': 5000,
            'num_threads': 1,
            'window_size': 1,  # Default window size for LMAPF
            'time_limit': 0.0,  # Default time limit for LMAPF
        }
    }
    
    # Convert to EasyDict to enable attribute access (required by LMAPF)
    config = EasyDict(config)
    
    return config

def convert_to_mapf_mode(env, start_positions, goal_positions):
    """Configure the environment to work in MAPF mode (agents done when reaching goal)."""
    # This function would need to be implemented based on the specific LMAPF environment
    # For now, this is a placeholder - the actual implementation would depend on 
    # how the LMAPF environment can be configured to work in MAPF mode
    pass

def run_single_test(dataset_dir, map_name, num_agents, test_id, policy_ref, device_ref, 
                   map_weights_path, WPPL_mode, rollout_length):
    """Run a single test case using a pre-created environment."""
    
    temp_dir = None
    try:
         # Load map and scenario
        map_data, start_positions, goal_positions = load_map_and_scenario(
            dataset_dir, map_name, num_agents, test_id
        )
        
        # # Debug prints for map, starts, and goals
        # Logger.info(f"=== DEBUG INFO for {map_name}, {num_agents} agents, test {test_id} ===")
        # Logger.info(f"Map shape: {map_data.shape}")
        # Logger.info(f"Map data (first 10x10 or full if smaller):")
        # display_rows = min(10, map_data.shape[0])
        # display_cols = min(10, map_data.shape[1])
        # for i in range(display_rows):
        #     row_str = " ".join([f"{map_data[i, j]:2.0f}" for j in range(display_cols)])
        #     if map_data.shape[1] > 10:
        #         row_str += " ..."
        #     Logger.info(f"  Row {i:2d}: {row_str}")
        # if map_data.shape[0] > 10:
        #     Logger.info(f"  ... (showing first 10 rows of {map_data.shape[0]})")
        
        # Logger.info(f"Start positions ({len(start_positions)} agents):")
        # for i, pos in enumerate(start_positions):
        #     Logger.info(f"  Agent {i}: {pos} (row={pos[0]}, col={pos[1]})")
        
        # Logger.info(f"Goal positions ({len(goal_positions)} agents):")
        # for i, pos in enumerate(goal_positions):
        #     Logger.info(f"  Agent {i}: {pos} (row={pos[0]}, col={pos[1]})")
        
        # # Verify positions are within map bounds
        # map_height, map_width = map_data.shape
        # for i, (start, goal) in enumerate(zip(start_positions, goal_positions)):
        #     if not (0 <= start[0] < map_height and 0 <= start[1] < map_width):
        #         Logger.warning(f"Agent {i} start position {start} is out of map bounds!")
        #     if not (0 <= goal[0] < map_height and 0 <= goal[1] < map_width):
        #         Logger.warning(f"Agent {i} goal position {goal} is out of map bounds!")
        #     if map_data[start[0], start[1]] == -1:
        #         Logger.warning(f"Agent {i} start position {start} is on an obstacle!")
        #     if map_data[goal[0], goal[1]] == -1:
        #         Logger.warning(f"Agent {i} goal position {goal} is on an obstacle!")
        
        # Logger.info("=== END DEBUG INFO ===")
        
        # Use existing map file from dataset instead of creating temporary one
        dataset_path = Path(dataset_dir)
        map_file_path = dataset_path / map_name / "input" / "map" / f"{map_name}.map"
        
        if not map_file_path.exists():
            raise FileNotFoundError(f"Map file not found: {map_file_path}")
        
        map_file = str(map_file_path)
        Logger.debug(f"Using existing map file: {map_file}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create temporary start and goal files
            start_file, goal_file = create_temporary_start_goal_files(
                start_positions, goal_positions, temp_path, map_data.shape[1]
            )
            Logger.debug(f"Created temporary start file: {start_file}")
            Logger.debug(f"Created temporary goal file: {goal_file}")
            
            # Create environment configuration
            env_config = create_mapf_environment_config(
                map_file, num_agents, rollout_length,
                WPPL_mode
            )

            # Use the map_name parameter directly since we're using the actual dataset map
            # (not extracting from temporary file path)
            Logger.debug(f"Using map: {map_name} with {num_agents} agents")
            
            # Create environment and add our map to the map manager
            env = MultiLMAPFEnv("test", 42, env_config, device_ref, None, None)
            
            # Create a Map object and register it with the environment's map manager
            # temp_map = Map()
            # temp_map.load(map_file)
            # print("Map name:", map_name)
            # temp_map.name = map_name  # Set the name that will be used for lookup
            # temp_map.set_agent_bins([num_agents])  # Set the agent bins for this map
            
            # Add the map to the environment's map manager
            # env.map_manager.add_map(temp_map)
            
            Logger.debug(f"Available maps: {list(env.map_manager.maps_dict.keys())}")
            
            # Set the current environment to use our map
            # env.set_curr_env2(map_name, num_agents, verbose=False)
            # Logger.info(f"Set current environment to use map '{map_name}' with {num_agents} agents")
            
            # Load start and goal positions
            env.load_starts(start_file)
            env.load_tasks(goal_file)
            
            # Set up behavior policies
            agent = "agent_0"
            policy_id = "policy_0"
            behavior_policies = {
                agent: (policy_id, policy_ref),
            }
            
            # Run evaluation
            start_time = time.time()
            rollout_desc = RolloutDesc(test_id, "agent_0", None, None, None, None, None)
        
        rollout_results = rollout_func(
            eval=True,
            rollout_worker=None,
            rollout_desc=rollout_desc,
            env=env,
            behavior_policies=behavior_policies,
            data_server=None,
            rollout_length=rollout_length,
            render=False,
            verbose=False,
            collect_data=True,
            collect_log=False,
            device=device_ref,
            instance=(map_name, num_agents)
        )
        
        elapsed_time = time.time() - start_time

        Logger.debug(f"Rollout results for test {test_id}: {rollout_results}")

        # Extract results
        if rollout_results and "results" in rollout_results and len(rollout_results["results"]) > 0:
                result_data = rollout_results["results"][0]
                stats = result_data.get("stats", {}).get("agent_0", {})
                imitation_data = rollout_results.get('imitation_data', {})

                positions = imitation_data.get('curr_positions', [])
                goals = imitation_data.get('target_positions', [])
                
                # Extract goals - handle nested structure
                if len(goals) > 0:
                    goals = goals[0] if isinstance(goals[0], (list, np.ndarray)) else goals

                Logger.debug(f"Rollout results keys: {list(rollout_results.keys())}")
                Logger.debug(f"Result data keys: {list(result_data.keys())}")
                Logger.debug(f"Stats keys: {list(stats.keys())}")
                Logger.debug(f"Positions shape: {np.array(positions).shape if len(positions) > 0 else 'None'}")
                Logger.debug(f"Goals shape: {np.array(goals).shape if len(goals) > 0 else 'None'}")
                              
                Logger.debug(f"Positions: {positions}")
                Logger.debug(f"Goals: {goals}")
                # Primary method: Check if agents reached their goals using position data
                if len(positions) > 0 and len(goals) > 0:
                    map_width = map_data.shape[1]
                    success = check_agents_reached_goals(positions, goals, map_width)
                    Logger.debug(f"Success via goal achievement check: {success}")
                
                # Initialize variables
                solution = []
                
                # Compute metrics using actual position data if available
                if success and len(positions) > 0 and len(goals) > 0:
                    positions_array = np.array(positions)
                    map_height = map_data.shape[0]
                    map_width = map_data.shape[1]
                    
                    # Convert positions to solution format for metrics computation
                    solution = compute_solution_from_positions(positions, goals, map_height, map_width)
                    Logger.debug(solution)

                    
                    # Calculate episode length from actual data
                    actual_episode_length = 0
                    for agent_idx in range(len(solution)):
                        if len(solution[agent_idx]) > 0:
                            # Ensure the solution is not empty
                            actual_episode_length = max(actual_episode_length, len(solution[agent_idx])-1)
                    episode_length = max(actual_episode_length, 0)
                    
                    # Calculate steps and costs for each agent (like in test_custom_env.py)
                    agent_steps = []
                    agent_costs = []
                    
                    for agent_idx in range(num_agents):
                        if agent_idx < len(solution) and len(solution[agent_idx]) > 0:
                            # Count actual movements (steps), not wait actions
                            steps = 0
                            agent_path = solution[agent_idx]
                            
                            for i in range(1, len(agent_path)):
                                current_pos = agent_path[i]
                                previous_pos = agent_path[i-1]
                                # Only count as a step if the agent actually moved
                                if current_pos != previous_pos:
                                    steps += 1
                            
                            # Cost equals path length - 1 (excluding initial position)
                            cost = len(agent_path) - 1
                            agent_steps.append(steps)
                            agent_costs.append(cost)
                        else:
                            agent_steps.append(0)
                            agent_costs.append(0)
                    
                    # Compute aggregate metrics
                    total_steps = sum(agent_steps)
                    avg_steps = np.mean(agent_steps) if agent_steps else 0
                    max_steps = max(agent_steps) if agent_steps else 0
                    min_steps = min(agent_steps) if agent_steps else 0
                    
                    total_costs = sum(agent_costs)
                    avg_costs = np.mean(agent_costs) if agent_costs else 0
                    max_costs = max(agent_costs) if agent_costs else 0
                    min_costs = min(agent_costs) if agent_costs else 0
                    
                    # Count collisions using the corrected function
                    agent_coll, obs_coll = count_collisions(positions, map_data, map_width)
                    
                    # Calculate collision rates
                    if episode_length > 0 and num_agents > 0:
                        total_agent_timesteps = episode_length * num_agents
                        agent_coll_rate = agent_coll / total_agent_timesteps
                        obstacle_coll_rate = obs_coll / total_agent_timesteps
                        total_coll_rate = (agent_coll + obs_coll) / total_agent_timesteps
                    else:
                        agent_coll_rate = 0.0
                        obstacle_coll_rate = 0.0
                        total_coll_rate = 0.0
                    
                    crashed = (agent_coll + obs_coll) > 0
                    
                else:
                    # Failed case
                    episode_length = 0
                    total_steps = 0
                    avg_steps = 0
                    max_steps = 0
                    min_steps = 0
                    total_costs = 0
                    avg_costs = 0
                    max_costs = 0
                    min_costs = 0
                    agent_coll_rate = 0.0
                    obstacle_coll_rate = 0.0
                    total_coll_rate = 0.0
                    crashed = True
                
                result = {
                    'finished': success,
                    'time': elapsed_time,
                    'episode_length': episode_length,
                    'total_steps': total_steps,
                    'avg_steps': avg_steps,
                    'max_steps': max_steps,
                    'min_steps': min_steps,
                    'total_costs': total_costs,
                    'avg_costs': avg_costs,
                    'max_costs': max_costs,
                    'min_costs': min_costs,
                    'agent_coll_rate': agent_coll_rate,
                    'obstacle_coll_rate': obstacle_coll_rate,
                    'total_coll_rate': total_coll_rate,
                    'crashed': crashed
                }
                
        else:
            # Failed to get results - add detailed logging
            Logger.warning(f"No valid rollout results obtained")
            if rollout_results:
                Logger.warning(f"Rollout results structure: {rollout_results}")
            else:
                Logger.warning("rollout_results is None or empty")
                
            result = {
                'finished': False,
                'time': elapsed_time,
                'episode_length': 0,
                'total_steps': 0,
                'avg_steps': 0.0,
                'max_steps': 0,
                'min_steps': 0,
                'total_costs': 0,
                'avg_costs': 0.0,
                'max_costs': 0,
                'min_costs': 0,
                'agent_coll_rate': 0.0,
                'obstacle_coll_rate': 0.0,
                'total_coll_rate': 0.0,
                'crashed': True
            }
            solution = []
        
        return result, solution
        
    except Exception as e:
        Logger.error(f"Error in test {test_id}: {e}")
        traceback.print_exc()
        
        # Return failed result
        result = {
            'finished': False,
            'time': 0.0,
            'episode_length': 0,
            'total_steps': 0,
            'avg_steps': 0.0,
            'max_steps': 0,
            'min_steps': 0,
            'total_costs': 0,
            'avg_costs': 0.0,
            'max_costs': 0,
            'min_costs': 0,
            'agent_coll_rate': 0.0,
            'obstacle_coll_rate': 0.0,
            'total_coll_rate': 0.0,
            'crashed': True
        }
        return result, []
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def initialize_policy(model_path, num_processes, num_devices):
    """Initialize the policy model."""
    global policy
    global device
    
    if num_processes != 1:
        current = multiprocessing.current_process()
        worker_id = current._identity[0]
    else:
        worker_id = 0    
        
    if num_devices != 1:
        device = f"cuda:{worker_id % num_devices}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Logger.info(f"Worker {worker_id} using device {device}")

    # Load policy
    policy = MAPPO.load(model_path, env_agent_id="agent_0")
    policy = policy.to_device(device)
    Logger.info("Policy loaded successfully")

def evaluate_map_config(config, dataset_dir, output_folder, map_weights_path, WPPL_mode, rollout_length):
    """Evaluate a single map configuration.
    
    Environment is created once per map configuration to avoid the overhead of
    recreating it for each test. The environment is reset for each test with
    new start and goal positions.
    """
    map_name = config["map_name"]
    size = config["size"] 
    n_tests = config["n_tests"]
    list_num_agents = config["list_num_agents"]
    
    Logger.info(f"Processing map: {map_name}")
    
    # Create output directory
    output_dir = Path(output_folder) / map_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup CSV logger
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    sanitized_map_name = map_name.replace("/", "_").replace("\\", "_")
    csv_filename_base = f'SILLM_{WPPL_mode}_{sanitized_map_name}_{date}'
    csv_file, csv_logger = get_csv_logger(str(output_dir), csv_filename_base)
    
    # CSV header
    header = ["n_agents", 
              "success_rate", "time", "time_std", "time_min", "time_max",
              "episode_length", "episode_length_std", "episode_length_min", "episode_length_max",
              "total_steps", "total_steps_std", "total_steps_min", "total_steps_max",
              "avg_steps", "avg_steps_std", "avg_steps_min", "avg_steps_max",
              "max_steps", "max_steps_std", "max_steps_min", "max_steps_max",
              "min_steps", "min_steps_std", "min_steps_min", "min_steps_max",
              "total_costs", "total_costs_std", "total_costs_min", "total_costs_max",
              "avg_costs", "avg_costs_std", "avg_costs_min", "avg_costs_max",
              "max_costs", "max_costs_std", "max_costs_min", "max_costs_max",
              "min_costs", "min_costs_std", "min_costs_min", "min_costs_max",
              "agent_collision_rate", "agent_collision_rate_std", "agent_collision_rate_min", "agent_collision_rate_max",
              "obstacle_collision_rate", "obstacle_collision_rate_std", "obstacle_collision_rate_min", "obstacle_collision_rate_max",
              "total_collision_rate", "total_collision_rate_std", "total_collision_rate_min", "total_collision_rate_max"]
    
    csv_logger.writerow(header)
    csv_file.flush()
    
    # Process each agent count
    for num_agents in list_num_agents:
        Logger.info(f"Testing {num_agents} agents on {map_name}")
        
        # Initialize result storage
        results = {
            'finished': [], 'time': [], 'episode_length': [],
            'total_steps': [], 'avg_steps': [], 'max_steps': [], 'min_steps': [],
            'total_costs': [], 'avg_costs': [], 'max_costs': [], 'min_costs': [],
            'crashed': [], 'agent_coll_rate': [], 'obstacle_coll_rate': [], 'total_coll_rate': []
        }
        
        # Run tests
        for test_id in tqdm(range(n_tests), desc=f"{num_agents} agents"):
            try:
                Logger.debug(f"Running test {test_id} for {num_agents} agents on {map_name}")
                result, solution = run_single_test(
                    dataset_dir, map_name, num_agents, test_id, policy, device, 
                    map_weights_path, WPPL_mode, rollout_length
                )
                
                Logger.debug(f"Test {test_id} completed. Success: {result['finished']}")
                
                # Collect results (convert any tensors to scalars)
                results['finished'].append(tensor_to_scalar(result['finished']))
                if result['finished']:
                    results['time'].append(tensor_to_scalar(result['time']))
                    results['episode_length'].append(tensor_to_scalar(result['episode_length']))
                    results['total_steps'].append(tensor_to_scalar(result['total_steps']))
                    results['avg_steps'].append(tensor_to_scalar(result['avg_steps']))
                    results['max_steps'].append(tensor_to_scalar(result['max_steps']))
                    results['min_steps'].append(tensor_to_scalar(result['min_steps']))
                    results['total_costs'].append(tensor_to_scalar(result['total_costs']))
                    results['avg_costs'].append(tensor_to_scalar(result['avg_costs']))
                    results['max_costs'].append(tensor_to_scalar(result['max_costs']))
                    results['min_costs'].append(tensor_to_scalar(result['min_costs']))
                    results['agent_coll_rate'].append(tensor_to_scalar(result['agent_coll_rate']))
                    results['obstacle_coll_rate'].append(tensor_to_scalar(result['obstacle_coll_rate']))
                    results['total_coll_rate'].append(tensor_to_scalar(result['total_coll_rate']))
                    results['crashed'].append(tensor_to_scalar(result['crashed']))
                
            except Exception as e:
                Logger.error(f"Failed test {test_id}: {e}")
                import traceback
                Logger.error(f"Traceback: {traceback.format_exc()}")
                results['finished'].append(False)
        
        # Calculate aggregated metrics
        success_rate = np.mean(results['finished']) if results['finished'] else 0
        
        # Calculate statistics for successful runs
        metrics = {}
        metric_keys = ['time', 'episode_length', 'total_steps', 'avg_steps', 'max_steps', 'min_steps',
                      'total_costs', 'avg_costs', 'max_costs', 'min_costs',
                      'agent_coll_rate', 'obstacle_coll_rate', 'total_coll_rate']
        
        for key in metric_keys:
            if results[key]:
                metrics[f"{key}_mean"] = np.mean(results[key])
                metrics[f"{key}_std"] = np.std(results[key])
                metrics[f"{key}_min"] = np.min(results[key])
                metrics[f"{key}_max"] = np.max(results[key])
            else:
                metrics[f"{key}_mean"] = 0
                metrics[f"{key}_std"] = 0
                metrics[f"{key}_min"] = 0
                metrics[f"{key}_max"] = 0
        
        crashed_rate = np.mean(results['crashed']) if results['crashed'] else 0
        
        # Write to CSV
        csv_data = [
            num_agents,
            success_rate * 100,  # Convert to percentage
            metrics['time_mean'], metrics['time_std'], metrics['time_min'], metrics['time_max'],
            metrics['episode_length_mean'], metrics['episode_length_std'], metrics['episode_length_min'], metrics['episode_length_max'],
            metrics['total_steps_mean'], metrics['total_steps_std'], metrics['total_steps_min'], metrics['total_steps_max'],
            metrics['avg_steps_mean'], metrics['avg_steps_std'], metrics['avg_steps_min'], metrics['avg_steps_max'],
            metrics['max_steps_mean'], metrics['max_steps_std'], metrics['max_steps_min'], metrics['max_steps_max'],
            metrics['min_steps_mean'], metrics['min_steps_std'], metrics['min_steps_min'], metrics['min_steps_max'],
            metrics['total_costs_mean'], metrics['total_costs_std'], metrics['total_costs_min'], metrics['total_costs_max'],
            metrics['avg_costs_mean'], metrics['avg_costs_std'], metrics['avg_costs_min'], metrics['avg_costs_max'],
            metrics['max_costs_mean'], metrics['max_costs_std'], metrics['max_costs_min'], metrics['max_costs_max'],
            metrics['min_costs_mean'], metrics['min_costs_std'], metrics['min_costs_min'], metrics['min_costs_max'],
            metrics['agent_coll_rate_mean'], metrics['agent_coll_rate_std'], metrics['agent_coll_rate_min'], metrics['agent_coll_rate_max'],
            metrics['obstacle_coll_rate_mean'], metrics['obstacle_coll_rate_std'], metrics['obstacle_coll_rate_min'], metrics['obstacle_coll_rate_max'],
            metrics['total_coll_rate_mean'], metrics['total_coll_rate_std'], metrics['total_coll_rate_min'], metrics['total_coll_rate_max']
        ]
        
        csv_logger.writerow(csv_data)
        csv_file.flush()
        
        Logger.info(f"Results for {num_agents} agents: Success rate: {success_rate*100:.1f}%")
    
    csv_file.close()
    Logger.info(f"Completed evaluation for map: {map_name}")

def tensor_to_scalar(value):
    """Convert torch tensor to Python scalar, handling CUDA tensors."""
    if torch.is_tensor(value):
        return value.cpu().item() if value.numel() == 1 else value.cpu().numpy()
    return value

def reset_environment_for_test(env, dataset_dir, map_name, num_agents, test_id):
    """Reset environment for a new test with different agents and start/goal positions.
    
    This function is called for each individual test to load the specific 
    start and goal positions without recreating the entire environment.
    """
    
    # Load map and scenario for this specific test
    map_data, start_positions, goal_positions = load_map_and_scenario(
        dataset_dir, map_name, num_agents, test_id
    )
    
    # Create temporary directory that will be managed by the caller
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Create temporary start and goal files for this test
    start_file, goal_file = create_temporary_start_goal_files(
        start_positions, goal_positions, temp_path, map_data.shape[1]
    )
    
    # Load start and goal positions into the environment
    env.load_starts(start_file)
    env.load_tasks(goal_file)
    
    return map_data, start_positions, goal_positions, temp_dir

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    #  Sample arguments for testing
    # python evaluate_warehouse.py --dataset_dir /home/giuffrida/code/master_thesis_MAPF_DRL/baselines/Dataset/ --model_path pretrained_models/static_guidance/v3/RL/warehouse/best --WPPL_mode PIBT-RL --rollout_length 256 --num_processes 1
    
    # Configuration
    dataset_dir = Path(args.dataset_dir)
    model_path = args.model_path
    map_weights_path = args.map_weights_path
    WPPL_mode = args.WPPL_mode
    num_processes = args.num_processes
    num_devices = args.num_devices
    rollout_length = args.rollout_length
    output_folder = args.output_folder
    
    # Output setup
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_folder = os.path.join(output_folder, f"{timestamp}_SILLM_{WPPL_mode}")
    log_folder = os.path.join(output_folder, "log")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    
    Logger.info("Starting custom warehouse evaluation for SILLM")
    
    # Get map configurations
    map_configurations = get_map_configurations()
    
    # Initialize policy (single process for now)
    if num_processes == 1:
        initialize_policy(model_path, num_processes, num_devices)
        
        # Run evaluation for each map configuration
        for config in map_configurations:
            try:
                evaluate_map_config(config, dataset_dir, output_folder, 
                                  map_weights_path, WPPL_mode, rollout_length)
            except Exception as e:
                Logger.error(f"Failed to evaluate config {config}: {e}")
    else:
        Logger.warning("Multi-process evaluation not yet implemented")
        # TODO: Implement multi-process evaluation similar to original evaluate_copy.py
    
    Logger.info("Evaluation completed!")
