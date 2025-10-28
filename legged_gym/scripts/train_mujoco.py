"""
Mujoco Training Script
Train policies using Mujoco physics (no Isaac Gym required)
Works on Google Colab with CPU/GPU
"""

import os
import argparse
from datetime import datetime
import torch

# Import ONLY Mujoco-compatible modules (no Isaac Gym!)
from legged_gym.envs.g1.mujoco_g1_env import MujocoG1Robot
from legged_gym.envs.g1.mujoco_g1_config import MujocoG1RoughCfg, MujocoG1RoughCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict

from rsl_rl.runners import OnPolicyRunner


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train locomotion policy with Mujoco')

    # Task
    parser.add_argument('--task', type=str, default='g1_mujoco',
                        help='Task name (currently only g1_mujoco supported)')

    # Training
    parser.add_argument('--num_envs', type=int, default=None,
                        help='Number of parallel environments (default: 256 for CPU)')
    parser.add_argument('--max_iterations', type=int, default=None,
                        help='Maximum training iterations (default: 10000)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')

    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training (cpu or cuda)')
    parser.add_argument('--rl_device', type=str, default=None,
                        help='Device for RL algorithm (default: same as --device)')

    # Logging
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')
    parser.add_argument('--run_name', type=str, default='',
                        help='Run name for logging')

    # Resume
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--load_run', type=str, default=None,
                        help='Run name to load from (if resuming)')
    parser.add_argument('--checkpoint', type=int, default=-1,
                        help='Checkpoint to load (-1 for latest)')

    args = parser.parse_args()

    # Set rl_device to device if not specified
    if args.rl_device is None:
        args.rl_device = args.device

    return args


def train(args):
    """Main training function"""
    print("\n" + "="*60)
    print("MUJOCO TRAINING (Colab-Compatible)")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Device: {args.device}")
    print(f"RL Device: {args.rl_device}")
    print(f"Seed: {args.seed}")
    print("="*60 + "\n")

    # Set random seed
    torch.manual_seed(args.seed)

    # Create environment configuration
    env_cfg = MujocoG1RoughCfg()
    train_cfg = MujocoG1RoughCfgPPO()

    # Override from command line
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    if args.run_name:
        train_cfg.runner.run_name = args.run_name
    if args.resume:
        train_cfg.runner.resume = True
        if args.load_run:
            train_cfg.runner.load_run = args.load_run
        if args.checkpoint >= 0:
            train_cfg.runner.checkpoint = args.checkpoint

    env_cfg.seed = args.seed
    train_cfg.seed = args.seed

    print(f"Creating {env_cfg.env.num_envs} environments...")

    # Create environment
    env = MujocoG1Robot(cfg=env_cfg, device=args.device)

    print(f"\nEnvironment created successfully!")
    print(f"  - Observations: {env.num_obs}")
    print(f"  - Actions: {env.num_actions}")
    print(f"  - Max episode length: {env.max_episode_length}")

    # Create PPO runner
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    if train_cfg.runner.resume:
        log_dir = None  # Will be set by runner when loading
    else:
        log_dir = os.path.join(
            log_root,
            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name
        )

    print(f"\nLog directory: {log_dir if log_dir else 'Resuming from checkpoint'}")

    train_cfg_dict = class_to_dict(train_cfg)

    # Ensure all required runner config fields are present
    # (class_to_dict may not properly handle nested class inheritance)
    required_runner_fields = {
        'num_steps_per_env': 24,
        'algorithm_class_name': 'PPO',
        'policy_class_name': 'ActorCriticRecurrent',
        'max_iterations': 10000,
        'save_interval': 500,
        'experiment_name': 'g1_colab_training',
        'run_name': 'run_001',
        'resume': False,
        'load_run': -1,
        'checkpoint': -1,
    }

    if 'runner' not in train_cfg_dict or not isinstance(train_cfg_dict['runner'], dict):
        train_cfg_dict['runner'] = {}

    for key, default_value in required_runner_fields.items():
        if key not in train_cfg_dict['runner']:
            # Try to get from config class first
            if hasattr(train_cfg.runner, key):
                train_cfg_dict['runner'][key] = getattr(train_cfg.runner, key)
            else:
                train_cfg_dict['runner'][key] = default_value
            print(f"[WARNING] Added missing runner config: {key} = {train_cfg_dict['runner'][key]}")

    # Also ensure algorithm config is present
    if 'algorithm' not in train_cfg_dict or not isinstance(train_cfg_dict['algorithm'], dict):
        train_cfg_dict['algorithm'] = {}
        algorithm_attrs = [attr for attr in dir(train_cfg.algorithm) if not attr.startswith('_')]
        for attr in algorithm_attrs:
            train_cfg_dict['algorithm'][attr] = getattr(train_cfg.algorithm, attr)

    # Ensure policy config is present
    if 'policy' not in train_cfg_dict or not isinstance(train_cfg_dict['policy'], dict):
        train_cfg_dict['policy'] = {}
        policy_attrs = [attr for attr in dir(train_cfg.policy) if not attr.startswith('_')]
        for attr in policy_attrs:
            train_cfg_dict['policy'][attr] = getattr(train_cfg.policy, attr)

    print(f"\nTraining configuration:")
    print(f"  - Policy: {train_cfg_dict['policy'].get('policy_class_name', train_cfg_dict['runner'].get('policy_class_name'))}")
    print(f"  - Steps per env: {train_cfg_dict['runner']['num_steps_per_env']}")
    print(f"  - Learning rate: {train_cfg_dict['algorithm'].get('learning_rate', 'N/A')}")

    runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)

    # Load checkpoint if resuming
    if train_cfg.runner.resume:
        from legged_gym.utils.helpers import get_load_path
        resume_path = get_load_path(
            log_root,
            load_run=train_cfg.runner.load_run,
            checkpoint=train_cfg.runner.checkpoint
        )
        print(f"Loading model from: {resume_path}")
        runner.load(resume_path)

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Max iterations: {train_cfg.runner.max_iterations}")
    print(f"Save interval: {train_cfg.runner.save_interval}")
    print("="*60 + "\n")

    # Start training
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Models saved to: {runner.log_dir}")
    print("\nTo visualize in Mujoco:")
    print(f"  python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml")
    print("\nTo download models from Colab:")
    print(f"  from google.colab import files")
    print(f"  files.download('{runner.log_dir}/model_<iteration>.pt')")
    print("="*60 + "\n")


if __name__ == '__main__':
    args = get_args()
    train(args)
