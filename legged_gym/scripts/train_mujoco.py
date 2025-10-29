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

    # rsl_rl OnPolicyRunner expects BOTH flat keys AND nested sections
    # Build a config dict that has both structures
    final_cfg = {}

    # Build nested sections
    runner_cfg = {}
    algorithm_cfg = {}
    policy_cfg = {}

    # Extract runner attributes
    runner_attrs = [attr for attr in dir(train_cfg.runner) if not attr.startswith('_')]
    for attr in runner_attrs:
        value = getattr(train_cfg.runner, attr)
        runner_cfg[attr] = value
        final_cfg[attr] = value  # Also add to top level

    # Extract algorithm attributes
    algorithm_attrs = [attr for attr in dir(train_cfg.algorithm) if not attr.startswith('_')]
    for attr in algorithm_attrs:
        value = getattr(train_cfg.algorithm, attr)
        algorithm_cfg[attr] = value
        final_cfg[attr] = value  # Also add to top level

    # Extract policy attributes
    policy_attrs = [attr for attr in dir(train_cfg.policy) if not attr.startswith('_')]
    for attr in policy_attrs:
        value = getattr(train_cfg.policy, attr)
        policy_cfg[attr] = value
        final_cfg[attr] = value  # Also add to top level

    # Add nested sections
    final_cfg['runner'] = runner_cfg
    final_cfg['algorithm'] = algorithm_cfg
    final_cfg['policy'] = policy_cfg

    # rsl_rl expects 'class_name' in policy config dict
    # Get it from policy_class_name at top level
    if 'policy_class_name' in final_cfg and 'class_name' not in policy_cfg:
        policy_cfg['class_name'] = final_cfg['policy_class_name']
        final_cfg['policy']['class_name'] = final_cfg['policy_class_name']

    # Add top-level attributes (like seed, runner_class_name)
    for key in ['seed', 'runner_class_name']:
        if hasattr(train_cfg, key):
            final_cfg[key] = getattr(train_cfg, key)

    # Ensure all critical fields are present
    required_fields = {
        'num_steps_per_env': 24,
        'algorithm_class_name': 'PPO',
        'policy_class_name': 'ActorCriticRecurrent',
        'max_iterations': 10000,
        'save_interval': 500,
        # obs_groups defines which observation dict keys go to which network
        # Explicitly map observation dict keys to actor/critic
        'obs_groups': {
            'policy': ['policy'],  # Policy network uses 'policy' observations
            'critic': ['critic'],  # Critic network uses 'critic' observations (privileged)
        },
        'privileged_obs_groups': {},
    }

    for key, default_value in required_fields.items():
        if key not in final_cfg:
            final_cfg[key] = default_value
            print(f"[WARNING] Added missing config: {key} = {default_value}")

    print(f"\nTraining configuration:")
    print(f"  - Policy: {final_cfg.get('policy_class_name')}")
    print(f"  - Steps per env: {final_cfg.get('num_steps_per_env')}")
    print(f"  - Learning rate: {final_cfg.get('learning_rate', 'N/A')}")
    print(f"  - Algorithm: {final_cfg.get('algorithm_class_name')}")

    runner = OnPolicyRunner(env, final_cfg, log_dir, device=args.rl_device)

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
