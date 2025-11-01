"""
JAX/MJX PPO Training Script for Unitree Robots
GPU-accelerated training using Brax PPO and MuJoCo MJX

Based on MuJoCo Playground's training approach, adapted for Unitree robots.
"""

import os
import argparse
from datetime import datetime
from typing import Callable, Dict, Any
import functools

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# Brax for PPO training
try:
    from brax import envs
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.io import model
except ImportError:
    print("ERROR: Brax not installed. Install with: pip install brax")
    exit(1)

# Logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging disabled")

from tensorboardX import SummaryWriter

# Import Unitree environments
from legged_gym.envs.g1.mjx_g1_env import MJXG1Robot, create_g1_env
from legged_gym.envs.g1.mujoco_g1_config import MujocoG1RoughCfg
from legged_gym import LEGGED_GYM_ROOT_DIR


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Unitree robots with JAX/MJX PPO')

    # Environment
    parser.add_argument('--robot', type=str, default='g1',
                        choices=['g1', 'go2', 'h1', 'h1_2'],
                        help='Robot to train')
    parser.add_argument('--num_envs', type=int, default=1024,
                        help='Number of parallel environments (T4: 1024, RTX4090: 4096, A100: 8192)')
    parser.add_argument('--backend', type=str, default='mjx',
                        choices=['mjx', 'mujoco'],
                        help='Physics backend (mjx=GPU, mujoco=CPU)')

    # Training
    parser.add_argument('--num_timesteps', type=int, default=100_000_000,
                        help='Total training timesteps')
    parser.add_argument('--episode_length', type=int, default=1000,
                        help='Maximum episode length')
    parser.add_argument('--num_minibatches', type=int, default=32,
                        help='Number of PPO minibatches')
    parser.add_argument('--num_updates_per_batch', type=int, default=4,
                        help='Number of PPO updates per batch')
    parser.add_argument('--reward_scaling', type=float, default=1.0,
                        help='Reward scaling factor')
    parser.add_argument('--entropy_cost', type=float, default=1e-2,
                        help='Entropy cost coefficient')
    parser.add_argument('--discounting', type=float, default=0.97,
                        help='Discount factor gamma')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--unroll_length', type=int, default=10,
                        help='Unroll length for PPO')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training')
    parser.add_argument('--num_evals', type=int, default=10,
                        help='Number of evaluation episodes')

    # Network architecture
    parser.add_argument('--policy_hidden_layer_sizes', type=int, nargs='+',
                        default=[256, 256, 256],
                        help='Policy network hidden layer sizes')
    parser.add_argument('--value_hidden_layer_sizes', type=int, nargs='+',
                        default=[256, 256, 256],
                        help='Value network hidden layer sizes')

    # Checkpointing
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Checkpoint save interval (in iterations)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory (default: logs/<robot>_jax/<timestamp>)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for wandb')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (in iterations)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')

    # Device
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    return parser.parse_args()


def create_environment(args):
    """Create Unitree robot environment"""
    print(f"\n{'='*60}")
    print(f"Creating {args.robot.upper()} environment")
    print(f"{'='*60}")

    if args.robot == 'g1':
        cfg = MujocoG1RoughCfg()
        cfg.env.num_envs = args.num_envs
        env = MJXG1Robot(cfg, backend=args.backend)
    else:
        raise NotImplementedError(f"Robot {args.robot} not yet ported to JAX/MJX")

    print(f"Environment: {args.robot}")
    print(f"Num envs: {args.num_envs}")
    print(f"Backend: {args.backend}")
    print(f"Observations: {env.num_obs}")
    print(f"Actions: {env.num_actions}")
    print(f"{'='*60}\n")

    return env


def wrap_for_brax(env: MJXG1Robot):
    """
    Wrap MJX environment to be compatible with Brax training.

    Brax expects environments to have:
    - reset(rng) -> state
    - step(state, action) -> next_state
    - And state should have obs, reward, done fields
    """

    class BraxWrapper:
        """Wrapper to make MJX environment Brax-compatible"""

        def __init__(self, env):
            self.env = env
            self.num_envs = env.num_envs
            self.observation_size = env.num_obs
            self.action_size = env.num_actions

        def reset(self, rng: jnp.ndarray):
            """Reset environment"""
            state, obs = self.env.reset(rng)

            # Create Brax-compatible state with obs field
            return state, obs

        def step(self, state, action: jnp.ndarray):
            """Step environment"""
            next_state, obs, reward, info = self.env.step(state, action)

            # Brax expects: next_state with .obs, .reward, .done
            # We need to attach these to the state for Brax compatibility
            # This is a workaround - in practice you'd modify the EnvState dataclass

            return next_state, obs, reward, info

    return BraxWrapper(env)


def make_ppo_networks(
    observation_size: int,
    action_size: int,
    policy_hidden_layer_sizes: tuple = (256, 256, 256),
    value_hidden_layer_sizes: tuple = (256, 256, 256),
):
    """Create PPO policy and value networks"""
    return ppo_networks.make_ppo_networks(
        observation_size,
        action_size,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
    )


def create_train_fn(env_wrapper, args):
    """Create Brax-compatible training function"""

    def env_reset_fn(rng):
        """Reset function for Brax"""
        state, obs = env_wrapper.reset(rng)
        return state, obs

    def env_step_fn(state, action):
        """Step function for Brax"""
        next_state, obs, reward, info = env_wrapper.step(state, action)
        done = next_state.done
        return next_state, obs, reward, done, info

    # Create network factory
    network_factory = functools.partial(
        make_ppo_networks,
        policy_hidden_layer_sizes=tuple(args.policy_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(args.value_hidden_layer_sizes),
    )

    return env_reset_fn, env_step_fn, network_factory


def train(args):
    """Main training function"""
    print(f"\n{'='*60}")
    print("JAX/MJX PPO TRAINING - GPU ACCELERATED")
    print(f"{'='*60}")
    print(f"Robot: {args.robot}")
    print(f"Num environments: {args.num_envs}")
    print(f"Backend: {args.backend}")
    print(f"Total timesteps: {args.num_timesteps:,}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Setup checkpoint directory
    if args.checkpoint_dir is None:
        timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        checkpoint_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            'logs',
            f'{args.robot}_jax',
            timestamp
        )
    else:
        checkpoint_dir = args.checkpoint_dir

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}\n")

    # Initialize wandb
    if args.use_wandb and WANDB_AVAILABLE:
        exp_name = args.experiment_name or f"{args.robot}_jax"
        wandb.init(
            project="unitree_rl_jax",
            name=exp_name,
            config=vars(args)
        )

    # Initialize TensorBoard
    tb_writer = SummaryWriter(log_dir=checkpoint_dir)

    # Create environment
    env = create_environment(args)
    env_wrapper = wrap_for_brax(env)

    # Create training functions
    env_reset_fn, env_step_fn, network_factory = create_train_fn(env_wrapper, args)

    # Initialize RNG
    rng = random.PRNGKey(args.seed)

    print("Starting training...")
    print(f"{'='*60}\n")

    # Training loop - simplified version
    # In practice, you'd use Brax's full PPO training loop
    # For now, this is a structure showing how to organize the training

    training_state = None  # Would be initialized by Brax PPO
    iteration = 0

    # Placeholder for actual Brax PPO training
    # You would call something like:
    # train_fn = ppo.train(...)
    # training_state = train_fn(rng, ...)

    print("\n" + "="*60)
    print("TRAINING SETUP COMPLETE")
    print("="*60)
    print("\nNOTE: Full Brax integration requires:")
    print("1. Converting environment step/reset to exact Brax format")
    print("2. Using brax.training.agents.ppo.train() for the training loop")
    print("3. Implementing proper state handling for vectorized envs")
    print("\nThe environment and network architecture are ready!")
    print("See: https://github.com/google/brax/tree/main/brax/training")
    print("="*60 + "\n")

    # Save configuration
    import json
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to: {config_path}")

    # Cleanup
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    tb_writer.close()

    return checkpoint_dir


def evaluate(args, checkpoint_path: str):
    """Evaluate trained policy"""
    print(f"\nEvaluating checkpoint: {checkpoint_path}")

    # Create environment
    env = create_environment(args)

    # Load checkpoint
    # params = model.load_params(checkpoint_path)

    # Run evaluation episodes
    # ...

    print("Evaluation complete!")


if __name__ == '__main__':
    # Check JAX GPU availability
    print("JAX devices:", jax.devices())
    print("JAX default backend:", jax.default_backend())

    # Enable TF32 for better performance on Ampere GPUs
    # Note: For training stability, might need to set JAX_DEFAULT_MATMUL_PRECISION=highest
    import os
    if 'JAX_DEFAULT_MATMUL_PRECISION' not in os.environ:
        print("\nWARNING: JAX_DEFAULT_MATMUL_PRECISION not set!")
        print("For NVIDIA Ampere GPUs (RTX 30/40 series), run:")
        print("  export JAX_DEFAULT_MATMUL_PRECISION=highest")
        print("to avoid TF32 precision issues.\n")

    args = get_args()

    # Train
    checkpoint_dir = train(args)

    print(f"\n{'='*60}")
    print("TRAINING SCRIPT EXECUTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    print("\nNext steps:")
    print("1. Complete Brax environment wrapper integration")
    print("2. Implement full PPO training loop with Brax")
    print("3. Add checkpoint saving/loading")
    print("4. Add evaluation and video rendering")
    print(f"{'='*60}\n")
