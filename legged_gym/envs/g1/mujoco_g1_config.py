"""
Mujoco G1 Robot Configuration
Inherits from G1RoughCfg with Mujoco-specific adjustments
"""

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO


class MujocoG1RoughCfg(G1RoughCfg):
    """
    Configuration for G1 robot in Mujoco.
    Most settings inherited from G1RoughCfg.
    """

    class env(G1RoughCfg.env):
        # Reduce number of environments for CPU training
        num_envs = 256  # Down from 4096 (Isaac Gym default)
        # Can be overridden from command line with --num_envs

    class terrain(G1RoughCfg.terrain):
        # Use plane terrain for Mujoco (heightfield not implemented yet)
        mesh_type = 'plane'
        curriculum = False

    class domain_rand(G1RoughCfg.domain_rand):
        # Keep domain randomization for robustness
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5


class MujocoG1RoughCfgPPO(G1RoughCfgPPO):
    """
    PPO training configuration for Mujoco G1.
    Uses same hyperparameters as Isaac Gym version.
    """

    class runner(G1RoughCfgPPO.runner):
        experiment_name = 'g1_mujoco'
        # Training may take longer on CPU, but keeping same iteration count
        max_iterations = 10000
        # Save more frequently since training is slower
        save_interval = 500
