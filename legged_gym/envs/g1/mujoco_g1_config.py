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

    class policy(G1RoughCfgPPO.policy):
        # Inherit all policy settings from parent
        # G1 uses: init_noise_std=0.8, actor/critic hidden_dims=[32], activation='elu'
        # rnn_type='lstm', rnn_hidden_size=64, rnn_num_layers=1
        pass

    class algorithm(G1RoughCfgPPO.algorithm):
        # Inherit all algorithm settings from parent
        # G1 uses: entropy_coef=0.01, plus all base settings
        pass

    class runner(G1RoughCfgPPO.runner):
        # Explicitly inherit required attributes to ensure they're present
        policy_class_name = "ActorCriticRecurrent"  # From G1RoughCfgPPO
        algorithm_class_name = 'PPO'  # From LeggedRobotCfgPPO
        num_steps_per_env = 24  # From LeggedRobotCfgPPO

        # Mujoco-specific overrides for Colab
        experiment_name = 'g1_colab_training'
        run_name = 'run_001'
        max_iterations = 1000  # ~3.5 hours on T4 GPU (can stop early or resume)
        save_interval = 50  # Save every ~100 minutes (~1.7 hours)

        # Resume settings (from base)
        resume = False
        load_run = -1
        checkpoint = -1
