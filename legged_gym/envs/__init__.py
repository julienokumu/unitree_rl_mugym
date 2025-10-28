from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry

# Mujoco-based environments (Colab-compatible, no Isaac Gym required)
# Import these first as they don't depend on Isaac Gym
from legged_gym.envs.g1.mujoco_g1_config import MujocoG1RoughCfg, MujocoG1RoughCfgPPO
from legged_gym.envs.g1.mujoco_g1_env import MujocoG1Robot

# Register Mujoco environments (always available)
task_registry.register("g1_mujoco", MujocoG1Robot, MujocoG1RoughCfg(), MujocoG1RoughCfgPPO())

# Isaac Gym environments (optional - only available if Isaac Gym is installed)
try:
    from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
    from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
    from legged_gym.envs.h1.h1_env import H1Robot
    from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
    from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
    from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
    from legged_gym.envs.g1.g1_env import G1Robot
    from .base.legged_robot import LeggedRobot

    # Register Isaac Gym environments
    task_registry.register("go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
    task_registry.register("h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
    task_registry.register("h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
    task_registry.register("g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())

    print("[INFO] Isaac Gym environments loaded successfully")
except ImportError as e:
    print(f"[INFO] Isaac Gym not available. Only Mujoco environments will be available.")
    print(f"[INFO] To use Isaac Gym environments, install Isaac Gym locally.")
