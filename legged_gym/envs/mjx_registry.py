"""
MJX Environment Registry
Central registry for JAX/MJX-based Unitree robot environments
"""

from typing import Dict, Callable, Any
from legged_gym.envs.base.mjx_legged_robot import MJXLeggedRobot


# Global registry
_MJX_ENV_REGISTRY: Dict[str, Callable] = {}


def register_mjx_env(name: str, creator_fn: Callable):
    """
    Register a MJX environment creator function.

    Args:
        name: Environment name (e.g., 'g1', 'go2')
        creator_fn: Function that creates the environment
    """
    _MJX_ENV_REGISTRY[name.lower()] = creator_fn
    print(f"[MJX Registry] Registered environment: {name}")


def create_mjx_env(name: str, **kwargs) -> MJXLeggedRobot:
    """
    Create a MJX environment by name.

    Args:
        name: Environment name
        **kwargs: Additional arguments passed to environment creator

    Returns:
        MJX environment instance
    """
    name = name.lower()

    if name not in _MJX_ENV_REGISTRY:
        available = ', '.join(_MJX_ENV_REGISTRY.keys())
        raise ValueError(
            f"Environment '{name}' not found in registry. "
            f"Available environments: {available}"
        )

    creator_fn = _MJX_ENV_REGISTRY[name]
    return creator_fn(**kwargs)


def list_mjx_envs():
    """List all registered MJX environments"""
    return list(_MJX_ENV_REGISTRY.keys())


# ============ Register Available Environments ============

def _register_g1():
    """Register G1 humanoid robot"""
    from legged_gym.envs.g1.mjx_g1_env import MJXG1Robot, create_g1_env
    from legged_gym.envs.g1.mujoco_g1_config import MujocoG1RoughCfg

    def creator(num_envs: int = 2048, backend: str = 'mjx', **kwargs):
        return create_g1_env(num_envs=num_envs, backend=backend)

    register_mjx_env('g1', creator)
    register_mjx_env('g1_humanoid', creator)  # Alias


def _register_go2():
    """Register Go2 quadruped robot (placeholder)"""
    def creator(num_envs: int = 2048, backend: str = 'mjx', **kwargs):
        raise NotImplementedError(
            "Go2 not yet ported to JAX/MJX. "
            "Please use PyTorch version or help port it!"
        )

    register_mjx_env('go2', creator)


def _register_h1():
    """Register H1 humanoid robot (placeholder)"""
    def creator(num_envs: int = 2048, backend: str = 'mjx', **kwargs):
        raise NotImplementedError(
            "H1 not yet ported to JAX/MJX. "
            "Please use PyTorch version or help port it!"
        )

    register_mjx_env('h1', creator)


def _register_h1_2():
    """Register H1_2 humanoid robot (placeholder)"""
    def creator(num_envs: int = 2048, backend: str = 'mjx', **kwargs):
        raise NotImplementedError(
            "H1_2 not yet ported to JAX/MJX. "
            "Please use PyTorch version or help port it!"
        )

    register_mjx_env('h1_2', creator)


# Auto-register all environments
def _init_registry():
    """Initialize registry with all available environments"""
    _register_g1()
    _register_go2()
    _register_h1()
    _register_h1_2()


# Initialize on import
_init_registry()


# ============ Convenience Functions ============

def create_g1_env(num_envs: int = 2048, backend: str = 'mjx'):
    """Convenience function to create G1 environment"""
    return create_mjx_env('g1', num_envs=num_envs, backend=backend)


def create_go2_env(num_envs: int = 2048, backend: str = 'mjx'):
    """Convenience function to create Go2 environment"""
    return create_mjx_env('go2', num_envs=num_envs, backend=backend)


def create_h1_env(num_envs: int = 2048, backend: str = 'mjx'):
    """Convenience function to create H1 environment"""
    return create_mjx_env('h1', num_envs=num_envs, backend=backend)


if __name__ == '__main__':
    # Test registry
    print("\nMJX Environment Registry")
    print("=" * 60)
    print(f"Available environments: {list_mjx_envs()}")
    print("=" * 60)

    # Try creating G1
    print("\nTesting G1 environment creation...")
    try:
        env = create_g1_env(num_envs=4)
        print(f"✓ Successfully created G1 environment with {env.num_envs} environments")
    except Exception as e:
        print(f"✗ Failed to create G1 environment: {e}")
