#!/usr/bin/env python3
"""
Test JAX/MJX Setup
Quick script to verify GPU-accelerated training environment is working
"""

import sys
import traceback


def test_jax():
    """Test JAX installation and GPU availability"""
    print("\n" + "="*60)
    print("1. Testing JAX Installation")
    print("="*60)

    try:
        import jax
        import jax.numpy as jnp
        print(f"✓ JAX version: {jax.__version__}")
        print(f"✓ JAX backend: {jax.default_backend()}")
        print(f"✓ JAX devices: {jax.devices()}")

        # Test GPU computation
        x = jnp.ones(10)
        y = jnp.sum(x)
        print(f"✓ JAX computation test passed (sum={y})")

        # Check for GPU
        has_gpu = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in jax.devices())
        if has_gpu:
            print("✓ GPU detected and available!")
        else:
            print("⚠ WARNING: No GPU detected. Training will be slow.")
            print("  Install JAX with CUDA: pip install --upgrade 'jax[cuda12]'")

        return True
    except ImportError as e:
        print(f"✗ JAX not installed: {e}")
        print("  Install with: pip install --upgrade 'jax[cuda12]'")
        return False
    except Exception as e:
        print(f"✗ JAX test failed: {e}")
        traceback.print_exc()
        return False


def test_mujoco_mjx():
    """Test MuJoCo MJX installation"""
    print("\n" + "="*60)
    print("2. Testing MuJoCo MJX")
    print("="*60)

    try:
        import mujoco
        from mujoco import mjx
        print(f"✓ MuJoCo version: {mujoco.__version__}")
        print("✓ MuJoCo MJX available")
        return True
    except ImportError as e:
        print(f"✗ MuJoCo MJX not installed: {e}")
        print("  Install with: pip install mujoco-mjx>=3.2.0")
        return False
    except Exception as e:
        print(f"✗ MuJoCo MJX test failed: {e}")
        traceback.print_exc()
        return False


def test_brax():
    """Test Brax installation"""
    print("\n" + "="*60)
    print("3. Testing Brax")
    print("="*60)

    try:
        import brax
        from brax.training.agents import ppo
        print(f"✓ Brax available")
        print("✓ Brax PPO available")
        return True
    except ImportError as e:
        print(f"✗ Brax not installed: {e}")
        print("  Install with: pip install brax>=0.10.0")
        return False
    except Exception as e:
        print(f"✗ Brax test failed: {e}")
        traceback.print_exc()
        return False


def test_other_deps():
    """Test other dependencies"""
    print("\n" + "="*60)
    print("4. Testing Other Dependencies")
    print("="*60)

    deps = {
        'flax': 'Flax (neural networks)',
        'optax': 'Optax (optimization)',
        'orbax': 'Orbax (checkpointing)',
    }

    all_ok = True
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} not installed")
            all_ok = False

    # Optional dependencies
    try:
        import wandb
        print("✓ Weights & Biases (optional) installed")
    except ImportError:
        print("⚠ Weights & Biases (optional) not installed")

    return all_ok


def test_environment():
    """Test G1 environment creation"""
    print("\n" + "="*60)
    print("5. Testing G1 Environment Creation")
    print("="*60)

    try:
        from legged_gym.envs.mjx_registry import create_g1_env
        import jax

        print("Creating G1 environment with 4 parallel envs...")
        env = create_g1_env(num_envs=4, backend='mjx')

        print(f"✓ Environment created successfully")
        print(f"  - Num envs: {env.num_envs}")
        print(f"  - Observations: {env.num_obs}")
        print(f"  - Actions: {env.num_actions}")

        # Test reset
        print("\nTesting environment reset...")
        rng = jax.random.PRNGKey(0)
        state, obs = env.reset(rng)
        print(f"✓ Reset successful")
        print(f"  - Observation shape: {obs.shape}")

        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        traceback.print_exc()
        return False


def test_precision_settings():
    """Test GPU precision settings"""
    print("\n" + "="*60)
    print("6. Checking GPU Precision Settings")
    print("="*60)

    import os
    precision = os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', 'not set')
    print(f"JAX_DEFAULT_MATMUL_PRECISION: {precision}")

    if precision == 'not set':
        print("\n⚠ WARNING: Precision not configured!")
        print("For NVIDIA Ampere GPUs (RTX 30/40 series), run:")
        print("  export JAX_DEFAULT_MATMUL_PRECISION=highest")
        print("Add to ~/.bashrc for permanent effect.")
    elif precision == 'highest':
        print("✓ Precision set to 'highest' (recommended for training)")
    else:
        print(f"⚠ Precision set to '{precision}'")
        print("  Consider using 'highest' for stable training")

    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("JAX/MJX SETUP TEST")
    print("="*60)

    results = {
        'JAX': test_jax(),
        'MuJoCo MJX': test_mujoco_mjx(),
        'Brax': test_brax(),
        'Dependencies': test_other_deps(),
        'Environment': test_environment(),
        'Precision': test_precision_settings(),
    }

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n🎉 All tests passed! You're ready to train.")
        print("\nTry running:")
        print("  python legged_gym/scripts/train_jax_ppo.py --robot g1 --num_envs 512")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        print("\nQuick fix: Install missing dependencies:")
        print("  pip install -e .[jax_gpu]")

    print()
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
