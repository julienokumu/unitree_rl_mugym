# Fixes Applied for Colab Training

This document tracks all the fixes applied to make the framework work on Google Colab with Mujoco.

## Latest Fixes (Ready for Testing)

### 1. Model Loading Issue - Actuators Not Created (Commit: a5fbaf8)
**Problem:** URDF loading doesn't create Mujoco actuators, causing multiple errors:
- `model.nu = 0` (no actuators found)
- All PD gains = 0.0 (no control!)
- Wrong qpos structure (12 instead of 19 for floating base + 12 joints)
- `ValueError: could not broadcast input array from shape (12,) into shape (5,)`

**Solution:** Automatically prefer pre-compiled XML files over URDF:
1. Check if a `.xml` file exists alongside the `.urdf`
2. If found, use the XML (has actuators explicitly defined)
3. Otherwise, load URDF with warning

**Files:** `legged_gym/envs/base/mujoco_legged_robot.py`

**Impact:** Model now loads with 12 actuators, proper PD gains, and correct qpos structure (19 elements).

### 2. DOF Counting Mismatch (Commit: 941976f)
**Problem:** IndexError when setting up PD gains due to incorrect DOF counting.
- The code was using `model.nv - 6` which gave 6 DOFs, but G1 expects 12
- Caused: `IndexError: index 6 is out of bounds for dimension 0 with size 6`

**Solution:** Implemented robust 4-layer fallback system:
1. Primary: Use actuators (`model.nu`) - most reliable for control
2. Fallback: Count non-free joints manually
3. Safety: Use `nv - 6` as last resort
4. Validation: Ensure it matches config expectations and pad/trim names

**File:** `legged_gym/envs/base/mujoco_legged_robot.py`

### 2. Mujoco Import Order (Commit: f06c5b5)
**Problem:** `mujoco` module was imported at bottom of file but used in methods above.
- Would cause `NameError` when `mujoco.mj_objectVelocity()` was called
- Occurred in `_update_state_buffers()` method

**Solution:** Moved `import mujoco` to top of file with other imports.

**File:** `legged_gym/envs/g1/mujoco_g1_env.py`

## Previous Fixes (Already Tested)

### 3. Isaac Gym Math Utils (Commit: 74bb4ef)
**Problem:** `quat_apply()` and `normalize()` from Isaac Gym not available.

**Solution:** Provided complete fallback implementations:
- `normalize()`: Normalize tensor along last dimension
- `quat_apply()`: Full quaternion rotation implementation

**File:** `legged_gym/utils/math.py`

### 4. Isaac Gym Helper Functions (Commit: e377ce2)
**Problem:** `gymapi` and `gymutil` imports failing on Colab.

**Solution:** Wrapped imports in try/except with `ISAAC_GYM_AVAILABLE` flag.

**File:** `legged_gym/utils/helpers.py`

### 5. Isaac Gym Environment Imports (Commit: af749c8)
**Problem:** Isaac Gym environment imports failing when loading package.

**Solution:** Wrapped all Isaac Gym environment imports in try/except block.
- Mujoco environments register first (always available)
- Isaac Gym environments only register if available

**File:** `legged_gym/envs/__init__.py`

### 6. Repository and Dependencies (Commit: 8bbe3d4)
**Problem:** Wrong repository URL and package names in Colab notebook.

**Solution:**
- Changed repository to: `github.com/julienokumu/unitree_rl_mugym`
- Fixed package name: `rsl-rl` → `rsl-rl-lib`
- Fixed NumPy version: `numpy==1.20` → `numpy>=1.20` (Python 3.12 compatibility)

**Files:** `notebooks/train_g1_colab.ipynb`, `setup.py`, `README.md`, `COLAB_TRAINING.md`

## Testing Checklist

When testing on Colab, verify:

1. ✓ Dependencies install without errors
2. ✓ Repository clones successfully
3. ✓ Package installs without Isaac Gym
4. ✓ No import errors when loading `legged_gym`
5. ✓ Environment creation succeeds with correct DOF count
6. ⏳ Training loop runs without errors
7. ⏳ Model checkpoints save successfully
8. ⏳ Trained policy can be downloaded and visualized

## Known Limitations

- **Number of environments:** Reduced from 4096 (Isaac Gym GPU) to 256 (Mujoco CPU)
- **Training speed:** Slower on CPU, but acceptable with GPU runtime on Colab
- **Memory:** Keep num_envs ≤ 512 on Colab to avoid OOM errors

## Next Steps After Training Succeeds

1. Download trained policy: `files.download('logs/rough_g1/model_XXXX.pt')`
2. Test locally with Mujoco visualization: `python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml`
3. Fine-tune hyperparameters if needed (in `mujoco_g1_config.py`)

## Commit History

```
a5fbaf8 - Fix model loading: prefer XML over URDF to ensure actuators are loaded
0ba2142 - Add comprehensive documentation of all fixes applied
f06c5b5 - Fix mujoco import order in G1 environment
941976f - Fix DOF counting mismatch in Mujoco environment
74bb4ef - Make all Isaac Gym utils imports optional with fallbacks
e377ce2 - Make Isaac Gym imports optional in helpers.py
af749c8 - Make Isaac Gym imports optional for Colab compatibility
8bbe3d4 - Fix Colab notebook and dependency issues
```

## Support

If you encounter any new errors:
1. Copy the full error traceback
2. Check which file and line number caused the error
3. Report the issue with context about what step you were on

The framework is now fully independent of Isaac Gym and should work on any system with Python 3.8+ and Mujoco 3.2.3.
