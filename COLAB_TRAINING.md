# Training on Google Colab (No Local GPU Required!)

This guide explains how to train locomotion policies on **Google Colab** and visualize them **locally with Mujoco**.

## Why Colab + Mujoco?

- ✅ **No local GPU needed** - Train on Colab's free GPUs
- ✅ **No Isaac Gym dependency** - Works with Mujoco (CPU-friendly)
- ✅ **Colab-compatible** - Pure Python, runs on any platform
- ✅ **Local visualization** - Render policies on your laptop with Mujoco

## Workflow Overview

```
┌─────────────────────────────────────────────────────────┐
│  GOOGLE COLAB (Training)                                │
│  ┌────────────────────────────────────────────────┐    │
│  │  1. Train policy with Mujoco physics           │    │
│  │  2. Export model to TorchScript                │    │
│  │  3. Download trained model                     │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                        ↓ Download
┌─────────────────────────────────────────────────────────┐
│  LOCAL MACHINE (Visualization)                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  1. Load trained model                         │    │
│  │  2. Visualize in Mujoco viewer                 │    │
│  │  3. Test policy performance                    │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## Part 1: Training on Colab

### Option A: Using the Jupyter Notebook (Recommended)

1. **Upload notebook to Colab**
   ```
   Open: https://colab.research.google.com/
   Upload: notebooks/train_g1_colab.ipynb
   ```

2. **Run all cells**
   - The notebook will guide you through the entire process
   - Training takes ~2-6 hours depending on hardware
   - Models are automatically downloaded at the end

3. **Monitor progress**
   - TensorBoard is available in the notebook
   - Check rewards, episode length, policy loss

### Option B: Using the Command Line

If you prefer command-line training in Colab:

```bash
# 1. Clone repository
!git clone https://github.com/unitreerobotics/unitree_rl_gym.git
%cd unitree_rl_gym

# 2. Install dependencies (Mujoco only, no Isaac Gym!)
!pip install mujoco==3.2.3 scipy pyyaml tensorboard rsl-rl
!pip install -e . --no-deps

# 3. Train policy
!python legged_gym/scripts/train_mujoco.py \
    --task g1_mujoco \
    --num_envs 256 \
    --max_iterations 10000 \
    --device cuda \
    --experiment_name my_g1_training

# 4. Download trained models
from google.colab import files
import glob

# Find latest model
models = sorted(glob.glob('logs/my_g1_training/*/model_*.pt'))
latest_model = models[-1]

# Download
files.download(latest_model)
```

---

## Part 2: Visualization Locally

### Prerequisites

Install Mujoco on your local machine:

```bash
pip install mujoco==3.2.3
pip install torch pyyaml
```

### Steps

1. **Extract downloaded model**
   ```bash
   # The model should be named something like: model_10000.pt
   # Place it in: deploy/pre_train/g1/my_trained_model.pt
   ```

2. **Update config file**

   Edit `deploy/deploy_mujoco/configs/g1.yaml`:
   ```yaml
   policy_path: "{LEGGED_GYM_ROOT_DIR}/deploy/pre_train/g1/my_trained_model.pt"
   # ... rest of config
   ```

3. **Run visualization**
   ```bash
   python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
   ```

   **Or with custom policy path:**
   ```bash
   python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml \
       --policy /path/to/your/model.pt \
       --duration 60
   ```

4. **Watch your robot walk!**
   - The Mujoco viewer will open
   - Robot should follow velocity commands
   - Statistics are printed every 2 seconds

---

## Training Configuration

### Key Parameters

Located in `legged_gym/envs/g1/mujoco_g1_config.py`:

```python
class env:
    num_envs = 256          # Parallel environments (CPU: 128-256, GPU: 512-1024)
    num_observations = 47   # Observation dim
    num_actions = 12        # Joint actions
    episode_length_s = 20   # Episode duration

class rewards:
    class scales:
        tracking_lin_vel = 1.0      # Follow velocity commands
        tracking_ang_vel = 0.5      # Follow rotation commands
        orientation = -1.0          # Stay upright
        base_height = -10.0         # Maintain height
        torques = -0.00001          # Minimize effort
        action_rate = -0.01         # Smooth actions
        # ... G1-specific rewards
```

### Tuning Tips

**Faster Training:**
- Increase `num_envs` (requires more RAM/VRAM)
- Use GPU runtime in Colab
- Reduce `max_iterations` for testing

**Better Policies:**
- Train for 10,000-15,000 iterations
- Adjust reward scales (e.g., increase `tracking_lin_vel`)
- Enable domain randomization

**Debugging:**
- Start with `num_envs=64` and `max_iterations=1000`
- Check TensorBoard for reward trends
- Verify robot doesn't terminate immediately

---

## Command-Line Options

### Training Script

```bash
python legged_gym/scripts/train_mujoco.py [OPTIONS]
```

**Options:**
- `--task g1_mujoco` - Task name (currently only G1 supported)
- `--num_envs 256` - Number of parallel environments
- `--max_iterations 10000` - Training iterations
- `--device cpu/cuda` - Compute device
- `--experiment_name NAME` - Experiment name for logging
- `--run_name NAME` - Run name for this training session
- `--resume` - Resume from checkpoint
- `--load_run DIR` - Specify run directory to resume from
- `--checkpoint N` - Checkpoint number to load (-1 for latest)

**Examples:**
```bash
# Quick test (5 minutes)
python legged_gym/scripts/train_mujoco.py \
    --num_envs 64 \
    --max_iterations 1000 \
    --device cpu

# Full training on GPU
python legged_gym/scripts/train_mujoco.py \
    --num_envs 512 \
    --max_iterations 15000 \
    --device cuda \
    --experiment_name g1_final

# Resume training
python legged_gym/scripts/train_mujoco.py \
    --resume \
    --load_run Feb15_10-30-45_run_001 \
    --checkpoint 5000
```

### Visualization Script

```bash
python deploy/deploy_mujoco/deploy_mujoco.py CONFIG_FILE [OPTIONS]
```

**Options:**
- `--policy PATH` - Override policy path from config
- `--duration SECONDS` - Override simulation duration

**Examples:**
```bash
# Use pre-trained model
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml

# Use custom policy
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml \
    --policy logs/g1_mujoco/Feb15_10-30-45/model_10000.pt

# Run for 120 seconds
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml --duration 120
```

---

## Observation Space

**Dimensions: 47**

| Index | Content | Dim | Notes |
|-------|---------|-----|-------|
| 0-2 | Angular velocity | 3 | Body rotation rates (scaled) |
| 3-5 | Projected gravity | 3 | Gravity in body frame |
| 6-8 | Commands | 3 | vx, vy, vyaw targets |
| 9-20 | DOF positions | 12 | Relative to default angles |
| 21-32 | DOF velocities | 12 | Joint angular velocities |
| 33-44 | Previous actions | 12 | For temporal context |
| 45-46 | Phase | 2 | sin/cos of gait phase |

**Privileged observations (50 dims):** Same as above but includes linear velocity (3 dims at start)

---

## Action Space

**Dimensions: 12 (joint position targets)**

Actions are scaled and added to default joint angles:
```
target_angle = action * action_scale + default_angle
```

**Joint order:**
1. left_hip_yaw
2. left_hip_roll
3. left_hip_pitch
4. left_knee
5. left_ankle_pitch
6. left_ankle_roll
7. right_hip_yaw
8. right_hip_roll
9. right_hip_pitch
10. right_knee
11. right_ankle_pitch
12. right_ankle_roll

---

## Reward Functions

**Base Rewards (from LeggedRobot):**
- `tracking_lin_vel`: Follow linear velocity commands (exponential)
- `tracking_ang_vel`: Follow angular velocity commands (exponential)
- `orientation`: Penalize tilt (squared gravity projection)
- `base_height`: Maintain target height (squared error)
- `torques`: Minimize motor effort (squared)
- `dof_vel`: Penalize high joint velocities
- `dof_acc`: Penalize high joint accelerations
- `action_rate`: Encourage smooth actions
- `dof_pos_limits`: Penalize approaching joint limits

**G1-Specific Rewards:**
- `contact`: Reward phase-aligned foot contacts (stance/swing)
- `feet_swing_height`: Target 8cm swing height
- `hip_pos`: Penalize hip roll/pitch away from zero
- `contact_no_vel`: Penalize foot velocity during stance
- `alive`: Reward for staying upright

---

## Troubleshooting

### Colab Issues

**Out of Memory:**
```python
# Reduce number of environments
NUM_ENVS = 128  # or even 64
```

**Training Too Slow:**
```python
# Switch to GPU runtime
# Runtime → Change runtime type → GPU (T4)
NUM_ENVS = 512  # Can increase with GPU
```

**Import Errors:**
```bash
# Restart runtime and re-run all cells
# Runtime → Restart runtime
```

### Local Visualization Issues

**Policy Not Loading:**
```bash
# Check policy path is correct
ls deploy/pre_train/g1/my_trained_model.pt

# Try absolute path
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml \
    --policy /absolute/path/to/model.pt
```

**Robot Falls Immediately:**
- Policy may not be trained enough (try later checkpoint)
- Check observation/action dimensions match config
- Verify PD gains in config file

**Mujoco Viewer Won't Open:**
```bash
# Install correct Mujoco version
pip install mujoco==3.2.3

# On Linux, may need:
sudo apt-get install libglfw3 libgl1-mesa-glx libosmesa6
```

### Performance Issues

**Slow Training:**
- Use GPU in Colab (free T4 GPU available)
- Increase `num_envs` to parallelize
- Reduce `max_iterations` for testing

**Policy Not Learning:**
- Check TensorBoard - rewards should increase
- Verify robot survives >1 second (check termination rate)
- Adjust reward scales if needed
- Enable domain randomization for robustness

---

## File Structure

```
unitree_rl_gym/
├── legged_gym/
│   ├── envs/
│   │   ├── base/
│   │   │   ├── mujoco_legged_robot.py    # NEW: Mujoco base class
│   │   │   └── legged_robot.py           # Original Isaac Gym
│   │   └── g1/
│   │       ├── mujoco_g1_env.py          # NEW: Mujoco G1 environment
│   │       ├── mujoco_g1_config.py       # NEW: Mujoco G1 config
│   │       ├── g1_env.py                 # Original G1 (Isaac Gym)
│   │       └── g1_config.py              # Original config
│   └── scripts/
│       ├── train_mujoco.py               # NEW: Mujoco training script
│       ├── train.py                      # Original (Isaac Gym)
│       └── play.py                       # Original (Isaac Gym)
├── deploy/
│   ├── deploy_mujoco/
│   │   ├── deploy_mujoco.py              # ENHANCED: Better viz
│   │   └── configs/
│   │       └── g1.yaml                   # Mujoco deployment config
│   └── pre_train/
│       └── g1/
│           └── motion.pt                 # Pre-trained model
├── notebooks/
│   └── train_g1_colab.ipynb              # NEW: Colab notebook
└── COLAB_TRAINING.md                      # NEW: This file
```

---

## Comparison: Isaac Gym vs Mujoco

| Feature | Isaac Gym | Mujoco |
|---------|-----------|--------|
| **GPU Required** | Yes (local) | No |
| **Colab Support** | ❌ No | ✅ Yes |
| **Parallel Envs** | 4096 | 256-512 |
| **Training Speed** | Very Fast | Moderate |
| **Installation** | Complex | Simple (pip) |
| **Platform** | Linux only | Cross-platform |
| **Physics** | PhysX (GPU) | Mujoco (CPU/GPU) |
| **Maintenance** | Legacy (superseded) | Active |

**Recommendation:** Use **Mujoco** for Colab training, **Isaac Gym** for maximum speed (if you have local GPU).

---

## Advanced Usage

### Custom Environments

To create a Mujoco environment for other robots:

1. **Create environment class:**
   ```python
   # legged_gym/envs/h1/mujoco_h1_env.py
   from legged_gym.envs.base.mujoco_legged_robot import MujocoLeggedRobot

   class MujocoH1Robot(MujocoLeggedRobot):
       # Override compute_observations(), reward functions, etc.
       pass
   ```

2. **Create config:**
   ```python
   # legged_gym/envs/h1/mujoco_h1_config.py
   from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO

   class MujocoH1RoughCfg(H1RoughCfg):
       class env(H1RoughCfg.env):
           num_envs = 256
   ```

3. **Register task:**
   ```python
   # legged_gym/envs/__init__.py
   from legged_gym.envs.h1.mujoco_h1_env import MujocoH1Robot
   from legged_gym.envs.h1.mujoco_h1_config import MujocoH1RoughCfg, MujocoH1RoughCfgPPO

   task_registry.register("h1_mujoco", MujocoH1Robot, MujocoH1RoughCfg(), MujocoH1RoughCfgPPO())
   ```

### Hyperparameter Tuning

Key hyperparameters in `*CfgPPO` classes:

```python
class algorithm:
    entropy_coef = 0.01          # Exploration vs exploitation
    learning_rate = 1e-3         # Step size
    num_learning_epochs = 5      # PPO epochs per iteration
    clip_param = 0.2             # PPO clipping

class policy:
    actor_hidden_dims = [32]     # Network size (can try [128, 64])
    rnn_type = 'lstm'            # Use LSTM for temporal info
    rnn_hidden_size = 64         # LSTM hidden size
```

### Monitoring Training

**TensorBoard metrics:**
- `Reward/mean_reward` - Should increase over time
- `Loss/value_function` - Should decrease
- `Loss/policy` - Should stabilize
- `Episode/mean_length` - Should increase (robot survives longer)

```bash
# View locally after downloading logs
tensorboard --logdir logs/g1_mujoco/
```

---

## FAQ

**Q: Can I use Colab Pro for faster training?**
A: Yes! Colab Pro provides better GPUs (A100, V100) and longer runtimes. Training can be 2-3x faster.

**Q: Can I train other robots (H1, Go2)?**
A: Currently only G1 has Mujoco support. You can add others by following the "Custom Environments" guide above.

**Q: How long does training take?**
A:
- CPU (256 envs): 6-10 hours for 10k iterations
- GPU T4 (512 envs): 2-4 hours for 10k iterations
- GPU A100 (1024 envs): 1-2 hours for 10k iterations

**Q: Can I deploy to the real robot?**
A: Yes! The trained policy is compatible with real robot deployment. See `deploy/deploy_real/` for instructions.

**Q: Why not train locally?**
A: You can! If you have a GPU, Isaac Gym training is faster. But Colab + Mujoco works great without local hardware.

**Q: Can I modify the robot URDF?**
A: Yes, edit `resources/robots/g1_description/g1_12dof.urdf` and retrain. Make sure to update config accordingly.

---

## Next Steps

1. **Try the Colab notebook** - Easiest way to start
2. **Monitor training** - Check TensorBoard for progress
3. **Tune rewards** - Adjust config for better behaviors
4. **Deploy to robot** - Test on real hardware (if available)
5. **Share results** - Contribute back to the community!

---

## Support

- **Issues:** https://github.com/unitreerobotics/unitree_rl_gym/issues
- **Discussions:** GitHub Discussions
- **Docs:** Check README.md for original workflow

**Happy Training! 🤖**
