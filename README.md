# Unitree RL MuGym

**Mujoco-based Reinforcement Learning for Unitree Robots - Train on Google Colab, Visualize Locally**

This is a modified version of [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym) that replaces Isaac Gym with Mujoco for training. The key advantage: **train policies on Google Colab without requiring a local GPU**, then visualize them locally with Mujoco.

---

## 🎯 Key Features

- ✅ **No Local GPU Required** - Train on Google Colab's free GPUs
- ✅ **No Isaac Gym Dependency** - Uses Mujoco physics (CPU-friendly)
- ✅ **Colab-Compatible** - Works in Jupyter notebooks with free tier
- ✅ **Local Visualization** - Render policies on any machine with Mujoco
- ✅ **Same Algorithm** - Uses proven RSL-RL PPO implementation
- ✅ **Easy Setup** - Single `pip install` for dependencies

---

## 📖 Background

### Original Framework: Unitree RL Gym

This repository is based on [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym), an excellent framework for training locomotion policies on Unitree robots using Isaac Gym.

**Original Features:**
- Isaac Gym-based training (4096 parallel environments on GPU)
- Supports Go2, H1, H1_2, G1 robots
- Sim2Real deployment
- Pre-trained models

**Limitations:**
- Requires local GPU with Isaac Gym
- Isaac Gym doesn't work on Google Colab
- Isaac Gym is legacy (superseded by Isaac Lab)

### Our Modification: Unitree RL MuGym

We've extended the framework with **Mujoco-based training environments** that:
- Run on Google Colab (no local GPU needed)
- Use Mujoco physics instead of Isaac Gym
- Support CPU training (slower but accessible)
- Maintain compatibility with the original deployment pipeline

**What's New:**
- `MujocoLeggedRobot` - Base Mujoco environment class
- `MujocoG1Robot` - G1-specific Mujoco implementation
- `train_mujoco.py` - Colab-compatible training script
- `train_g1_colab.ipynb` - Complete Colab notebook
- `COLAB_TRAINING.md` - Comprehensive guide

**What's Preserved:**
- Original Isaac Gym environments (still work if you have local GPU)
- Deployment scripts (Sim2Sim, Sim2Real)
- Configuration system
- Reward functions
- Pre-trained models

---

## 🚀 Quick Start

### Option 1: Train on Google Colab (Recommended)

1. **Open Colab Notebook**
   - Upload `notebooks/train_g1_colab.ipynb` to [Google Colab](https://colab.research.google.com/)
   - Or open directly: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/julienokumu/unitree_rl_mugym/blob/main/notebooks/train_g1_colab.ipynb)

2. **Run All Cells**
   - Training takes 2-6 hours depending on hardware
   - TensorBoard available for monitoring
   - Model automatically downloaded at end

3. **Visualize Locally**
   ```bash
   # Install on local machine
   pip install mujoco==3.2.3 torch pyyaml

   # Run visualization
   python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml \
       --policy /path/to/downloaded/model.pt
   ```

### Option 2: Train Locally (If You Have CPU/GPU)

```bash
# Clone repository
git clone https://github.com/julienokumu/unitree_rl_mugym.git
cd unitree_rl_mugym

# Install dependencies (Mujoco-only, no Isaac Gym)
pip install mujoco==3.2.3 scipy pyyaml tensorboard rsl-rl-lib torch
pip install -e .

# Train policy
python legged_gym/scripts/train_mujoco.py \
    --task g1_mujoco \
    --num_envs 256 \
    --max_iterations 10000 \
    --device cpu  # or 'cuda' if you have GPU

# Visualize
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

---

## 📚 Documentation

- **[COLAB_TRAINING.md](COLAB_TRAINING.md)** - Complete guide for Colab training workflow
- **[README_ORIGINAL.md](README_ORIGINAL.md)** - Original Unitree RL Gym documentation
- **[Notebooks](notebooks/)** - Jupyter notebooks for training

---

## 🤖 Supported Robots

| Robot | Mujoco Support | Isaac Gym Support | DOF | Type |
|-------|---------------|-------------------|-----|------|
| **G1** | ✅ Yes | ✅ Yes | 12 | Humanoid |
| **H1** | 🚧 Coming Soon | ✅ Yes | 12 | Humanoid |
| **H1_2** | 🚧 Coming Soon | ✅ Yes | 12 | Humanoid |
| **Go2** | 🚧 Coming Soon | ✅ Yes | 12 | Quadruped |

Currently, only **G1** has Mujoco support. Other robots can be added by following the implementation pattern.

---

## 📦 Installation

### For Colab Training (Minimal)

```bash
# In Colab notebook
!pip install mujoco==3.2.3 scipy pyyaml tensorboard rsl-rl
!git clone https://github.com/julienokumu/unitree_rl_mugym.git
!pip install -e unitree_rl_mugym --no-deps
```

### For Local Use (Full)

```bash
# Clone repository
git clone https://github.com/julienokumu/unitree_rl_mugym.git
cd unitree_rl_mugym

# Install Mujoco-only dependencies
pip install mujoco==3.2.3 scipy pyyaml tensorboard rsl-rl-lib torch matplotlib numpy==1.20
pip install -e .

# Or install everything (including Isaac Gym if you have it)
pip install -e .
```

### For Local Visualization Only

```bash
pip install mujoco==3.2.3 torch pyyaml numpy
```

---

## 🎓 How It Works

### Training Flow (Mujoco)

```
┌─────────────────────────────────────────┐
│  Google Colab / Local Machine           │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ MujocoG1Robot (256-512 envs)   │   │
│  │  - Mujoco physics simulation    │   │
│  │  - Vectorized CPU/GPU           │   │
│  │  - Same rewards as Isaac Gym   │   │
│  └──────────────┬──────────────────┘   │
│                 ↓                       │
│  ┌─────────────────────────────────┐   │
│  │ RSL-RL PPO Algorithm            │   │
│  │  - Policy network (LSTM)        │   │
│  │  - Value network                │   │
│  │  - 10k iterations (~2-6 hours) │   │
│  └──────────────┬──────────────────┘   │
│                 ↓                       │
│  ┌─────────────────────────────────┐   │
│  │ Trained Policy (TorchScript)    │   │
│  │  - model_10000.pt               │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
              ↓ Download
┌─────────────────────────────────────────┐
│  Local Machine (Visualization)          │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Mujoco Viewer                   │   │
│  │  - Load trained policy          │   │
│  │  - Render robot walking         │   │
│  │  - Real-time visualization      │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

## 📊 Performance Comparison

| Setup | Environments | Time (10k iter) | Hardware |
|-------|-------------|-----------------|----------|
| **Isaac Gym** | 4096 | 2-4 hours | Local GPU (required) |
| **Mujoco (CPU)** | 256 | 6-10 hours | Any CPU (Colab free) |
| **Mujoco (T4 GPU)** | 512 | 2-4 hours | Colab free GPU |
| **Mujoco (A100 GPU)** | 1024 | 1-2 hours | Colab Pro |

**Trade-off:** Mujoco is slower but accessible anywhere without local GPU.

---

## 🎮 Usage Examples

### Training

```bash
# Quick test (5 minutes)
python legged_gym/scripts/train_mujoco.py \
    --num_envs 64 \
    --max_iterations 100 \
    --device cpu

# Full training on CPU
python legged_gym/scripts/train_mujoco.py \
    --task g1_mujoco \
    --num_envs 256 \
    --max_iterations 10000 \
    --device cpu \
    --experiment_name my_g1_training

# Full training on GPU (faster)
python legged_gym/scripts/train_mujoco.py \
    --task g1_mujoco \
    --num_envs 512 \
    --max_iterations 10000 \
    --device cuda \
    --experiment_name my_g1_training

# Resume from checkpoint
python legged_gym/scripts/train_mujoco.py \
    --resume \
    --load_run Feb15_10-30-45_my_run \
    --checkpoint 5000
```

### Visualization

```bash
# Use pre-trained model
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml

# Use custom trained model
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml \
    --policy logs/g1_mujoco/Feb15_10-30-45/model_10000.pt

# Run for longer duration
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml \
    --policy my_model.pt \
    --duration 120
```

### Monitoring

```bash
# Launch TensorBoard
tensorboard --logdir logs/g1_mujoco/

# Monitor specific run
tensorboard --logdir logs/g1_mujoco/Feb15_10-30-45_my_run/
```

---

## 🛠️ Project Structure

```
unitree_rl_mugym/
├── legged_gym/
│   ├── envs/
│   │   ├── base/
│   │   │   ├── mujoco_legged_robot.py    # 🆕 Mujoco base class
│   │   │   └── legged_robot.py           # Original Isaac Gym
│   │   ├── g1/
│   │   │   ├── mujoco_g1_env.py          # 🆕 Mujoco G1
│   │   │   ├── mujoco_g1_config.py       # 🆕 Config
│   │   │   ├── g1_env.py                 # Original
│   │   │   └── g1_config.py              # Original
│   │   └── [h1, h1_2, go2]/              # Original robots
│   └── scripts/
│       ├── train_mujoco.py               # 🆕 Mujoco training
│       ├── train.py                      # Original Isaac Gym
│       └── play.py                       # Original
├── deploy/
│   ├── deploy_mujoco/
│   │   ├── deploy_mujoco.py              # ✨ Enhanced
│   │   └── configs/
│   │       └── g1.yaml
│   ├── deploy_real/                      # Original (works with both)
│   └── pre_train/                        # Pre-trained models
├── notebooks/
│   └── train_g1_colab.ipynb              # 🆕 Colab notebook
├── resources/                             # Robot URDFs and meshes
├── COLAB_TRAINING.md                      # 🆕 Complete guide
├── README.md                              # 🆕 This file
└── README_ORIGINAL.md                     # Original documentation
```

**Legend:**
- 🆕 New files added in this fork
- ✨ Enhanced/modified existing files
- No icon = Original files from unitree_rl_gym

---

## 🔍 Differences from Original

| Feature | Original (Isaac Gym) | This Fork (Mujoco) |
|---------|---------------------|-------------------|
| **Training Backend** | Isaac Gym (GPU only) | Mujoco (CPU/GPU) |
| **Colab Support** | ❌ No | ✅ Yes |
| **Local GPU Required** | ✅ Yes | ❌ No |
| **Parallel Envs** | 4096 | 256-1024 |
| **Training Speed** | Very Fast | Moderate |
| **Installation** | Complex | Simple (pip) |
| **Platform** | Linux only | Cross-platform |
| **Deployment** | ✅ Same | ✅ Same |
| **Pre-trained Models** | ✅ Compatible | ✅ Compatible |

**Both versions share:**
- Same PPO algorithm (rsl_rl)
- Same reward functions
- Same observation/action spaces
- Same deployment pipeline
- Compatible model formats

---

## 🐛 Troubleshooting

### Common Issues

**Out of Memory on Colab:**
```python
# Reduce number of environments
NUM_ENVS = 128  # or 64
```

**Policy Not Loading:**
```bash
# Check model file exists
ls -lh path/to/model.pt

# Use absolute path
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml \
    --policy /absolute/path/to/model.pt
```

**Robot Falls Immediately:**
- Policy may need more training iterations
- Check observation/action dimensions match
- Verify PD gains in config file

**Mujoco Import Error:**
```bash
# Reinstall Mujoco
pip uninstall mujoco
pip install mujoco==3.2.3

# On Linux, install system dependencies
sudo apt-get install libglfw3 libgl1-mesa-glx libosmesa6
```

See [COLAB_TRAINING.md](COLAB_TRAINING.md) for more troubleshooting tips.

---

## 🤝 Contributing

Contributions are welcome! Areas where we'd love help:

1. **Add Mujoco support for other robots** (H1, H1_2, Go2)
2. **Improve training efficiency** (better vectorization, GPU utilization)
3. **Add terrain support** (heightfield, curriculum learning)
4. **Enhance visualization** (keyboard controls, better camera)
5. **Documentation improvements**

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/awesome-feature`)
3. Commit your changes (`git commit -m 'Add awesome feature'`)
4. Push to the branch (`git push origin feature/awesome-feature`)
5. Open a Pull Request

---

## 📝 Citation

### This Repository (Unitree RL MuGym)

```bibtex
@software{unitree_rl_mugym2025,
  title={Unitree RL MuGym: Mujoco-based Training for Unitree Robots},
  author={Your Name},
  year={2025},
  url={https://github.com/julienokumu/unitree_rl_mugym},
  note={Extended from Unitree RL Gym with Mujoco support for Google Colab training}
}
```

### Original Framework (Unitree RL Gym)

```bibtex
@software{unitree_rl_gym2024,
  title={Unitree RL Gym},
  author={Unitree Robotics},
  year={2024},
  url={https://github.com/unitreerobotics/unitree_rl_gym}
}
```

### Dependencies

- **Mujoco**: [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- **RSL-RL**: [RSL-RL Repository](https://github.com/leggedrobotics/rsl_rl)
- **Isaac Gym** (original): [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)

---

## 🙏 Acknowledgments

- **Unitree Robotics** - For the original [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) framework
- **RSL Lab (ETH Zurich)** - For the [rsl_rl](https://github.com/leggedrobotics/rsl_rl) PPO implementation
- **DeepMind/Google** - For [Mujoco](https://mujoco.org/) physics engine
- **NVIDIA** - For Isaac Gym (used in original framework)

---

## 📄 License

This project maintains the same license as the original Unitree RL Gym: **BSD-3-Clause**

See [LICENSE](LICENSE) for details.

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/julienokumu/unitree_rl_mugym/issues)
- **Discussions**: [GitHub Discussions](https://github.com/julienokumu/unitree_rl_mugym/discussions)
- **Original Framework**: [Unitree RL Gym Issues](https://github.com/unitreerobotics/unitree_rl_gym/issues)

---

**Happy Training! 🤖**

Train anywhere, deploy everywhere.
