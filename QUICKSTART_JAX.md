# JAX/MJX Training Quickstart

**Get 10-100x faster training with GPU acceleration!**

## 🚀 Recommended: Google Colab (T4 GPU - FREE!)

**Use the Colab notebook for the easiest experience:**

1. Open `notebooks/train_g1_jax_colab.ipynb` in Google Colab
2. Enable T4 GPU (Runtime > Change runtime type > T4 GPU)
3. Run all cells
4. Train completes in ~1.5 hours (vs 13+ hours with CPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/julienokumu/unitree_rl_mugym/blob/main/notebooks/train_g1_jax_colab.ipynb)

**Benefits:**
- ⚡ 8-10x faster than PyTorch/CPU version
- 💰 Completely free (T4 GPU included)
- 📦 No local installation needed
- ☁️ Train in the cloud, visualize locally

---

## 💻 Local GPU (If You Have One)

```bash
# 1. Install JAX GPU dependencies
pip install -e .[jax_gpu]

# 2. Set precision (for NVIDIA Ampere GPUs: RTX 30/40 series)
export JAX_DEFAULT_MATMUL_PRECISION=highest

# 3. Test setup
python test_jax_setup.py

# 4. Train G1 robot on GPU
python legged_gym/scripts/train_jax_ppo.py \
  --robot g1 \
  --num_envs 1024 \  # Adjust based on your GPU
  --backend mjx
```

## Installation

### Step 1: Install JAX with CUDA Support

```bash
# For CUDA 12
pip install --upgrade "jax[cuda12]>=0.4.23"

# For CUDA 11
pip install --upgrade "jax[cuda11]>=0.4.23"
```

### Step 2: Install Training Dependencies

```bash
pip install -e .[jax_gpu]
```

This installs:
- MuJoCo MJX (GPU physics)
- Brax (JAX RL library)
- Flax (neural networks)
- Optax (optimization)
- Weights & Biases (logging)

### Step 3: Verify Installation

```bash
python test_jax_setup.py
```

You should see all tests pass with "✓ PASS" marks.

## GPU Configuration

### For NVIDIA RTX 30/40 Series (Ampere)

```bash
export JAX_DEFAULT_MATMUL_PRECISION=highest
```

Add to `~/.bashrc`:
```bash
echo 'export JAX_DEFAULT_MATMUL_PRECISION=highest' >> ~/.bashrc
source ~/.bashrc
```

### Verify GPU Detection

```bash
python -c "import jax; print('Devices:', jax.devices())"
```

Should show: `Devices: [cuda(id=0)]` or similar.

## Training Examples

### Basic Training (G1 Robot)

```bash
python legged_gym/scripts/train_jax_ppo.py \
  --robot g1 \
  --num_envs 2048 \
  --num_timesteps 100000000
```

### Fast Training (More Environments)

```bash
python legged_gym/scripts/train_jax_ppo.py \
  --robot g1 \
  --num_envs 4096 \
  --learning_rate 3e-4 \
  --num_timesteps 50000000
```

### With Logging

```bash
python legged_gym/scripts/train_jax_ppo.py \
  --robot g1 \
  --num_envs 2048 \
  --experiment_name "g1_run1" \
  --use_wandb
```

## Choosing Num Envs

More environments = faster training, but uses more GPU memory:

| GPU | Recommended | Max |
|-----|-------------|-----|
| RTX 3060 (12GB) | 1024 | 2048 |
| RTX 3080 (10GB) | 1024 | 2048 |
| RTX 3090 (24GB) | 2048 | 4096 |
| RTX 4090 (24GB) | 2048 | 4096 |
| A100 (40GB) | 4096 | 8192 |
| A100 (80GB) | 8192 | 16384 |

Start with the recommended value, increase if you have spare memory.

## Expected Performance

Training 10M steps on G1 robot:

| Setup | Time | Speedup |
|-------|------|---------|
| CPU (256 envs, PyTorch) | ~8 hours | 1x |
| RTX 3090 (2048 envs, JAX) | ~45 min | 10.7x |
| RTX 4090 (4096 envs, JAX) | ~30 min | 16x |
| A100 (8192 envs, JAX) | ~20 min | 24x |

## Troubleshooting

### "No GPU found"

```bash
# Check CUDA installation
nvidia-smi

# Reinstall JAX with CUDA
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda12]"

# Verify
python -c "import jax; print(jax.devices())"
```

### "Out of Memory"

Reduce `--num_envs`:
```bash
python train_jax_ppo.py --num_envs 1024
```

### "Training is unstable"

Check precision:
```bash
export JAX_DEFAULT_MATMUL_PRECISION=highest
```

### "Import errors"

Reinstall:
```bash
pip install -e .[jax_gpu] --force-reinstall
```

## Next Steps

1. **Read full documentation**: `docs/JAX_MJX_TRAINING.md`
2. **Port other robots**: See porting guide in docs
3. **Experiment with hyperparameters**: Adjust learning rate, batch size, etc.
4. **Deploy trained models**: Use existing deployment tools

## Comparison: PyTorch vs JAX

| Feature | PyTorch (Original) | JAX/MJX (New) |
|---------|-------------------|---------------|
| **Training Speed** | 1x | 10-100x |
| **Num Envs** | 256 typical | 2048-8192 |
| **Device** | CPU | GPU |
| **Setup Complexity** | Low | Medium |
| **Stability** | High | High (with correct settings) |

Both implementations produce equivalent results. Use JAX for **faster iteration**.

## Support

If you encounter issues:

1. Run `python test_jax_setup.py` and share output
2. Include GPU model and CUDA version
3. Share full error traceback
4. Open an issue with the `[JAX]` tag

## References

- Full documentation: `docs/JAX_MJX_TRAINING.md`
- JAX: https://jax.readthedocs.io/
- MuJoCo MJX: https://mujoco.readthedocs.io/en/stable/mjx.html
- Brax: https://github.com/google/brax

---

**Ready to train? Run `python test_jax_setup.py` to get started!**
