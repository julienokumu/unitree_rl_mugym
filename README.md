<div align="center">
  <h1 align="center">Unitree RL GYM</h1>
  <p align="center">
    <span> 🌎English </span> | <a href="README_zh.md"> 🇨🇳中文 </a>
  </p>
</div>

<p align="center">
  <strong>Reinforcement learning for Unitree robots (Go2, H1, H1_2, G1) with Isaac Gym and Mujoco support.</strong>
</p>

<p align="center">
  ✨ <strong>Now with Google Colab training support - train without a local GPU!</strong> ✨
</p>

<div align="center">

| <div align="center"> Isaac Gym </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/32f06dc9dfe4452dac300dda45e86b34.GIF" width="240px">](https://oss-global-cdn.unitree.com/static/5bbc5ab1d551407080ca9d58d7bec1c8.mp4) | [<img src="https://oss-global-cdn.unitree.com/static/244cd5c4f823495fbfb67ef08f56aa33.GIF" width="240px">](https://oss-global-cdn.unitree.com/static/5aa48535ffd641e2932c0ba45c8e7854.mp4) | [<img src="https://oss-global-cdn.unitree.com/static/78c61459d3ab41448cfdb31f6a537e8b.GIF" width="240px">](https://oss-global-cdn.unitree.com/static/0818dcf7a6874b92997354d628adcacd.mp4) |

</div>

---

## 🆕 What's New

This fork adds **Mujoco-based training for Google Colab**, enabling training without a local GPU:

- ✅ Train on Google Colab free tier (T4 GPU)
- ✅ No Isaac Gym installation required for training
- ✅ Visualize trained policies locally with Mujoco
- ✅ Resume training across multiple Colab sessions
- ✅ [Ready-to-use Colab notebook](notebooks/train_g1_mujoco_colab.ipynb)

---

## 📦 Installation

### Option 1: Google Colab Training (No Local GPU Required)

1. **Open the Colab Notebook**: [train_g1_mujoco_colab.ipynb](notebooks/train_g1_mujoco_colab.ipynb)
2. **Enable GPU**: `Runtime > Change runtime type > GPU (T4)`
3. **Run all cells** - training takes ~2 hours per 50 iterations
4. **Download trained policy** and visualize locally

**Local visualization only** (no training):
```bash
pip install mujoco==3.2.3 torch pyyaml numpy
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml --policy /path/to/downloaded/policy_1.pt
```

### Option 2: Local Training (Isaac Gym)

For full Isaac Gym setup, see [setup.md](/doc/setup_en.md).

---

## 🔁 Workflow

`Train` → `Play` → `Sim2Sim` → `Sim2Real`

- **Train**: Train policy in simulation (Isaac Gym or Mujoco)
- **Play**: Visualize and verify trained policy
- **Sim2Sim**: Test policy in different simulators
- **Sim2Real**: Deploy to physical robot

---

## 🛠️ Usage

### 1. Training

#### Mujoco Training (Colab/Local)
```bash
python legged_gym/scripts/train_mujoco.py --task=g1_mujoco
```

**Parameters:**
- `--device`: `cuda` or `cpu`
- `--num_envs`: Number of parallel environments (default: 512)
- `--max_iterations`: Training iterations (default: 1000)
- `--resume`: Resume from latest checkpoint

**Training tips:**
- 50-100 iterations: Basic coordination emerges (~2-4 hours on T4)
- 500 iterations: Stable walking policy (~20 hours, use resume)
- 1000+ iterations: Robust, efficient walking

#### Isaac Gym Training (Local Only)
```bash
python legged_gym/scripts/train.py --task=g1
```

**Parameters:**
- `--task`: Robot type (go2, g1, h1, h1_2)
- `--headless`: Run without GUI (faster)
- `--resume`: Resume training
- `--experiment_name`, `--run_name`: Organize experiments
- `--num_envs`: Parallel environments
- `--max_iterations`: Training iterations

**Models saved to**: `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

---

### 2. Play (Visualization in Gym)

Visualize training results:
```bash
python legged_gym/scripts/play.py --task=g1
```

**Exports policy** to `logs/{experiment_name}/exported/policies/policy_1.pt` for deployment.

---

### 3. Sim2Sim (Mujoco)

Test policy in Mujoco simulator:
```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

**Custom policy:**
```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml --policy /path/to/policy_1.pt
```

**Configuration:** Edit `deploy/deploy_mujoco/configs/g1.yaml` to customize policy path, control parameters, etc.

#### Mujoco Simulation Results

| G1 | H1 | H1_2 |
|--- | --- | --- |
| [![mujoco_g1](https://oss-global-cdn.unitree.com/static/244cd5c4f823495fbfb67ef08f56aa33.GIF)](https://oss-global-cdn.unitree.com/static/5aa48535ffd641e2932c0ba45c8e7854.mp4)  |  [![mujoco_h1](https://oss-global-cdn.unitree.com/static/7ab4e8392e794e01b975efa205ef491e.GIF)](https://oss-global-cdn.unitree.com/static/8934052becd84d08bc8c18c95849cf32.mp4)  |  [![mujoco_h1_2](https://oss-global-cdn.unitree.com/static/2905e2fe9b3340159d749d5e0bc95cc4.GIF)](https://oss-global-cdn.unitree.com/static/ee7ee85bd6d249989a905c55c7a9d305.mp4) |

---

### 4. Sim2Real (Physical Robot)

Deploy trained policy to physical robot (requires robot in debug mode):
```bash
python deploy/deploy_real/deploy_real.py {net_interface} {config_name}
```

**Parameters:**
- `net_interface`: Network interface name (e.g., `eth0`, `enp3s0`)
- `config_name`: Config file (e.g., `g1.yaml`, `h1.yaml`)

See [Physical Deployment Guide](deploy/deploy_real/README.md) for details.

#### Deployment Results

| G1 | H1 | H1_2 |
|--- | --- | --- |
| [![real_g1](https://oss-global-cdn.unitree.com/static/78c61459d3ab41448cfdb31f6a537e8b.GIF)](https://oss-global-cdn.unitree.com/static/0818dcf7a6874b92997354d628adcacd.mp4) | [![real_h1](https://oss-global-cdn.unitree.com/static/fa07b2fd2ad64bb08e6b624d39336245.GIF)](https://oss-global-cdn.unitree.com/static/ea0084038d384e3eaa73b961f33e6210.mp4) | [![real_h1_2](https://oss-global-cdn.unitree.com/static/a88915e3523546128a79520aa3e20979.GIF)](https://oss-global-cdn.unitree.com/static/12d041a7906e489fae79d55b091a63dd.mp4) |

---

## 🎉 Acknowledgments

Built upon these excellent open-source projects:

- [legged_gym](https://github.com/leggedrobotics/legged_gym): Training framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): RL algorithms
- [mujoco](https://github.com/google-deepmind/mujoco.git): Physics simulation
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python.git): Hardware interface
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym): Original repository

---

## 🔖 License

This project is licensed under the [BSD 3-Clause License](./LICENSE).

For details, please read the full [LICENSE file](./LICENSE).
