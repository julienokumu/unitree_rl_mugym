"""
Convert rsl_rl checkpoint to TorchScript for deployment
Usage: python convert_checkpoint_to_jit.py <checkpoint_path> <output_path>
"""

import sys
import os
import torch
import copy

# Add repo to path
sys.path.insert(0, os.path.dirname(__file__))

from legged_gym.envs.g1.mujoco_g1_config import MujocoG1RoughCfg, MujocoG1RoughCfgPPO
from rsl_rl.modules import ActorCriticRecurrent


class PolicyExporterLSTM(torch.nn.Module):
    """Export LSTM policy to TorchScript"""
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
        print(f"✓ Exported policy to: {path}")


def convert_checkpoint(checkpoint_path, output_path):
    """Convert rsl_rl checkpoint to TorchScript"""

    print(f"\n{'='*60}")
    print("CHECKPOINT TO TORCHSCRIPT CONVERTER")
    print(f"{'='*60}")
    print(f"Input:  {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✓ Checkpoint loaded (iteration {checkpoint.get('iter', 'unknown')})")

    # Get config
    env_cfg = MujocoG1RoughCfg()
    train_cfg = MujocoG1RoughCfgPPO()

    # Create policy network
    print("\nCreating policy network...")
    num_obs = env_cfg.env.num_observations
    num_critic_obs = env_cfg.env.num_privileged_obs or num_obs
    num_actions = env_cfg.env.num_actions

    # Create dummy observation dict for initialization (rsl_rl requirement)
    dummy_obs = {
        'policy': torch.zeros(1, num_obs),
        'critic': torch.zeros(1, num_critic_obs)
    }

    obs_groups = {
        'policy': ['policy'],
        'critic': ['critic']
    }

    actor_critic = ActorCriticRecurrent(
        obs=dummy_obs,
        num_actor_obs=num_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        obs_groups=obs_groups,
        actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
        activation=train_cfg.policy.activation,
        rnn_type=train_cfg.policy.rnn_type,
        rnn_hidden_size=train_cfg.policy.rnn_hidden_size,
        rnn_num_layers=train_cfg.policy.rnn_num_layers,
    )
    print("✓ Policy network created")

    # Load weights
    print("\nLoading weights from checkpoint...")
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    actor_critic.eval()
    print("✓ Weights loaded successfully")

    # Export to TorchScript
    print("\nExporting to TorchScript...")
    exporter = PolicyExporterLSTM(actor_critic)
    exporter.export(output_path)

    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nYou can now use the exported policy:")
    print(f"  python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml \\")
    print(f"      --policy {os.path.join(output_path, 'policy_1.pt')}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_checkpoint_to_jit.py <checkpoint_path> <output_path>")
        print("\nExample:")
        print("  python convert_checkpoint_to_jit.py \\")
        print("      logs/g1_colab_training/run_001/model_49.pt \\")
        print("      logs/g1_colab_training/run_001/exported")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    convert_checkpoint(checkpoint_path, output_path)
