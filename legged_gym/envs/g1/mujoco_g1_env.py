"""
Mujoco-based G1 Robot Environment
Extends MujocoLeggedRobot with G1-specific features
"""

import numpy as np
import torch
import mujoco
from legged_gym.envs.base.mujoco_legged_robot import MujocoLeggedRobot


class MujocoG1Robot(MujocoLeggedRobot):
    """
    G1 humanoid robot environment using Mujoco physics.
    Includes phase-based gait control and G1-specific rewards.
    """

    def __init__(self, cfg, device='cpu'):
        super().__init__(cfg, device)

    def _init_buffers(self):
        """Initialize buffers including G1-specific ones"""
        super()._init_buffers()

        # G1-specific: foot state tracking
        self.feet_num = len(self.feet_indices)

        # Feet positions and velocities
        self.feet_pos = torch.zeros(self.num_envs, self.feet_num, 3, device=self.device)
        self.feet_vel = torch.zeros(self.num_envs, self.feet_num, 3, device=self.device)

        # Phase information for gait
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_left = torch.zeros(self.num_envs, device=self.device)
        self.phase_right = torch.zeros(self.num_envs, device=self.device)
        self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device)  # Left, right

        print(f"[MujocoG1Robot] Initialized with {self.feet_num} feet")

    def _update_state_buffers(self, env_ids=None):
        """Update state buffers including foot states"""
        super()._update_state_buffers(env_ids)

        # Update foot positions and velocities
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        for env_id in env_ids:
            mjd = self.mjdata_list[env_id.item()]

            # Get foot states from body positions
            for i, foot_idx in enumerate(self.feet_indices):
                self.feet_pos[env_id, i] = torch.from_numpy(mjd.xpos[foot_idx]).to(self.device)

                # Get foot velocity from body velocities
                # Mujoco stores velocities in qvel, but we need body velocities
                body_vel = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, mjd, mujoco.mjtObj.mjOBJ_BODY, foot_idx, body_vel, 0)
                self.feet_vel[env_id, i] = torch.from_numpy(body_vel[:3]).to(self.device)

    def _post_physics_step_callback(self):
        """Update phase and call parent callback"""
        # Update gait phase
        period = 0.8  # Gait period in seconds
        offset = 0.5  # Phase offset between legs

        self.phase = (self.episode_length_buf.float() * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1.0
        self.leg_phase = torch.stack([self.phase_left, self.phase_right], dim=1)

        # Call parent callback
        super()._post_physics_step_callback()

    def compute_observations(self):
        """
        Compute observations for G1 robot.

        Observation space (47 dims):
        - angular velocity (3)
        - projected gravity (3)
        - commands (3)
        - dof positions (12)
        - dof velocities (12)
        - previous actions (12)
        - phase sin/cos (2)
        """
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)

        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase
        ), dim=-1)

        # Privileged observations (includes linear velocity)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.cat((
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                sin_phase,
                cos_phase
            ), dim=-1)

    # ============ G1-Specific Reward Functions ============

    def _reward_contact(self):
        """
        Reward foot contact that matches gait phase.
        Stance phase (< 0.55): foot should be in contact
        Swing phase (>= 0.55): foot should not be in contact
        """
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55

            # Check if foot is in contact (force > 1N)
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0

            # Reward when contact matches expected phase
            res += ~(contact ^ is_stance)  # XOR then NOT = XNOR (match)

        return res.float()

    def _reward_feet_swing_height(self):
        """
        Penalize swing foot height away from target (8cm).
        Only penalize when foot is not in contact.
        """
        target_height = 0.08

        # Check contact for each foot
        contact = torch.zeros(self.num_envs, self.feet_num, dtype=torch.bool, device=self.device)
        for i in range(self.feet_num):
            contact[:, i] = torch.norm(self.contact_forces[:, self.feet_indices[i], :3], dim=1) > 1.0

        # Penalize height error when not in contact
        pos_error = torch.square(self.feet_pos[:, :, 2] - target_height) * (~contact)

        return torch.sum(pos_error, dim=1)

    def _reward_alive(self):
        """Reward for staying alive"""
        return torch.ones(self.num_envs, device=self.device)

    def _reward_contact_no_vel(self):
        """
        Penalize foot velocity when in contact (should be stationary).
        Encourages static stance phase.
        """
        # Check contact for each foot
        contact = torch.zeros(self.num_envs, self.feet_num, dtype=torch.bool, device=self.device)
        for i in range(self.feet_num):
            contact[:, i] = torch.norm(self.contact_forces[:, self.feet_indices[i], :3], dim=1) > 1.0

        # Get foot velocities when in contact
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)

        # Penalize velocity magnitude
        penalize = torch.square(contact_feet_vel[:, :, :3])

        return torch.sum(penalize, dim=(1, 2))

    def _reward_hip_pos(self):
        """
        Penalize hip roll and pitch joint positions away from zero.
        Encourages upright hip configuration.
        G1 hip indices: [1, 2] for left, [7, 8] for right
        """
        hip_indices = [1, 2, 7, 8]  # hip_roll and hip_pitch for both legs
        return torch.sum(torch.square(self.dof_pos[:, hip_indices]), dim=1)
