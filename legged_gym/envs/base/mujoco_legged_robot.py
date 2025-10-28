"""
Mujoco-based Legged Robot Environment
Replaces Isaac Gym with Mujoco for Colab-compatible training
"""

import os
import numpy as np
import mujoco
import torch
from typing import Tuple, Dict
from rsl_rl.env import VecEnv

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg


class MujocoLeggedRobot(VecEnv):
    """
    Mujoco-based vectorized environment for legged robot locomotion training.

    Compatible with rsl_rl PPO training and runs on CPU (Colab-friendly).
    Implements the same interface as Isaac Gym LeggedRobot.
    """

    def __init__(self, cfg: LeggedRobotCfg, device='cpu'):
        """
        Initialize Mujoco environment.

        Args:
            cfg: Environment configuration
            device: Compute device ('cpu' or 'cuda')
        """
        self.cfg = cfg
        self.device = device

        # Parse configuration
        self._parse_cfg(cfg)

        # Create Mujoco models (one per environment)
        self._load_mujoco_model()

        # Initialize buffers
        self._init_buffers()

        # Prepare reward function
        self._prepare_reward_function()

        # Initialize environments
        self.reset()

        print(f"[MujocoLeggedRobot] Created {self.num_envs} environments")
        print(f"[MujocoLeggedRobot] Observation dim: {self.num_obs}")
        print(f"[MujocoLeggedRobot] Action dim: {self.num_actions}")

    def _parse_cfg(self, cfg):
        """Parse configuration parameters"""
        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # Timing
        self.sim_dt = 0.002  # 500 Hz simulation
        self.control_decimation = cfg.control.decimation
        self.dt = self.control_decimation * self.sim_dt  # Policy update rate

        # Episode length
        self.max_episode_length_s = cfg.env.episode_length_s
        self.max_episode_length = int(self.max_episode_length_s / self.dt)

        # Scales
        self.obs_scales = cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(cfg.rewards.scales)
        self.command_ranges = class_to_dict(cfg.commands.ranges)

        # Control
        self.action_scale = cfg.control.action_scale

        # Commands scale
        self.commands_scale = torch.tensor([
            self.obs_scales.lin_vel,
            self.obs_scales.lin_vel,
            self.obs_scales.ang_vel
        ], device=self.device)

        # Domain randomization
        self.push_interval = int(cfg.domain_rand.push_interval_s / self.dt)

    def _load_mujoco_model(self):
        """Load Mujoco model from URDF"""
        # Get URDF path
        urdf_path = self.cfg.asset.file.replace('{LEGGED_GYM_ROOT_DIR}', LEGGED_GYM_ROOT_DIR)

        # Convert URDF to Mujoco XML if needed
        if urdf_path.endswith('.urdf'):
            # Load URDF into Mujoco
            self.model = mujoco.MjModel.from_xml_path(urdf_path)
        else:
            self.model = mujoco.MjModel.from_xml_path(urdf_path)

        self.model.opt.timestep = self.sim_dt

        # Store model info
        self.num_dof = self.model.nv - 6  # Exclude floating base
        self.num_bodies = self.model.nbody

        # Get DOF names
        self.dof_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                          for i in range(self.model.njnt) if self.model.jnt_type[i] != 0]  # Exclude free joint

        # Extract PD gains from config
        self._setup_pd_control()

        # Get default joint angles
        self._setup_default_joint_angles()

        # Get DOF limits
        self._setup_dof_limits()

        # Find body indices
        self._find_body_indices()

        print(f"[MujocoLeggedRobot] Loaded model: {self.model.nq} qpos, {self.model.nv} qvel, {self.num_dof} actuated DOF")

    def _setup_pd_control(self):
        """Setup PD controller gains from config"""
        stiffness = self.cfg.control.stiffness
        damping = self.cfg.control.damping

        # Extract gains for each joint
        self.p_gains = torch.zeros(self.num_dof, device=self.device)
        self.d_gains = torch.zeros(self.num_dof, device=self.device)

        for i, dof_name in enumerate(self.dof_names):
            # Match joint name to config (e.g., "left_hip_yaw_joint" -> "hip_yaw")
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains[i] = stiffness[key]
                    self.d_gains[i] = damping[key]
                    break

        print(f"[MujocoLeggedRobot] PD gains: Kp={self.p_gains.tolist()}, Kd={self.d_gains.tolist()}")

    def _setup_default_joint_angles(self):
        """Get default joint angles from config"""
        default_angles_dict = self.cfg.init_state.default_joint_angles

        self.default_dof_pos = torch.zeros(self.num_dof, device=self.device)

        for i, dof_name in enumerate(self.dof_names):
            if dof_name in default_angles_dict:
                self.default_dof_pos[i] = default_angles_dict[dof_name]

        print(f"[MujocoLeggedRobot] Default DOF positions: {self.default_dof_pos.tolist()}")

    def _setup_dof_limits(self):
        """Extract DOF limits from Mujoco model"""
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dof, device=self.device)
        self.torque_limits = torch.zeros(self.num_dof, device=self.device)

        # Get limits from Mujoco model (skip free joint)
        actuated_joint_idx = 0
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] == 0:  # Free joint
                continue

            # Position limits
            self.dof_pos_limits[actuated_joint_idx, 0] = self.model.jnt_range[i, 0]
            self.dof_pos_limits[actuated_joint_idx, 1] = self.model.jnt_range[i, 1]

            # Velocity limits (from model)
            self.dof_vel_limits[actuated_joint_idx] = 10.0  # Default if not specified

            # Torque limits (from actuator force range)
            if actuated_joint_idx < self.model.nu:
                self.torque_limits[actuated_joint_idx] = self.model.actuator_forcerange[actuated_joint_idx, 1]

            actuated_joint_idx += 1

        # Apply soft limits
        soft_limit = self.cfg.rewards.soft_dof_pos_limit
        for i in range(self.num_dof):
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * soft_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * soft_limit

    def _find_body_indices(self):
        """Find body indices for feet and contact bodies"""
        body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                      for i in range(self.model.nbody)]

        # Find feet
        foot_name = self.cfg.asset.foot_name
        self.feet_indices = [i for i, name in enumerate(body_names) if foot_name in name]

        # Find penalized contact bodies
        self.penalised_contact_indices = []
        for contact_name in self.cfg.asset.penalize_contacts_on:
            self.penalised_contact_indices.extend([i for i, name in enumerate(body_names) if contact_name in name])

        # Find termination contact bodies
        self.termination_contact_indices = []
        for contact_name in self.cfg.asset.terminate_after_contacts_on:
            self.termination_contact_indices.extend([i for i, name in enumerate(body_names) if contact_name in name])

        print(f"[MujocoLeggedRobot] Feet indices: {self.feet_indices}")
        print(f"[MujocoLeggedRobot] Penalized contact indices: {self.penalised_contact_indices}")
        print(f"[MujocoLeggedRobot] Termination contact indices: {self.termination_contact_indices}")

    def _init_buffers(self):
        """Initialize state buffers for vectorized environments"""
        # Create Mujoco data instances for each environment
        self.mjdata_list = [mujoco.MjData(self.model) for _ in range(self.num_envs)]

        # State buffers (num_envs, dim)
        self.base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)

        self.dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.last_dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)

        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)

        # Commands
        self.commands = torch.zeros(self.num_envs, 4, device=self.device)  # vx, vy, vyaw, heading

        # Rewards
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Observations
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)
        else:
            self.privileged_obs_buf = None

        # Contact forces
        self.contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)

        # Torques
        self.torques = torch.zeros(self.num_envs, self.num_dof, device=self.device)

        # Extras
        self.extras = {}

        # Gravity vector
        self.gravity_vec = torch.tensor([0, 0, -1], device=self.device).repeat(self.num_envs, 1)

        # Episode sums for logging
        self.episode_sums = {name: torch.zeros(self.num_envs, device=self.device)
                             for name in self.reward_scales.keys()}

        # Friction coefficients for domain randomization
        if self.cfg.domain_rand.randomize_friction:
            friction_range = self.cfg.domain_rand.friction_range
            self.friction_coeffs = torch.rand(self.num_envs, device=self.device) * \
                                   (friction_range[1] - friction_range[0]) + friction_range[0]
        else:
            self.friction_coeffs = torch.ones(self.num_envs, device=self.device) * \
                                   self.cfg.terrain.static_friction

        # Base mass randomization
        self.base_mass_offsets = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.domain_rand.randomize_base_mass:
            mass_range = self.cfg.domain_rand.added_mass_range
            self.base_mass_offsets = torch.rand(self.num_envs, device=self.device) * \
                                     (mass_range[1] - mass_range[0]) + mass_range[0]

    def _prepare_reward_function(self):
        """Prepare reward functions from config"""
        self.reward_functions = []
        self.reward_names = []

        for name, scale in self.reward_scales.items():
            if scale == 0:
                continue
            if name == "termination":
                continue
            self.reward_names.append(name)
            reward_fn_name = '_reward_' + name
            if hasattr(self, reward_fn_name):
                self.reward_functions.append(getattr(self, reward_fn_name))
            else:
                print(f"[Warning] Reward function {reward_fn_name} not found, skipping")

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step the environments.

        Args:
            actions: Action tensor (num_envs, num_actions)

        Returns:
            obs: Observations (num_envs, num_obs)
            privileged_obs: Privileged observations or None
            rewards: Rewards (num_envs,)
            dones: Done flags (num_envs,)
            extras: Extra info dict
        """
        # Clip and store actions
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions)

        # Step physics for each decimation step
        for dec_step in range(self.control_decimation):
            # Compute torques from actions
            self.torques = self._compute_torques(self.actions)

            # Apply torques and step each environment
            for env_id in range(self.num_envs):
                mjd = self.mjdata_list[env_id]
                mjd.ctrl[:] = self.torques[env_id].cpu().numpy()
                mujoco.mj_step(self.model, mjd)

            # Update state buffers after physics step
            self._update_state_buffers()

        # Post-physics step callback
        self._post_physics_step_callback()

        # Check terminations
        self._check_termination()

        # Compute rewards
        self._compute_rewards()

        # Compute observations
        self.compute_observations()

        # Reset environments that are done
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._reset_envs(env_ids)

        # Update episode length
        self.episode_length_buf += 1

        # Store last actions
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # Clip observations
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        """Reset all environments"""
        self._reset_envs(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        return self.obs_buf, self.privileged_obs_buf

    def _reset_envs(self, env_ids):
        """Reset specific environments"""
        if len(env_ids) == 0:
            return

        for env_id in env_ids:
            mjd = self.mjdata_list[env_id.item()]

            # Reset to default pose
            mujoco.mj_resetData(self.model, mjd)

            # Set default joint positions with randomization
            default_qpos = self.default_dof_pos.cpu().numpy()
            noise = np.random.uniform(0.5, 1.5, size=self.num_dof)
            mjd.qpos[7:7+self.num_dof] = default_qpos * noise

            # Reset base position
            mjd.qpos[2] = self.cfg.init_state.pos[2]  # Height

            # Reset velocities
            mjd.qvel[:] = 0

            # Forward kinematics
            mujoco.mj_forward(self.model, mjd)

        # Reset buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.time_out_buf[env_ids] = False
        self.last_actions[env_ids] = 0
        self.last_dof_vel[env_ids] = 0

        # Resample commands
        self._resample_commands(env_ids)

        # Update state buffers
        self._update_state_buffers(env_ids)

        # Reset episode sums
        for key in self.episode_sums.keys():
            self.episode_sums[key][env_ids] = 0

    def _update_state_buffers(self, env_ids=None):
        """Update state buffers from Mujoco data"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        for env_id in env_ids:
            mjd = self.mjdata_list[env_id.item()]

            # Base state
            self.base_pos[env_id] = torch.from_numpy(mjd.qpos[:3]).to(self.device)
            self.base_quat[env_id] = torch.from_numpy(mjd.qpos[3:7]).to(self.device)

            # Base velocity (in world frame, need to convert to base frame)
            lin_vel_world = torch.from_numpy(mjd.qvel[:3]).to(self.device)
            ang_vel_world = torch.from_numpy(mjd.qvel[3:6]).to(self.device)

            # Rotate to base frame
            self.base_lin_vel[env_id] = self._quat_rotate_inverse(self.base_quat[env_id], lin_vel_world)
            self.base_ang_vel[env_id] = self._quat_rotate_inverse(self.base_quat[env_id], ang_vel_world)

            # Projected gravity
            self.projected_gravity[env_id] = self._quat_rotate_inverse(self.base_quat[env_id], self.gravity_vec[env_id])

            # DOF state
            self.dof_pos[env_id] = torch.from_numpy(mjd.qpos[7:7+self.num_dof]).to(self.device)
            self.dof_vel[env_id] = torch.from_numpy(mjd.qvel[6:6+self.num_dof]).to(self.device)

            # Contact forces
            for i in range(self.model.nbody):
                # Sum contact forces on this body
                contact_force = np.zeros(3)
                for j in range(mjd.ncon):
                    contact = mjd.contact[j]
                    if contact.geom1 in self._get_body_geoms(i) or contact.geom2 in self._get_body_geoms(i):
                        # Get contact force
                        c_array = np.zeros(6)
                        mujoco.mj_contactForce(self.model, mjd, j, c_array)
                        contact_force += c_array[:3]

                self.contact_forces[env_id, i] = torch.from_numpy(contact_force).to(self.device)

    def _get_body_geoms(self, body_id):
        """Get list of geom IDs for a body"""
        geoms = []
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] == body_id:
                geoms.append(geom_id)
        return geoms

    def _quat_rotate_inverse(self, quat, vec):
        """Rotate vector by inverse quaternion"""
        quat_np = quat.cpu().numpy() if isinstance(quat, torch.Tensor) else quat
        vec_np = vec.cpu().numpy() if isinstance(vec, torch.Tensor) else vec

        # Quaternion conjugate for inverse rotation
        quat_conj = np.array([quat_np[0], -quat_np[1], -quat_np[2], -quat_np[3]])

        # Convert to rotation matrix and apply
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat([quat_np[1], quat_np[2], quat_np[3], quat_np[0]])  # scipy uses xyzw
        rotated = rot.inv().apply(vec_np)

        return torch.from_numpy(rotated).to(self.device)

    def _compute_torques(self, actions):
        """Compute torques from actions using PD control"""
        actions_scaled = actions * self.action_scale
        target_dof_pos = actions_scaled + self.default_dof_pos

        # PD control
        torques = self.p_gains * (target_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel

        # Clip torques
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _post_physics_step_callback(self):
        """Callback after physics step"""
        # Resample commands at intervals
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        # Random pushes (domain randomization)
        if self.cfg.domain_rand.push_robots:
            env_ids = (self.episode_length_buf % self.push_interval == 0).nonzero(as_tuple=False).flatten()
            self._push_robots(env_ids)

    def _resample_commands(self, env_ids):
        """Resample velocity commands"""
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * \
                                     (self.command_ranges["lin_vel_x"][1] - self.command_ranges["lin_vel_x"][0]) + \
                                     self.command_ranges["lin_vel_x"][0]
        self.commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * \
                                     (self.command_ranges["lin_vel_y"][1] - self.command_ranges["lin_vel_y"][0]) + \
                                     self.command_ranges["lin_vel_y"][0]
        self.commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * \
                                     (self.command_ranges["ang_vel_yaw"][1] - self.command_ranges["ang_vel_yaw"][0]) + \
                                     self.command_ranges["ang_vel_yaw"][0]

        # Zero out small commands
        lin_vel_norm = torch.norm(self.commands[env_ids, :2], dim=1)
        self.commands[env_ids, :2] *= (lin_vel_norm > 0.2).unsqueeze(1)

    def _push_robots(self, env_ids):
        """Apply random pushes to robots"""
        if len(env_ids) == 0:
            return

        max_push_vel = self.cfg.domain_rand.max_push_vel_xy

        for env_id in env_ids:
            push_vel = torch.rand(2, device=self.device) * 2 * max_push_vel - max_push_vel
            mjd = self.mjdata_list[env_id.item()]
            mjd.qvel[:2] += push_vel.cpu().numpy()

    def _check_termination(self):
        """Check termination conditions"""
        # Timeout
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length

        # Contact on termination bodies
        termination_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for body_idx in self.termination_contact_indices:
            termination_contact |= torch.norm(self.contact_forces[:, body_idx, :], dim=-1) > 1.0

        # Orientation (fallen over)
        body_upright = self.projected_gravity[:, 2] > -0.5  # More than 60 degrees tilt

        # Combine conditions
        self.reset_buf = self.time_out_buf | termination_contact | ~body_upright

    def _compute_rewards(self):
        """Compute rewards"""
        self.rew_buf[:] = 0

        for i, reward_fn in enumerate(self.reward_functions):
            name = self.reward_names[i]
            reward = reward_fn() * self.reward_scales[name]
            self.rew_buf += reward
            self.episode_sums[name] += reward

        # Clip negative rewards if configured
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf, min=0.0)

    def compute_observations(self):
        """Compute observations - OVERRIDE IN SUBCLASS"""
        # Base implementation (should be overridden)
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.cat((
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions
            ), dim=-1)

    # ============ Reward Functions ============

    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity"""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity"""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """Penalize non-flat base orientation"""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """Penalize base height away from target"""
        return torch.square(self.base_pos[:, 2] - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        """Penalize torques"""
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """Penalize dof velocities"""
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """Penalize dof accelerations"""
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        """Penalize changes in actions"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        """Penalize collisions on selected bodies"""
        collision_penalty = torch.zeros(self.num_envs, device=self.device)
        for body_idx in self.penalised_contact_indices:
            collision_penalty += (torch.norm(self.contact_forces[:, body_idx, :], dim=-1) > 0.1).float()
        return collision_penalty

    def _reward_dof_pos_limits(self):
        """Penalize dof positions too close to limits"""
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands"""
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    # ============ Properties for rsl_rl compatibility ============

    @property
    def num_envs(self):
        return self._num_envs

    @num_envs.setter
    def num_envs(self, value):
        self._num_envs = value

    @property
    def num_obs(self):
        return self._num_obs

    @num_obs.setter
    def num_obs(self, value):
        self._num_obs = value

    @property
    def num_privileged_obs(self):
        return self._num_privileged_obs

    @num_privileged_obs.setter
    def num_privileged_obs(self, value):
        self._num_privileged_obs = value

    @property
    def num_actions(self):
        return self._num_actions

    @num_actions.setter
    def num_actions(self, value):
        self._num_actions = value

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf
