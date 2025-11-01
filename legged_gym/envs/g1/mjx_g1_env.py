"""
JAX/MJX G1 Robot Environment
GPU-accelerated G1 humanoid with phase-based gait control
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, Any
import mujoco
from mujoco import mjx
import numpy as np

from legged_gym.envs.base.mjx_legged_robot import MJXLeggedRobot, EnvState
from legged_gym.envs.g1.mujoco_g1_config import MujocoG1RoughCfg


class MJXG1Robot(MJXLeggedRobot):
    """
    JAX/MJX-based G1 humanoid robot environment.
    Includes phase-based gait rewards and G1-specific observations.
    Fully vectorized for GPU acceleration.
    """

    def __init__(self, cfg: MujocoG1RoughCfg = None, backend: str = 'mjx'):
        """
        Initialize G1 environment.

        Args:
            cfg: G1 configuration (uses default if None)
            backend: Physics backend ('mjx' for GPU)
        """
        if cfg is None:
            cfg = MujocoG1RoughCfg()

        super().__init__(cfg, backend)

        # G1-specific setup
        self._setup_g1_indices()

    def _setup_g1_indices(self):
        """Find G1-specific body indices"""
        # Find feet bodies
        foot_name = self.cfg.asset.foot_name  # e.g., "ankle"
        self.feet_indices = []

        for i in range(self.mj_model.nbody):
            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and foot_name in body_name:
                self.feet_indices.append(i)

        self.num_feet = len(self.feet_indices)
        self.feet_indices = jnp.array(self.feet_indices)

        # Find hip joints (for hip position penalty)
        # G1 has: left_hip_roll, left_hip_pitch, right_hip_roll, right_hip_pitch
        # Typically at indices [1, 2, 7, 8] in the DOF array
        self.hip_indices = jnp.array([1, 2, 7, 8])

        # Find penalized contact bodies
        self.penalized_contact_indices = []
        for contact_name in self.cfg.asset.penalize_contacts_on:
            for i in range(self.mj_model.nbody):
                body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and contact_name in body_name:
                    self.penalized_contact_indices.append(i)

        self.penalized_contact_indices = jnp.array(self.penalized_contact_indices)

        print(f"[MJXG1Robot] Found {self.num_feet} feet: {self.feet_indices.tolist()}")
        print(f"[MJXG1Robot] Hip indices: {self.hip_indices.tolist()}")

    def _compute_observations(self, state: EnvState) -> jnp.ndarray:
        """
        Compute G1 observations including gait phase.

        Observation space (47 dims):
        - angular velocity (3)
        - projected gravity (3)
        - commands (3)
        - dof positions relative to default (12)
        - dof velocities (12)
        - previous actions (12)
        - phase sin/cos (2)
        """
        pipeline_state = state.pipeline_state

        # Extract state components
        base_quat = pipeline_state.qpos[3:7]
        base_ang_vel = pipeline_state.qvel[3:6]

        # Projected gravity (rotate gravity vector by inverse base quaternion)
        gravity_world = jnp.array([0., 0., -1.])
        projected_gravity = self._quat_rotate_inverse(base_quat, gravity_world)

        # DOF state
        dof_pos = self._get_dof_pos(pipeline_state)
        dof_vel = self._get_dof_vel(pipeline_state)

        # Phase encoding
        sin_phase = jnp.sin(2 * jnp.pi * state.phase)
        cos_phase = jnp.cos(2 * jnp.pi * state.phase)

        # Concatenate observations
        obs = jnp.concatenate([
            base_ang_vel * self.ang_vel_scale,
            projected_gravity,
            state.commands[:3] * self.commands_scale,
            (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
            dof_vel * self.dof_vel_scale,
            state.last_actions,
            jnp.array([sin_phase, cos_phase])
        ])

        return obs

    def _quat_rotate_inverse(self, quat: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
        """Rotate vector by inverse of quaternion (from world to body frame)"""
        # Quaternion format: [w, x, y, z]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # Compute rotation using quaternion formula
        # v' = q^{-1} * v * q  where q^{-1} = [w, -x, -y, -z] / ||q||^2
        # Optimized version for unit quaternions

        # Build rotation matrix from quaternion
        R = jnp.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

        # For inverse rotation, use transpose (since R is orthogonal)
        return R.T @ vec

    def _get_contact_forces(self, pipeline_state: Any) -> jnp.ndarray:
        """Extract contact forces for all bodies"""
        if self.backend == 'mjx':
            # MJX stores contact data differently
            # For now, use simplified version based on contact.force
            # Shape: (max_contacts, 3) for normal force
            contact_force = pipeline_state.contact.force if hasattr(pipeline_state, 'contact') else jnp.zeros((0, 3))
            return contact_force
        else:
            # CPU version
            return jnp.zeros((self.mj_model.nbody, 3))

    def _check_foot_contact(self, pipeline_state: Any, foot_idx: int) -> jnp.ndarray:
        """Check if specific foot is in contact (force > 1N)"""
        # Simplified: check contact force magnitude
        # In full MJX implementation, would iterate through contact pairs
        # For now, use placeholder that will work with JIT
        contact_forces = self._get_contact_forces(pipeline_state)

        # Check if any contact involves this foot body
        # This is simplified - full version would check contact.geom1/geom2
        in_contact = jnp.any(jnp.linalg.norm(contact_forces, axis=1) > 1.0)

        return in_contact

    def _get_feet_positions(self, pipeline_state: Any) -> jnp.ndarray:
        """Get positions of all feet"""
        if self.backend == 'mjx':
            # xpos contains all body positions
            # Shape: (nbody, 3)
            feet_pos = pipeline_state.xpos[self.feet_indices]
        else:
            feet_pos = jnp.zeros((self.num_feet, 3))

        return feet_pos

    def _get_feet_velocities(self, pipeline_state: Any) -> jnp.ndarray:
        """Get velocities of all feet"""
        # For simplicity, compute from qvel using jacobian
        # Full implementation would use mujoco.mj_objectVelocity equivalent
        # Placeholder for now
        return jnp.zeros((self.num_feet, 3))

    def _check_termination(self, pipeline_state: Any, episode_length: jnp.ndarray) -> jnp.ndarray:
        """Check G1-specific termination conditions"""
        # Timeout
        timeout = episode_length >= self.max_episode_length

        # Check if robot has fallen (projected gravity z-component)
        base_quat = pipeline_state.qpos[3:7]
        gravity_world = jnp.array([0., 0., -1.])
        projected_gravity = self._quat_rotate_inverse(base_quat, gravity_world)

        # Fallen if tilted more than 60 degrees (proj_grav_z > -0.5)
        fallen = projected_gravity[2] > -0.5

        # Contact on non-foot bodies (e.g., torso hitting ground)
        # Simplified version - full implementation would check contact bodies
        contact_termination = False

        return timeout | fallen | contact_termination

    # ============ G1-Specific Reward Functions ============

    def _reward_alive(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Reward for staying alive"""
        return 1.0

    def _reward_contact(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Reward foot contact that matches gait phase.
        Stance phase (< 0.55): foot should be in contact
        Swing phase (>= 0.55): foot should not be in contact
        """
        # Compute phase for each leg (left vs right with offset)
        phase_left = state.phase
        phase_right = (state.phase + 0.5) % 1.0

        # For simplicity, assume 2 feet: left and right
        # In full implementation, would iterate through all feet
        reward = 0.0

        # Left foot
        is_stance_left = phase_left < 0.55
        contact_left = self._check_foot_contact(pipeline_state, self.feet_indices[0])
        reward += jnp.where(is_stance_left == contact_left, 1.0, 0.0)

        # Right foot (if exists)
        if self.num_feet > 1:
            is_stance_right = phase_right < 0.55
            contact_right = self._check_foot_contact(pipeline_state, self.feet_indices[1])
            reward += jnp.where(is_stance_right == contact_right, 1.0, 0.0)

        return reward

    def _reward_feet_swing_height(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Penalize swing foot height away from target (8cm).
        Only penalize when foot is not in contact.
        """
        target_height = 0.08

        feet_pos = self._get_feet_positions(pipeline_state)

        # Check contact for each foot
        # Simplified: penalize height error for all feet during swing
        height_error = 0.0

        for i in range(self.num_feet):
            in_contact = self._check_foot_contact(pipeline_state, self.feet_indices[i])
            foot_height = feet_pos[i, 2]

            # Penalize only if not in contact
            error = (foot_height - target_height)**2
            height_error += jnp.where(in_contact, 0.0, error)

        return height_error

    def _reward_contact_no_vel(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Penalize foot velocity when in contact (should be stationary).
        Encourages static stance phase.
        """
        feet_vel = self._get_feet_velocities(pipeline_state)

        penalty = 0.0
        for i in range(self.num_feet):
            in_contact = self._check_foot_contact(pipeline_state, self.feet_indices[i])
            vel_magnitude = jnp.sum(feet_vel[i]**2)

            # Penalize only if in contact
            penalty += jnp.where(in_contact, vel_magnitude, 0.0)

        return penalty

    def _reward_hip_pos(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Penalize hip roll and pitch joint positions away from zero.
        Encourages upright hip configuration.
        """
        dof_pos = self._get_dof_pos(pipeline_state)
        hip_pos = dof_pos[self.hip_indices]

        return jnp.sum(hip_pos**2)

    def _reward_orientation(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Penalize non-flat base orientation"""
        base_quat = pipeline_state.qpos[3:7]
        gravity_world = jnp.array([0., 0., -1.])
        projected_gravity = self._quat_rotate_inverse(base_quat, gravity_world)

        # Penalize x and y components (should be zero when upright)
        return jnp.sum(projected_gravity[:2]**2)

    def _reward_lin_vel_z(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Penalize z-axis base linear velocity"""
        base_lin_vel = pipeline_state.qvel[:3]

        # Transform to body frame
        base_quat = pipeline_state.qpos[3:7]
        base_lin_vel_body = self._quat_rotate_inverse(base_quat, base_lin_vel)

        return base_lin_vel_body[2]**2

    def _reward_ang_vel_xy(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Penalize xy-axes base angular velocity"""
        base_ang_vel = pipeline_state.qvel[3:6]
        return jnp.sum(base_ang_vel[:2]**2)

    def _reward_base_height(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Penalize base height away from target"""
        base_height = pipeline_state.qpos[2]
        target_height = self.cfg.rewards.base_height_target
        return (base_height - target_height)**2

    def _reward_tracking_lin_vel(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Reward tracking linear velocity commands"""
        base_lin_vel = pipeline_state.qvel[:3]

        # Transform to body frame
        base_quat = pipeline_state.qpos[3:7]
        base_lin_vel_body = self._quat_rotate_inverse(base_quat, base_lin_vel)

        # Track commands (vx, vy)
        lin_vel_error = jnp.sum((state.commands[:2] - base_lin_vel_body[:2])**2)
        return jnp.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Reward tracking angular velocity commands"""
        base_ang_vel = pipeline_state.qvel[3:6]

        # Track yaw velocity command
        ang_vel_error = (state.commands[2] - base_ang_vel[2])**2
        return jnp.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_collision(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Penalize collisions on non-foot bodies"""
        # Simplified: check if any penalized bodies have contact
        # Full implementation would iterate through contact list
        penalty = 0.0

        # Placeholder - would check contact forces on penalized bodies
        return penalty

    def _reward_dof_acc(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Penalize dof accelerations"""
        dof_vel = self._get_dof_vel(pipeline_state)
        dof_acc = (dof_vel - state.last_dof_vel) / self.dt
        return jnp.sum(dof_acc**2)

    def _reward_dof_pos_limits(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Penalize dof positions too close to limits"""
        dof_pos = self._get_dof_pos(pipeline_state)

        # Get limits from model
        # For now, use soft clipping - full implementation would extract from model
        # Assuming limits are already applied in soft_dof_pos_limit during setup

        # Penalize positions near limits
        # Simplified: assume symmetric limits around default position
        max_deviation = 1.0  # radians
        deviation = jnp.abs(dof_pos - self.default_dof_pos)
        out_of_limits = jnp.maximum(deviation - max_deviation, 0.0)

        return jnp.sum(out_of_limits**2)


def create_g1_env(num_envs: int = 512, backend: str = 'mjx') -> MJXG1Robot:
    """
    Factory function to create G1 environment with common settings.

    Args:
        num_envs: Number of parallel environments
        backend: Physics backend ('mjx' for GPU)

    Returns:
        Configured G1 environment
    """
    cfg = MujocoG1RoughCfg()
    cfg.env.num_envs = num_envs

    return MJXG1Robot(cfg, backend)
