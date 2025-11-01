"""
JAX/MJX-based Legged Robot Environment
GPU-accelerated vectorized environment using MuJoCo MJX
Compatible with Brax PPO training
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, Any
import mujoco
from mujoco import mjx
from flax import struct
import chex

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym import LEGGED_GYM_ROOT_DIR


@struct.dataclass
class EnvState:
    """State of the environment (immutable using flax.struct)"""
    # MJX data
    pipeline_state: Any  # mjx.Data

    # Episode tracking
    episode_length: jnp.ndarray  # (num_envs,)
    done: jnp.ndarray  # (num_envs,)

    # Commands
    commands: jnp.ndarray  # (num_envs, 4) [vx, vy, vyaw, heading]

    # Action history
    last_actions: jnp.ndarray  # (num_envs, num_actions)
    last_dof_vel: jnp.ndarray  # (num_envs, num_dof)

    # Gait phase (for phase-based rewards)
    phase: jnp.ndarray  # (num_envs,)

    # Domain randomization parameters
    friction_coeffs: jnp.ndarray  # (num_envs,)
    base_mass_offsets: jnp.ndarray  # (num_envs,)

    # Reward accumulation (for logging)
    reward_sums: Dict[str, jnp.ndarray]  # {reward_name: (num_envs,)}

    # RNG key
    rng: jnp.ndarray


class MJXLeggedRobot:
    """
    JAX/MJX-based vectorized legged robot environment.

    All operations are functional (pure functions) for JIT compilation.
    Supports massive GPU parallelization (1000s of environments).
    Compatible with Brax PPO training.
    """

    def __init__(self, cfg: LeggedRobotCfg, backend: str = 'mjx'):
        """
        Initialize MJX environment.

        Args:
            cfg: Environment configuration
            backend: Physics backend ('mjx' for GPU, 'mujoco' for CPU)
        """
        self.cfg = cfg
        self.backend = backend

        # Parse configuration
        self._parse_cfg(cfg)

        # Load MuJoCo model
        self._load_model()

        # Create MJX model (GPU-optimized)
        if backend == 'mjx':
            self.mjx_model = mjx.put_model(self.mj_model)
        else:
            self.mjx_model = self.mj_model

        # Setup control parameters
        self._setup_control_params()

        # Prepare reward functions
        self._prepare_reward_functions()

        print(f"[MJXLeggedRobot] Initialized with {self.num_envs} environments")
        print(f"[MJXLeggedRobot] Backend: {backend}")
        print(f"[MJXLeggedRobot] Observations: {self.num_obs}")
        print(f"[MJXLeggedRobot] Actions: {self.num_actions}")

    def _parse_cfg(self, cfg):
        """Parse configuration parameters"""
        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # Timing
        self.sim_dt = 0.002  # 500 Hz simulation
        self.control_decimation = cfg.control.decimation
        self.dt = self.control_decimation * self.sim_dt

        # Episode length
        self.max_episode_length_s = cfg.env.episode_length_s
        self.max_episode_length = int(self.max_episode_length_s / self.dt)

        # Extract config values
        self.obs_scales = cfg.normalization.obs_scales
        self.reward_scales = self._dict_from_class(cfg.rewards.scales)
        self.command_ranges = self._dict_from_class(cfg.commands.ranges)
        self.action_scale = cfg.control.action_scale
        self.clip_actions = cfg.normalization.clip_actions
        self.clip_observations = cfg.normalization.clip_observations

        # Domain randomization
        self.domain_rand_cfg = cfg.domain_rand
        self.push_interval = int(cfg.domain_rand.push_interval_s / self.dt)
        self.command_resample_interval = int(cfg.commands.resampling_time / self.dt)

    def _dict_from_class(self, cls):
        """Convert class attributes to dictionary"""
        return {key: getattr(cls, key) for key in dir(cls) if not key.startswith('_')}

    def _load_model(self):
        """Load MuJoCo model from file"""
        asset_path = self.cfg.asset.file.replace('{LEGGED_GYM_ROOT_DIR}', LEGGED_GYM_ROOT_DIR)

        # Prefer XML over URDF
        if asset_path.endswith('.urdf'):
            xml_path = asset_path.replace('.urdf', '.xml')
            import os
            if os.path.exists(xml_path):
                print(f"[MJXLeggedRobot] Using compiled XML: {xml_path}")
                asset_path = xml_path

        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(asset_path)
        self.mj_model.opt.timestep = self.sim_dt

        # Store model info
        self.num_dof = self.mj_model.nu if self.mj_model.nu > 0 else self.cfg.env.num_actions

        print(f"[MJXLeggedRobot] Loaded model: {self.mj_model.nq} qpos, {self.mj_model.nv} qvel, {self.num_dof} DOF")

    def _setup_control_params(self):
        """Setup PD control gains and default positions"""
        stiffness = self._dict_from_class(self.cfg.control.stiffness)
        damping = self._dict_from_class(self.cfg.control.damping)
        default_angles = self._dict_from_class(self.cfg.init_state.default_joint_angles)

        # Get actuator names
        actuator_names = []
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            actuator_names.append(name if name else f"actuator_{i}")

        # Match gains to actuators
        p_gains = []
        d_gains = []
        default_pos = []

        for name in actuator_names:
            # Match actuator name to config keys
            matched_p = 0.0
            matched_d = 0.0
            matched_pos = 0.0

            for key in stiffness.keys():
                if key in name:
                    matched_p = stiffness[key]
                    matched_d = damping[key]
                    break

            for key in default_angles.keys():
                if key in name:
                    matched_pos = default_angles[key]
                    break

            p_gains.append(matched_p)
            d_gains.append(matched_d)
            default_pos.append(matched_pos)

        # Convert to JAX arrays
        self.p_gains = jnp.array(p_gains)
        self.d_gains = jnp.array(d_gains)
        self.default_dof_pos = jnp.array(default_pos)

        # Observation scales as JAX arrays
        self.lin_vel_scale = self.obs_scales.lin_vel
        self.ang_vel_scale = self.obs_scales.ang_vel
        self.dof_pos_scale = self.obs_scales.dof_pos
        self.dof_vel_scale = self.obs_scales.dof_vel

        # Command scales
        self.commands_scale = jnp.array([
            self.obs_scales.lin_vel,
            self.obs_scales.lin_vel,
            self.obs_scales.ang_vel
        ])

    def _prepare_reward_functions(self):
        """Prepare list of reward functions with their names"""
        self.reward_functions = []
        self.reward_names = []

        for name, scale in self.reward_scales.items():
            if scale == 0 or name == "termination":
                continue

            reward_fn_name = f'_reward_{name}'
            if hasattr(self, reward_fn_name):
                self.reward_functions.append(getattr(self, reward_fn_name))
                self.reward_names.append(name)

    def reset(self, rng: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray]:
        """
        Reset all environments.

        Args:
            rng: JAX random key

        Returns:
            state: Initial environment state
            obs: Initial observations (num_envs, num_obs)
        """
        rng, reset_rng = random.split(rng)

        # Create initial MJX state
        if self.backend == 'mjx':
            pipeline_state = mjx.make_data(self.mjx_model)
        else:
            pipeline_state = mujoco.MjData(self.mj_model)

        # Initialize state for all environments
        state = self._reset_all_envs(pipeline_state, reset_rng)

        # Compute initial observations
        obs = self._compute_observations(state)

        return state, obs

    def _reset_all_envs(self, pipeline_state: Any, rng: jnp.ndarray) -> EnvState:
        """Initialize state for all environments"""
        rng, cmd_rng, friction_rng, mass_rng = random.split(rng, 4)

        # Initialize qpos/qvel with default pose + noise
        qpos = jnp.tile(self.mj_model.qpos0, (self.num_envs, 1))
        qvel = jnp.zeros((self.num_envs, self.mj_model.nv))

        # Add noise to joint positions
        noise_rng, rng = random.split(rng)
        joint_noise = random.uniform(noise_rng, (self.num_envs, self.num_dof), minval=0.5, maxval=1.5)
        qpos = qpos.at[:, 7:7+self.num_dof].set(self.default_dof_pos * joint_noise)

        # Set base height
        qpos = qpos.at[:, 2].set(self.cfg.init_state.pos[2])

        # Vectorized physics state (if using MJX)
        if self.backend == 'mjx':
            pipeline_state = pipeline_state.replace(qpos=qpos[0], qvel=qvel[0])
            # MJX will vmap over these

        # Initialize commands
        commands = self._resample_commands(jnp.arange(self.num_envs), cmd_rng)

        # Domain randomization
        if self.domain_rand_cfg.randomize_friction:
            friction_range = self.domain_rand_cfg.friction_range
            friction_coeffs = random.uniform(
                friction_rng,
                (self.num_envs,),
                minval=friction_range[0],
                maxval=friction_range[1]
            )
        else:
            friction_coeffs = jnp.ones(self.num_envs) * self.cfg.terrain.static_friction

        if self.domain_rand_cfg.randomize_base_mass:
            mass_range = self.domain_rand_cfg.added_mass_range
            base_mass_offsets = random.uniform(
                mass_rng,
                (self.num_envs,),
                minval=mass_range[0],
                maxval=mass_range[1]
            )
        else:
            base_mass_offsets = jnp.zeros(self.num_envs)

        # Initialize reward sums
        reward_sums = {name: jnp.zeros(self.num_envs) for name in self.reward_names}

        return EnvState(
            pipeline_state=pipeline_state,
            episode_length=jnp.zeros(self.num_envs, dtype=jnp.int32),
            done=jnp.zeros(self.num_envs, dtype=bool),
            commands=commands,
            last_actions=jnp.zeros((self.num_envs, self.num_actions)),
            last_dof_vel=jnp.zeros((self.num_envs, self.num_dof)),
            phase=jnp.zeros(self.num_envs),
            friction_coeffs=friction_coeffs,
            base_mass_offsets=base_mass_offsets,
            reward_sums=reward_sums,
            rng=rng
        )

    def step(self, state: EnvState, actions: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """
        Step the environment.

        Args:
            state: Current environment state
            actions: Actions (num_envs, num_actions)

        Returns:
            next_state: Updated state
            obs: Observations (num_envs, num_obs)
            rewards: Rewards (num_envs,)
            info: Additional info dict
        """
        # Clip actions
        actions = jnp.clip(actions, -self.clip_actions, self.clip_actions)

        # Step physics (with control decimation)
        pipeline_state = state.pipeline_state
        for _ in range(self.control_decimation):
            # Compute torques using PD control
            torques = self._compute_torques(pipeline_state, actions)

            # Step physics
            if self.backend == 'mjx':
                pipeline_state = mjx.step(self.mjx_model, pipeline_state)
            else:
                # For CPU fallback
                pipeline_state.ctrl[:] = torques
                mujoco.mj_step(self.mj_model, pipeline_state)

        # Update episode length
        episode_length = state.episode_length + 1

        # Update phase
        phase = (episode_length.astype(jnp.float32) * self.dt) % 0.8 / 0.8

        # Check termination
        done = self._check_termination(pipeline_state, episode_length)

        # Compute rewards
        rewards, reward_components = self._compute_rewards(state, pipeline_state, actions)

        # Update reward sums
        reward_sums = {
            name: state.reward_sums[name] + reward_components[name]
            for name in self.reward_names
        }

        # Resample commands if needed
        rng, cmd_rng = random.split(state.rng)
        should_resample = (episode_length % self.command_resample_interval) == 0
        env_ids = jnp.where(should_resample)[0]
        commands = jax.lax.cond(
            env_ids.size > 0,
            lambda: self._resample_commands_partial(state.commands, env_ids, cmd_rng),
            lambda: state.commands
        )

        # Create next state
        next_state = state.replace(
            pipeline_state=pipeline_state,
            episode_length=episode_length,
            done=done,
            commands=commands,
            last_actions=actions,
            last_dof_vel=self._get_dof_vel(pipeline_state),
            phase=phase,
            reward_sums=reward_sums,
            rng=rng
        )

        # Reset done environments
        next_state = self._auto_reset(next_state)

        # Compute observations
        obs = self._compute_observations(next_state)
        obs = jnp.clip(obs, -self.clip_observations, self.clip_observations)

        # Info dict
        info = {
            'episode_length': episode_length,
            'reward_components': reward_components
        }

        return next_state, obs, rewards, info

    def _compute_torques(self, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Compute torques from actions using PD control"""
        # Get current DOF state
        dof_pos = self._get_dof_pos(pipeline_state)
        dof_vel = self._get_dof_vel(pipeline_state)

        # Compute target positions
        actions_scaled = actions * self.action_scale
        target_dof_pos = actions_scaled + self.default_dof_pos

        # PD control
        torques = self.p_gains * (target_dof_pos - dof_pos) - self.d_gains * dof_vel

        return torques

    def _get_dof_pos(self, pipeline_state: Any) -> jnp.ndarray:
        """Extract DOF positions from pipeline state"""
        if self.backend == 'mjx':
            return pipeline_state.qpos[7:7+self.num_dof]
        else:
            return pipeline_state.qpos[7:7+self.num_dof]

    def _get_dof_vel(self, pipeline_state: Any) -> jnp.ndarray:
        """Extract DOF velocities from pipeline state"""
        if self.backend == 'mjx':
            return pipeline_state.qvel[6:6+self.num_dof]
        else:
            return pipeline_state.qvel[6:6+self.num_dof]

    def _check_termination(self, pipeline_state: Any, episode_length: jnp.ndarray) -> jnp.ndarray:
        """Check termination conditions"""
        # Timeout
        timeout = episode_length >= self.max_episode_length

        # Body orientation (fallen over)
        # Get projected gravity (z-component in body frame)
        quat = pipeline_state.qpos[3:7]  # Base quaternion
        # Simplified: check if robot flipped (will be vectorized properly in subclass)
        fallen = False  # Placeholder, implement in subclass with proper quaternion math

        return timeout | fallen

    def _compute_rewards(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> Tuple[jnp.ndarray, Dict]:
        """Compute total rewards and components"""
        reward_components = {}
        total_reward = jnp.zeros(self.num_envs)

        for reward_fn, name in zip(self.reward_functions, self.reward_names):
            reward = reward_fn(state, pipeline_state, actions)
            scaled_reward = reward * self.reward_scales[name]
            reward_components[name] = scaled_reward
            total_reward += scaled_reward

        # Clip negative rewards if configured
        if self.cfg.rewards.only_positive_rewards:
            total_reward = jnp.maximum(total_reward, 0.0)

        return total_reward, reward_components

    def _compute_observations(self, state: EnvState) -> jnp.ndarray:
        """Compute observations - OVERRIDE IN SUBCLASS"""
        # Base implementation (override in subclass)
        pipeline_state = state.pipeline_state

        # Extract state
        base_ang_vel = pipeline_state.qvel[3:6]  # Angular velocity
        projected_gravity = jnp.array([0, 0, -1])  # Simplified
        dof_pos = self._get_dof_pos(pipeline_state)
        dof_vel = self._get_dof_vel(pipeline_state)

        obs = jnp.concatenate([
            base_ang_vel * self.ang_vel_scale,
            projected_gravity,
            state.commands[:3] * self.commands_scale,
            (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
            dof_vel * self.dof_vel_scale,
            state.last_actions
        ])

        return obs

    def _resample_commands(self, env_ids: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        """Sample new commands for specified environments"""
        rng_x, rng_y, rng_yaw = random.split(rng, 3)

        num_envs = env_ids.shape[0] if env_ids.ndim > 0 else self.num_envs

        cmd_x = random.uniform(
            rng_x, (num_envs,),
            minval=self.command_ranges['lin_vel_x'][0],
            maxval=self.command_ranges['lin_vel_x'][1]
        )

        cmd_y = random.uniform(
            rng_y, (num_envs,),
            minval=self.command_ranges['lin_vel_y'][0],
            maxval=self.command_ranges['lin_vel_y'][1]
        )

        cmd_yaw = random.uniform(
            rng_yaw, (num_envs,),
            minval=self.command_ranges['ang_vel_yaw'][0],
            maxval=self.command_ranges['ang_vel_yaw'][1]
        )

        # Zero out small commands
        lin_vel_norm = jnp.sqrt(cmd_x**2 + cmd_y**2)
        mask = lin_vel_norm > 0.2
        cmd_x = cmd_x * mask
        cmd_y = cmd_y * mask

        commands = jnp.stack([cmd_x, cmd_y, cmd_yaw, jnp.zeros(num_envs)], axis=1)

        return commands

    def _resample_commands_partial(self, current_commands: jnp.ndarray, env_ids: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        """Update commands only for specified environments"""
        new_commands = self._resample_commands(env_ids, rng)
        return current_commands.at[env_ids].set(new_commands)

    def _auto_reset(self, state: EnvState) -> EnvState:
        """Automatically reset done environments"""
        # For now, simple version - full implementation needs proper MJX handling
        # This will be expanded in practice with proper state reinitialization
        return state

    # ============ Reward Functions (Base) ============

    def _reward_tracking_lin_vel(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Reward tracking linear velocity commands"""
        base_lin_vel = pipeline_state.qvel[:3]  # Simplified
        lin_vel_error = jnp.sum((state.commands[:2] - base_lin_vel[:2])**2)
        return jnp.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Reward tracking angular velocity commands"""
        base_ang_vel = pipeline_state.qvel[5]  # Yaw velocity
        ang_vel_error = (state.commands[2] - base_ang_vel)**2
        return jnp.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_torques(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Penalize torques"""
        torques = self._compute_torques(pipeline_state, actions)
        return jnp.sum(torques**2)

    def _reward_action_rate(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Penalize changes in actions"""
        return jnp.sum((actions - state.last_actions)**2)

    def _reward_dof_vel(self, state: EnvState, pipeline_state: Any, actions: jnp.ndarray) -> jnp.ndarray:
        """Penalize DOF velocities"""
        dof_vel = self._get_dof_vel(pipeline_state)
        return jnp.sum(dof_vel**2)
