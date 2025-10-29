import time
import sys

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


class KeyboardController:
    """Keyboard controller for commanding robot velocities"""
    def __init__(self):
        self.cmd = np.zeros(3, dtype=np.float32)  # vx, vy, vyaw
        self.cmd_increment = 0.1
        self.yaw_increment = 0.3

    def update(self, viewer):
        """Update command based on keyboard input"""
        # Note: Mujoco viewer doesn't have direct keyboard callbacks
        # This is a placeholder for future keyboard integration
        pass

    def print_controls(self):
        """Print keyboard controls"""
        print("\n" + "="*60)
        print("KEYBOARD CONTROLS")
        print("="*60)
        print("  W/S: Forward/Backward velocity")
        print("  A/D: Left/Right velocity")
        print("  Q/E: Rotate left/right")
        print("  R: Reset commands to zero")
        print("  ESC: Exit")
        print("="*60)
        print(f"Commands: vx={self.cmd[0]:.2f}, vy={self.cmd[1]:.2f}, vyaw={self.cmd[2]:.2f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MUJOCO POLICY VISUALIZATION")
    print("="*60 + "\n")

    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser(description='Visualize trained policy in Mujoco')
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument("--policy", type=str, default=None, help="Override policy path from config")
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration")
    args = parser.parse_args()
    config_file = args.config_file

    print(f"Loading configuration: {config_file}")

    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        # Override policy path if provided
        if args.policy:
            policy_path = args.policy
        else:
            policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        # Override duration if provided
        if args.duration:
            simulation_duration = args.duration
        else:
            simulation_duration = config["simulation_duration"]

        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)

    print(f"  Policy: {policy_path}")
    print(f"  Model: {xml_path}")
    print(f"  Duration: {simulation_duration}s")
    print(f"  Control rate: {1.0/(control_decimation*simulation_dt):.1f}Hz")
    print(f"  Initial command: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, vyaw={cmd[2]:.2f}")

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    print(f"\nLoading policy...")
    try:
        policy = torch.jit.load(policy_path)
        policy.eval()

        # Reset LSTM memory if policy has recurrent layers
        if hasattr(policy, 'reset_memory'):
            policy.reset_memory()
            print("✓ Policy loaded successfully (LSTM memory reset)!")
        else:
            print("✓ Policy loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading policy: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("STARTING VISUALIZATION")
    print("="*60)
    print("  Close the viewer window or press ESC to exit")
    print("="*60 + "\n")

    # Initialize keyboard controller
    keyboard = KeyboardController()

    # Statistics tracking
    step_count = 0
    policy_inference_count = 0
    total_distance = 0.0
    last_pos = d.qpos[:2].copy()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        last_print_time = start

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # PD control
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau

            # Step physics
            mujoco.mj_step(m, d)
            step_count += 1

            counter += 1
            if counter % control_decimation == 0:
                policy_inference_count += 1

                # Create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)

                # Policy inference
                with torch.no_grad():
                    action = policy(obs_tensor).detach().numpy().squeeze()

                # Transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

                # Track distance traveled
                current_pos = d.qpos[:2].copy()
                total_distance += np.linalg.norm(current_pos - last_pos)
                last_pos = current_pos

            # Print statistics every 2 seconds
            current_time = time.time()
            if current_time - last_print_time >= 2.0:
                elapsed = current_time - start
                base_height = d.qpos[2]
                base_vel = np.linalg.norm(d.qvel[:2])

                print(f"[{elapsed:.1f}s] Height: {base_height:.3f}m | "
                      f"Velocity: {base_vel:.3f}m/s | "
                      f"Distance: {total_distance:.2f}m | "
                      f"Steps: {step_count} | "
                      f"Policy calls: {policy_inference_count}")

                last_print_time = current_time

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Final statistics
    elapsed_total = time.time() - start
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"  Duration: {elapsed_total:.1f}s")
    print(f"  Total steps: {step_count}")
    print(f"  Policy inferences: {policy_inference_count}")
    print(f"  Total distance: {total_distance:.2f}m")
    print(f"  Average velocity: {total_distance/elapsed_total:.3f}m/s")
    print("="*60 + "\n")
