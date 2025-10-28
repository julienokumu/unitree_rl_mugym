import torch
from torch import Tensor
import numpy as np
from typing import Tuple

# Isaac Gym utilities (optional - provide fallbacks if not available)
try:
    from isaacgym.torch_utils import quat_apply, normalize
    ISAAC_GYM_AVAILABLE = True
except ImportError:
    ISAAC_GYM_AVAILABLE = False

    # Fallback implementations
    def normalize(x):
        """Normalize tensor along last dimension"""
        return x / torch.norm(x, dim=-1, keepdim=True)

    def quat_apply(quat, vec):
        """Apply quaternion rotation to vector"""
        # quat: (N, 4) in [w, x, y, z] format
        # vec: (N, 3)
        quat = quat.contiguous()
        vec = vec.contiguous()

        w, x, y, z = quat[:, 0:1], quat[:, 1:2], quat[:, 2:3], quat[:, 3:4]
        vx, vy, vz = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]

        # Quaternion multiplication: q * v * q^-1
        t_x = 2.0 * (y * vz - z * vy)
        t_y = 2.0 * (z * vx - x * vz)
        t_z = 2.0 * (x * vy - y * vx)

        result_x = vx + w * t_x + (y * t_z - z * t_y)
        result_y = vy + w * t_y + (z * t_x - x * t_z)
        result_z = vz + w * t_z + (x * t_y - y * t_x)

        return torch.cat([result_x, result_y, result_z], dim=-1)

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower