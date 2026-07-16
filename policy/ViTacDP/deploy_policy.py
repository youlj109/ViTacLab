import sys
sys.path.append("/home/ubuntu/YzmSpace/new/ViTacLab")
from policy.ViTacDP.DP import DP
import numpy as np
import torch
from PIL import Image


def _quat_wxyz_to_rot6d(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norm, 1.0e-8)
    w, x, y, z = np.moveaxis(quat, -1, 0)

    two_s = 2.0
    matrix = np.empty((*quat.shape[:-1], 3, 3), dtype=quat.dtype)
    matrix[..., 0, 0] = 1.0 - two_s * (y * y + z * z)
    matrix[..., 0, 1] = two_s * (x * y - z * w)
    matrix[..., 0, 2] = two_s * (x * z + y * w)
    matrix[..., 1, 0] = two_s * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - two_s * (x * x + z * z)
    matrix[..., 1, 2] = two_s * (y * z - x * w)
    matrix[..., 2, 0] = two_s * (x * z - y * w)
    matrix[..., 2, 1] = two_s * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - two_s * (x * x + y * y)

    # Match PyTorch3D matrix_to_rotation_6d: flatten the first two rows.
    return matrix[..., :2, :].reshape(*quat.shape[:-1], 6)


class Encapsulation:
    def __init__(self, init_dict, num_envs):
        self.init_dict = init_dict
        self.num_envs = num_envs
        self.policy = DP_Policy(init_dict, num_envs)
        self.action_len = 4
        self.action_idx = 0
        self.actions = None

    def get_action(self, observation):
        actions = []
        if self.actions is None:
            self.actions = self.policy.get_action(observation)
        else:
            self.policy.update_obs(observation)
        for i in range(self.num_envs):
            actions.append(self.actions[i][self.action_idx])
        self.action_idx += 1
        if self.action_idx >= self.action_len:
            self.actions = None
            self.action_idx = 0
        return np.array(actions)

    def reset(self):
        self.policy = DP_Policy(self.init_dict, self.num_envs)
        self.action_idx = 0
        self.actions = None


class DP_Policy:
    def __init__(self, init_dict, num_envs):
        task_name = init_dict["task_name"]
        data_num = init_dict["data_num"]
        checkpoint_num = init_dict["checkpoint_num"]
        self.num_envs = num_envs
        self.policy = DP(
            task_name=task_name,
            data_num=data_num,
            checkpoint_num=checkpoint_num,
            num_envs=num_envs,
        )
    
    def pose_quat_to_pose6d(self, pose_array: np.ndarray) -> np.ndarray:
        pos = pose_array[:, :3]
        quat = pose_array[:, 3:7]
        rot6d = _quat_wxyz_to_rot6d(quat).astype(pose_array.dtype, copy=False)
        return np.concatenate([pos, rot6d], axis=-1)

    def encode_obs(self, observation):
        obs = [dict() for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            # obs[i]["head_cam"] = (
            #     np.moveaxis(observation[i]["third_person_camera"].numpy(), -1, -3) / 255.0
            # )
            # obs[i]["twist_cam"] = (
            #     np.moveaxis(observation[i]["twist_camera"].numpy(), -1, -3) / 255.0
            # )
            camera_array = []
            camera_pos_array = []
            for key in sorted(observation[i].keys()):
                if key.endswith("camera"):
                    camera_array.append(
                        np.moveaxis(observation[i][key].numpy(), -1, -3) / 255.0
                    )
                    camera_pos_array.append(
                        observation[i][key + "_pos"].numpy()
                    )
            obs[i]["camera"] = np.stack(camera_array, axis=0)
            obs[i]["camera_pos"] = self.pose_quat_to_pose6d(np.concatenate(camera_pos_array, axis=0))
            tactile_normal_force = observation[i]["tactile_normal_force"].numpy()
            tactile_shear_force = observation[i]["tactile_shear_force"].numpy()
            tactile_force = np.concatenate([tactile_normal_force, tactile_shear_force], axis=-1)
            obs[i]["tac_force"] = tactile_force
            tactile_rgb_image = observation[i]["tactile_rgb_image"].numpy()
            tactile_rgb_image = np.moveaxis(tactile_rgb_image, -1, -3) / 255.0
            obs[i]["tac_rgb"] = tactile_rgb_image
            tactile_pos = observation[i]["tactile_pos"].numpy()
            obs[i]["tac_pos"] = self.pose_quat_to_pose6d(tactile_pos)
            obs[i]["agent_pos"] = observation[i]["joint_pos"].numpy()
        return obs

    def get_action(self, observation):
        obs = self.encode_obs(observation)
        action = self.policy.get_action(obs)
        return action

    def update_obs(self, observation):
        obs = self.encode_obs(observation)
        self.policy.update_obs(obs)


def _npz_frame_to_policy_obs(data, t: int) -> list[dict]:
    """单环境一条观测，字段与仿真 `obs['record'][i]` 一致，供 `DP_Policy.encode_obs` 使用。"""
    rec = {
        "third_person_camera": torch.from_numpy(data["third_person_camera"][t]),
        "twist_camera": torch.from_numpy(data["twist_camera"][t]),
        "tactile_normal_force": torch.from_numpy(data["tactile_normal_force"][t]),
        "tactile_shear_force": torch.from_numpy(data["tactile_shear_force"][t]),
        "joint_pos": torch.from_numpy(data["joint_pos"][t]),
    }
    return [rec]


if __name__ == "__main__":
    npz_path = "/home/ubuntu/YzmSpace/new/ViTacLab/data/rsl_rl/Isaac-Forge-PegInsert-Direct-v0_42/2026-04-09_14-57-50/episode_1.npz"
    init_dict = {
        "task_name": "Isaac-Forge-PegInsert-Direct-v0_42randtwistv60",
        "data_num": 200,
        "checkpoint_num": 1500,
    }
    num_envs = 1
    enc = Encapsulation(init_dict, num_envs)
    enc.reset()

    z = np.load(npz_path)
    n_frames = int(z["joint_pos"].shape[0])
    use_action_key = "action" in z.files
    total_sse = 0.0
    n_elems = 0

    for t in range(n_frames - 1):
        obs_t = _npz_frame_to_policy_obs(z, t)
        pred = enc.get_action(obs_t)
        pred_vec = np.asarray(pred[0], dtype=np.float64)
        if use_action_key:
            gt = np.asarray(z["action"][t + 1], dtype=np.float64).reshape(-1)
        else:
            gt = np.asarray(z["joint_pos"][t + 1], dtype=np.float64).reshape(-1)
        m = min(pred_vec.size, gt.size)
        if m == 0:
            continue
        diff = pred_vec[:m] - gt[:m]
        total_sse += float(np.sum(diff * diff))
        n_elems += m

    z.close()
    overall_mse = total_sse / n_elems if n_elems > 0 else float("nan")
    print(
        f"MSE vs {'action' if use_action_key else 'joint_pos (next frame)'}: "
        f"{overall_mse:.6g} (SSE={total_sse:.6g}, n={n_elems})"
    )
