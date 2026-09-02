import sys
sys.path.append("/home/ubuntu/YzmSpace/new/ViTacLab")
from policy.Diffusion_Policy.DP import DP
import numpy as np
import torch
from PIL import Image

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
        return actions

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
        self.policy = DP(task_name=task_name, data_num=data_num, checkpoint_num=checkpoint_num, num_envs=num_envs)
        self._expects_twist_cam = False
        try:
            normalizer = getattr(getattr(self.policy, "policy", None), "normalizer", None)
            params_dict = getattr(normalizer, "params_dict", None)
            if params_dict is not None:
                self._expects_twist_cam = ("twist_cam" in list(params_dict.keys()))
        except Exception:
            self._expects_twist_cam = False

    def encode_obs(self, observation):
        obs = [dict() for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            obs[i]["head_cam"] = np.moveaxis(observation[i]["third_person_camera"].numpy(), -1, -3) / 255.0
            if self._expects_twist_cam:
                if "twist_camera" in observation[i]:
                    obs[i]["twist_cam"] = np.moveaxis(observation[i]["twist_camera"].numpy(), -1, -3) / 255.0
                else:
                    # Keep backward compatibility for checkpoints trained with twist_cam.
                    obs[i]["twist_cam"] = np.moveaxis(np.random.rand(480, 640, 3), -1, -3) / 255.0
            # obs["head_cam"] = np.moveaxis(np.random.rand(480, 640, 3), -1, -3) / 255.0
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
    """单环境观测：仅相机与关节状态（无触觉），供 `DP_Policy.encode_obs` 使用。"""
    rec = {
        "third_person_camera": torch.from_numpy(data["third_person_camera"][t]),
        # "twist_camera": torch.from_numpy(data["twist_camera"][t]),
        "joint_pos": torch.from_numpy(data["joint_pos"][t]),
    }
    return [rec]


if __name__ == "__main__":
    npz_path = "/home/glzn/new/xsq/ViTacLab/play_records/full_tra_v3_pickup/episode_0000.npz"
    init_dict = {
        "task_name": "Isaac-UR10eShadowHand-Pickup-Direct-v1",
        "data_num": 200,
        "checkpoint_num": 2500,
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