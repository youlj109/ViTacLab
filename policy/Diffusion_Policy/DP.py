import sys
import os
import torch
import hydra
import dill
from policy.Diffusion_Policy.diffusion_policy.workspace.robotworkspace import RobotWorkspace
from policy.Diffusion_Policy.diffusion_policy.common.pytorch_util import dict_apply
from policy.Diffusion_Policy.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from policy.Diffusion_Policy.diffusion_policy.env_runner.dp_runner import DPRunner


def _strip_module_prefix_from_state_dict(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


class DP:
    def __init__(self, task_name, checkpoint_num: int, data_num: int, num_envs: int):
        print(f'checkpoints/{task_name}_{data_num}/{checkpoint_num}.ckpt')
        self.policy = get_policy(f'checkpoints/{task_name}_{data_num}/{checkpoint_num}.ckpt', None, 'cuda:0')
        self.num_envs = num_envs
        self.runner = [DPRunner(output_dir=None) for _ in range(num_envs)]

    def update_obs(self, observation):
        for i in range(self.num_envs):
            self.runner[i].update_obs(observation[i])
    
    def get_action(self, observation):
        device, dtype = self.policy.device, self.policy.dtype
        first_obs = None
        for obs_i in observation:
            if obs_i is not None:
                first_obs = obs_i
                break
        if first_obs is None:
            raise RuntimeError("All observations are None in DP.get_action().")
        obs_keys = [k for k in ("head_cam", "twist_cam", "agent_pos") if k in first_obs]
        obs_dict_input_all = {k: [] for k in obs_keys}
        for i in range(self.num_envs):
            
            if observation[i] is not None:
                self.runner[i].obs.append(observation[i])  # update
            obs = self.runner[i].get_n_steps_obs()

            # create obs dict
            np_obs_dict = dict(obs)
            # device transfer
            obs_dict = dict_apply(
                np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
            )
            
            for k in obs_keys:
                if k not in obs_dict:
                    raise KeyError(f"Missing key '{k}' in runner observation for env {i}.")
                obs_dict_input_all[k].append(obs_dict[k])
            
        for k in obs_keys:
            obs_dict_input_all[k] = torch.stack(obs_dict_input_all[k], dim=0)
        action_dict = self.policy.predict_action(obs_dict_input_all)

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
        actions = np_action_dict["action"]
        return actions

    def get_last_obs(self):
        return [self.runner[i].obs[-1] for i in range(self.num_envs)]

    def reset_obs(self):
        for i in range(self.num_envs):
            self.runner[i].reset_obs()
    
def get_policy(checkpoint, output_dir, device):
    # load checkpoint
    payload = torch.load(open('./policy/Diffusion_Policy/'+checkpoint, 'rb'), pickle_module=dill)
    if "state_dicts" in payload and isinstance(payload["state_dicts"], dict):
        for key, value in payload["state_dicts"].items():
            payload["state_dicts"][key] = _strip_module_prefix_from_state_dict(value)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy