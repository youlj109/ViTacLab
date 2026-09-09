import torch
import hydra
import dill
from policy.ViTacDP.diffusion_policy.workspace.robotworkspace import RobotWorkspace
from policy.ViTacDP.diffusion_policy.common.pytorch_util import dict_apply
from policy.ViTacDP.diffusion_policy.env_runner.dp_runner import DPRunner


def _strip_module_prefix_from_state_dict(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


class DP:
    def __init__(self, task_name, checkpoint_num: int, data_num: int, num_envs: int = 1):
        self.policy, self.cfg = get_policy(
            f"checkpoints/{task_name}_{data_num}/{checkpoint_num}.ckpt",
            None,
            "cuda:0",
        )
        self.num_envs = num_envs
        self.runner = [DPRunner(output_dir=None) for _ in range(num_envs)]
        self.tac_type = self.cfg["task"]["shape_meta"]["obs"]["tac"]["type"]
        self.camera_num = self.cfg["task"]["camera_num"]
        self.tac_num = self.cfg["task"]["tac_num"]

    def update_obs(self, observation):
        for i in range(self.num_envs):
            self.runner[i].update_obs(observation[i])

    def get_action(self, observation):
        device = self.policy.device
        obs_dict_input_all = dict()
        obs_dict_input_all["agent_pos"] = []
        assert self.camera_num == len(observation[0]["camera"]), "camera_num mismatch"
        assert self.tac_num == len(observation[0][self.tac_type]), "tac_num mismatch"
        for i in range(self.camera_num):
            obs_dict_input_all[f"camera_{i}"] = []
            obs_dict_input_all[f"camera_pos_{i}"] = []
        for i in range(self.tac_num):
            obs_dict_input_all[f"tac_{i}"] = []
            obs_dict_input_all[f"tac_pos_{i}"] = []
        for i in range(self.num_envs):
            if observation[i] is not None:
                self.runner[i].obs.append(observation[i])
            obs = self.runner[i].get_n_steps_obs()
            for key in obs.keys():
                print("obs", key, obs[key].shape)
            np_obs_dict = dict(obs)
            obs_dict = dict_apply(
                np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
            )
            for i in range(self.camera_num):
                obs_dict_input_all[f"camera_{i}"].append(obs_dict["camera"][:, i])
                obs_dict_input_all[f"camera_pos_{i}"].append(obs_dict["camera_pos"][:, i])
            for i in range(self.tac_num):
                obs_dict_input_all[f"tac_{i}"].append(obs_dict[self.tac_type][:, i])
                obs_dict_input_all[f"tac_pos_{i}"].append(obs_dict["tac_pos"][:, i])
            obs_dict_input_all["agent_pos"].append(obs_dict["agent_pos"])
        for key in obs_dict_input_all.keys():
            obs_dict_input_all[key] = torch.stack(obs_dict_input_all[key], dim=0)
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict_input_all)
        np_action_dict = dict_apply(
            action_dict, lambda x: x.detach().to("cpu").numpy()
        )
        actions = np_action_dict["action"]
        return actions

    def get_last_obs(self):
        return [self.runner[i].obs[-1] for i in range(self.num_envs)]

    def reset_obs(self):
        for i in range(self.num_envs):
            self.runner[i].reset_obs()


def get_policy(checkpoint, output_dir, device):
    # load checkpoint
    payload = torch.load(open("./policy/ViTacDP/" + checkpoint, "rb"), pickle_module=dill)
    if "state_dicts" in payload and isinstance(payload["state_dicts"], dict):
        for key, value in payload["state_dicts"].items():
            payload["state_dicts"][key] = _strip_module_prefix_from_state_dict(value)
    cfg = payload["cfg"]
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

    return policy, cfg
