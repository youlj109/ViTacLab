from policy.Diffusion_Policy.DP import DP
import numpy as np
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
        self.actions = [None for _ in range(self.num_envs)]

class DP_Policy:
    def __init__(self, init_dict, num_envs):
        task_name = init_dict["task_name"]
        data_num = init_dict["data_num"]
        checkpoint_num = init_dict["checkpoint_num"]
        self.num_envs = num_envs
        self.policy = DP(task_name=task_name, data_num=data_num, checkpoint_num=checkpoint_num, num_envs=num_envs)

    def encode_obs(self, observation):
        obs = [dict() for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            obs[i]["head_cam"] = np.moveaxis(observation[i]["third_person_camera"].numpy(), -1, -3) / 255.0
            # obs[i]["twist_cam"] = np.moveaxis(observation[i]["twist_camera_rgb"].numpy(), -1, -3) / 255.0
            if "twist_camera" in observation[i]:
                obs[i]["twist_cam"] = np.moveaxis(observation[i]["twist_camera"].numpy(), -1, -3) / 255.0
            else:
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
        
if __name__ == "__main__":
    init_dict = {
        "task_name": "Isaac-Forge-PegInsert-Direct-v0",
        "data_num": 200,
        "checkpoint_num": 1500,
    }
    num_envs = 1
    policy = Encapsulation(init_dict, num_envs)
    while True:
        a = 1