import gymnasium as gym

from ViTacLab.tasks.direct.simple_gripper import agents as simple_agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Forge-PegInsert-Breakable-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeBreakableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskPegInsertBreakableCfg",
        "rl_games_cfg_entry_point": f"{simple_agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{simple_agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Forge-GearMesh-Breakable-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeBreakableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskGearMeshBreakableCfg",
        "rl_games_cfg_entry_point": f"{simple_agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{simple_agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Forge-NutThread-Breakable-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeBreakableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskNutThreadBreakableCfg",
        "rl_games_cfg_entry_point": f"{simple_agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
        "rsl_rl_cfg_entry_point": f"{simple_agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

