import gymnasium as gym

from .shadow_hand import agents as shadow_hand_agents

##
# Register Gym environments for simple dexterous direct-control tasks.
##

# UR10e + ShadowHand pickup with DexCube and TacSL tactile sensing
gym.register(
    id="Isaac-UR10eShadowHand-Pickup-Direct-v0",
    entry_point=f"{__name__}.hand_pickup.hand_pickup_env:UR10eShadowHandPickupEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hand_pickup.hand_pickup_env_cfg:UR10eShadowHandPickupEnvCfg",
        # RSL-RL PPO config (tactile-aware) reused from ShadowHand tactile task
        "rsl_rl_cfg_entry_point": f"{shadow_hand_agents.__name__}.rsl_rl_ppo_cfg:ShadowHandTactilePPORunnerCfg",
    },
)

