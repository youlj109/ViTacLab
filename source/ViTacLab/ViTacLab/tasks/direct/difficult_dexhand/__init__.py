import gymnasium as gym

##
# Register Gym environments for difficult dexterous tasks.
##


gym.register(
    id="Isaac-UR10eShadowHand-PourDeformable-Direct-v0",
    entry_point=f"{__name__}.ur10e_shadowhand_pour_env:UR10eShadowHandPourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_shadowhand_pour_env_cfg:UR10eShadowHandPourEnvCfg",
    },
)

