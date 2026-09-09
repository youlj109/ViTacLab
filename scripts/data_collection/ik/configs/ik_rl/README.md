# IK-RL 配置说明

这里保存 `play_ik_policy.py` 和 IK-RL 训练入口共同使用的唯一配置版本。
数据采集入口位于 `scripts/data_collection/ik/`，训练入口仅复用这些控制参数，
不再在 `scripts/rsl_rl` 下保存第二份配置。

- `ik_rl_pickup.yaml`：`Isaac-UR10eShadowHand-Pickup-Direct-v0`。
- `ik_rl_pour.yaml`：`Isaac-UR10eShadowHand-PourDeformable-Direct-v0`。

命令行显式传入的同名参数优先于 YAML。`--ik-config none` 表示不加载配置。
字段含义和完整命令参见 `scripts/data_collection/README.md`。
