### BreakableObject 使用说明

#### 1. 相关文件

- **`breakable.py`**（当前目录）
  - 定义：
    - `BreakableObjectCfg`：可破坏物体的配置类
    - `BreakableObject`：运行时类（封装刚体 + contact sensor + 破坏逻辑）
- **`__init__.py`**（当前目录）
  - 导出：
    - `BreakableObject`, `BreakableObjectCfg`
- 示例环境：
  - `tasks/kitchen/loft_breakable_bi_cfg.py`
  - `tasks/kitchen/loft_breakable_bi.py`

---

#### 2. `BreakableObjectCfg` 配置说明

定义位置：`breakable.py`

- **`rigid_cfg: RigidObjectCfg`**  
  要被破坏的刚体对象配置，通常使用：
  - `prim_path`：刚体在 stage 中的路径，例如 `/World/Breakable/Object`
  - `spawn=UsdFileCfg(usd_path=...)`：指定 USD 资产路径
  - `init_state`：初始位姿（`pos`, `rot` 等）

- **`contact_cfg: ContactSensorCfg`**  
  挂在刚体上的 contact sensor，用来读取：
  - `net_forces_w`（净接触力）
  - `contact_pos_w`（接触点位置）

- **`mesh_root_path: Optional[str]`**  
  执行网格切割时的 mesh 根路径：  
  - 默认为 `rigid_cfg.prim_path`  
  - 若刚体 prim 下有子 mesh，可显式设置为那个 mesh 根

- **`break_force_threshold: float`**  
  触发破坏的力阈值（单位 N），基于网格上的净接触力模长。

- **`max_cuts_per_env: int`**  
  每个 env 在一个 episode 中，最多允许被切几次。

- **`success_min_cuts: int`**  
  至少被切多少次，才视为任务成功（`is_success()` 返回 True）。

- **`cut_direction_mode: str`**  
  切割平面的法向策略：
  - `"force"`：沿 net contact force 方向
  - `"opposite_force"`：沿反方向
  - `"world_z"`：使用世界系 `(0,0,1)` 方向

---

#### 3. `BreakableObject` 类说明

定义位置：`breakable.py`

##### 3.1 属性

- **`cfg: BreakableObjectCfg`**  
  本实例使用的配置。

- **`rigid: RigidObject`**  
  底层刚体对象，由 `cfg.rigid_cfg` 创建。

- **`contact: ContactSensor`**  
  底层接触传感器对象，由 `cfg.contact_cfg` 创建。

- **`_cut_counts: Optional[torch.Tensor]`**  
  尺寸 `[num_envs]`，记录每个 env 当前被切割次数，通过 `initialize_counters` 分配。

##### 3.2 公共方法

- **`register_to_scene(scene, rigid_name: str, contact_name: str)`**  
  - 作用：将内部 `rigid` 和 `contact` 注册到 `InteractiveScene` 中。  
  - 典型用法：  
    - `scene.rigid_objects[rigid_name] = self.rigid`  
    - `scene.sensors[contact_name] = self.contact`

- **`initialize_counters(num_envs: int, device)`**  
  - 作用：根据 env 数量和 device 初始化 `_cut_counts`。  
  - 必须在 env 创建后、使用前调用一次。

- **`step_breaking()`**  
  - 每个 sim step 调用一次。  
  - 流程：
    1. 从 `contact.data` 读取 `net_forces_w` 和 `contact_pos_w`。  
    2. 对每个 env：
       - 若该 env 的切割次数已达到 `max_cuts_per_env`，跳过。  
       - 找到力模最大的 body，若小于 `break_force_threshold`，跳过。  
       - 从对应 body 的 `contact_pos_w` 中取一个有效接触点。  
       - 根据 `cut_direction_mode` 选择切割法向（沿力 / 反向 / 世界 Z）。  
       - 调用 `cut_mesh_with_world_plane(mesh_root_path, plane_center, plane_normal)` 执行网格切割。  
       - 对应 env 的 `_cut_counts[env_id] += 1`。

- **`reset(env_ids: Sequence[int] | None = None)`**  
  - 作用：重置刚体到默认 root state，并把指定 env 的切割计数清零。  
  - 若 `env_ids` 为 `None`，默认对所有 env 生效。

- **`cut_counts` 属性**  
  - 返回当前每个 env 的切割次数（`torch.Tensor[num_envs]`）。

- **`is_success() -> torch.Tensor`**  
  - 返回 `bool` 张量（尺寸 `[num_envs]`），表示是否满足：  
    - `cut_counts >= success_min_cuts`。

---

#### 4. 在环境中使用示例（以 `LoftBreakableEnv` 为例）

##### 4.1 在 EnvCfg 里配置 `BreakableObjectCfg`

文件：`tasks/kitchen/loft_breakable_bi_cfg.py`

```python
from lehome.assets.object import BreakableObjectCfg

@configclass
class LoftBreakableEnvCfg(BaseEnvCfg):
    ...
    breakable_usd_path: str = (
        os.getcwd()
        + "/Assets/objects/Paper_Release/burger/Assets/Burger_Bread002/Burger_Bread002.usd"
    )

    breakable: BreakableObjectCfg = BreakableObjectCfg(
        rigid_cfg=RigidObjectCfg(
            prim_path="/World/Breakable/Object",
            spawn=sim_utils.UsdFileCfg(usd_path=breakable_usd_path),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-3.55, 5.2, 0.83),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        ),
        contact_cfg=ContactSensorCfg(
            prim_path="/World/Breakable/Object",
            update_period=0.0,
            history_length=4,
            track_contact_points=True,
            debug_vis=False,
            filter_prim_paths_expr=["/World/.*"],
        ),
        mesh_root_path="/World/Breakable/Object",
        break_force_threshold=50.0,
        max_cuts_per_env=1,
        success_min_cuts=1,
        cut_direction_mode="force",
    )
```

##### 4.2 在 Env 中实例化与注册

文件：`tasks/kitchen/loft_breakable_bi.py`

**构造函数中创建并初始化：**

```python
class LoftBreakableEnv(BaseEnv):
    ...
    def __init__(..., cfg: BaseEnvCfg | LoftBreakableEnvCfg, ...):
        super().__init__(cfg, render_mode, **kwargs)
        ...
        self.breakable = BreakableObject(self.cfg.breakable)
        self.breakable.initialize_counters(self.num_envs, self.device)
```

**`_setup_scene` 中注册到 scene：**

```python
def _setup_scene(self):
    super()._setup_scene()
    ...
    self.scene.articulations["left_arm"] = self.left_arm
    self.scene.articulations["right_arm"] = self.right_arm
    self.scene.sensors["top_camera"] = self.top_camera
    self.scene.sensors["left_camera"] = self.left_camera
    self.scene.sensors["right_camera"] = self.right_camera

    # 注册 breakable 对象到 scene
    self.breakable.register_to_scene(
        self.scene, rigid_name="breakable_object", contact_name="breakable_contact"
    )
```

##### 4.3 每个 step 中调用破坏逻辑

```python
def _apply_action(self) -> None:
    self.left_arm.set_joint_position_target(self.actions[:, :6])
    self.right_arm.set_joint_position_target(self.actions[:, 6:])

    # 使用 BreakableObject 执行“是否破坏 + 如何切割”的逻辑
    self.breakable.step_breaking()
```

##### 4.4 奖励与成功判定中使用

```python
def _get_rewards(self) -> torch.Tensor:
    if self.breakable.cut_counts is None:
        return torch.zeros((self.num_envs,), device=self.device)
    return (self.breakable.cut_counts > 0).float()

def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
    success_mask = self.breakable.is_success().to(device=self.device)
    return success_mask
```

##### 4.5 Reset 时重置可破坏物体

```python
def _reset_idx(self, env_ids: Sequence[int] | None):
    ...
    # 重置 breakable 对象 + 计数
    self.breakable.reset(env_ids)
```

---

#### 5. 在其他环境中复用的步骤

1. 在新的 `EnvCfg` 中添加一个或多个 `BreakableObjectCfg` 字段。  
2. 在 env 构造中实例化 `BreakableObject` 并调用 `initialize_counters`。  
3. 在 `_setup_scene` 中调用 `breakable.register_to_scene(...)` 注册到 scene。  
4. 在 `_apply_action` 中调用 `breakable.step_breaking()`。  
5. 在 `_get_rewards` / `_get_success` 中使用 `cut_counts` / `is_success()`。  
6. 在 `_reset_idx` 中调用 `breakable.reset(env_ids)` 重置状态。

