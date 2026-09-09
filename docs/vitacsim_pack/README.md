# ViTacSim Pack Manifest

本目录定义 **可移植 ViTacSim 打包** 的文件清单与说明。

## 一键打包

在 ViTacLab 仓库根目录：

```bash
bash bash_command/vitacsim_pack.sh
```

输出：

- `dist/vitacsim_pack_YYYYMMDD_HHMMSS/` —  staging 目录
- `dist/vitacsim_pack_YYYYMMDD_HHMMSS.tar.gz` — 可拷贝到新机器的压缩包
- 包内 `README_PACK.md` — 新机器安装步骤
- 包内 `MANIFEST.json` — 文件列表与 SHA256

## 可选参数

```bash
# 只 staging，不压缩（便于检查内容）
SKIP_TAR=1 bash bash_command/vitacsim_pack.sh

# 打包整个 source/ViTacLab（仍排除 Scene/ + Objects/）
INCLUDE_FULL_SOURCE=1 bash bash_command/vitacsim_pack.sh

# 自定义输出路径
OUT_DIR=/tmp/vitac_pack ARCHIVE=/tmp/vitac_pack.tar.gz bash bash_command/vitacsim_pack.sh
```

## 清单文件

- [`PATHS.include`](PATHS.include) — 默认打包路径（相对 repo 根）

修改 sensor / demo / task 路径后，请同步更新 `PATHS.include` 并重新打包。

## 相关文档

| 文档 | 用途 |
|------|------|
| [`../VITACSIM_PRINCIPLES.md`](../VITACSIM_PRINCIPLES.md) | **原理说明**（架构、数据流、集成） |
| [`../VITACSIM_PHYSX_VALIDATION.md`](../VITACSIM_PHYSX_VALIDATION.md) | PhysX 验证与 no-fallback 说明 |
| [`../../README_send.md`](../../README_send.md) | 对外模块摘要 |

## 体积预估

| 组件 | 约 |
|------|-----|
| Sensors USD/meshes | ~110 MB |
| ShadowHand + Franka GelSight | ~140 MB |
| Python 代码 + scripts + docs | ~5 MB |
| **合计（不含 Scene/Objects）** | **~250 MB** 压缩后更小 |

## 新机器最小验证

```bash
bash bash_command/visuotactile_physx_alignment_check.sh
bash bash_command/forge_tactile_feedback_insert_demo.sh
```

两者均通过即可认为 ViTacSim 核心链路可用。
