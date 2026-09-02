"""Run dependency-free static acceptance checks for the ViTacLab repository.

This audit intentionally does not import Isaac Lab, Isaac Sim, CUDA, camera, or
tactile packages.  It verifies the repository invariants that can be checked on
any machine: Python syntax, canonical Gym registrations, entry-point targets,
script documentation, argparse help, collector placement, stale/versioned
filenames, duplicate source files, obsolete policy paths, and the static
31-task TacSL/GelSight sensor/record manifest.

Examples:

    python scripts/audit_project.py
    python scripts/audit_project.py --root /workspace/ViTacLab --verbose

Exit status is zero only when every required check passes.  Warnings identify
items that require remote runtime validation but do not make the static audit
fail.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re
import sys
from typing import Any


VERSIONED_STEM_RE = re.compile(r"(?:^|[_-])v\d+$|(?:^|[_-])(copy|bak|backup|old|new)$", re.IGNORECASE)
OBSOLETE_TEXT = (
    "Diffusion_Policy_v6",
    "play_policy_v6.py",
    "Bi_Vi_Tac_Encoder_v2.py",
    "ViTacEncoder_bak.py",
    "scripts/rsl_rl/full_tra/",
    "scripts/rsl_rl/full_ik/play_full_ik_single.py",
    "scripts/rsl_rl/ik_rl/play_ik_rl_single.py",
    "scripts/debug/run_ur10e_shadowhand_single.py",
)

# Isaac Lab registers these upstream IDs before ViTacLab is imported. Reusing
# them causes Gym to silently replace the upstream spec based on import order,
# so every ViTacLab-customized environment must use its own collision-free ID.
UPSTREAM_TASK_ID_COLLISIONS = {
    "Isaac-Forge-PegInsert-Direct-v0",
    "Isaac-Forge-GearMesh-Direct-v0",
    "Isaac-Forge-NutThread-Direct-v0",
    "Isaac-Repose-Cube-Shadow-Direct-v0",
    "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0",
    "Isaac-Repose-Cube-Shadow-Vision-Direct-v0",
    "Isaac-Shadow-Hand-Over-Direct-v0",
}

# This exact Franka layer was found to contain itself as a sublayer.  USD
# composition then reports a cycle and Forge environments can stall while the
# stage is opening.  Keep the hash here so the dependency-free audit catches a
# regression even though binary USDC layers cannot be safely parsed without
# launching a USD/Isaac Sim runtime.
KNOWN_CORRUPT_ASSET_SHA256 = {
    "source/ViTacLab/ViTacLab/assets/data/Robots/Franka/Franka_R15/franka_mimic_edit.usd": (
        "0a5344b6e988dc2a3fe47d0ea36dd24e2a9feaed7288ae1815ec02244fb5456f"
    ),
}

# Project-level tactile acceptance manifest.  Every canonical task must create
# this many real TacSL/GelSight sensors when ``enable_cameras=True`` and expose
# the four canonical record fields.  Policy observation dimensions are a
# separate task/checkpoint contract and are intentionally not encoded here.
CANONICAL_TACTILE_SENSOR_COUNT = {
    "Isaac-ViTac-Forge-GearMesh-Breakable-Direct-v0": 2,
    "Isaac-ViTac-Forge-GearMesh-Direct-v0": 2,
    "Isaac-ViTac-Forge-NutThread-Breakable-Direct-v0": 2,
    "Isaac-ViTac-Forge-NutThread-Direct-v0": 2,
    "Isaac-ViTac-Forge-PegInsert-Breakable-Direct-v0": 2,
    "Isaac-ViTac-Forge-PegInsert-Direct-v0": 2,
    "Isaac-GelsightFinger-FrictionPretrain-Direct-v0": 1,
    "Isaac-GelsightFinger-MassPretrain-Direct-v0": 1,
    "Isaac-UR10eShadowHand-Repose-Cube-Direct-v0": 5,
    "Isaac-UR10eShadowHand-Repose-Cube-OpenAI-FF-Direct-v0": 5,
    "Isaac-UR10eShadowHand-Repose-Cube-Tactile-Direct-v0": 5,
    "Isaac-UR10eShadowHand-Repose-Cube-Vision-Direct-v0": 5,
    "Isaac-ViTac-Shadow-Hand-Over-Direct-v0": 10,
    "Isaac-UR10e-Dual-Shadow-Hand-BiBlindBinDrop-Direct-v0": 10,
    "Isaac-UR10e-Dual-Shadow-Hand-BiBlindGrasp-Direct-v0": 10,
    "Isaac-UR10e-Dual-Shadow-Hand-BiBlindInhand-Direct-v0": 10,
    "Isaac-UR10e-Dual-Shadow-Hand-BiBlindPeg-Direct-v0": 10,
    "Isaac-UR10e-Dual-Shadow-Hand-BiPeg-Direct-v0": 10,
    "Isaac-UR10e-Dual-Shadow-Hand-BiStab-Direct-v0": 10,
    "Isaac-UR10e-Dual-Shadow-Hand-Over-Direct-v0": 10,
    "Isaac-UR10e-Dual-Shadow-Hand-PourDeformable-Direct-v0": 10,
    "Isaac-UR10e-Dual-Shadow-Hand-UnscrewBottleCap-Direct-v0": 10,
    "Isaac-UR10eShadowHand-BlindClassification-Direct-v0": 5,
    "Isaac-UR10eShadowHand-BlindGrasp-Direct-v0": 5,
    "Isaac-UR10eShadowHand-BlindGraspReplay-Direct-v0": 5,
    "Isaac-UR10eShadowHand-BlindRetrieval-Direct-v0": 5,
    "Isaac-UR10eShadowHand-ForgeGearMesh-Direct-v0": 5,
    "Isaac-UR10eShadowHand-ForgeNutThread-Direct-v0": 5,
    "Isaac-UR10eShadowHand-ForgePegInsert-Direct-v0": 5,
    "Isaac-UR10eShadowHand-Pickup-Direct-v0": 5,
    "Isaac-UR10eShadowHand-PourDeformable-Direct-v0": 5,
}


class Audit:
    """Collect static audit errors, warnings, and informational counters."""

    def __init__(self, root: Path, verbose: bool = False):
        self.root = root
        self.verbose = verbose
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.info.append(message)


def _is_main_guard(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(isinstance(item, ast.Constant) and item.value == "__main__" for item in ast.walk(node.test))
    )


def _module_name(source_root: Path, path: Path) -> str:
    relative = path.relative_to(source_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_import_from(current_module: str, node: ast.ImportFrom) -> str:
    package = current_module if node.level == 0 else current_module.rsplit(".", node.level)[0]
    if node.module:
        return f"{package}.{node.module}" if node.level else node.module
    return package


def _eval_static(node: ast.AST | None, names: dict[str, Any]) -> Any:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _eval_static(node.value, names)
        return f"{base}.{node.attr}" if isinstance(base, str) else None
    if isinstance(node, ast.JoinedStr):
        chunks: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                chunks.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                rendered = _eval_static(value.value, names)
                if rendered is None:
                    return None
                chunks.append(str(rendered))
            else:
                return None
        return "".join(chunks)
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            resolved_key = _eval_static(key, names)
            if resolved_key is None:
                return None
            result[resolved_key] = _eval_static(value, names)
        return result
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval_static(item, names) for item in node.elts]
    return None


def _entry_target_exists(root: Path, entry: str) -> tuple[bool, str]:
    if not isinstance(entry, str) or ":" not in entry:
        return False, f"malformed entry point {entry!r}"
    module, symbol = entry.split(":", 1)
    candidates = [
        root / "source" / "ViTacLab" / Path(*module.split(".")),
        root / "source" / "video_teleop" / Path(*module.split(".")),
    ]
    module_path = None
    for base in candidates:
        py_file = base.with_suffix(".py")
        init_file = base / "__init__.py"
        if py_file.is_file():
            module_path = py_file
            break
        if init_file.is_file():
            module_path = init_file
            break
    if module_path is None:
        return False, f"module not found for {entry}"
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    except SyntaxError as exc:
        return False, f"cannot parse {module_path}: {exc}"
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    imported = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    if symbol not in defined and symbol not in imported:
        return False, f"symbol {symbol!r} not found in {module_path.relative_to(root)}"
    return True, ""


def audit_python(audit: Audit) -> None:
    python_files = [path for path in audit.root.rglob("*.py") if ".git" not in path.parts]
    executable_count = 0
    vitac_source_root = audit.root / "source" / "ViTacLab"
    export_cache: dict[Path, set[str]] = {}

    def _module_file(module_name: str) -> Path | None:
        base = vitac_source_root / Path(*module_name.split("."))
        if base.with_suffix(".py").is_file():
            return base.with_suffix(".py")
        if (base / "__init__.py").is_file():
            return base / "__init__.py"
        return None

    def _module_exports(module_path: Path) -> set[str]:
        cached = export_cache.get(module_path)
        if cached is not None:
            return cached
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        exports: set[str] = set()
        for statement in tree.body:
            if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                exports.add(statement.name)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                for alias in statement.names:
                    exports.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    exports.update(node.id for node in ast.walk(target) if isinstance(node, ast.Name))
            elif isinstance(statement, ast.AnnAssign):
                exports.update(node.id for node in ast.walk(statement.target) if isinstance(node, ast.Name))
        export_cache[module_path] = exports
        return exports

    for path in python_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            audit.error(f"Python parse failure: {audit.rel(path)}: {exc}")
            continue

        is_executable = any(_is_main_guard(node) for node in tree.body)
        is_env_module = path.name.endswith(("_env.py", "_env_cfg.py"))
        if is_executable:
            executable_count += 1
        if (is_executable or is_env_module) and ast.get_docstring(tree) is None:
            audit.error(f"Missing module usage/role docstring: {audit.rel(path)}")

        # Verify repository-local absolute imports at symbol level. Python
        # syntax compilation cannot detect a removed/reorganized exported name;
        # this catches regressions such as importing a task constant that no
        # longer exists after consolidation without importing Isaac Sim.
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 0 or not node.module:
                continue
            if not node.module.startswith("ViTacLab."):
                continue
            module_path = _module_file(node.module)
            if module_path is None:
                audit.error(
                    f"Missing repository import module: {audit.rel(path)}:{node.lineno}: {node.module}"
                )
                continue
            exports = _module_exports(module_path)
            module_base = vitac_source_root / Path(*node.module.split("."))
            for alias in node.names:
                if alias.name == "*" or alias.name in exports:
                    continue
                # ``from package import child_module`` is legal even when the
                # package __init__ does not explicitly re-export the child.
                child_file = module_base / f"{alias.name}.py"
                child_init = module_base / alias.name / "__init__.py"
                if child_file.is_file() or child_init.is_file():
                    continue
                audit.error(
                    "Missing repository import symbol: "
                    f"{audit.rel(path)}:{node.lineno}: {node.module}.{alias.name}"
                )

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("ViTacLab.")
                and ":" in node.value
            ):
                ok, reason = _entry_target_exists(audit.root, node.value)
                if not ok:
                    audit.error(f"Invalid static module:Symbol reference: {audit.rel(path)}:{node.lineno}: {reason}")
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue
            if not any(keyword.arg == "help" for keyword in node.keywords):
                audit.error(f"argparse option without help=: {audit.rel(path)}:{node.lineno}")

    audit.note(f"Parsed {len(python_files)} Python modules; {executable_count} executable entries.")


def audit_registrations(audit: Audit) -> list[dict[str, str]]:
    source_root = audit.root / "source" / "ViTacLab"
    registration_root = source_root / "ViTacLab" / "tasks" / "direct"
    registrations: list[dict[str, str]] = []
    for path in registration_root.rglob("__init__.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = _module_name(source_root, path)
        names: dict[str, Any] = {"__name__": module}

        for statement in tree.body:
            if isinstance(statement, ast.ImportFrom):
                imported_module = _resolve_import_from(module, statement)
                for alias in statement.names:
                    alias_name = alias.asname or alias.name
                    names[alias_name] = f"{imported_module}.{alias.name}" if statement.module else f"{imported_module}.{alias.name}"
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    names[alias.asname or alias.name.split(".")[0]] = alias.name
            elif isinstance(statement, ast.Assign):
                value = _eval_static(statement.value, names)
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        names[target.id] = value

            calls = [statement] if isinstance(statement, ast.Expr) else []
            for expr in calls:
                call = expr.value
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "register"
                ):
                    continue
                keywords = {keyword.arg: keyword.value for keyword in call.keywords}
                task_id = _eval_static(keywords.get("id"), names)
                entry = _eval_static(keywords.get("entry_point"), names)
                kwargs = _eval_static(keywords.get("kwargs"), names) or {}
                cfg = kwargs.get("env_cfg_entry_point") if isinstance(kwargs, dict) else None
                registrations.append(
                    {"id": task_id, "entry": entry, "cfg": cfg, "file": audit.rel(path)}
                )

    ids = Counter(item["id"] for item in registrations)
    for task_id, count in ids.items():
        if task_id is None:
            audit.error("A Gym registration has a statically unresolved id.")
        elif task_id in UPSTREAM_TASK_ID_COLLISIONS:
            audit.error(f"ViTacLab Gym id collides with an upstream Isaac Lab task: {task_id}")
        elif count > 1:
            audit.error(f"Duplicate Gym task id ({count}): {task_id}")

    pairs: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for item in registrations:
        if not isinstance(item["entry"], str) or not isinstance(item["cfg"], str):
            audit.error(f"Unresolved env/config entry for {item['id']} in {item['file']}")
            continue
        pairs[(item["entry"], item["cfg"])].append(item["id"])
        for label in ("entry", "cfg"):
            ok, reason = _entry_target_exists(audit.root, item[label])
            if not ok:
                audit.error(f"Invalid {label} for {item['id']}: {reason}")
    for pair, task_ids in pairs.items():
        if len(task_ids) > 1:
            audit.error(f"Multiple task ids share the same env/config pair {pair}: {task_ids}")

    audit.note(f"Found {len(registrations)} unique canonical Gym registrations.")
    return sorted(registrations, key=lambda item: item["id"] or "")


def audit_tactile_contract(audit: Audit, registrations: list[dict[str, str]]) -> None:
    """Enforce the static half of the 31/31 camera/TacSL record contract."""

    registered_ids = {item["id"] for item in registrations if isinstance(item["id"], str)}
    manifested_ids = set(CANONICAL_TACTILE_SENSOR_COUNT)
    missing_manifest = sorted(registered_ids - manifested_ids)
    stale_manifest = sorted(manifested_ids - registered_ids)
    if missing_manifest:
        audit.error(f"Canonical tasks missing tactile sensor-count manifest entries: {missing_manifest}")
    if stale_manifest:
        audit.error(f"Tactile sensor-count manifest contains unregistered tasks: {stale_manifest}")
    if len(registered_ids) != 31 or len(manifested_ids) != 31:
        audit.error(
            f"Tactile acceptance requires exactly 31 canonical tasks; "
            f"registered={len(registered_ids)} manifested={len(manifested_ids)}"
        )

    required_source_tokens: dict[str, tuple[str, ...]] = {
        "scripts/common/sensor_diagnostics.py": (
            "_REQUIRED_TACTILE_RECORD_KEYS",
            "_build_record_dict",
            "[SENSOR-DIAG-PASS]",
            "tactile RGB is entirely zero",
            "tactile RGB is constant",
        ),
        "source/ViTacLab/ViTacLab/assets/robot/ur10e_shadowhand_direct_base_single/ur10e_shadowhand_direct_base_env.py": (
            '"tactile_pos"',
            '"tactile_normal_force"',
            '"tactile_shear_force"',
            '"tactile_rgb_image"',
            "physical fingertip",
        ),
        "source/ViTacLab/ViTacLab/assets/robot/ur10e_dual_shadowhand_direct_base/ur10e_dual_shadowhand_direct_base_env.py": (
            "_expected_tactile_sensor_names",
            '"tactile_pos"',
            '"tactile_normal_force"',
            '"tactile_shear_force"',
            '"tactile_rgb_image"',
        ),
        "source/ViTacLab/ViTacLab/tasks/direct/simple_gripper/forge_env.py": (
            "_expected_tactile_sensor_names",
            '"tactile_pos"',
            '"tactile_normal_force"',
            '"tactile_shear_force"',
            '"tactile_rgb_image"',
        ),
        "source/ViTacLab/ViTacLab/tasks/direct/pretraining/gelsight_finger_pretrain_base_env.py": (
            "_expected_tactile_sensor_names",
            "def _build_record_dict",
            '"tactile_pos"',
            '"tactile_rgb_image"',
        ),
        "source/ViTacLab/ViTacLab/assets/sensor/shadow_hand_tacsl.py": (
            "build_shadow_hand_tacsl_sensor_cfgs",
            "build_shadow_hand_tactile_record",
            '"tactile_pos"',
            '"tactile_normal_force"',
            '"tactile_shear_force"',
            '"tactile_rgb_image"',
        ),
        "source/ViTacLab/ViTacLab/tasks/direct/simple_dexhand/shadow_hand/shadow_hand_vision_env.py": (
            "ShadowHandSceneCfg",
            "def _build_record_dict",
            "initialize_tacsl_nominal_render",
        ),
        "source/ViTacLab/ViTacLab/tasks/direct/simple_dexhand/shadow_hand_over/shadow_hand_over_env.py": (
            "build_shadow_hand_tacsl_sensor_cfgs",
            "shadow_hand_tacsl_sensor_keys(\"right_\")",
            "shadow_hand_tacsl_sensor_keys(\"left_\")",
            "def _build_record_dict",
        ),
        "source/ViTacLab/ViTacLab/tasks/direct/medium_dexhand/bi_peg/bi_peg_env_cfg.py": (
            "class UR10eDualShadowHandBiPegSceneCfg",
            '"/World/envs/env_.*/(hole|peg)"',
            "scene: UR10eDualShadowHandBiPegSceneCfg",
        ),
        "source/ViTacLab/ViTacLab/tasks/direct/difficult_dexhand/bi_blind_grasp/bi_blind_grasp_env_cfg.py": (
            "UR10eDualShadowHandBiPegSceneCfg",
            "scene: UR10eDualShadowHandBiPegSceneCfg",
        ),
        "source/ViTacLab/ViTacLab/tasks/direct/difficult_dexhand/bi_blind_peg/bi_blind_peg_env_cfg.py": (
            "UR10eDualShadowHandBiPegSceneCfg",
            "scene: UR10eDualShadowHandBiPegSceneCfg",
        ),
    }
    for relative, tokens in required_source_tokens.items():
        path = audit.root / relative
        if not path.is_file():
            audit.error(f"Missing tactile-contract implementation file: {relative}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in tokens:
            if token not in text:
                audit.error(f"Missing tactile-contract marker {token!r}: {relative}")

    profile_counts = Counter(CANONICAL_TACTILE_SENSOR_COUNT.values())
    audit.note(
        "Tactile manifest covers 31/31 tasks "
        f"(1-sensor={profile_counts[1]}, 2-sensor={profile_counts[2]}, "
        f"5-sensor={profile_counts[5]}, 10-sensor={profile_counts[10]})."
    )


def audit_layout_and_duplicates(audit: Audit) -> None:
    for path in audit.root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() in {".py", ".sh"} and VERSIONED_STEM_RE.search(path.stem):
            audit.error(f"Version/backup-suffixed executable filename: {audit.rel(path)}")

    scripts_root = audit.root / "scripts"
    canonical_collectors = scripts_root / "data_collection"
    for path in scripts_root.rglob("*.py"):
        if canonical_collectors in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\b(?:np|numpy)\.savez(?:_compressed)?\s*\(", text):
            audit.error(f"Dataset-writing script outside scripts/data_collection: {audit.rel(path)}")

    # Diffusion Policy and ViTacDP intentionally own separate copies of their
    # framework code. Detect duplicates within each maintained area, while
    # allowing common algorithms to exist once in each independent policy.
    duplicate_scopes = (
        audit.root / "scripts",
        audit.root / "source" / "ViTacLab",
        audit.root / "policy" / "Diffusion_Policy",
        audit.root / "policy" / "ViTacDP",
    )
    for base in duplicate_scopes:
        digest_map: defaultdict[tuple[str, int], list[Path]] = defaultdict(list)
        for path in base.rglob("*.py"):
            data = path.read_bytes()
            if len(data) < 256 or path.name == "__init__.py":
                continue
            digest_map[(hashlib.sha256(data).hexdigest(), len(data))].append(path)
        for paths in digest_map.values():
            if len(paths) > 1:
                audit.error("Byte-for-byte duplicate Python modules: " + ", ".join(audit.rel(path) for path in paths))

    searchable_suffixes = {".py", ".md", ".yaml", ".yml", ".toml", ".sh", ".json"}
    for path in audit.root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in searchable_suffixes or ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in OBSOLETE_TEXT:
            if token in text:
                audit.error(f"Obsolete path/name reference {token!r}: {audit.rel(path)}")

    shared_core = audit.root / "policy" / "dp_core"
    if shared_core.exists():
        audit.error("Shared policy/dp_core is forbidden; Diffusion Policy and ViTacDP must remain independent.")

    independent_cores = {
        "Diffusion_Policy": audit.root / "policy" / "Diffusion_Policy" / "diffusion_policy_core",
        "ViTacDP": audit.root / "policy" / "ViTacDP" / "vitacdp_core",
    }
    for policy_name, core in independent_cores.items():
        if not core.is_dir():
            audit.error(f"Missing independent {policy_name} core: {audit.rel(core)}")
        elif core.is_symlink():
            audit.error(f"Independent {policy_name} core must not be a symlink: {audit.rel(core)}")
    if all(core.is_dir() for core in independent_cores.values()):
        if independent_cores["Diffusion_Policy"].resolve() == independent_cores["ViTacDP"].resolve():
            audit.error("Diffusion Policy and ViTacDP resolve to the same core directory.")
        audit.note("Found two physically independent policy cores (Diffusion Policy and ViTacDP).")

    forbidden_cross_imports = {
        "Diffusion_Policy": "vitacdp_core",
        "ViTacDP": "diffusion_policy_core",
    }
    for policy_name, forbidden_name in forbidden_cross_imports.items():
        policy_root = audit.root / "policy" / policy_name
        for path in policy_root.rglob("*.py"):
            if forbidden_name in path.read_text(encoding="utf-8", errors="replace"):
                audit.error(
                    f"{policy_name} imports the other policy core {forbidden_name!r}: {audit.rel(path)}"
                )

    reference_pattern = re.compile(
        r"(?<![\w./-])(scripts/data_collection/[A-Za-z0-9_./-]+\.(?:ya?ml|json))(?![\w./-])"
    )
    for path in (audit.root / "scripts" / "data_collection").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for relative in reference_pattern.findall(text):
            if not (audit.root / relative).is_file():
                audit.error(f"Missing data-collection config reference {relative!r}: {audit.rel(path)}")


def audit_assets(audit: Audit) -> None:
    """Check dependency-free asset invariants and known broken binary layers."""

    fixed_base_cfg = (
        audit.root
        / "source/ViTacLab/ViTacLab/assets/robot/ur10e_shadowhand_direct_base_single/ur10e_shadowhand_direct_base_cfg.py"
    )
    if not fixed_base_cfg.is_file():
        audit.error(f"Missing fixed-base UR10e configuration: {audit.rel(fixed_base_cfg)}")
    elif not re.search(r"\bfix_root_link\s*=\s*True\b", fixed_base_cfg.read_text(encoding="utf-8")):
        audit.error("Canonical tabletop UR10e/ShadowHand configuration must set fix_root_link=True.")

    for relative, corrupt_digest in KNOWN_CORRUPT_ASSET_SHA256.items():
        path = audit.root / relative
        if not path.is_file():
            audit.error(f"Missing required asset: {relative}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == corrupt_digest:
            audit.error(f"Known self-referencing/cyclic USD asset: {relative}")

    # Text USDA layers can be checked without importing pxr.  Binary USDC
    # files are covered by explicit known-bad hashes and remote runtime tests.
    for path in audit.root.rglob("*.usd"):
        if ".git" in path.parts:
            continue
        data = path.read_bytes()
        if not data.lstrip().startswith(b"#usda"):
            continue
        text = data.decode("utf-8", errors="replace")
        if f"./{path.name}" in text:
            audit.error(f"Text USD layer references itself: {audit.rel(path)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dependency-free ViTacLab repository acceptance checks.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="ViTacLab repository root (default: inferred from this script).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print all informational audit counters.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "source" / "ViTacLab").is_dir() or not (root / "scripts").is_dir():
        raise SystemExit(f"Not a ViTacLab repository root: {root}")

    audit = Audit(root, verbose=args.verbose)
    audit_python(audit)
    registrations = audit_registrations(audit)
    audit_tactile_contract(audit, registrations)
    audit_layout_and_duplicates(audit)
    audit_assets(audit)

    if args.verbose:
        for message in audit.info:
            print(f"[INFO] {message}")
    for message in audit.warnings:
        print(f"[WARNING] {message}")
    for message in audit.errors:
        print(f"[ERROR] {message}")
    print(
        f"[SUMMARY] errors={len(audit.errors)} warnings={len(audit.warnings)} "
        f"checks_info={len(audit.info)}"
    )
    return 1 if audit.errors else 0


if __name__ == "__main__":
    sys.exit(main())
