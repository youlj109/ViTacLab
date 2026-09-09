#!/usr/bin/env python3
"""汇总 play_policy 写入的 validation 结果（all_success.txt）。

目录结构示例::
    data/validation/<task>/<policy>_<observation_profile>/{data_num}_{ckpt}_seed{seed}/all_success.txt

每个 all_success.txt 中每行对应一个并行 env 在本轮 global episode 是否成功（True/False）。
num_envs=20 时，每 20 行通常为一轮；若重复运行同一目录会追加行。

用法（仓库根目录）::
    python scripts/policy/summarize_validation.py \\
        data/validation/Isaac-UR10eShadowHand-Pickup-Direct-v0/ViTacDP_force

    python scripts/policy/summarize_validation.py \\
        data/validation/Isaac-UR10eShadowHand-Pickup-Direct-v0/ViTacDP_force \\
        --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

RUN_DIR_RE = re.compile(r"^(?P<data_num>\d+)_(?P<ckpt>\d+)_seed(?P<seed>\d+)$")


@dataclass(frozen=True)
class RunResult:
    data_num: int
    ckpt: int
    seed: int
    n_trials: int
    n_success: int

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_trials if self.n_trials else 0.0


def _parse_bool(line: str) -> bool | None:
    s = line.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    return None


def load_run_result(run_dir: Path) -> RunResult | None:
    m = RUN_DIR_RE.match(run_dir.name)
    if m is None:
        return None

    success_file = run_dir / "all_success.txt"
    if not success_file.is_file():
        return None

    flags: list[bool] = []
    for line in success_file.read_text(encoding="utf-8").splitlines():
        val = _parse_bool(line)
        if val is not None:
            flags.append(val)

    if not flags:
        return None

    return RunResult(
        data_num=int(m.group("data_num")),
        ckpt=int(m.group("ckpt")),
        seed=int(m.group("seed")),
        n_trials=len(flags),
        n_success=sum(flags),
    )


def collect_results(policy_dir: Path) -> list[RunResult]:
    results: list[RunResult] = []
    for child in sorted(policy_dir.iterdir()):
        if not child.is_dir():
            continue
        row = load_run_result(child)
        if row is not None:
            results.append(row)
    return results


def _fmt_pct(rate: float) -> str:
    return f"{100.0 * rate:5.1f}%"


def print_summary(results: list[RunResult], policy_dir: Path) -> None:
    if not results:
        print(f"[WARN] 未在 {policy_dir} 下找到有效的 all_success.txt")
        return

    data_nums = sorted({r.data_num for r in results})
    if len(data_nums) > 1:
        print(f"[WARN] 发现多个 data_num: {data_nums}，汇总时未按 data_num 分组")

    print(f"策略目录: {policy_dir.resolve()}")
    print(f"运行数: {len(results)}  |  总 trial 数: {sum(r.n_trials for r in results)}")
    print()

    header = f"{'ckpt':>6} {'seed':>5} {'trials':>7} {'success':>8} {'rate':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: (x.ckpt, x.seed)):
        print(
            f"{r.ckpt:6d} {r.seed:5d} {r.n_trials:7d} {r.n_success:8d} {_fmt_pct(r.success_rate):>8}"
        )

    print()
    print("按 checkpoint 聚合（跨 seed 的 success rate 均值 ± 标准差）:")
    by_ckpt: dict[int, list[float]] = {}
    for r in results:
        by_ckpt.setdefault(r.ckpt, []).append(r.success_rate)

    agg_header = f"{'ckpt':>6} {'#seed':>6} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}"
    print(agg_header)
    print("-" * len(agg_header))
    for ckpt in sorted(by_ckpt):
        rates = by_ckpt[ckpt]
        mean = statistics.mean(rates)
        std = statistics.pstdev(rates) if len(rates) > 1 else 0.0
        print(
            f"{ckpt:6d} {len(rates):6d} {_fmt_pct(mean):>8} {_fmt_pct(std):>8} "
            f"{_fmt_pct(min(rates)):>8} {_fmt_pct(max(rates)):>8}"
        )

    overall = sum(r.n_success for r in results) / sum(r.n_trials for r in results)
    print()
    print(f"总体成功率（所有 seed × ckpt 合并）: {_fmt_pct(overall)}")


def write_csv(results: list[RunResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["data_num", "ckpt", "seed", "n_trials", "n_success", "success_rate"])
        for r in sorted(results, key=lambda x: (x.ckpt, x.seed)):
            w.writerow([r.data_num, r.ckpt, r.seed, r.n_trials, r.n_success, f"{r.success_rate:.6f}"])
    print(f"[INFO] 已写入 CSV: {csv_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 validation 的 all_success.txt")
    parser.add_argument(
        "policy_dir",
        type=Path,
        nargs="?",
        default=Path(
            "data/validation/Isaac-UR10eShadowHand-Pickup-Direct-v0/ViTacDP_force"
        ),
        help="策略结果目录，例如 .../ViTacDP_force",
    )
    parser.add_argument("--csv", type=Path, default=None, help="可选：导出逐 run 明细 CSV")
    args = parser.parse_args()

    policy_dir = args.policy_dir
    if not policy_dir.is_dir():
        raise SystemExit(f"目录不存在: {policy_dir}")

    results = collect_results(policy_dir)
    print_summary(results, policy_dir)

    if args.csv is not None:
        write_csv(results, args.csv)


if __name__ == "__main__":
    main()
