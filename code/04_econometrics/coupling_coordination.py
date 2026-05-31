#!/usr/bin/env python3
"""coupling_coordination.py — 计算 NQP 与 HQD 两系统的耦合协调度 D。

输入：含两列归一化指数的面板 CSV（默认 ``nqp_index`` 与 ``hqd_index``）。
若指数不在 [0, 1]，脚本会先按各列 min-max 归一化再算 D。
输出：在原表追加 ``coupling_C``、``coordination_T``、``coupling_D``、``coord_grade`` 四列。

用法：
    python code/04_econometrics/coupling_coordination.py \
        --panel data/processed/demo_panel.csv \
        --out   data/processed/demo_coupling.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nqp.coupling import (
    classify_coordination,
    coordination_index,
    coupling_coordination,
    coupling_degree,
)
from nqp.index import minmax_normalize


def main() -> int:
    ap = argparse.ArgumentParser(description="耦合协调度模型")
    ap.add_argument("--panel", required=True, help="含两列指数的面板 CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--system-a", default="nqp_index", help="系统 A 列名")
    ap.add_argument("--system-b", default="hqd_index", help="系统 B 列名")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    path = Path(args.panel)
    if not path.exists():
        print(f"[error] 找不到 {path}", file=sys.stderr)
        return 2

    df = pd.read_csv(path)
    for col in (args.system_a, args.system_b):
        if col not in df.columns:
            print(f"[error] 缺少列 {col}", file=sys.stderr)
            return 2

    sub = df[[args.system_a, args.system_b]].astype(float)
    # 若不在 [0,1] 区间，先归一化
    if sub.min().min() < 0 or sub.max().max() > 1:
        sub = minmax_normalize(sub)
    ua, ub = sub[args.system_a], sub[args.system_b]

    df = df.copy()
    df["coupling_C"] = coupling_degree(ua, ub)
    df["coordination_T"] = coordination_index(ua, ub)
    df["coupling_D"] = coupling_coordination(ua, ub)
    df["coord_grade"] = classify_coordination(df["coupling_D"].to_numpy())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[ok] 耦合协调度 D 范围 [{df.coupling_D.min():.4f}, {df.coupling_D.max():.4f}]，"
          f"均值 {df.coupling_D.mean():.4f}")
    print("[ok] 等级分布：")
    for grade, n in df["coord_grade"].value_counts().items():
        print(f"     {grade:<8} {n}")
    print(f"[ok] 写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
