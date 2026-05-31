#!/usr/bin/env python3
"""panel_regression.py — 双向固定效应面板回归：NQP 对 HQD 的影响。

模型：HQD_it = β·NQP_it + γ·X_it + μ_i（省份）+ λ_t（年份）+ ε_it
标准误：省份层面聚类稳健。

用法：
    python code/04_econometrics/panel_regression.py \
        --panel data/processed/demo_panel.csv \
        --dependent hqd_index --regressors nqp_index rd_intensity
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nqp.panel import twoway_fe


def main() -> int:
    ap = argparse.ArgumentParser(description="双向固定效应面板回归")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--dependent", default="hqd_index")
    ap.add_argument("--regressors", nargs="+", default=["nqp_index"])
    ap.add_argument("--entity", default="province")
    ap.add_argument("--time", default="year")
    ap.add_argument("--out", default=None, help="把系数表写到这个 CSV（可选）")
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
    if "is_synthetic" in df.columns and df["is_synthetic"].any():
        print("[warn] 输入含合成数据（is_synthetic=1），以下结果仅用于流程演示。\n")

    res = twoway_fe(
        df, dependent=args.dependent, regressors=args.regressors,
        entity=args.entity, time=args.time,
    )

    print(f"=== 双向固定效应回归：{args.dependent} ~ {' + '.join(args.regressors)} ===")
    print(f"个体（{args.entity}）固定效应 + 时间（{args.time}）固定效应，省份层面聚类标准误")
    print(f"观测数 {res.nobs}，个体 {res.n_entities}，期数 {res.n_periods}，"
          f"组内 R² = {res.rsquared_within:.4f}\n")
    print(res.coef_table().to_string(float_format=lambda x: f"{x:.4f}"))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.coef_table().to_csv(args.out, encoding="utf-8-sig")
        print(f"\n[ok] 系数表写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
