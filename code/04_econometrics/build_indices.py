#!/usr/bin/env python3
"""build_indices.py — 由 LLM 维度评分合成「新质生产力综合指数」(NQP index)。

输入：LLM 打标结果 CSV（含 5 个维度评分列，见 configs/labeling_schema.yaml）。
输出：在原表基础上追加 ``nqp_index`` 列，并打印各维度的熵权。

用法：
    python code/04_econometrics/build_indices.py \
        --labels data/interim/central_labels.csv \
        --out    data/processed/central_nqp_index.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.config import NQP_DIMENSION_IDS
from nqp.index import composite_index


def main() -> int:
    ap = argparse.ArgumentParser(description="构建新质生产力综合指数（熵权法）")
    ap.add_argument("--labels", required=True, help="LLM 打标 CSV 路径")
    ap.add_argument("--out", required=True, help="输出 CSV 路径")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"[error] 找不到 {labels_path}，先跑 LLM 打标 pipeline。", file=sys.stderr)
        return 2

    df = pd.read_csv(labels_path)
    missing = [c for c in NQP_DIMENSION_IDS if c not in df.columns]
    if missing:
        print(f"[error] 打标文件缺少维度列：{missing}", file=sys.stderr)
        return 2

    dims = df[NQP_DIMENSION_IDS].astype(float)
    score, weights = composite_index(dims)
    df = df.copy()
    df["nqp_index"] = score.values

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("=== 熵权（各维度对综合指数的客观权重）===")
    for dim, w in weights.sort_values(ascending=False).items():
        print(f"  {dim:<22} {w:.4f}")
    print()
    print(f"[ok] 样本数 {len(df)}，NQP 指数范围 [{score.min():.4f}, {score.max():.4f}]")
    print(f"[ok] 写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
