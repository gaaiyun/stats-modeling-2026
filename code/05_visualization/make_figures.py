#!/usr/bin/env python3
"""make_figures.py — 由处理后的数据出图。

目前支持两类图：

1. ``nqp_trend``   —— 5 维评分的逐年折线（输入 LLM 打标 CSV，需有 ``year`` 列）。
2. ``coord_hist``  —— 耦合协调度等级分布柱状图（输入耦合协调 CSV，需有 ``coord_grade``）。

用法：
    python code/05_visualization/make_figures.py \
        --labels  data/interim/central_labels.csv \
        --coupling data/processed/demo_coupling.csv \
        --out-dir figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.config import NQP_DIMENSIONS, configure_matplotlib_cjk
from nqp.coupling import COORDINATION_LABELS


def plot_nqp_trend(labels_csv: Path, out_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(labels_csv)
    if "year" not in df.columns:
        print(f"[warn] {labels_csv} 无 year 列，跳过趋势图。", file=sys.stderr)
        return None
    df = df.sort_values("year")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for dim_id, dim_name in NQP_DIMENSIONS.items():
        if dim_id in df.columns:
            ax.plot(df["year"], df[dim_id], marker="o", label=dim_name)
    ax.set_xlabel("年份")
    ax.set_ylabel("LLM 语义评分 (1—10)")
    ax.set_title("新质生产力五维语义强度演化")
    ax.set_ylim(0, 10.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    out = out_dir / "nqp_dimension_trend.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_coord_hist(coupling_csv: Path, out_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(coupling_csv)
    if "coord_grade" not in df.columns:
        print(f"[warn] {coupling_csv} 无 coord_grade 列，跳过等级图。", file=sys.stderr)
        return None
    order = [g for g in COORDINATION_LABELS if g in set(df["coord_grade"])]
    counts = df["coord_grade"].value_counts().reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(counts)), counts.values, color="#4C72B0")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.set_ylabel("样本数")
    ax.set_title("耦合协调度等级分布")
    fig.tight_layout()
    out = out_dir / "coordination_grade_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="出图")
    ap.add_argument("--labels", help="LLM 打标 CSV（画维度趋势）")
    ap.add_argument("--coupling", help="耦合协调 CSV（画等级分布）")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    if not args.labels and not args.coupling:
        print("[error] 至少提供 --labels 或 --coupling 之一。", file=sys.stderr)
        return 2

    configure_matplotlib_cjk()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    made = []
    if args.labels:
        p = plot_nqp_trend(Path(args.labels), out_dir)
        if p:
            made.append(p)
    if args.coupling:
        p = plot_coord_hist(Path(args.coupling), out_dir)
        if p:
            made.append(p)

    for p in made:
        print(f"[ok] 出图 {p}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
