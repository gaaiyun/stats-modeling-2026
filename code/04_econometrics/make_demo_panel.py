#!/usr/bin/env python3
"""make_demo_panel.py — 生成【合成】省级面板，仅供端到端冒烟测试。

⚠️ 这不是真实数据。真实分析需要 31 省 × 2014—2025 的政府工作报告 LLM 评分
（见 code/01_scrape 收集指南）加上统计年鉴指标。本脚本用固定随机种子造一份
结构合理、真系数已知的合成面板，让耦合协调 / 面板回归脚本在没有真实省级数据时
也能跑通并被验证。

生成的文件含一列 ``is_synthetic=1`` 作为显式标记，下游和论文都不应把它当真实数据。

用法：
    python code/04_econometrics/make_demo_panel.py --out data/processed/demo_panel.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.config import PROVINCES, YEARS


def make_demo_panel(beta: float = 0.6, seed: int = 20260517) -> pd.DataFrame:
    """造一份真系数 (NQP→HQD = beta) 已知的合成面板。"""
    rng = np.random.default_rng(seed)
    prov_effect = {p: rng.normal(0, 0.15) for p in PROVINCES}
    year_trend = {y: 0.02 * (y - YEARS[0]) for y in YEARS}

    rows = []
    for p in PROVINCES:
        base_nqp = rng.uniform(0.2, 0.6)
        for y in YEARS:
            nqp = np.clip(base_nqp + year_trend[y] + rng.normal(0, 0.05), 0, 1)
            rd_intensity = np.clip(0.5 * nqp + rng.normal(0, 0.1), 0, 1)  # 中介：研发投入
            hqd = np.clip(
                beta * nqp + 0.3 * rd_intensity + prov_effect[p]
                + year_trend[y] + rng.normal(0, 0.05),
                0, 1,
            )
            rows.append({
                "province": p, "year": y,
                "nqp_index": round(float(nqp), 4),
                "rd_intensity": round(float(rd_intensity), 4),
                "hqd_index": round(float(hqd), 4),
                "is_synthetic": 1,
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成合成省级面板（仅冒烟测试用）")
    ap.add_argument("--out", required=True, help="输出 CSV 路径")
    ap.add_argument("--beta", type=float, default=0.6, help="NQP→HQD 真系数")
    ap.add_argument("--seed", type=int, default=20260517)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    df = make_demo_panel(beta=args.beta, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[ok] 合成面板 {len(df)} 行（{df.province.nunique()} 省 × {df.year.nunique()} 年）"
          f"，真 beta={args.beta}")
    print(f"[ok] 写入 {args.out}（含 is_synthetic=1 标记，切勿当真实数据）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
