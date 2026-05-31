#!/usr/bin/env python3
"""keyword_evolution.py — 政府工作报告的关键词演化与维度词频。

对收集到的报告做三件事，全部基于真实文本：

1. 每篇 TF-IDF 高权重关键词（看每年报告最突出的议题）。
2. 各报告在 5 个新质生产力维度上的种子词命中次数（``data/processed/dim_term_freq.csv``），
   可与 LLM 评分做相关性校验（稳健性证据：文本里某维度关键词越多，LLM 该维度评分理应越高）。
3. 维度词频的逐年趋势汇总打印。

用法：
    python code/03_text_mining/keyword_evolution.py \
        --reports-dir data/raw/gov_reports/central \
        --out-dir     data/processed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nqp.textfeatures import dimension_term_frequency, tfidf_matrix

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_dimension_keywords() -> dict[str, list[str]]:
    schema = yaml.safe_load((CONFIGS_DIR / "labeling_schema.yaml").read_text(encoding="utf-8"))
    return {d["id"]: d["keywords_seed"] for d in schema["dimensions"]}


def discover(reports_dir: Path) -> list[tuple[str, int, Path]]:
    items = []
    for p in sorted(reports_dir.rglob("*.txt")):
        parts = p.stem.split("_")
        if len(parts) != 2:
            continue
        try:
            year = int(parts[1])
        except ValueError:
            continue
        items.append((parts[0], year, p))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="关键词演化与维度词频")
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=15, help="每篇报告保留的 TF-IDF 关键词数")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    reports_dir = Path(args.reports_dir)
    items = discover(reports_dir)
    if not items:
        print(f"[error] {reports_dir} 下没有 *.txt 报告。", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = [p.read_text(encoding="utf-8") for _, _, p in items]
    labels = [f"{region}_{year}" for region, year, _ in items]

    # 1. TF-IDF
    tfidf = tfidf_matrix(docs, doc_labels=labels, top_k=args.top_k)
    tfidf_path = out_dir / "tfidf_keywords.csv"
    tfidf.to_csv(tfidf_path, index=False, encoding="utf-8-sig")

    # 2. 维度词频
    dim_kw = load_dimension_keywords()
    freq_rows = []
    for (region, year, _), doc in zip(items, docs):
        row = {"region": region, "year": year}
        row.update(dimension_term_frequency(doc, dim_kw))
        freq_rows.append(row)
    freq = pd.DataFrame(freq_rows)
    freq_path = out_dir / "dim_term_freq.csv"
    freq.to_csv(freq_path, index=False, encoding="utf-8-sig")

    print(f"[ok] TF-IDF 关键词写入 {tfidf_path}（{tfidf['doc'].nunique()} 篇）")
    print(f"[ok] 维度词频写入 {freq_path}")
    print("\n=== 各维度种子词命中次数（逐年）===")
    dim_cols = list(dim_kw.keys())
    print(freq.sort_values("year")[["year", *dim_cols]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
