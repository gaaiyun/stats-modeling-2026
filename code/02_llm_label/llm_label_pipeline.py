#!/usr/bin/env python3
"""
llm_label_pipeline.py — LLM 多维语义打标 pipeline

输入：data/raw/gov_reports/{province}_{year}.txt
输出：data/interim/llm_labels.csv（含 5 维度评分 + evidence）

每篇报告调用 SiliconFlow Qwen 一次，prompt 含 schema，输出 JSON。
含重试 / 缓存 / 断点续跑逻辑。

用法:
    python code/02_llm_label/llm_label_pipeline.py                    # 跑全部
    python code/02_llm_label/llm_label_pipeline.py --province 北京     # 单省
    python code/02_llm_label/llm_label_pipeline.py --year 2024         # 单年
    python code/02_llm_label/llm_label_pipeline.py --dry-run --limit 3 # 验证 prompt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_REPORTS_DIR = DATA_DIR / "raw" / "gov_reports"
INTERIM_DIR = DATA_DIR / "interim"
LLM_CACHE_DIR = INTERIM_DIR / "llm_cache"
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ============ 配置加载 ============

def load_api_config() -> dict:
    cfg_path = CONFIGS_DIR / "api_keys.yaml"
    if not cfg_path.exists():
        print(f"[error] {cfg_path} 不存在。\n"
              f"请先：cp {CONFIGS_DIR / 'api_keys.yaml.example'} {cfg_path}\n"
              f"然后填入 SiliconFlow API key", file=sys.stderr)
        sys.exit(2)
    if yaml is None:
        print("[error] PyYAML 未安装。pip install pyyaml", file=sys.stderr)
        sys.exit(2)
    return yaml.safe_load(cfg_path.read_text(encoding='utf-8'))


def load_labeling_schema() -> dict:
    return yaml.safe_load((CONFIGS_DIR / "labeling_schema.yaml").read_text(encoding='utf-8'))


# ============ Prompt 构造 ============

def build_prompt(report_text: str, schema: dict, max_text_len: int = 6000) -> str:
    """构造 LLM prompt。报告太长时截断（保留 head + tail）。"""
    if len(report_text) > max_text_len:
        head = report_text[:max_text_len // 2]
        tail = report_text[-max_text_len // 2:]
        report_text = f"{head}\n\n...[中间省略]...\n\n{tail}"

    dims_text = []
    for d in schema['dimensions']:
        kw = "、".join(d['keywords_seed'][:8])
        dims_text.append(f"- **{d['name']}（{d['id']}）**：{d['description'].strip()}\n  参考关键词：{kw}")
    dims_block = "\n\n".join(dims_text)

    return f"""你是一名经济学者，需要对一份省级政府工作报告做新质生产力相关的语义评分。

# 评分维度（5 个，每个 1-10 分）

{dims_block}

# 评分规则

- 分数 1-10：1 = 完全没有提及；5 = 一般性提及但缺具体；10 = 非常深入、有明确举措。
- 重要：分数应基于**报告里实际写了什么**，不是省份的实际情况（不要靠你的"先验"猜测）。
- 每个维度需要给出 1 句不超过 50 字的"evidence"——支撑你给分的关键句子或词组。
- 最后给一段 100 字内的 overall_summary。

# 输出格式（严格 JSON，不要任何其他文字）

```json
{{
  "tech_innovation": <1-10>,
  "tech_innovation_evidence": "...",
  "industrial_upgrade": <1-10>,
  "industrial_upgrade_evidence": "...",
  "green_low_carbon": <1-10>,
  "green_low_carbon_evidence": "...",
  "digital_empowerment": <1-10>,
  "digital_empowerment_evidence": "...",
  "talent_support": <1-10>,
  "talent_support_evidence": "...",
  "overall_summary": "..."
}}
```

# 待评分文本

{report_text}

# 输出（严格 JSON）：
"""


# ============ LLM 调用 ============

def cache_key(province: str, year: int, model: str, prompt: str) -> str:
    h = hashlib.md5(f"{model}|{prompt}".encode('utf-8')).hexdigest()[:12]
    return f"{province}_{year}_{h}"


def llm_call(client: 'OpenAI', model: str, prompt: str,
             max_retries: int = 3, retry_delay: int = 2) -> dict:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )
            text = (resp.choices[0].message.content or "").strip()
            # 提取 JSON（可能包了 ```json ... ```）
            if text.startswith("```"):
                lines = text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_err = f"JSON parse error: {e}; raw: {text[:200]}"
        except Exception as e:
            last_err = f"API error: {e}"
        if attempt < max_retries - 1:
            print(f"      [retry {attempt+1}/{max_retries}] {last_err}", file=sys.stderr)
            time.sleep(retry_delay * (2 ** attempt))  # 指数退避
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


# ============ Pipeline 主逻辑 ============

def discover_reports(province_filter: str | None = None,
                     year_filter: int | None = None) -> list[tuple[str, int, Path]]:
    """扫描 data/raw/gov_reports/，返回 [(province, year, path)]"""
    if not RAW_REPORTS_DIR.exists():
        print(f"[warn] {RAW_REPORTS_DIR} 不存在。先跑 fetch_gov_reports.py", file=sys.stderr)
        return []
    items = []
    for p in sorted(RAW_REPORTS_DIR.glob("*.txt")):
        # 文件名约定：{province}_{year}.txt，如 北京_2024.txt
        parts = p.stem.split("_")
        if len(parts) != 2:
            continue
        province, year_str = parts
        try:
            year = int(year_str)
        except ValueError:
            continue
        if province_filter and province != province_filter:
            continue
        if year_filter and year != year_filter:
            continue
        items.append((province, year, p))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 多维语义打标 pipeline")
    parser.add_argument('--province', help='只跑某省')
    parser.add_argument('--year', type=int, help='只跑某年')
    parser.add_argument('--limit', type=int, help='最多处理 N 个文件')
    parser.add_argument('--dry-run', action='store_true', help='只打印 prompt，不调 API')
    parser.add_argument('--output', default=None, help='输出 csv 路径（默认 data/interim/llm_labels.csv）')
    parser.add_argument('--model', default=None, help='覆盖配置里的默认 model')
    args = parser.parse_args()

    cfg = load_api_config()
    schema = load_labeling_schema()

    sf = cfg['siliconflow']
    model = args.model or sf['default_model']
    api_key = sf['api_key']

    if api_key.startswith("sk-your_") or not api_key:
        print(f"[error] 请先在 {CONFIGS_DIR/'api_keys.yaml'} 里填入真实 SiliconFlow key", file=sys.stderr)
        return 2

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output) if args.output else INTERIM_DIR / 'llm_labels.csv'

    items = discover_reports(args.province, args.year)
    if args.limit:
        items = items[:args.limit]

    if not items:
        print("[info] 没有找到报告。先跑 fetch_gov_reports.py 准备数据。", file=sys.stderr)
        # demo 模式：用一段假文本演示 prompt
        demo_text = "去年我省深入实施创新驱动发展战略，加强关键核心技术攻关，推进战略性新兴产业培育，新能源汽车产量增长 40%。新型基础设施建设取得突破，5G 基站新增 1.2 万个。深入推进碳达峰行动..."
        prompt = build_prompt(demo_text, schema)
        print("=" * 70)
        print("DEMO PROMPT（无报告时显示）：")
        print("=" * 70)
        print(prompt)
        return 0

    if args.dry_run:
        # 只打印第一个的 prompt
        province, year, p = items[0]
        text = p.read_text(encoding='utf-8')
        prompt = build_prompt(text, schema)
        print(f"[dry-run] 会处理 {len(items)} 个文件")
        print(f"[dry-run] 第一个：{province} {year} ({len(text)} 字符)")
        print(f"[dry-run] Prompt 长度 ~{len(prompt)} 字符")
        print()
        print("=" * 70)
        print("PROMPT：")
        print("=" * 70)
        print(prompt[:3000] + ("\n...[truncated]..." if len(prompt) > 3000 else ""))
        return 0

    if OpenAI is None:
        print("[error] openai 未装。pip install openai", file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key, base_url=sf['base_url'])

    print(f"[info] 模型：{model}")
    print(f"[info] 待处理：{len(items)} 个文件")
    print(f"[info] 输出：{out_path}")
    print()

    write_header = not out_path.exists()
    success, skipped, failed = 0, 0, 0

    with out_path.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'province', 'year', 'model',
            'tech_innovation', 'tech_innovation_evidence',
            'industrial_upgrade', 'industrial_upgrade_evidence',
            'green_low_carbon', 'green_low_carbon_evidence',
            'digital_empowerment', 'digital_empowerment_evidence',
            'talent_support', 'talent_support_evidence',
            'overall_summary',
        ])
        if write_header:
            writer.writeheader()

        for idx, (province, year, p) in enumerate(items, 1):
            text = p.read_text(encoding='utf-8')
            prompt = build_prompt(text, schema)
            ck = cache_key(province, year, model, prompt)
            cache_path = LLM_CACHE_DIR / f"{ck}.json"

            if cache_path.exists():
                try:
                    result = json.loads(cache_path.read_text(encoding='utf-8'))
                    print(f"[{idx}/{len(items)}] {province} {year} - cached")
                    skipped += 1
                except json.JSONDecodeError:
                    cache_path.unlink()
                    result = None
            else:
                result = None

            if result is None:
                print(f"[{idx}/{len(items)}] {province} {year} - calling LLM ...", end='', flush=True)
                try:
                    result = llm_call(client, model, prompt,
                                      max_retries=sf['max_retries'],
                                      retry_delay=sf['retry_delay_sec'])
                    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                          encoding='utf-8')
                    print(" OK")
                    success += 1
                    # 简单限流
                    time.sleep(60.0 / sf['rate_limit_qpm'])
                except Exception as e:
                    print(f" FAILED: {e}")
                    failed += 1
                    continue

            row = {'province': province, 'year': year, 'model': model}
            for d in schema['dimensions']:
                k = d['id']
                row[k] = result.get(k)
                row[f"{k}_evidence"] = result.get(f"{k}_evidence", "")
            row['overall_summary'] = result.get('overall_summary', '')
            writer.writerow(row)

    print()
    print(f"=== Done. success={success} cached={skipped} failed={failed} ===")
    print(f"Output: {out_path}")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
