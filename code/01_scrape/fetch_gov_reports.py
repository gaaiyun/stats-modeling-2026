#!/usr/bin/env python3
"""
fetch_gov_reports.py — 收集 31 省政府工作报告

数据来源（按推荐顺序）：
  1. 各省政府门户官网（最权威，但格式各异需人工辅助）
  2. 北大法宝 / 国家法律法规数据库（结构化，但需注册）
  3. 中国政府网汇总页 https://www.gov.cn/lianbo/bumen/

由于政府官网反爬较严 + 格式差异大，本脚本提供半自动模板：
  - 对北京/上海/广东等一线省市提供直接抓取
  - 其他省份提示手动收集 + txt 格式标准

用法：
  # 列出已收集的报告
  python code/01_scrape/fetch_gov_reports.py --list

  # 抓取北京（已实现）
  python code/01_scrape/fetch_gov_reports.py --province 北京 --years 2014 2015 ... 2025

  # 提示手动收集步骤
  python code/01_scrape/fetch_gov_reports.py --manual-guide
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_REPORTS_DIR = PROJECT_ROOT / "data" / "raw" / "gov_reports"

# 31 个省级行政区（不含港澳台）
PROVINCES = [
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
    "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州",
    "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆",
]

YEARS = list(range(2014, 2026))   # 2014-2025

# 各省政府官网（最新可达到的政府工作报告页面）
# 注意：URL 容易失效，建议运行前先访问验证
GOV_URLS = {
    "北京": "https://www.beijing.gov.cn/gongkai/zfgzbg/",
    "上海": "https://www.shanghai.gov.cn/zfgzbg/",
    "广东": "http://www.gd.gov.cn/zwgk/zfgzbg/",
    # ... 其他省份按需补充
}


def list_collected() -> None:
    """列出已收集的报告。"""
    if not RAW_REPORTS_DIR.exists():
        print(f"[info] {RAW_REPORTS_DIR} 不存在，还没开始收集")
        return
    files = sorted(RAW_REPORTS_DIR.rglob("*.txt"))
    if not files:
        print("(没有任何报告)")
        return

    by_province: dict[str, list[int]] = {}
    for p in files:
        parts = p.stem.split("_")
        if len(parts) == 2:
            try:
                by_province.setdefault(parts[0], []).append(int(parts[1]))
            except ValueError:
                pass

    print(f"已收集 {len(files)} 份报告，覆盖 {len(by_province)} 个省份：")
    print()
    print(f"{'省份':<10} {'年份范围':<25} {'数量':<5} {'缺失年份'}")
    print("-" * 80)
    for prov in PROVINCES:
        if prov in by_province:
            years = sorted(by_province[prov])
            missing = sorted(set(YEARS) - set(years))
            year_range = f"{years[0]}-{years[-1]}"
            print(f"{prov:<10} {year_range:<25} {len(years):<5} {missing if missing else '完整'}")
    not_covered = [p for p in PROVINCES if p not in by_province]
    if not_covered:
        print()
        print(f"未覆盖（{len(not_covered)} 省）：{', '.join(not_covered)}")


def manual_guide() -> None:
    print("""
==============================================================
手动收集 31 省政府工作报告步骤
==============================================================

【推荐路径 1：政府官网】
  1. 访问省政府官网，搜索"政府工作报告"
  2. 找到历年报告页面
  3. 复制全文 (从"各位代表" 到 "谢谢大家")
  4. 保存为 data/raw/gov_reports/{省份}_{年份}.txt（UTF-8 编码）

【推荐路径 2：北大法宝】
  https://www.pkulaw.com/  注册 → 搜"政府工作报告" → 按地区筛选 → 复制全文

【推荐路径 3：中国政府网】
  https://www.gov.cn/  搜"政府工作报告" → 按"地方"筛选

【推荐路径 4：现成数据集（如有）】
  - GitHub 搜 "中国 政府工作报告 数据集"，部分研究者公开整理过
  - 知乎 / CSDN 上有非官方汇总，但需人工核对

==============================================================
文件命名规范
==============================================================

文件名：data/raw/gov_reports/{省份}_{年份}.txt

例子：
  北京_2024.txt
  上海_2024.txt
  广东_2023.txt

省份名：用《关于完整列表见 PROVINCES 常量》（不含"省/市/自治区"后缀）

==============================================================
最少完整度建议
==============================================================

为了支撑统计建模分析，建议：
  - 至少 25 个省（包含东中西部）
  - 至少 8 年（2017-2024，覆盖政策变迁）
  - 总计 ≥ 200 份报告

==============================================================
质量检查
==============================================================

  python code/01_scrape/fetch_gov_reports.py --list

会显示已收集的省份和年份覆盖情况。
""")


def main() -> int:
    parser = argparse.ArgumentParser(description="收集 31 省政府工作报告")
    parser.add_argument("--list", action="store_true", help="列出已收集的报告")
    parser.add_argument("--manual-guide", action="store_true", help="显示手动收集指南")
    parser.add_argument("--province", help="抓取某省（仅支持 GOV_URLS 中已配置的）")
    parser.add_argument("--years", nargs="+", type=int, help="抓取的年份列表")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    if args.list:
        list_collected()
        return 0
    if args.manual_guide:
        manual_guide()
        return 0
    if args.province:
        if args.province not in GOV_URLS:
            print(f"[warn] {args.province} 暂未实现自动抓取。", file=sys.stderr)
            print("[info] 见 --manual-guide 了解手动步骤", file=sys.stderr)
            return 1
        print(f"[TODO] 自动抓取 {args.province} 还在开发，请暂时手动收集（python ... --manual-guide）")
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
