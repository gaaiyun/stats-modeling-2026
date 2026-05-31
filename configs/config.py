"""项目全局路径与常量。

API key 一律从环境变量或 ``configs/api_keys.yaml``（已 gitignore）读取，
本文件不含任何密钥。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
GOV_REPORT_DIR = RAW_DIR / "gov_reports"
STATS_DIR = RAW_DIR / "stats"
FIGURES_DIR = ROOT / "figures"
PAPER_DIR = ROOT / "paper"
LOGS_DIR = ROOT / "logs"

SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

# 31 个省级行政区（不含港澳台），顺序按统计年鉴习惯
PROVINCES = [
    "北京", "天津", "河北", "山西", "内蒙古",
    "辽宁", "吉林", "黑龙江",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "广西", "海南",
    "重庆", "四川", "贵州", "云南", "西藏",
    "陕西", "甘肃", "青海", "宁夏", "新疆",
]

YEARS = list(range(2014, 2026))  # 2014—2025

# 新质生产力 5 维度。键名与 configs/labeling_schema.yaml、LLM 输出 CSV 完全一致，
# 这是全项目维度键的唯一权威来源，下游代码请引用这里而不要另写一份。
NQP_DIMENSIONS = {
    "tech_innovation": "科技创新",
    "industrial_upgrade": "产业升级",
    "green_low_carbon": "绿色低碳",
    "digital_empowerment": "数字赋能",
    "talent_support": "人才支撑",
}
NQP_DIMENSION_IDS = list(NQP_DIMENSIONS.keys())


def configure_matplotlib_cjk() -> None:
    """按需设置 matplotlib 中文字体。出图脚本显式调用，避免导入即触发副作用。"""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def ensure_dirs() -> None:
    """创建数据 / 输出目录。脚本运行时调用，导入本模块时不产生副作用。"""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, GOV_REPORT_DIR, STATS_DIR,
              FIGURES_DIR, PAPER_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
