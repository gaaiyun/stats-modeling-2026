"""项目全局配置——勿提交带 key 的版本到公开仓库"""
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

for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, GOV_REPORT_DIR, STATS_DIR,
          FIGURES_DIR, PAPER_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
LLM_FALLBACK_MODELS = [
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Pro/Qwen/Qwen2.5-7B-Instruct",
]

PROVINCES = [
    "北京", "天津", "河北", "山西", "内蒙古",
    "辽宁", "吉林", "黑龙江",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "广西", "海南",
    "重庆", "四川", "贵州", "云南", "西藏",
    "陕西", "甘肃", "青海", "宁夏", "新疆",
]

YEARS = list(range(2014, 2026))

NQP_DIMENSIONS = {
    "tech_innovation": "科技创新（基础研究/关键核心技术/原创性突破）",
    "industry_upgrade": "产业升级（战略性新兴产业/未来产业/先进制造）",
    "green_transform": "绿色低碳（双碳目标/清洁能源/生态文明）",
    "digital_empower": "数字赋能（数字经济/人工智能/数字化转型）",
    "talent_support": "人才支撑（创新人才/教育/职业培训）",
}

import matplotlib

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False
