# 基于大语言模型语义量化的省级"新质生产力"政策文本演化与高质量发展耦合协调机制研究

> **2026 年（第十二届）全国大学生统计建模大赛参赛作品**
>
> 主题：服务国家战略 创新统计赋能

## 一、研究问题

围绕 2024 年提出的国家战略——**新质生产力**，回答两个核心问题：

1. 2014—2025 年中国 31 个省级政府工作报告中"新质生产力"语义维度（科技创新 / 产业升级 / 绿色低碳 / 数字赋能 / 人才支撑）如何演化？
2. 政策文本表征的"新质生产力强度"与各省高质量发展水平之间是否存在耦合协调与因果驱动关系？

## 二、技术路线

```
省政府工作报告(原始文本)
        │
        ▼
┌────────────────────┐    硅基流动 Qwen3.5-27B
│  LLM 多维度打标     │◀──────────────────────
│  (5 维 × 1-10 分)  │
└────────────────────┘
        │
        ▼
┌────────────────────┐
│ 文本挖掘（LDA/共现）│
└────────────────────┘
        │
        ▼
┌────────────────────┐    统计局 / 国家知识产权局 / CEADs
│  耦合协调度模型     │◀──────────────────────
│  + 双向固定面板回归 │
│  + 空间杜宾 (SDM)   │
│  + Bootstrap 中介  │
└────────────────────┘
        │
        ▼
   可视化 + 论文
```

## 三、目录结构

```
g:\统计建模\
├── data/
│   ├── raw/                    # 原始爬取/下载数据
│   │   ├── gov_reports/        # 31 省政府工作报告 txt
│   │   └── stats/              # 统计年鉴/专利/碳排放等
│   ├── interim/                # LLM 打标/中间结果
│   └── processed/              # 最终面板数据
├── code/
│   ├── 01_scrape/              # 数据采集脚本
│   ├── 02_llm_label/           # LLM 多维打标
│   ├── 03_text_mining/         # LDA / TF-IDF / 共现网络
│   ├── 04_econometrics/        # 耦合协调 / 面板 / 空间计量
│   └── 05_visualization/       # 出图脚本
├── figures/                    # 论文用图
├── paper/                      # 论文 (.docx / .pdf)
├── logs/                       # 运行日志
├── configs/                    # API key / 字段映射
└── README.md
```

## 四、数据来源（全部公开合规）

| 类别 | 来源 | 说明 |
|---|---|---|
| 省级政府工作报告 | 31 省政府官网 / 北大法宝公开版 | 2014—2025，共 ~372 篇 |
| 经济总量 / 三产结构 | 国家统计局 | 年度面板 |
| R&D 经费 / 专利 | 国家知识产权局、统计年鉴 | 创新投入 / 产出 |
| 数字经济规模 | 中国信通院《数字经济发展白皮书》 | 数字赋能维度 |
| 碳排放 | CEADs（清华公开） | 绿色低碳维度 |
| 人均受教育年限 | 第七次人口普查 + 年鉴 | 人才支撑维度 |

## 五、复现指引

```bash
pip install -r requirements.txt

# 1. 爬取 / 整理政府工作报告
python code/01_scrape/fetch_gov_reports.py

# 2. LLM 多维打标
python code/02_llm_label/llm_label_pipeline.py

# 3. 文本挖掘
python code/03_text_mining/lda_topic.py
python code/03_text_mining/keyword_network.py

# 4. 计量分析
python code/04_econometrics/build_indices.py
python code/04_econometrics/coupling_coordination.py
python code/04_econometrics/panel_regression.py
python code/04_econometrics/spatial_dm.py

# 5. 可视化
python code/05_visualization/make_all_figures.py
```

## 六、AI 工具使用声明

按比赛要求，本项目使用大语言模型（硅基流动 Qwen3.5-27B）**仅用于政策文本的多维语义打标**，
所有统计模型、回归方程、图表与正文文字均由参赛队员独立撰写，详见《AI 工具使用情况表》。
