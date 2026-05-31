# 省级"新质生产力"政策文本的语义量化与耦合协调分析

用大语言模型给政府工作报告的"新质生产力"语义强度打分，再用熵权法、耦合协调度模型和双向固定效应面板回归，分析政策文本强度与高质量发展之间的关系。

2026 年（第十二届）全国大学生统计建模大赛参赛作品，主题"服务国家战略 创新统计赋能"。

## 这个项目做什么

政府工作报告里"新质生产力"相关的表述，过去靠人工读、人工编码，主观且难规模化。这里换一种做法：让 LLM 按五个维度（科技创新 / 产业升级 / 绿色低碳 / 数字赋能 / 人才支撑）对每篇报告打 1—10 分，并给出支撑句。打分结果再进入两条统计链路：

1. **语义演化**：看各维度评分逐年怎么变，并用 TF-IDF、维度关键词词频做交叉验证。
2. **耦合协调机制**：把新质生产力综合指数（NQP）与高质量发展指数（HQD）放进耦合协调度模型，再用面板回归估计 NQP 对 HQD 的影响。

LLM 只负责文本打分这一步。指数合成、耦合协调、回归全部是常规计量方法，代码在 `nqp/` 包里，可被单元测试覆盖。

## 仓库现状（先看这里）

| 环节 | 状态 | 数据 |
|---|---|---|
| 中央政府工作报告采集 | 可跑 | 2014—2025 共 12 篇，已入库 `data/raw/gov_reports/central/` |
| LLM 五维打标 | 可跑，已实测 | 12 篇中央报告的真实评分见 `data/interim/central_labels.csv` |
| 文本挖掘（TF-IDF / 维度词频） | 可跑 | 输出在 `data/processed/` |
| 综合指数 / 耦合协调 / 面板回归 | 可跑 | 见下方说明 |
| 31 省政府工作报告 | **未收集** | 见 `code/01_scrape` 的手动收集指南 |

耦合协调和面板回归需要"省份 × 年份"的面板，还需要高质量发展指标。这两类真实省级数据尚未收集齐（采集指南见下文）。在拿到真实省级数据之前，`code/04_econometrics/make_demo_panel.py` 会生成一份**带 `is_synthetic=1` 标记的合成面板**，仅用于把后面几步跑通和做单元测试，不能当真实结果。代码本身（熵权、耦合协调、双向 FE）是真的，换上真实 CSV 即可出真实结论。

## 安装

需要 Python 3.10+。

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 或装成可导入的包 + 测试依赖
pip install -e ".[dev]"
```

## 凭证

LLM 打标走 OpenAI 兼容端点（硅基流动、DeepSeek 等均可）。**密钥绝不写进仓库**，从环境变量或一个仓库外的文件读：

```bash
# 方式一：环境变量
set LLM_API_KEY=你的key
set LLM_BASE_URL=https://api.siliconflow.cn/v1
set LLM_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct

# 方式二：configs/api_keys.yaml（已被 .gitignore 排除）
copy configs\api_keys.yaml.example configs\api_keys.yaml
notepad configs\api_keys.yaml
```

不填凭证也能跑 `--dry-run`（只构造 prompt）和全部统计 / 测试环节，只是不能真正调模型打分。

## 跑一遍（不需要省级数据也能验证全流程）

```bash
# 1. 验证 prompt（无需凭证，用仓库里的中央报告）
python code/02_llm_label/llm_label_pipeline.py --province 中央 --year 2024 --dry-run

# 2. 真正打标（需凭证）。仓库已附带 12 篇中央报告的打标结果，可跳过这步
python code/02_llm_label/llm_label_pipeline.py --province 中央 \
    --output data/interim/central_labels.csv

# 3. 文本挖掘：TF-IDF 关键词 + 五维种子词词频
python code/03_text_mining/keyword_evolution.py \
    --reports-dir data/raw/gov_reports/central --out-dir data/processed

# 4. 综合指数（熵权法）
python code/04_econometrics/build_indices.py \
    --labels data/interim/central_labels.csv \
    --out    data/processed/central_nqp_index.csv

# 5. 出图：五维演化折线
python code/05_visualization/make_figures.py \
    --labels data/interim/central_labels.csv --out-dir figures
```

跑耦合协调和面板回归（用合成面板演示，结果仅示流程）：

```bash
python code/04_econometrics/make_demo_panel.py --out data/processed/demo_panel.csv
python code/04_econometrics/coupling_coordination.py \
    --panel data/processed/demo_panel.csv --out data/processed/demo_coupling.csv
python code/04_econometrics/panel_regression.py \
    --panel data/processed/demo_panel.csv \
    --dependent hqd_index --regressors nqp_index rd_intensity
```

拿到真实省级面板后，把 `--panel` 换成真实 CSV（需含 `province`、`year` 和各指数列）即可。

## 目录结构

```
.
├── nqp/                         # 核心算法库（被单元测试覆盖）
│   ├── index.py                 #   min-max 归一化 + 熵权法 + 综合指数
│   ├── coupling.py              #   耦合度 C / 协调指数 T / 耦合协调度 D + 等级划分
│   ├── textfeatures.py          #   分词 / 维度词频 / TF-IDF / 共现
│   └── panel.py                 #   双向固定效应回归封装
├── code/
│   ├── 01_scrape/               # 报告采集（中央可跑，省级附手动指南）
│   ├── 02_llm_label/            # LLM 五维打标 pipeline（缓存 / 重试 / 断点续跑）
│   ├── 03_text_mining/          # 关键词演化
│   ├── 04_econometrics/         # 指数 / 耦合协调 / 面板回归 / 合成面板
│   └── 05_visualization/        # 出图
├── configs/
│   ├── config.py                # 路径与维度常量（无密钥）
│   ├── labeling_schema.yaml     # 五维定义 + JSON schema（维度键的权威来源）
│   └── api_keys.yaml.example    # 凭证模板
├── data/
│   ├── raw/gov_reports/         # 原始报告 txt
│   ├── interim/                 # LLM 打标结果
│   └── processed/               # 指数 / 耦合协调 / 词频等
├── tests/                       # pytest（核心算法 + 端到端冒烟）
└── figures/                     # 输出图
```

## 方法说明

**综合指数（熵权法）**。先按指标方向 min-max 归一化（负向指标取 `max - x`），再按各指标的离散程度客观赋权：信息熵 `e_j = -k·Σ p_ij ln p_ij`（`k = 1/ln n`），差异系数 `g_j = 1 - e_j`，权重 `w_j = g_j / Σ g_j`。见 `nqp/index.py`。

**耦合协调度**。耦合度 `C = n·(∏U_i)^(1/n) / ΣU_i` 衡量子系统咬合紧密程度；综合协调指数 `T = Σα_i U_i`；耦合协调度 `D = √(C·T)`，兼顾"咬合"与"水平"。等级按廖重斌（1999）十分类，从"极度失调"到"优质协调"。见 `nqp/coupling.py`。

**面板回归**。基准模型为双向固定效应：

```
HQD_it = β·NQP_it + γ·X_it + μ_i + λ_t + ε_it
```

同时控制省份和年份固定效应，省份层面聚类稳健标准误。见 `nqp/panel.py`。

## 数据来源

| 类别 | 来源 | 说明 |
|---|---|---|
| 中央政府工作报告 | 中国政府网 / 新华网 | 2014—2025，已入库 |
| 省级政府工作报告 | 各省政府官网 / 北大法宝 | 待收集，见 `01_scrape` 指南 |
| 经济 / 三产 / R&D / 专利 | 国家统计局、国家知识产权局 | 高质量发展指标，待整理 |
| 数字经济规模 | 中国信通院 | 数字赋能维度 |
| 碳排放 | CEADs（清华公开） | 绿色低碳维度 |

收集省级报告：

```bash
python code/01_scrape/fetch_gov_reports.py --manual-guide   # 手动收集步骤与命名规范
python code/01_scrape/fetch_gov_reports.py --list           # 查看已收集覆盖情况
python code/01_scrape/fetch_central_reports.py              # 重新抓取中央报告
```

## 测试

```bash
pytest
```

覆盖熵权法、耦合协调度、文本特征、面板回归的数值正确性，以及一条不调用任何 API 的端到端冒烟链路。

## AI 工具使用声明

按比赛要求：大语言模型仅用于政府工作报告的五维语义打标这一步。所有统计模型、回归方程、表格、图表与论文正文由参赛队员独立完成；打分质量通过维度关键词词频交叉验证和文献基准对比保证。

## 许可

MIT，见 [LICENSE](LICENSE)。
