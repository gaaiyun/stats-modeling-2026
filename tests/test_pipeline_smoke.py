"""端到端冒烟测试：合成面板 -> 指数 -> 耦合协调 -> 面板回归，全程不调用任何 API。

也覆盖 LLM 打标脚本里不需要网络的纯函数（报告发现、prompt 构造）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script(rel_path: str):
    """按文件路径加载 code/0X 下的脚本模块（目录名以数字开头，不能正常 import）。"""
    path = ROOT / rel_path
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_demo_panel_to_regression(tmp_path):
    """合成面板真系数能被双向 FE 回归大致恢复。"""
    make_demo = _load_script("code/04_econometrics/make_demo_panel.py")
    from nqp.coupling import coupling_coordination
    from nqp.index import composite_index
    from nqp.panel import twoway_fe

    df = make_demo.make_demo_panel(beta=0.6, seed=123)
    assert df["province"].nunique() == 31
    assert df["year"].nunique() == 12
    assert (df["is_synthetic"] == 1).all()

    # 综合指数能算出来且单调有效
    score, _ = composite_index(df[["nqp_index", "rd_intensity"]])
    assert score.between(0, 1).all()

    # 耦合协调度有效
    d = coupling_coordination(df["nqp_index"].clip(0, 1), df["hqd_index"].clip(0, 1))
    assert (d >= 0).all() and (d <= 1).all()

    # 回归恢复 beta
    res = twoway_fe(df, dependent="hqd_index", regressors=["nqp_index", "rd_intensity"])
    assert 0.4 < res.params["nqp_index"] < 0.8
    assert res.pvalues["nqp_index"] < 0.05
    assert res.nobs == 31 * 12


def test_build_indices_cli(tmp_path):
    """build_indices 脚本能在小样本 CSV 上跑通并写出 nqp_index 列。"""
    labels = pd.DataFrame({
        "year": [2014, 2015, 2016],
        "tech_innovation": [5, 7, 9],
        "industrial_upgrade": [4, 6, 8],
        "green_low_carbon": [8, 6, 7],
        "digital_empowerment": [3, 5, 8],
        "talent_support": [3, 4, 6],
    })
    in_csv = tmp_path / "labels.csv"
    out_csv = tmp_path / "out.csv"
    labels.to_csv(in_csv, index=False)

    mod = _load_script("code/04_econometrics/build_indices.py")
    sys.argv = ["build_indices.py", "--labels", str(in_csv), "--out", str(out_csv)]
    assert mod.main() == 0
    out = pd.read_csv(out_csv)
    assert "nqp_index" in out.columns
    assert out["nqp_index"].is_monotonic_increasing  # 三行指标递增 -> 指数递增


def test_pipeline_discover_and_prompt():
    """LLM 打标脚本：能递归发现 central 报告、prompt 含层级标注。"""
    pipe = _load_script("code/02_llm_label/llm_label_pipeline.py")
    schema = pipe.load_labeling_schema()
    prompt = pipe.build_prompt("加强关键核心技术攻关。", schema, region="中央")
    assert "中央政府工作报告" in prompt
    assert "tech_innovation" in prompt

    items = pipe.discover_reports(province_filter="中央")
    # 仓库内含 12 篇中央报告
    assert len(items) >= 1
    assert all(region == "中央" for region, _, _ in items)
