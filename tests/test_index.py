"""nqp.index 单元测试：min-max 归一化 + 熵权法。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqp.index import composite_index, entropy_weights, minmax_normalize


def test_minmax_basic_range():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    out = minmax_normalize(df)
    assert out["a"].tolist() == [0.0, 0.5, 1.0]
    assert out["b"].tolist() == [0.0, 0.5, 1.0]
    assert out.min().min() == 0.0
    assert out.max().max() == 1.0


def test_minmax_negative_indicator_inverts():
    # 负向指标：越大越差，归一化后最大值应得 0
    df = pd.DataFrame({"emission": [1.0, 2.0, 3.0]})
    out = minmax_normalize(df, negative_cols=["emission"])
    assert out["emission"].tolist() == [1.0, 0.5, 0.0]


def test_minmax_constant_column_is_zero():
    df = pd.DataFrame({"c": [5.0, 5.0, 5.0]})
    out = minmax_normalize(df)
    assert (out["c"] == 0.0).all()


def test_entropy_weights_sum_to_one():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((20, 4)), columns=list("abcd"))
    norm = minmax_normalize(df)
    w = entropy_weights(norm)
    assert pytest.approx(w.sum(), abs=1e-9) == 1.0
    assert (w >= 0).all()


def test_entropy_gives_higher_weight_to_more_dispersed_indicator():
    # b 列方差极小（几乎无区分度），a 列分散；a 的权重应明显更高
    n = 50
    a = np.linspace(0, 1, n)
    b = np.full(n, 0.5) + np.random.default_rng(1).normal(0, 1e-4, n)
    norm = minmax_normalize(pd.DataFrame({"a": a, "b": b}))
    w = entropy_weights(norm)
    assert w["a"] > w["b"]


def test_entropy_requires_at_least_two_samples():
    df = pd.DataFrame({"a": [0.5]})
    with pytest.raises(ValueError):
        entropy_weights(df)


def test_composite_index_monotone_in_inputs():
    # 单调性：所有指标都更大的样本，综合得分应更高
    df = pd.DataFrame(
        {"x": [0.1, 0.5, 0.9], "y": [0.2, 0.6, 1.0]},
        index=["low", "mid", "high"],
    )
    score, w = composite_index(df)
    assert score["low"] < score["mid"] < score["high"]
    assert pytest.approx(w.sum(), abs=1e-9) == 1.0


def test_composite_index_accepts_external_weights():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 0.0]})
    # 全压在 x 上：得分应等于 x 的归一化值
    score, w = composite_index(df, weights=pd.Series({"x": 1.0, "y": 0.0}))
    assert score.tolist() == [0.0, 1.0]
    assert w["x"] == 1.0
