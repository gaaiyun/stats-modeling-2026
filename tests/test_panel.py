"""nqp.panel 单元测试：双向固定效应能否恢复已知系数。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqp.panel import twoway_fe


def _make_panel(beta: float = 0.7, seed: int = 0) -> pd.DataFrame:
    """造一个真系数已知的面板：含个体效应、时间效应和噪声。"""
    rng = np.random.default_rng(seed)
    provinces = [f"p{i}" for i in range(20)]
    years = list(range(2014, 2026))
    entity_fe = {p: rng.normal(0, 2) for p in provinces}
    time_fe = {t: rng.normal(0, 1) for t in years}
    rows = []
    for p in provinces:
        for t in years:
            nqp = rng.random()
            hqd = beta * nqp + entity_fe[p] + time_fe[t] + rng.normal(0, 0.05)
            rows.append({"province": p, "year": t, "nqp_index": nqp, "hqd_index": hqd})
    return pd.DataFrame(rows)


def test_twoway_fe_recovers_known_beta():
    df = _make_panel(beta=0.7)
    res = twoway_fe(df, dependent="hqd_index", regressors=["nqp_index"])
    assert res.params["nqp_index"] == pytest.approx(0.7, abs=0.05)
    assert res.pvalues["nqp_index"] < 0.01
    assert res.n_entities == 20
    assert res.n_periods == 12
    assert 0.0 <= res.rsquared_within <= 1.0


def test_twoway_fe_reports_dimensions():
    df = _make_panel()
    res = twoway_fe(df, dependent="hqd_index", regressors=["nqp_index"])
    assert res.nobs == 20 * 12
    tbl = res.coef_table()
    assert "coef" in tbl.columns and "p" in tbl.columns


def test_twoway_fe_missing_column_raises():
    df = _make_panel()
    with pytest.raises(ValueError):
        twoway_fe(df, dependent="nope", regressors=["nqp_index"])


def test_twoway_fe_needs_multiple_entities():
    df = _make_panel()
    one = df[df["province"] == "p0"]
    with pytest.raises(ValueError):
        twoway_fe(one, dependent="hqd_index", regressors=["nqp_index"])
