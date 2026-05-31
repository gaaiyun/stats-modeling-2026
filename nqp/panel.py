"""面板数据建模：双向固定效应回归的轻封装。

围绕 ``linearmodels.PanelOLS`` 做一层方便的接口，固定本项目的基准设定：

    HQD_it = β · NQP_it + γ · X_it + μ_i + λ_t + ε_it

即同时控制个体（省份）固定效应和时间固定效应，聚类稳健标准误聚到个体层面。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["PanelResult", "twoway_fe"]


@dataclass
class PanelResult:
    """精简后的回归结果，便于打印成表和写 CSV。"""

    params: pd.Series
    std_errors: pd.Series
    tstats: pd.Series
    pvalues: pd.Series
    rsquared_within: float
    nobs: int
    n_entities: int
    n_periods: int
    summary: str

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.std_errors,
                "t": self.tstats,
                "p": self.pvalues,
            }
        )


def twoway_fe(
    df: pd.DataFrame,
    dependent: str,
    regressors: list[str],
    entity: str = "province",
    time: str = "year",
    cluster_entity: bool = True,
) -> PanelResult:
    """估计双向固定效应面板回归。

    Parameters
    ----------
    df:
        长面板，至少含 ``entity``、``time``、``dependent`` 及 ``regressors`` 各列。
    dependent:
        因变量列名（如 ``hqd_index``）。
    regressors:
        解释变量列名列表（核心解释变量放第一个，如 ``nqp_index``）。
    entity, time:
        个体维与时间维列名。
    cluster_entity:
        是否用个体层面聚类稳健标准误（默认是）。

    Returns
    -------
    PanelResult
    """
    from linearmodels.panel import PanelOLS

    missing = [c for c in [entity, time, dependent, *regressors] if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少这些列：{missing}")

    data = df.dropna(subset=[dependent, *regressors]).copy()
    if data[entity].nunique() < 2:
        raise ValueError("个体数不足 2，无法估计个体固定效应。")
    if data[time].nunique() < 2:
        raise ValueError("时间期数不足 2，无法估计时间固定效应。")

    panel = data.set_index([entity, time])
    y = panel[dependent]
    x = panel[regressors]

    mod = PanelOLS(y, x, entity_effects=True, time_effects=True, drop_absorbed=True)
    cov_kwargs = {"cov_type": "clustered", "cluster_entity": True} if cluster_entity else {}
    res = mod.fit(**cov_kwargs)

    return PanelResult(
        params=res.params,
        std_errors=res.std_errors,
        tstats=res.tstats,
        pvalues=res.pvalues,
        rsquared_within=float(res.rsquared_within),
        nobs=int(res.nobs),
        n_entities=int(panel.index.get_level_values(0).nunique()),
        n_periods=int(panel.index.get_level_values(1).nunique()),
        summary=str(res.summary),
    )
