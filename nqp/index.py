"""综合指数构建：min-max 归一化 + 熵权法。

熵权法（entropy weight method）按指标的离散程度客观赋权，是新质生产力测度
文献里最常用的合成方法（韩文龙等 2024；周颖刚等 2025）。这里实现的是标准
做法，并把容易出错的细节固定下来：

1. 先做 min-max 归一化，区分正向 / 负向指标（负向指标取 ``(max - x)``）。
2. 归一化后整列做平移，避免出现 0（``ln 0`` 无定义）。平移量为列内最小正值的
   一半，既保证为正、又几乎不影响相对大小。
3. 信息熵 ``e_j = -k * Σ p_ij ln p_ij``，其中 ``k = 1 / ln(n)``，``n`` 为样本数。
4. 差异系数 ``g_j = 1 - e_j``，权重 ``w_j = g_j / Σ g_j``。

所有函数对 ``pandas.DataFrame`` 操作，行=样本（如省份-年份），列=指标。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "minmax_normalize",
    "entropy_weights",
    "composite_index",
]


def minmax_normalize(
    df: pd.DataFrame,
    negative_cols: list[str] | None = None,
) -> pd.DataFrame:
    """对每列做 min-max 归一化到 [0, 1]。

    Parameters
    ----------
    df:
        行=样本，列=指标，全部为数值。
    negative_cols:
        负向指标列名（数值越大越差，如能耗、碳排放）。这些列用
        ``(max - x) / (max - min)`` 归一，其余用 ``(x - min) / (max - min)``。

    Notes
    -----
    某列所有值相等时，分母为 0，该列整体记为 0.0（无区分度，熵权法里也会得到
    接近 0 的权重）。
    """
    if df.empty:
        raise ValueError("输入数据为空，无法归一化。")
    negative = set(negative_cols or [])
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        x = df[col].astype(float)
        lo, hi = x.min(), x.max()
        rng = hi - lo
        if rng == 0:
            out[col] = 0.0
            continue
        if col in negative:
            out[col] = (hi - x) / rng
        else:
            out[col] = (x - lo) / rng
    return out


def entropy_weights(normalized: pd.DataFrame) -> pd.Series:
    """对已归一化的指标矩阵计算熵权。

    Parameters
    ----------
    normalized:
        min-max 归一化后的矩阵（值域 [0, 1]）。

    Returns
    -------
    pandas.Series
        index 为列名，值为权重，和为 1。
    """
    if normalized.empty:
        raise ValueError("输入数据为空，无法计算熵权。")
    n = len(normalized)
    if n < 2:
        raise ValueError(f"熵权法至少需要 2 个样本，当前只有 {n} 个。")

    mat = normalized.astype(float).copy()

    # 平移到正区间：每列加上列内最小正值的一半（若整列为 0 则加一个极小量）。
    shifted = pd.DataFrame(index=mat.index, columns=mat.columns, dtype=float)
    for col in mat.columns:
        x = mat[col]
        positives = x[x > 0]
        eps = (positives.min() / 2.0) if not positives.empty else 1e-6
        shifted[col] = x + eps

    col_sums = shifted.sum(axis=0)
    p = shifted.divide(col_sums, axis=1)  # 列归一为概率分布

    k = 1.0 / np.log(n)
    # p * ln p，约定 0 处贡献为 0
    plnp = p * np.log(p.where(p > 0, np.nan))
    plnp = plnp.fillna(0.0)
    entropy = -k * plnp.sum(axis=0)        # 信息熵 e_j ∈ [0, 1]
    diversity = 1.0 - entropy              # 差异系数 g_j

    total = diversity.sum()
    if total <= 0:
        # 所有指标都没有区分度：退化为等权
        return pd.Series(1.0 / len(mat.columns), index=mat.columns)
    return diversity / total


def composite_index(
    df: pd.DataFrame,
    negative_cols: list[str] | None = None,
    weights: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """构造综合指数。

    先 min-max 归一化，再用熵权（或外部给定权重）加权求和。

    Returns
    -------
    (index_score, weights)
        ``index_score`` 为每个样本的综合得分（Series，与 ``df`` 同 index），
        ``weights`` 为各指标权重。
    """
    normalized = minmax_normalize(df, negative_cols=negative_cols)
    if weights is None:
        weights = entropy_weights(normalized)
    else:
        weights = weights.reindex(normalized.columns)
        if weights.isna().any():
            missing = list(weights[weights.isna()].index)
            raise ValueError(f"外部权重缺少这些列：{missing}")
        weights = weights / weights.sum()
    score = normalized.mul(weights, axis=1).sum(axis=1)
    return score, weights
