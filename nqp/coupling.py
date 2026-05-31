"""耦合协调度模型（coupling coordination degree）。

衡量两个或多个子系统（这里是「新质生产力水平 NQP」与「高质量发展水平 HQD」）
相互作用的协调程度。文献里的标准三步：

1. 耦合度  C = n * (∏ U_i)^(1/n) / (Σ U_i)，n 为子系统个数，U_i ∈ [0, 1]。
   两系统时退化为常见形式 C = 2 √(U1 U2) / (U1 + U2)。C 高只说明两系统“咬合紧”，
   不代表水平高（两个都很低也可能 C≈1）。
2. 综合协调指数  T = Σ α_i U_i，α_i 为子系统重要性权重（默认等权）。
3. 耦合协调度  D = √(C · T)，D ∈ [0, 1]，兼顾“咬合”与“水平”。

等级划分采用应用最广的十分类（廖重斌 1999），从「极度失调」到「优质协调」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "coupling_degree",
    "coordination_index",
    "coupling_coordination",
    "classify_coordination",
    "COORDINATION_BINS",
    "COORDINATION_LABELS",
]

# 廖重斌 (1999)《环境与经济协调发展的定量评判及其分类体系》的十级划分
COORDINATION_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
COORDINATION_LABELS = [
    "极度失调",
    "严重失调",
    "中度失调",
    "轻度失调",
    "濒临失调",
    "勉强协调",
    "初级协调",
    "中级协调",
    "良好协调",
    "优质协调",
]


def _as_2d(*systems: pd.Series | np.ndarray) -> np.ndarray:
    """把若干等长子系统序列堆成 (n_samples, n_systems) 矩阵，并做取值检查。"""
    arrs = [np.asarray(s, dtype=float) for s in systems]
    lengths = {a.shape[0] for a in arrs}
    if len(lengths) != 1:
        raise ValueError(f"各子系统长度不一致：{[a.shape[0] for a in arrs]}")
    mat = np.column_stack(arrs)
    if np.any(mat < 0) or np.any(mat > 1):
        raise ValueError("子系统取值必须先归一化到 [0, 1]。")
    return mat


def coupling_degree(*systems: pd.Series | np.ndarray) -> np.ndarray:
    """耦合度 C ∈ [0, 1]。

    C = n * (∏ U_i)^(1/n) / (Σ U_i)。任一子系统为 0 时几何平均为 0，C=0。
    所有子系统全为 0 时分子分母同为 0，约定 C=0（无耦合）。
    """
    mat = _as_2d(*systems)
    n = mat.shape[1]
    if n < 2:
        raise ValueError("耦合度至少需要 2 个子系统。")
    geo = np.prod(mat, axis=1) ** (1.0 / n)   # 几何平均
    denom = mat.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(denom > 0, n * geo / denom, 0.0)
    return c


def coordination_index(
    *systems: pd.Series | np.ndarray,
    weights: list[float] | None = None,
) -> np.ndarray:
    """综合协调指数 T = Σ α_i U_i。默认子系统等权。"""
    mat = _as_2d(*systems)
    n = mat.shape[1]
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != n:
            raise ValueError(f"权重个数 {w.shape[0]} 与子系统个数 {n} 不符。")
        if w.sum() <= 0:
            raise ValueError("权重之和必须为正。")
        w = w / w.sum()
    return mat @ w


def coupling_coordination(
    *systems: pd.Series | np.ndarray,
    weights: list[float] | None = None,
) -> np.ndarray:
    """耦合协调度 D = √(C · T) ∈ [0, 1]。"""
    c = coupling_degree(*systems)
    t = coordination_index(*systems, weights=weights)
    return np.sqrt(c * t)


def classify_coordination(d: pd.Series | np.ndarray | float):
    """把耦合协调度 D 映射到十级文字等级。

    标量返回字符串；序列返回与输入等长的字符串数组。
    """
    scalar = np.isscalar(d)
    arr = np.atleast_1d(np.asarray(d, dtype=float))
    # 右闭区间：0.1 归入「极度失调/严重失调」边界时落在上一档，与文献一致
    idx = np.digitize(arr, COORDINATION_BINS[1:-1], right=True)
    idx = np.clip(idx, 0, len(COORDINATION_LABELS) - 1)
    labels = np.array(COORDINATION_LABELS)[idx]
    return labels[0] if scalar else labels
