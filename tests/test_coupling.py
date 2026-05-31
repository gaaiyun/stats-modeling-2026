"""nqp.coupling 单元测试：耦合度 C、协调指数 T、耦合协调度 D、等级划分。"""
from __future__ import annotations

import numpy as np
import pytest

from nqp.coupling import (
    classify_coordination,
    coordination_index,
    coupling_coordination,
    coupling_degree,
)


def test_coupling_max_when_systems_equal():
    # 两系统相等时耦合度 C = 1（咬合最紧）
    u = np.array([0.3, 0.6, 0.9])
    c = coupling_degree(u, u)
    np.testing.assert_allclose(c, 1.0, atol=1e-12)


def test_coupling_zero_when_one_system_zero():
    c = coupling_degree(np.array([0.0, 0.5]), np.array([0.8, 0.0]))
    # 任一子系统为 0，几何平均为 0 -> C = 0
    np.testing.assert_allclose(c, 0.0, atol=1e-12)


def test_coupling_two_system_closed_form():
    # 对照解析式 C = 2√(U1 U2)/(U1+U2)
    u1 = np.array([0.2, 0.7])
    u2 = np.array([0.8, 0.3])
    expected = 2 * np.sqrt(u1 * u2) / (u1 + u2)
    np.testing.assert_allclose(coupling_degree(u1, u2), expected, atol=1e-12)


def test_coupling_rejects_out_of_range():
    with pytest.raises(ValueError):
        coupling_degree(np.array([1.5]), np.array([0.5]))


def test_coupling_rejects_single_system():
    with pytest.raises(ValueError):
        coupling_degree(np.array([0.5, 0.6]))


def test_coordination_index_equal_weight_is_mean():
    u1 = np.array([0.4, 0.8])
    u2 = np.array([0.6, 0.2])
    np.testing.assert_allclose(coordination_index(u1, u2), (u1 + u2) / 2)


def test_coordination_index_custom_weights():
    u1 = np.array([1.0, 0.0])
    u2 = np.array([0.0, 1.0])
    t = coordination_index(u1, u2, weights=[3.0, 1.0])  # 归一为 0.75 / 0.25
    np.testing.assert_allclose(t, [0.75, 0.25])


def test_coupling_coordination_bounded_and_le_coordination():
    rng = np.random.default_rng(2)
    u1 = rng.random(100)
    u2 = rng.random(100)
    d = coupling_coordination(u1, u2)
    assert np.all(d >= 0) and np.all(d <= 1)
    # D = √(C·T) ≤ √T ≤ max(T, ...)，且 C ≤ 1 -> D ≤ √T
    t = coordination_index(u1, u2)
    assert np.all(d <= np.sqrt(t) + 1e-12)


def test_coupling_coordination_perfect():
    # 两系统都为 1：C=1, T=1, D=1
    d = coupling_coordination(np.array([1.0]), np.array([1.0]))
    np.testing.assert_allclose(d, 1.0, atol=1e-12)


def test_classify_scalar_and_vector():
    assert classify_coordination(0.95) == "优质协调"
    assert classify_coordination(0.05) == "极度失调"
    assert classify_coordination(0.55) == "勉强协调"
    labels = classify_coordination(np.array([0.05, 0.55, 0.95]))
    assert list(labels) == ["极度失调", "勉强协调", "优质协调"]


def test_classify_boundaries():
    # 边界值落档：0.6 -> 初级协调 (0.5,0.6] 上沿
    assert classify_coordination(0.6) == "勉强协调"
    assert classify_coordination(0.61) == "初级协调"
