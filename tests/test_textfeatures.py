"""nqp.textfeatures 单元测试：分词、维度词频、TF-IDF、共现。"""
from __future__ import annotations

import pytest

from nqp.textfeatures import (
    cooccurrence,
    dimension_term_frequency,
    tfidf_matrix,
    tokenize,
)


def test_tokenize_drops_short_and_stopwords():
    toks = tokenize("我们要加快建设现代化产业体系，推动科技创新。")
    # “我们/建设/推动/加快”是停用词，单字被过滤
    assert "现代化" in toks or "产业" in toks
    assert "我们" not in toks
    assert all(len(t) >= 2 for t in toks)


def test_dimension_term_frequency_counts():
    text = "加强关键核心技术攻关，关键核心技术取得突破；发展数字经济。"
    kws = {"tech": ["关键核心技术", "攻关"], "digital": ["数字经济"]}
    freq = dimension_term_frequency(text, kws)
    assert freq["tech"] == 3  # “关键核心技术”×2 + “攻关”×1
    assert freq["digital"] == 1


def test_tfidf_matrix_shape_and_columns():
    docs = [
        "科技创新 关键核心技术 基础研究 原创性突破 攻关",
        "绿色低碳 双碳目标 清洁能源 碳达峰 碳中和",
        "数字经济 人工智能 算力 数据要素 工业互联网",
    ]
    df = tfidf_matrix(docs, doc_labels=["a", "b", "c"], top_k=3)
    assert set(df.columns) == {"doc", "term", "tfidf"}
    assert set(df["doc"].unique()) == {"a", "b", "c"}
    assert (df["tfidf"] > 0).all()
    # 每篇最多 top_k 个词
    assert df.groupby("doc").size().max() <= 3


def test_tfidf_empty_raises():
    with pytest.raises(ValueError):
        tfidf_matrix([])


def test_cooccurrence_symmetric():
    text = "科技创新 和 产业升级 同句出现。绿色低碳 单独一句。"
    mat = cooccurrence(text, vocab=["科技创新", "产业升级", "绿色低碳"])
    assert mat.loc["科技创新", "产业升级"] == 1
    assert mat.loc["产业升级", "科技创新"] == 1  # 对称
    assert mat.loc["科技创新", "绿色低碳"] == 0  # 不同句
    # 对角线为 0（不和自己共现）
    assert mat.loc["科技创新", "科技创新"] == 0
