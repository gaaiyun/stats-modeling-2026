"""中文政策文本的特征提取：分词、按维度词频、TF-IDF、关键词共现。

不引入重型 NLP 依赖，只用 ``jieba`` 分词 + ``scikit-learn`` 的 TF-IDF。
所有函数都接受「文档列表」（每篇报告一段字符串），返回 pandas 结构，方便
下游直接画图或并入面板。
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

import jieba
import pandas as pd

__all__ = [
    "tokenize",
    "dimension_term_frequency",
    "tfidf_matrix",
    "cooccurrence",
]

# 政策文本里高频但无信息量的功能词 / 套话，做停用处理
_DEFAULT_STOPWORDS = {
    "我们", "我省", "我市", "全省", "全市", "全年", "去年", "今年", "以来",
    "进一步", "扎实", "持续", "深入", "积极", "不断", "切实", "大力", "加快",
    "推进", "推动", "加强", "实现", "工作", "发展", "建设", "支持", "提升",
    "推动", "促进", "做好", "确保", "完善", "提高", "继续", "坚持", "围绕",
    "各位", "代表", "报告", "国务院", "政府", "情况", "方面", "有关", "以及",
    "亿元", "万元", "增长", "同比", "左右", "其中", "等等", "目前", "重点",
}

# 仅保留 2 字及以上的中文词
_CN_WORD = re.compile(r"^[一-龥]{2,}$")


def tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    """分词并过滤：仅保留 ≥2 字的中文词，去停用词。"""
    stop = _DEFAULT_STOPWORDS if stopwords is None else stopwords
    words = jieba.lcut(text)
    return [w for w in words if _CN_WORD.match(w) and w not in stop]


def dimension_term_frequency(
    text: str,
    dimension_keywords: dict[str, list[str]],
) -> dict[str, int]:
    """统计每个维度的种子关键词在文本中的出现总次数。

    用于和 LLM 评分做交叉验证 / 稳健性对照（文本里关键词越多，
    理应与 LLM 给出的该维度得分正相关）。
    """
    counts: dict[str, int] = {}
    for dim, kws in dimension_keywords.items():
        total = 0
        for kw in kws:
            total += text.count(kw)
        counts[dim] = total
    return counts


def tfidf_matrix(
    docs: list[str],
    doc_labels: list[str] | None = None,
    top_k: int = 30,
    stopwords: set[str] | None = None,
) -> pd.DataFrame:
    """计算文档集的 TF-IDF，返回每篇文档 top_k 关键词的稀疏长表。

    Returns
    -------
    pandas.DataFrame
        列：``doc``, ``term``, ``tfidf``，按文档和 tfidf 降序。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not docs:
        raise ValueError("文档列表为空。")
    labels = doc_labels or [f"doc{i}" for i in range(len(docs))]
    if len(labels) != len(docs):
        raise ValueError("doc_labels 数量与文档数不一致。")

    tokenized = [" ".join(tokenize(d, stopwords=stopwords)) for d in docs]
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    mat = vec.fit_transform(tokenized)
    vocab = vec.get_feature_names_out()

    rows = []
    for i, label in enumerate(labels):
        row = mat[i].toarray().ravel()
        if row.size == 0:
            continue
        order = row.argsort()[::-1][:top_k]
        for j in order:
            if row[j] <= 0:
                break
            rows.append({"doc": label, "term": vocab[j], "tfidf": float(row[j])})
    return pd.DataFrame(rows, columns=["doc", "term", "tfidf"])


def cooccurrence(
    text: str,
    vocab: list[str],
    window: int = 1,
    stopwords: set[str] | None = None,
) -> pd.DataFrame:
    """关键词共现矩阵（句子级）。

    把文本按标点切句，统计 ``vocab`` 中任意两词在同一句出现的次数。
    ``window`` 暂保留接口（句子级共现忽略它），便于以后扩展为滑动窗口。

    Returns
    -------
    pandas.DataFrame
        对称的方阵，行列均为 ``vocab``，值为共现次数。
    """
    vocab_set = set(vocab)
    sentences = re.split(r"[。！？；\n]", text)
    pair_counts: Counter = Counter()
    for sent in sentences:
        present = sorted({w for w in vocab_set if w in sent})
        for a, b in combinations(present, 2):
            pair_counts[(a, b)] += 1

    mat = pd.DataFrame(0, index=vocab, columns=vocab, dtype=int)
    for (a, b), n in pair_counts.items():
        mat.loc[a, b] = n
        mat.loc[b, a] = n
    return mat
