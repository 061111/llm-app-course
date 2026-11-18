"""中文分词：既要"中文能切"，又要"英文一点没变"。

第二条比第一条重要 —— 加中文能力时最容易踩的坑是把英文切坏了，
而英文那 283 条测试跑的是上层逻辑，未必能抓到分词粒度的细微变化。
"""
from __future__ import annotations

from retrieval.bm25 import tokenize
from retrieval.cjk import cjk_ratio, cjk_terms, has_cjk


def test_英文分词与改造前逐词一致():
    # 这是加中文之前的输出，一个词都不能变
    assert tokenize("Computing the Gini Index of a Single Node") == [
        "computing",
        "gini",
        "index",
        "single",
        "node",
    ]


def test_英文数字与符号仍按原规则():
    assert tokenize("Gini = 0.278, k-NN with n=14") == ["gini", "0.278", "nn", "14"]


def test_中文被切成二元组():
    assert tokenize("朴素贝叶斯") == ["朴素", "素贝", "贝叶", "叶斯"]


def test_中文问题与中文正文有交集():
    """改造前这条是空集 —— 那正是中文一条都搜不到的原因。"""
    q = set(tokenize("贝叶斯分类器的核心假设是什么"))
    d = set(tokenize("朴素贝叶斯分类器假设各属性在给定类别下条件独立"))
    assert q & d


def test_全停用字的二元组被丢掉():
    assert "的是" not in tokenize("这是一个的是")


def test_含实词的二元组保留():
    grams = tokenize("的核")
    assert grams == ["的核"], "只有两个字都为功能字时才丢"


def test_单字中文术语保留():
    assert tokenize("熵") == ["熵"]
    assert tokenize("的") == []


def test_中英混排两边都切():
    grams = tokenize("用 C4.5 计算信息熵")
    assert "4.5" in grams  # C 是单字母被丢，4.5 是既有行为
    assert "信息" in grams and "息熵" in grams


def test_cjk_terms_空串():
    assert cjk_terms("") == []


def test_汉字占比的实测分档():
    # 纯中文 → 1.0；纯英文 → 0.0；中英混排落在中间
    assert cjk_ratio("多元统计分析") == 1.0
    assert cjk_ratio("multivariate analysis") == 0.0
    assert 0.15 < cjk_ratio("统计 analysis") < 0.25, "2 个汉字 / (2+8) 个字母"


def test_has_cjk():
    assert has_cjk("第3章")
    assert not has_cjk("Slide 38: Apriori")
