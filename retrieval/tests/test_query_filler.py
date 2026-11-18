"""M5 补丁验收：问句剥离后，同一个问题换三种问法应当检索出同一批证据。

这条规则的来历是一次实测：30 页中文资料上，同一个术语用
「什么是X」「X的定义是什么」「X是如何定义的」三种问法检索，
取回的前 3 页几乎不重合（Jaccard 0.235）。原因不是 BM25 坏了，
而是 bigram 把问法本身切成了查询词 —— "什么""如何""义的"在正文里到处都是。

所以这里测两件事：
  1. 剥离后三种问法的词元集合完全相同（这是"同问法同结果"的必要条件）；
  2. 剥离不误伤实词（"定义与定理的区别"里的"定义"必须保留）。
"""
from __future__ import annotations

from retrieval.bm25 import tokenize
from retrieval.cjk import strip_query_filler


def test_三种问法剥离后词元完全相同():
    variants = ("什么是主成分", "主成分的定义是什么", "主成分是如何定义的")
    token_sets = {tuple(tokenize(strip_query_filler(q))) for q in variants}
    assert len(token_sets) == 1, f"三种问法切出了不同词元：{token_sets}"
    assert token_sets == {("主成", "成分")}


def test_英文查询原样不动():
    assert strip_query_filler("what is PCA") == "what is PCA"


def test_全是套话时回退原文():
    # 清空查询会让 BM25 召回空结果，看起来像"资料里没有"，实际是查询被洗没了
    assert strip_query_filler("是什么？") == "是什么？"


def test_不删实词():
    # 定义 / 含义 是教材里的章节标题，删了会让这类问题失去一半检索词
    assert "定义" in strip_query_filler("定义与定理的区别")
    assert "含义" in strip_query_filler("含义和用法")


def test_问句里的术语原样保留():
    assert strip_query_filler("为什么信息增益偏向多值属性") == "信息增益偏向多值属性"
