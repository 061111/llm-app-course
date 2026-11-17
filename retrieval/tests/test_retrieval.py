"""M3 检索层的测试。全部离线：BM25 是纯函数，翻译用注入的假函数。

重点测三件事：
  1. BM25 的排序是否符合它的公式（不是"能跑"）。
  2. 翻译缓存是否真的省调用、且 refresh 能穿透缓存。
  3. 检索器在"没翻译"时对中文问题返回空 —— 这是跨语言问题的证据，
     如果哪天它意外返回了结果，说明 tokenize 出错了。
"""
from __future__ import annotations

import pytest

from corpus.chunker import Chunk, PageSpan
from retrieval.bm25 import BM25, index_text, tokenize
from retrieval.eval_retrieval import ItemHit, Report
from retrieval.query import Translator
from retrieval.search import Searcher


def mk(i: int, doc: str, page: int, title: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc}#{i:03d}",
        doc_id=doc,
        title=title,
        page_start=page,
        page_end=page,
        text=text,
        spans=(PageSpan(page=page, start=0, end=len(text)),),
        is_empty=False,
    )


CORPUS = [
    mk(0, "a", 1, "Rule Ordering Schemes", "Rule-based ordering ranks rules by quality. Class-based ordering sorts by class."),
    mk(1, "a", 2, "Gain Ratio", "GainRatio is Gain_split divided by SplitInfo. Used in C4.5. High entropy partitioning is penalized."),
    mk(2, "a", 3, "Confusion Matrix", "False Positive is Type I Error. False Negative is Type II Error."),
]


# ---------------------------------------------------------------- 分词


def test_分词小写化并去停用词():
    assert tokenize("The Rule is OF quality") == ["rule", "quality"]


def test_分词丢掉单字母():
    # 教材里 k、n、v 是符号，但太短太常见，留着会污染公式页
    assert tokenize("k nearest n v") == ["nearest"]


def test_分词保留小数():
    assert tokenize("Gini = 0.278 and 0.444") == ["gini", "0.278", "0.444"]


def test_分词去掉分音符():
    # 讲义里写的是 "Naïve"（带分音符），用户查询写的是 "naive"。
    # 不去音符两者是不同的词元，实测 p17/p18 里 "naive" 的词频是 0，
    # 命中全靠 "bayes" 一个词撑着。
    assert "naive" in tokenize("Naïve Bayes classifier")


def test_分词还原数学斜体():
    # chap03 p59 的公式里 "Gini Index" 是数学斜体 𝐺𝑖𝑛𝑖𝐼𝑛𝑑𝑒𝑥。归一化后：
    #   "𝐺𝑖𝑛𝑖= 1"  → "Gini= 1"  → 切出 gini  ✓（实测 p59 里 gini 出现 5 次）
    #   "𝐺𝑖𝑛𝑖𝐼𝑛𝑑𝑒𝑥" → "GiniIndex" → 仍是一个词 giniindex  ✗（已知残留，粘连无法靠规则拆开）
    assert "gini" in tokenize("𝐺𝑖𝑛𝑖= 1")
    assert tokenize("𝐺𝑖𝑛𝑖𝐼𝑛𝑑𝑒𝑥") == ["giniindex"]


def test_索引文本包含标题():
    assert index_text("Gain Ratio", "正文") == "Gain Ratio\n正文"
    assert index_text("", "正文") == "正文"


# ---------------------------------------------------------------- BM25


def test_罕见词的_idf_高于常见词():
    idx = BM25([tokenize(c.text) for c in CORPUS])
    assert idx.idf("confusion") > idx.idf("ordering")


def test_命中两个词的块排第一():
    s = Searcher(CORPUS)
    results = s.search_terms(["gain ratio", "C4.5"], top_k=3)
    assert results[0].chunk.chunk_id == "a#001"
    assert results[0].rank == 1


def test_没有词的块不出现在结果里():
    s = Searcher(CORPUS)
    results = s.search_terms(["confusion"], top_k=5)
    assert [r.chunk.chunk_id for r in results] == ["a#002"], "零分块不该进结果"


def test_长文档同词频时分数更低():
    # b=0.75 的长度归一化：同样出现 1 次，短文档应该分更高
    short = tokenize("gain ratio")
    long = tokenize("gain ratio " + "filler " * 100)
    idx = BM25([short, long])
    assert idx.score(["gain"], 0) > idx.score(["gain"], 1)


def test_top_k_被遵守():
    s = Searcher(CORPUS)
    assert len(s.search_terms(["ordering", "class", "gain", "confusion"], top_k=2)) == 2


def test_同一查询两次结果一致():
    s = Searcher(CORPUS)
    a = [r.chunk.chunk_id for r in s.search_terms(["rule", "class"], top_k=3)]
    b = [r.chunk.chunk_id for r in s.search_terms(["rule", "class"], top_k=3)]
    assert a == b


# ---------------------------------------------------------------- 翻译与缓存


def make_translator(tmp_path, calls: list[str]):
    def fake(question: str) -> list[str]:
        calls.append(question)
        return ["confusion matrix", "Type I Error"]

    return Translator(cache_path=tmp_path / "cache.json", call_fn=fake)


def test_第二次同问题命中缓存不调用模型(tmp_path):
    calls: list[str] = []
    t = make_translator(tmp_path, calls)
    assert t("问题一") == ["confusion matrix", "Type I Error"]
    assert t("问题一") == ["confusion matrix", "Type I Error"]
    assert len(calls) == 1
    assert t.calls == 1


def test_缓存会落盘并被新实例读到(tmp_path):
    calls: list[str] = []
    make_translator(tmp_path, calls)("问题一")
    fresh = make_translator(tmp_path, calls)
    assert fresh("问题一") == ["confusion matrix", "Type I Error"]
    assert len(calls) == 1, "新实例应该直接读文件，不该再调模型"


def test_refresh_能穿透缓存(tmp_path):
    calls: list[str] = []
    make_translator(tmp_path, calls)("问题一")
    forced = Translator(cache_path=tmp_path / "cache.json", call_fn=lambda q: ["forced"], refresh=True)
    assert forced("问题一") == ["forced"]
    assert forced.calls == 1


def test_翻译结果里的中文和停用词被清掉(tmp_path):
    t = Translator(cache_path=tmp_path / "c.json", call_fn=lambda q: ["混淆矩阵", "the", "confusion"])
    assert t("问题") == ["confusion"]


def test_清洗后为空时报错而不是返回空查询(tmp_path):
    t = Translator(cache_path=tmp_path / "c.json", call_fn=lambda q: ["混淆矩阵"])
    with pytest.raises(ValueError, match="清洗后为空"):
        t("问题")


def test_坏掉的缓存文件不影响启动(tmp_path):
    p = tmp_path / "cache.json"
    p.write_text("{not json", encoding="utf-8")
    t = Translator(cache_path=p, call_fn=lambda q: ["ok"])
    assert t("问题") == ["ok"]


# ---------------------------------------------------------------- 检索器


def test_纯中文问题不翻译时检索不到东西():
    # 这不是缺陷，是跨语言问题本身：中文词元对不上英文语料。
    # 注意必须是**纯中文**的问题 —— 黄金集里有一半题目夹着英文术语
    # （"Type I Error"、"C4.5"），那些题不翻译也能蒙中几个词，
    # 所以"不翻译 = 全灭"是错的，只能对纯中文句成立。
    s = Searcher(CORPUS)
    assert s.search("混淆矩阵里第一类错误对应哪一格", translate=False) == []


def test_夹着英文术语时中文问题能蒙中部分():
    # 记录这个事实，防止以后有人把"不翻译也有分"当成翻译无效的证据
    s = Searcher(CORPUS)
    results = s.search("Type I Error 对应哪一格", translate=False)
    assert results, "含英文术语的中文问题会被字面命中"


def test_翻译后能命中():
    s = Searcher(CORPUS, translator=lambda q: ["confusion matrix", "Type I Error"])
    results = s.search("混淆矩阵里 Type I Error 对应哪一格", top_k=3)
    assert results[0].chunk.chunk_id == "a#002"


def test_没配翻译器却要求翻译时报错():
    s = Searcher(CORPUS)
    with pytest.raises(ValueError, match="translator"):
        s.search("问题", translate=True)


def test_空块列表直接报错():
    with pytest.raises(ValueError, match="为空"):
        Searcher([])


# ---------------------------------------------------------------- 指标


def test_hit_at_和_mrr_计算正确():
    items = (
        ItemHit(gold_id="G01", rank=1, top1="x"),
        ItemHit(gold_id="G02", rank=3, top1="x"),
        ItemHit(gold_id="G03", rank=0, top1="x"),
    )
    r = Report(mode="t", top_k=5, items=items)
    assert r.hit_at[1] == 1
    assert r.hit_at[3] == 2
    assert r.hit_at[5] == 2
    assert r.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert r.misses == ["G03"]


def test_排名超过_top_k_不算命中():
    items = (ItemHit(gold_id="G01", rank=9, top1="x"),)
    assert Report(mode="t", top_k=5, items=items).hit_at[5] == 0
