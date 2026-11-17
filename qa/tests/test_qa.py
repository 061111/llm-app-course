"""M4 的测试。全部离线：假客户端 + 假块，不花一分钱。

重点：
  1. 引用解析（含跨页、倒序、重复）。
  2. **引用越界检测** —— 这是 M4 唯一能机械核对的正确性维度，必须测准。
  3. prompt 里确实带上了出处头（否则模型没法引用，M2 的 header 就白做了）。
"""
from __future__ import annotations

import pytest

from corpus.chunker import Chunk, PageSpan
from llmkit.config import Settings
from llmkit.tests.fakes import FakeClient, make_completion
from qa.answer import Answer, answer_question, parse_citations, unsupported_citations
from qa.grading import Verdict, numeric_hits
from qa.prompt import build_context, build_prompt
from retrieval.search import SearchResult, Searcher


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


CHUNKS = [
    mk(0, "chap03", 70, "Computing Entropy of a Single Node", "Entropy = 0.65 for (1,5)"),
    mk(1, "chap03", 57, "Measure of Impurity: GINI", "Gini index used in CART, SLIQ, SPRINT"),
]

SETTINGS = Settings(api_key="test-key", base_url="http://fake", model="fake-model")


def fake_searcher(translator=None) -> Searcher:
    return Searcher(CHUNKS, translator=translator or (lambda q: ["entropy"]))


# ---------------------------------------------------------------- 引用解析


def test_解析单页引用():
    refs = parse_citations("结论是 0.65【chap03:p70】。")
    assert len(refs) == 1
    assert refs[0].source() == "chap03:p70"


def test_解析跨页引用():
    refs = parse_citations("两条规则各占一段【chap04a:p12-13】")
    assert refs[0].page_start == 12
    assert refs[0].page_end == 13
    assert refs[0].source() == "chap04a:p12-13"


def test_页序写反时纠正():
    refs = parse_citations("【chap04a:p13-12】")
    assert (refs[0].page_start, refs[0].page_end) == (12, 13)


def test_重复引用去重():
    refs = parse_citations("【chap03:p70】又说了一遍【chap03:p70】")
    assert len(refs) == 1


def test_没有引用时返回空():
    assert parse_citations("讲义未覆盖，无法回答。") == ()


def test_格式不对的引用不被识别():
    # 只认 prompt 里规定死的格式；"第 70 页"这种无法机械核对
    assert parse_citations("见第 70 页") == ()


# ---------------------------------------------------------------- 引用越界


def test_引用在检索到的块内算合法():
    results = fake_searcher().search_terms(["entropy"], top_k=2)
    refs = parse_citations("【chap03:p70】")
    assert unsupported_citations(refs, results) == ()


def test_引用未检索到的页算越界():
    results = fake_searcher().search_terms(["entropy"], top_k=2)
    refs = parse_citations("【chap05:p3】")
    assert unsupported_citations(refs, results) == ("chap05:p3",)


def test_只引用跨页块中的一页也算合法():
    # 块覆盖 p12-13，模型只写【chap04a:p13】应当接受
    chunk = Chunk(
        chunk_id="chap04a#000",
        doc_id="chap04a",
        title="Rule Ordering Schemes",
        page_start=12,
        page_end=13,
        text="正文",
        spans=(PageSpan(page=12, start=0, end=2),),
        is_empty=False,
    )
    results = [
        SearchResult(chunk=chunk, score=1.0, rank=1, terms=("rule",)),
    ]
    assert unsupported_citations(parse_citations("【chap04a:p13】"), results) == ()


def test_区间引用跨两个独立块时合法():
    # 实测反例：模型写【chap03:p20-21】，p20、p21 是两个独立块但都在结果里。
    # 按"单块必须覆盖整个区间"判，这里会误报越界。
    def one(page: int) -> SearchResult:
        c = Chunk(
            chunk_id=f"chap03#{page}",
            doc_id="chap03",
            title=f"T{page}",
            page_start=page,
            page_end=page,
            text="正文",
            spans=(PageSpan(page=page, start=0, end=2),),
            is_empty=False,
        )
        return SearchResult(chunk=c, score=1.0, rank=1, terms=("x",))

    results = [one(20), one(21)]
    assert unsupported_citations(parse_citations("【chap03:p20-21】"), results) == ()


def test_区间里有一页没检索到就是越界():
    # 实测反例：模型写【chap04a:p27-28】，但只检索到 p27
    c = Chunk(
        chunk_id="chap04a#1",
        doc_id="chap04a",
        title="T",
        page_start=27,
        page_end=27,
        text="正文",
        spans=(PageSpan(page=27, start=0, end=2),),
        is_empty=False,
    )
    results = [SearchResult(chunk=c, score=1.0, rank=1, terms=("x",))]
    assert unsupported_citations(parse_citations("【chap04a:p27-28】"), results) == (
        "chap04a:p27-28",
    )


# ---------------------------------------------------------------- prompt


def test_context_带出处头():
    results = fake_searcher().search_terms(["entropy"], top_k=1)
    ctx = build_context(results)
    assert "【chap03:p70 · Computing Entropy of a Single Node】" in ctx
    assert "0.65" in ctx


def test_prompt_包含问题与规则():
    results = fake_searcher().search_terms(["entropy"], top_k=2)
    p = build_prompt("熵是多少", results)
    assert "问题：熵是多少" in p
    assert "讲义未覆盖" in p
    assert "【doc_id:p页码】" in p


def test_context_超长时截断且至少保留一块():
    results = fake_searcher().search_terms(["entropy"], top_k=2)
    ctx = build_context(results, max_chars=10)
    assert ctx, "不能截成空上下文"


# ---------------------------------------------------------------- 端到端


def test_answer_question_串起全链路():
    client = FakeClient([make_completion("(1,5) 的熵约 0.65【chap03:p70】")])
    ans = answer_question(
        "熵是多少", searcher=fake_searcher(), client=client, settings=SETTINGS
    )
    assert isinstance(ans, Answer)
    assert ans.cited_sources == ("chap03:p70",)
    assert ans.unsupported == ()
    assert ans.retrieved[0] == "chap03:p70"
    # prompt 真的进了模型：检查请求体里带上了片段。
    # messages 是 [system, user]，片段在最后一条（user）里，不在 system 里。
    body = client.calls[0]["messages"][-1]["content"]
    assert "Computing Entropy" in body
    assert "问题：熵是多少" in body


def test_模型编造出处会被抓出来():
    client = FakeClient([make_completion("答案见【chap05:p99】")])
    ans = answer_question("熵是多少", searcher=fake_searcher(), client=client, settings=SETTINGS)
    assert ans.unsupported == ("chap05:p99",)


def test_承认讲义未覆盖能被识别():
    client = FakeClient([make_completion("讲义未覆盖这个问题。")])
    ans = answer_question("课本外的问题", searcher=fake_searcher(), client=client, settings=SETTINGS)
    assert ans.claims_uncovered is True
    assert ans.citations == ()


# ---------------------------------------------------------------- 判分


def test_数值核对支持等价写法():
    ok, missing = numeric_hits("support=2/5=0.4, confidence=2/3", (("0.4", "2/5"), ("0.67", "2/3")))
    assert ok == 2
    assert missing == ()


def test_数值核对抓出错值():
    # 0.86 是照抄印刷错误公式得到的值，必须判成缺
    ok, missing = numeric_hits("熵是 0.86", (("0.65",),))
    assert ok == 0
    assert missing == ("0.65",)


def test_判卷结论只能是四档之一():
    with pytest.raises(Exception):
        Verdict(verdict="excellent", reason="x")
