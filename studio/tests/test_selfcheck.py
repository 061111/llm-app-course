"""M6-T1 验收：领域无关自检的四个判据是否真的机械可判。

这里**不测语义对不对** —— 那是自检本身也做不到的事（见 studio/selfcheck.py 的说明）。
测的是"判定逻辑有没有按设计工作"：
  · 随机串真的不在正文里（否则"编造"这一项就失效了）
  · 引用页上真的有那个词才算通过，张冠李戴要被抓出来
  · 资料外的问题被编出来时，必须记 fabricated 并把红绿灯降到"有风险"
  · 三种问法落在同一页才算稳定

用假 studio 喂死数据，不联网、不花钱。
"""
from __future__ import annotations

import random
from types import SimpleNamespace

from llmkit.types import Usage
from qa.answer import parse_citations
from studio.glossary import Term
from studio.selfcheck import (
    make_nonce,
    pick_terms,
    run_selfcheck,
    squash,
    term_on_pages,
)
from studio.service import Answer, AnswerView, Evidence

DOC = "u001"


def _view(text: str, *, with_evidence: bool = True, cost: float = 0.0) -> AnswerView:
    """拼一个真实的 AnswerView（pydantic 对象，不走网络）。

    with_evidence=False 用来模拟"检索什么都没捞到"—— AnswerView.denied 的
    第一条判据就是"证据为空即拒答"。
    """
    answer = Answer(
        question="q",
        text=text,
        citations=parse_citations(text),
        retrieved=(),
        unsupported=(),
        usage=Usage(cost_cny=cost),
    )
    evidence = (
        (Evidence(source=f"{DOC}:p1", title="t", score=1.0, text="x"),)
        if with_evidence
        else ()
    )
    return AnswerView(
        answer=answer, evidence=evidence, doc_ids=(DOC,), top_k=5, translated=False
    )


class FakeSearcher:
    """只回答"首条命中在哪一页"，不稳定性的判定只需要这一个数。"""

    def __init__(self, page_by_question: dict[str, int | None]) -> None:
        self.page_by_question = page_by_question

    def search(self, question: str, *, top_k: int = 5, translate: bool = True) -> list:
        page = self.page_by_question.get(question)
        if page is None:
            return []
        return [SimpleNamespace(chunk=SimpleNamespace(page_start=page))]


class FakeStudio:
    def __init__(
        self,
        pages: dict[int, str],
        terms: list[Term],
        answers,
        pages_by_question: dict[str, int | None] | None = None,
    ) -> None:
        self._pages = pages
        self._terms = terms
        self._answers = answers
        self._searcher = FakeSearcher(pages_by_question or {})

    def pages(self, doc_id: str) -> list:
        return [SimpleNamespace(page=p, text=t) for p, t in self._pages.items()]

    def terms(self, doc_id: str, top_n: int = 40) -> list[Term]:
        return self._terms

    def searcher(self, doc_ids: list[str]):
        return self._searcher

    def ask(
        self,
        question: str,
        *,
        doc_ids=None,
        top_k: int = 5,
        model: str | None = None,
    ) -> AnswerView:
        return self._answers(question)

    def settings_for(self, model: str | None = None):
        """自检要记录"这轮用的哪个模型"，所以替身也得有这个接口。

        替身不实现真实接口，测试就会在"自检报错"上红——而红的原因跟自检无关。
        """
        return SimpleNamespace(model=model or "fake-model")


def _term(text: str = "方差") -> Term:
    return Term(text=text, freq=5, score=1.0, occurrences=())


def test_squash_压掉空白并转小写():
    assert squash("Gini  In\ndex") == "giniindex"
    assert squash("  主成分  ") == "主成分"


def test_术语在被引页上才算可核验():
    pages = {1: "方差分析的基本假设", 2: "与本题无关的正文"}
    assert term_on_pages(pages, "方差", 1, 1)
    assert term_on_pages(pages, "方差", 1, 2), "区间引用只要有一页命中就算通过"
    assert not term_on_pages(pages, "方差", 2, 2)
    assert not term_on_pages(pages, "", 1, 2), "空术语不能算命中"


def test_随机串保证不在正文里():
    rng = random.Random(7)
    corpus = "这段正文里有一堆汉字和若干 English words，但不会有随机标识符。"
    token = make_nonce(rng, corpus)
    assert len(token) == 6
    assert token.lower() not in squash(corpus)
    assert make_nonce(random.Random(7), corpus) == token, "同种子应当可复现"


def test_随机串撞上正文会重新生成():
    rng = random.Random(1)
    first = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
    # 把这个串塞进正文，生成器必须换一个
    token = make_nonce(random.Random(1), f"正文里恰好出现了 {first} 这个串")
    assert token != first


def test_等距抽样覆盖整份文档():
    terms = [_term(f"术语{i}") for i in range(40)]
    picked = pick_terms(terms, 5)
    assert len(picked) == 5
    assert picked[0].text == "术语0"
    assert picked[-1].text == "术语39", "等距抽样要能取到文档末尾的术语"
    assert len({t.text for t in picked}) == 5
    assert pick_terms(terms, 0) == []
    assert pick_terms([], 3) == []
    assert len(pick_terms(terms[:2], 5)) == 2, "术语不够时全部返回，不报错"


def test_引用对得上页时判定为可溯源():
    studio = FakeStudio(
        pages={1: "方差是各数据与均值之差的平方的平均。"},
        terms=[_term("方差")],
        answers=lambda q: (
            _view("方差衡量离散程度，是平方的平均。【u001:p1】")
            if "是什么意思" in q
            else _view("资料未覆盖：这份资料里没有这个内容。")
        ),
        pages_by_question={q: 1 for q in ("什么是方差", "方差的定义是什么", "方差是如何定义的")},
    )
    report = run_selfcheck(studio, [DOC], n_terms=1, n_negative=1, translate=False)

    assert report.metrics.citations_total == 1
    assert report.metrics.citation_verified == 1.0
    assert report.metrics.grounded == 1.0
    assert report.metrics.fabricated == 0
    assert report.metrics.retrieval_stability == 1.0
    assert report.verdict == "可溯源"


def test_引用张冠李戴会被抓出来():
    studio = FakeStudio(
        pages={1: "方差是平方的平均。", 2: "这一页讲的是别的东西。"},
        terms=[_term("方差")],
        answers=lambda q: _view("方差是平方的平均。【u001:p2】"),
        pages_by_question={},
    )
    report = run_selfcheck(studio, [DOC], n_terms=1, n_negative=1, translate=False)

    assert report.metrics.citation_verified == 0.0
    assert report.items[0].bad_citations == ("u001:p2",)
    assert report.verdict == "有风险"


def test_资料外的问题被编造会降档():
    studio = FakeStudio(
        pages={1: "方差是平方的平均。"},
        terms=[_term("方差")],
        answers=lambda q: _view("这个方法的定义是……【u001:p1】"),
        pages_by_question={},
    )
    report = run_selfcheck(studio, [DOC], n_terms=1, n_negative=2, translate=False)

    assert report.metrics.fabricated == 2
    assert report.metrics.refusal == 0.0
    assert report.verdict == "有风险"
    assert any("不存在" in r for r in report.verdict_reasons)


def test_正确拒答计入拒答率():
    def answers(question: str) -> AnswerView:
        if "是什么意思" in question:
            return _view("方差是平方的平均。【u001:p1】")
        return _view("资料未覆盖：这份资料里没有这个内容。", with_evidence=True)

    studio = FakeStudio(
        pages={1: "方差是平方的平均。"},
        terms=[_term("方差")],
        answers=answers,
        pages_by_question={},
    )
    report = run_selfcheck(studio, [DOC], n_terms=1, n_negative=2, translate=False)

    assert report.metrics.refusal == 1.0
    assert report.metrics.fabricated == 0


def test_换问法落到不同页记为不稳定():
    questions = ("什么是方差", "方差的定义是什么", "方差是如何定义的")
    studio = FakeStudio(
        pages={1: "方差是平方的平均。"},
        terms=[_term("方差")],
        answers=lambda q: _view("方差是平方的平均。【u001:p1】"),
        pages_by_question={questions[0]: 1, questions[1]: 1, questions[2]: 9},
    )
    report = run_selfcheck(studio, [DOC], n_terms=1, n_negative=1, translate=False)

    # 前 3 条结果的页码集合两两比：{1} 对 {1} 完全重合，{1} 对 {9} 为 0，均值 0.5
    assert abs(report.metrics.retrieval_stability - 0.5) < 1e-9


def test_没有引用时不给引用可核验率():
    """一条引用都没有时返回 None，而不是 1.0 —— "没东西可查"不等于"全对"。"""
    studio = FakeStudio(
        pages={1: "方差是平方的平均。"},
        terms=[_term("方差")],
        answers=lambda q: _view("方差是平方的平均。"),
        pages_by_question={},
    )
    report = run_selfcheck(studio, [DOC], n_terms=1, n_negative=1, translate=False)

    assert report.metrics.citations_total == 0
    assert report.metrics.citation_verified is None
    assert report.metrics.grounded == 0.0
    assert report.verdict == "有风险"
