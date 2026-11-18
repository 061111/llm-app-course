"""M6 · 领域无关自检：不给标准答案，也能判断"这份资料能不能放心问"。

【为什么需要这个模块】
  `corpus/gold.py` 那 18 条能测整个链路，但它们绑死在 5 份英文课程讲义上 ——
  谁上传一份法律合同、一份产品手册、一份病历，那 18 条一条都用不上。
  而没有标准答案，就没法判断"答得对不对"：
    · 人工出题：每份新文档都要人读一遍、写题、核页，不可行；
    · 让模型自己出题再自己判卷：等于自己给自己打分，测出来的是"模型像不像模型"。

  所以这里换一个思路：**不判语义，只判能被机械证实的事**。
  一条答案是否可信，最硬的三个条件是：
    1. 它标的页真的存在（引用越界 —— 已在 qa/answer.py 里做）；
    2. 它标的页上真的写着它说的那个东西（本模块的"引用可核验"）；
    3. 资料里根本没有的东西，它敢不敢说"没有"（本模块的"非资料问题拒答"）。
  这三条都不需要人给标准答案，任何语言的任何文档都能跑。

【四项指标，各自抓什么错】
  citation_verified  引用可核验率。拿术语表里的词问"是什么意思"，答案标的页里
                     是否真的出现这个词。抓"引用真实但张冠李戴"——引用页合法
                     （在检索结果里），但那一页压根没这个概念。
                     局限：词出现在页上 ≠ 那页支持了那句结论。后者机器判不了，
                     这里只保证"引用不是随便挑的一页"。

  grounded           有据回答率。被问到的术语本来就从原文挖出来的，答完却一个
                     引用都没有，说明模型在凭记忆说话。抓"无出处作答"。

  refusal            非资料问题拒答率。取一个**保证不在正文里**的随机串当术语
                     去问（见 make_nonce），模型应当说"资料未覆盖"。抓"编造"。
                     这是四项里唯一能百分百机械判定对错的一项：
                     随机串不可能在文档里，任何实质回答都是编的。

  retrieval_stability 检索稳定性。同一个术语用三种问法检索，top-1 落在同一页的
                     比例。抓"换个问法就答不出来"——这是检索层不健壮的直接证据。
                     它只走检索、不调模型，零成本。

【阈值的来历，别当成定律】
  VERDICT 里的 0.9 / 0.8 / 0.6 是在 30 页中文简答资料（2026-11-18 实测）上
  定的经验值，只有一份样本，没有跨领域校准。它们的作用是"给用户一个红绿灯"，
  不是科学判据。真要用它做验收，至少先跑十份不同领域的资料再改这些数。

【为什么默认只问 5 + 3 个问题】
  每问一次是一次模型调用（约 ¥0.003）。5 + 3 = 8 次 ≈ ¥0.03、约 20 秒 ——
  这是"用户愿意点一下"的规模。默认值调大不会更准，只会更贵：
  这些指标测的是"有没有系统性问题"，8 个样本已经能暴露引用对不上、敢编造。
"""
from __future__ import annotations

import random
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # 只在类型检查时导入：service 会反过来 import 本模块，避免成环
    from studio.glossary import Term
    from studio.service import AnswerView, Studio

# 随机串用的字母表：去掉 I/O/0/1，避免模型把它读成别的字符而"歪打正着"。
NONCE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

# 判定为"可疑"的三种结局，报告里要单独点名。
OUTCOME_OK = "ok"
OUTCOME_NO_CITE = "no_citation"
OUTCOME_REFUSED = "refused"
OUTCOME_FABRICATED = "fabricated"
OUTCOME_UNCITED = "answered_without_citation"
OUTCOME_MISS = "no_hit"
OUTCOME_PARTIAL = "partial"

TERM_QUESTION = "「{term}」在这份资料里是什么意思？"
NONCE_QUESTIONS = (
    "「{nonce}」在这份资料里是怎么定义的？",
    "这份资料里关于「{nonce}」的结论是什么？",
)
ZH_PHRASINGS = ("什么是{term}", "{term}的定义是什么", "{term}是如何定义的")
EN_PHRASINGS = ("what is {term}", "definition of {term}", "{term} meaning")

# 经验阈值：一份中文样本上定的，见模块说明。
THRESHOLD_CITATION = 0.9
THRESHOLD_GROUNDED = 0.8
THRESHOLD_REFUSAL = 0.9
THRESHOLD_STABILITY = 0.6

_SQUASH_RE = re.compile(r"\s+")


def squash(text: str) -> str:
    """压掉空白并转小写。

    为什么要压空白：PDF 抽出来的英文常带断行与多余空格（"Gini  In dex"），
    直接把术语拿去 `in` 判断会漏检。中文不受影响，但统一处理更省心。
    """
    return _SQUASH_RE.sub("", text).lower()


def term_on_pages(pages: dict[int, str], term: str, start: int, end: int) -> bool:
    """术语是否出现在 [start, end] 的**至少一页**上。

    为什么是"至少一页"而不是"每一页"：区间引用（p12-13）常见于跨页展开的推导，
    术语往往只在其中一页出现。要求每页都有会把合法引用判成错，误报比漏报更伤
    —— 用户看到一堆红叉就不再信任这个自检了。
    """
    needle = squash(term)
    if not needle:
        return False
    return any(needle in squash(pages.get(p, "")) for p in range(start, end + 1))


def make_nonce(rng: random.Random, corpus: str) -> str:
    """生成一个**保证不在正文里**的 6 位随机串。

    这是"非资料问题"能机械判定的根基：只要串不在正文里，模型对它的任何实质
    回答都是编的，不需要人来判断"这个问题超纲了吗"。
    循环上限 200 次是防御性的：正常文档一次就中，真撞上了说明文档里有大量随机串，
    那种情况下多试几次也无所谓。
    """
    flat = squash(corpus)
    for _ in range(200):
        token = "".join(rng.choice(NONCE_ALPHABET) for _ in range(6))
        if token.lower() not in flat:
            return token
    raise RuntimeError("生成不出不在正文里的随机串，请检查输入文本")


def pick_terms(terms: list["Term"], n: int) -> list["Term"]:
    """等距抽样，而不是取前 n 个。

    取前 n 个会全落在文档最前面（术语分往往按频次排，前三页的高频词占满），
    抽出来的"样本"其实是同一片区域 —— 后面章节的引用错误就测不到。
    等距抽样能覆盖整份文档，代价可接受。
    """
    if n <= 0 or not terms:
        return []
    if len(terms) <= n:
        return list(terms)
    if n == 1:
        return [terms[len(terms) // 2]]
    # 首尾都取：只按 n 等分会在最后一段留空，文档末尾的引用错误就永远测不到。
    step = (len(terms) - 1) / (n - 1)
    return [terms[round(i * step)] for i in range(n)]


class SelfCheckItem(BaseModel):
    """一条自检记录。detail 是给人看的证据，不是给程序解析的。"""

    model_config = ConfigDict(frozen=True)

    check: str = Field(description="citation / negative / stability")
    subject: str
    outcome: str
    detail: str = ""
    citations: tuple[str, ...] = Field(default_factory=tuple)
    bad_citations: tuple[str, ...] = Field(default_factory=tuple)


class SelfCheckMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    citations_total: int = 0
    citations_ok: int = 0
    citation_verified: float | None = Field(
        default=None, description="None 表示一条引用都没出现，无从统计"
    )
    grounded: float = 0.0
    refusal: float = 0.0
    fabricated: int = 0
    retrieval_stability: float = 0.0


class SelfCheckReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_ids: tuple[str, ...]
    terms: tuple[str, ...]
    items: tuple[SelfCheckItem, ...]
    metrics: SelfCheckMetrics
    verdict: str
    verdict_reasons: tuple[str, ...] = Field(default_factory=tuple)
    model: str = Field(default="", description="跑这轮自检用的模型")
    cost_cny: float = 0.0
    seconds: float = 0.0
    generated_at: str = ""

    def to_payload(self) -> dict:
        """给 HTTP 用的扁平结构。前端要直接画表，不想到处挖 items。"""
        return {
            "doc_ids": list(self.doc_ids),
            "terms": list(self.terms),
            "metrics": self.metrics.model_dump(),
            "verdict": self.verdict,
            "verdict_reasons": list(self.verdict_reasons),
            "model": self.model,
            "items": [i.model_dump() for i in self.items],
            "cost_cny": self.cost_cny,
            "seconds": self.seconds,
            "generated_at": self.generated_at,
        }


def _verdict(m: SelfCheckMetrics) -> tuple[str, tuple[str, ...]]:
    """三档红绿灯。每一条降档理由都要说人话，不堆指标名。"""
    reasons: list[str] = []

    if m.fabricated:
        reasons.append(f"{m.fabricated} 个资料里不存在的问题被编了答案")
    if m.citation_verified is None:
        reasons.append("所有回答都没有标注出处，无法溯源")
    elif m.citation_verified < THRESHOLD_CITATION:
        bad = m.citations_total - m.citations_ok
        reasons.append(f"{m.citations_total} 条引用里有 {bad} 条在所指页上找不到对应内容")
    if m.grounded < THRESHOLD_GROUNDED:
        reasons.append(f"只有 {m.grounded:.0%} 的回答带了出处")
    if m.refusal < THRESHOLD_REFUSAL:
        reasons.append(f"资料外的问题只有 {m.refusal:.0%} 被拒答，其余在硬答")
    if m.retrieval_stability < THRESHOLD_STABILITY:
        reasons.append(f"换个问法后只有 {m.retrieval_stability:.0%} 落在同一页，检索不稳")

    # 一条引用都没有也算"有风险"：没有出处的答案无法核对，不能按"没问题"放过去。
    if (
        m.fabricated
        or m.citation_verified is None
        or m.citation_verified < THRESHOLD_CITATION
    ):
        return "有风险", tuple(reasons)
    if reasons:
        return "基本可用", tuple(reasons)
    return "可溯源", ("引用对得上页，资料外的问题会拒答，换问法检索结果稳定",)


def run_selfcheck(
    studio: "Studio",
    doc_ids: list[str],
    *,
    n_terms: int = 5,
    n_negative: int = 3,
    top_k: int = 5,
    translate: bool | None = None,
    seed: int | None = None,
    model: str | None = None,
) -> SelfCheckReport:
    """对一组文档跑四项检查。全程不需要标准答案。

    translate 交给调用方定（和 Studio.ask 的规则一致：英文资料才翻译），
    这里不重复判断语种 —— 语种决策只该有一处。
    """
    started = time.perf_counter()
    rng = random.Random(seed)

    pages: dict[str, dict[int, str]] = {}
    corpus_parts: list[str] = []
    for doc_id in doc_ids:
        pages[doc_id] = {p.page: p.text for p in studio.pages(doc_id)}
        corpus_parts.extend(pages[doc_id].values())
    corpus = "\n".join(corpus_parts)

    terms = pick_terms(studio.terms(doc_ids[0], top_n=40), n_terms)
    searcher = studio.searcher(doc_ids)
    use_translate = bool(translate)
    items: list[SelfCheckItem] = []
    cost = 0.0

    # ---- 1) 引用可核验 + 有据回答：拿术语问"是什么意思"
    grounded_hits = 0
    cite_total = 0
    cite_ok = 0
    for term in terms:
        view: "AnswerView" = studio.ask(
            TERM_QUESTION.format(term=term.text),
            doc_ids=list(doc_ids),
            top_k=top_k,
            model=model,
        )
        cost += view.answer.usage.cost_cny
        cites = view.answer.citations
        bad: list[str] = []
        for c in cites:
            cite_total += 1
            page_texts = pages.get(c.doc_id, {})
            if term_on_pages(page_texts, term.text, c.page_start, c.page_end):
                cite_ok += 1
            else:
                bad.append(c.source())
        if cites and not view.denied:
            grounded_hits += 1

        if view.denied and not cites:
            outcome = OUTCOME_REFUSED
            detail = "模型说资料未覆盖。可能是术语抽取的噪音词（如“结构”“结果”），不是错误。"
        elif not cites:
            outcome = OUTCOME_NO_CITE
            detail = "答了但没有标出处，无法核对。"
        elif bad:
            outcome = OUTCOME_NO_CITE
            detail = f"标了 {len(cites)} 处出处，其中 {len(bad)} 处所指页上没有这个词。"
        else:
            outcome = OUTCOME_OK
            detail = f"{len(cites)} 处出处均能在所指页上找到该词。"
        items.append(
            SelfCheckItem(
                check="citation",
                subject=term.text,
                outcome=outcome,
                detail=detail,
                citations=tuple(c.source() for c in cites),
                bad_citations=tuple(bad),
            )
        )

    # ---- 2) 非资料问题：随机串必然不在正文里，答了就是编的
    nonce = make_nonce(rng, corpus)
    refused = 0
    fabricated = 0
    for i in range(n_negative):
        question = NONCE_QUESTIONS[i % len(NONCE_QUESTIONS)].format(nonce=nonce)
        view = studio.ask(question, doc_ids=list(doc_ids), top_k=top_k, model=model)
        cost += view.answer.usage.cost_cny
        if view.answer.citations:
            fabricated += 1
            outcome = OUTCOME_FABRICATED
            detail = f"给不存在的东西标了出处：{', '.join(c.source() for c in view.answer.citations)}"
        elif view.denied:
            refused += 1
            outcome = OUTCOME_REFUSED
            detail = "正确拒答。"
        else:
            outcome = OUTCOME_UNCITED
            detail = "没有标出处，也没有明确说资料未覆盖 —— 无法确认是在拒答还是凭常识作答。"
        items.append(
            SelfCheckItem(check="negative", subject=nonce, outcome=outcome, detail=detail)
        )

    # ---- 3) 检索稳定性：三种问法取回的证据重合度（零模型调用）
    #
    # 第一版比的是"首条命中落在哪一页"，实测发现测错了东西：术语本来散落在多页，
    # 问法一变，命中的是"另一处"，页码当然不同（同一术语出现 p2 / p18 / p20）。
    # 那不是检索不稳，是同一个词的不同出处。改成比前 3 条结果的**页码集合重合度**
    # （Jaccard）：只要三种问法取回的是同一片证据，就算稳定；跑到完全不同的页，才算不稳。
    stability_scores: list[float] = []
    for term in terms:
        patterns = EN_PHRASINGS if term.text.isascii() else ZH_PHRASINGS
        evidence_pages: list[set[int]] = []
        for pattern in patterns:
            hits = searcher.search(
                pattern.format(term=term.text), top_k=top_k, translate=use_translate
            )
            evidence_pages.append({h.chunk.page_start for h in hits[:3]})
        base = evidence_pages[0]
        if not any(evidence_pages):
            items.append(
                SelfCheckItem(
                    check="stability",
                    subject=term.text,
                    outcome=OUTCOME_MISS,
                    detail="三种问法都检索不到内容。",
                )
            )
            stability_scores.append(0.0)
            continue
        overlaps = []
        for other in evidence_pages[1:]:
            union = base | other
            overlaps.append(len(base & other) / len(union) if union else 0.0)
        score = sum(overlaps) / len(overlaps)
        stability_scores.append(score)
        shown = "、".join(str(sorted(s)) for s in evidence_pages)
        items.append(
            SelfCheckItem(
                check="stability",
                subject=term.text,
                outcome=OUTCOME_OK if score >= 0.999 else (OUTCOME_PARTIAL if score else OUTCOME_MISS),
                detail=f"三种问法取回的前 3 条结果分别落在第 {shown} 页，重合度 {score:.0%}。",
            )
        )

    metrics = SelfCheckMetrics(
        citations_total=cite_total,
        citations_ok=cite_ok,
        citation_verified=(cite_ok / cite_total) if cite_total else None,
        grounded=(grounded_hits / len(terms)) if terms else 0.0,
        refusal=(refused / n_negative) if n_negative else 0.0,
        fabricated=fabricated,
        retrieval_stability=(sum(stability_scores) / len(stability_scores)) if stability_scores else 0.0,
    )
    verdict, reasons = _verdict(metrics)
    return SelfCheckReport(
        doc_ids=tuple(doc_ids),
        terms=tuple(t.text for t in terms),
        items=tuple(items),
        metrics=metrics,
        verdict=verdict,
        verdict_reasons=reasons,
        model=getattr(studio.settings_for(model), "model", "") or "",
        cost_cny=round(cost, 6),
        seconds=round(time.perf_counter() - started, 2),
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )
