"""领域画像的机械校验：这是"不编造"这一条的唯一实现证据。

重点不是"模型输出长什么样"，而是"模型说错话时系统怎么处理"：
丢哪个、留哪个、留的凭什么可信。
"""
from __future__ import annotations

import json

from corpus.parser import PageText

from studio.glossary import mine_terms
from studio.profile import (
    LlmProfile,
    build_profile,
    save_profile,
    load_profile,
    statistical_profile,
    verify,
)

PAGES = [
    PageText(
        doc_id="u001",
        page=1,
        text="系统聚类法第 1 步先计算距离矩阵。距离矩阵记录样品两两之间的距离。距离矩阵的规模随样品数增长。",
        raw_chars=60,
        is_empty=False,
    ),
    PageText(
        doc_id="u001",
        page=2,
        text="判别分析使用马氏距离。马氏距离考虑了协方差阵。主成分分析用来降维。主成分分析保留方差最大的方向。",
        raw_chars=60,
        is_empty=False,
    ),
]
DOC_IDS = ("u001",)


def _terms():
    return mine_terms(PAGES, top_n=20)


def _raw(**kw) -> LlmProfile:
    base = {"domain": "多元统计", "summary": "聚类与判别", "terms": [], "phrasings": []}
    base.update(kw)
    return LlmProfile.model_validate(base)


def test_原文里没有的术语被拒绝并计数():
    raw = _raw(
        terms=[
            {"term": "距离矩阵", "note": ""},
            {"term": "量子纠缠矩阵", "note": ""},
        ]
    )
    profile = verify(raw, pages=PAGES, candidates=_terms(), doc_ids=DOC_IDS)
    assert profile.term_texts == ("距离矩阵",)
    assert profile.rejected == ("量子纠缠矩阵",)


def test_页码由系统重算而不是模型给():
    raw = _raw(terms=[{"term": "主成分分析", "note": ""}])
    profile = verify(raw, pages=PAGES, candidates=_terms(), doc_ids=DOC_IDS)
    assert profile.terms[0].pages == ("u001:p2",)


def test_说明里的数字必须能在原文找到():
    raw = _raw(
        terms=[
            {"term": "距离矩阵", "note": "规模是 n×n"},  # 无数字，保留
            {"term": "主成分分析", "note": "通常保留 85% 方差"},  # 85 原文没有
        ]
    )
    profile = verify(raw, pages=PAGES, candidates=_terms(), doc_ids=DOC_IDS)
    notes = {t.term: t.note for t in profile.terms}
    assert notes["距离矩阵"] == "规模是 n×n"
    assert notes["主成分分析"] == ""
    assert profile.notes_dropped == ("主成分分析",)


def test_说明里的数字若真在原文里则保留():
    # 反例：原文没有"2"，所以说明里的 2 会被判成编造
    raw = _raw(terms=[{"term": "距离矩阵", "note": "原文出现 2 次的方法"}])
    profile = verify(raw, pages=PAGES, candidates=_terms(), doc_ids=DOC_IDS)
    assert profile.terms[0].note == ""
    # 正例：原文第 1 页确实写着"第 1 步"，说明里的 1 有据可查
    raw2 = _raw(terms=[{"term": "距离矩阵", "note": "见第 1 页开头的定义"}])
    profile2 = verify(raw2, pages=PAGES, candidates=_terms(), doc_ids=DOC_IDS)
    assert profile2.terms[0].note == "见第 1 页开头的定义"


def test_引用未通过术语的问题被丢弃():
    raw = _raw(
        terms=[{"term": "距离矩阵", "note": ""}],
        phrasings=[
            {"question": "距离矩阵怎么算？", "terms": ["距离矩阵"]},
            {"question": "量子纠缠矩阵怎么算？", "terms": ["量子纠缠矩阵"]},
            {"question": "没说术语的问题", "terms": []},
        ],
    )
    profile = verify(raw, pages=PAGES, candidates=_terms(), doc_ids=DOC_IDS)
    assert [p.question for p in profile.phrasings] == ["距离矩阵怎么算？"]


def test_重复术语只留一次():
    raw = _raw(terms=[{"term": "距离矩阵"}, {"term": "距离矩阵"}])
    profile = verify(raw, pages=PAGES, candidates=_terms(), doc_ids=DOC_IDS)
    assert profile.term_texts == ("距离矩阵",)


def test_证据片段确实含这个术语():
    raw = _raw(terms=[{"term": "主成分分析", "note": ""}])
    profile = verify(raw, pages=PAGES, candidates=_terms(), doc_ids=DOC_IDS)
    assert "主成分分析" in profile.terms[0].evidence


def test_纯统计画像不编领域名也不编概括():
    terms = _terms()
    profile = statistical_profile(pages=PAGES, terms=terms, doc_ids=DOC_IDS)
    assert profile.source == "statistical"
    assert profile.domain == "未命名领域"
    assert "候选术语" in profile.summary
    assert profile.phrasings == ()
    assert profile.rejected == ()
    for t in profile.terms:
        assert t.pages, "每个术语都要有出处"


def test_模型调用失败时降级而不是报错():
    def boom(_prompt: str) -> str:
        raise RuntimeError("模型挂了")

    profile = build_profile(
        pages=PAGES, terms=_terms(), doc_ids=DOC_IDS, llm=boom, allow_fallback=True
    )
    assert profile.source == "statistical"
    assert profile.terms


def test_不允许降级时把异常抛出来():
    def boom(_prompt: str) -> str:
        raise RuntimeError("模型挂了")

    try:
        build_profile(
            pages=PAGES, terms=_terms(), doc_ids=DOC_IDS, llm=boom, allow_fallback=False
        )
    except RuntimeError as exc:
        assert "模型挂了" in str(exc)
    else:
        raise AssertionError("应该抛异常")


def test_模型输出不是_JSON_时降级():
    profile = build_profile(
        pages=PAGES,
        terms=_terms(),
        doc_ids=DOC_IDS,
        llm=lambda _p: "我觉得这份资料讲的是聚类。",
        allow_fallback=True,
    )
    assert profile.source == "statistical"


def test_正常输出走_llm_路径():
    payload = json.dumps(
        {
            "domain": "多元统计",
            "summary": "聚类分析与判别分析",
            "terms": [{"term": "距离矩阵", "note": ""}],
            "phrasings": [],
        },
        ensure_ascii=False,
    )
    profile = build_profile(
        pages=PAGES, terms=_terms(), doc_ids=DOC_IDS, llm=lambda _p: payload
    )
    assert profile.source == "llm"
    assert profile.domain == "多元统计"
    assert profile.term_texts == ("距离矩阵",)


def test_候选块里带原文片段():
    from studio.profile import build_candidates_block

    block = build_candidates_block(_terms())
    assert "距离矩阵" in block
    assert "次）" in block


def test_画像可以落盘再读回(tmp_path):
    payload = json.dumps(
        {
            "domain": "多元统计",
            "summary": "s",
            "terms": [{"term": "距离矩阵", "note": ""}],
            "phrasings": [{"question": "距离矩阵怎么算", "terms": ["距离矩阵"]}],
        },
        ensure_ascii=False,
    )
    profile = build_profile(
        pages=PAGES, terms=_terms(), doc_ids=DOC_IDS, llm=lambda _p: payload
    )
    path = tmp_path / "u001.json"
    save_profile(profile, path)
    again = load_profile(path)
    assert again == profile


def test_坏画像文件当作没有(tmp_path):
    path = tmp_path / "u001.json"
    path.write_text("{oops", encoding="utf-8")
    assert load_profile(path) is None


def test_prompt_block_只放术语不放说明():
    payload = json.dumps(
        {
            "domain": "多元统计",
            "summary": "s",
            "terms": [{"term": "距离矩阵", "note": "一句很长的说明"}],
        },
        ensure_ascii=False,
    )
    profile = build_profile(
        pages=PAGES, terms=_terms(), doc_ids=DOC_IDS, llm=lambda _p: payload
    )
    block = profile.prompt_block()
    assert "距离矩阵" in block
    assert "一句很长的说明" not in block
