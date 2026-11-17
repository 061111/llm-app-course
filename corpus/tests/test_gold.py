"""M2-T1 验收：黄金集的形状对不对。

这里**不测答案内容是否正确** —— 答案对不对只能靠人去核对原文，
代码测不出来。代码能测的是"这张表作为数据结构是否可用"：
条数够不够、id 有没有重复、页号有没有越界、五种讲义是不是都覆盖到了。

两个断言特别值得注意：
  test_空答案会被_pydantic_拦下  —— 验证 Field(min_length) 真的在挡
  test_doc_id_写错会立刻报错     —— 验证校验时机是"构造后立刻查"，不是跑到 T4 才发现
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from corpus.docs import DOCS, DocMeta
from corpus.gold import GOLD, GoldItem, GoldKind, validate_all


def test_黄金集是_18_条():
    assert len(GOLD) == 18


def test_id_格式统一且不重复():
    ids = [g.id for g in GOLD]
    assert len(set(ids)) == len(ids), "有重复 id"
    # 按顺序编号 G01..G18，方便报告里点名
    assert ids == [f"G{i:02d}" for i in range(1, 19)]


def test_每份讲义都至少出了一道题():
    covered = {g.doc_id for g in GOLD}
    assert covered == set(DOCS), f"没覆盖到的讲义：{set(DOCS) - covered}"


def test_三类题都有():
    kinds = {g.kind for g in GOLD}
    assert kinds == {GoldKind.FACT, GoldKind.CALC, GoldKind.TRAP}


def test_全部通过页号校验():
    validate_all()  # 不抛异常即通过


def test_答案都不是敷衍的一句话():
    # Pydantic 只保证 >=10 字符；这里提高门槛，逼答案写完整
    for g in GOLD:
        assert len(g.answer) >= 30, f"{g.id} 的答案太短：{g.answer!r}"


def test_source_拼出可读出处():
    assert GOLD[0].source() == "chap03:p15"


def test_空答案会被_pydantic_拦下():
    with pytest.raises(ValidationError):
        GoldItem(
            id="G99",
            question="问题够长了吧？",
            answer="短",
            doc_id="chap03",
            page=1,
            kind=GoldKind.FACT,
        )


def test_doc_id_写错会立刻报错():
    item = GoldItem(
        id="G99",
        question="一个足够长的问题",
        answer="一个足够长的答案，长度超过三十个字符，用来通过字段校验。",
        doc_id="chap99",
        page=1,
        kind=GoldKind.FACT,
    )
    with pytest.raises(ValueError, match="未知文档"):
        item.check_known_doc()


def test_页码越界会被抓出来():
    # chap04b 只有 14 页，写到第 30 页就是记错了出处
    item = GoldItem(
        id="G99",
        question="一个足够长的问题",
        answer="一个足够长的答案，长度超过三十个字符，用来通过字段校验。",
        doc_id="chap04b",
        page=30,
        kind=GoldKind.FACT,
    )
    with pytest.raises(ValueError, match="超过"):
        item.check_known_doc()


def test_字段名写错不会被静默接受():
    # extra="forbid"：把 doc_id 写成 docid 应该报错，而不是多出一个废字段
    with pytest.raises(ValidationError):
        GoldItem(
            id="G99",
            question="一个足够长的问题",
            answer="一个足够长的答案，长度超过三十个字符，用来通过字段校验。",
            docid="chap03",  # 故意的错拼
            page=1,
            kind=GoldKind.FACT,
        )


def test_全部已核对():
    # 2026-11-17 核对完毕。这条断言的作用是防止以后新增条目忘了核对就混进来：
    # 新加一条如果没写 verified_by_user=True，这里立刻红。
    not_verified = [g.id for g in GOLD if not g.verified_by_user]
    assert not not_verified, f"这些条目还没核对过：{not_verified}"


def test_doc_meta_页数与实测一致():
    # 2026-11-17 用 pymupdf 数出来的页数，合计 199 页
    assert {k: v.pages for k, v in DOCS.items()} == {
        "chap03": 82,
        "chap04a": 37,
        "chap04b": 14,
        "chap04c": 28,
        "chap05": 38,
    }
    assert sum(v.pages for v in DOCS.values()) == 199


def test_doc_meta_不可变():
    with pytest.raises(ValidationError):
        DOCS["chap03"].pages = 999  # type: ignore[misc]


def test_doc_meta_path_指向语料目录下的真实文件名():
    meta = DOCS["chap05"]
    assert meta.path().name == "chap05_Association_Analysis.pdf"


def test_doc_meta_页数不能为负():
    with pytest.raises(ValidationError):
        DocMeta(doc_id="x", filename="x.pdf", pages=0, topic="t")
