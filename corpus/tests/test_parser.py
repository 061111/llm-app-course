"""M2-T2 验收：解析层。

分两层：
  单元层  用假字符串测清洗规则，不依赖任何 PDF，换机器也全绿。
  集成层  用真实 PDF 测 199 页的解析结果，本机没有语料就自动跳过。

集成层为什么值得写：单元测试只能证明"规则按我设想的方式执行"，
证明不了"这 199 页真的被洗干净了"。比如 chap04b p7 是纯图片页，
只有跑真的才知道它会被标成空页。
"""
from __future__ import annotations

import pytest

from corpus.docs import DOCS
from corpus.parser import (
    FOOTER_Y_RATIO,
    MIN_CHARS,
    PageText,
    clean_page_text,
    corpus_available,
    detect_numeric_footer,
    extract_pages,
    is_copyright,
    is_numeric_footer,
    map_pua,
    strip_footer_blocks,
)

# ---------------------------------------------------------------- 单元层


def test_逻辑符号被映射而不是删掉():
    # 这两条是全项目最脆的地方：删掉就变成 "(a) (b) Birds"，规则语义消失
    assert map_pua("\uf0d9") == "∧"
    assert map_pua("\uf0ae") == "⇒"


def test_乘号被映射():
    # 4/7 × 1/7 × 0.0072 = 0.0006，乘号丢了就变成三个数字并排
    assert map_pua("4/7 \uf0b4 1/7") == "4/7 × 1/7"


def test_项目符号被映射():
    assert map_pua("\uf0a7Examples") == "•Examples"


def test_未登记的私有区字符变成空格():
    # U+F0F7 实测出现 2 次，含义不明，任务是"别让乱码进正文"
    assert map_pua("a\uf0f7b") == "a b"


def test_无私有区字符时原样返回():
    assert map_pua("正常文本 ¶ ⊕") == "正常文本 ¶ ⊕"


def test_纯数字行判定():
    assert is_numeric_footer("12") is True
    assert is_numeric_footer("  7  ") is True
    assert is_numeric_footer("12a") is False
    assert is_numeric_footer("Section 12") is False
    assert is_numeric_footer("") is False


def test_版权行判定():
    assert is_copyright("Slides: Introduction to Data Mining, 2nd Edition b") is True
    assert is_copyright("Introduction to Data Mining, 2nd Edition, page 12") is True
    assert is_copyright("正文里提到 Introduction to Data Mining 这本书") is False


def test_版权行总被删掉():
    raw = "标题\n正文内容\nSlides: Introduction to Data Mining, 2nd Edition b"
    assert "Slides" not in clean_page_text(raw)


# ---------------------------------------------------------------- 块级过滤
# 这组用假 blocks 测坐标判断，不需要真 PDF。
# 坐标数值照抄 chap03 p31/p60 的实测值（页高 540，页码 y=517.2，表格数字 y=456.5）。

PAGE_H = 540.0


def test_页码按坐标删_与它在列表里的位置无关():
    # 模拟 chap03 p60：页码在抽取顺序里排在标题之后、正文之前
    blocks = [
        (20.1, 50.0, "Computing Gini Index for a Collection of Nodes"),
        (517.2, 30.0, "60"),  # 页码夹在中间
        (100.0, 50.0, "• When a node p is split into k partitions"),
    ]
    out = strip_footer_blocks(blocks, PAGE_H, drop_page_number=True)
    assert [t for _y, _x, t in out] == [
        "Computing Gini Index for a Collection of Nodes",
        "• When a node p is split into k partitions",
    ]


def test_表格里的数字不会被当成页码():
    # chap03 p31 实测：TID=10 在 y=456.5（底部距 75），页码 31 在 y=517.2
    blocks = [(456.5, 50.0, "10"), (517.2, 30.0, "31")]
    out = strip_footer_blocks(blocks, PAGE_H, drop_page_number=True)
    assert [t for _y, _x, t in out] == ["10"]


def test_关闭页脚清洗时页码保留_版权行仍删():
    blocks = [
        (456.5, 50.0, "10"),
        (517.2, 30.0, "31"),
        (520.0, 200.0, "Slides: Introduction to Data Mining, 2nd Edition b"),
    ]
    out = strip_footer_blocks(blocks, PAGE_H, drop_page_number=False)
    texts = [t for _y, _x, t in out]
    assert "31" in texts
    assert "10" in texts
    assert not any(t.startswith("Slides:") for t in texts)


def test_底部区域内的非数字内容不删():
    # 正文最后一行可能正好落在底部 5% 里，它不是页码就得分明地留着
    blocks = [(520.0, 50.0, "relative to the total number of data points in all nodes")]
    out = strip_footer_blocks(blocks, PAGE_H, drop_page_number=True)
    assert len(out) == 1


def test_页脚阈值边界():
    # 恰好等于 95% 的算页脚（实测页码在 95.8%，留了余量）
    y_at_ratio = PAGE_H * FOOTER_Y_RATIO
    out = strip_footer_blocks([(y_at_ratio, 50.0, "7")], PAGE_H, drop_page_number=True)
    assert out == []
    # 略高于阈值的一律保留
    out2 = strip_footer_blocks([(y_at_ratio - 1, 50.0, "7")], PAGE_H, drop_page_number=True)
    assert len(out2) == 1


def test_首行标题必须保留():
    # 实测每页首行是标题，不是页眉。删了它，T3 就没有切分依据了
    raw = "Direct Method: RIPPER\n规则内容在这里"
    assert clean_page_text(raw).startswith("Direct Method: RIPPER")


def test_空行被去掉():
    raw = "标题\n\n\n正文\n\n"
    assert clean_page_text(raw) == "标题\n正文"


def test_孤立项目符号行并入上一行():
    raw = "正文\n•\n下一句"
    out = clean_page_text(raw)
    assert out == "正文 •\n下一句"


def test_page_text_的派生属性():
    p = PageText(doc_id="chap03", page=7, text="abc", raw_chars=10, is_empty=False)
    assert p.char_count == 3
    assert p.dropped_chars == 7
    assert p.source() == "chap03:p7"


def test_page_text_不可变():
    p = PageText(doc_id="chap03", page=7, text="abc", raw_chars=10, is_empty=False)
    with pytest.raises(Exception):
        p.text = "changed"  # type: ignore[misc]


def test_未知_doc_id_且不给_path_时报错():
    with pytest.raises(ValueError, match="未知 doc_id"):
        extract_pages("chap99")


# ---------------------------------------------------------------- 集成层

needs_corpus = pytest.mark.skipif(
    not corpus_available(), reason="本机没有语料 PDF，跳过集成用例"
)


@needs_corpus
def test_五份讲义一共_199_页():
    total = 0
    for doc_id, meta in DOCS.items():
        total += len(extract_pages(doc_id))
    assert total == 199


@needs_corpus
def test_页码连续且与登记表一致():
    for doc_id, meta in DOCS.items():
        pages = extract_pages(doc_id)
        assert [p.page for p in pages] == list(range(1, meta.pages + 1))


@needs_corpus
def test_解析结果里没有私有区字符残留():
    for doc_id in DOCS:
        for p in extract_pages(doc_id):
            leftovers = [c for c in p.text if 0xE000 <= ord(c) <= 0xF8FF]
            assert not leftovers, f"{p.source()} 还残留 {leftovers}"


@needs_corpus
def test_chap03_的页码页脚被识别并删掉():
    pages = extract_pages("chap03")
    # 抽查 6 页（含 p31/34/38/41 这几页有表格的），页码必须不再单独成行。
    #
    # 注意这里不能断言"末行不是数字"：p31 删掉页码 31 之后，末行还剩一个 "10"，
    # 那是事务表的 TID 值，本来就该保留。第一版断言写成"末行不是纯数字"，
    # 结果 4 条假失败 —— 错的是断言，不是代码。这 4 页恰好证明页脚规则
    # 只删了最后一行、没有把它上面那行一起吞掉。
    for page_no in (15, 31, 34, 38, 41, 60):
        p = pages[page_no - 1]
        assert str(page_no) not in p.text.splitlines(), f"p{page_no} 的页码没被删掉"


@needs_corpus
def test_另四份讲义不误删末行数字():
    # chap05 p3 末行是 "Bread, Milk, Diaper, Coke"，本来就不是数字；
    # 这条测的是"没有页码的文档里，末行的数字若存在也不会被删"
    for doc_id in ("chap04a", "chap04b", "chap04c", "chap05"):
        pages = extract_pages(doc_id)
        for p in pages:
            if p.text and is_numeric_footer(p.text.splitlines()[-1]):
                # 确实存在这种页 → 说明我们没滥用页脚规则
                assert len(p.text.splitlines()) > 1


@needs_corpus
def test_纯图片页被标成空页而不是丢弃():
    pages = extract_pages("chap04b")
    empty = [p.page for p in pages if p.is_empty]
    assert empty == [7], f"chap04b 的空页应只有 p7，实际 {empty}"
    # 关键：它仍在列表里，页码没断
    assert len(pages) == 14


@needs_corpus
def test_每页标题都还在():
    # 抽三页核对首行是不是原书标题
    assert extract_pages("chap04a")[24].text.startswith("Direct Method: RIPPER")
    assert extract_pages("chap04a")[11].text.startswith("Rule Ordering Schemes")
    assert extract_pages("chap05")[2].text.startswith("Definition: Frequent Itemset")


@needs_corpus
def test_规则符号在真实解析结果里变成可读箭头():
    text = extract_pages("chap04a")[2].text  # p3，五条规则
    assert "⇒" in text and "∧" in text


@needs_corpus
def test_空页判定的阈值与最短有效页不冲突():
    # 全部 199 页里，被标空的最长页 与 被保留的最短页，界限要清晰
    all_pages = []
    for doc_id in DOCS:
        all_pages.extend(extract_pages(doc_id))
    empties = [p.char_count for p in all_pages if p.is_empty]
    kept = [p.char_count for p in all_pages if not p.is_empty]
    assert max(empties) < MIN_CHARS <= min(kept)


@needs_corpus
def test_强制关闭页脚清洗会保留数字():
    pages = extract_pages("chap03", force_footer=False)
    numeric_tails = [p.page for p in pages if p.text and is_numeric_footer(p.text.splitlines()[-1])]
    assert len(numeric_tails) > 40  # 实测 53 页


@needs_corpus
def test_页脚检测是按文档自适应的():
    import pymupdf

    flags = {}
    for doc_id, meta in DOCS.items():
        doc = pymupdf.open(meta.path())
        flags[doc_id] = detect_numeric_footer(doc)
        doc.close()
    # 实测：只有 chap03 有页码页脚
    assert flags == {
        "chap03": True,
        "chap04a": False,
        "chap04b": False,
        "chap04c": False,
        "chap05": False,
    }
