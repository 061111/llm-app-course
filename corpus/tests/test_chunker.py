"""M2-T3 的单元测试。

分两层：
  纯函数层（不需要 PDF）：标题清洗、分组、拼接偏移、切分、合并。换机器也能跑。
  集成层（需要 PDF）：真解析 199 页、真书签、真切分。没有语料就 skip。
"""
from __future__ import annotations

import pytest

from corpus.chunker import (
    MAX_CHUNK_CHARS,
    Chunk,
    PageSpan,
    build_all,
    build_chunks,
    corpus_available,
    group_pages,
    join_pages,
    merge_tiny,
    normalize_title,
    read_titles,
    slice_spans,
    split_long,
    title_stem,
)
from corpus.parser import PageText

needs_corpus = pytest.mark.skipif(
    not corpus_available(), reason="本机没有 DataMining PDF 语料，跳过集成用例"
)


def pt(page: int, text: str, doc: str = "d") -> PageText:
    """造一个假的 PageText。is_empty 的阈值取 50，和 parser.MIN_CHARS 一致。"""
    return PageText(
        doc_id=doc,
        page=page,
        text=text,
        raw_chars=len(text),
        is_empty=len(text) < 50,
    )


def body(n: int, seed: str = "x") -> str:
    """造 n 个字符的正文，带换行，方便测切分。"""
    line = f"{seed}" * 40
    return "\n".join([line] * (n // len(line) + 1))[:n]


# ---------------------------------------------------------------- 标题清洗


def test_normalize_去掉_slide_前缀():
    assert normalize_title("Slide 7: Rule Ordering Schemes") == "Rule Ordering Schemes"


def test_normalize_只有页码没有标题_洗成空串():
    # chap03 p38/p39/p40/p42 的书签就是这样，实测
    assert normalize_title("Slide 38") == ""


def test_normalize_去掉尾部省略号和项目符号():
    assert normalize_title("Slide 19: Example of Sequential Covering…") == (
        "Example of Sequential Covering"
    )
    assert normalize_title("Slide 19: Measures of Node Impurity •") == (
        "Measures of Node Impurity"
    )


def test_stem_去掉冒号后的分级后缀():
    assert title_stem("Model Evaluation Metric: Accuracy") == "Model Evaluation Metric"
    assert title_stem("Model Evaluation Metric: Error rate") == "Model Evaluation Metric"


def test_stem_单字前缀不合并():
    # 冒号前只有 1 个词，说明冒号不是分级符，别把不相干的页绑一起
    assert title_stem("TID: something") == "TID: something"


def test_stem_无冒号时原样返回():
    t = "Computing Gini Index of a Single Node"
    assert title_stem(t) == t


# ---------------------------------------------------------------- 分组


TITLES = {
    1: "A",
    2: "A",
    3: "B",
    4: "B",
    5: "B",
    6: "C",
}


def test_page_模式每页一组():
    groups = group_pages(TITLES, mode="page")
    assert len(groups) == 6
    assert all(len(pgs) == 1 for _t, pgs in groups)


def test_title_模式合并连续同标题():
    groups = group_pages(TITLES, mode="title")
    assert [(t, pgs) for t, pgs in groups] == [("A", [1, 2]), ("B", [3, 4, 5]), ("C", [6])]


def test_title_模式不合并不连续的相同标题():
    # 同标题但中间隔了别的标题，不能合并 —— 否则页序会乱
    titles = {1: "A", 2: "B", 3: "A"}
    groups = group_pages(titles, mode="title")
    assert len(groups) == 3


def test_stem_模式把冒号后缀的连续页合并():
    titles = {18: "Model Evaluation Metric: Accuracy", 19: "Model Evaluation Metric: Error rate"}
    groups = group_pages(titles, mode="stem")
    assert len(groups) == 1
    assert groups[0][1] == [18, 19]
    # 显示用标题取 stem，比带后缀的更适合当章节名
    assert groups[0][0] == "Model Evaluation Metric"


def test_stem_模式把连续无标题页并成一组():
    # 连续两页都没标题 = 同一段的延续（chap03 p38/p39 就是这种情况，
    # 只是 read_titles 会先把标题继承下来）。合并比切成两块合理。
    titles = {1: "", 2: "", 3: "Real"}
    groups = group_pages(titles, mode="stem")
    assert len(groups) == 2
    assert groups[0][1] == [1, 2]


# ---------------------------------------------------------------- 拼接与偏移


def test_join_pages_记录每页偏移():
    text, spans = join_pages([pt(1, "aaa"), pt(2, "bb")])
    assert text == "aaa\n\nbb"
    assert spans == (PageSpan(page=1, start=0, end=3), PageSpan(page=2, start=5, end=7))
    # 偏移能反查：第 5 个字符起是第 2 页
    assert text[spans[0].end : spans[1].start] == "\n\n"


def test_join_pages_跳过空页但不丢页号():
    text, spans = join_pages([pt(1, "aaa"), pt(2, ""), pt(3, "ccc")])
    assert text == "aaa\n\nccc"
    assert [s.page for s in spans] == [1, 3]


def test_slice_spans_裁到区间内():
    spans = (PageSpan(page=1, start=0, end=10), PageSpan(page=2, start=12, end=22))
    cut = slice_spans(spans, 5, 15)
    assert cut == (PageSpan(page=1, start=5, end=10), PageSpan(page=2, start=12, end=15))


def test_slice_spans_丢掉完全在区间外的页():
    spans = (PageSpan(page=1, start=0, end=10), PageSpan(page=2, start=12, end=22))
    assert slice_spans(spans, 10, 12) == ()


# ---------------------------------------------------------------- 切分


def test_短文本不切():
    text = "短"
    pieces = split_long(
        text, (PageSpan(page=1, start=0, end=1),), max_chars=100, overlap=10
    )
    assert pieces == [(text, (PageSpan(page=1, start=0, end=1),))]


def test_每块都不超过上限():
    text = body(3000)
    pieces = split_long(text, (PageSpan(page=1, start=0, end=len(text)),), max_chars=600, overlap=100)
    assert len(pieces) > 1
    assert all(len(p[0]) <= 600 for p in pieces)


def test_相邻块之间有重叠():
    text = body(2000)
    pieces = split_long(text, (PageSpan(page=1, start=0, end=len(text)),), max_chars=600, overlap=100)
    for (a, _), (b, _) in zip(pieces, pieces[1:]):
        # b 的开头应当能在 a 的结尾找到（重叠至少 1 个字符）
        assert any(a.endswith(b[:k]) for k in range(1, 101)), "相邻块没有共享内容"


def test_切分不丢字():
    # 用偏移重建覆盖区间：并集必须覆盖整段文本
    text = body(2500)
    spans = (PageSpan(page=1, start=0, end=len(text)),)
    pieces = split_long(text, spans, max_chars=600, overlap=100)
    covered = bytearray(len(text))
    for _body, pspans in pieces:
        for s in pspans:
            for i in range(s.start, s.end):
                covered[i] = 1
    assert all(covered), f"有 {covered.count(0)} 个字符没被任何块覆盖"


def test_切分块内的_spans_与文本长度一致():
    text = body(2500)
    spans = (PageSpan(page=1, start=0, end=len(text)),)
    for b, pspans in split_long(text, spans, max_chars=600, overlap=100):
        assert sum(s.length for s in pspans) == len(b)


def test_参数不合理时直接报错():
    with pytest.raises(ValueError):
        split_long("abc", (), max_chars=100, overlap=100)


def test_超长单行也能切_不依赖换行():
    text = "x" * 1500  # 一个换行都没有
    pieces = split_long(text, (PageSpan(page=1, start=0, end=1500),), max_chars=600, overlap=50)
    assert len(pieces) >= 3
    assert all(len(p[0]) <= 600 for p in pieces)


# ---------------------------------------------------------------- 小组合并


def test_过短的页被后吞():
    groups = [("A", [1]), ("B", [2])]
    merged = merge_tiny(groups, [30, 400], min_chars=200)
    assert merged == [("A", [1, 2])]


def test_正常的页不被合并():
    groups = [("A", [1]), ("B", [2])]
    merged = merge_tiny(groups, [400, 400], min_chars=200)
    assert len(merged) == 2


def test_连续多个短页被一起吞下():
    groups = [("A", [1]), ("B", [2]), ("C", [3])]
    merged = merge_tiny(groups, [0, 50, 400], min_chars=200)
    assert merged == [("A", [1, 2, 3])]


# ---------------------------------------------------------------- 组装


def test_page_模式一页一块():
    pages = [pt(i, body(300, str(i))) for i in range(1, 4)]
    titles = {1: "A", 2: "B", 3: "C"}
    chunks = build_chunks("d", mode="page", pages=pages, titles=titles)
    assert len(chunks) == 3
    assert [c.chunk_id for c in chunks] == ["d#000", "d#001", "d#002"]
    assert all(c.page_start == c.page_end for c in chunks)


def test_没有页被丢掉():
    pages = [pt(i, body(300, str(i))) for i in range(1, 6)]
    titles = {i: f"T{i}" for i in range(1, 6)}
    chunks = build_chunks("d", mode="title", pages=pages, titles=titles)
    covered = set()
    for c in chunks:
        covered.update(range(c.page_start, c.page_end + 1))
    assert covered == set(range(1, 6))


def test_空页不单独成块():
    # chap03 p40/p42 这种整页无文字的页，单独成块就是废块
    pages = [pt(1, body(300, "a")), pt(2, ""), pt(3, body(300, "c"))]
    titles = {1: "A", 2: "A", 3: "C"}
    chunks = build_chunks("d", mode="page", pages=pages, titles=titles)
    assert all(c.text.strip() for c in chunks)
    assert len(chunks) == 2


def test_合并后标题取正文最多的那页():
    # 封面（30 字）被吞进正文页（400 字），标题不该是封面那个
    pages = [pt(1, body(30, "cover")), pt(2, body(400, "real"))]
    titles = {1: "CS422 : Data Mining", 2: "Association Rule Mining"}
    chunks = build_chunks("d", mode="page", pages=pages, titles=titles)
    assert len(chunks) == 1
    assert chunks[0].title == "Association Rule Mining"


def test_page_of_能定位到页():
    pages = [pt(1, "a" * 100), pt(2, "b" * 100)]
    chunks = build_chunks("d", mode="title", pages=pages, titles={1: "T", 2: "T"})
    assert len(chunks) == 1
    c = chunks[0]
    assert c.page_of(0) == 1
    assert c.page_of(99) == 1
    assert c.page_of(100) is None  # 页间分隔符不属于任何一页
    assert c.page_of(102) == 2


def test_page_text_只取那一页的字():
    pages = [pt(1, "a" * 100), pt(2, "b" * 100)]
    chunks = build_chunks("d", mode="title", pages=pages, titles={1: "T", 2: "T"})
    c = chunks[0]
    assert c.page_text(1) == "a" * 100
    assert c.page_text(2) == "b" * 100


def test_header_带出处和标题():
    c = Chunk(
        chunk_id="d#000",
        doc_id="d",
        title="Rule Ordering Schemes",
        page_start=12,
        page_end=13,
        text="正文",
        spans=(PageSpan(page=12, start=0, end=2),),
        is_empty=False,
    )
    assert c.header() == "【d:p12-13 · Rule Ordering Schemes】"
    assert c.source() == "d:p12-13"


# ---------------------------------------------------------------- 集成


@needs_corpus
def test_书签缺标题的页继承上一页():
    # chap03 p38/p39 的书签是 "Slide 38" / "Slide 39"，实测无标题。
    # 注意原文用的是弯引号 ’（U+2019），不是 ASCII 的 '
    titles = read_titles("chap03")
    assert titles[36] == "Hunt’s Algorithm: loan borrower classification problem"
    assert titles[38] == titles[36]
    assert titles[38] == titles[39]


@needs_corpus
def test_真语料切分后不丢页且不超长():
    chunks = build_all()
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    assert set(by_doc) == {"chap03", "chap04a", "chap04b", "chap04c", "chap05"}

    from corpus.docs import DOCS

    for doc_id, meta in DOCS.items():
        covered = set()
        for c in by_doc[doc_id]:
            covered.update(range(c.page_start, c.page_end + 1))
            assert c.char_count <= MAX_CHUNK_CHARS, f"{c.chunk_id} 超长 {c.char_count}"
        assert covered == set(range(1, meta.pages + 1)), f"{doc_id} 有页没进任何块"


@needs_corpus
def test_真语料没有空块():
    chunks = build_all()
    empty = [c.chunk_id for c in chunks if c.is_empty]
    assert empty == [], f"出现了无法检索的空块：{empty}"
