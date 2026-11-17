"""M2-T4 的测试：落盘往返是否保真。

这里最值得测的不是"能不能写文件"，而是**写出去再读回来是否一模一样**。
Pydantic 的 tuple/BaseModel 嵌套在 JSON 里会退化成 list/dict，
如果还原不回来，下游拿到的 spans 就是 dict 而不是 PageSpan，
直到 M4 算页码时才炸 —— 所以现在测。
"""
from __future__ import annotations

import json

import pytest

from corpus.chunker import Chunk, PageSpan, build_all, corpus_available
from corpus.store import (
    ChunkFileMeta,
    default_paths,
    format_stats,
    load_chunks,
    read_meta,
    save_chunks,
    stats,
)

needs_corpus = pytest.mark.skipif(
    not corpus_available(), reason="本机没有 DataMining PDF 语料，跳过集成用例"
)


def fake_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="d#000",
            doc_id="d",
            title="Rule Ordering Schemes",
            page_start=12,
            page_end=13,
            text="正文一\n\n正文二",
            spans=(
                PageSpan(page=12, start=0, end=3),
                PageSpan(page=13, start=5, end=8),
            ),
            is_empty=False,
        ),
        Chunk(
            chunk_id="d#001",
            doc_id="d",
            title="",
            page_start=14,
            page_end=14,
            text="只有一页",
            spans=(PageSpan(page=14, start=0, end=4),),
            is_empty=False,
        ),
    ]


def test_往返完全一致(tmp_path):
    src = fake_chunks()
    chunks_path, meta_path = default_paths(tmp_path)
    save_chunks(src, chunks_path, meta_path)
    back = load_chunks(chunks_path, meta_path)
    assert back == src, "读回来的块和写进去的不一样"


def test_spans_读回来仍是_PageSpan_而不是_dict(tmp_path):
    chunks_path, meta_path = default_paths(tmp_path)
    save_chunks(fake_chunks(), chunks_path, meta_path)
    back = load_chunks(chunks_path, meta_path)
    assert isinstance(back[0].spans, tuple)
    assert isinstance(back[0].spans[0], PageSpan)
    # 派生方法在往返后仍然可用 —— 这才是真正要保的东西
    assert back[0].page_of(0) == 12
    assert back[0].page_of(6) == 13


def test_jsonl_一行一块且是人类可读的(tmp_path):
    chunks_path, meta_path = default_paths(tmp_path)
    save_chunks(fake_chunks(), chunks_path, meta_path)
    lines = chunks_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["chunk_id"] == "d#000"
    assert first["spans"][0] == {"page": 12, "start": 0, "end": 3}


def test_meta_记录了生成参数(tmp_path):
    chunks_path, meta_path = default_paths(tmp_path)
    save_chunks(fake_chunks(), chunks_path, meta_path, mode="page", max_chars=900, overlap=100)
    meta = read_meta(meta_path)
    assert isinstance(meta, ChunkFileMeta)
    assert (meta.mode, meta.max_chars, meta.overlap) == ("page", 900, 100)
    assert meta.chunk_count == 2
    assert meta.total_chars == sum(len(c.text) for c in fake_chunks())


def test_块数对不上时报错而不是沉默(tmp_path):
    chunks_path, meta_path = default_paths(tmp_path)
    save_chunks(fake_chunks(), chunks_path, meta_path)
    # 手改文件：删掉一行，模拟"写一半中断"或"人工编辑"
    lines = chunks_path.read_text(encoding="utf-8").splitlines()
    chunks_path.write_text(lines[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="块数不一致"):
        load_chunks(chunks_path, meta_path)


def test_文件不存在时给出可执行的提示(tmp_path):
    chunks_path, meta_path = default_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="corpus.store save"):
        load_chunks(chunks_path, meta_path)


def test_单行坏掉时报出第几行(tmp_path):
    chunks_path, meta_path = default_paths(tmp_path)
    save_chunks(fake_chunks(), chunks_path, meta_path)
    lines = chunks_path.read_text(encoding="utf-8").splitlines()
    lines[1] = '{"chunk_id": "坏的"}'  # 缺字段
    chunks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        load_chunks(chunks_path, meta_path)


def test_stats_统计正确():
    s = stats(fake_chunks())
    assert s.chunk_count == 2
    assert s.min_chars == 4
    assert s.max_chars == 8
    assert s.by_doc == {"d": 2}
    assert s.empty == 0
    assert "块数      2" in format_stats(s)


@needs_corpus
def test_真语料落盘往返一致(tmp_path):
    chunks = build_all()
    chunks_path, meta_path = default_paths(tmp_path)
    save_chunks(chunks, chunks_path, meta_path)
    back = load_chunks(chunks_path, meta_path)
    assert back == chunks
    assert len(back) == 181
