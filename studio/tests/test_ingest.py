"""解析层：安检顺序、文本分页、语种判定。

PDF 的真实解析在 corpus/ 已有 90 条测试，这里只测**上传层新增的那部分**：
后缀识别、魔数、空文件、文本文件的虚拟分页。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio.ingest import classify_lang, detect_kind, ingest, paginate_text
from studio.errors import CorruptFile, EmptyFile, FileTooLarge, UnsupportedType


def test_只认三种后缀():
    assert detect_kind("a.PDF") == (".pdf", "pdf")
    assert detect_kind("b.txt") == (".txt", "text")
    assert detect_kind("c.md") == (".md", "text")


def test_docx_被拒绝且给了下一步建议():
    with pytest.raises(UnsupportedType) as exc:
        detect_kind("论文.docx")
    assert "PDF" in exc.value.hint


def test_空文件():
    with pytest.raises(EmptyFile):
        ingest("u001", "a.txt", b"")


def test_超大文件在读之前就被拒(monkeypatch: pytest.MonkeyPatch):
    from studio import ingest as mod

    monkeypatch.setattr(mod, "MAX_BYTES", 10)
    with pytest.raises(FileTooLarge):
        ingest("u001", "a.txt", b"x" * 11)


def test_假_PDF_靠魔数识破():
    """把 .txt 改名成 .pdf 是最常见的误用，必须在解析前拦住。"""
    with pytest.raises(CorruptFile) as exc:
        ingest("u001", "a.pdf", b"just some text")
    assert "%PDF" in str(exc.value)


def test_文本分页按空行分段且不超过目标字数():
    text = "\n\n".join("第%d段。" % i + "内容" * 200 for i in range(6))
    pages = paginate_text(text)
    assert len(pages) > 1
    assert [p[0] for p in pages] == list(range(1, len(pages) + 1))
    assert all(len(body) <= 1200 for _no, _t, body in pages)


def test_文本分页取最近的标题():
    text = "# 第一章\n\n正文一。\n\n## 1.1 小节\n\n正文二。"
    pages = paginate_text(text)
    assert pages[0][1] == "1.1 小节"


def test_文本资料端到端并切成块():
    text = "\n\n".join(
        f"第{i}段：系统聚类法需要先计算距离矩阵，再按最短距离合并类别。"
        f"这一段用来把正文撑长，保证虚拟分页会切出多页。" + "补充说明。" * 12
        for i in range(12)
    )
    result = ingest("u001", "notes.md", text.encode("utf-8"))
    assert result.kind == "text"
    assert result.page_count >= 2
    assert result.chunk_count >= 2
    assert result.lang == "zh"
    assert result.chunks[0].doc_id == "u001"
    # 块的页码引用必须落在真实页范围内
    assert result.chunks[0].page_start >= 1
    assert result.chunks[0].page_end <= result.page_count


def test_gbk_编码也能读():
    text = "第一章 基本概念\n\n" + "方差的无偏估计是样本协方差阵。" * 30
    result = ingest("u001", "old.txt", text.encode("gbk"))
    assert "方差" in result.full_text()


def test_乱码编码报_CorruptFile():
    with pytest.raises(CorruptFile):
        ingest("u001", "bad.txt", bytes([0xFF, 0xFE, 0x00, 0x01, 0x80, 0x81]))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("多元统计分析主要知识点", "zh"),
        ("Association rule mining with Apriori", "en"),
        ("", "en"),
    ],
)
def test_语种分档(text: str, expected: str):
    assert classify_lang(text) == expected


def test_中英混排判成_mixed():
    assert classify_lang("第 3 章 Clustering 聚类分析与 k-means 算法") == "mixed"


# ---------------------------------------------------------------- 真实 PDF（可跳过）

ZH_PDF = Path.home() / "Desktop" / "多元统计分析主要知识点（2026秋）.pdf"


@pytest.mark.skipif(not ZH_PDF.is_file(), reason="本机没有中文 PDF 样本")
def test_真实中文_PDF_能抽出中文正文():
    result = ingest("u001", ZH_PDF.name, ZH_PDF.read_bytes())
    assert result.lang == "zh"
    assert result.page_count == 2
    assert "聚类" in result.full_text()
    assert result.chunks and all(c.spans for c in result.chunks)
