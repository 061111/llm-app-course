"""上传登记：doc_id 生成规则、去重、删除的一致性。

这里测的都是"看起来会成功但数据错了"的地方 —— 那种 bug 不报错，
只在几天后表现为"文档列表里少了一份"。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio import library
from studio.errors import DocNotFound, DuplicateDoc, TooManyDocs

PDF_BYTES = b"%PDF-1.4 fake"


def _register(root: Path, name: str = "a.pdf", data: bytes = PDF_BYTES) -> library.LibraryDoc:
    return library.register(
        filename=name,
        data=data,
        pages=3,
        chars=100,
        lang="zh",
        suffix=".pdf",
        root=root,
    )


def test_doc_id_取文件名里的_ascii(tmp_path: Path):
    doc = _register(tmp_path, "Slide 03 Analysis.pdf")
    assert doc.doc_id == "slide_03_analysis"


def test_纯中文名退化成顺序号(tmp_path: Path):
    doc = _register(tmp_path, "多元统计分析.pdf")
    assert doc.doc_id == "u001"


def test_只有数字的名字也用顺序号():
    """实测踩过：'（计算题 3-6章 ）' 洗出 '3_6'，合法但像页码。"""
    assert library.alloc_doc_id("（计算题 3-6章 ）.pdf", set()) == "u001"


def test_重名自动加后缀():
    assert library.alloc_doc_id("a.pdf", {"a"}) == "a_2"
    assert library.alloc_doc_id("a.pdf", {"a", "a_2"}) == "a_3"


def test_doc_id_必须能被引用正则接受():
    """CITE_RE 只认 [A-Za-z0-9_]+，中文 doc_id 会让引用解析不出来。"""
    with pytest.raises(ValueError):
        library.LibraryDoc(
            doc_id="中文",
            filename="x.pdf",
            stored_name="x.pdf",
            pages=1,
            chars=1,
            sha256="0" * 64,
            lang="zh",
            uploaded_at="2026-01-01T00:00:00",
        )


def test_同一份文件重复上传会被识别(tmp_path: Path):
    _register(tmp_path)
    with pytest.raises(DuplicateDoc) as exc:
        _register(tmp_path, "b.pdf")
    assert exc.value.doc_id == "a"


def test_内容不同但同名可以共存(tmp_path: Path):
    a = _register(tmp_path, "a.pdf", PDF_BYTES)
    b = _register(tmp_path, "a.pdf", PDF_BYTES + b"x")
    assert (a.doc_id, b.doc_id) == ("a", "a_2")


def test_删除同时清掉文件和记录(tmp_path: Path):
    doc = _register(tmp_path)
    stored = library.stored_path(doc, tmp_path)
    assert stored.is_file()
    library.remove(doc.doc_id, tmp_path)
    assert not stored.exists()
    assert library.load_manifest(tmp_path).docs == []


def test_删除不存在的文档报_DocNotFound(tmp_path: Path):
    with pytest.raises(DocNotFound):
        library.remove("nope", tmp_path)


def test_登记表坏掉时当空表而不是崩溃(tmp_path: Path):
    path = library.manifest_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ 这不是 json", encoding="utf-8")
    assert library.load_manifest(tmp_path).docs == []


def test_登记表写入是原子替换(tmp_path: Path):
    """写完不该留下 .tmp 中间文件，否则下次读到半个 JSON。"""
    _register(tmp_path)
    leftovers = list(library.manifest_path(tmp_path).parent.glob("*.tmp"))
    assert leftovers == []


def test_保存后重新读回字段一致(tmp_path: Path):
    doc = _register(tmp_path)
    again = library.get(doc.doc_id, tmp_path)
    assert again == doc


def test_超过上限拒绝(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(library, "MAX_DOCS", 1)
    _register(tmp_path)
    with pytest.raises(TooManyDocs):
        _register(tmp_path, "b.pdf", PDF_BYTES + b"y")


def test_topic_回填不改变其它字段(tmp_path: Path):
    doc = _register(tmp_path)
    updated = library.touch_topic(doc.doc_id, "多元统计", tmp_path)
    assert updated.topic == "多元统计"
    assert updated.sha256 == doc.sha256
    assert library.get(doc.doc_id, tmp_path).topic == "多元统计"


def test_manifest_里的_json_可读(tmp_path: Path):
    _register(tmp_path)
    raw = json.loads(library.manifest_path(tmp_path).read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["docs"][0]["doc_id"] == "a"
