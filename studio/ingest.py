"""M5-T3 · 解析：任何受支持的文件 → 逐页文本 → 块。

这一层刻意**薄**：PDF 的清洗规则（按 bbox 删页码、私有区字符映射、书签取标题）
已经在 corpus/parser.py 和 corpus/chunker.py 里，且每条规则都有实测依据。
上传层再写一套，等于维护两个真相。所以这里只做三件事：

  1. **安检**：后缀、大小、魔数、是否加密、是否有文本层；
  2. **补齐 PDF 没有的输入**：.txt / .md 的分页与标题；
  3. **复用**：`extract_pages` + `read_titles` + `build_chunks`，参数一个不改。

【为什么 .txt/.md 也要支持】
  用户手头的资料不一定是 PDF（笔记、导出的会议纪要、从网页摘的正文）。
  给它一个真实可用的入口，比要求"请先转成 PDF"更合理。
  但文本文件没有"页"，所以采用**虚拟分页**：按空行分段，累积到约 1000 字算一页。
  引用里的 pN 就指这个虚拟页 —— 文档里必须写清楚这一点，否则用户会去找不存在的页码。

【安检顺序是有讲究的】
  后缀 → 大小 → 魔数 → 解析 → 文本量。前三个不需要打开文件，最便宜；
  "有没有文本层"只能解析完才知道，所以放最后。反过来写，一个 200 MB 的
  假 PDF 会先被完整读进内存再告诉你格式不对。
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from corpus.chunker import Chunk, build_chunks, read_titles
from corpus.parser import PageText, extract_pages
from retrieval.cjk import cjk_ratio
from studio.errors import (
    CorruptFile,
    EmptyFile,
    EncryptedPdf,
    FileTooLarge,
    NoTextLayer,
    UnsupportedType,
)

SUPPORTED_SUFFIXES = {".pdf": "pdf", ".txt": "text", ".md": "text"}

MAX_BYTES = 50 * 1024 * 1024

# 平均每页少于这么多字符，就当"没有文本层"。
# 依据：本机中文 PDF 实测正常页 500~900 字符，扫描件页 0 字符；
# 20 这个阈值给"只有一张图 + 一句图注"的页面留了余量。
MIN_CHARS_PER_PAGE = 20

# 文本文件的虚拟分页目标字数。取 1000 与 MAX_CHUNK_CHARS=1200 同量级：
# 一页基本对应一个块，页码才不至于和块完全脱节。
TEXT_PAGE_CHARS = 1000

Kind = Literal["pdf", "text"]


class IngestResult(BaseModel):
    """解析结果。pages 是原始页，chunks 是检索单元。"""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    doc_id: str
    kind: Kind
    suffix: str
    pages: tuple[PageText, ...]
    chunks: tuple[Chunk, ...]
    lang: Literal["zh", "en", "mixed"]
    chars: int = Field(ge=0)
    empty_pages: int = Field(ge=0)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


def detect_kind(filename: str) -> tuple[str, Kind]:
    """返回 (后缀, 类型)。不认识的扩展名直接 415，不做内容嗅探以外的猜测。"""
    suffix = Path(filename).suffix.lower()
    kind = SUPPORTED_SUFFIXES.get(suffix)
    if kind is None:
        raise UnsupportedType(
            f"不支持的文件类型：{suffix or '（无扩展名）'}",
            hint="只接受 .pdf / .txt / .md；Word 请先另存为 PDF",
        )
    return suffix, kind  # type: ignore[return-value]


def classify_lang(text: str) -> Literal["zh", "en", "mixed"]:
    """按汉字占"汉字+拉丁字母"的比例分档。三条线是实测定的（2026-11-18）：

      中文习题解答 0.945 / 中文考点 0.906 / 中文样题 0.882  → zh
      chap03 英文讲义 0.000                              → en
      本项目 README 0.493 / bm25.py 0.337                 → mixed

    0.60 / 0.10 两条线把三者干净分开。mixed 不是"出错"，它表示资料里两种语言
    都有（典型是"英文教材 + 中文批注"）；检索时按**不翻译**处理，
    因为中文查询词在中文块里能直接命中，翻译反而会把它们送进英文语义空间。
    """
    r = cjk_ratio(text)
    if r >= 0.60:
        return "zh"
    if r <= 0.10:
        return "en"
    return "mixed"


def paginate_text(text: str) -> list[tuple[int, str, str]]:
    """把纯文本拆成 [(页码, 标题, 正文)]。按空行分段，累积到约 TEXT_PAGE_CHARS 字算一页。

    标题取该页内**最后一个** Markdown 一级/二级标题（正文里最近的章节名）；
    没有标题就留空 —— 不编一个"第 N 部分"出来。
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    if not blocks:
        return []

    heading_re = re.compile(r"^#{1,2}\s+(.+)$", re.M)
    out: list[tuple[int, str, str]] = []
    buf: list[str] = []
    size = 0
    for block in blocks:
        if buf and size + len(block) > TEXT_PAGE_CHARS:
            out.append((len(out) + 1, _last_heading(buf, heading_re), "\n\n".join(buf)))
            buf, size = [], 0
        buf.append(block)
        size += len(block)
    if buf:
        out.append((len(out) + 1, _last_heading(buf, heading_re), "\n\n".join(buf)))
    return out


def _last_heading(blocks: list[str], heading_re: re.Pattern[str]) -> str:
    for block in reversed(blocks):
        m = heading_re.search(block)
        if m:
            return m.group(1).strip()
    return ""


def _ingest_pdf(doc_id: str, path: Path, data: bytes) -> IngestResult:
    try:
        pages = extract_pages(doc_id, path=path)
    except FileNotFoundError as exc:  # 理论上不会发生：临时文件刚写完
        raise CorruptFile(f"临时文件丢失：{path}") from exc
    except Exception as exc:  # noqa: BLE001 - pymupdf 的异常类型不稳定
        msg = str(exc).lower()
        if "password" in msg or "encrypt" in msg:
            raise EncryptedPdf("PDF 已加密，无法读取正文") from exc
        raise CorruptFile(f"PDF 解析失败：{exc}") from exc

    chars = sum(len(p.text) for p in pages)
    if pages and chars / len(pages) < MIN_CHARS_PER_PAGE:
        raise NoTextLayer(
            f"{len(pages)} 页只抽出 {chars} 个字符（平均 {chars / len(pages):.1f} 字/页）"
        )

    titles = read_titles(doc_id, path=path)
    chunks = build_chunks(doc_id, pages=pages, titles=titles)
    return IngestResult(
        doc_id=doc_id,
        kind="pdf",
        suffix=".pdf",
        pages=tuple(pages),
        chunks=tuple(chunks),
        lang=classify_lang("\n".join(p.text for p in pages)),
        chars=chars,
        empty_pages=sum(1 for p in pages if p.is_empty),
    )


def _ingest_text(doc_id: str, path: Path, data: bytes) -> IngestResult:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("gbk")  # 中文 Windows 导出的 txt 常见编码
        except UnicodeDecodeError as exc:
            raise CorruptFile("文本既不是 UTF-8 也不是 GBK，无法确定编码") from exc

    if not text.strip():
        raise EmptyFile("文本文件里没有内容")

    paginated = paginate_text(text)
    pages = [
        PageText(
            doc_id=doc_id,
            page=no,
            text=body,
            raw_chars=len(body),
            is_empty=len(body) < 50,
        )
        for no, _title, body in paginated
    ]
    titles = {no: title for no, title, _body in paginated}
    chunks = build_chunks(doc_id, pages=pages, titles=titles)
    return IngestResult(
        doc_id=doc_id,
        kind="text",
        suffix=path.suffix.lower(),
        pages=tuple(pages),
        chunks=tuple(chunks),
        lang=classify_lang(text),
        chars=sum(len(p.text) for p in pages),
        empty_pages=sum(1 for p in pages if p.is_empty),
    )


def ingest(doc_id: str, filename: str, data: bytes) -> IngestResult:
    """安检 + 解析。**不写任何持久文件**（除了一个解析用的临时文件）。

    调用方拿到 IngestResult 之后才去 library.register() 落盘登记，
    这样"解析失败"永远不会留下半条记录 —— 顺序反过来就会。
    """
    suffix, kind = detect_kind(filename)
    if not data:
        raise EmptyFile("上传的文件是 0 字节")
    if len(data) > MAX_BYTES:
        raise FileTooLarge(f"文件 {len(data) / 1024 / 1024:.1f} MB，超过 50 MB 上限")
    if kind == "pdf" and not data[:5].startswith(b"%PDF-"):
        raise CorruptFile(
            "这个文件没有 PDF 文件头（%PDF-），可能是把别的格式改了扩展名"
        )

    if kind == "pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return _ingest_pdf(doc_id, tmp_path, data)
        finally:
            tmp_path.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return _ingest_text(doc_id, tmp_path, data)
    finally:
        tmp_path.unlink(missing_ok=True)
