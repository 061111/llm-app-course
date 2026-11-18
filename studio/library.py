"""M5-T2 · 上传登记：把用户丢进来的文件变成一条可寻址的 doc_id。

【为什么不复用 corpus/docs.py 的 DOCS】
  DOCS 是**评测语料**的登记表：18 道黄金题逐题绑在 chap03…chap05 上，
  并且有测试断言"每份讲义至少出一道题""页数合计 199"。
  用户上传的文档不属于评测集，一旦塞进 DOCS，评测就被污染了
  （见 docs/upload-service.md 里"为什么必须拆成两张表"）。

  所以这里另起一张表：artifacts/library.json。两个关键差别：
    1. 可写。DOCS 是代码里的常量，上传是运行时行为。
    2. 带 sha256。同一份文件重复上传必须能识别出来，否则索引里会有两份
       一模一样的块，BM25 的 IDF 会被稀释，检索结果莫名其妙地变差。

【doc_id 只能是什么】
  引用格式被 CITE_RE 锁死为 `【doc_id:pN】`，其中 doc_id 匹配 `[A-Za-z0-9_]+`。
  所以中文文件名不能直接当 doc_id —— "多元统计分析.pdf" 生成不了可解析的引用。
  规则：取文件名的 ASCII 部分做 slug；一个 ASCII 字符都没有（纯中文名）就用
  `u001` 这样的顺序号。两种都能被引用正则接受。

【文件放哪】
  artifacts/uploads/{doc_id}{后缀}。artifacts/ 已在 .gitignore 里，
  用户上传的资料不会被误提交到仓库 —— 这点比"方便"重要。
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from studio.errors import DocNotFound, DuplicateDoc, StudioError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "artifacts"

UPLOAD_SUBDIR = "uploads"
MANIFEST_NAME = "library.json"

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_DOCS = 200

Lang = Literal["zh", "en", "mixed"]

DOC_ID_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")

# 进程内串行化读改写。FastAPI 的同步端点跑在线程池里，两个上传同时进来
# 会各自读到旧 manifest 再各自写回 —— 丢一条记录，且症状是"上传成功但列表里没有"。
_LOCK = threading.RLock()


class LibraryDoc(BaseModel):
    """一份上传资料的身份证。frozen：登记后不该被悄悄改页数。"""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="引用里用的短名，只能 ASCII 字母数字下划线")
    filename: str = Field(min_length=1, description="用户上传时的原始文件名")
    stored_name: str = Field(min_length=1, description="落盘后的文件名（doc_id + 后缀）")
    pages: int = Field(gt=0)
    chars: int = Field(ge=0, description="正文总字符数（清洗后）")
    sha256: str = Field(min_length=64, max_length=64)
    lang: Lang
    topic: str = Field(default="", description="一句话主题；画像生成后回填")
    uploaded_at: str = Field(description="ISO 时间，展示用")

    @field_validator("doc_id")
    @classmethod
    def _check_doc_id(cls, v: str) -> str:
        if not DOC_ID_RE.match(v):
            raise ValueError(f"doc_id 只能由字母/数字/下划线组成（引用格式要求），收到 {v!r}")
        return v


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    docs: list[LibraryDoc] = Field(default_factory=list)


def _root(root: Path | None) -> Path:
    return Path(root) if root else DEFAULT_ROOT


def manifest_path(root: Path | None = None) -> Path:
    return _root(root) / MANIFEST_NAME


def upload_dir(root: Path | None = None) -> Path:
    return _root(root) / UPLOAD_SUBDIR


def stored_path(doc: LibraryDoc, root: Path | None = None) -> Path:
    return upload_dir(root) / doc.stored_name


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(root: Path | None = None) -> Manifest:
    """读登记表。文件不存在 = 空表；文件坏了 = 空表 + 不覆盖（下次写才重建）。

    损坏时选择"当空表"而不是抛异常：上传是主流程，一张坏掉的登记表
    不应该让整个服务起不来。但也不能静默删掉它 —— 这里不写回，
    用户下次上传时才覆盖，至少给人留了一次手动抢救的机会。
    """
    p = manifest_path(root)
    if not p.is_file():
        return Manifest()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return Manifest.model_validate(raw)
    except Exception:  # noqa: BLE001 - 兜底：坏文件不阻塞主流程
        return Manifest()


def save_manifest(manifest: Manifest, root: Path | None = None) -> None:
    p = manifest_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    # 先写临时文件再替换：写一半被打断也不会留下半个 JSON
    tmp.write_text(
        json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def alloc_doc_id(filename: str, existing: set[str]) -> str:
    """从文件名生成 doc_id，保证唯一。

    chap03.pdf            → chap03（已存在 → chap03_2）
    paper_v2.pdf          → paper_v2
    多元统计分析.pdf       → u001（无 ASCII 字母）
    计算题 3-6章.pdf       → u001（只有数字，没有字母 —— 见下）

    **要求 slug 里至少有一个 ASCII 字母**。这条是实测踩出来的：
    "（计算题 3-6章 ）" 洗出来的 slug 是 "3_6"，虽然合法、能被引用解析，
    但一个纯数字的 doc_id 既没有信息量，又容易被误读成页码。
    """
    stem = Path(filename).stem
    slug = _SLUG_RE.sub("_", stem).strip("_").lower()[:24]
    usable = bool(slug) and any(ch.isalpha() for ch in slug)
    if usable and slug not in existing:
        return slug

    if usable:
        for i in range(2, 100):
            cand = f"{slug}_{i}"
            if cand not in existing:
                return cand

    n = 1
    while f"u{n:03d}" in existing:
        n += 1
    return f"u{n:03d}"


def register(
    *,
    filename: str,
    data: bytes,
    pages: int,
    chars: int,
    lang: Lang,
    suffix: str,
    doc_id: str | None = None,
    root: Path | None = None,
) -> LibraryDoc:
    """写盘 + 登记。**解析成功之后**才调用 —— 解析失败不留半条记录。

    doc_id 允许由调用方预先指定：解析阶段（corpus.parser）就要把 doc_id 写进
    每页和每个块的 id 里，等解析完再改名意味着整批复写。所以流程是
    "先占号 → 用这个号解析 → 登记时校验号还没被抢"。
    """
    if not data:
        raise StudioError("文件内容为空")
    digest = sha256_bytes(data)

    with _LOCK:
        manifest = load_manifest(root)
        for doc in manifest.docs:
            if doc.sha256 == digest:
                raise DuplicateDoc(
                    f"这份文件已经上传过，doc_id={doc.doc_id}",
                    doc_id=doc.doc_id,
                )
        if len(manifest.docs) >= MAX_DOCS:
            from studio.errors import TooManyDocs

            raise TooManyDocs(f"已有 {len(manifest.docs)} 份文档，达到上限 {MAX_DOCS}")

        existing = {d.doc_id for d in manifest.docs}
        if doc_id is None:
            doc_id = alloc_doc_id(filename, existing)
        elif doc_id in existing:
            raise StudioError(
                f"doc_id={doc_id!r} 已被占用，请重新上传",
                code="doc_id_taken",
                status=409,
            )

        stored = f"{doc_id}{suffix}"
        dest = upload_dir(root) / stored
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

        doc = LibraryDoc(
            doc_id=doc_id,
            filename=filename,
            stored_name=stored,
            pages=pages,
            chars=chars,
            sha256=digest,
            lang=lang,
            uploaded_at=datetime.now().isoformat(timespec="seconds"),
        )
        manifest.docs.append(doc)
        save_manifest(manifest, root)
        return doc


def get(doc_id: str, root: Path | None = None) -> LibraryDoc:
    for doc in load_manifest(root).docs:
        if doc.doc_id == doc_id:
            return doc
    raise DocNotFound(f"没有 doc_id={doc_id!r} 这份文档")


def remove(doc_id: str, root: Path | None = None) -> LibraryDoc:
    """删记录 + 删文件 + 删画像。记录优先：文件删不掉也不能留下悬空条目。"""
    with _LOCK:
        manifest = load_manifest(root)
        keep = [d for d in manifest.docs if d.doc_id != doc_id]
        if len(keep) == len(manifest.docs):
            raise DocNotFound(f"没有 doc_id={doc_id!r} 这份文档")
        removed = next(d for d in manifest.docs if d.doc_id == doc_id)
        save_manifest(Manifest(version=manifest.version, docs=keep), root)

    for p in (
        stored_path(removed, root),
        _root(root) / "profiles" / f"{doc_id}.json",
    ):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    return removed


def touch_topic(doc_id: str, topic: str, root: Path | None = None) -> LibraryDoc:
    """画像生成后回填一句话主题。frozen 模型 → 用 model_copy 造新对象。"""
    with _LOCK:
        manifest = load_manifest(root)
        docs = [
            (d.model_copy(update={"topic": topic}) if d.doc_id == doc_id else d)
            for d in manifest.docs
        ]
        if not any(d.doc_id == doc_id for d in manifest.docs):
            raise DocNotFound(f"没有 doc_id={doc_id!r} 这份文档")
        save_manifest(Manifest(version=manifest.version, docs=docs), root)
        return next(d for d in docs if d.doc_id == doc_id)
