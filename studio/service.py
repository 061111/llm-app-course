"""M5-T6 · 服务编排：上传 → 解析 → 画像 → 检索 → 回答，全部在一个类里。

这一层只做"顺序"和"缓存"，不做算法：
  - 解析规则在 corpus/*，术语在 studio/glossary.py，画像在 studio/profile.py，
    检索在 retrieval/*，回答在 qa/*；
  - 这里负责把它们按正确顺序串起来，并在正确的时机复用（不重复解析 30 页 PDF）。

【缓存的三条规则】
  1. 解析结果按 (doc_id, 文件大小, 修改时间) 缓存。文件被替换时键自动失效。
  2. 检索器按"文档集合"缓存。同一批文档问 10 个问题只建一次 BM25 索引。
  3. 画像**不缓存内存**，只读磁盘 JSON —— 它是产物，可能被外部改，
     以磁盘为准比以内存为准更容易解释。

【为什么 translate 由文档语种决定，而不是让用户选】
  语料是英文时，中文问题必须先翻成英文关键词才可能命中（M3 的实测：不翻译 72%，
  翻译 94%）；语料是中文时，翻译反而把查询送进英文语义空间，命中率直接归零。
  这是**资料属性**决定的事，不该变成用户要理解的一个开关。
  留一个参数只为评估时能做对照实验（"如果不翻译会怎样"）。
"""
from __future__ import annotations

import threading
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from corpus.chunker import Chunk
from corpus.parser import PageText
from qa.answer import Answer, answer_question
from retrieval.search import SearchResult, Searcher
from studio import library
from studio.errors import DocNotFound, StudioError
from studio.glossary import Term, mine_terms
from studio.ingest import IngestResult, ingest
from studio.profile import (
    DomainProfile,
    build_profile,
    load_profile,
    profile_path,
    save_profile,
)


class DocView(BaseModel):
    """给前端看的文档卡片。不吐内部字段（sha256 / stored_name 没必要暴露）。"""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    filename: str
    pages: int
    chars: int
    lang: str
    topic: str
    uploaded_at: str
    has_profile: bool
    chunks: int = 0


class UploadResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc: DocView
    terms: tuple[str, ...] = Field(description="统计层抽到的前 8 个候选，供上传后立即展示")

    @property
    def doc_id(self) -> str:
        return self.doc.doc_id


class Evidence(BaseModel):
    """一条检索依据。text 截断到 400 字，够核对，不至于把响应撑到几百 KB。"""

    model_config = ConfigDict(frozen=True)

    source: str
    title: str
    score: float
    text: str


class AnswerView(BaseModel):
    """一次问答的完整结果：答案 + 引用校验 + 依据 + 走了哪条路径。"""

    model_config = ConfigDict(frozen=True)

    answer: Answer
    evidence: tuple[Evidence, ...]
    doc_ids: tuple[str, ...]
    top_k: int
    translated: bool = Field(description="是否走了中文问题到英文关键词的翻译")
    profile_source: str = Field(default="", description="llm / statistical / 空")
    expanded_terms: tuple[str, ...] = Field(default=(), description="用领域措辞扩展出的检索词")

    @property
    def denied(self) -> bool:
        """是否"完全拒答"。

        判定要区分两种情况（实测出来的）：
          - 模型说"资料未覆盖"且**没有任何引用** → 真拒答；
          - 模型先答了资料里有的部分，再补一句"X 的具体含义资料未覆盖" → 这是
            **部分覆盖**，答案有效，不能挂"拒答"横幅，否则用户会以为白问了。
        所以要求"声明未覆盖"与"零引用"同时成立，或检索本身为空。
        """
        if not self.evidence:
            return True
        return self.answer.claims_uncovered and not self.answer.citations


class Studio:
    def __init__(
        self,
        root: Path | None = None,
        *,
        settings=None,
        client=None,
        llm=None,
        translator=None,
    ) -> None:
        self.root = Path(root) if root else library.DEFAULT_ROOT
        # 没显式给配置就自己读一次 .env：上传服务是长驻进程，
        # 不该在每次提问时才去解析一遍环境变量。
        # 读不到就保持 None —— 上传/术语/画像（纯统计）仍然可用，
        # 只有真正要调模型的那一步才会报 ConfigError。
        if settings is None:
            from llmkit.config import Settings
            from llmkit.errors import ConfigError

            try:
                settings = Settings.from_env()
            except ConfigError:
                settings = None
        self.settings = settings
        self.client = client
        self.llm = llm
        self.translator = translator
        self._lock = threading.RLock()
        self._ingest_cache: dict[tuple[str, int, float], IngestResult] = {}
        self._searcher_cache: dict[tuple[str, ...], Searcher] = {}

    # ---------------------------------------------------------------- 文档

    def _stored(self, doc: library.LibraryDoc) -> Path:
        p = library.stored_path(doc, self.root)
        if not p.is_file():
            raise StudioError(
                f"文件丢了：{p.name}（登记表里有，磁盘上没有）",
                code="file_missing",
                hint="删除这份文档后重新上传",
            )
        return p

    def _ingest(self, doc: library.LibraryDoc) -> IngestResult:
        path = self._stored(doc)
        stat = path.stat()
        key = (doc.doc_id, stat.st_size, stat.st_mtime)
        with self._lock:
            hit = self._ingest_cache.get(key)
            if hit is not None:
                return hit
            result = ingest(doc.doc_id, doc.filename, path.read_bytes())
            self._ingest_cache = {k: v for k, v in self._ingest_cache.items() if k[0] == doc.doc_id}
            self._ingest_cache[key] = result
            return result

    def pages(self, doc_id: str) -> list[PageText]:
        return list(self._ingest(library.get(doc_id, self.root)).pages)

    def chunks(self, doc_id: str) -> list[Chunk]:
        return list(self._ingest(library.get(doc_id, self.root)).chunks)

    def terms(self, doc_id: str, *, top_n: int = 40) -> list[Term]:
        return mine_terms(self.pages(doc_id), top_n=top_n)

    def _view(self, doc: library.LibraryDoc) -> DocView:
        chunks = 0
        try:
            chunks = self._ingest(doc).chunk_count
        except StudioError:
            chunks = 0
        return DocView(
            doc_id=doc.doc_id,
            filename=doc.filename,
            pages=doc.pages,
            chars=doc.chars,
            lang=doc.lang,
            topic=doc.topic,
            uploaded_at=doc.uploaded_at,
            has_profile=profile_path(doc.doc_id, self.root / "profiles").is_file(),
            chunks=chunks,
        )

    def docs(self) -> list[DocView]:
        return [self._view(d) for d in library.load_manifest(self.root).docs]

    def upload(self, filename: str, data: bytes) -> UploadResult:
        """解析成功才登记。顺序见 studio/ingest.py 的说明。"""
        existing = {d.doc_id for d in library.load_manifest(self.root).docs}
        doc_id = library.alloc_doc_id(filename, existing)

        result = ingest(doc_id, filename, data)
        doc = library.register(
            filename=filename,
            data=data,
            pages=result.page_count,
            chars=result.chars,
            lang=result.lang,
            suffix=result.suffix,
            doc_id=doc_id,
            root=self.root,
        )
        with self._lock:
            self._searcher_cache.clear()
        terms = mine_terms(result.pages, top_n=8)
        return UploadResult(doc=self._view(doc), terms=tuple(t.text for t in terms))

    def remove(self, doc_id: str) -> None:
        library.remove(doc_id, self.root)
        with self._lock:
            self._searcher_cache.clear()
            self._ingest_cache = {k: v for k, v in self._ingest_cache.items() if k[0] != doc_id}

    # ---------------------------------------------------------------- 画像

    def profile(self, doc_id: str, *, refresh: bool = False) -> DomainProfile:
        doc = library.get(doc_id, self.root)
        path = profile_path(doc_id, self.root / "profiles")
        if not refresh:
            cached = load_profile(path)
            if cached is not None:
                return cached

        pages = self.pages(doc_id)
        candidates = mine_terms(pages, top_n=40)
        profile = build_profile(
            pages=pages,
            terms=candidates,
            doc_ids=(doc_id,),
            llm=self.llm,
            settings=self.settings,
            client=self.client,
            allow_fallback=True,
            topic_hint=doc.topic or Path(doc.filename).stem,
        )
        save_profile(profile, path)
        if profile.source == "llm" and profile.summary:
            library.touch_topic(doc_id, profile.domain, self.root)
        return profile

    def get_profile(self, doc_id: str) -> DomainProfile | None:
        return load_profile(profile_path(doc_id, self.root / "profiles"))

    # ---------------------------------------------------------------- 检索与问答

    def _selected(self, doc_ids: list[str] | None) -> list[library.LibraryDoc]:
        all_docs = library.load_manifest(self.root).docs
        if not doc_ids:
            return all_docs
        wanted = set(doc_ids)
        picked = [d for d in all_docs if d.doc_id in wanted]
        missing = wanted - {d.doc_id for d in picked}
        if missing:
            raise DocNotFound(f"这些 doc_id 不存在：{sorted(missing)}")
        return picked

    def searcher(self, doc_ids: list[str] | None = None) -> Searcher:
        docs = self._selected(doc_ids)
        if not docs:
            raise StudioError(
                "还没有上传任何资料",
                code="empty_library",
                status=409,
                hint="先上传一份 PDF 或 txt，再提问",
            )
        key = tuple(d.doc_id for d in docs)
        with self._lock:
            hit = self._searcher_cache.get(key)
            if hit is not None:
                return hit
            chunks: list[Chunk] = []
            for d in docs:
                chunks.extend(self._ingest(d).chunks)
            # 中文资料不能走"中文→英文"翻译：翻完查询词和中文正文一个字都对不上。
            need_translate = all(d.lang == "en" for d in docs)
            searcher = Searcher(
                chunks,
                translator=(self.translator if need_translate else None),
            )
            self._searcher_cache[key] = searcher
            return searcher

    def _profiles_for(self, docs: list[library.LibraryDoc]) -> DomainProfile | None:
        """多文档时取"已生成画像"的合并视图；一份都没生成就返回 None。"""
        profiles = [
            p for p in (self.get_profile(d.doc_id) for d in docs) if p is not None
        ]
        if not profiles:
            return None
        if len(profiles) == 1:
            return profiles[0]
        seen: dict[str, None] = {}
        for p in profiles:
            for t in p.terms:
                seen.setdefault(t.term, None)
        from studio.profile import ProfileTerm

        merged_terms = tuple(t for p in profiles for t in p.terms)
        uniq: dict[str, ProfileTerm] = {}
        for t in merged_terms:
            uniq.setdefault(t.term, t)
        return DomainProfile(
            doc_ids=tuple(d.doc_id for d in docs),
            domain="、".join(dict.fromkeys(p.domain for p in profiles)),
            summary="",
            terms=tuple(uniq.values()),
            phrasings=tuple(ph for p in profiles for ph in p.phrasings),
            source="llm" if any(p.source == "llm" for p in profiles) else "statistical",
        )

    def ask(
        self,
        question: str,
        *,
        doc_ids: list[str] | None = None,
        top_k: int = 5,
        translate: bool | None = None,
    ) -> AnswerView:
        question = question.strip()
        if not question:
            raise StudioError(
                "问题为空", code="empty_question", status=400, hint="说一句想问什么"
            )

        docs = self._selected(doc_ids)
        searcher = self.searcher([d.doc_id for d in docs])
        profile = self._profiles_for(docs)
        if translate is None:
            translate = all(d.lang == "en" for d in docs)

        results: list[SearchResult] = searcher.search(
            question, top_k=top_k, translate=translate
        )
        expanded: tuple[str, ...] = ()
        if profile is not None and not translate:
            # 领域措辞扩展：问题与资料里"可能怎么问"高度重合时，把它标注的术语一并检索。
            # 只在第一次检索**一个都没命中**时启用 —— 命中时不扩展，
            # 避免用一个"差不多的问题"的术语污染本来正确的排序。
            extra = profile.expansion_terms(question)
            if extra and not results:
                expanded = extra
                results = searcher.search_terms(list(extra) + [question], top_k=top_k)

        answer = answer_question(
            question,
            searcher=searcher,
            top_k=top_k,
            settings=self.settings,
            client=self.client,
            translate=translate,
            profile=profile,
            results=results if expanded else None,
        )
        return AnswerView(
            answer=answer,
            evidence=tuple(
                Evidence(
                    source=r.chunk.source(),
                    title=r.chunk.title,
                    score=r.score,
                    text=r.chunk.text[:400],
                )
                for r in results
            ),
            doc_ids=tuple(d.doc_id for d in docs),
            top_k=top_k,
            translated=translate,
            profile_source=profile.source if profile else "",
            expanded_terms=expanded,
        )
