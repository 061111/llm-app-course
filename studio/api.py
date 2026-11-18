"""M5-T7 · HTTP 接口：把 Studio 暴露成一组 REST 端点 + 一个单页界面。

【为什么用 FastAPI，而 pyproject 里它只在 web extra】
  核心四层（llmkit / corpus / retrieval / qa）不需要 Web 也能跑完 283 条测试。
  把 fastapi 放进基础依赖，等于让"只想跑评估"的人也装一套 Web 框架。
  所以：`uv sync --extra web` 才装，缺它时这个模块 import 失败是**预期行为**。

【错误处理只有一个出口】
  所有 StudioError 都带 status / code / hint，统一由异常处理器转成
  {"error": {"code", "message", "hint"}}。前端只需要读 error.hint 直接展示，
  不需要为每种错误写一个分支 —— 这也是为什么异常类里要带 hint（见 studio/errors.py）。

【上传为什么要自己读流】
  `await file.read()` 会把整个文件先读进内存再判大小。一个 2 GB 的"PDF"
  足以把进程打爆。这里按 1 MB 分块读，超过上限立刻中止并返回 413。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from llmkit.errors import ConfigError
from studio.errors import StudioError
from studio.library import MAX_UPLOAD_BYTES
from studio.service import Studio

STATIC_DIR = Path(__file__).resolve().parent / "static"
CHUNK = 1024 * 1024


def create_app(studio: Studio | None = None) -> FastAPI:
    app = FastAPI(title="资料问答服务", version="0.1.0")
    store = studio or Studio()
    app.state.studio = store

    # ------------------------------------------------------------ 错误出口

    @app.exception_handler(StudioError)
    async def _studio_error(_request: Request, exc: StudioError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"error": exc.as_dict()})

    @app.exception_handler(ConfigError)
    async def _config_error(_request: Request, exc: ConfigError) -> JSONResponse:
        """缺密钥不是 500：服务本身好的，是配置没给。

        返回 503 并直接告诉用户该设哪个变量，比 "Internal Server Error" 有用得多。
        """
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "llm_not_configured",
                    "message": str(exc),
                    "hint": "在项目根目录的 .env 里写 LLMKIT_API_KEY=...，然后重启服务",
                }
            },
        )

    # ------------------------------------------------------------ 页面

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        docs = store.docs()
        return {
            "ok": True,
            "docs": len(docs),
            "model": getattr(store.settings, "model", "") or "（未配置）",
            "base_url": getattr(store.settings, "base_url", "") or "",
        }

    # ------------------------------------------------------------ 文档

    @app.get("/api/docs")
    def list_docs() -> dict:
        return {"docs": [d.model_dump() for d in store.docs()]}

    @app.post("/api/docs", status_code=201)
    async def upload(file: UploadFile = File(...)) -> dict:
        name = file.filename or "未命名"
        buf = bytearray()
        while True:
            chunk = await file.read(CHUNK)
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) > MAX_UPLOAD_BYTES:
                from studio.errors import FileTooLarge

                raise FileTooLarge(f"{name} 超过 50 MB 上限")
        result = store.upload(name, bytes(buf))
        return {
            "doc": result.doc.model_dump(),
            "terms": list(result.terms),
        }

    @app.delete("/api/docs/{doc_id}")
    def delete_doc(doc_id: str) -> dict:
        store.remove(doc_id)
        return {"deleted": doc_id}

    # ------------------------------------------------------------ 术语与画像

    @app.get("/api/docs/{doc_id}/terms")
    def terms(doc_id: str, top_n: int = Query(default=20, ge=5, le=60)) -> dict:
        items = store.terms(doc_id, top_n=top_n)
        return {
            "doc_id": doc_id,
            "terms": [
                {
                    "term": t.text,
                    "freq": t.freq,
                    "score": t.score,
                    "kind": t.kind,
                    "pages": list(t.pages()),
                    "evidence": t.occurrences[0].context if t.occurrences else "",
                }
                for t in items
            ],
        }

    @app.get("/api/docs/{doc_id}/profile")
    def get_profile(doc_id: str) -> dict:
        profile = store.get_profile(doc_id)
        if profile is None:
            return {"doc_id": doc_id, "profile": None}
        return {"doc_id": doc_id, "profile": profile.model_dump()}

    @app.post("/api/docs/{doc_id}/profile")
    def build_profile(doc_id: str, refresh: bool = False) -> dict:
        profile = store.profile(doc_id, refresh=refresh)
        return {"doc_id": doc_id, "profile": profile.model_dump()}

    # ------------------------------------------------------------ 问答

    @app.post("/api/ask")
    def ask(payload: dict) -> dict:
        question = str(payload.get("question", "")).strip()
        doc_ids = payload.get("doc_ids") or None
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        top_k = int(payload.get("top_k", 5) or 5)
        if not 1 <= top_k <= 10:
            raise StudioError(
                "top_k 必须在 1~10 之间", code="bad_top_k", status=400, hint="一般用 5"
            )
        view = store.ask(question, doc_ids=doc_ids, top_k=top_k)
        return {
            "question": view.answer.question,
            "answer": view.answer.text,
            "citations": [c.source() for c in view.answer.citations],
            "unsupported": list(view.answer.unsupported),
            "denied": view.denied,
            "doc_ids": list(view.doc_ids),
            "top_k": view.top_k,
            "translated": view.translated,
            "profile_source": view.profile_source,
            "expanded_terms": list(view.expanded_terms),
            "latency_ms": view.answer.latency_ms,
            "cost_cny": view.answer.usage.cost_cny,
            "evidence": [e.model_dump() for e in view.evidence],
        }

    return app


app = create_app()
