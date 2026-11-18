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

from llmkit.errors import ConfigError, LLMError
from studio import balance
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

    @app.exception_handler(LLMError)
    async def _llm_error(_request: Request, exc: LLMError) -> JSONResponse:
        """模型侧失败也要有话说，不能掉成裸 500。

        实测（2026-11-18）：账户余额不足时上游返回 402，异常是 `FatalError`，
        没有处理器就变成 500 + 空响应，前端只能显示 HTTP 500。
        这里把上游状态码翻译成"用户能行动"的提示：401/402/403 是账户问题（503），
        其余（超时、5xx）是临时故障（502）。
        """
        status = 503 if exc.status in (401, 402, 403) else 502
        hint = (
            "检查 .env 里的 LLMKIT_API_KEY，以及账户余额/配额"
            if status == 503
            else "上游暂时不可用，稍后重试；连续失败先看服务端日志"
        )
        return JSONResponse(
            status_code=status,
            content={"error": {"code": "llm_error", "message": str(exc), "hint": hint}},
        )

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
            "has_key": bool(getattr(store.settings, "api_key", "")),
        }

    # ------------------------------------------------------------ 余额

    @app.get("/api/balance")
    def coupon_balance(refresh: bool = False) -> dict:
        """代金券余额。查的是阿里云账单接口，不是模型接口。

        三种状态都返回 200：未配置 AK、上游拒绝、查到了。
        前端据此显示"怎么配"、"为什么失败"或数字——余额查不到不该变成红色报错，
        它跟问答本身没关系，不该让人以为服务坏了。
        """
        return balance.query(store.settings, refresh=refresh).to_payload()

    # ------------------------------------------------------------ 模型

    @app.get("/api/models")
    def models(refresh: bool = False) -> dict:
        """可选模型清单。

        清单来自上游 `GET /models`（缓存 10 分钟），拿不到时降级成内置推荐表，
        `source` 字段如实标注——前端据此提示"这份清单可能不是最新的"。
        故意不抛异常：模型列表拉不到不该让整个界面打不开。
        """
        info = store.models(refresh=refresh)
        return {
            "current": info.current,
            "base_url": info.base_url,
            "source": info.source,
            "note": info.note,
            "checked_at": info.checked_at,
            "recommended": [m.model_dump() for m in info.recommended],
            "others": list(info.others),
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
    def build_profile(
        doc_id: str, refresh: bool = False, model: str | None = None
    ) -> dict:
        profile = store.profile(doc_id, refresh=refresh, model=model)
        return {"doc_id": doc_id, "profile": profile.model_dump()}

    # ------------------------------------------------------------ 自检

    @app.post("/api/docs/{doc_id}/selfcheck")
    def selfcheck(
        doc_id: str,
        n_terms: int = Query(default=5, ge=1, le=8),
        n_negative: int = Query(default=3, ge=1, le=5),
        model: str | None = None,
    ) -> dict:
        """领域无关自检：只判"引用对不对得上页""资料外的问题拒没拒"。

        为什么是 POST：它会真的调模型、花钱（默认约 ¥0.03）。GET 带副作用
        容易被浏览器预取、被爬虫点到，账单会莫名其妙涨。
        """
        report = store.selfcheck(
            [doc_id], n_terms=n_terms, n_negative=n_negative, model=model
        )
        return {"doc_id": doc_id, **report.to_payload()}

    # ------------------------------------------------------------ 问答

    @app.post("/api/ask")
    def ask(payload: dict) -> dict:
        question = str(payload.get("question", "")).strip()
        doc_ids = payload.get("doc_ids") or None
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        model = str(payload.get("model") or "").strip() or None
        top_k = int(payload.get("top_k", 5) or 5)
        if not 1 <= top_k <= 10:
            raise StudioError(
                "top_k 必须在 1~10 之间", code="bad_top_k", status=400, hint="一般用 5"
            )
        view = store.ask(question, doc_ids=doc_ids, top_k=top_k, model=model)
        return {
            "question": view.answer.question,
            "answer": view.answer.text,
            "citations": [c.source() for c in view.answer.citations],
            "unsupported": list(view.answer.unsupported),
            "denied": view.denied,
            "doc_ids": list(view.doc_ids),
            "top_k": view.top_k,
            "model": view.model,
            "translated": view.translated,
            "profile_source": view.profile_source,
            "expanded_terms": list(view.expanded_terms),
            "latency_ms": view.answer.latency_ms,
            "cost_cny": view.answer.usage.cost_cny,
            "evidence": [e.model_dump() for e in view.evidence],
        }

    return app


app = create_app()
