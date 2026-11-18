"""M5-T0 · 上传层的异常：每个错误码对应一个"用户能照做"的处置。

为什么要单独定义，而不是到处 `raise ValueError`：
  上传是唯一由**用户输入**驱动的入口，出错原因必须能被前端原样展示。
  ValueError 只能给一句人话，前端还得自己猜是"格式不支持"还是"文件太大"。

  这里每个异常固定三件事：`code`（机器判分支）、`status`（HTTP 码）、
  `hint`（告诉用户下一步做什么）。三样都缺一个，前端就会退化成"上传失败"四个字。

【一个刻意的取舍】
  所有业务异常都带 `status`，但 `service.py` 不依赖 FastAPI ——
  CLI 和测试也走同一套异常，只是不看 status。这样"服务层"不用为了 Web 而存在。
"""
from __future__ import annotations


class StudioError(Exception):
    """上传/解析/画像三类失败的基类。

    `status` / `code` / `hint` 都可以在构造时覆盖：基类默认 500 是"没预料到的错"，
    而"问题为空""资料库为空"这类是**可预期的坏请求**，必须显式给 400/409 ——
    否则前端会把用户自己能改的输入问题显示成"服务器错误"。
    """

    code = "internal"
    status = 500
    hint = ""

    def __init__(
        self,
        message: str,
        *,
        hint: str = "",
        code: str | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if hint:
            self.hint = hint
        if code:
            self.code = code
        if status:
            self.status = status

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


class UnsupportedType(StudioError):
    code = "unsupported_type"
    status = 415
    hint = "只接受 .pdf / .txt / .md"


class FileTooLarge(StudioError):
    code = "file_too_large"
    status = 413
    hint = "单个文件上限 50 MB；超大的扫描件请先拆分或压缩"


class EmptyFile(StudioError):
    code = "empty_file"
    status = 400
    hint = "文件是空的，检查是不是上传了一个 0 字节的占位文件"


class CorruptFile(StudioError):
    code = "corrupt_file"
    status = 422
    hint = "PDF 可能已损坏或不是真正的 PDF（改扩展名不算）"


class EncryptedPdf(StudioError):
    code = "encrypted_pdf"
    status = 422
    hint = "带打开密码的 PDF 无法解析；请先另存为不加密的副本"


class NoTextLayer(StudioError):
    code = "no_text_layer"
    status = 422
    hint = "这是扫描件（图片型 PDF）。需要先做 OCR 生成文本层，本服务不含 OCR"


class DuplicateDoc(StudioError):
    code = "duplicate_doc"
    status = 409
    hint = "同一份文件已经上传过；如需重建索引请先删除原记录"

    def __init__(self, message: str, *, doc_id: str = "", hint: str = "") -> None:
        super().__init__(message, hint=hint)
        self.doc_id = doc_id


class DocNotFound(StudioError):
    code = "doc_not_found"
    status = 404
    hint = "刷新文档列表，确认 doc_id 是否存在"


class TooManyDocs(StudioError):
    code = "too_many_docs"
    status = 429
    hint = "先删除不用的文档；本机版本按单进程内存检索，不建议挂几百份"


class LlmUnavailable(StudioError):
    code = "llm_unavailable"
    status = 503
    hint = "画像生成依赖模型；可先用纯统计画像（不调模型，术语仍带出处）"
