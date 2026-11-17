"""llmkit —— 项目 A 的模型调用层。

外部只需记四个名字：

    from llmkit import ask, ask_structured, stream_ask, Settings

    result = ask("用一句话解释贝叶斯定理")
    print(result.text, result.usage.total_tokens, result.latency_ms)

    class Answer(BaseModel):
        answer: str
        confidence: float

    obj = ask_structured("…", Answer)

    for piece in stream_ask("…"):
        print(piece, end="")

设计原则：这个包之外，任何人都不需要 import openai。
换供应商只要改 LLMKIT_BASE_URL / LLMKIT_MODEL 两个环境变量。
"""
from __future__ import annotations

from llmkit.client import ask, ask_structured, collect, collect_partial, stream_ask
from llmkit.config import Settings
from llmkit.errors import (
    ConfigError,
    FatalError,
    LLMError,
    LLMParseError,
    RetryableError,
    classify,
)
from llmkit.obs import setup_logging
from llmkit.types import AskOptions, AskResult, Citation, Usage

__all__ = [
    "ask",
    "ask_structured",
    "stream_ask",
    "collect",
    "collect_partial",
    "Settings",
    "AskOptions",
    "AskResult",
    "Citation",
    "Usage",
    "LLMError",
    "RetryableError",
    "FatalError",
    "ConfigError",
    "LLMParseError",
    "classify",
    "setup_logging",
]

__version__ = "0.1.0"
