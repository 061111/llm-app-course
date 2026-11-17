"""W2-04 · 测试与演示共用的假流对象。

真实 SDK 的 Stream 背后是一条活着的 HTTP 连接，测试里造不出来（也不该造）。
这里用最小实现顶替它：可迭代 + 有 close() + 记录自己有没有被关。

一个刻意的设计：pieces 列表里可以混入异常实例，表示「遍历到这一格就炸」。
于是「发三段然后断流」只需要写 [chunk("第一段"), TimeoutError("断了")]。

这个模块不产生任何输出，也没有演示入口。
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from types import SimpleNamespace
from typing import Any

from openai.types.chat import ChatCompletionChunk

MODEL = "gpt-4o-mini"


def make_chunk(content: str | None, finish_reason: str | None = None) -> ChatCompletionChunk:
    """造一个流式碎片。content=None 用于模拟结束块。"""
    return ChatCompletionChunk.model_validate({
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "delta": {"role": None, "content": content},
        }],
    })


def empty_chunk() -> ChatCompletionChunk:
    """流真正结束时发的那一块：choices 是空列表。"""
    return ChatCompletionChunk.model_validate({
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [],
    })


class FakeStream:
    """顶替 SDK 的 Stream：能 for、能 close、能被检查。"""

    def __init__(self, pieces: Iterable[Any]) -> None:
        self._pieces = list(pieces)
        self.closed = False
        self.iterated = 0

    def __iter__(self) -> Iterator[Any]:
        for piece in self._pieces:
            if isinstance(piece, BaseException):
                raise piece
            self.iterated += 1
            yield piece

    def close(self) -> None:
        self.closed = True


class FakeClient:
    """顶替 openai.OpenAI：只需要 client.chat.completions.create 这一条链。"""

    def __init__(self, pieces: Iterable[Any]) -> None:
        self.pieces = list(pieces)
        self.last_stream: FakeStream | None = None
        self.create_calls = 0
        self.last_kwargs: dict[str, Any] = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> FakeStream:
        self.create_calls += 1
        self.last_kwargs = kwargs
        if kwargs.get("stream") is not True:
            raise AssertionError("这个假 client 只服务流式路径")
        self.last_stream = FakeStream(self.pieces)
        return self.last_stream
