"""测试与演示共用的假客户端。

核心思路：openai SDK 的 `ChatCompletion` / `ChatCompletionChunk` 都是 Pydantic 模型，
可以 `model_validate({...})` 从字典直接造出来。所以造一个"看起来像真的"响应
完全不需要联网、不需要 key。

`FakeCompletions` 的剧本元素可以是：
  - 一个 ChatCompletion / 流对象 → 正常返回
  - 一个异常实例               → 抛出它
这样"前两次超时第三次成功"这种场景就是三行数据。
"""
from __future__ import annotations

from types import SimpleNamespace

from openai.types.chat import ChatCompletion, ChatCompletionChunk


def make_completion(
    text: str = "示例答案",
    *,
    model: str = "fake-model",
    prompt_tokens: int = 12,
    completion_tokens: int = 8,
    with_usage: bool = True,
) -> ChatCompletion:
    payload: dict = {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
    }
    if with_usage:
        payload["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    return ChatCompletion.model_validate(payload)


def make_chunk(content: str | None, finish: str | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "delta": {"role": None, "content": content},
                }
            ],
        }
    )


def empty_chunk() -> ChatCompletionChunk:
    """真实 API 在流末尾会发一个 choices 为空数组的块。"""
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "fake-model",
            "choices": [],
        }
    )


class FakeStream:
    """可迭代 + 有 close() + 记录是否被关。元素里的异常会在遍历时抛出。"""

    def __init__(self, items: list) -> None:
        self._items = list(items)
        self.closed = False

    def __iter__(self):
        for item in self._items:
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self) -> None:
        self.closed = True


class FakeCompletions:
    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("剧本已用完，但还有人继续调用 create")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    """够用就好：只要暴露 chat.completions.create 这一个入口。"""

    def __init__(self, script: list) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(script))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls

    @property
    def call_count(self) -> int:
        return len(self.chat.completions.calls)
