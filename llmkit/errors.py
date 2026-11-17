"""M1 · 异常层：从 w02/errors.py 搬过来，并补两个 M1 才需要的异常。

跟 w02 版本的两处差别（其余逐字一致）：
  1. 新增 `ConfigError` / `LLMParseError`，都继承 `FatalError`——
     它们确实"重试无意义"：key 没配好，重试一万次也还是没配好。
  2. 去掉了 `_demo()`，演示留在 w02 的讲义里，产品代码不放演示。

为什么不直接 `from w02.errors import ...`：
  `w02/` 是教材目录，会被整目录删改；`llmkit/` 是要被 import 的产品包。
  产品依赖教材，等于把课堂笔记当运行时依赖。
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class LLMError(Exception):
    """本项目所有 LLM 相关异常的基类。业务代码只 except 这一层。"""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status

    def __str__(self) -> str:
        return self.message if self.status is None else f"[{self.status}] {self.message}"


class RetryableError(LLMError):
    """瞬时故障：等一会儿再试可能就成功（超时、连接断、429、5xx）。"""


class FatalError(LLMError):
    """确定性故障：重试一百次也一样失败（401、400、格式不符）。"""


class ConfigError(FatalError):
    """配置缺失或非法。发生在调用之前，不是"调用失败"。

    继承 FatalError 的理由：上层写 `except LLMError` 能一并兜住，
    写 `except RetryableError` 不会误捕它。
    """


class LLMParseError(FatalError):
    """模型返回的内容无法解析成目标结构。

    带 `raw` 字段保存原始输出——排查"模型到底吐了什么"时这是唯一线索。
    """

    def __init__(self, message: str, *, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


_RETRYABLE_BUILTIN = (TimeoutError, ConnectionError)

# SDK 里不带 status_code 的连接层异常，只能按类型名兜底
_RETRYABLE_TYPE_NAMES = frozenset(
    {"APITimeoutError", "APIConnectionError", "APIConnectionTimeoutError"}
)


def classify(exc: BaseException) -> LLMError:
    """把任意异常翻译成 RetryableError 或 FatalError。

    判断顺序（别乱调，顺序本身是逻辑）：
      1. 已经是我们自己的异常 → 原样返回；
      2. 标准库超时/连接 → 可重试；
      3. 429 → 可重试；5xx → 可重试；4xx（除 429）→ 不可重试；
      4. 没有 status_code 的 SDK 连接异常 → 按类型名判；
      5. 其余未知异常 → 一律不可重试（fail fast，别赌）。
    """
    if isinstance(exc, LLMError):
        return exc

    if isinstance(exc, _RETRYABLE_BUILTIN):
        return RetryableError(f"连接层失败：{type(exc).__name__}: {exc}")

    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)

    if status == 429:
        return RetryableError("被限流，稍后重试", status=429)
    if isinstance(status, int) and 500 <= status < 600:
        return RetryableError(f"服务端故障 {status}，稍后重试", status=status)
    if isinstance(status, int) and 400 <= status < 500:
        return FatalError(f"请求本身有问题 {status}，重试无意义", status=status)

    if type(exc).__name__ in _RETRYABLE_TYPE_NAMES:
        return RetryableError(f"连接层失败：{type(exc).__name__}")

    return FatalError(f"未知异常，按不可重试处理：{type(exc).__name__}: {exc}")


def backoff_delay(
    attempt: int,
    *,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.5,
) -> float:
    """第 attempt 次失败后该等多久：指数增长 + 上下浮动。"""
    raw = min(base_delay * (2 ** (attempt - 1)), max_delay)
    return raw * (1 + random.uniform(-jitter, jitter))


def retry_call(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, LLMError, float], None] | None = None,
) -> T:
    """执行 fn，遇到可重试错误就退避后重来，最多 max_attempts 次。

    抛出的一定是 LLMError 的子类（原始异常挂在 __cause__ 上）。
    """
    if max_attempts < 1:
        raise ValueError("max_attempts 至少为 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except BaseException as exc:  # 边界层就是要兜住一切，逐层翻译
            err = classify(exc)
            if not isinstance(err, RetryableError) or attempt >= max_attempts:
                raise err from exc
            delay = backoff_delay(
                attempt, base_delay=base_delay, max_delay=max_delay, jitter=jitter
            )
            if on_retry is not None:
                on_retry(attempt, err, delay)
            sleep(delay)

    raise AssertionError("循环一定会在里面 return 或 raise，走不到这里")
