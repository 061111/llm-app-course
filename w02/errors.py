"""W2-02 · 异常体系与重试。

一句话：把 SDK 那十几种异常收敛成「能重试」和「不能重试」两类，
       再给能重试的那一类套一个带指数退避的重试环。

为什么需要这一层：
  1. 业务代码不该 import openai / anthropic / 任何 SDK 的异常类——换供应商就全崩；
  2. 「重试」不是无脑循环，401 重试三次只是浪费三次时间；
  3. 重试要能测试，所以 sleep 必须可注入。

运行演示：..\\.venv\\Scripts\\python.exe -m w02.errors
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class LLMError(Exception):
    """本项目所有 LLM 相关异常的基类。

    业务代码只 except 这一层，永远不 import 第三方 SDK 的异常类。
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status

    def __str__(self) -> str:
        return self.message if self.status is None else f"[{self.status}] {self.message}"


class RetryableError(LLMError):
    """瞬时故障：等一会儿再试可能就成功（超时、连接断、429 限流、5xx）。"""


class FatalError(LLMError):
    """确定性故障：重试一百次也一样失败（401 密钥错、400 参数错、格式不符）。"""


# 标准库层面的瞬时故障
_RETRYABLE_BUILTIN = (TimeoutError, ConnectionError)

# SDK 里不带 status_code 的连接层异常，只能按类型名兜底
_RETRYABLE_TYPE_NAMES = frozenset(
    {"APITimeoutError", "APIConnectionError", "APIConnectionTimeoutError"}
)


def classify(exc: BaseException) -> LLMError:
    """把任意异常翻译成 RetryableError 或 FatalError。

    判断顺序很重要，别乱调：
      1. 已经是我们自己的异常 → 原样返回；
      2. 标准库超时/连接 → 可重试；
      3. 429 → 可重试；5xx → 可重试；4xx（除 429）→ 不可重试；
      4. 没有 status_code 的 SDK 连接异常 → 按类型名判可重试；
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
    """第 attempt 次失败后该等多久：指数增长 + 上下浮动。

    attempt=1 → 0.5s 上下浮动 50% → [0.25, 0.75]
    attempt=2 → 1.0s 上下浮动 50% → [0.5, 1.5]
    attempt=3 → 2.0s 上下浮动 50% → [1.0, 3.0]
    """
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


def _demo() -> None:
    def fake_sleep(seconds: float) -> None:
        print(f"      …… 实际会睡 {seconds:.2f}s（演示里不睡）")

    def log(attempt: int, err: LLMError, delay: float) -> None:
        print(f"      第 {attempt} 次失败：{err} → 退避 {delay:.2f}s")

    print("场景 1：前两次超时，第三次成功")
    state = {"n": 0}

    def flaky() -> str:
        state["n"] += 1
        if state["n"] < 3:
            raise TimeoutError("read timeout=30.0")
        return "第三次成功"

    print("   结果：", retry_call(flaky, sleep=fake_sleep, jitter=0, on_retry=log))
    print("   总调用次数：", state["n"])

    print("\n场景 2：401，不该重试")
    def unauthorized() -> str:
        raise FatalError("密钥无效", status=401)

    try:
        retry_call(unauthorized, sleep=fake_sleep, jitter=0, on_retry=log)
    except LLMError as e:
        print("   结果：直接失败 →", e)

    print("\n场景 3：三次都是 503，耗尽重试")
    state2 = {"n": 0}

    def always_503() -> str:
        state2["n"] += 1
        raise RetryableError("服务端故障 503，稍后重试", status=503)

    try:
        retry_call(always_503, sleep=fake_sleep, jitter=0, on_retry=log)
    except LLMError as e:
        print("   结果：", type(e).__name__, "→", e)
        print("   总调用次数：", state2["n"])


if __name__ == "__main__":
    _demo()
