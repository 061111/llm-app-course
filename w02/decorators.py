"""W2-03 · 装饰器：把重试、计时从业务函数里搬出去。

一句话：装饰器 = 接收函数、返回新函数的函数；`@` 只是语法糖。

为什么需要这一层：
  1. `retry_call(fn)` 要求调用点现场造 lambda，20 个调用点就有 20 个 lambda；
  2. 重试 / 计时 / 打日志属于「横切关注点」，写进业务函数会把主逻辑淹掉；
  3. 包一层 wrapper 会丢掉 __name__ / __doc__ / 签名，必须用 functools.wraps 补回来。

运行演示：..\\.venv\\Scripts\\python.exe -m w02.decorators
"""
from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from w02.errors import LLMError, retry_call

T = TypeVar("T")

logger = logging.getLogger(__name__)


def retry(
    fn: Callable[..., T] | None = None,
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, LLMError, float], None] | None = None,
):
    """把 `retry_call` 包成装饰器，两种写法都支持。

        @retry                          # 用法一：一个括号都不写
        def f() -> str: ...

        @retry(max_attempts=5)          # 用法二：先配参数，再贴函数
        def g() -> str: ...

    参数含义与 `w02.errors.retry_call` 完全一致，这里只是原样往下传。
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_call(
                lambda: func(*args, **kwargs),
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter=jitter,
                sleep=sleep,
                on_retry=on_retry,
            )

        return wrapper

    if fn is None:          # @retry(...) 形式：函数还没来，先把 decorator 交出去
        return decorator
    return decorator(fn)    # @retry 形式：函数已经到了，直接包


def timed(
    fn: Callable[..., T] | None = None,
    *,
    timer: Callable[[], float] = time.perf_counter,
    sink: Callable[[str, float], None] | None = None,
):
    """记录函数耗时，成功和失败都记。

    `timer` 和 `sink` 都是可注入的：测试里换掉它们，就能在不真等的前提下
    断言「耗时被记录了几次、每次多少」。
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            started = timer()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = timer() - started
                if sink is not None:
                    sink(func.__name__, elapsed)
                else:
                    logger.info("%s 耗时 %.3fs", func.__name__, elapsed)

        return wrapper

    if fn is None:
        return decorator
    return decorator(fn)


def _demo() -> None:
    def fake_sleep(seconds: float) -> None:
        print(f"      …… 实际会睡 {seconds:.2f}s（演示里不睡）")

    print("① @retry 不带任何参数")
    calls: list[str] = []

    @retry(sleep=fake_sleep)
    def fetch(text: str) -> str:
        calls.append(text)
        if len(calls) < 3:
            raise TimeoutError("mock timeout")
        return f"摘要：{text}"

    print("      返回:", fetch("原文"))
    print("      调用次数:", len(calls))
    print("      __name__:", fetch.__name__)

    print()
    print("② @retry(max_attempts=5)：次数可配")
    attempts: list[int] = []

    @retry(max_attempts=5, sleep=lambda seconds: None)
    def always_timeout() -> None:
        attempts.append(1)
        raise TimeoutError("mock timeout")

    try:
        always_timeout()
    except LLMError as exc:
        print(f"      共尝试 {len(attempts)} 次后放弃，抛出 {type(exc).__name__}")

    print()
    print("③ @timed：记录单次耗时")
    records: list[tuple[str, float]] = []
    ticks = iter([0.0, 0.25])

    @timed(timer=lambda: next(ticks), sink=lambda name, s: records.append((name, s)))
    def write_doc() -> str:
        return "ok"

    write_doc()
    print(f"      记录到 {records}")

    print()
    print("④ 叠加顺序：谁在外层，谁看到的是总耗时")

    def make_sink(bucket: list[tuple[str, float]]) -> Callable[[str, float], None]:
        def sink(name: str, elapsed: float) -> None:
            bucket.append((name, round(elapsed, 3)))

        return sink

    print("      A. @timed 在外层 → 记 1 条，包含全部重试")
    bucket_a: list[tuple[str, float]] = []
    state_a = {"calls": 0}

    @timed(sink=make_sink(bucket_a))
    @retry(sleep=lambda seconds: time.sleep(0.02))
    def task_a() -> str:
        state_a["calls"] += 1
        if state_a["calls"] < 3:
            raise TimeoutError("mock timeout")
        return "ok"

    task_a()
    print("         ", bucket_a)

    print("      B. @retry 在外层 → 记 3 条，各是单次尝试")
    bucket_b: list[tuple[str, float]] = []
    state_b = {"calls": 0}

    @retry(sleep=lambda seconds: time.sleep(0.02))
    @timed(sink=make_sink(bucket_b))
    def task_b() -> str:
        state_b["calls"] += 1
        if state_b["calls"] < 3:
            raise TimeoutError("mock timeout")
        return "ok"

    task_b()
    print("         ", bucket_b)


if __name__ == "__main__":
    _demo()
