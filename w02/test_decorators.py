"""W2-03 · 装饰器的测试。

12 条用例，四组：
  写法组（4）—— @retry / @retry(...) / @timed / @timed(...) 四种写法都能用
  透传组（3）—— 参数、返回值、不可重试错误
  元信息组（2）—— __name__ / __doc__ / 签名有没有被 wrapper 盖掉
  叠加组（3）—— 两个装饰器谁在外层，计时结果差在哪
"""
from __future__ import annotations

import inspect
import logging

import httpx
import openai
import pytest

from w02.decorators import retry, timed
from w02.errors import FatalError, LLMError, RetryableError


def _sdk_error(cls, status):
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://api.example.com")
    )
    return cls(message="mock", response=response, body=None)


def no_sleep(_seconds: float) -> None:
    """替代 time.sleep：立刻返回，整套测试零耗时。"""


# ---------------------------------------------------------------- 写法组


def test_裸_retry_能用() -> None:
    """@retry 不带括号，命中 fn 不是 None 的那条分支。"""

    @retry
    def work() -> int:
        return 7

    assert work() == 7


def test_带参数的_retry_次数生效() -> None:
    """@retry(max_attempts=5) 命中 fn 是 None 的那条分支。"""
    calls: list[int] = []

    @retry(max_attempts=5, sleep=no_sleep)
    def always_timeout() -> None:
        calls.append(1)
        raise TimeoutError("mock timeout")

    with pytest.raises(RetryableError):
        always_timeout()
    assert len(calls) == 5


def test_裸_timed_能用() -> None:
    @timed
    def work() -> int:
        return 42

    assert work() == 42


def test_带参数的_timed_能用() -> None:
    ticks = iter([0.0, 0.25])
    records: list[tuple[str, float]] = []

    @timed(timer=lambda: next(ticks), sink=lambda name, s: records.append((name, s)))
    def work() -> str:
        return "ok"

    assert work() == "ok"
    assert records == [("work", 0.25)]


# ---------------------------------------------------------------- 透传组


def test_位置参数和关键字参数都透传() -> None:
    seen: dict[str, int] = {}

    @retry(sleep=no_sleep)
    def collect(a, b=2, *, c=3):
        seen.update(a=a, b=b, c=c)
        return a + b + c

    assert collect(1, b=20, c=300) == 321
    assert seen == {"a": 1, "b": 20, "c": 300}


def test_重试后第三次成功返回正确值() -> None:
    calls: list[int] = []

    @retry(sleep=no_sleep)
    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("mock timeout")
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 3


def test_不可重试的错误只调一次() -> None:
    calls: list[int] = []

    @retry(sleep=no_sleep)
    def bad_key() -> None:
        calls.append(1)
        raise _sdk_error(openai.AuthenticationError, 401)

    with pytest.raises(FatalError):
        bad_key()
    assert len(calls) == 1


# ---------------------------------------------------------------- 元信息组


def test_wraps_保住名字和文档() -> None:
    @retry(sleep=no_sleep)
    def fetch(text: str) -> str:
        """把文本写成摘要。"""
        return text

    assert fetch.__name__ == "fetch"
    assert fetch.__doc__ == "把文本写成摘要。"


def test_没有_wraps_签名就丢了() -> None:
    """反例：手写 wrapper 不套 functools.wraps，签名会退化成 (*args, **kwargs)。"""
    import functools

    def without_wraps(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def with_wraps(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def original(text: str, limit: int = 10) -> str:
        return text * limit

    assert list(inspect.signature(original).parameters) == ["text", "limit"]
    assert list(inspect.signature(without_wraps(original)).parameters) == [
        "args",
        "kwargs",
    ]
    assert list(inspect.signature(with_wraps(original)).parameters) == ["text", "limit"]


# ---------------------------------------------------------------- 叠加组


def test_timed_异常时也记录() -> None:
    ticks = iter([0.0, 0.5])
    records: list[tuple[str, float]] = []

    @timed(timer=lambda: next(ticks), sink=lambda name, s: records.append((name, s)))
    def boom() -> None:
        raise ValueError("故意失败")

    with pytest.raises(ValueError):
        boom()
    assert records == [("boom", 0.5)]


def test_timed_在外层只记一条总耗时() -> None:
    ticks = iter([0.0, 3.0])
    records: list[tuple[str, float]] = []
    calls: list[int] = []

    @timed(timer=lambda: next(ticks), sink=lambda name, s: records.append((name, s)))
    @retry(sleep=no_sleep)
    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("mock timeout")
        return "ok"

    assert flaky() == "ok"
    assert len(records) == 1
    assert records[0] == ("flaky", 3.0)


def test_retry_在外层每次尝试各记一条() -> None:
    ticks = iter([0.0, 0.1, 1.0, 1.1, 2.0, 2.1])
    records: list[tuple[str, float]] = []
    calls: list[int] = []

    @retry(sleep=no_sleep)
    @timed(timer=lambda: next(ticks), sink=lambda name, s: records.append((name, s)))
    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("mock timeout")
        return "ok"

    assert flaky() == "ok"
    assert len(records) == 3
    assert [round(seconds, 6) for _, seconds in records] == [0.1, 0.1, 0.1]
