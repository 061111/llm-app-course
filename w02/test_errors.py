"""W2-02 的测试：异常分类 + 重试行为。

测试的关键不是「跑通」，而是每一条都对应一个真实故障场景。
"""
from __future__ import annotations

import httpx
import openai
import pytest

from w02.errors import (
    FatalError,
    LLMError,
    RetryableError,
    backoff_delay,
    classify,
    retry_call,
)


def _sdk_error(cls: type[Exception], status: int) -> Exception:
    """构造一个真实的 openai SDK 异常对象（不需要 API key）。"""
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://api.example.com/v1/chat")
    )
    return cls(message="mock", response=response, body=None)


# ---------------- 分类 ----------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (TimeoutError("read timeout"), RetryableError),
        (ConnectionResetError("连接被重置"), RetryableError),
        (_sdk_error(openai.RateLimitError, 429), RetryableError),
        (_sdk_error(openai.InternalServerError, 500), RetryableError),
        (_sdk_error(openai.InternalServerError, 503), RetryableError),
        (openai.APITimeoutError(request=httpx.Request("POST", "https://x")), RetryableError),
        (_sdk_error(openai.AuthenticationError, 401), FatalError),
        (_sdk_error(openai.BadRequestError, 400), FatalError),
        (ValueError("完全陌生"), FatalError),
    ],
)
def test_异常分类(exc: Exception, expected: type[LLMError]) -> None:
    assert isinstance(classify(exc), expected)


def test_未知异常默认不可重试() -> None:
    """赌一把重试未知异常，可能重试一百次都是同一个程序 bug。"""
    assert isinstance(classify(RuntimeError("谁知道呢")), FatalError)


def test_已经是自己的异常就原样返回() -> None:
    err = FatalError("密钥无效", status=401)
    assert classify(err) is err


def test_分类保留状态码() -> None:
    assert classify(_sdk_error(openai.RateLimitError, 429)).status == 429


# ---------------- 重试 ----------------


def test_一次就成功只调用一次() -> None:
    calls = []

    def ok() -> str:
        calls.append(1)
        return "ok"

    assert retry_call(ok, sleep=lambda s: None) == "ok"
    assert len(calls) == 1


def test_前两次失败第三次成功() -> None:
    calls = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("timeout")
        return "ok"

    assert retry_call(flaky, sleep=lambda s: None, jitter=0) == "ok"
    assert len(calls) == 3


def test_不可重试的立刻失败且只调用一次() -> None:
    calls = []

    def unauthorized() -> str:
        calls.append(1)
        raise _sdk_error(openai.AuthenticationError, 401)

    with pytest.raises(FatalError):
        retry_call(unauthorized, sleep=lambda s: None)
    assert len(calls) == 1


def test_耗尽次数后抛可重试异常() -> None:
    calls = []

    def always_503() -> str:
        calls.append(1)
        raise _sdk_error(openai.InternalServerError, 503)

    with pytest.raises(RetryableError):
        retry_call(always_503, max_attempts=3, sleep=lambda s: None, jitter=0)
    assert len(calls) == 3


def test_原始异常挂在_cause_上() -> None:
    original = _sdk_error(openai.InternalServerError, 503)

    def boom() -> str:
        raise original

    with pytest.raises(RetryableError) as info:
        retry_call(boom, max_attempts=1, sleep=lambda s: None)
    assert info.value.__cause__ is original


def test_退避时间按指数增长() -> None:
    slept: list[float] = []

    def always_timeout() -> str:
        raise TimeoutError("timeout")

    with pytest.raises(RetryableError):
        retry_call(
            always_timeout,
            max_attempts=4,
            base_delay=0.5,
            jitter=0,
            sleep=slept.append,
        )
    assert slept == [0.5, 1.0, 2.0]


def test_退避时间有上限() -> None:
    slept: list[float] = []

    def always_timeout() -> str:
        raise TimeoutError("timeout")

    with pytest.raises(RetryableError):
        retry_call(
            always_timeout,
            max_attempts=7,
            base_delay=0.5,
            max_delay=2.0,
            jitter=0,
            sleep=slept.append,
        )
    assert slept == [0.5, 1.0, 2.0, 2.0, 2.0, 2.0]


def test_抖动落在区间内() -> None:
    for attempt in (1, 2, 3):
        for _ in range(50):
            d = backoff_delay(attempt, base_delay=0.5, jitter=0.5)
            raw = 0.5 * 2 ** (attempt - 1)
            assert raw * 0.5 <= d <= raw * 1.5


def test_on_retry_回调收到正确次数() -> None:
    seen: list[tuple[int, str, float]] = []

    def always_timeout() -> str:
        raise TimeoutError("timeout")

    with pytest.raises(RetryableError):
        retry_call(
            always_timeout,
            max_attempts=3,
            jitter=0,
            sleep=lambda s: None,
            on_retry=lambda a, e, d: seen.append((a, type(e).__name__, d)),
        )
    assert [s[0] for s in seen] == [1, 2]
    assert all(s[1] == "RetryableError" for s in seen)


def test_max_attempts_非法直接报错() -> None:
    with pytest.raises(ValueError):
        retry_call(lambda: "x", max_attempts=0)


def test_业务层只需要一个_except() -> None:
    """这是整个设计的验收标准：上层永远只写 `except LLMError`。"""

    def boom() -> str:
        raise _sdk_error(openai.BadRequestError, 400)

    try:
        retry_call(boom, sleep=lambda s: None)
    except LLMError as e:  # 不关心是哪种 SDK 异常
        assert isinstance(e, FatalError)
    else:
        pytest.fail("应该抛 LLMError")
