"""M1-T1.3 验收：调用层。

这一组全部离线跑，靠 `FakeClient` 编排剧本。四个关键断言：
  1. 一次成功时，真的只调用一次；
  2. 401 立刻失败，**不重试**；
  3. 超时重试到上限，抛 RetryableError，抛的不是裸 SDK 异常；
  4. sleep 被注入后，重试路径零耗时。
"""
from __future__ import annotations

import httpx
import openai
import pytest

from llmkit.client import ask, ask_structured
from llmkit.config import Settings
from llmkit.errors import FatalError, LLMError, RetryableError
from llmkit.tests.fakes import FakeClient, make_completion
from llmkit.types import AskOptions

SETTINGS = Settings(api_key="sk-test", model="fake-model", price_in_cny_per_mtok=2.0,
                    price_out_cny_per_mtok=8.0)


def sdk_error(cls, status: int):
    """造一个"看起来像真的"SDK 异常。不需要联网。"""
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.example.com"))
    return cls(message="mock", response=response, body=None)


def no_sleep(_seconds: float) -> None:
    """注入用的假 sleep：什么都不做，让重试路径零耗时。"""


def test_一次成功只调用一次():
    client = FakeClient([make_completion("你好，我是答案")])
    result = ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)

    assert result.text == "你好，我是答案"
    assert result.attempts == 1
    assert client.call_count == 1


def test_返回值带耗时与_trace_id():
    client = FakeClient([make_completion("x")])
    result = ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)

    assert result.latency_ms >= 0
    assert len(result.trace_id) == 12


def test_请求参数真的传到了_create():
    client = FakeClient([make_completion("x")])
    ask(
        "问题",
        options=AskOptions(temperature=0.7, max_tokens=256, system="你是助教"),
        settings=SETTINGS,
        client=client,
        sleep=no_sleep,
    )

    sent = client.calls[0]
    assert sent["model"] == "fake-model"
    assert sent["temperature"] == 0.7
    assert sent["max_tokens"] == 256
    assert sent["messages"] == [
        {"role": "system", "content": "你是助教"},
        {"role": "user", "content": "问题"},
    ]


def test_system_为空时不放_system_消息():
    client = FakeClient([make_completion("x")])
    ask("问题", options=AskOptions(system=""), settings=SETTINGS, client=client, sleep=no_sleep)
    assert client.calls[0]["messages"] == [{"role": "user", "content": "问题"}]


def test_401_立刻失败只调用一次():
    client = FakeClient([sdk_error(openai.AuthenticationError, 401)])

    with pytest.raises(FatalError) as info:
        ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)

    assert "401" in str(info.value)
    assert client.call_count == 1          # 关键：没有重试


def test_超时重试到上限后抛_RetryableError():
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://api.example.com"))
    client = FakeClient([timeout, timeout, timeout])

    with pytest.raises(RetryableError):
        ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)

    assert client.call_count == 3


def test_前两次超时第三次成功():
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://api.example.com"))
    client = FakeClient([timeout, timeout, make_completion("终于成功")])

    result = ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)

    assert result.text == "终于成功"
    assert result.attempts == 3
    assert client.call_count == 3


def test_max_attempts_可配置():
    timeout = TimeoutError("read timeout")
    client = FakeClient([timeout] * 4)

    with pytest.raises(RetryableError):
        ask("问题", options=AskOptions(max_attempts=4), settings=SETTINGS,
            client=client, sleep=no_sleep)

    assert client.call_count == 4


def test_抛出的异常不会是裸_SDK_异常():
    client = FakeClient([sdk_error(openai.InternalServerError, 503)] * 3)

    try:
        ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)
    except BaseException as exc:
        assert isinstance(exc, LLMError)          # 一定是我们自己的类型
        assert not isinstance(exc, openai.OpenAIError)
        assert exc.__cause__ is not None          # 原始异常还挂在 __cause__ 上
    else:
        pytest.fail("应该抛异常")


def test_成本按_token_算():
    client = FakeClient([make_completion("x", prompt_tokens=1000, completion_tokens=500)])
    result = ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)

    # 1000/1M*2 + 500/1M*8 = 0.002 + 0.004
    assert result.usage.cost_cny == pytest.approx(0.006)
    assert result.usage.total_tokens == 1500


def test_服务端不返回_usage_时记_0_而不是崩():
    client = FakeClient([make_completion("x", with_usage=False)])
    result = ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)
    assert result.usage.prompt_tokens == 0
    assert result.usage.cost_cny == 0.0


def test_空_choices_抛_FatalError():
    from openai.types.chat import ChatCompletion

    empty = ChatCompletion.model_validate(
        {"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": []}
    )
    client = FakeClient([empty])

    with pytest.raises(FatalError, match="空 choices"):
        ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)


def test_空正文会重试():
    """空正文不是"答得短"，是没答：必须重发，不能把空串当答案交给下游。

    实测来源（2026-11-18）：deepseek-flash 把 max_tokens 全花在 reasoning_tokens 上时，
    content 是空串、finish_reason=length。旧实现直接返回 ""，
    评估里表现为"回答为空"，用户看到一片空白。
    """
    client = FakeClient([make_completion(""), make_completion("第二次答上了")])
    result = ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)

    assert result.text == "第二次答上了"
    assert result.attempts == 2
    assert client.call_count == 2


def test_一直空正文就如实报错():
    client = FakeClient([make_completion(""), make_completion(""), make_completion("")])

    with pytest.raises(RetryableError, match="空正文"):
        ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)
    assert client.call_count == 3


def test_ask_structured_第一次就合法():
    from pydantic import BaseModel

    class Point(BaseModel):
        x: int
        y: int

    client = FakeClient([make_completion('{"x": 1, "y": 2}')])
    point = ask_structured("给个坐标", Point, settings=SETTINGS, client=client, sleep=no_sleep)

    assert isinstance(point, Point)
    assert (point.x, point.y) == (1, 2)
    assert client.call_count == 1


def test_ask_structured_第二次才合法():
    from pydantic import BaseModel

    class Point(BaseModel):
        x: int
        y: int

    client = FakeClient([make_completion('{"x": 1}'), make_completion('{"x": 1, "y": 2}')])
    point = ask_structured("给个坐标", Point, settings=SETTINGS, client=client, sleep=no_sleep)

    assert point.y == 2
    assert client.call_count == 2

    # 第二次请求的 prompt 里必须真的带上"上次错在哪"，否则模型没有修正依据
    repair_prompt_text = client.calls[1]["messages"][-1]["content"]
    assert "只输出一个 JSON" in repair_prompt_text
    assert "错误信息" in repair_prompt_text
