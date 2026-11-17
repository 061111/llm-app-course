"""M1-T1.5 验收：流式。

这组的重点不是"能不能拼出整段"，而是三条边界：
  1. 空 choices 末块、content=None 的结束块，必须跳过而不是崩；
  2. 中途断流，已收到的部分要留下来（collect_partial）；
  3. 无论怎么结束，流的 close() 都要被调用。

还有一条来自 04 的结论：**建流失败会重试，遍历失败不会**。
"""
from __future__ import annotations

import pytest

from llmkit.client import collect, collect_partial, stream_ask
from llmkit.config import Settings
from llmkit.errors import ConfigError, RetryableError
from llmkit.tests.fakes import FakeClient, FakeStream, empty_chunk, make_chunk

SETTINGS = Settings(api_key="sk-test", model="fake-model")


def no_sleep(_seconds: float) -> None:
    pass


def test_正常流拼成整段():
    stream = FakeStream([make_chunk("贝叶"), make_chunk("斯"), make_chunk(None, "stop"), empty_chunk()])
    client = FakeClient([stream])

    pieces = stream_ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)
    assert collect(pieces) == "贝叶斯"


def test_跳过空_choices_与_None_内容():
    stream = FakeStream([make_chunk("a"), empty_chunk(), make_chunk(None), empty_chunk(), make_chunk("b")])
    client = FakeClient([stream])

    pieces = list(stream_ask("问题", settings=SETTINGS, client=client, sleep=no_sleep))
    assert pieces == ["a", "b"]


def test_空串片段也被跳过():
    stream = FakeStream([make_chunk("a"), make_chunk(""), make_chunk("b")])
    client = FakeClient([stream])

    pieces = list(stream_ask("问题", settings=SETTINGS, client=client, sleep=no_sleep))
    assert pieces == ["a", "b"]


def test_流式请求带_stream_True():
    client = FakeClient([FakeStream([make_chunk("x")])])
    list(stream_ask("问题", settings=SETTINGS, client=client, sleep=no_sleep))

    assert client.calls[0]["stream"] is True


def test_建流阶段失败会重试():
    timeout = TimeoutError("建立连接超时")
    client = FakeClient([timeout, timeout, FakeStream([make_chunk("终于连上")])])

    pieces = stream_ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)
    assert collect(pieces) == "终于连上"
    assert client.call_count == 3


def test_建流一直失败就抛_RetryableError():
    client = FakeClient([TimeoutError("x"), TimeoutError("x"), TimeoutError("x")])

    with pytest.raises(RetryableError):
        stream_ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)

    assert client.call_count == 3


def test_遍历中断时保留已收到的部分():
    stream = FakeStream([make_chunk("前半"), TimeoutError("流断了")])
    client = FakeClient([stream])

    pieces = stream_ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)
    text, error = collect_partial(pieces)

    assert text == "前半"
    assert isinstance(error, TimeoutError)
    assert client.call_count == 1          # 遍历阶段失败不重试


def test_正常结束后流被关闭():
    stream = FakeStream([make_chunk("a")])
    client = FakeClient([stream])

    list(stream_ask("问题", settings=SETTINGS, client=client, sleep=no_sleep))
    assert stream.closed is True


def test_中断后流也被关闭():
    stream = FakeStream([make_chunk("a"), TimeoutError("断")])
    client = FakeClient([stream])

    text, error = collect_partial(stream_ask("问题", settings=SETTINGS, client=client, sleep=no_sleep))
    assert text == "a"
    assert stream.closed is True


def test_提前_break_后流被关闭():
    stream = FakeStream([make_chunk("a"), make_chunk("b"), make_chunk("c")])
    client = FakeClient([stream])

    pieces = stream_ask("问题", settings=SETTINGS, client=client, sleep=no_sleep)
    for piece in pieces:
        assert piece == "a"
        break
    # 生成器被回收/关闭时 finally 才执行，这里显式关闭以保证确定性
    pieces.close()

    assert stream.closed is True


def test_配置缺失在调用时就报错而不是第一次_next(monkeypatch, tmp_path):
    # stream_ask 不是生成器函数（内部没有 yield），所以配置检查是同步发生的。
    #
    # 注意：不能直接让 stream_ask 自己去读 .env —— 那会让这条测试依赖"你本机
    # .env 里恰好没填密钥"。填上真密钥后它就绿转红（2026-11-17 实际踩到）。
    # 所以这里把 from_env 的输入钉死成"空环境变量 + 不存在的文件"，
    # 被测的仍是真实的 from_env 逻辑，只是输入完全确定。
    #
    # 打桩对象是**类**的 classmethod，不是某个实例 —— 必须先把原函数抓在手里，
    # 否则假函数调 Settings.from_env 就是调自己，直接 RecursionError。
    real_from_env = Settings.from_env

    def only_empty_env(*_args, **_kwargs):
        return real_from_env(env={}, env_file=tmp_path / "no_such.env")

    monkeypatch.setattr("llmkit.client.Settings.from_env", only_empty_env)

    with pytest.raises(ConfigError):
        stream_ask("问题", settings=None, client=None)
