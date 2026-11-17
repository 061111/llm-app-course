"""W2-04 · streaming 的测试。

16 条，四组：切分 5、惰性 2、上下文 5、半截内容 4。

全部不联网、不真等、不花钱：FakeClient 顶替 SDK 的客户端，
FakeStream 顶替那条 HTTP 连接，并且会记录自己有没有被 close()。

注意这里一律用 FakeStream(...) 而不是 iter([...])：只有 FakeStream 知道
「pieces 列表里混进来一个异常实例，意思是遍历到这一格就抛」。
"""
from __future__ import annotations

import pytest

from w02.fakes import FakeClient, FakeStream, empty_chunk, make_chunk
from w02.streaming import collect, collect_partial, iter_text, open_text_stream


def full_pieces() -> list:
    """一条完整的流：三个真片段 + 一个结束块 + 一个空 choices 末块。"""
    return [
        make_chunk("大模型"),
        make_chunk("应用"),
        make_chunk("开发"),
        make_chunk(None, finish_reason="stop"),
        empty_chunk(),
    ]


def test_正常流拼成整段() -> None:
    assert collect(FakeStream(full_pieces())) == "大模型应用开发"


def test_空choices的末块被跳过() -> None:
    assert collect(FakeStream([empty_chunk()])) == ""


def test_content为None的块被跳过() -> None:
    pieces = [make_chunk("前半"), make_chunk(None, finish_reason="stop")]
    assert collect(FakeStream(pieces)) == "前半"


def test_空字符串片段不被产出() -> None:
    pieces = [make_chunk(""), make_chunk("有内容"), make_chunk("")]
    assert list(iter_text(FakeStream(pieces))) == ["有内容"]


def test_纯空白片段会被保留() -> None:
    pieces = [make_chunk("a"), make_chunk(" "), make_chunk("b")]
    assert list(iter_text(FakeStream(pieces))) == ["a", " ", "b"]


def test_iter_text_调用时不消费流() -> None:
    client = FakeClient(full_pieces())
    with open_text_stream(client, "原文") as stream:
        gen = iter_text(stream)
        assert stream.iterated == 0
        next(gen)
        assert stream.iterated == 1


def test_生成器遍历一次就空() -> None:
    gen = iter_text(FakeStream(full_pieces()))
    assert "".join(gen) == "大模型应用开发"
    assert list(gen) == []


def test_正常结束连接被关闭() -> None:
    client = FakeClient(full_pieces())
    with open_text_stream(client, "原文") as stream:
        collect(stream)
    assert client.last_stream is not None
    assert client.last_stream.closed is True


def test_提前break连接也关闭() -> None:
    client = FakeClient(full_pieces())
    with open_text_stream(client, "原文") as stream:
        for piece in iter_text(stream):
            assert piece == "大模型"
            break
    assert client.last_stream.closed is True


def test_中途异常连接也关闭() -> None:
    client = FakeClient([make_chunk("第一段"), TimeoutError("流断了")])
    with pytest.raises(TimeoutError):
        with open_text_stream(client, "原文") as stream:
            collect(stream)
    assert client.last_stream.closed is True


def test_不进入with不发请求() -> None:
    client = FakeClient(full_pieces())
    cm = open_text_stream(client, "原文")
    assert client.create_calls == 0
    with cm as stream:
        assert client.create_calls == 1
        collect(stream)


def test_请求参数带stream为True() -> None:
    client = FakeClient(full_pieces())
    with open_text_stream(client, "原文", model="gpt-4o-mini") as stream:
        collect(stream)
    assert client.last_kwargs["stream"] is True
    assert client.last_kwargs["model"] == "gpt-4o-mini"
    assert client.last_kwargs["messages"] == [{"role": "user", "content": "原文"}]


def test_collect_中途异常直接抛() -> None:
    pieces = [make_chunk("第一段"), TimeoutError("流断了")]
    with pytest.raises(TimeoutError):
        collect(FakeStream(pieces))


def test_collect_partial_保留已收到内容() -> None:
    pieces = [make_chunk("第一段"), make_chunk("第二段"), TimeoutError("流断了")]
    text, error = collect_partial(FakeStream(pieces))
    assert text == "第一段第二段"
    assert isinstance(error, TimeoutError)


def test_collect_partial_正常时异常为None() -> None:
    text, error = collect_partial(FakeStream(full_pieces()))
    assert text == "大模型应用开发"
    assert error is None


def test_collect_partial_空流不报错() -> None:
    text, error = collect_partial(FakeStream([empty_chunk()]))
    assert text == ""
    assert error is None
