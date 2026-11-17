"""W2-04 · 生成器 + 上下文管理器：把流式响应切成文本，并保证连接一定关掉。

一句话：`yield` 让函数边算边给，`with` 让收尾一定发生。

为什么需要这一层：
  1. 流式接口返回的是迭代器，不是一个完整对象 —— 只能往前走一遍；
  2. 迭代器背后挂着一条 HTTP 连接，没关就是资源泄漏；
  3. 碎片流里有 choices 为空的块和 content 为 None 的块，直接取字段会崩。

运行演示：..\\.venv\\Scripts\\python.exe -m w02.streaming
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

DEFAULT_MODEL = "gpt-4o-mini"


@contextmanager
def open_text_stream(
    client: Any,
    text: str,
    *,
    model: str = DEFAULT_MODEL,
) -> Iterator[Iterable[Any]]:
    """打开一条流，离开 with 块时一定 close()。

    注意「打开」这件事发生在 __enter__ 里（也就是第一次 with 时），
    不是在调用 open_text_stream() 的那一刻 —— @contextmanager 的函数体
    要等 with 才跑。所以单纯拿到这个对象不会发出任何请求。
    """
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        stream=True,
    )
    try:
        yield stream
    finally:
        stream.close()


def iter_text(stream: Iterable[Any]) -> Iterator[str]:
    """把 chunk 流切成纯文本片段。

    两次跳过分别对应两种真实崩溃：
      - choices 为空 → 直接取 [0] 会 IndexError
      - content 为 None → 拼进字符串会 TypeError
    """
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content
        if piece:
            yield piece


def collect(stream: Iterable[Any]) -> str:
    """拼成整段文本。中途断流就抛，已经收到的碎片随异常一起丢掉。"""
    return "".join(iter_text(stream))


def collect_partial(stream: Iterable[Any]) -> tuple[str, Exception | None]:
    """拼成整段文本；中途断流则返回「已收到的部分」和那个异常。

    返回值是二元组而不是抛异常，因为调用方往往两样都要：
    既要告诉用户「下面没了」，又要保住上面已经显示出来的字。
    """
    parts: list[str] = []
    try:
        for piece in iter_text(stream):
            parts.append(piece)
    except Exception as exc:
        return "".join(parts), exc
    return "".join(parts), None


def _demo() -> None:
    from w02.fakes import FakeClient, empty_chunk, make_chunk

    def full_pieces() -> list[Any]:
        return [
            make_chunk("大模型"),
            make_chunk("应用"),
            make_chunk("开发"),
            make_chunk(None, finish_reason="stop"),
            empty_chunk(),
        ]

    print("① 完整地收集一条流")
    client = FakeClient(full_pieces())
    with open_text_stream(client, "原文") as stream:
        print("   结果:", repr(collect(stream)))
    print("   stream.closed =", client.last_stream.closed)

    print("② 边收边打印（真正的流式体验）")
    client = FakeClient(full_pieces())
    with open_text_stream(client, "原文") as stream:
        for piece in iter_text(stream):
            print("    收到:", piece)

    print("③ 提前 break：连接照样关闭")
    client = FakeClient(full_pieces())
    with open_text_stream(client, "原文") as stream:
        for piece in iter_text(stream):
            print("    只取第一段:", piece)
            break
    print("   stream.closed =", client.last_stream.closed)

    print("④ 流到一半断掉：已收到的内容还在")
    client = FakeClient([make_chunk("第一段"), TimeoutError("流断了")])
    with open_text_stream(client, "原文") as stream:
        text, error = collect_partial(stream)
    print("   已收到:", repr(text), "| 异常:", type(error).__name__)
    print("   stream.closed =", client.last_stream.closed)

    print("⑤ 不进入 with 就不发请求")
    client = FakeClient(full_pieces())
    cm = open_text_stream(client, "原文")
    print("   拿到上下文管理器后 create_calls =", client.create_calls)
    with cm as stream:
        print("   进入 with 后 create_calls =", client.create_calls)


if __name__ == "__main__":
    _demo()
