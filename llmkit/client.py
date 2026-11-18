"""M1-T1.3 / T1.5 · 调用层：唯一一个 import openai 的文件。

三个对外函数：
  ask()             同步一次性调用 → AskResult
  ask_structured()  同步 + 结构化输出 → 你指定的 Pydantic 模型
  stream_ask()      流式 → 逐段吐 str

设计上只有一处"必须懂"：
  **重试包住了 create，但没包住遍历。** 流式那条路径里，`retry_call` 只负责把
  "流建立起来"这个动作重试掉；一旦开始 `for chunk in stream`，再断就救不回来了——
  已经吐给用户的字不会消失，重来只会重复。这是 04 实测出的结论。
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from time import perf_counter
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from llmkit.config import Settings
from llmkit.errors import FatalError, LLMError, RetryableError, retry_call
from llmkit.obs import log_event, new_trace_id
from llmkit.parsing import parse_with_repair, repair_prompt
from llmkit.types import AskOptions, AskResult, Usage

M = TypeVar("M", bound=BaseModel)


def make_client(settings: Settings) -> OpenAI:
    """构造 SDK 客户端。抽成函数是为了让测试能替换成假的。"""
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_s,
    )


def _messages(prompt: str, system: str) -> list[dict[str, str]]:
    """chat 接口吃的是消息数组，不是裸字符串。system 为空时就不放，别塞空串。"""
    if system:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": prompt}]


def _text_of(resp: object) -> str:
    """从 ChatCompletion 里取正文，带两层防御：choices 可能为空，content 可能为 None。"""
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise FatalError("服务端返回了空 choices 列表")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content or ""


def _usage_of(resp: object, settings: Settings) -> Usage:
    """有些兼容服务不返回 usage。缺就记 0，不要报错——不能因为记账失败就让整次调用失败。"""
    raw = getattr(resp, "usage", None)
    prompt_tokens = int(getattr(raw, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(raw, "completion_tokens", 0) or 0)
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_cny=round(settings.cost_cny(prompt_tokens, completion_tokens), 6),
    )


def _retry_logger(trace_id: str, max_attempts: int) -> Callable[[int, LLMError, float], None]:
    """重试时打一条结构化日志。回调签名必须匹配 retry_call 的 on_retry。"""

    def _log(attempt: int, err: LLMError, delay: float) -> None:
        log_event(
            "llm.retry",
            trace_id=trace_id,
            attempt=attempt,
            max_attempts=max_attempts,
            error=type(err).__name__,
            detail=str(err),
            delay_s=round(delay, 3),
        )

    return _log


def ask(
    prompt: str,
    *,
    options: AskOptions | None = None,
    settings: Settings | None = None,
    client: object | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> AskResult:
    """同步调用一次模型。

    参数注入点有三个，都只为测试服务（生产调用只传 prompt）：
      settings  不给就读环境变量
      client    不给就真建 SDK 客户端
      sleep     不给就真睡；测试传个记录器，重试路径零耗时
    """
    opts = options or AskOptions()
    settings = settings or Settings.from_env()
    client = client or make_client(settings)

    trace_id = new_trace_id()
    model = opts.model or settings.model
    started = perf_counter()
    calls = {"n": 0}

    def _once() -> object:
        calls["n"] += 1
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=_messages(prompt, opts.system),
            temperature=opts.temperature,
            max_tokens=opts.max_tokens,
        )
        # 空正文不是"答了但很短"，是没答。实测（2026-11-18）：推理模型把
        # max_tokens 全花在 reasoning_tokens 上时，content 就是空串、finish_reason=length。
        # 以前这里静默返回 ""，下游把它当答案渲染出去，评估里表现为"回答为空"。
        # 空正文按**可重试**处理：同一请求重发往往会正常返回；重试用尽后如实报错。
        if not _text_of(resp).strip():
            raise RetryableError("模型返回空正文（可能被 max_tokens 截断或推理 token 占满）")
        return resp

    try:
        resp = retry_call(
            _once,
            max_attempts=opts.max_attempts,
            sleep=sleep,
            on_retry=_retry_logger(trace_id, opts.max_attempts),
        )
    except LLMError as exc:
        log_event(
            "llm.call",
            trace_id=trace_id,
            status="error",
            model=model,
            attempts=calls["n"],
            latency_ms=round((perf_counter() - started) * 1000, 2),
            error=type(exc).__name__,
            detail=str(exc),
        )
        raise

    result = AskResult(
        text=_text_of(resp),
        model=model,
        latency_ms=round((perf_counter() - started) * 1000, 2),
        usage=_usage_of(resp, settings),
        attempts=calls["n"],
        trace_id=trace_id,
    )
    log_event(
        "llm.call",
        trace_id=trace_id,
        status="ok",
        model=model,
        attempts=calls["n"],
        latency_ms=result.latency_ms,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        cost_cny=result.usage.cost_cny,
    )
    return result


def ask_structured(
    prompt: str,
    model_cls: type[M],
    *,
    options: AskOptions | None = None,
    settings: Settings | None = None,
    client: object | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_parse_attempts: int = 2,
) -> M:
    """要求模型输出能对上 model_cls 的 JSON；对不上就把错误回灌重问一次。"""
    first = ask(prompt, options=options, settings=settings, client=client, sleep=sleep)
    latest = {"raw": first.text}

    def _reask(error_text: str) -> str:
        follow_up = repair_prompt(prompt, latest["raw"], error_text)
        retry = ask(follow_up, options=options, settings=settings, client=client, sleep=sleep)
        latest["raw"] = retry.text
        return retry.text

    return parse_with_repair(
        first.text, model_cls, _reask, max_attempts=max_parse_attempts
    )


def stream_ask(
    prompt: str,
    *,
    options: AskOptions | None = None,
    settings: Settings | None = None,
    client: object | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[str]:
    """流式调用。**返回时流已经建立好了**，配置错误会立刻抛，不会拖到第一次 next。

    注意这个函数本身不是生成器（没有 yield），所以 retry_call 在调用时就执行了。
    真正的遍历在 _iter_pieces 里，它才是生成器。
    """
    opts = options or AskOptions()
    settings = settings or Settings.from_env()
    client = client or make_client(settings)

    trace_id = new_trace_id()
    model = opts.model or settings.model
    started = perf_counter()
    calls = {"n": 0}

    def _open() -> object:
        calls["n"] += 1
        return client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=_messages(prompt, opts.system),
            temperature=opts.temperature,
            max_tokens=opts.max_tokens,
            stream=True,
        )

    stream = retry_call(
        _open,
        max_attempts=opts.max_attempts,
        sleep=sleep,
        on_retry=_retry_logger(trace_id, opts.max_attempts),
    )
    return _iter_pieces(stream, trace_id=trace_id, started=started, model=model)


def _iter_pieces(
    stream: object, *, trace_id: str, started: float, model: str
) -> Iterator[str]:
    """把 chunk 流切成纯文本片段。两种块必须跳过，否则崩：

    1. `choices` 为空的末块  → 取 [0] 会 IndexError
    2. `delta.content` 为 None 的结束块 → "".join 会 TypeError
    """
    chars = 0
    status = "ok"
    try:
        for chunk in stream:  # type: ignore[union-attr]
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            piece = getattr(choices[0].delta, "content", None)
            if not piece:
                continue
            chars += len(piece)
            yield piece
    except BaseException:
        status = "error"
        raise
    finally:
        closer = getattr(stream, "close", None)
        if callable(closer):
            closer()
        log_event(
            "llm.stream",
            trace_id=trace_id,
            status=status,
            model=model,
            chars=chars,
            latency_ms=round((perf_counter() - started) * 1000, 2),
        )


def collect(pieces: Iterable[str]) -> str:
    """把片段拼成整段。"""
    return "".join(pieces)


def collect_partial(pieces: Iterable[str]) -> tuple[str, BaseException | None]:
    """尽力收集：中途断了也把已收到的部分还给你，同时回传异常。

    为什么要这个：流式答案给用户看了一半断掉，比"什么都没给"体验好得多。
    """
    buffer: list[str] = []
    try:
        for piece in pieces:
            buffer.append(piece)
    except BaseException as exc:  # noqa: BLE001 - 这里就是要接住一切再交出去
        return "".join(buffer), exc
    return "".join(buffer), None
