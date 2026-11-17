"""M1-T1.4 · 解析层：从"模型吐的一段文字"里抠出 JSON，再交给 Pydantic。

模型不会乖乖只回 JSON。实测会遇到这三种：
  1. 干净 JSON                     → 直接解析
  2. ```json 围栏包着               → 扒掉围栏
  3. 前面一句"好的，以下是结果："    → 跳过前缀找 JSON 起点

第 3 种不能靠"找第一个 { 到最后一个 }"——字符串里出现大写括号就错位了。
正解是 `json.JSONDecoder().raw_decode()`：它从指定位置开始解析，解析成功就告诉你
结束位置，失败就抛异常。于是一个一个 { 试过去即可。
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from llmkit.errors import LLMParseError

M = TypeVar("M", bound=BaseModel)

FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.S)
_DECODER = json.JSONDecoder()


def _first_json(s: str) -> str | None:
    """扫描字符串，返回第一个能完整解析的 JSON 片段原文。找不到返回 None。"""
    for index, char in enumerate(s):
        if char not in "{[":
            continue
        try:
            _, end = _DECODER.raw_decode(s, index)
        except json.JSONDecodeError:
            continue
        return s[index:end]
    return None


def extract_json(text: str) -> str:
    """把模型输出变成一段合法 JSON 文本。失败抛 LLMParseError（带原文）。"""
    if not text or not text.strip():
        raise LLMParseError("模型返回空内容，无法解析", raw=text)

    # 优先信任围栏里的内容：围栏是模型明确标注"这是代码/数据"的信号
    fenced = [f for f in FENCE_RE.findall(text) if f.strip()]
    for candidate in [*fenced, text]:
        found = _first_json(candidate)
        if found is not None:
            return found

    raise LLMParseError("在模型输出里找不到合法 JSON", raw=text)


def parse_into(text: str, model_cls: type[M]) -> M:
    """一步到位：抽 JSON → Pydantic 校验。"""
    payload = extract_json(text)
    try:
        return model_cls.model_validate_json(payload)
    except ValidationError as exc:
        raise LLMParseError(
            f"JSON 合法但不符合 {model_cls.__name__}：{exc.error_count()} 处错误", raw=text
        ) from exc


def repair_prompt(original: str, bad_output: str, error: str) -> str:
    """把"你上次错在哪"喂回去。这是结构化输出能稳定工作的关键一步。"""
    return (
        f"{original}\n\n"
        "你上一次的输出无法解析，请不要输出任何解释文字，只输出一个 JSON 对象。\n"
        f"上一次的输出是：\n{bad_output}\n\n"
        f"错误信息：\n{error}"
    )


def parse_with_repair(
    text: str,
    model_cls: type[M],
    ask_fn: Callable[[str], str],
    *,
    max_attempts: int = 2,
) -> M:
    """解析失败就把错误回灌、重新问一次。

    `ask_fn` 是"给一段 prompt，返回模型原始文本"——由 client 注入，
    所以这层不需要知道 http 或 SDK 的存在。
    """
    if max_attempts < 1:
        raise ValueError("max_attempts 至少为 1")

    current = text
    last_error: LLMParseError | None = None
    for _ in range(max_attempts):
        try:
            return parse_into(current, model_cls)
        except LLMParseError as exc:
            last_error = exc
            current = ask_fn(str(exc))

    assert last_error is not None
    raise last_error
