"""M1-T1.1 验收：契约层。

三个问题：
  1. 缺必填字段时，报的是 ValidationError 还是别的？
  2. extra="forbid" 是否真的拦住了多余字段？
  3. total_tokens 是不是计算属性（不进 JSON）？
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmkit.types import AskOptions, AskResult, Citation, Usage


def test_缺必填字段报_ValidationError():
    # text 给了、model 和 latency_ms 没给
    with pytest.raises(ValidationError) as info:
        AskResult(text="你好")  # type: ignore[call-arg]

    missing = {err["loc"][0] for err in info.value.errors()}
    assert missing == {"model", "latency_ms"}


def test_多传字段被拒绝():
    with pytest.raises(ValidationError):
        AskResult(text="x", model="m", latency_ms=1.0, hacked=True)  # type: ignore[call-arg]


def test_usage_的_total_tokens_是计算属性():
    usage = Usage(prompt_tokens=12, completion_tokens=8)
    assert usage.total_tokens == 20
    # 它不在序列化结果里
    assert "total_tokens" not in usage.model_dump()


def test_usage_不接受负数():
    with pytest.raises(ValidationError):
        Usage(prompt_tokens=-1)


def test_citation_的_page_允许_None_但不允许_0():
    assert Citation(source="a.pdf").page is None
    with pytest.raises(ValidationError):
        Citation(source="a.pdf", page=0)


def test_temperature_越界会被拦():
    with pytest.raises(ValidationError):
        AskOptions(temperature=3.0)


def test_AskResult_能完整序列化再反序列化():
    original = AskResult(
        text="答案",
        model="fake",
        latency_ms=12.5,
        citations=[Citation(source="a.pdf", page=3, quote="原文")],
        usage=Usage(prompt_tokens=5, completion_tokens=7, cost_cny=0.0001),
        attempts=2,
        trace_id="abc123",
    )
    restored = AskResult.model_validate_json(original.model_dump_json())
    assert restored == original


def test_schema_里能看到每个字段的约束():
    schema = AskResult.model_json_schema()
    assert schema["required"] == ["text", "model", "latency_ms"]
    assert "trace_id" in schema["properties"]
