"""W2 第一讲：类型注解 + Pydantic —— 给 LLM 的输出上锁。

为什么 LLM 应用第一件事是干这个：
模型返回的是**字符串**，你的程序要的是**结构体**。中间这道校验层没了，
后面所有代码都在猜字段在不在、类型对不对、值合不合法。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class LLMParseError(RuntimeError):
    """所有候选输出都解析失败。调用方应该降级或重问，而不是崩。"""


class Citation(BaseModel):
    doc_id: str = Field(min_length=1, description="文档标识")
    quote: str = Field(min_length=1, max_length=500, description="原文引用片段")


class Answer(BaseModel):
    """RAG 生成层的输出契约。以后所有 prompt 都必须按这个 schema 产出。"""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, description="给用户的回答")
    citations: list[Citation] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal["ok", "refused"] = "ok"

    @field_validator("answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("answer 不能是空白字符串")
        return v.strip()

    @model_validator(mode="after")
    def _refused_rule(self) -> Answer:
        """跨字段规则只能写在 model_validator 里 —— field_validator 看不到后面的字段。"""
        if self.status == "refused" and self.citations:
            raise ValueError("拒答时不应该附带引用")
        if self.status == "refused" and self.confidence > 0.5:
            raise ValueError("拒答说明没把握，confidence 不该这么高")
        return self


def _extract_json(raw: str) -> str:
    """模型爱加围栏、爱在 JSON 前后写废话，这里把裸 JSON 抠出来。"""
    if m := FENCE.search(raw):
        return m.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("输出里没有 JSON 对象")
    return raw[start : end + 1]


def parse_answer(raw: str) -> Answer:
    """单次解析：抽 JSON → json.loads → Pydantic 校验。任一步失败都抛异常。"""
    try:
        data = json.loads(_extract_json(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"不是合法 JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"顶层必须是对象，拿到 {type(data).__name__}")
    return Answer.model_validate(data)


def parse_with_retry(candidates: Iterable[str]) -> Answer:
    """把模型的多轮输出依次试一遍 —— 这就是「解析失败重试」的最小实现。

    真实场景里 candidates 来自「重问一次」，这里用可迭代对象让逻辑可测。
    """
    errors: list[str] = []
    for i, raw in enumerate(candidates, 1):
        try:
            return parse_answer(raw)
        except ValueError as e:  # ValidationError 本身就是 ValueError 子类，不用重复捕获
            errors.append(f"第 {i} 次: {e}")
    raise LLMParseError("；".join(errors) or "没有任何候选输出")


if __name__ == "__main__":
    good = '{"answer": "K-means 的 K 要手动定", "citations": [{"doc_id": "d1", "quote": "K 需预设"}], "confidence": 0.8}'
    bad = '好的，以下是结果：\n```json\n{"answer": "", "confidence": 2}\n```'
    print(parse_with_retry([bad, good]).model_dump_json(indent=2))
