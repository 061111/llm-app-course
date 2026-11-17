"""M1-T1.4 验收：解析层。

这组测试钉住三件事：
  1. 三种"不干净"的模型输出都能被正确抽出 JSON；
  2. 抽不出来 / 抽出来但不合 schema，抛的都是 LLMParseError 且带原文；
  3. 回灌重试能把"第一次坏、第二次好"救回来。
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from llmkit.errors import LLMParseError
from llmkit.parsing import extract_json, parse_into, parse_with_repair, repair_prompt


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)


def test_干净_json_原样返回():
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_剥掉_json_围栏():
    text = '```json\n{"answer": "是", "confidence": 0.9}\n```'
    assert extract_json(text) == '{"answer": "是", "confidence": 0.9}'


def test_跳过前缀废话():
    text = '好的，以下是结果：\n{"answer": "是", "confidence": 0.9}\n希望有帮助。'
    assert extract_json(text) == '{"answer": "是", "confidence": 0.9}'


def test_前缀里出现大括号也不会错位():
    # 这就是不能用"第一个 { 到最后一个 }"的原因：前面的 {示例} 是干扰项
    text = '格式应该是 {answer, confidence} 这样，实际回答：{"answer": "是", "confidence": 0.5}'
    assert extract_json(text) == '{"answer": "是", "confidence": 0.5}'


def test_字符串里的大括号不会截断():
    text = '{"answer": "用 {} 表示空字典", "confidence": 0.8}'
    parsed = parse_into(text, Answer)
    assert parsed.answer == "用 {} 表示空字典"


def test_空输出抛_LLMParseError_并保留原文():
    with pytest.raises(LLMParseError) as info:
        extract_json("   ")
    assert info.value.raw == "   "


def test_完全没有_json_时抛错():
    with pytest.raises(LLMParseError):
        extract_json("我不知道该怎么回答这个问题。")


def test_json_合法但不符合模型时抛错():
    with pytest.raises(LLMParseError) as info:
        parse_into('{"answer": "是", "confidence": 3.0}', Answer)
    # 置信度 3.0 越界，错误消息里应该点出这是校验问题
    assert "Answer" in str(info.value)


def test_缺字段也会被拦():
    with pytest.raises(LLMParseError):
        parse_into('{"answer": "是"}', Answer)


def test_回灌重试能从坏到好():
    # 初始 text 是坏的；fake_ask 只负责"补问"那一次，所以它返回好结果
    outputs = iter(['{"answer": "第二版", "confidence": 0.7}'])
    prompts: list[str] = []

    def fake_ask(prompt: str) -> str:
        prompts.append(prompt)
        return next(outputs)

    result = parse_with_repair('{"answer": "第一版"}', Answer, fake_ask, max_attempts=2)
    assert result.answer == "第二版"
    assert len(prompts) == 1              # 只补问了一次
    # 解析层交给 ask_fn 的是"错误描述"，不是完整 prompt——
    # 拼 prompt 是 client 层的事（见 test_client 的对应用例）
    assert "不符合 Answer" in prompts[0]


def test_两次都坏就抛最后一次的错():
    def always_bad(prompt: str) -> str:
        return "还是不给你 JSON"

    with pytest.raises(LLMParseError) as info:
        parse_with_repair("不是 JSON", Answer, always_bad, max_attempts=2)
    assert info.value.raw == "还是不给你 JSON"


def test_repair_prompt_包含原始问题与坏输出():
    prompt = repair_prompt("原问题", "坏输出", "字段缺失")
    assert "原问题" in prompt
    assert "坏输出" in prompt
    assert "字段缺失" in prompt
    assert "只输出一个 JSON" in prompt
