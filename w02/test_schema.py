"""W2 第一讲配套测试：11 个用例，覆盖「正常 / 脏输出 / 越界 / 重试」。

跑法： .venv\\Scripts\\python.exe -m pytest
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from w02.schema import Answer, LLMParseError, parse_answer, parse_with_retry

GOOD = (
    '{"answer": "混合检索能提升召回", '
    '"citations": [{"doc_id": "d1", "quote": "BM25 补充向量召回"}], '
    '"confidence": 0.82}'
)


def test_正常解析() -> None:
    a = parse_answer(GOOD)
    assert a.confidence == pytest.approx(0.82)
    assert a.citations[0].doc_id == "d1"
    assert a.status == "ok"


def test_带_markdown_围栏() -> None:
    assert parse_answer(f"```json\n{GOOD}\n```").answer == "混合检索能提升召回"


def test_前后有废话() -> None:
    raw = f"好的，结果如下：\n{GOOD}\n希望对你有帮助。"
    assert parse_answer(raw).answer == "混合检索能提升召回"


def test_citations_可以省略() -> None:
    a = parse_answer('{"answer": "不知道", "confidence": 0.1}')
    assert a.citations == []


def test_缺字段报错() -> None:
    with pytest.raises(ValidationError) as ei:
        parse_answer('{"confidence": 0.5}')
    assert "answer" in str(ei.value)


def test_confidence_越界() -> None:
    with pytest.raises(ValidationError):
        parse_answer('{"answer": "x", "confidence": 1.5}')


def test_answer_全是空白() -> None:
    with pytest.raises(ValidationError):
        parse_answer('{"answer": "   ", "confidence": 0.5}')


def test_answer_首尾空白被裁掉() -> None:
    assert parse_answer('{"answer": "  hi  ", "confidence": 0.5}').answer == "hi"


def test_citations_超过_5_条() -> None:
    cits = ",".join(f'{{"doc_id":"d{i}","quote":"q"}}' for i in range(6))
    with pytest.raises(ValidationError):
        parse_answer(f'{{"answer":"x","confidence":0.5,"citations":[{cits}]}}')


def test_不是_JSON_就报错() -> None:
    with pytest.raises(ValueError, match="JSON"):
        parse_answer("我觉得答案是 42")


def test_重试取第一个成功的() -> None:
    bad = '{"answer": "", "confidence": 2}'
    a = parse_with_retry([bad, "随便说说", GOOD])
    assert a.confidence == pytest.approx(0.82)


def test_全部失败抛_LLMParseError() -> None:
    with pytest.raises(LLMParseError):
        parse_with_retry(['{"answer": ""}', "not json"])


def test_多余字段报错() -> None:
    """模型爱编字段。默认 Pydantic 会静默丢掉，我们开了 forbid 才看得见。"""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        parse_answer('{"answer": "x", "confidence": 0.5, "理由": "我自己加的"}')


def test_拒答却带引用() -> None:
    with pytest.raises(ValidationError, match="不应该附带引用"):
        parse_answer(
            '{"answer": "资料里没有", "confidence": 0.2, "status": "refused",'
            ' "citations": [{"doc_id": "d1", "quote": "q"}]}'
        )


def test_拒答但很自信() -> None:
    with pytest.raises(ValidationError, match="confidence 不该这么高"):
        parse_answer('{"answer": "资料里没有", "confidence": 0.9, "status": "refused"}')
