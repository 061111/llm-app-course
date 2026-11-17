"""M1-T1.6 验收：观测层。

日志必须是**一行一个 JSON**，否则没法被机器聚合。
这里用 capsys 抓 stderr 再 json.loads——能解析才算合格。
"""
from __future__ import annotations

import json
import logging

from llmkit.obs import LOGGER_NAME, JsonFormatter, new_trace_id, setup_logging


def test_日志是一行合法_JSON(capsys):
    logger = setup_logging("INFO")
    logger.info("llm.call", extra={"trace_id": "abc", "attempts": 2, "latency_ms": 12.5})

    line = capsys.readouterr().err.strip()
    payload = json.loads(line)

    assert payload["event"] == "llm.call"
    assert payload["trace_id"] == "abc"
    assert payload["attempts"] == 2
    assert payload["level"] == "INFO"


def test_中文不被转成_unicode_转义(capsys):
    logger = setup_logging("INFO")
    logger.info("报错", extra={"detail": "密钥无效"})

    line = capsys.readouterr().err
    assert "密钥无效" in line
    assert "\\u" not in line


def test_setup_logging_幂等不会叠加_handler(capsys):
    setup_logging("INFO")
    setup_logging("INFO")
    setup_logging("INFO")

    logger = logging.getLogger(LOGGER_NAME)
    assert len(logger.handlers) == 1


def test_级别可调(capsys):
    logger = setup_logging("WARNING")
    logger.info("不该出现")
    logger.warning("该出现")

    err = capsys.readouterr().err
    assert "该出现" in err
    assert "不该出现" not in err


def test_exception_信息会被写进_json(capsys):
    logger = setup_logging("ERROR")
    try:
        raise ValueError("内部炸了")
    except ValueError:
        logger.exception("llm.call")

    payload = json.loads(capsys.readouterr().err.strip())
    assert "ValueError: 内部炸了" in payload["exc"]


def test_trace_id_是_12_位且不重复():
    ids = {new_trace_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(len(i) == 12 for i in ids)


def test_formatter_默认值缺失时用_str_兜底():
    # 传一个 json 序列化不了的对象，不能因此抛异常把日志打挂
    record = logging.LogRecord(
        name="llmkit", level=logging.INFO, pathname=__file__, lineno=1,
        msg="x", args=(), exc_info=None,
    )
    record.weird = object()
    text = JsonFormatter().format(record)
    assert json.loads(text)["weird"].startswith("<object")
