"""M1-T1.6 · 观测层：结构化日志 + trace id。

为什么不 `print`：
  `print` 的输出是给人的，没法按字段过滤。一行 JSON 才能被 `jq`、被日志平台、
  被 pytest 的 caplog 按字段断言。

日志格式固定为一行一个 JSON：
  {"ts": "...", "level": "INFO", "event": "llm.call", "trace_id": "...", "attempt": 1, ...}

`event` 是我们自己的字段名，不是 logging 的——logging 负责"什么时候打"，
我们负责"打什么"。两者分工在 `JsonFormatter` 里体现。
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, TextIO

LOGGER_NAME = "llmkit"
_RESERVED = frozenset(
    {"name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
     "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
     "relativeCreated", "thread", "threadName", "processName", "process", "taskName"}
)


class JsonFormatter(logging.Formatter):
    """把 LogRecord 拍成一行 JSON。

    `extra={...}` 传进来的键会被合并进去——这就是我们能打 `trace_id`/`attempt` 的方式。
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        # ensure_ascii=False 让中文日志可读，不变成 \uXXXX
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO", stream: TextIO | None = None) -> logging.Logger:
    """给 llmkit 装上 JSON handler。重复调用不会叠加 handler（幂等）。"""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False  # 不让消息冒到 root，避免和外部配置各打一遍

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:  # 没配过就配一个默认的，保证日志不会凭空消失
        setup_logging()
    return logger


def new_trace_id() -> str:
    """12 位十六进制。够短能看，够长不撞。"""
    return uuid.uuid4().hex[:12]


def log_event(event: str, **fields: Any) -> None:
    """打一条结构化事件。字段名冲突由 logging 自己报错，这里不吞。"""
    get_logger().info(event, extra=fields)
