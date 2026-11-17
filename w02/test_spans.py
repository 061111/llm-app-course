"""W2-04 · spans 的测试。

9 条，三组：计时 4、异常路径 3、日志兜底 2。

关键手法跟 03 一样：把 timer 注入成一个「按剧本报数」的函数，
于是耗时断言可以写等号，而不是「大约」。
"""
from __future__ import annotations

import logging

import pytest

from w02.spans import TimedSpan, timed_span


class Ticker:
    """按剧本依次返回时间戳，用完之后报错（防止多调一次被悄悄放过）。"""

    def __init__(self, *values: float) -> None:
        self._values = list(values)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if not self._values:
            raise AssertionError("timer 被调用的次数超出剧本")
        return self._values.pop(0)


def test_类写法记录耗时() -> None:
    records: list[tuple[str, float]] = []
    ticker = Ticker(0.0, 3.0)
    with TimedSpan("调模型", timer=ticker, sink=lambda n, s: records.append((n, s))) as span:
        pass
    assert span.elapsed == 3.0
    assert records == [("调模型", 3.0)]


def test_生成器写法记录耗时() -> None:
    records: list[tuple[str, float]] = []
    ticker = Ticker(10.0, 10.5)
    with timed_span("解析", timer=ticker, sink=lambda n, s: records.append((n, s))) as span:
        pass
    assert span.elapsed == 0.5
    assert records == [("解析", 0.5)]


def test_构造时不调用timer() -> None:
    ticker = Ticker(0.0, 1.0)
    span = TimedSpan("x", timer=ticker, sink=lambda n, s: None)
    assert ticker.calls == 0
    assert span.elapsed is None
    with span:
        pass
    assert ticker.calls == 2


def test_每次进入各自计时() -> None:
    records: list[tuple[str, float]] = []
    span = TimedSpan("复用", timer=Ticker(0.0, 2.0, 100.0, 100.25),
                     sink=lambda n, s: records.append((n, s)))
    with span:
        pass
    first = span.elapsed
    with span:
        pass
    assert first == 2.0
    assert span.elapsed == 0.25
    assert len(records) == 2


def test_正常结束时exc_type为None() -> None:
    with timed_span("正常", timer=Ticker(0.0, 1.0), sink=lambda n, s: None) as span:
        pass
    assert span.exc_type is None


def test_块内异常也记录耗时() -> None:
    records: list[tuple[str, float]] = []
    with pytest.raises(ValueError):
        with timed_span("会炸", timer=Ticker(0.0, 0.75),
                        sink=lambda n, s: records.append((n, s))) as span:
            raise ValueError("块里炸了")
    assert span.elapsed == 0.75
    assert span.exc_type is ValueError
    assert records == [("会炸", 0.75)]


def test_块内异常不被吞() -> None:
    with pytest.raises(ValueError, match="块里炸了"):
        with timed_span("会炸", timer=Ticker(0.0, 0.0), sink=lambda n, s: None):
            raise ValueError("块里炸了")


def test_sink为None时写日志(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="w02.spans"):
        with TimedSpan("兜底", timer=Ticker(0.0, 1.25)):
            pass
    assert "兜底" in caplog.text
    assert "1.250s" in caplog.text


def test_出错时日志带标记(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="w02.spans"):
        with pytest.raises(RuntimeError):
            with TimedSpan("兜底", timer=Ticker(0.0, 0.5)):
                raise RuntimeError("boom")
    assert "块内出错" in caplog.text
