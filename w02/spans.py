"""W2-04 · 上下文管理器：把「一定收尾」做成可复用的协议。

和 03 的 @timed 对照着看：
  装饰器     包住「一次函数调用」，跨函数复用；
  上下文管理器 包住「一段代码块」，跨语句复用 —— 一段 with 里可以有十行。

计时逻辑是同一套，区别只在包的是什么。

运行演示：..\\.venv\\Scripts\\python.exe -m w02.spans
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import TracebackType

logger = logging.getLogger(__name__)

Timer = Callable[[], float]
Sink = Callable[[str, float], None]


class TimedSpan:
    """用类实现的上下文管理器：__enter__ 开始计时，__exit__ 结束时记录。

    计时起点放在 __enter__ 而不是 __init__，是因为 with 的展开顺序是
    「先构造、再 enter」：构造函数跑完到真正进入代码块之间可能隔了一段
    （比如你还写了别的语句），那段时间不该算进来。
    """

    def __init__(
        self,
        name: str,
        *,
        timer: Timer = time.perf_counter,
        sink: Sink | None = None,
    ) -> None:
        self.name = name
        self.elapsed: float | None = None
        self.exc_type: type[BaseException] | None = None
        self._timer = timer
        self._sink = sink
        self._started: float | None = None

    def __enter__(self) -> TimedSpan:
        self._started = self._timer()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if self._started is None:
            raise RuntimeError("TimedSpan 必须作为上下文管理器使用")
        self.elapsed = self._timer() - self._started
        self.exc_type = exc_type

        if self._sink is not None:
            self._sink(self.name, self.elapsed)
        else:
            logger.info(
                "%s 耗时 %.3fs%s",
                self.name,
                self.elapsed,
                "（块内出错）" if exc_type is not None else "",
            )

        return False


@contextmanager
def timed_span(
    name: str,
    *,
    timer: Timer = time.perf_counter,
    sink: Sink | None = None,
) -> Iterator[TimedSpan]:
    """同一件事的生成器写法。yield 之前是 __enter__，之后是 __exit__。

    这里把整个工作委托给 TimedSpan，不重复实现，是为了让两种写法的
    行为严格一致 —— 否则你会在「异常时到底记不记录」这种细节上踩两次坑。
    """
    with TimedSpan(name, timer=timer, sink=sink) as span:
        yield span


def _demo() -> None:
    records: list[tuple[str, float]] = []

    def sink(name: str, elapsed: float) -> None:
        records.append((name, elapsed))

    print("① 类写法")
    ticks = iter([0.0, 3.0])
    with TimedSpan("调模型", timer=lambda: next(ticks), sink=sink) as span:
        print("   （假装这里调了一次模型）")
    print("   span.elapsed =", span.elapsed)

    print("② 生成器写法（同一个东西）")
    ticks = iter([10.0, 10.5])
    with timed_span("解析 JSON", timer=lambda: next(ticks), sink=sink) as span:
        print("   （假装这里解析了一次）")
    print("   span.elapsed =", span.elapsed)

    print("③ 块里抛异常：照样记录，异常不被吞")
    ticks = iter([0.0, 0.0])
    try:
        with timed_span("会炸的块", timer=lambda: next(ticks), sink=sink) as span:
            raise ValueError("块里炸了")
    except ValueError as exc:
        print("   外层抓到:", type(exc).__name__, exc)
    print("   span.exc_type =", span.exc_type.__name__)
    print("   span.elapsed  =", span.elapsed)

    print("④ 记录汇总（含正常与异常两类）")
    for name, elapsed in records:
        print(f"   {name:<10} {elapsed}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
