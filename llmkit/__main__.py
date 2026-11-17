"""M1-T1.7 · 命令行入口。

用法：
    python -m llmkit "贝叶斯分类器的假设是什么"
    python -m llmkit --stream "讲一句话"
    python -m llmkit --json "讲一句话"
    python -m llmkit --check           # 只检查配置，不发请求

退出码（脚本里能判断）：
    0  成功
    1  调用失败（LLMError 及其子类）
    2  配置缺失（ConfigError）
"""
from __future__ import annotations

import argparse
import sys

from llmkit.client import ask, collect_partial, stream_ask
from llmkit.config import Settings
from llmkit.errors import ConfigError, LLMError
from llmkit.obs import setup_logging

EXIT_OK = 0
EXIT_CALL_FAILED = 1
EXIT_CONFIG = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmkit",
        description="项目 A 的模型调用层：一条命令问一句。",
    )
    parser.add_argument("prompt", nargs="?", help="要问的问题")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    parser.add_argument("--json", action="store_true", help="以 JSON 打印完整结果")
    parser.add_argument("--check", action="store_true", help="只校验配置，不发请求")
    parser.add_argument("--debug", action="store_true", help="打印结构化调用日志")
    parser.add_argument("--system", default=None, help="覆盖默认 system prompt")
    return parser


def _print_result(result: object) -> None:
    text = getattr(result, "text", "")
    usage = getattr(result, "usage", None)
    print(f"答案: {text}")
    if usage is None:
        return
    cost = usage.cost_cny
    cost_text = f"¥{cost:.6f}" if cost > 0 else "未配置单价"
    print(
        f"tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out"
        f"  |  耗时 {getattr(result, 'latency_ms', 0):.0f}ms"
        f"  |  成本 {cost_text}"
        f"  |  尝试 {getattr(result, 'attempts', 1)} 次"
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.debug:
        setup_logging("INFO")

    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return EXIT_CONFIG

    if args.check:
        print(f"配置可用：base_url={settings.base_url} model={settings.model} key={settings.masked_key}")
        return EXIT_OK

    if not args.prompt:
        print("请给一个问题，例如：python -m llmkit \"贝叶斯定理是什么\"", file=sys.stderr)
        return EXIT_CONFIG

    from llmkit.types import AskOptions

    options = AskOptions(system=args.system) if args.system else AskOptions()

    try:
        if args.stream:
            pieces = stream_ask(args.prompt, options=options, settings=settings)
            text, error = collect_partial(pieces)
            print(text)
            if error is not None:
                print(f"\n[中断] {type(error).__name__}: {error}（已保留上面收到的部分）", file=sys.stderr)
                return EXIT_CALL_FAILED
            return EXIT_OK

        result = ask(args.prompt, options=options, settings=settings)
    except LLMError as exc:
        print(f"调用失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_CALL_FAILED

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_result(result)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
