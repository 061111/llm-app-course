"""`python -m studio` —— 启一个本地服务。

用法：
    uv run python -m studio                 # http://127.0.0.1:8000
    uv run python -m studio --port 9000

只绑 127.0.0.1。这不是"忘了开 0.0.0.0"，是刻意的：上传的资料是私有文件，
服务没有任何鉴权，暴露到局域网等于把资料目录开放给同网段所有人。
要对外提供，必须先加认证（见 docs/upload-service.md 的"不做的事"）。
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="启动资料问答服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="改代码自动重启（开发用）")
    args = ap.parse_args(argv)

    import uvicorn

    uvicorn.run("studio.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
