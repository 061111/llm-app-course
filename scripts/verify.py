"""全量验证：静态检查 → 单元测试 → 索引完整性 → 配置可用性。

用法：
    python scripts/verify.py              # 全部跑一遍
    python scripts/verify.py --no-lint    # 没装 ruff 时跳过
    python scripts/verify.py --no-network # 跳过需要真实 API 的那一项

这一项存在的理由：项目有 5 个可跑的命令行入口和 283 条测试，
改完配置容易"代码没错、环境错了"。把它当提交前的一把闸。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
RUFF = ROOT / ".venv" / ("Scripts/ruff.exe" if sys.platform == "win32" else "bin/ruff")


def run(label: str, cmd: list[str], *, ok_codes: tuple[int, ...] = (0,)) -> bool:
    print(f"\n== {label} ==")
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = (proc.stdout or "") + (proc.stderr or "")
    if output.strip():
        print("\n".join(output.strip().splitlines()[-18:]))
    ok = proc.returncode in ok_codes
    print(f"→ {'通过' if ok else '失败'} (exit={proc.returncode})")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="提交前全量验证")
    ap.add_argument("--no-lint", action="store_true", help="跳过 ruff")
    ap.add_argument("--no-network", action="store_true", help="跳过需要真实 API 的配置检查")
    args = ap.parse_args(argv)

    results: list[tuple[str, bool]] = []

    if not args.no_lint:
        if RUFF.is_file():
            results.append(("静态检查", run("静态检查 (ruff)", [str(RUFF), "check", "."])))
        else:
            print("\n== 静态检查 ==\n未安装 ruff，跳过。装它：uv sync --extra dev")
            results.append(("静态检查", True))

    results.append(("单元测试", run("单元测试 (pytest)", [PY, "-m", "pytest", "-o", "addopts="])))

    # 索引完整性：只读 meta 对账，不重新解析 PDF（那要 2 秒且需要语料）
    results.append(("索引完整性", run("索引完整性", [PY, "-m", "corpus.store", "stats"], ok_codes=(0, 1))))

    if not args.no_network:
        # --check 只读 .env 不打网络请求，但需要 key 已填；
        # 没填时返回非 0，这是"待配置"而不是"坏了"，所以不算失败。
        ok = run("配置可用性", [PY, "-m", "llmkit", "--check"], ok_codes=(0, 1, 2))
        results.append(("配置可用性", ok))

    print("\n== 汇总 ==")
    for name, ok in results:
        print(f"  {'通过' if ok else '失败'}  {name}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} 项失败：" + "、".join(failed))
        return 1
    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
