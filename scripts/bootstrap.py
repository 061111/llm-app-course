"""一键准备开发环境：虚拟环境 + 依赖 + .env + 语料检查。

用法：
    python scripts/bootstrap.py            # 缺什么补什么，已有的一律不动
    python scripts/bootstrap.py --check    # 只体检，不安装任何东西

设计约束：**幂等且不破坏现状**。已经有 .venv 就只在里面核对依赖，
不会删掉重建——重建会把你装过的调试工具一起清掉。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_PYTHON = (3, 12)
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def say(ok: bool, label: str, detail: str = "") -> None:
    mark = "OK  " if ok else "MISS"
    print(f"[{mark}] {label}" + (f"  {detail}" if detail else ""))


def check_python() -> bool:
    cur = sys.version_info[:2]
    ok = cur >= MIN_PYTHON
    say(ok, "Python 版本", f"{cur[0]}.{cur[1]}（需要 >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}）")
    return ok


def check_venv() -> bool:
    ok = VENV_PY.is_file()
    say(ok, "虚拟环境", str(VENV_PY.relative_to(ROOT)) if ok else "未创建")
    return ok


def create_venv() -> None:
    print("  → 创建 .venv ...")
    subprocess.run([sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True)


def install_deps() -> None:
    """有 uv 用 uv（快、且尊重 uv.lock），没有就退回 pip。"""
    if shutil.which("uv"):
        print("  → uv sync --extra dev")
        subprocess.run(
            ["uv", "sync", "--extra", "dev"],
            cwd=ROOT,
            check=True,
            env={**os.environ, "UV_DEFAULT_INDEX": _index()},
        )
    else:
        print("  → pip install -e .[dev]")
        subprocess.run([str(VENV_PY), "-m", "pip", "install", "-e", ".[dev]"], cwd=ROOT, check=True)


def _index() -> str:
    """国内镜像：默认源在部分网络环境下会超时到放弃。"""
    return "https://mirrors.cloud.tencent.com/pypi/simple"


def check_env_file() -> tuple[bool, bool]:
    """返回 (存在, 看起来已填)。已填指的是 key 不是占位符也不是空。"""
    env = ROOT / ".env"
    if not env.is_file():
        return False, False
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("LLMKIT_API_KEY="):
            value = line.split("=", 1)[1].strip()
            filled = bool(value) and "把这里换" not in value
            return True, filled
    return True, False


def check_corpus() -> bool:
    sys.path.insert(0, str(ROOT))
    from corpus.chunker import corpus_available
    from corpus.docs import DOCS, corpus_dir

    ok = corpus_available()
    say(ok, "语料 PDF", f"{len(DOCS)} 个文件于 {corpus_dir()}" if ok else f"{corpus_dir()} 下缺文件")
    if not ok:
        print("       缺文件的机器上，268 条纯函数测试照跑，15 条集成测试会跳过。")
        print("       有 PDF 的话：set LLMKIT_CORPUS_DIR=<目录> 再跑本脚本。")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="准备开发环境")
    ap.add_argument("--check", action="store_true", help="只体检，不安装")
    args = ap.parse_args(argv)

    print("== 环境体检 ==")
    py_ok = check_python()
    if not py_ok:
        print("\nPython 版本不够，先升级再回来。")
        return 1

    venv_ok = check_venv()
    env_exists, env_filled = check_env_file()
    say(env_exists, ".env 配置文件", "已存在" if env_exists else "未创建")
    if env_exists:
        say(env_filled, "  └ API Key", "已填写" if env_filled else "仍为空或占位符")
    corpus_ok = check_corpus()

    if args.check:
        return 0 if (venv_ok and env_filled) else 1

    print("\n== 补齐缺的部分 ==")
    if not venv_ok:
        create_venv()
    install_deps()

    if not env_exists:
        shutil.copyfile(ROOT / ".env.example", ROOT / ".env")
        print("  → 已从 .env.example 复制出 .env，请填入 LLMKIT_API_KEY")

    print("\n== 下一步 ==")
    if not env_filled:
        print("  1. 打开 .env，把 LLMKIT_API_KEY 换成你的密钥")
        print("  2. 校验配置：python -m llmkit --check")
    if corpus_ok:
        print("  3. 建索引：python scripts/build_index.py")
        print("  4. 提问：  python -m llmkit \"贝叶斯的核心假设是什么\"")
    print("  全量验证：  python scripts/verify.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
