"""W1 环境体检 —— 只用标准库，任何 Python 3.9+ 都能跑。

用法：
    python w01/check_env.py

它会告诉你：装了什么、缺什么、下一步该敲哪条命令。
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

OK = "OK  "
MISS = "MISS"
WARN = "WARN"


@dataclass
class Item:
    name: str
    status: str
    detail: str
    fix: str = ""


def run(cmd: list[str]) -> str | None:
    """跑一条命令拿首行输出，失败返回 None。"""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, shell=False
        )
        text = (out.stdout or out.stderr).strip()
        return text.splitlines()[0] if text else ""
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None


def probe(exe: str, extra: list[str] | None = None) -> str | None:
    """先在 PATH 里找，再去常见安装目录兜底（新装的工具常要重开终端才进 PATH）。"""
    for exe_path in [shutil.which(exe), *(extra or [])]:
        if exe_path and Path(exe_path).exists():
            return exe_path
    return None


def check_python() -> Item:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 13):
        return Item("Python", WARN, f"{ver} —— 3.13 部分生态未适配，建议 3.12", "装 3.12")
    if (v.major, v.minor) < (3, 11):
        return Item("Python", MISS, f"{ver} —— 太旧，fastapi/pydantic v2 会难受", "装 3.12")
    return Item("Python", OK, f"{ver}  ({sys.executable})")


def check_module(mod: str, label: str) -> Item:
    spec = importlib.util.find_spec(mod)
    if spec is None:
        return Item(label, MISS, "未安装", f"uv add {mod}")
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
    except Exception as e:  # noqa: BLE001 - 体检脚本要吞掉一切
        return Item(label, WARN, f"已装但导入失败：{type(e).__name__}", "")
    return Item(label, OK, f"{ver}")


def main() -> int:
    items: list[Item] = [check_python()]

    home = Path.home()
    uv_path = probe("uv", [str(home / ".local" / "bin" / "uv.exe"), str(home / ".local/bin/uv")])
    if uv_path:
        items.append(Item("uv", OK, run([uv_path, "--version"]) or uv_path))
    else:
        items.append(
            Item("uv", MISS, "未安装（依赖管理，强烈建议）",
                 "powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        )

    git = run(["git", "--version"])
    items.append(Item("git", OK, git) if git else Item("git", MISS, "未安装", "winget install Git.Git"))

    node = run(["node", "--version"])
    items.append(Item("node", OK, node) if node else Item("node", WARN, "未装（前端面板时才需要）", ""))

    docker = run(["docker", "--version"])
    items.append(
        Item("docker", OK, docker) if docker else Item("docker", MISS, "未装 —— W4 打包镜像必用",
                                                       "装 Docker Desktop for Windows")
    )

    redis = shutil.which("redis-server") or shutil.which("redis-cli")
    items.append(
        Item("redis", OK, redis) if redis else Item("redis", MISS, "未装 —— W4 起要用",
                                                    "Docker 起一个即可，不必装 Windows 版")
    )

    for mod, label in [
        ("pydantic", "pydantic"),
        ("fastapi", "fastapi"),
        ("httpx", "httpx"),
        ("pytest", "pytest"),
        ("openai", "openai SDK"),
    ]:
        items.append(check_module(mod, label))

    width = max(len(i.name) for i in items)
    print("=" * 68)
    print(f"W1 环境体检  |  {platform.system()} {platform.release()}  |  {platform.machine()}")
    print("=" * 68)
    for it in items:
        print(f"[{it.status}] {it.name.ljust(width)}  {it.detail}")
    print("-" * 68)

    todo = [i for i in items if i.status == MISS and i.fix]
    if not todo:
        print("全部就绪，可以进 W2。")
    else:
        print(f"还差 {len(todo)} 项：")
        for i, it in enumerate(todo, 1):
            print(f"  {i}. {it.name:<12} → {it.fix}")
    print("=" * 68)

    blockers = [i for i in items if i.status == MISS and i.name in {"Python", "uv", "git"}]
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
