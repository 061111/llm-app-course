"""离线建索引：解析 PDF → 切分 → 落盘。

用法：
    python scripts/build_index.py                # 输出到 artifacts/
    python scripts/build_index.py --out data/    # 换目录
    python scripts/build_index.py --dry-run      # 只统计，不写文件

这一步和在线问答完全解耦：改完切分参数只需重跑本脚本，
qa/retrieval/llmkit 一行代码都不用动。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corpus.chunker import build_all, corpus_available  # noqa: E402
from corpus.docs import DOCS, corpus_dir  # noqa: E402
from corpus.store import default_paths, format_stats, save_chunks, stats  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="解析并切分语料，生成 chunks.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="输出目录（默认 artifacts/）")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = ap.parse_args(argv)

    if not corpus_available():
        print(f"语料缺失：{corpus_dir()} 下找不到全部 {len(DOCS)} 个 PDF。", file=sys.stderr)
        print("用环境变量指定目录，例如：", file=sys.stderr)
        print('  set LLMKIT_CORPUS_DIR=D:\\path\\to\\pdfs     (Windows)', file=sys.stderr)
        print('  export LLMKIT_CORPUS_DIR=/path/to/pdfs      (Linux/macOS)', file=sys.stderr)
        return 2

    print(f"语料目录：{corpus_dir()}")
    chunks = build_all()
    s = stats(chunks)

    print("\n== 切分结果 ==")
    print(format_stats(s))

    if args.dry_run:
        print("\n--dry-run：未写文件")
        return 0

    chunks_path, meta_path = default_paths(args.out)
    meta = save_chunks(chunks, chunks_path, meta_path)

    size_kb = chunks_path.stat().st_size / 1024
    print("\n== 已落盘 ==")
    print(f"  {chunks_path.relative_to(ROOT)}  ({size_kb:.0f} KB, {meta.chunk_count} 行)")
    print(f"  {meta_path.relative_to(ROOT)}  (切分参数与块数，用于加载时对账)")

    if meta.chunk_count != len(chunks):
        print(f"\n块数不一致：meta={meta.chunk_count} 实际={len(chunks)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
