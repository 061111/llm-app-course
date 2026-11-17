"""M2-T4 · 落盘与加载：切好的块存成文件，后续所有环节读文件。

为什么要有这一层：解析 199 页 PDF 要 ~1.5 秒，切分再 ~1 秒。听起来不多，
但 M3 调检索参数、M4 调 prompt，每个实验都要重新解析一遍的话，
"改一个参数跑 20 次"就从 2 分钟变成 40 分钟，而你会在第 5 次时开始
懒得改参数 —— 这才是真正的代价。

三层职责分开之后，责任也清楚了：
    PDF 变了 → 重跑 save
    切分参数变了 → 重跑 save
    检索/prompt 变了 → 只读文件，不碰 PDF

【为什么是 jsonl 不是 pickle / parquet】
  1. jsonl 是一行一个 JSON，可以直接用编辑器打开看第 37 块长什么样。
     pickle 是二进制，parquet 要装 pyarrow，都为了一点点速度牺牲了可读性。
  2. 181 块、约 9 万字符，jsonl 只有几百 KB，读写都不是瓶颈。
  3. 出问题时能 diff：改切分参数前后各存一份，`diff` 就能看出哪些块变了。

【为什么要一个 meta 文件】
  jsonl 本身不知道自己是用什么参数生成的。等你三天后回来，
  "这个文件是 1200 还是 600 切的"就得靠猜；更糟的是拿过期文件跑评估，
  然后花半小时研究"为什么指标变了"。
  meta.json 记住参数、块数、字数、生成时间，load 时对一下块数 ——
  对不上就直接报错，而不是让错误往下游传。

【一个必须处理的细节：spans 是 tuple of BaseModel】
  Pydantic 的 model_dump_json 会把 tuple 写成数组，model_validate_json 会读回来。
  但 json.dumps(model_dump()) 里 PageSpan 会变成 dict，Chunk 的 tuple 会变成 list，
  再 dump 一次还能读回去 —— 所以统一走 model_dump_json / model_validate_json，
  不混用这两种序列化路径。混用是这块最容易出的 bug。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from corpus.chunker import (
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    OVERLAP_CHARS,
    Chunk,
    build_all,
)
from corpus.docs import DOCS

# 默认存放位置：项目根目录下的 artifacts/chunks.jsonl。
# 不放 corpus/ 里面，因为它是**产物**不是**源码**，而且会被 .gitignore 忽略。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "artifacts"
CHUNKS_NAME = "chunks.jsonl"
META_NAME = "chunks.meta.json"


class ChunkFileMeta(BaseModel):
    """描述这份 jsonl 是怎么来的。字段少而硬，只记能用来校验的东西。"""

    model_config = ConfigDict(extra="forbid")

    built_at: str = Field(description="生成时间，人看的")
    mode: str = Field(description="分组方式：page / title / stem")
    max_chars: int = Field(gt=0)
    min_chars: int = Field(ge=0)
    overlap: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    total_chars: int = Field(ge=0)
    doc_chunks: dict[str, int] = Field(description="每个文档切了多少块")


class StoreStats(BaseModel):
    chunk_count: int
    total_chars: int
    median_chars: int
    min_chars: int
    max_chars: int
    by_doc: dict[str, int]
    empty: int


def default_paths(directory: Path | None = None) -> tuple[Path, Path]:
    d = Path(directory) if directory else DEFAULT_DIR
    return d / CHUNKS_NAME, d / META_NAME


def save_chunks(
    chunks: list[Chunk],
    path: Path,
    meta_path: Path,
    *,
    mode: str = "page",
    max_chars: int = MAX_CHUNK_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> ChunkFileMeta:
    """写 jsonl + meta。返回写进去的 meta，方便调用方打印。

    一行一个 Chunk，用 model_dump_json —— 不要用 model_dump() 再 json.dumps，
    那条路径会把 tuple 变成 list 且不会自动还原（见文件头）。
    """
    path = Path(path)
    meta_path = Path(meta_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(c.model_dump_json())
            f.write("\n")

    counts = Counter(c.doc_id for c in chunks)
    meta = ChunkFileMeta(
        built_at=datetime.now().isoformat(timespec="seconds"),
        mode=mode,
        max_chars=max_chars,
        min_chars=min_chars,
        overlap=overlap,
        chunk_count=len(chunks),
        total_chars=sum(c.char_count for c in chunks),
        doc_chunks={doc_id: counts.get(doc_id, 0) for doc_id in DOCS},
    )
    meta_path.write_text(
        json.dumps(meta.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def load_chunks(path: Path, meta_path: Path | None = None) -> list[Chunk]:
    """读 jsonl。如果 meta 存在，核对块数 —— 对不上立刻报错。

    为什么要核对：文件被手改过、或者写一半中断，都能让块数对不上。
    不做这个检查，症状会变成"检索结果莫名其妙变差"，很难查。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到块文件：{path}\n先运行：python -m corpus.store save"
        )

    chunks: list[Chunk] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            chunks.append(Chunk.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001 - 要把行号带进错误里
            raise ValueError(f"{path} 第 {lineno} 行解析失败：{exc}") from exc

    if meta_path is None:
        meta_path = path.parent / META_NAME
    meta_path = Path(meta_path)
    if meta_path.is_file():
        meta = ChunkFileMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        if meta.chunk_count != len(chunks):
            raise ValueError(
                f"块数不一致：meta 说 {meta.chunk_count} 块，文件里 {len(chunks)} 块。"
                f"文件可能被改过或写坏了，重跑 python -m corpus.store save"
            )
    return chunks


def read_meta(meta_path: Path) -> ChunkFileMeta | None:
    """读 meta。不存在返回 None —— 缺 meta 不算错，只是失去校验能力。"""
    meta_path = Path(meta_path)
    if not meta_path.is_file():
        return None
    return ChunkFileMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))


def stats(chunks: list[Chunk]) -> StoreStats:
    sizes = sorted(c.char_count for c in chunks)
    return StoreStats(
        chunk_count=len(chunks),
        total_chars=sum(sizes),
        median_chars=sizes[len(sizes) // 2] if sizes else 0,
        min_chars=sizes[0] if sizes else 0,
        max_chars=sizes[-1] if sizes else 0,
        by_doc=dict(Counter(c.doc_id for c in chunks)),
        empty=sum(1 for c in chunks if c.is_empty),
    )


def format_stats(s: StoreStats) -> str:
    lines = [
        f"块数      {s.chunk_count}",
        f"总字数    {s.total_chars}",
        f"字数分布  最小 {s.min_chars} / 中位 {s.median_chars} / 最大 {s.max_chars}",
        f"空块      {s.empty}",
        "各文档块数 " + "  ".join(f"{k}={v}" for k, v in sorted(s.by_doc.items())),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    action = argv[0] if argv else "save"
    chunks_path, meta_path = default_paths()

    if action == "save":
        chunks = build_all()
        meta = save_chunks(chunks, chunks_path, meta_path)
        print(f"已写入 {meta.chunk_count} 块 → {chunks_path}")
        print(f"meta → {meta_path}")
        print(format_stats(stats(chunks)))
        return 0

    if action == "stats":
        chunks = load_chunks(chunks_path, meta_path)
        meta = read_meta(meta_path)
        if meta:
            print(
                f"文件生成于 {meta.built_at}  mode={meta.mode} "
                f"max_chars={meta.max_chars} overlap={meta.overlap}"
            )
        print(format_stats(stats(chunks)))
        return 0

    print(f"不认识的命令：{action}\n用法：python -m corpus.store [save|stats]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
