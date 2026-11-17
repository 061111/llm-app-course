"""M2 语料层。

四个里程碑里 M2 的职责：把 PDF 变成带出处的文本块。
  docs.py   5 个 PDF 的身份证（doc_id / 文件名 / 页数）
  gold.py   黄金集：18 条人工核对的标准问答，用来判断后面检索得准不准
  parser.py （T2）解析：PDF → 每页纯文本
  chunker.py（T3）切分：每页 → 若干带坐标的块
  store.py  （T4）落盘：块 → jsonl，可加载

为什么不把 PDF 路径写死在代码里：语料是你的私人文件，可能换机器、
可能换目录。所以路径统一从 corpus_dir() 取，可被环境变量覆盖。
"""
from corpus.docs import DOCS, DEFAULT_CORPUS_DIR, CORPUS_DIR_ENV, DocMeta, corpus_dir
from corpus.gold import GOLD, GoldItem, GoldKind

__all__ = [
    "DOCS",
    "DEFAULT_CORPUS_DIR",
    "CORPUS_DIR_ENV",
    "DocMeta",
    "corpus_dir",
    "GOLD",
    "GoldItem",
    "GoldKind",
]
