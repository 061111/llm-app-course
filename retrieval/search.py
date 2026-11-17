"""M3-T3 · 检索入口：把块、BM25、翻译串起来。

三层结构（和 M1 一个思路：接口定住，实现可换）：
    Chunk[]  --索引-->  BM25  --查询-->  [(chunk, score)]
                            ↑
                      Translator（中文 → 英文关键词）

换检索算法时（比如以后上向量），只改 Searcher 内部的 index 类型，
search() 的签名和返回的 SearchResult 不变，M4 一行都不用动。

【为什么 search() 要能关掉翻译】
  评估时必须能对比"不翻译"和"翻译"两种模式，差多少就是翻译这一步的贡献。
  一个不能关掉组件的系统，就没办法证明这个组件有用。
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from corpus.chunker import Chunk
from retrieval.bm25 import BM25, index_text, tokenize
from retrieval.query import Translator


class SearchResult(BaseModel):
    """一条检索结果。带 chunk 全文，M4 直接拿去拼 prompt。"""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    terms: tuple[str, ...] = Field(description="实际参与打分的词元（已 tokenize）")

    @property
    def source(self) -> str:
        return self.chunk.source()


class Searcher:
    """在内存里做检索。181 个块，构建索引约 10ms，不需要持久化索引。"""

    def __init__(self, chunks: list[Chunk], *, translator: Translator | None = None) -> None:
        if not chunks:
            raise ValueError("块列表为空，没什么可检索的")
        self.chunks = list(chunks)
        self._docs = [tokenize(index_text(c.title, c.text)) for c in self.chunks]
        self._index = BM25(self._docs)
        self.translator = translator

    @classmethod
    def from_store(cls, path: Path | None = None, **kwargs) -> "Searcher":
        """从落盘的 jsonl 建检索器。M3/M4 的默认入口。"""
        from corpus.store import default_paths, load_chunks

        chunks_path, meta_path = default_paths(path.parent if path else None)
        return cls(load_chunks(chunks_path, meta_path), **kwargs)

    def search_terms(self, terms: list[str], *, top_k: int = 5) -> list[SearchResult]:
        """用已经给定的词（可以是中文或英文）检索。

        词元**按首次出现去重**。不去重的话，模型给出 ["P(X|No)", "P(X|Yes)"]
        这类关键词，分词后 "no"/"yes"/"probability" 会重复计入，等于把同一个词
        的权重翻倍 —— BM25 的 tf 是按文档算的，查询侧重复没有理论依据。
        """
        tokens: list[str] = []
        for t in terms:
            for tok in tokenize(t):
                if tok not in tokens:
                    tokens.append(tok)
        hits = self._index.search(tokens, top_k=top_k)
        return [
            SearchResult(
                chunk=self.chunks[i],
                score=round(score, 4),
                rank=rank,
                terms=tuple(tokens),
            )
            for rank, (i, score) in enumerate(hits, 1)
        ]

    def search(
        self, question: str, *, top_k: int = 5, translate: bool = True
    ) -> list[SearchResult]:
        """中文问题直接检索。

        translate=False 时把原问题当作"词"直接 tokenize —— 中文会被切成中文字符串，
        和英文语料一个词都对不上，结果为空。这不是 bug，是把跨语言问题暴露出来。
        """
        if translate:
            if self.translator is None:
                raise ValueError("translate=True 但没有配置 translator")
            terms = self.translator(question)
        else:
            terms = [question]
        return self.search_terms(terms, top_k=top_k)
