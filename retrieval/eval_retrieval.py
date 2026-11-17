"""M3 的尺子：用黄金集量检索。

【这道题在问什么】
  给定 18 个中文问题，检索器把"含答案的那一页"排到第几名？
  这是 RAG 里最硬的指标 —— 检索没取到，生成再强也答不出来。

【为什么用"页"而不是"块"判定命中】
  黄金集记录的出处是页码（chap03:p70）。块是检索单元，页是事实单元。
  判定规则：top-k 里有任意一个块覆盖了该题的**任一**必需页，就算命中。
  用"任一"而不是"全部"：G07 需要 3 页才能答全，但检索只要取到其中一页
  就有机会答对一半，这个区别用 hit@1/hit@3 的差异体现，不该塞进命中判定里。

【三个数】
  hit@k  前 k 名里有没有含答案的块。k=1 决定"直接问"能不能成，k=5 决定 RAG 的上下文预算。
  MRR    第一个命中块的排名倒数（第 1 名=1.0，第 3 名=0.33，没命中=0）。比 hit@k 更细。
  缺口    未命中的题号。这些是 M4 会答错的题，要单独看。

【对照组的价值】
  translate=False 时中文原句直接进 BM25。**注意：它不是 0。**
  黄金集有近一半题目夹着英文术语（"Type I Error"、"C4.5"、"k-NN"），
  这些题不翻译也能字面命中。所以"不翻译 = 全灭"是错的，
  这个基线的作用是量化"翻译到底带来了多少"，而不是证明"没有翻译就不行"。

【实测结果（2026-11-17，18 题，top_k=5，全部来自真实运行）】

    模式        hit@1   hit@3   hit@5   MRR    未命中
    不翻译       56%     61%     72%    0.596  G07,G12,G13,G17,G18
    翻译         67%     83%     94%    0.769  G15（排第 6）

  翻译的净收益：hit@5 +22 个百分点，MRR +0.17。
  G15 是唯一在 k=5 下漏掉的题，k=8 时能命中（排名第 6）。

【调参过程中三个真实修复，每个都有实测支撑】
  1. NFKD 归一化：数学斜体 𝐺𝑖𝑛𝑖 不还原，p59 的 gini 一次也匹配不上（G04 必挂）。
  2. 去分音符：讲义 "Naïve" 与查询 "naive" 是不同词元。
  3. 保留单字符数字：G04 的问题不含答案数字，唯一区分信号是 (0,6)/(1,5)/(2,4)。
     实测 hit@5 89% → 94%。
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from corpus.eval_chunking import SPECS
from retrieval.query import Translator
from retrieval.search import SearchResult, Searcher


class ItemHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    gold_id: str
    rank: int = Field(ge=0, description="第一个命中块的排名；0 表示 top_k 内没命中")
    top1: str = Field(description="第一名的块 id 与出处，用来人工看排序是否合理")
    hit_source: str = Field(default="", description="命中的那个块覆盖的页")

    @property
    def hit(self) -> bool:
        return self.rank > 0


class Report(BaseModel):
    mode: str
    top_k: int
    items: tuple[ItemHit, ...]

    @property
    def hit_at(self) -> dict[int, int]:
        """各 k 的命中数。k 只需要 <= top_k 的值。"""
        return {k: sum(1 for i in self.items if 0 < i.rank <= k) for k in range(1, self.top_k + 1)}

    @property
    def mrr(self) -> float:
        return sum((1.0 / i.rank) if i.rank else 0.0 for i in self.items) / len(self.items)

    @property
    def misses(self) -> list[str]:
        return [i.gold_id for i in self.items if not i.hit]


def _covers(result: SearchResult, doc_id: str, pages: tuple[int, ...]) -> bool:
    c = result.chunk
    return c.doc_id == doc_id and any(c.page_start <= p <= c.page_end for p in pages)


def evaluate(searcher: Searcher, *, mode: str, translate: bool, top_k: int = 5) -> Report:
    items: list[ItemHit] = []
    for spec in SPECS:
        results = searcher.search(_question(spec.gold_id), top_k=top_k, translate=translate)
        rank = 0
        hit_source = ""
        for r in results:
            if _covers(r, spec.doc_id, spec.pages):
                rank = r.rank
                hit_source = f"{r.chunk.source()}"
                break
        top1 = f"{results[0].chunk.chunk_id} {results[0].chunk.source()}" if results else "(无结果)"
        items.append(ItemHit(gold_id=spec.gold_id, rank=rank, top1=top1, hit_source=hit_source))
    return Report(mode=mode, top_k=top_k, items=tuple(items))


def _question(gold_id: str) -> str:
    """从黄金集取问题原文。延迟导入，避免模块级循环依赖。"""
    from corpus.gold import GOLD

    for g in GOLD:
        if g.id == gold_id:
            return g.question
    raise KeyError(gold_id)


def format_report(r: Report) -> str:
    n = len(r.items)
    lines = [f"=== {r.mode}  top_k={r.top_k} ==="]
    for k, hits in r.hit_at.items():
        lines.append(f"  hit@{k}  {hits}/{n}  {hits / n:.0%}")
    lines.append(f"  MRR    {r.mrr:.3f}")
    if r.misses:
        lines.append(f"  未命中  {','.join(r.misses)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    top_k = int(argv[0]) if argv else 5

    translator = Translator()
    searcher = Searcher.from_store(translator=translator)

    no_tr = evaluate(searcher, mode="不翻译（中文原句直接检索）", translate=False, top_k=top_k)
    with_tr = evaluate(searcher, mode="翻译成英文关键词", translate=True, top_k=top_k)

    print(format_report(no_tr))
    print()
    print(format_report(with_tr))
    print()
    print(f"本轮真实模型调用 {translator.calls} 次（其余命中缓存）")
    print()
    print("=== 逐题（翻译模式）===")
    for item in with_tr.items:
        mark = "命中" if item.hit else "未命中"
        rank = f"第{item.rank}名" if item.hit else "-"
        print(f"  {item.gold_id} {mark:4} {rank:6} 命中块={item.hit_source or '-':16} top1={item.top1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
