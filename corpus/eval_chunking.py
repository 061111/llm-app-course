"""M2-T3 的尺子：用黄金集量切分质量。

为什么需要这把尺子：切分参数（max_chars、分组方式）有十几种组合，
没有可复算的指标，"这个切法更好"就只能靠感觉。而感觉在切分这件事上特别不可靠——
改完以后你去读几个块，总能找到"看起来还挺合理"的证据。

【三个指标，以及它们各自惩罚什么】

1. keys_found（答案关键词覆盖率）
   每道题声明几个"答案必须包含的字符串"，检查覆盖该题所需页的块里有没有它们。
   惩罚：切太碎 —— 算法和它算出的 0.278 被切到两个块里，这个块就不足以回答。

2. chunks_needed（回答一题需要几个块）
   同一题所需页落在几个不同的块里。数目越大，检索要同时命中多个块才答得出来。
   惩罚：切太碎 + 分组太保守。

3. crosstalk（串味块数）
   一个块里同时含有两道**不同题**的页 —— 说明它把两个话题焊在了一起。
   惩罚：切太大 / 合并过度。检索时这种块对两道题都"部分相关"，排序会乱。

三个指标互相拉扯：把 max_chars 调大，1、2 变好，3 变差。所以不能只看一个数，
必须在同一张表里一起看，然后选一个没有明显短板的点。这就是 eval_chunking.py 干的事。

【关键词怎么定的，以及一个反直觉的发现】
   逐条从原文抄，不用公式符号。实测发现公式被 PDF 抽成了斜体 Unicode：
     Accuracy 公式在文本里是 "𝑨𝒄𝒄𝒖𝒓𝒂𝒄𝒚=" 而不是 "Accuracy ="
     SplitInfo 是 "𝑆𝑝𝑙𝑖𝑡𝐼𝑛𝑓𝑜"
   所以任何"用公式当关键词"的做法都会永远匹配不到。刻度只能是散文措辞
   （如 "complement of accuracy"、"penalized"）。
   这也是 M3 检索必须用语义而不是字符串匹配的另一个理由。
"""
from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from corpus.chunker import GroupMode, build_all
from corpus.parser import extract_pages


class Spec(BaseModel):
    """一道题的"答案证据"：必须覆盖哪些页、这些页里必须出现哪些字符串。"""

    model_config = ConfigDict(frozen=True)

    gold_id: str = Field(pattern=r"^G\d{2}$")
    doc_id: str
    pages: tuple[int, ...] = Field(min_length=1)
    keys: tuple[str, ...] = Field(min_length=1, description="必须在这些页里出现的原文措辞")


# pages 来自 gold.py 的 note 字段；keys 逐条对照原文页正文选定。
# 注意 G02/G07/G09/G15 天然跨页 —— 这不是标注失误，是"一个答案由连续几页共同说明"。
SPECS: list[Spec] = [
    Spec(gold_id="G01", doc_id="chap03", pages=(15,),
         keys=("False Positive", "False Negative", "Type I Error", "Type II Error")),
    Spec(gold_id="G02", doc_id="chap03", pages=(18, 19),
         keys=("Accuracy", "Error rate", "complement of accuracy")),
    Spec(gold_id="G03", doc_id="chap03", pages=(57,),
         keys=("CART", "SLIQ", "SPRINT", "equally distributed")),
    Spec(gold_id="G04", doc_id="chap03", pages=(59,),
         keys=("0.278", "0.444")),
    Spec(gold_id="G05", doc_id="chap03", pages=(70,),
         keys=("0.65", "0.92")),
    Spec(gold_id="G06", doc_id="chap03", pages=(73,),
         keys=("Split Information", "C4.5", "penalized")),
    Spec(gold_id="G07", doc_id="chap03", pages=(80, 81, 82),
         keys=("Underfitting", "greedy", "pruning")),
    Spec(gold_id="G08", doc_id="chap04a", pages=(5,),
         keys=("Coverage", "Confidence Factor", "antecedent", "consequent")),
    Spec(gold_id="G09", doc_id="chap04a", pages=(12, 13),
         keys=("Rule-based ordering", "negation", "Class-based ordering")),
    Spec(gold_id="G10", doc_id="chap04a", pages=(25,),
         keys=("prevalence", "default")),
    Spec(gold_id="G11", doc_id="chap04b", pages=(2,),
         keys=("Eager Learners", "Lazy Learners", "Decision Tree", "Rule-Based Classifier")),
    Spec(gold_id="G12", doc_id="chap04b", pages=(11,),
         keys=("Voronoi", "local classifiers", "susceptible to noise")),
    Spec(gold_id="G13", doc_id="chap04c", pages=(10,),
         keys=("conditionally independent", "naïve Bayes")),
    Spec(gold_id="G14", doc_id="chap04c", pages=(19,),
         keys=("Laplace", "hyper-parameter", "non-zero")),
    Spec(gold_id="G15", doc_id="chap04c", pages=(18,),
         keys=("Divorced | No) = 0", "Refund = Yes | Yes) = 0")),
    Spec(gold_id="G16", doc_id="chap05", pages=(3, 4),
         keys=("Support count", "Confidence")),
    Spec(gold_id="G17", doc_id="chap05", pages=(11,),
         keys=("Complete search", "DHP", "Reduce the number of comparisons")),
    Spec(gold_id="G18", doc_id="chap05", pages=(35,),
         keys=("candidate association rules", "non-empty subsets")),
]


# ---------------------------------------------------------------- 关键词自检

WS_RE = re.compile(r"\s+")


def flat(text: str) -> str:
    """把空白压成一个空格。匹配前必须做，否则关键词会因 PDF 换行随机失效。

    实测：chap04b p11 原文是 "are quite susceptible to\\nnoise."，
    硬找 "susceptible to noise" 找不到。这不是切分的问题，是匹配的问题。
    """
    return WS_RE.sub(" ", text)


def verify_keys() -> list[str]:
    """先确认每个关键词真的在它声明的页里，否则后面的指标全是假的。

    这一步是给"我"用的：如果我抄错一个关键词，这里会立刻报出来，
    而不是等到指标异常时去怀疑切分。
    """
    problems: list[str] = []
    cache: dict[str, dict[int, str]] = {}
    for spec in SPECS:
        if spec.doc_id not in cache:
            cache[spec.doc_id] = {p.page: p.text for p in extract_pages(spec.doc_id)}
        joined = flat("\n".join(cache[spec.doc_id].get(p, "") for p in spec.pages))
        for key in spec.keys:
            if flat(key) not in joined:
                problems.append(f"{spec.gold_id} 的关键词 {key!r} 不在 {spec.doc_id} {spec.pages} 里")
    return problems


# ---------------------------------------------------------------- 指标


class ItemResult(BaseModel):
    gold_id: str
    chunks_needed: int
    missing_keys: tuple[str, ...]

    @property
    def answered(self) -> bool:
        return not self.missing_keys


class Report(BaseModel):
    mode: str
    max_chars: int
    chunk_count: int
    median_chars: int
    max_chunk_chars: int
    items: tuple[ItemResult, ...]
    crosstalk: int
    crosstalk_detail: tuple[str, ...]
    titled: int = Field(default=0, description="正文里含自己标题的块数")

    @property
    def keys_ok(self) -> int:
        return sum(1 for i in self.items if i.answered)

    @property
    def keys_ok_rate(self) -> float:
        return self.keys_ok / len(self.items)

    @property
    def titled_rate(self) -> float:
        """块正文里出现自己标题的比例。

        这一列是给"按页切"准备的：chap05 p19–p23 是 Apriori 表格的续页，
        它们的标题是图片，正文从 "TID" 开始。单独成块时，读它的人和模型
        都不知道这张表在讲什么 —— 这样的块哪怕字数够，也是不可用的。
        """
        return self.titled / self.chunk_count if self.chunk_count else 0.0

    @property
    def avg_chunks_needed(self) -> float:
        return sum(i.chunks_needed for i in self.items) / len(self.items)

    @property
    def single_chunk_rate(self) -> float:
        """一题只需一个块的比例。越高说明切分越"整"。"""
        return sum(1 for i in self.items if i.chunks_needed == 1) / len(self.items)


def evaluate(chunks, *, mode: str, max_chars: int) -> Report:
    # 每页被哪些块覆盖
    by_page: dict[tuple[str, int], list] = defaultdict(list)
    for c in chunks:
        for p in range(c.page_start, c.page_end + 1):
            by_page[(c.doc_id, p)].append(c)

    items: list[ItemResult] = []
    chunk_of_gold: dict[int, set[str]] = defaultdict(set)
    for spec in SPECS:
        seen: dict[str, object] = {}
        for p in spec.pages:
            for c in by_page.get((spec.doc_id, p), []):
                if c.text:
                    seen[c.chunk_id] = c
        text = flat("\n".join(c.text for c in seen.values()))  # type: ignore[attr-defined]
        missing = tuple(k for k in spec.keys if flat(k) not in text)
        items.append(
            ItemResult(gold_id=spec.gold_id, chunks_needed=len(seen), missing_keys=missing)
        )
        for cid in seen:
            chunk_of_gold[cid].add(spec.gold_id)

    # 串味：一个块里出现两道以上不同题的证据
    cross = {cid: ids for cid, ids in chunk_of_gold.items() if len(ids) > 1}
    detail = tuple(
        f"{cid} ← {','.join(sorted(ids))}" for cid, ids in sorted(cross.items())[:8]
    )

    sizes = sorted(len(c.text) for c in chunks)
    titled = sum(
        1
        for c in chunks
        if c.title and flat(c.title).casefold() in flat(c.text).casefold()
    )
    return Report(
        mode=mode,
        max_chars=max_chars,
        chunk_count=len(chunks),
        median_chars=sizes[len(sizes) // 2] if sizes else 0,
        max_chunk_chars=sizes[-1] if sizes else 0,
        items=tuple(items),
        crosstalk=len(cross),
        crosstalk_detail=detail,
        titled=titled,
    )


def run_matrix() -> list[Report]:
    """跑 3 种分组 × 4 个 max_chars，供人一次看完再选。"""
    reports: list[Report] = []
    modes: list[GroupMode] = ["page", "title", "stem"]
    for mode in modes:
        for max_chars in (600, 1200, 1800, 3000):
            chunks = build_all(mode=mode, max_chars=max_chars)
            reports.append(evaluate(chunks, mode=mode, max_chars=max_chars))
    return reports


def main() -> int:
    problems = verify_keys()
    if problems:
        print("关键词自检失败，先修关键词，不要看指标：")
        for p in problems:
            print("  ✗", p)
        return 1

    print("关键词自检通过：18 题的关键词全部落在其声明的页里。")
    print()
    print("读表须知：「覆盖」是「切分不丢字」这个下限的 sanity check，")
    print("它不是质量指标 —— 切分只要不丢页，这一列必然是 18/18。")
    print("真正有区分度的是「单块」「串味」「带题」三列。")
    print()
    header = (
        f"{'mode':6}{'max':>6}{'块数':>6}{'中位':>7}{'最大':>7}{'覆盖':>8}"
        f"{'单块':>7}{'串味':>6}{'带题':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in run_matrix():
        print(
            f"{r.mode:6}{r.max_chars:>6}{r.chunk_count:>6}{r.median_chars:>7}"
            f"{r.max_chunk_chars:>7}{r.keys_ok:>4}/{len(r.items):<3}"
            f"{r.single_chunk_rate:>7.0%}{r.crosstalk:>6}{r.titled_rate:>7.0%}"
        )

    print()
    best = build_all(mode="page", max_chars=1200)
    rep = evaluate(best, mode="page", max_chars=1200)
    print("=== 选定方案 page / 1200 的逐题明细 ===")
    for i in rep.items:
        mark = "OK " if i.answered else "缺 "
        miss = "" if i.answered else f"  缺={','.join(i.missing_keys)}"
        print(f"  {mark}{i.gold_id}  块数={i.chunks_needed}{miss}")
    print()
    print(f"块数={rep.chunk_count} 中位={rep.median_chars} 最大={rep.max_chunk_chars}")
    if rep.crosstalk_detail:
        print("串味块：")
        for line in rep.crosstalk_detail:
            print("  ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
