"""中文资料验收：这是"能读懂中文"这一条的可复算证据。

样本是真实资料（30 页 / 2.35 万汉字 / 59 块），不是造的。
题是我对着原文写的 —— 所以这是**开卷成绩**：它证明链路通、切片准、术语对得上，
不证明泛化能力。这一点必须写在结论里，否则 10/10 会被误读成"中文问答已解决"。

路径用环境变量 `LLMKIT_CN_SAMPLE` 覆盖；本机没有样本时整组跳过（skip），
不会把别人的机器搞红。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from retrieval.search import Searcher
from studio.ingest import ingest

# 样本路径由环境变量决定，仓库里不写死任何人的绝对路径。
# 本机实测用的是一份 30 页中文简答资料（2.35 万汉字）；没有样本时整组跳过。
SAMPLE = Path(
    os.environ.get(
        "LLMKIT_CN_SAMPLE",
        str(Path.home() / "Desktop" / "中文资料样本.pdf"),
    )
)

pytestmark = pytest.mark.skipif(not SAMPLE.is_file(), reason="本机没有中文验收样本")

# (问题, 答案所在页的集合)。多页是因为有些答案本身就跨页 —— 按"任一页命中"算对。
GOLD: list[tuple[str, set[int]]] = [
    ("无偏估计 S 的分母为什么是 n−1", {1}),
    ("p 值是什么，怎么用它判断是否拒绝原假设", {3}),
    ("Hotelling T² 化成 F 统计量的系数怎么算", {3}),
    ("协方差阵不满秩时马氏距离和 T² 会怎样", {1}),
    ("动态聚类法的基本思想和步骤", {5}),
    ("第 k 主成分的方差贡献率怎么定义", {14, 15, 16}),
    ("因子旋转为什么不会改变共同度", {17, 18}),
    ("做因子分析前为什么要先看 KMO 和 Bartlett", {23}),
    ("共同度的含义和它为什么越大越好", {16, 17}),
    ("距离判别、贝叶斯判别、Fisher 判别的区别", {8, 9}),
]


@pytest.fixture(scope="module")
def searcher() -> Searcher:
    result = ingest("u001", SAMPLE.name, SAMPLE.read_bytes())
    assert result.lang == "zh"
    return Searcher(list(result.chunks))


def _covered(result) -> set[int]:
    return set(range(result.chunk.page_start, result.chunk.page_end + 1))


def _rank_of(searcher: Searcher, question: str, expected: set[int], k: int = 5) -> int | None:
    for rank, item in enumerate(searcher.search(question, top_k=k, translate=False), 1):
        if _covered(item) & expected:
            return rank
    return None


@pytest.mark.parametrize("question,expected", GOLD)
def test_每条问题都能在前三命中(searcher: Searcher, question: str, expected: set[int]):
    rank = _rank_of(searcher, question, expected, k=3)
    assert rank is not None, f"前 3 条都没命中 {sorted(expected)}：{question}"


def test_整体指标不低于验收线(searcher: Searcher):
    """实测（2026-11-18）：hit@1 10/10、hit@3 10/10、MRR 1.000。

    验收线刻意定得低于实测值（0.70 / 0.90）：这条测试要抓的是"改动把中文
    检索弄坏了"，不是"每天必须满分"。真实场景的问题不会都像这批一样
    术语密集，把阈值定在 100% 只会让人学会改阈值。
    """
    ranks = [_rank_of(searcher, q, exp) for q, exp in GOLD]
    n = len(GOLD)
    hit1 = sum(1 for r in ranks if r == 1) / n
    hit3 = sum(1 for r in ranks if r and r <= 3) / n
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"\nhit@1={hit1:.0%} hit@3={hit3:.0%} MRR={mrr:.3f}（n={n}）")
    assert hit1 >= 0.70
    assert hit3 >= 0.90


def test_语料外问题检索层仍会返回结果(searcher: Searcher):
    """这是一个**边界声明**，不是 bug。

    "量子纠缠的贝尔不等式"在中文资料里仍能靠"推导/不等式"等二元组命中几块。
    也就是说：**拒答不能靠检索为空来实现**，必须靠生成层（prompt 里的
    "资料未覆盖"约束）。实测这条问题在真实模型下回答"资料未覆盖"，
    见 docs/upload-service.md 的验收表。
    """
    hits = searcher.search("量子纠缠的贝尔不等式怎么推导", top_k=5, translate=False)
    assert hits, "如果哪天这里为空，说明召回被收得太紧，正常问题也会被误杀"
