"""M5-T4 · 术语抽取：从用户资料里挖出"这份文档在说什么"。

【为什么不是"用模型列一下术语"】
  直接问模型"这份资料的术语有哪些"，它会给出一份**看起来很专业**的词表，
  其中有一部分根本不在资料里 —— 这正是"编造"的入口，而且无法机械发现。
  本模块的做法是：**先统计，再让模型挑**。

  统计层（这里）只做一件事：从字面把"像术语的候选"捞出来，每个候选都带
  原文页码和上下文。因为候选是从原文里数出来的，所以它**天生不可能编造**。
  模型在下一步（studio/profile.py）只能从这份候选里挑，挑不出来的词会被丢弃。

【算法：新词发现的两个经典判据】
  一个连续汉字串要成为"词"，必须同时满足：

  1. **内部结合紧**（PMI / 凝固度）
     把串在所有位置切开，取最松的那一刀。若 "贝叶斯" 真的常连用，
     那么 P(贝叶斯) ≫ P(贝)·P(叶斯)，比值取对数就是凝固度。
     取"最松的一刀"而不是平均值：只要有一处能随便拆开，它就不是一个整体词。
     例："的系数" 里 "的" 与 "系数" 各自独立 → 最松的一刀很小 → 被淘汰。

  2. **外部位置自由**（左右邻字熵）
     一个词能出现在各种上下文里（"这个**方差**是…"、"**方差**的估计…"），
     所以它左右相邻的字应该很杂，熵高。
     反过来，"氏距离" 左边几乎永远是"马/欧"，熵低 → 它是碎片不是词。
     这条判据专治"半个词"。

  3. 再加两条工程过滤器：出现次数下限、首尾不能是功能字（的/是/不/就…）。

  生活类比：判断"张伟"是不是一个人名，看两点 —— 这两个字是不是总黏在一起出现
  （而不是"张"和"伟"各自随便配），以及它前面能不能接各种字（"找张伟""张伟说"）。
  类比失效处：人名不需要"专业性"，而术语需要；本模块不判断专业性，
  专业性由模型在候选里挑（且只能在候选里挑）。

【阈值是调出来的，不是拍的】
  在 30 页 / 23528 汉字的中文简答资料上实测（2026-11-18），
  freq≥3 / PMI≥2.0 / 熵≥1.0 时留 443 个候选，Top 30 里 27 个是本领域术语
  （因子、载荷、公共因子、协差阵、方差贡献、主成分、判别函数、马氏距离…），
  另 3 个是"结构/结果/数据"这类跨领域通用词 —— 它们会在模型挑选阶段被筛掉。
  调参过程与失败的几版见 docs/upload-service.md。

【已知限制，不修但要说清】
  1. 小于约 2000 汉字的资料，统计不显著：频次门槛自动降到 2，候选会偏碎。
     这类资料的术语表应当视为"候选"，不能当结论。
  2. 单个拉丁字母（p 值里的 p、F 检验里的 F）不成词，会被丢掉 ——
     它在中文句子里是符号不是词。带数字或连字符的（K-means、C4.5、T²）保留。
  3. 抽取是**字面**的：同义异形（"协方差阵" vs "协差阵"）算两个词，
     合并交给模型（模型只做"归并"，不做"发明"）。
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from corpus.parser import PageText
from retrieval.bm25 import STOPWORDS
from retrieval.cjk import CJK_RUN_RE

# 首尾是这些字的串不作为术语。理由：它们是纯功能字，
# 出现在词首/词尾说明这个串是"半句话"而不是"一个词"。
# 实测案例："前面的系数"(4 次) 首字"前"不在表内但尾字"数"… 所以另有停用词表兜底；
# 真正被这条干掉的是 "不能""就是""说明的" 这类。
CJK_BOUNDARY_STOP = frozenset(
    "的了着过是在和与或及等这那为对不很最更又再只才把被让从到会能可要还想呢吗吧啊呀嘛哦嗯"
    "而就都也其之于以每各某两些该本"
)

# 通用高频词。它们满足凝固度和熵，但没有领域信息量。
# 这张表**只收跨领域无信息量的词**，收进"分析/数据/结构"会误伤本领域术语。
GENERIC_STOPWORDS = frozenset(
    """没有 就是 可以 说明 三条 两者 不能 完全 证明 计算 考虑 结论 初始 上述 下面 上面 因此 所以
    但是 如果 因为 由于 通过 根据 对于 关于 以及 并且 或者 而且 然后 这时 此时 这里 那里 其中
    一个 一种 这样 那样 什么 怎么 为什么 如何 是否 需要 必须 应该 可能 只是 已经 仍然 还是 也是
    都是 不是 有些 一些 很多 很少 全部 所有 每个 各个 如下 以下 可见 显然 例如 比如 通常 一般
    主要 基本 直接 间接 分别 同时 此外 另外 只要 只有 无论 不管 不仅 而是 之一 之间 之后 之前
    以上 我们 你们 他们 自己 大家 这个 那个 这些 那些 时间 情况 方面 问题 部分 地方 东西""".split()
)

# 拉丁词：K-means / C4.5 / T2 / SPSS / Hotelling。
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[.\-_][A-Za-z0-9]+)*")

MAX_TERM_LEN = 6
MAX_OCCURRENCES = 3
CONTEXT_PAD = 18


class Occurrence(BaseModel):
    """术语在原文里的一次出现。page+offset 足以让人翻回去核对。"""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    page: int = Field(ge=1)
    start: int = Field(ge=0, description="页内字符偏移，左闭")
    end: int = Field(ge=0, description="页内字符偏移，右开")
    context: str = Field(description="前后各约 18 字，用于展示依据")

    def source(self) -> str:
        return f"{self.doc_id}:p{self.page}"


class Term(BaseModel):
    """一个候选术语。带分数是为了排序和后续筛选，不是给人看的。"""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    freq: int = Field(ge=1, description="在全库出现的次数")
    score: float
    cohesion: float = Field(default=0.0, description="最小切分点的凝固度（PMI），仅汉字词有意义")
    freedom: float = Field(default=0.0, description="左右邻字熵的较小值")
    kind: str = Field(default="cjk", description="cjk / latin")
    occurrences: tuple[Occurrence, ...] = Field(default_factory=tuple)

    def pages(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(o.source() for o in self.occurrences))


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 1:
        return 0.0
    return -sum(c / total * math.log2(c / total) for c in counter.values())


def _freedom(left: Counter[str], right: Counter[str]) -> float:
    """左右邻字熵的较小值，**但边界标记不参与**。

    为什么必须排除边界：PPT / 笔记这类资料里，术语经常整行出现在行首
    （实测："距离矩阵是聚类分析的核心。" 每句都是独立 CJK 段，左边全是边界）。
    把边界算作一种"邻字"，同一侧 6 次全是边界 → 熵 = 0 → 一个明明很规整的术语
    被误杀。这是"版式造成的假信号"，不是语言学事实。

    一侧内部样本不足 2 个时，用另一侧；两侧都不足 2 个，返回 inf 表示
    "此判据无信息"，交由凝固度把关（碎片仍有 _dedupe 兜底）。
    """
    scores: list[float] = []
    for counter in (left, right):
        inner = Counter({k: v for k, v in counter.items() if k not in {"^", "$"}})
        if sum(inner.values()) >= 2:
            scores.append(_entropy(inner))
    return min(scores) if scores else math.inf


def _context(page_text: str, start: int, end: int) -> str:
    lo = max(0, start - CONTEXT_PAD)
    hi = min(len(page_text), end + CONTEXT_PAD)
    snippet = page_text[lo:hi].replace("\n", " ")
    return re.sub(r"\s{2,}", " ", snippet).strip()


class GlossaryBuilder:
    """可增量喂入文档（一次上传一份，也能一次建多份的联合词表）。

    统计量是**全库**算的：一份资料的词表受其他资料影响（IDF 同理）。
    所以上传新文档后，旧文档的画像要不要重建是一个明确的产品决定 ——
    本项目的答案是"不自动重建"，见 docs/upload-service.md。
    """

    def __init__(
        self,
        *,
        min_freq: int | None = None,
        min_cohesion: float = 2.0,
        min_freedom: float = 1.0,
        max_len: int = MAX_TERM_LEN,
    ) -> None:
        self.min_freq = min_freq
        self.min_cohesion = min_cohesion
        self.min_freedom = min_freedom
        self.max_len = max_len
        self._pages: list[PageText] = []

    def add_pages(self, pages: Iterable[PageText]) -> "GlossaryBuilder":
        self._pages.extend(pages)
        return self

    # ---------------------------------------------------------------- 统计

    def _cjk_runs(self) -> list[tuple[PageText, int, str]]:
        """[(页, 段在页内的起始偏移, 段文本)]。偏移要留着，术语落点靠它。"""
        out: list[tuple[PageText, int, str]] = []
        for page in self._pages:
            for m in CJK_RUN_RE.finditer(page.text):
                out.append((page, m.start(), m.group(0)))
        return out

    def _mine_cjk(self) -> list[Term]:
        runs = self._cjk_runs()
        total_chars = sum(len(r) for _p, _s, r in runs)
        if not total_chars:
            return []

        min_freq = self.min_freq
        if min_freq is None:
            # 自适应：小资料降低门槛，否则一个候选都出不来。
            # 23528 字 → 4；1200 字 → 2。实测见文件头。
            min_freq = max(2, round(total_chars / 6000))

        freq: Counter[str] = Counter()
        left: dict[str, Counter[str]] = defaultdict(Counter)
        right: dict[str, Counter[str]] = defaultdict(Counter)
        spots: dict[str, list[tuple[PageText, int]]] = defaultdict(list)

        for page, base, run in runs:
            n_run = len(run)
            for n in range(1, self.max_len + 1):
                for i in range(n_run - n + 1):
                    gram = run[i : i + n]
                    freq[gram] += 1
                    left[gram][run[i - 1] if i > 0 else "^"] += 1
                    right[gram][run[i + n] if i + n < n_run else "$"] += 1
                    if n >= 2:
                        spots[gram].append((page, base + i))

        scored: list[Term] = []
        for gram, f in freq.items():
            if len(gram) < 2 or f < min_freq:
                continue
            if gram in GENERIC_STOPWORDS:
                continue
            if gram[0] in CJK_BOUNDARY_STOP or gram[-1] in CJK_BOUNDARY_STOP:
                continue

            # 最松的一刀：任一切分点都很紧，才算一个整体
            cohesion = min(
                math.log2(f * total_chars / (freq[gram[:i]] * freq[gram[i:]] + 1e-9) + 1e-9)
                for i in range(1, len(gram))
            )
            freedom = min(_freedom(left[gram], right[gram]), 9.99)  # 夹住 inf，否则落盘出 Infinity
            if cohesion < self.min_cohesion or freedom < self.min_freedom:
                continue

            score = cohesion * freedom * math.log2(1 + f) * (1 + 0.35 * (len(gram) - 2))
            scored.append(
                Term(
                    text=gram,
                    freq=f,
                    score=round(score, 4),
                    cohesion=round(cohesion, 4),
                    freedom=round(freedom, 4),
                    kind="cjk",
                    occurrences=self._occurrences(gram, spots[gram]),
                )
            )
        return self._dedupe(scored)

    def _occurrences(self, term: str, spots: list[tuple[PageText, int]]) -> tuple[Occurrence, ...]:
        """每页只取第一处，最多取 3 页。

        为什么不是"取前 3 次出现"：那 3 次经常落在同一页（长文档尤其明显），
        于是"出处"栏里三行写着同一个页码，等于只给了一个出处。
        按页去重后，用户能直接跳到 3 个不同的位置核对 —— 这才是出处的用处。
        """
        out: list[Occurrence] = []
        seen: set[tuple[str, int]] = set()
        for page, start in spots:
            key = (page.doc_id, page.page)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Occurrence(
                    doc_id=page.doc_id,
                    page=page.page,
                    start=start,
                    end=start + len(term),
                    context=_context(page.text, start, start + len(term)),
                )
            )
            if len(out) >= MAX_OCCURRENCES:
                break
        return tuple(out)

    def _dedupe(self, terms: list[Term]) -> list[Term]:
        """去掉"半个词"。

        判定：若候选 g 是某个更长的已保留术语 k 的子串，且 **g 的出现次数
        大部分被这些长词吃掉**（sum(freq_k) ≥ 0.8 · freq_g），g 就是碎片。

        实测案例："氏距离"(27 次) 里 22 次来自"马氏距离"(11) + "欧氏距离"(11)，
        22 ≥ 0.8×27 → 丢掉。而 "主成分"(57) 只有 11 次来自"主成分分析"(11)，
        11 < 0.8×57 → 保留，因为它确实能独立出现。

        第一版用的是"短词频次 < 2×长词频就丢"，结果 "氏距离" 27 ≥ 2×11 被留下，
        词表里同时出现"马氏距离"和"氏距离"，谁也看不懂。
        """
        terms.sort(key=lambda t: -t.score)
        kept: list[Term] = []
        for t in terms:
            absorbed = sum(k.freq for k in kept if t.text in k.text)
            if absorbed >= 0.8 * t.freq:
                continue
            kept.append(t)
        return kept

    def _mine_latin(self) -> list[Term]:
        counts: Counter[str] = Counter()
        spots: dict[str, list[tuple[PageText, int]]] = defaultdict(list)
        for page in self._pages:
            for m in LATIN_RE.finditer(page.text):
                tok = m.group(0)
                if len(tok) < 3 or tok.lower() in STOPWORDS:
                    continue
                if tok.islower() and len(tok) < 4:
                    # 全小写短词（and/the…）八成是普通英文词；缩写通常大写
                    continue
                counts[tok] += 1
                spots[tok].append((page, m.start()))

        out: list[Term] = []
        for tok, f in counts.items():
            if f < 2:
                continue
            acronym = tok.isupper() and len(tok) >= 2
            score = math.log2(1 + f) * (2.0 if acronym else 1.0)
            out.append(
                Term(
                    text=tok,
                    freq=f,
                    score=round(score, 4),
                    kind="latin",
                    occurrences=self._occurrences(tok, spots[tok]),
                )
            )
        out.sort(key=lambda t: -t.score)
        return out

    # ---------------------------------------------------------------- 对外

    def build(self, *, top_n: int = 40) -> list[Term]:
        """汉字词与拉丁词合并排序。两类分数不同量纲，不做归一化 ——
        排序只在同类内有意义，展示时按 kind 分组更诚实。
        """
        cjk = self._mine_cjk()[: top_n * 3]
        latin = self._mine_latin()[: top_n]
        merged = cjk + latin
        merged.sort(key=lambda t: -t.score)
        return merged[:top_n]


def mine_terms(pages: Iterable[PageText], *, top_n: int = 40) -> list[Term]:
    """一份/一批资料的一次性抽取。"""
    return GlossaryBuilder().add_pages(pages).build(top_n=top_n)
