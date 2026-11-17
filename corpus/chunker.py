"""M2-T3 · 切分层：页 → 块。

一句话：**块的边界决定检索的天花板**。块切碎了，答案的证据被劈成两半；
块切大了，一个块混三个主题，检索命中了也是噪音。换多好的模型都救不回来。

【实测依据，2026-11-17】

实测 1：标题必须从 PDF 书签取，不能取文本首行。
    文本首行有 7 页不是标题：
      chap05 p6/p19/p20/p21/p22 首行是 "TID"（表格的第一列列名，标题是图片）
      chap05 p15 首行是公式 "∀𝑋, 𝑌: (𝑋⊂𝑌) ⇒ 𝑠𝑌≤𝑠(𝑋)"
    按首行切会把 chap05 p19–p23 五页并成一块 3933 字的巨型块。
    书签 199/199 页齐全，格式是 "Slide N: 标题"。

实测 2：书签也有 4 条坏的，全在 chap03（p38/p39/p40/p42），只有 "Slide 38" 没有标题。
    这 4 页夹在同一个话题中间（Hunt's Algorithm 那段），所以继承上一页标题是对的。

实测 3：同一话题常被拆成"标题只差一个冒号后缀"的连续页，按标题全等合并合不上：
    p18 "Model Evaluation Metric: Accuracy"
    p19 "Model Evaluation Metric: Error rate"
    而 G02 的答案（Accuracy 公式 + Error Rate 公式）横跨这两页 —— 黄金集直接证明了
    这是一个块，不是两个块。所以分组键用"冒号前的部分"（下称 stem）。

【三种切法，用数据选，不凭感觉】

    page   每页一块（最碎，但一页就是作者切好的语义单元）
    title  连续同标题合并（按全等）
    stem   连续同 stem 合并（冒号前部分相等）

用 corpus/eval_chunking.py 量 3 种分组 × 4 个 max_chars，结论（2026-11-17 实测）：

    mode     max   块数   中位   最大   单块率  串味  带题率
    page     600   265    438    599    33%     0    69%
    page    1200   181    507   1175    78%     0    96%   ← 选它
    page    1800   178    520   1402    78%     0    98%
    title   1200   152    645   1199    56%     3    93%
    title   3000   129    645   2998    78%     1    98%
    stem    1200   148    706   1199    61%     3    95%
    stem    3000   124    706   2998    83%     1    99%

（page 1800/3000 的数字一样，因为只有 3 页超过 1200 字符；再放大上限没有收益。）

为什么和"通用最佳实践"相反：通用建议（512 token、按章节切）来自**散文文档**，
那里一章几千字，必须切。但这批语料是 PPT，平均 513 字/页（约 90 个英文词），
每一页本来就是作者切好的一个概念。此时再按标题合并，是把两三个概念焊成一块：
    stem 模式把 chap04c 的 "Issues with Naïve Bayes" 5 页焊成 2695 字，
    于是 G14（零概率/平滑）和 G15（后验为 0 的算例）掉进同样 3 个块 → 串味 3。
    page 模式的串味是 0，而单块率还更高（78% vs 61%）。

为什么 page/600 的单块率只有 33%：一页被劈成两块，答案跟着碎成两半。
这是"切太碎"的实证，不是理论担忧。

哪个方案"更好"不看块数，看这三列同时是否难看：单块率低＝切太碎，
串味高＝切太大，带题率低＝块没有上下文。任何一列崩了都不能选。

【由此得到的一条判断规则，以后换语料要重跑】
    "该不该合并"取决于语料单位，不取决于 token 数。
    PPT / 教案 / 卡片式内容：一页一块。
    连续散文 / 论文 / 书：必须按段落切，并按章节合并。
    判断手段固定：拿黄金集量 单块率 vs 串味，两个指标一起看。

【块里为什么要带 offsets】
    M4 要评估"系统引用的那一页里是否真有这句话"。如果块只记 page_start/page_end，
    块跨 4 页时就没法知道某句话来自哪一页。所以每个块额外记 spans：
    (page, start, end) —— 块文本里 [start, end) 这一段来自第 page 页。
    这样任意一个字符偏移都能反查页码，引用溯源不需要再猜。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel, ConfigDict, Field

from corpus.docs import DOCS
from corpus.parser import PageText, extract_pages

# ---------------------------------------------------------------- 参数

# 切太碎的反面证据：黄金集里 G02（Accuracy + Error Rate）、G07（定义 + 原因 + 缓解）
# 的答案都横跨 3 页，说明"一页一块"对这些题不成立。
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 200
OVERLAP_CHARS = 150

GroupMode = Literal["page", "title", "stem"]

SLIDE_PREFIX_RE = re.compile(r"^Slide\s+\d+\s*:?\s*", re.I)
TRAILING_NOISE_RE = re.compile(r"[\s…•\u2022]+$")


# ---------------------------------------------------------------- 数据


class PageSpan(BaseModel):
    """块文本里的一段来自哪一页。start/end 是块内偏移，左闭右开。"""

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @property
    def length(self) -> int:
        return self.end - self.start


class Chunk(BaseModel):
    """一个检索单元。不可变：后面 M3 索引、M4 评估都读它，谁都不该改。"""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1, description="doc#序号，稳定可排序")
    doc_id: str = Field(min_length=1)
    title: str = Field(description="来自 PDF 书签；继承或空页时可能为空")
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1, description="块覆盖的最后一页（含）")
    text: str = Field(description="块正文，页与页之间用空行分隔")
    spans: tuple[PageSpan, ...] = Field(description="块内偏移 → 页码")
    is_empty: bool = Field(description="正文为空或极短，检索层应该跳过它")

    @property
    def char_count(self) -> int:
        return len(self.text)

    def source(self) -> str:
        if self.page_start == self.page_end:
            return f"{self.doc_id}:p{self.page_start}"
        return f"{self.doc_id}:p{self.page_start}-{self.page_end}"

    def page_of(self, offset: int) -> int | None:
        """块内第 offset 个字符来自哪一页。用于 M4 引用溯源。

        落在页间分隔符上时返回 None —— 那个字符不属于任何一页的正文。
        """
        for span in self.spans:
            if span.start <= offset < span.end:
                return span.page
        return None

    def header(self) -> str:
        """给模型看的出处头。

        为什么需要它：chap05 p19–p23 是 Apriori 表格的续页，正文从 "TID" 开始，
        标题是图片所以正文里没有标题。这类块单独看不知道在讲什么。
        块边界解决不了这个问题（p19 就那点内容），但拼 prompt 时加一行出处
        就能解决：模型看到"chap05 p19 · Illustrating Apriori Principle"就懂了。
        """
        if self.title:
            return f"【{self.source()} · {self.title}】"
        return f"【{self.source()}】"

    def page_text(self, page: int) -> str:
        """取出块里属于第 page 页的那段文字。"""
        parts = [self.text[s.start : s.end] for s in self.spans if s.page == page]
        return "\n".join(parts)


# ---------------------------------------------------------------- 标题


def normalize_title(raw: str) -> str:
    """把书签标题洗成纯标题：去 "Slide N:" 前缀、去尾部省略号和项目符号。

    "Slide 38" → ""（这条书签本来就没有标题）
    "Slide 19: Measures of Node Impurity •" → "Measures of Node Impurity"
    "Slide 19: Example of Sequential Covering…" → "Example of Sequential Covering"
    """
    t = SLIDE_PREFIX_RE.sub("", raw.strip())
    t = TRAILING_NOISE_RE.sub("", t)
    return t.strip()


def title_stem(title: str) -> str:
    """分组键：冒号前的部分，前提是它至少有 2 个词。

    "Model Evaluation Metric: Accuracy" → "Model Evaluation Metric"
    "Computing Gini Index of a Single Node" → 原样（无冒号，不动）
    "Rule-Based Classifier" → 原样（冒号前只有 1 个词，说明冒号不是分级符）

    为什么要求 ≥2 个词：像 "TID: ..." 这种单字前缀不是章节名，合并过去会把
    不相干的页绑在一起。
    """
    if ":" in title:
        head = title.split(":", 1)[0].strip()
        if len(head.split()) >= 2:
            return head
    return title


def read_titles(doc_id: str, *, path: Path | None = None) -> dict[int, str]:
    """从 PDF 书签读每页标题，缺的继承上一页。

    为什么继承：chap03 p38/p39 的书签只有 "Slide 38"，而这两页夹在
    "Hunt's Algorithm: loan borrower classification problem" 这段中间。
    继承 = 承认它们属于同一话题；留空 = 把一段话切成三块。
    """
    if path is None:
        meta = DOCS.get(doc_id)
        if meta is None:
            raise ValueError(f"未知 doc_id：{doc_id!r}")
        path = meta.path()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 PDF：{path}")

    doc = pymupdf.open(path)
    try:
        by_page: dict[int, str] = {}
        for _level, raw, page in doc.get_toc():
            by_page[page] = normalize_title(raw)
        total = doc.page_count
    finally:
        doc.close()

    out: dict[int, str] = {}
    prev = ""
    for p in range(1, total + 1):
        t = by_page.get(p, "")
        if t:
            prev = t
        out[p] = t or prev
    return out


# ---------------------------------------------------------------- 分组


def group_pages(
    titles: dict[int, str], *, mode: GroupMode
) -> list[tuple[str, list[int]]]:
    """把连续页分成组，返回 [(显示用标题, [页码...])]。"""
    pages = sorted(titles)
    groups: list[tuple[str, list[int]]] = []
    prev_key: str | None = None

    for p in pages:
        title = titles[p]
        if mode == "page":
            groups.append((title, [p]))
            continue
        key = title_stem(title) if mode == "stem" else title
        key_cf = key.casefold()
        if groups and prev_key == key_cf:
            groups[-1][1].append(p)
        else:
            # 显示用标题：stem 模式显示 stem（更像章节名），title 模式显示全标题
            display = key if mode == "stem" else title
            groups.append((display, [p]))
        prev_key = key_cf
    return groups


# ---------------------------------------------------------------- 拼接与切分


def join_pages(pages: list[PageText]) -> tuple[str, tuple[PageSpan, ...]]:
    """把若干页正文拼成一段，同时记下每页在拼接结果里的偏移。

    页间用 "\\n\\n" 分隔。空页跳过：它没有正文，但页码仍由调用方记在
    page_start/page_end 里，不会丢页。
    """
    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0
    for pt in pages:
        if not pt.text:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        spans.append(PageSpan(page=pt.page, start=cursor, end=cursor + len(pt.text)))
        parts.append(pt.text)
        cursor += len(pt.text)
    return "".join(parts), tuple(spans)


def slice_spans(spans: tuple[PageSpan, ...], start: int, end: int) -> tuple[PageSpan, ...]:
    """截取 [start, end) 覆盖到的那些 span，边界裁到区间内。"""
    out: list[PageSpan] = []
    for s in spans:
        a = max(s.start, start)
        b = min(s.end, end)
        if a < b:
            out.append(PageSpan(page=s.page, start=a, end=b))
    return tuple(out)


def split_long(
    text: str,
    spans: tuple[PageSpan, ...],
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list[tuple[str, tuple[PageSpan, ...]]]:
    """把过长的拼接文本切成若干块，相邻块之间留 overlap 个字符的重叠。

    断开点优先选换行：句中切断会让"目标类"和它的定义分家。
    找不到换行（超长单行）时退回硬切。

    overlap 的作用：让后一块开头带着前一块的尾巴，语义上不至于完全没有上下文。
    它不能替代真正的句子边界，只是减少"开头就是半句话"的概率。
    """
    if max_chars <= overlap:
        raise ValueError(f"max_chars({max_chars}) 必须大于 overlap({overlap})")

    n = len(text)
    if n <= max_chars:
        return [(text, spans)]

    pieces: list[tuple[str, tuple[PageSpan, ...]]] = []
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # 从后往前找换行，但至少保留一半的长度，避免切出极短的块
            cut = text.rfind("\n", start + max_chars // 2, end)
            if cut != -1:
                end = cut
        body = text[start:end]
        pieces.append((body, slice_spans(spans, start, end)))
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        nl = text.find("\n", nxt, end)
        start = nl + 1 if nl != -1 else nxt
    return pieces


def merge_tiny(
    groups: list[tuple[str, list[int]]],
    sizes: list[int],
    *,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[tuple[str, list[int]]]:
    """把过小的组合并进相邻组，避免切出"标题 26 字正文"这种没法用的块。

    实测案例：chap04b p7「KNN Classifiers: Algorithm」正文只有 26 字符 ——
    算法本体是图片，文本层里什么都没有。单独成块它永远检索不到；
    吞进 p8 后 457 字符，成了一个能正常用的块。

    规则是**单向前吞**：正在构建的这一块如果还不到 min_chars，就继续吃下一组，
    直到够长或吃完。不回填上一组 —— 回填会让前一块无界增长，
    而且会让已经合格的块被撑大。

    第一版写的是"并入前一组，但前一组不超过自身 3 倍"，实测在 chap04b p7 上
    直接失效：p7 的正文比前一组小得多，条件不成立，于是它还是单独成块。
    """
    out: list[tuple[str, list[int]]] = []
    out_sizes: list[int] = []
    for (title, pages), size in zip(groups, sizes):
        if out and out_sizes[-1] < min_chars and out_sizes[-1] + size <= max_chars:
            out[-1] = (out[-1][0], out[-1][1] + pages)
            out_sizes[-1] += size
            continue
        out.append((title, list(pages)))
        out_sizes.append(size)
    return out


# ---------------------------------------------------------------- 对外


def build_chunks(
    doc_id: str,
    *,
    mode: GroupMode = "page",
    max_chars: int = MAX_CHUNK_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
    pages: list[PageText] | None = None,
    titles: dict[int, str] | None = None,
) -> list[Chunk]:
    """把一份 PDF 切成块。pages/titles 可注入，测试就不用真 PDF。"""
    if pages is None:
        pages = extract_pages(doc_id)
    if titles is None:
        titles = read_titles(doc_id)

    by_page = {p.page: p for p in pages}
    groups = group_pages(titles, mode=mode)

    # 先量每组字数，再决定是否合并小组合
    sizes = []
    for _title, pgs in groups:
        sizes.append(sum(len(by_page[p].text) for p in pgs if p in by_page))
    groups = merge_tiny(groups, sizes, min_chars=min_chars, max_chars=max_chars)

    chunks: list[Chunk] = []
    for title, pgs in groups:
        if mode == "page" and len(pgs) > 1:
            # 页被合并（多半是它自己太短），标题要取"贡献正文最多的那页"。
            # 实测案例：chap05 p1 是封面（"CS422 : Data Mining"，182 字），
            # 被吞进 p2（"Association Rule Mining"，579 字）。
            # 若沿用第一页标题，整块会挂上"CS422 : Data Mining"这个没用的名字。
            dominant = max(pgs, key=lambda p: len(by_page[p].text) if p in by_page else 0)
            title = titles.get(dominant, title)
        text, spans = join_pages([by_page[p] for p in pgs if p in by_page])
        pieces = split_long(text, spans, max_chars=max_chars, overlap=overlap)
        for body, body_spans in pieces:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{len(chunks):03d}",
                    doc_id=doc_id,
                    title=title,
                    page_start=min(pgs),
                    page_end=max(pgs),
                    text=body,
                    spans=body_spans,
                    is_empty=len(body.strip()) < min_chars,
                )
            )
    return chunks


def build_all(**kwargs) -> list[Chunk]:
    """全部 5 份讲义切成一个列表，顺序与 DOCS 一致。"""
    out: list[Chunk] = []
    for doc_id in DOCS:
        out.extend(build_chunks(doc_id, **kwargs))
    return out


def corpus_available() -> bool:
    return all(meta.path().is_file() for meta in DOCS.values())
