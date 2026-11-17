"""M4-T2 · 回答：检索 → 拼 prompt → 调用 → 解析引用。

【这个模块最值得看的一件事：引用能不能被机械核对】
  模型很会"顺手编一个出处"。所以这里不止把文字返回，
  还把答案里所有【doc_id:pN】抓出来，逐个检查它是否真的在**本次检索到的块**里。
  不在的都进 suspicious —— 这叫"引用越界"。

  为什么这比"看起来对不对"重要：答案是自然语言，人审 18 条要半小时；
  引用越界是机械判定，一秒出结果，而且它抓的正好是最危险的一类错误
  （答案内容可能对，但出处是编的，等于不可溯源）。
  注意：引用合法 ≠ 内容正确。它只是必要条件，不是充分条件。

【为什么 temperature=0】
  评估要可复现。0.2 的默认温度下，同一个问题两次答案可能差一个数字，
  而黄金集里 4 道计算题的判分完全取决于那几个数字。
  代价是答案更呆板 —— 这是评估场景该付的代价，聊天场景不该。
"""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from llmkit.client import ask
from llmkit.types import AskOptions, Usage
from qa.prompt import SYSTEM, build_prompt
from retrieval.search import SearchResult, Searcher

# 【chap03:p70】或【chap03:p12-13】。只认这种格式：它是 prompt 里规定死的，
# 认死了才能机械核对；放宽成"第 70 页"之类，解析会立刻变得不可靠。
CITE_RE = re.compile(r"【([A-Za-z0-9_]+):p(\d+)(?:-(\d+))?】")


class CitationRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    raw: str = Field(description="答案里的原始写法，便于回指")

    def source(self) -> str:
        if self.page_start == self.page_end:
            return f"{self.doc_id}:p{self.page_start}"
        return f"{self.doc_id}:p{self.page_start}-{self.page_end}"


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    text: str = Field(description="模型原文回答")
    citations: tuple[CitationRef, ...] = Field(description="答案里出现的所有出处")
    retrieved: tuple[str, ...] = Field(description="本次检索到的块 source，按排名")
    unsupported: tuple[str, ...] = Field(
        description="引用了但不在本次检索结果里的出处 —— 疑似编造"
    )
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = Field(default=0.0, ge=0.0)
    trace_id: str = ""

    @property
    def cited_sources(self) -> tuple[str, ...]:
        return tuple(c.source() for c in self.citations)

    @property
    def claims_uncovered(self) -> bool:
        """模型是否自己承认"讲义未覆盖"。评估时要和"答错"分开统计。"""
        return "讲义未覆盖" in self.text


def parse_citations(text: str) -> tuple[CitationRef, ...]:
    """抓出答案里的出处标注，按出现顺序去重。"""
    out: list[CitationRef] = []
    seen: set[str] = set()
    for m in CITE_RE.finditer(text):
        doc_id, start, end = m.group(1), int(m.group(2)), m.group(3)
        start_page = start
        end_page = int(end) if end else start
        if end_page < start_page:  # 模型写反了页序，纠正而不是崩
            start_page, end_page = end_page, start_page
        ref = CitationRef(
            doc_id=doc_id, page_start=start_page, page_end=end_page, raw=m.group(0)
        )
        if ref.source() in seen:
            continue
        seen.add(ref.source())
        out.append(ref)
    return tuple(out)


def unsupported_citations(
    citations: tuple[CitationRef, ...], results: list[SearchResult]
) -> tuple[str, ...]:
    """哪些引用指向了本次没检索到的页。

    判定**逐页**做：区间 [start, end] 里每一页都必须被某个检索到的块覆盖。
    不能要求"单个块覆盖整个区间" —— 实测反例：模型写【chap03:p20-21】，
    而 p20 和 p21 是两个独立的块，两个都在检索结果里，此时区间引用完全合法，
    按"单块覆盖"判会误报。

    另一个实测反例（真错误）：模型写【chap04a:p27-28】，检索到的只有 p27，
    p28 那一块没进 top-5 —— 这是真越界，逐页判定会正确抓出来。

    为什么值得这么较真：这是整个 M4 里唯一能机械判定的正确性维度。
    它误报，你就得人工复核每一条；它漏报，编造的出处就混过去了。
    """
    covered: dict[str, set[int]] = {}
    for r in results:
        ch = r.chunk
        covered.setdefault(ch.doc_id, set()).update(
            range(ch.page_start, ch.page_end + 1)
        )

    out: list[str] = []
    for c in citations:
        pages = set(range(c.page_start, c.page_end + 1))
        if not pages <= covered.get(c.doc_id, set()):
            out.append(c.source())
    return tuple(out)


def answer_question(
    question: str,
    *,
    searcher: Searcher,
    top_k: int = 5,
    settings=None,
    client=None,
    max_chars: int = 6000,
) -> Answer:
    """完整走一遍：检索 → 拼 prompt → 调用 → 解析。

    settings / client 是给测试的注入点（和 M1 的 ask 一致）：
    不给就真调用，给了就能用假客户端跑到同一个校验路径。
    """
    results = searcher.search(question, top_k=top_k, translate=True)
    prompt = build_prompt(question, results, max_chars=max_chars)
    result = ask(
        prompt,
        options=AskOptions(temperature=0.0, max_tokens=800, system=SYSTEM),
        settings=settings,
        client=client,
    )
    cites = parse_citations(result.text)
    return Answer(
        question=question,
        text=result.text,
        citations=cites,
        retrieved=tuple(r.chunk.source() for r in results),
        unsupported=unsupported_citations(cites, results),
        usage=result.usage,
        latency_ms=result.latency_ms,
        trace_id=result.trace_id,
    )
