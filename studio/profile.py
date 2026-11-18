"""M5-T5 · 领域画像：把"候选术语"变成"这份资料的用词规范"，并且不许编造。

【核心问题：让模型写东西，怎样才能证明它没编】
  模型的输出是自然语言，人审一份术语表要十几分钟，而且看不出来
  "共同度"这个词到底在不在原文里。所以这里不靠"提示词写得严"，靠**机械校验**：

  | 模型能做什么 | 校验方式 | 不合格怎么办 |
  |---|---|---|
  | 从候选里挑术语 | 逐字在原文里查找 | 找不到 → 丢进 rejected，计入编造数 |
  | 给术语写一句说明 | 说明里的每个数字必须在该术语所在页出现 | 找不到 → 丢弃说明，保留术语 |
  | 写"用户可能怎么问" | 问题里引用的术语必须已在上面通过校验 | 有未通过的就丢弃这条问题 |
  | 写领域名 / 一句话概括 | 无法机械校验（这是概括，不是事实） | 限长 + 标注来源，不当作事实 |

  **页码一律由系统重算**：prompt 里根本没让模型输出页码，输出结构里也没有这个字段。
  术语的出处取"原文里实际出现的页"，模型没有机会编一个页码。

  这套设计的代价是模型只能"选"，不能"造"。看起来限制了能力，但它换来的是一个
  可以量化验收的指标：**编造术语数**。评测里这个数必须为 0，否则就是不合格。

【为什么画像要落盘】
  一次画像约 1 次模型调用、几秒、几百 token。问答时每次重算既慢又不稳定
  （temperature=0 也只是"尽量"可复现）。落盘到 artifacts/profiles/{doc_id}.json，
  回答问题时直接读 —— 于是"术语"成为**可审计的配置**，而不是每次现编的文本。

【降级：模型不可用时仍然可用】
  没有密钥 / 调用失败 → 退回纯统计画像：术语表照样带页码和上下文，
  只是没有模型筛选和"可能怎么问"。`source` 字段标明是 llm 还是 statistical，
  UI 上显示不同的角标 —— 用户有权知道这份词表有没有经过模型。
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from corpus.parser import PageText
from llmkit.client import ask
from llmkit.parsing import parse_with_repair, repair_prompt
from llmkit.types import AskOptions, Usage
from retrieval.bm25 import tokenize
from studio.errors import LlmUnavailable
from studio.glossary import Term, snippet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE_DIR = PROJECT_ROOT / "artifacts" / "profiles"

MAX_TERMS = 24
MAX_NOTE_CHARS = 40
MAX_PHRASINGS = 6

# 单次画像的输出预算。**这个数字是被实测逼出来的**（2026-11-18）：
# 本机配置的 deepseek-flash 是推理模型，选术语这种小任务也会先花掉几千 token
# 推理。max_tokens=1200 时实测 finish_reason=length、reasoning_tokens=1200、
# 正文长度为 0；max_tokens=8000 时 reasoning≈7057、正文 1140、正常收尾。
# 所以预算必须给推理留出空间，否则拿到的是"空内容"而不是"错内容"。
MAX_OUTPUT_TOKENS = 8000

SYSTEM = "你是资料整理助手。你只整理用户给的候选词，不引入任何外部知识。"

PROMPT = """下面是一份资料的候选术语表。每个候选都是**从原文里数出来的**，后面跟着出现次数和一段原文片段。

=== 候选术语 ===
{candidates}

=== 任务 ===
1. 从中挑出 12~20 个真正属于这份资料领域的术语（太通用的词不要，比如"结果""实际"）。
2. 每个术语写一句不超过 25 字的说明，**只能根据上面给出的原文片段**，不要补充外部知识。
3. 写 3~6 个"用户可能会怎么问这份资料"的问题，每个问题标注它涉及的术语。
4. 用一句话概括这份资料是什么（domain ≤ 16 字，summary ≤ 60 字）。

=== 硬性规则（违反会被系统丢弃并计数）===
- 术语必须从候选里**逐字复制**，不要改写、不要合并同义词、不要新造。
- 说明里如果出现数字，必须是原文片段里出现过的数字。
- 不同候选是同一个概念时，选出现次数多的那个，不要自己拼一个新词。

只输出 JSON：
{{"domain": "...", "summary": "...",
  "terms": [{{"term": "...", "note": "..."}}],
  "phrasings": [{{"question": "...", "terms": ["..."]}}]}}
"""

NUM_RE = re.compile(r"\d+(?:\.\d+)?")


# ---------------------------------------------------------------- LLM 的输出形状


class LlmTerm(BaseModel):
    model_config = ConfigDict(extra="ignore")

    term: str = ""
    note: str = ""


class LlmPhrasing(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = ""
    terms: list[str] = Field(default_factory=list)


class LlmProfile(BaseModel):
    """模型原始输出。字段全给默认值：模型少给一个键不该让整轮失败 ——
    缺的字段会被校验判成"缺失"，结果比抛异常更有用（能计入 rejected）。
    """

    model_config = ConfigDict(extra="ignore")

    domain: str = ""
    summary: str = ""
    terms: list[LlmTerm] = Field(default_factory=list)
    phrasings: list[LlmPhrasing] = Field(default_factory=list)


# ---------------------------------------------------------------- 落盘形状


class ProfileTerm(BaseModel):
    model_config = ConfigDict(frozen=True)

    term: str
    note: str = Field(default="", description="模型写的一句说明；未通过数字校验时为空")
    pages: tuple[str, ...] = Field(description="系统从原文重算的出处，形如 u001:p3")
    evidence: str = Field(description="术语在原页的上下文片段")
    freq: int = Field(default=0, ge=0)
    kind: str = "cjk"


class Phrasing(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    terms: tuple[str, ...]


class DomainProfile(BaseModel):
    """一份（或一批）资料的领域画像。这是"补充项目内容"的实际载体。"""

    model_config = ConfigDict(frozen=True)

    doc_ids: tuple[str, ...]
    domain: str = Field(description="领域名，≤16 字")
    summary: str = Field(description="一句话概括，≤60 字")
    terms: tuple[ProfileTerm, ...]
    phrasings: tuple[Phrasing, ...] = ()
    source: Literal["llm", "statistical"] = "llm"
    fallback_reason: str = Field(
        default="", description="降级原因。非空表示模型这一步失败了，必须让用户看见"
    )
    rejected: tuple[str, ...] = Field(
        default=(), description="模型给出但原文里找不到的词 —— 编造计数"
    )
    notes_dropped: tuple[str, ...] = Field(
        default=(), description="说明里出现原文没有的数字，被丢弃"
    )
    built_at: str = ""
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0

    @property
    def term_texts(self) -> tuple[str, ...]:
        return tuple(t.term for t in self.terms)

    def prompt_block(self, *, limit: int = 12) -> str:
        """给问答 prompt 用的领域段落。

        只放**已通过校验**的术语，且只放术语本身不放大段说明 —— 说明是概括，
        放进 prompt 会被模型当成事实继续引用，风险大于收益。
        """
        if not self.terms:
            return ""
        head = "、".join(self.term_texts[:limit])
        return (
            f"资料领域：{self.domain}。\n"
            f"资料里的常用术语：{head}。\n"
            "回答时优先使用这些术语本身，不要换成同义词或口语说法。"
        )

    def expansion_terms(self, question: str, *, threshold: float = 0.45) -> tuple[str, ...]:
        """领域措辞扩展：问题与"资料里可能怎么问"高度重合时，借它的术语去检索。

        为什么需要：中文资料里同一概念常有多种说法。资料写"协差阵"，用户问
        "协方差阵"—— bigram 只重合"方差"两个字，排名会掉。而画像里的 phrasings
        正是"资料自己的说法"，用它补检索词是最贴近原文的扩展方式。

        为什么阈值取 0.45：两侧都是中文 bigram 集合，同一个问题的措辞变体
        Jaccard 通常在 0.5~0.8；语义相近但话题不同的两句在 0.1~0.3。
        0.45 落在两个簇之间。**这条阈值只在中文资料上验证过**，
        英文资料走翻译路径，不会进这个函数。
        """
        if not self.phrasings:
            return ()
        q = set(tokenize(question))
        if not q:
            return ()
        best: Phrasing | None = None
        best_score = 0.0
        for ph in self.phrasings:
            p = set(tokenize(ph.question))
            if not p:
                continue
            score = len(q & p) / len(q | p)
            if score > best_score:
                best, best_score = ph, score
        if best is None or best_score < threshold:
            return ()
        return best.terms


# ---------------------------------------------------------------- 校验


def _number_ok(note: str, page_texts: list[str]) -> bool:
    """说明里的每个数字，都要在该术语出现的页里找得到。

    这一条抓的是模型最爱犯的错："共同度一般大于 0.7" —— 0.7 是它凭印象写的，
    原文根本没这个数。丢掉的只是这句话，术语本身保留，因为术语通过了逐字校验。
    """
    for num in NUM_RE.findall(note):
        if not any(num in text for text in page_texts):
            return False
    return True


def _pages_of(term: str, pages: list[PageText], *, limit: int = 3) -> tuple[str, ...]:
    """在原文里找出这个术语出现在哪些页。**出处的唯一来源**。"""
    out: list[str] = []
    for p in pages:
        if term in p.text:
            out.append(f"{p.doc_id}:p{p.page}")
        if len(out) >= limit:
            break
    return tuple(out)


def _evidence_of(term: str, pages: list[PageText]) -> str:
    """术语在原文里的一次出现，按"整行（表格一格）+ 表头前缀"取，不按字数硬切。"""
    for p in pages:
        idx = p.text.find(term)
        if idx >= 0:
            return snippet(p.text, idx, idx + len(term))
    return ""


def verify(
    raw: LlmProfile,
    *,
    pages: list[PageText],
    candidates: list[Term],
    doc_ids: tuple[str, ...],
    usage: Usage | None = None,
    latency_ms: float = 0.0,
) -> DomainProfile:
    """把模型输出压成"只含原文事实"的画像。所有丢弃都记录下来，不静默。"""
    cand_by_text = {c.text: c for c in candidates}
    full_text = "\n".join(p.text for p in pages)

    kept: list[ProfileTerm] = []
    rejected: list[str] = []
    notes_dropped: list[str] = []
    seen: set[str] = set()

    for item in raw.terms:
        term = item.term.strip()
        if not term or term in seen:
            continue
        # 唯一一条硬校验：这个词必须真的在资料里出现
        if term not in full_text:
            rejected.append(term)
            continue
        seen.add(term)
        page_refs = _pages_of(term, pages)
        if not page_refs:  # 理论上不可达（full_text 里有就一定有页），防御性保留
            rejected.append(term)
            continue

        cand = cand_by_text.get(term)
        note = (item.note or "").strip()[:MAX_NOTE_CHARS]
        if note:
            page_texts = [
                p.text for p in pages if f"{p.doc_id}:p{p.page}" in page_refs
            ] or [full_text]
            if not _number_ok(note, page_texts):
                notes_dropped.append(term)
                note = ""

        kept.append(
            ProfileTerm(
                term=term,
                note=note,
                pages=page_refs,
                evidence=_evidence_of(term, pages),
                freq=cand.freq if cand else 1,
                kind=cand.kind if cand else "cjk",
            )
        )
        if len(kept) >= MAX_TERMS:
            break

    kept_terms = {t.term for t in kept}
    phrasings: list[Phrasing] = []
    for ph in raw.phrasings:
        q = ph.question.strip()
        if not q or len(q) > 60:
            continue
        refs = tuple(t.strip() for t in ph.terms if t.strip())
        # 问题里引用的术语必须全部通过校验，否则这条问题是"编出来的问题"
        if not refs or not set(refs) <= kept_terms:
            continue
        phrasings.append(Phrasing(question=q, terms=refs))
        if len(phrasings) >= MAX_PHRASINGS:
            break

    domain = raw.domain.strip()[:16] or "未命名领域"
    summary = raw.summary.strip()[:60]

    return DomainProfile(
        doc_ids=doc_ids,
        domain=domain,
        summary=summary,
        terms=tuple(kept),
        phrasings=tuple(phrasings),
        source="llm",
        rejected=tuple(rejected),
        notes_dropped=tuple(notes_dropped),
        built_at=datetime.now().isoformat(timespec="seconds"),
        usage=usage or Usage(),
        latency_ms=round(latency_ms, 1),
    )


# ---------------------------------------------------------------- 构建


def statistical_profile(
    *,
    pages: list[PageText],
    terms: list[Term],
    doc_ids: tuple[str, ...],
    topic_hint: str = "",
    reason: str = "",
) -> DomainProfile:
    """不调模型的降级画像。

    不编领域名：domain 用传入的 topic_hint（通常是文件名）或"未命名领域"；
    summary 只陈述可数的东西（页数、字数、候选数），不做概括 ——
    概括是模型的活，没有模型就不硬凑。

    `reason` 记录**为什么降级**。第一版没有这个字段，结果是模型因为
    token 预算被推理吃光而返回空内容时，界面上只显示"纯统计画像"，
    看起来像设计选择，其实是失败 —— 这种静默降级比报错更坏。
    """
    chars = sum(len(p.text) for p in pages)
    return DomainProfile(
        doc_ids=doc_ids,
        domain=(topic_hint.strip()[:16] or "未命名领域"),
        summary=f"共 {len(pages)} 页 / {chars} 字，统计得到 {len(terms)} 个候选术语（未经模型筛选）",
        terms=tuple(
            ProfileTerm(
                term=t.text,
                note="",
                pages=t.pages() or _pages_of(t.text, pages),
                evidence=t.occurrences[0].context if t.occurrences else _evidence_of(t.text, pages),
                freq=t.freq,
                kind=t.kind,
            )
            for t in terms[:MAX_TERMS]
        ),
        source="statistical",
        fallback_reason=reason,
        built_at=datetime.now().isoformat(timespec="seconds"),
    )


def build_candidates_block(terms: list[Term], *, limit: int = 28) -> str:
    """候选块：术语 + 次数 + 一句原文。原文是让模型"有据可依"的关键。"""
    lines: list[str] = []
    for t in terms[:limit]:
        ctx = t.occurrences[0].context if t.occurrences else ""
        lines.append(f"- {t.text}（{t.freq} 次）：…{ctx}…")
    return "\n".join(lines)


def build_profile(
    *,
    pages: list[PageText],
    terms: list[Term],
    doc_ids: tuple[str, ...],
    llm: Callable[[str], str] | None = None,
    settings=None,
    client=None,
    allow_fallback: bool = True,
    topic_hint: str = "",
) -> DomainProfile:
    """生成画像。llm 可以是任意 `prompt -> 原始文本` 函数（测试注入点）。

    失败策略：调用/解析失败时，allow_fallback=True 就退回统计画像并继续，
    否则抛出。默认 True —— 上传资料的主流程不该因为模型挂了而不可用。
    """
    if not terms:
        return statistical_profile(
            pages=pages, terms=[], doc_ids=doc_ids, topic_hint=topic_hint,
            reason="没有候选术语（资料太短或汉字太少），不调模型",
        )

    prompt = PROMPT.format(candidates=build_candidates_block(terms))
    try:
        if llm is None:
            result = ask(
                prompt,
                options=AskOptions(
                    temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS, system=SYSTEM
                ),
                settings=settings,
                client=client,
            )
            raw_text, usage, latency = result.text, result.usage, result.latency_ms
            if not raw_text.strip():
                raise LlmUnavailable(
                    f"模型返回空内容（输出 {usage.completion_tokens} token，"
                    f"可能是推理占满了 max_tokens={MAX_OUTPUT_TOKENS}）"
                )

            def _reask(error_text: str) -> str:
                return ask(
                    repair_prompt(prompt, raw_text, error_text),
                    options=AskOptions(
                        temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS, system=SYSTEM
                    ),
                    settings=settings,
                    client=client,
                ).text
        else:
            raw_text, usage, latency = llm(prompt), Usage(), 0.0
            if not raw_text.strip():
                raise LlmUnavailable("模型返回空内容")

            def _reask(error_text: str) -> str:
                return llm(repair_prompt(prompt, raw_text, error_text))

        parsed = parse_with_repair(raw_text, LlmProfile, _reask)
        return verify(
            parsed,
            pages=pages,
            candidates=terms,
            doc_ids=doc_ids,
            usage=usage,
            latency_ms=latency,
        )
    except Exception as exc:  # noqa: BLE001 - 兜底见函数 docstring
        if not allow_fallback:
            raise
        return statistical_profile(
            pages=pages,
            terms=terms,
            doc_ids=doc_ids,
            topic_hint=topic_hint,
            reason=f"{type(exc).__name__}: {exc}"[:200],
        )


# ---------------------------------------------------------------- 存取


def profile_path(doc_id: str, directory: Path | None = None) -> Path:
    d = Path(directory) if directory else DEFAULT_PROFILE_DIR
    return d / f"{doc_id}.json"


def save_profile(profile: DomainProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_profile(path: Path) -> DomainProfile | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return DomainProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 坏画像当作没有，重新生成即可
        return None
