"""M4 · 端到端评估：18 道题真跑一遍，看答对几道。

这是整个项目的验收口。前面三层（切分 / 检索 / 生成）任何一层退步，
都会在这里掉分 —— 前提是评估本身可信，所以这个文件里做的事比"跑一遍打印"多：

  1. 每题都跑完整链路（翻译 → BM25 → 拼 prompt → 调模型 → 解析引用）。
  2. 判分双轨：数值机械核对 + 模型判卷，两者独立记录。
  3. 记录**引用越界**（答案标注的出处不在检索结果里）—— 这是 RAG 特有的错误。
  4. 全部原始结果落盘到 artifacts/qa_eval.json，报告可复算。

【成本预期，跑之前就该知道】
  18 题 × (检索 0 次调用，翻译命中缓存 + 回答 1 次 + 判卷 1 次) = 36 次调用。
  实测约 ¥0.2、约 2 分钟。这就是"能反复跑"的规模，再大就要考虑抽样。

用法：
    python -m qa.eval_qa            # top_k=5
    python -m qa.eval_qa 8          # top_k=8，用来验证"检索修好了答案会不会修好"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from corpus.gold import GOLD
from llmkit.client import ask_structured
from llmkit.types import AskOptions
from qa.answer import Answer, answer_question
from qa.grading import JUDGE_PROMPT, Verdict, numeric_hits
from retrieval.query import Translator
from retrieval.search import Searcher

RESULTS_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "qa_eval.json"

# 机械核对用的预期值，每组是"等价写法"。只收**数值形态固定**的题：
#   G04 三个 Gini 值中的两个（(0,6) 那题 Gini=0 太容易蒙，不列入）
#   G05 熵值 0.65 / 0.92（这道是陷阱题：按错误公式会算成 0.86）
#   G16 2/5=0.4 与 2/3≈0.67
NUMERIC: dict[str, tuple[tuple[str, ...], ...]] = {
    "G04": (("0.278",), ("0.444",)),
    "G05": (("0.65",), ("0.92",)),
    "G16": (("0.4", "2/5"), ("0.67", "2/3", "0.666")),
}


class ItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gold_id: str
    kind: str
    verdict: str = Field(description="correct / partial / wrong / uncovered / judge_failed")
    reason: str
    judge_error: str = Field(default="", description="判卷失败时的原始错误")
    missing: list[str] = Field(default_factory=list)
    numeric_ok: int | None = Field(default=None, description="机械核对命中组数；None=该题无数字")
    numeric_total: int | None = None
    numeric_missing: list[str] = Field(default_factory=list)
    cited: list[str] = Field(default_factory=list, description="答案里标注的出处")
    unsupported: list[str] = Field(default_factory=list, description="引用越界")
    retrieved: list[str] = Field(default_factory=list)
    answer: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_cny: float = 0.0


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int
    items: list[ItemResult]
    judge_calls: int = 0

    def count(self, verdict: str) -> int:
        return sum(1 for i in self.items if i.verdict == verdict)

    @property
    def score(self) -> float:
        """correct 记 1 分，partial 记 0.5。uncovered 和 wrong 都是 0。

        为什么 partial 给 0.5 而不是 0：18 道题里 partial 通常意味着
        "答对一半"（如只说了原因没说缓解手段），把它记为 0 会让分数
        对长答案题目严重低估，也无法体现改动是"变好一点"还是"没变"。
        这个折算系数是主观的，读分数时必须一起看四档分布，不能只看 0.78 这个数。
        """
        return (self.count("correct") + 0.5 * self.count("partial")) / len(self.items)


def judge(question: str, gold: str, answer: str) -> tuple[Verdict | None, str]:
    """让模型判卷。返回 (结论, 错误信息)。

    **实测教训**：第一次跑 18 题时，第 6 题的判卷返回了空内容，
    LLMParseError 直接把整轮评估炸掉，前 5 题的结果一起丢了。
    单个 API 抖动不该毁掉一次 2 分钟的评估，所以这里：
      1. 失败重试一次；
      2. 仍失败就返回 None + 错误信息，由调用方记为 judge_failed 继续跑。
    这不是"把错误藏起来"：judge_failed 会单独统计并在报告里点名，
    它既不算对也不算错。

    判卷也走 ask_structured：Verdict 的 verdict 字段是 Literal，
    模型返回 "excellent" 这种值会校验失败并触发重问。
    """
    last = ""
    for _attempt in (1, 2):
        try:
            return (
                ask_structured(
                    JUDGE_PROMPT.format(question=question, gold=gold, answer=answer),
                    Verdict,
                    options=AskOptions(temperature=0.0, max_tokens=600),
                ),
                "",
            )
        except Exception as exc:  # noqa: BLE001 - 任何失败都只影响这一题
            last = f"{type(exc).__name__}: {exc}"
    return None, last


def run(top_k: int = 5) -> Run:
    translator = Translator()
    searcher = Searcher.from_store(translator=translator)
    items: list[ItemResult] = []

    for g in GOLD:
        ans: Answer = answer_question(g.question, searcher=searcher, top_k=top_k)
        verdict, err = judge(g.question, g.answer, ans.text)
        expect = NUMERIC.get(g.id)
        ok, total, miss = (None, None, [])
        if expect:
            ok, miss = numeric_hits(ans.text, expect)
            total = len(expect)
        items.append(
            ItemResult(
                gold_id=g.id,
                kind=g.kind.value,
                verdict=verdict.verdict if verdict else "judge_failed",
                reason=verdict.reason if verdict else "判卷调用失败，本题不计入对错",
                judge_error=err,
                missing=verdict.missing if verdict else [],
                numeric_ok=ok,
                numeric_total=total,
                numeric_missing=list(miss),
                cited=list(ans.cited_sources),
                unsupported=list(ans.unsupported),
                retrieved=list(ans.retrieved),
                answer=ans.text,
                latency_ms=ans.latency_ms,
                prompt_tokens=ans.usage.prompt_tokens,
                completion_tokens=ans.usage.completion_tokens,
                cost_cny=ans.usage.cost_cny,
            )
        )
        print(
            f"  {g.id} {(verdict.verdict if verdict else 'judge_failed'):12} "
            f"{ans.cited_sources} {(verdict.reason if verdict else err)[:36]}",
            flush=True,
        )

    return Run(top_k=top_k, items=items)


def format_run(r: Run) -> str:
    n = len(r.items)
    lines = [
        f"=== M4 端到端评估  top_k={r.top_k}  {n} 题 ===",
        f"  correct   {r.count('correct'):>2}/{n}",
        f"  partial   {r.count('partial'):>2}/{n}",
        f"  wrong     {r.count('wrong'):>2}/{n}",
        f"  uncovered {r.count('uncovered'):>2}/{n}",
        f"  判卷失败  {r.count('judge_failed'):>2}/{n}",
        f"  折算得分  {r.score:.2f}   (correct 记 1，partial 记 0.5，判卷失败记 0)",
        "",
        f"  引用越界  {sum(1 for i in r.items if i.unsupported)} 题",
        f"  无引用    {sum(1 for i in r.items if not i.cited)} 题",
        f"  平均耗时  {sum(i.latency_ms for i in r.items) / n / 1000:.2f} 秒",
        f"  总花费    ¥{sum(i.cost_cny for i in r.items):.4f}（回答部分；判卷另有花费）",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    top_k = int(argv[0]) if argv else 5

    result = run(top_k=top_k)
    print()
    print(format_run(result))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump()
    payload["generated_at"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n原始结果 → {RESULTS_PATH}")

    print()
    print("=== 逐题 ===")
    for i in result.items:
        num = ""
        if i.numeric_total is not None:
            num = f"  数值 {i.numeric_ok}/{i.numeric_total}"
            if i.numeric_missing:
                num += f" 缺 {i.numeric_missing}"
        bad = f"  越界引用={i.unsupported}" if i.unsupported else ""
        print(f"  {i.gold_id} {i.kind:5} {i.verdict:9}{num}{bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
