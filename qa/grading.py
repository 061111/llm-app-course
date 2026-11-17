"""M4-T3 · 评分：怎么判断一个答案对不对。

【为什么不能只看"像不像标准答案"】
  黄金集里 4 道计算题的答案是一个数字，5 道概念题是一段话。
  用字符串相似度判分，会把"答对了但换了说法"判成错，
  也会把"抄了一遍标准答案但数字错"判成对。所以这里分两条腿：

  腿 1（机械）：数值核对。计算题的答案里有不可含糊的数字（0.278 / 2/3），
      模型答案里出现即算命中。这条便宜、可复现、零主观，但只覆盖有数字的题。
  腿 2（模型判卷）：给判卷模型看问题、标准答案、系统答案，要它判
      correct / partial / wrong / uncovered。覆盖全部题目，但判卷本身有随机性。

  两条腿不一致时**不强行统一**，而是把分歧列出来：分歧本身就是信息
  （要么机械规则太死，要么判卷模型太松）。第 5 节会展示一个真实分歧。

【判卷 prompt 的三条约束】
  1. 只看事实，不看措辞。否则"答得简洁"会被判错。
  2. 数字必须对上才算对。0.65 与 0.86 在这门课里是两个不同结论。
  3. 必须能从标准答案推出。防止判卷模型用自己的知识"宽容"了系统答案。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

JUDGE_PROMPT = """你是严格的判卷人。下面是一道题、教材口径的标准答案、以及学生（AI 系统）的回答。

判定规则：
1. **只看事实**，不看措辞、不看格式、不看长度。说得啰嗦但事实对，判 correct。
2. 答案里的**数字必须与标准答案一致**才算对。0.65 和 0.86 是两个不同结论，不能混。
3. 只能依据标准答案判定，不要用你自己的知识替学生补充或宽容。
4. 四档：
   correct   —— 关键事实全部正确，可以缺细枝末节
   partial   —— 部分正确，漏了关键点或有一处事实错误
   wrong     —— 主要结论错误
   uncovered —— 学生回答"讲义未覆盖"（这是检索没覆盖，不是答错，单独统计）
5. 只输出 JSON：{{"verdict": "...", "reason": "一句话", "missing": ["漏掉的点"]}}

问题：{question}

标准答案：{gold}

学生回答：{answer}
"""


class Verdict(BaseModel):
    """判卷结果。verdict 用 Literal，模型乱给值会直接校验失败并触发重问。"""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["correct", "partial", "wrong", "uncovered"]
    reason: str = Field(description="一句话说明判定理由")
    missing: list[str] = Field(default_factory=list, description="漏掉或答错的关键点")


def numeric_hits(
    answer_text: str, expected: tuple[tuple[str, ...], ...]
) -> tuple[int, tuple[str, ...]]:
    """机械核对：每组"预期值"里有没有任意一种写法出现在答案里。

    expected 是**组**的元组，每组内部是等价写法。例如 G16 的 confidence：
    标准答案是 2/3 ≈ 0.67，模型写成 "2/3" 或 "0.67" 都对，所以两组是
    ("2/3", "0.67")。如果用单一字符串匹配，写成 2/3 的答案会被判错 ——
    那是规则的问题，不是答案的问题。

    不做 0.4 / 0.40 这类数值归一化：归一化容易让 0.6 匹配上 0.65，
    边界情况反而更难解释。当前只用于数字形态固定的几道题。
    """
    missing: list[str] = []
    for group in expected:
        if not any(alt in answer_text for alt in group):
            missing.append(" 或 ".join(group))
    return len(expected) - len(missing), tuple(missing)
