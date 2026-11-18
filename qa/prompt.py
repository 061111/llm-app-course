"""M4-T1 · 拼 prompt：把检索到的块变成模型能读的上下文。

这一层看着只是字符串拼接，其实是整个 RAG 里最影响答案质量的地方，
因为模型只能看到你给它的东西。三个具体决定：

【决定 1：每个块前面加一行出处头】
    【chap05:p20 · Illustrating Apriori Principle】
    TID Items ...
  为什么必要：chap05 p19–p23 是 Apriori 表格的续页，正文直接从 "TID" 开始，
  标题是图片所以正文里没有标题。不加这行，模型看到的就是一张没有表头的表。
  加了这行，它知道这是"说明 Apriori 原理的算例"。
  （这正是 M2 里 Chunk.header() 存在的理由，现在用上了。）

【决定 2：显式给出"引用格式"并要求复制粘贴】
  要求写成【chap03:p70】这种形式，而不是"请注明出处"。
  理由：可解析。M4 要自动检查"引用的页是否真的支持这句话"，
  自由格式的"出处见第 70 页"没法机械核对。

【决定 3：允许并强制"讲义未覆盖"】
  不写这一条，模型遇到检索没覆盖的问题时会用训练知识补，
  于是你会得到"看起来对但没依据"的答案 —— 那比答错更危险，
  因为它没法被黄金集抓出来。这条是 RAG 可信度的底线。

【为什么把片段顺序打乱成"按分数降序"而不是"按页码序"】
  实测里同一道题的 top-5 常常跨两三个文档。按分数排能让最相关的
  片段离问题最近（模型对上下文首尾更敏感），代价是页码不连续。
  两者都试过一次后选了按分数 —— 这个选择没有强证据，属于工程偏好，
  记录在此以免以后误以为它是实测结论。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from retrieval.search import SearchResult

if TYPE_CHECKING:  # 只为类型检查；运行时导入会形成 qa → studio → qa 的环
    from studio.profile import DomainProfile

SYSTEM = (
    "你是数据挖掘课程的助教，只根据用户提供的讲义片段回答。"
    "你不使用任何外部知识，也不猜测。"
)

INSTRUCTIONS = """你是助教。请**只**根据下面的讲义片段回答问题。

规则：
1. 只使用片段里的内容。片段里没有的信息，直接回答"讲义未覆盖"，不要补充你的先验知识。
2. 每个结论后面标注出处，格式**必须**是【doc_id:p页码】，例如【chap03:p70】。
   页码用片段头里给出的那个；跨页时写成【chap03:p12-13】。
3. 如果片段之间说法冲突，明确指出冲突，不要挑一个当结论。
4. 用中文回答。先给结论，再给依据，不要超过 200 字。
5. 不要输出 LaTeX（\\(...\\)、\\frac{}{}、\\sum_{} 这类），公式用纯文本和 Unicode 符号写：
   写 h_i^2 = ∑ b_ij^2，不写 \\(h_{i}^{2}=\\sum_{j} b_{ij}^{2}\\)。答案直接显示在网页上，没有公式渲染。
6. 问题里的关键概念如果**一次都没出现在片段中**，就是"讲义未覆盖"，不要用相近的片段凑出处。
   拿另一页顶替、或者说"虽然没提到但通常指…"，都属于编造。
"""


def system_for(profile: "DomainProfile | None" = None) -> str:
    """有领域画像时换一句 system：把"数据挖掘课程的助教"换成这份资料的助教。

    为什么值得改：用户传的可能是医学指南或合同。继续自称"数据挖掘助教"，
    模型会倾向于把内容往数据挖掘上解释 —— 这是**领域漂移**，比答错更难发现。
    """
    if profile is None or not getattr(profile, "domain", ""):
        return SYSTEM
    return (
        f"你是《{profile.domain}》这份资料的助教，只根据用户提供的资料片段回答。"
        "你不使用任何外部知识，也不猜测。"
    )


def build_context(results: list[SearchResult], *, max_chars: int = 6000) -> str:
    """把检索结果拼成片段块。超长时按顺序截断（最相关的在前，先保它们）。

    max_chars 是硬上限：块本身最大 1175 字，5 块最多约 5900 字，
    正常情况下不会触发截断。留着是为了防止有人把 top_k 调到 20 之后
    prompt 突然膨胀到几万 token（那会直接超出模型上下文或烧钱）。
    """
    parts: list[str] = []
    used = 0
    for r in results:
        block = f"{r.chunk.header()}\n{r.chunk.text}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_prompt(
    question: str,
    results: list[SearchResult],
    *,
    max_chars: int = 6000,
    profile: "DomainProfile | None" = None,
) -> str:
    """拼 prompt。profile 为 None 时走的仍是 M4 的原路径，一个字没改。

    有画像时（上传资料场景）改三处措辞：
      1. "讲义" → "资料"，"讲义未覆盖" → "资料未覆盖"（用户传的不是讲义）；
      2. 插入领域与术语块（用词一致性，见 studio/profile.py 的 prompt_block）；
      3. 保留全部硬规则 —— 引用格式、拒答、冲突声明一条都不放松。
    """
    context = build_context(results, max_chars=max_chars)

    if profile is None:
        return (
            f"{INSTRUCTIONS}\n"
            f"=== 讲义片段（共 {len(results)} 段，按相关度从高到低）===\n"
            f"{context}\n"
            f"=== 片段结束 ===\n\n"
            f"问题：{question}"
        )

    block = profile.prompt_block()
    scope_rules = """你是助教。请**只**根据下面的资料片段回答问题。

规则：
1. 只使用片段里的内容。片段里没有的信息，直接回答"资料未覆盖"，不要补充你的先验知识。
2. 每个结论后面标注出处，格式**必须**是【doc_id:p页码】，例如【u001:p3】。
   页码用片段头里给出的那个；跨页时写成【u001:p12-13】。
3. 如果片段之间说法冲突，明确指出冲突，不要挑一个当结论。
4. 用中文回答。先给结论，再给依据，不要超过 200 字。
5. 不要输出 LaTeX（\\(...\\)、\\frac{}{}、\\sum_{} 这类），公式用纯文本和 Unicode 符号写：
   写 h_i^2 = ∑ b_ij^2，不写 \\(h_{i}^{2}=\\sum_{j} b_{ij}^{2}\\)。答案直接显示在网页上，没有公式渲染。
6. 问题里的关键概念如果**一次都没出现在片段中**，就是"资料未覆盖"，不要用相近的片段凑出处。
   拿另一页顶替、或者说"虽然没提到但通常指…"，都属于编造。
"""
    return (
        f"{scope_rules}\n"
        + (f"{block}\n\n" if block else "")
        + f"=== 资料片段（共 {len(results)} 段，按相关度从高到低）===\n"
        f"{context}\n"
        f"=== 片段结束 ===\n\n"
        f"问题：{question}"
    )
