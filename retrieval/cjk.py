"""M5-T1 · 中文切词：不改依赖，用字二元组（bigram）让 BM25 能检索中文。

【为什么必须有这一层】
  原来的 TOKEN_RE 是 `[^\\W\\d_]+`，而 Python 的 `\\w` **包含中日韩汉字**，
  所以一整句中文会被切成一个词元。实测（2026-11-18，中文语料 7 页）：

      问题："贝叶斯分类器的核心假设是什么"
      正文："朴素贝叶斯分类器假设各属性在给定类别下条件独立"
      两个词元集合的交集 = 空集 → BM25 得分恒为 0 → 一条都召不回来

  这不是"效果差一点"，是**可用与不可用**的区别。

【为什么是 bigram，不是 jieba】
  1. jieba 是一个新依赖（约 20 MB 词典），而本项目的依赖纪律是"只装真正需要的"。
  2. 检索场景里分词器**不需要语言学正确**，只需要"同一个词在文档和查询里被切成
     同一串符号"。bigram 天然满足：无论"贝叶斯"是不是一个词，
     文档和查询里都会切出 "贝叶""叶斯"。
  3. 代价是词表变大（N 个字 → N-1 个二元组），181 块量级下完全无所谓；
     而且 IDF 会自动压掉"的是""我们"这类到处都有的二元组 ——
     BM25 的 IDF 项本身就是噪音过滤器。

  生活类比：查人名时不要求知道复姓，"欧阳修"切成"欧阳/阳修"，
  另一个地方出现"欧阳修"也会切成"欧阳/阳修"，照样对得上。
  类比失效处：单字词（"熵""秩"）会被切没，所以只有一个字的中文串单独保留。
  bigram 也分不出"南京市长江大桥"的两种断句 —— 这在检索里不构成问题，
  因为两种断句的二元组大部分仍然相同。

【英文路径零改动】
  tokenize 先把文本按"汉字段 / 非汉字段"切开，非汉字段仍走原来的 TOKEN_RE。
  所以英文语料的分词结果与改造前**逐词相同**（283 条既有测试就是证据）。
"""
from __future__ import annotations

import re

# 汉字区段：基本区 U+4E00–U+9FFF、扩展 A U+3400–U+4DBF、兼容区 U+F900–U+FAFF。
# 不含标点、不含日文假名（假名段落会落到非汉字分支，按拉丁规则处理，
# 结果是整段假名成为一个词元 —— 当前语料用不到，已知不修）。
CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")

# 中文停用字。**只在"整个二元组都由停用字组成"时才丢**，
# 例如 "的是""了我""不的"；而 "的核"（的 + 核）会保留，
# 因为"核"可能是一个术语的开头（"核心假设" → "的核" 是噪声，但 "核心" 是信号）。
# 这张表刻意只收**纯功能字**，不收"一/上/下/中"这类可作实词的方位字。
CJK_STOP_CHARS = frozenset(
    "的了着过是在有和与或及等这那为对不我你他她它们个把被让从到会能可要还想很更最再又"
    "并且但如因所如果虽然而是就都也才只呢吗吧啊呀嘛哦嗯"
)


def cjk_terms(run: str) -> list[str]:
    """把一个连续汉字串切成词元。

    - 长度 ≥2：相邻两字组成一个二元组，丢掉"全停用字"的组合；
    - 长度 =1：单独保留（除非它是停用字），否则"熵"这种单字术语永远检索不到；
    - 空串：返回空列表。
    """
    if not run:
        return []
    if len(run) == 1:
        return [] if run in CJK_STOP_CHARS else [run]

    out: list[str] = []
    for i in range(len(run) - 1):
        gram = run[i : i + 2]
        if gram[0] in CJK_STOP_CHARS and gram[1] in CJK_STOP_CHARS:
            continue
        out.append(gram)
    return out


def has_cjk(text: str) -> bool:
    """文本里有没有汉字。用于判断资料语种、决定要不要走"中文问题→英文关键词"翻译。"""
    return bool(CJK_RUN_RE.search(text))


def cjk_ratio(text: str) -> float:
    """汉字占"字母类字符"的比例，用来给文档打语种标签（zh / en / mixed）。

    **分母不是全部字符，而是"汉字 + ASCII 字母"**。这一条是实测改出来的：
    第一版用"汉字 / 非空白字符"，中文习题解答只有 0.251 —— 因为那批 PDF 的公式
    抽出来是大量数字、运算符和私有区乱码，把分母撑大了，一份中文资料被判成 mixed。
    换成"汉字 / (汉字 + 拉丁字母)"后，同一份文件是 0.945。

    实测分布（2026-11-18，本机）：
        中文习题解答 0.945 / 中文考点 0.906 / 中文样题 0.882 / chap03（英文）0.000
        本项目 README（中英混排）0.493 / bm25.py（中文注释 + 英文代码）0.337
    """
    cjk = sum(len(m.group(0)) for m in CJK_RUN_RE.finditer(text))
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = cjk + latin
    return cjk / total if total else 0.0


def cjk_runs(text: str) -> list[str]:
    """抽出所有连续汉字串。术语抽取（studio/glossary.py）用它统计 n-gram。"""
    return CJK_RUN_RE.findall(text)
