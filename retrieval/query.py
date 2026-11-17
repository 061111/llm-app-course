"""M3-T2 · 查询侧：把中文问题变成英文检索词。

**这是整个检索层最关键的一步，也是 M3 真正的问题所在。**

语料是英文讲义，黄金集的问题是中文。BM25 只认字面 —— 拿
"混淆矩阵里 Type I Error 对应哪一格" 直接去检索，一个词都对不上，
recall 恒为 0。这跟 BM25 写得好不好无关，是**跨语言**问题。

两条路：
  A. 向量检索（embedding 天然跨语言）—— 需要 embedding 服务，这个账号没有。
  B. 查询翻译：先让对话模型把中文问题压成英文关键词，再走 BM25。

本文件走 B。它不是"退而求其次"：BM25 + 翻译在很多专有名词密集的场景里
比向量检索更准（术语是硬匹配），代价是多一次模型调用（约 1 秒、几百 token）。

【为什么输出的是关键词而不是完整英文句子】
  句子里的 "which / does / correspond" 全是停用词，进 BM25 一点用没有，
  反而增加噪音。关键词形式还能强制模型保留教材里的**原词形**
  （"Type I Error" 而不是 "first type mistake"），这正是字面匹配需要的。

【为什么要缓存】
  18 道题的评估会跑几十次。不缓存的话每次 18 次模型调用、20 秒、还有钱。
  缓存键是问题原文，值是关键词列表，存 artifacts/query_cache.json。
  缓存必须能显式刷新（refresh=True），否则改了 prompt 却还在用旧结果 ——
  那种 bug 极难发现。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from llmkit.client import ask_structured
from retrieval.bm25 import STOPWORDS

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "artifacts" / "query_cache.json"

PROMPT = """你是检索系统的查询改写器。把下面的中文问题转换成英文检索关键词。

规则：
1. 输出 4~10 个关键词或短语，用教材里会出现的**原始英文术语**（不要意译）。
   例："混淆矩阵" → "confusion matrix"；"增益率" → "gain ratio"；"支持度" → "support"。
2. 问题里出现的数字、公式符号、算法名必须原样保留：0.65、2/3、C4.5、CART、RIPPER、k-NN。
3. 不要输出句子、不要输出中文、不要输出 and/or 之类的连接词。
4. 只输出 JSON：{{"keywords": ["...", "..."]}}

中文问题：
{question}
"""


class Keywords(BaseModel):
    """模型返回的形状。关键词为空等于没翻出来，所以要求至少 1 个。"""

    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = Field(min_length=1, description="英文检索词")


def _load_cache(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}  # 缓存坏了就当没有，不要让主流程挂掉
    if not isinstance(raw, dict):
        return {}
    return {str(k): [str(x) for x in v] for k, v in raw.items() if isinstance(v, list)}


def _save_cache(path: Path, cache: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


class Translator:
    """中文问题 → 英文关键词。带缓存。

    call_fn 是为测试留的注入点：不给就用真的 ask_structured，
    给了就能在零网络的情况下测缓存逻辑。
    """

    def __init__(
        self,
        *,
        cache_path: Path | None = DEFAULT_CACHE,
        call_fn: Callable[[str], list[str]] | None = None,
        refresh: bool = False,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = _load_cache(self.cache_path) if self.cache_path else {}
        self.refresh = refresh
        self._call_fn = call_fn
        self.calls = 0  # 真实模型调用次数，评估时打印出来看花了多少

    def _default_call(self, question: str) -> list[str]:
        result = ask_structured(PROMPT.format(question=question), Keywords)
        return [k.strip() for k in result.keywords if k.strip()]

    def __call__(self, question: str) -> list[str]:
        if not self.refresh and question in self.cache:
            return self.cache[question]

        fn = self._call_fn or self._default_call
        self.calls += 1
        keywords = fn(question)
        # 二次清洗：模型偶尔会混进中文或整句，这里兜住
        cleaned: list[str] = []
        for k in keywords:
            k = k.strip()
            if not k or any("\u4e00" <= ch <= "\u9fff" for ch in k):
                continue
            if k.lower() in STOPWORDS:
                continue
            cleaned.append(k)
        if not cleaned:
            raise ValueError(f"翻译结果清洗后为空：{question!r} → {keywords!r}")

        if self.cache_path:
            self.cache[question] = cleaned
            _save_cache(self.cache_path, self.cache)
        return cleaned
