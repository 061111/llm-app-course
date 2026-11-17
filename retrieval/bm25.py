"""M3-T1 · 词法检索：BM25。

为什么先做 BM25，而不是直接上向量检索：
  1. 它是**可解释的基线**。分数由"哪些词命中、各命中几次、这个词多罕见"算出来，
     调不好时能一眼看出是"查询词没出现在语料里"还是"语料里到处都是这个词"。
  2. 向量检索需要 embedding 服务，而当前账号（DeepSeek）只提供对话模型。
     先有基线，等接上 embedding 才知道它到底提升了多少 —— 没有基线的"提升"
     无法量化。
  3. 纯 Python 一百行，零依赖，跑 181 个块一次几毫秒。

【BM25 是什么，一句话】
  给定一个查询，给每个文档打分：**查询词在这个文档里出现得越多分越高，
  但这个词在整个语料里越罕见、贡献越大，同时文档越长越要打折扣。**

  公式（本文件用的形式）：
      score(q, d) = Σ_{t∈q} IDF(t) · (tf · (k1+1)) / (tf + k1 · (1 - b + b · |d|/avgdl))
      IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

  k1 控制"词频饱和"：一个词出现 10 次不该是出现 1 次的 10 倍分。取 1.5。
  b 控制"长度归一化"：长文档天然更容易命中，要压一压。取 0.75。

  生活类比：查词典找一个人名。生僻名（IDF 高）一出现基本就锁定了；
  常见字（"张""王"）出现十次也没用。长条目要打折扣，因为它什么都可能提到。
  类比失效处：BM25 只看词形，不懂同义（"car" 与 "automobile" 算两个词），
  这也是它和向量检索的根本差别。

【一个刻意的设计：索引文本 = 标题 + 正文】
  块正文里不一定含自己的标题（chap05 p19–p23 的标题是图片）。
  把标题拼进索引文本，等于给每个块补一段"摘要"。
  用户问"规则排序"时，标题里有 "Rule Ordering" 的块应该更容易被命中。
  注意：这只影响**检索**，不影响喂给模型的正文（正文仍是 chunk.text）。
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

# 停用词：高频但几乎不携带区分度。
# 不删干净也能跑，只是"the / of"会把所有块的分一起抬高，排序更钝。
STOPWORDS = frozenset(
    """a an the of to in on for and or is are was were be been being it its this that these
    those with by as at from we you they he she not no if then than which who whom what when
    where how all any both each few more most other some such only own same so too very can
    will just should now do does did has have had also into their there here about over under
    between during before after above below up down out off again further once""".split()
)

# 词元：优先取字母（含重音，如 Naïve），退而取数字（含小数 0.278）。
TOKEN_RE = re.compile(r"[^\W\d_]+|\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """统一字符形态，再做分词。两步都不能省：

    1. NFKD 兼容分解：把数学斜体 𝐺𝑖𝑛𝑖 还原成 Gini、全角字母还原成半角。
       **实测**：chap03 p59 整页在算 Gini，公式里的 "Gini" 是数学斜体；
       归一化后那一页的 gini 出现 5 次（p57 定义页是 3 次），G04 才排得进去。
       残留问题：连写的 𝐺𝑖𝑛𝑖𝐼𝑛𝑑𝑒𝑥 会变成 "GiniIndex" 一个词（不是 gini + index），
       这种黏连无法靠规则拆开，已知不修。

    2. 去掉组合用音符（分音符、重音符等）：讲义写的是 "Naïve"，用户查询写的是
       "naive"。不做这一步两者是**不同的词元**，实测 p17/p18 里 "naive" 的词频是 0，
       命中完全靠 "bayes" 一个词撑着。
       生活类比：查人名时把 "José" 和 "Jose" 当成同一个人。类比失效处：
       有些语言里去音符会改变词义（如越南语），对本语料（英文学术 PPT）不适用。
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """字符归一化 + 小写化 + 抽词元 + 去停用词 + 丢掉单字母（数字除外）。

    单字母（k、n、v）在教材里是有意义的符号，但太短太常见，
    留着会让 "P(X|Y)" 这类公式页命中一堆无关块。

    **单字符数字保留**（这条是实测出来的）：黄金集里 G04 问 "(0,6)、(1,5)、(2,4)
    时 Gini 各是多少"，问题本身不含答案数字，唯一的区分信号就是那几个逗号数字。
    丢掉它们，p57（定义页）和 p59（算例页）就完全同分，算例页排不进去。
    实测保留后：hit@1 61%→67%，hit@5 89%→94%。
    """
    out: list[str] = []
    for tok in TOKEN_RE.findall(normalize(text).lower()):
        if (len(tok) < 2 and not tok.isdigit()) or tok in STOPWORDS:
            continue
        out.append(tok)
    return out


def index_text(title: str, text: str) -> str:
    """索引用的文本 = 标题 + 正文。见文件头"刻意的设计"。"""
    return f"{title}\n{text}" if title else text


class Hit(BaseModel):
    """一条检索结果。"""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    score: float = Field(description="BM25 分数，跨查询不可比")
    rank: int = Field(ge=1, description="1 表示第一名")


class BM25:
    """纯 Python 的 BM25。N 只有 181，不需要倒排索引，暴力扫一遍就够。"""

    def __init__(self, docs: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avgdl = (sum(self.lens) / self.n_docs) if self.n_docs else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.df = df

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: list[str], doc_index: int) -> float:
        tf = self.tfs[doc_index]
        dl = self.lens[doc_index] or 1
        total = 0.0
        for term in query:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            total += self.idf(term) * f * (self.k1 + 1) / denom
        return total

    def search(self, query: list[str], *, top_k: int = 5) -> list[tuple[int, float]]:
        """返回 [(文档下标, 分数)]，按分数降序，只留 >0 的。

        分数为 0 的块不进结果：它一个查询词都没命中，
        把它排进 top-k 只会让下游 Prompt 里塞进无关内容。
        """
        scored = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]
