"""M2-T1 · 语料登记表：5 个 PDF 的身份证。

为什么单独一个文件，而不是在 gold.py 里写 "chap03" 字符串：
  下游有三处要引用同一批 doc_id —— gold.py（黄金集）、parser.py（解析）、
  store.py（落盘）。散落在三处，改一个字母就有一处查半天。
  集中成 DOCS，任何地方写错 doc_id 都会在构造 Pydantic 对象时立刻炸。

页数（pages）不是装饰：黄金集测试要用它断言"引用的页码没有越界"。
没有它，你把 chap04b 的题写成第 30 页也测不出来（那个文件只有 14 页）。
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# 语料目录：默认在桌面，可用环境变量 LLMKIT_CORPUS_DIR 覆盖。
# 不在代码里写死绝对路径，因为这是你的私人文件，换机器就失效。
CORPUS_DIR_ENV = "LLMKIT_CORPUS_DIR"
DEFAULT_CORPUS_DIR = Path.home() / "Desktop" / "DataMining pdf"


def corpus_dir() -> Path:
    """返回语料目录。环境变量优先，否则用桌面默认路径。"""
    raw = os.environ.get(CORPUS_DIR_ENV, "").strip()
    return Path(raw) if raw else DEFAULT_CORPUS_DIR


class DocMeta(BaseModel):
    """一个 PDF 的身份。frozen：登记表建好后不该被谁偷偷改掉页数。"""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1, description="短名，黄金集和块元数据都用它")
    filename: str = Field(min_length=1, description="语料目录下的真实文件名")
    pages: int = Field(gt=0, description="总页数，用 pymupdf 数出来的")
    topic: str = Field(description="一句话说明这份讲义讲什么")

    def path(self) -> Path:
        """拼出完整路径。文件不存在时这里不报错，交给真正要读它的人报。"""
        return corpus_dir() / self.filename


# 2026-11-17 实测页数（pymupdf doc.page_count）：
#   chap03 82 / chap04a 37 / chap04b 14 / chap04c 28 / chap05 38，合计 199 页。
DOCS: dict[str, DocMeta] = {
    "chap03": DocMeta(
        doc_id="chap03",
        filename="chap03_Classification_Basic ConceptsandTechniques (1).pdf",
        pages=82,
        topic="分类基础：决策树、不纯度度量、模型评估",
    ),
    "chap04a": DocMeta(
        doc_id="chap04a",
        filename="chap04_Part_I_Rule_Based.pdf",
        pages=37,
        topic="基于规则的分类：覆盖度/准确率、顺序覆盖、RIPPER",
    ),
    "chap04b": DocMeta(
        doc_id="chap04b",
        filename="chap04_Part_II_knn.pdf",
        pages=14,
        topic="k-近邻：惰性学习、Voronoi 边界、缺失值",
    ),
    "chap04c": DocMeta(
        doc_id="chap04c",
        filename="chap04_Part_III_naive_bayes.pdf",
        pages=28,
        topic="贝叶斯分类：条件独立、朴素贝叶斯、概率平滑",
    ),
    "chap05": DocMeta(
        doc_id="chap05",
        filename="chap05_Association_Analysis.pdf",
        pages=38,
        topic="关联分析：支持度/置信度、Apriori、规则生成",
    ),
}
