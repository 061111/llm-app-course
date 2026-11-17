"""W2-01 练习：Chunk 模型（B1 参考答案）。

先别看这个文件。跟着讲解的五步自己敲一遍，敲完再来对。
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class Chunk(BaseModel):
    """RAG 切出来的一段文本。"""

    text: str = Field(min_length=1, max_length=500, description="切出来的一段原文")
    page: int = Field(ge=1, description="所在页码，从 1 开始")
    tags: list[Annotated[str, Field(min_length=1, max_length=20)]] = Field(
        default_factory=list,
        max_length=3,
        description="最多 3 个标签，每个 1-20 字",
    )

    @field_validator("text")
    @classmethod
    def _no_todo(cls, v: str) -> str:
        if "TODO" in v:
            raise ValueError("text 里不能出现 TODO（未完成的段落不该入库）")
        return v


if __name__ == "__main__":
    print(Chunk(text="K-means 的 K 要预先指定", page=1, tags=["聚类", "无监督"]))
