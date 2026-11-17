"""W2-01 练习 B1 的 3 个测试。"""

import pytest
from pydantic import ValidationError

from w02.ex01 import Chunk


def test_合法的_chunk_能构造出来() -> None:
    c = Chunk(text="K-means 的 K 要预先指定", page=1, tags=["聚类", "无监督"])
    assert c.page == 1
    assert c.tags == ["聚类", "无监督"]


def test_tags_传4个被拦() -> None:
    with pytest.raises(ValidationError):
        Chunk(text="x", page=1, tags=["a", "b", "c", "d"])


def test_text_含TODO被拦() -> None:
    with pytest.raises(ValidationError):
        Chunk(text="这一节 TODO 待补", page=1)
