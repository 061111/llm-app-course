"""W2-01 动手练习：只写字段声明，不写 validator。

需求 → 写法，每一行都对得上速查卡第二节。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Out(BaseModel):
    model_config = ConfigDict(extra="forbid")          # 多余字段直接报错

    answer: str = Field(min_length=1, description="给用户的回答")
    citations: list[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal["ok", "refused"] = "ok"


if __name__ == "__main__":
    print("=== 正常构造 ===")
    o = Out(answer="K-means 的 K 要预先指定", confidence=0.8)
    print(o)
    print(o.model_dump_json(indent=2))

    print("\n=== 四行字段各验一次，看谁在拦 ===")
    cases = [
        ("answer 空串", dict(answer="", confidence=0.8)),
        ("citations 6 条", dict(answer="x", confidence=0.8, citations=list("abcdef"))),
        ("confidence 1.5", dict(answer="x", confidence=1.5)),
        ("confidence 缺", dict(answer="x")),
        ("status 乱写", dict(answer="x", confidence=0.8, status="maybe")),
        ("多一个 mood", dict(answer="x", confidence=0.8, mood="good")),
    ]
    for name, kw in cases:
        try:
            Out(**kw)
            print(f"  {name:16s} -> 通过（说明这行没起作用，有问题）")
        except Exception as e:
            print(f"  {name:16s} -> 拦住：{type(e).__name__}")
