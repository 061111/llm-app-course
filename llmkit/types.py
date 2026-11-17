"""M1-T1.1 · 契约层：先定"返回长什么样"，再写"怎么实现"。

这个文件里没有一行网络代码，全部是 Pydantic 模型。
它的作用是让调用方（未来的检索层、Agent 层）**只依赖这个形状**，
不依赖 openai SDK，也不依赖我们怎么调。

为什么先写它：
  1. 测试可以先写——不用等实现；
  2. 换供应商时这个文件一个字不用改；
  3. 上层拿到的一定是校验过的对象，不会有 `resp.choices[0]` 这种裸取值。
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Usage(BaseModel):
    """一次调用的 token 账。

    `extra="forbid"` 让"多传字段"直接报错——防止上游把 SDK 的 usage 对象整个塞进来，
    那种对象字段名带下划线（prompt_tokens 是有的，但 total_tokens 是计算出来的属性），
    混进来以后序列化会出奇怪的问题。
    """

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0, description="输入 token 数")
    completion_tokens: int = Field(default=0, ge=0, description="输出 token 数")
    cost_cny: float = Field(default=0.0, ge=0.0, description="本次花费，人民币元")

    @property
    def total_tokens(self) -> int:
        """总 token。注意它是 property，不是字段——不会被写进 JSON。"""
        return self.prompt_tokens + self.completion_tokens


class Citation(BaseModel):
    """一条引用来源。M1 阶段还不会真的产生它，先把形状定下来。

    M4 让模型吐出 citations 时，这里就是落点；现在留空列表即可。
    """

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, description="来源标识，如文件名或文档 id")
    page: int | None = Field(default=None, ge=1, description="页码，1 起算；不知道就 None")
    quote: str = Field(default="", description="被引用的原文片段")


class AskOptions(BaseModel):
    """单次调用的参数。跟 Settings 的区别：

    - `Settings` 是**环境级**的（key、地址、默认模型），一次进程一份；
    - `AskOptions` 是**调用级**的（这次用哪个模型、多大温度），每次可以不同。
    """

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(default=None, description="不填就用 Settings.model")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    max_attempts: int = Field(default=3, ge=1, le=10)
    system: str = Field(default="你是一个严谨的技术助手。回答简洁，不确定就说不确定。")


class AskResult(BaseModel):
    """`ask()` 的唯一返回类型。

    三个字段没有默认值：text、model、latency_ms。
    没有默认值 = 调用方必须给 = 少了就报 ValidationError，而不是悄悄给你一个空串。
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="模型返回的正文")
    model: str = Field(min_length=1, description="实际使用的模型名")
    latency_ms: float = Field(ge=0.0, description="端到端耗时，毫秒")
    citations: list[Citation] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    attempts: int = Field(default=1, ge=1, description="实际调用了几次（含首发）")
    trace_id: str = Field(default="", description="本次调用的追踪 id，与日志里的同一个")
