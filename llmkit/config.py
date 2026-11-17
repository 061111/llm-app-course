"""M1-T1.2 · 配置层：密钥、地址、默认模型。

三条纪律，写在这个文件里由代码强制：
  1. 密钥只从环境变量 / .env 来，代码里永不出现字面量；
  2. 缺密钥时在**启动瞬间**报错，而不是等到发请求后被服务端回 401；
  3. 打印配置时密钥必须脱敏——否则日志一落盘，密钥就泄露了。

依赖顺序：进程环境变量 > .env 文件 > 代码默认值。
（`.env` 是给你本机用的，不要提交到 git。）
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from llmkit.errors import ConfigError

# 环境变量名集中在这里，改一处即可全局生效
ENV_API_KEY = "LLMKIT_API_KEY"
ENV_BASE_URL = "LLMKIT_BASE_URL"
ENV_MODEL = "LLMKIT_MODEL"
ENV_TIMEOUT = "LLMKIT_TIMEOUT_S"
ENV_PRICE_IN = "LLMKIT_PRICE_IN"
ENV_PRICE_OUT = "LLMKIT_PRICE_OUT"

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """读一个极简 .env。不引入 python-dotenv，因为它没写进 pyproject 的依赖。

    只支持 `KEY=VALUE` 和 `#` 注释，不支持多行、变量插值、export 前缀。
    够用就好——生产环境应该由容器/平台注入环境变量，不该靠 .env 文件。
    """
    p = Path(path)
    if not p.is_file():
        return {}

    result: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            result[key] = value
    return result


class Settings(BaseModel):
    """进程级配置。`frozen=True` 让它创建后不可改——配置被中途改动是极难查的 bug。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key: str = Field(min_length=1, description="服务商密钥")
    base_url: str = Field(default=DEFAULT_BASE_URL, description="兼容 OpenAI 协议的地址")
    model: str = Field(default=DEFAULT_MODEL, min_length=1)
    timeout_s: float = Field(default=60.0, gt=0, le=600)
    max_attempts: int = Field(default=3, ge=1, le=10)
    price_in_cny_per_mtok: float = Field(
        default=0.0, ge=0.0, description="输入每百万 token 价格（元）。0 表示不估算成本"
    )
    price_out_cny_per_mtok: float = Field(default=0.0, ge=0.0)

    @property
    def masked_key(self) -> str:
        """给日志/CLI 用的脱敏展示：只留最后 4 位。"""
        if len(self.api_key) <= 8:
            return "***"
        return f"***{self.api_key[-4:]}"

    def cost_cny(self, prompt_tokens: int, completion_tokens: int) -> float:
        """按单价估算成本。单价为 0 时返回 0.0，不做假估算。"""
        return (
            prompt_tokens / 1_000_000 * self.price_in_cny_per_mtok
            + completion_tokens / 1_000_000 * self.price_out_cny_per_mtok
        )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        env_file: str | Path = ".env",
    ) -> Settings:
        """从环境变量构造。缺失密钥时抛 ConfigError，消息里带上该设哪个变量。

        `env` 参数是为了测试能传一个假的 dict，不去污染真实进程环境。
        """
        dotenv = load_dotenv(env_file)
        # 进程环境优先：显式设置的变量应该盖过文件里的值
        merged = {**dotenv, **(os.environ if env is None else env)}

        api_key = merged.get(ENV_API_KEY, "").strip()
        if not api_key:
            raise ConfigError(
                f"缺少 API 密钥：请设置环境变量 {ENV_API_KEY}（或在项目根建 .env 写入 "
                f"{ENV_API_KEY}=你的密钥）。当前 base_url={merged.get(ENV_BASE_URL, DEFAULT_BASE_URL)}"
            )

        def _float(name: str, default: float) -> float:
            raw = merged.get(name, "").strip()
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise ConfigError(f"环境变量 {name} 应为数字，实际是 {raw!r}") from exc

        return cls(
            api_key=api_key,
            base_url=merged.get(ENV_BASE_URL, "").strip() or DEFAULT_BASE_URL,
            model=merged.get(ENV_MODEL, "").strip() or DEFAULT_MODEL,
            timeout_s=_float(ENV_TIMEOUT, 60.0),
            price_in_cny_per_mtok=_float(ENV_PRICE_IN, 0.0),
            price_out_cny_per_mtok=_float(ENV_PRICE_OUT, 0.0),
        )
