"""M1-T1.2 验收：配置层。

这一组测试在回答："key 没配好时，报错发生在哪、报得够不够清楚？"
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmkit.config import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_PRICE_IN,
    Settings,
    load_dotenv,
)
from llmkit.errors import ConfigError, FatalError, LLMError


def test_缺密钥抛_ConfigError_且消息里有变量名(tmp_path):
    with pytest.raises(ConfigError) as info:
        Settings.from_env({}, env_file=tmp_path / "none.env")

    assert ENV_API_KEY in str(info.value)


def test_ConfigError_能被_LLMError_兜住():
    # 上层写 except LLMError 时必须能接住配置错误，否则调用点要写两套 except
    assert issubclass(ConfigError, FatalError)
    assert issubclass(ConfigError, LLMError)


def test_有密钥时默认值正确(tmp_path):
    settings = Settings.from_env({ENV_API_KEY: "sk-1234567890"}, env_file=tmp_path / "n.env")
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.model == "gpt-4o-mini"
    assert settings.timeout_s == 60.0


def test_进程环境变量覆盖_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{ENV_API_KEY}=from-file\n{ENV_MODEL}=file-model\n", encoding="utf-8")

    settings = Settings.from_env({ENV_API_KEY: "from-env"}, env_file=env_file)
    assert settings.api_key == "from-env"         # 环境变量优先
    assert settings.model == "file-model"          # 只在文件里有的照用


def test_dotenv_解析忽略注释与空行(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# 注释\n\nA=1\nB = 'two' \nC=\"three\"\n没有等号的一行\n", encoding="utf-8"
    )
    parsed = load_dotenv(env_file)
    assert parsed == {"A": "1", "B": "two", "C": "three"}


def test_非法数字报_ConfigError_而不是裸_ValueError(tmp_path):
    with pytest.raises(ConfigError) as info:
        Settings.from_env(
            {ENV_API_KEY: "sk-1234567890", ENV_PRICE_IN: "便宜"},
            env_file=tmp_path / "n.env",
        )
    assert ENV_PRICE_IN in str(info.value)


def test_密钥脱敏只留末四位(tmp_path):
    settings = Settings.from_env({ENV_API_KEY: "sk-abcdefgh1234"}, env_file=tmp_path / "n.env")
    assert settings.masked_key == "***1234"
    assert "abcdefgh" not in settings.masked_key


def test_短密钥全部打码(tmp_path):
    settings = Settings.from_env({ENV_API_KEY: "short"}, env_file=tmp_path / "n.env")
    assert settings.masked_key == "***"


def test_成本按单价计算():
    settings = Settings(
        api_key="sk-1", price_in_cny_per_mtok=2.0, price_out_cny_per_mtok=8.0
    )
    # 100 万输入 + 100 万输出 = 2 + 8 = 10 元
    assert settings.cost_cny(1_000_000, 1_000_000) == pytest.approx(10.0)
    # 1000 输入 + 500 输出
    assert settings.cost_cny(1000, 500) == pytest.approx(0.002 + 0.004)


def test_单价为_0_时成本就是_0_不做假估算():
    settings = Settings(api_key="sk-1")
    assert settings.cost_cny(1_000_000, 1_000_000) == 0.0


def test_配置创建后不可修改():
    settings = Settings(api_key="sk-1")
    with pytest.raises(ValidationError):
        settings.model = "别的模型"  # type: ignore[misc]


def test_base_url_可以自定义(tmp_path):
    settings = Settings.from_env(
        {ENV_API_KEY: "sk-1", ENV_BASE_URL: "https://api.deepseek.com/v1"},
        env_file=tmp_path / "n.env",
    )
    assert settings.base_url == "https://api.deepseek.com/v1"
