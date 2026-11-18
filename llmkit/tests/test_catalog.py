"""M7-T1 验收：模型目录。

这组测试回答三件事：
  1. 上游清单拿不到时，界面还能不能开（必须能，降级不算失败）；
  2. 会不会把"画图的模型"塞进聊天下拉（会的话用户必然撞参数错误）；
  3. 换模型时那句"推理型要给 4000 输出预算"的规矩有没有真的生效。
"""
from __future__ import annotations

import pytest

from llmkit import catalog
from llmkit.catalog import (
    DEFAULT_MAX_TOKENS,
    RECOMMENDED,
    REASONING_MAX_TOKENS,
    SOURCE_FALLBACK,
    SOURCE_UNCONFIGURED,
    SOURCE_UPSTREAM,
    entry_for,
    is_chat_model,
    list_available,
    max_tokens_for,
)
from llmkit.config import Settings

SETTINGS = Settings(
    api_key="sk-test-1234567890",
    base_url="https://example.invalid/v1",
    model="qwen3.8-flash",
)


@pytest.fixture(autouse=True)
def _clean_cache():
    """每个用例都从空缓存开始，否则前一个用例的清单会漏进来。"""
    catalog.clear_cache()
    yield
    catalog.clear_cache()


def test_过滤掉画图念语音的模型():
    """上游把 184 个模型混在一起返回。选错一个的后果是"参数错误"而不是"答得不好"，
    所以必须在下拉里就挡住，而不是等调用失败再解释。"""
    for bad in (
        "qwen-image-3.0",
        "wan2.7-image-pro",
        "qwen3-tts-flash",
        "qwen-audio-3.0-asr-flash",
        "qwen-vl-ocr",
        "qwen3.7-text-embedding",
        "qwen3.7-text-rerank",
        "qwen3-omni-flash-realtime",
        "qwen-mt-plus",
        "z-image-turbo",
    ):
        assert not is_chat_model(bad), bad

    for good in ("qwen3.8-max", "qwen-plus", "deepseek-v4-pro", "kimi-k3.1-pro"):
        assert is_chat_model(good), good


def test_推理型模型的输出预算更大():
    """实测：deepseek-v4-pro 在 max_tokens=200 时正文为空、200 全花在 reasoning 上。
    问答 prompt 有 6000 字符上下文，思考只会更长，所以必须是 4000。"""
    assert max_tokens_for("deepseek-v4-pro") == REASONING_MAX_TOKENS
    assert max_tokens_for("kimi-k3.1-pro") == REASONING_MAX_TOKENS
    # 非推理型：上限给宽一点无害（模型自己 stop 就停），给窄了会吐空
    assert max_tokens_for("qwen3.8-max") == DEFAULT_MAX_TOKENS
    assert DEFAULT_MAX_TOKENS > 800


def test_默认模型是推荐表第一行():
    """前端在上游清单拉不到、或 .env 里配的模型不在清单里时，会退回推荐表第一行。

    所以第一行和 .env 的 LLMKIT_MODEL 必须是同一个模型，否则用户会看到
    "下拉显示 A、实际回答用 B"，而且只有翻 .env 才发现。
    """
    first = RECOMMENDED[0]
    assert first.id == "deepseek-v4.1-flash"
    # 实测 reasoning_tokens=0，所以不该占推理预算
    assert first.reasoning is False
    assert first.max_tokens == DEFAULT_MAX_TOKENS


def test_没实测过的模型按默认预算_不猜它是不是推理型():
    """名字里带 -think 不代表它真的思考。猜错的代价是钱和延迟，而默认值两边都不亏。"""
    entry = entry_for("some-brand-new-model")
    assert entry.id == "some-brand-new-model"
    assert entry.reasoning is False
    assert entry.max_tokens == DEFAULT_MAX_TOKENS


def test_推荐表里的模型带着实测备注():
    """备注是给人选的依据（快/慢/推理型），不是装饰。空备注等于没写。"""
    for entry in RECOMMENDED:
        assert entry.note, entry.id
        assert "实测" in entry.note, entry.id


def test_正常拉取时只保留上游真实存在的推荐():
    upstream = ("qwen3.8-flash", "qwen-plus", "qwen-image-3.0", "brand-new-chat")
    info = list_available(SETTINGS, upstream=lambda: upstream)

    assert info.source == SOURCE_UPSTREAM
    assert [m.id for m in info.recommended] == ["qwen3.8-flash", "qwen-plus"]
    # 上游有、但不在推荐表里的聊天模型进 others；画图模型被丢掉
    assert info.others == ("brand-new-chat",)


def test_上游挂了也要能开界面():
    """模型清单拉不到时整页白屏是最糟的结果：用户连"上传资料"都做不了。
    降级成推荐表，并把原因写在 note 里。"""

    def boom():
        raise TimeoutError("上游 10 秒没响应")

    info = list_available(SETTINGS, upstream=boom)

    assert info.source == SOURCE_FALLBACK
    assert len(info.recommended) == len(RECOMMENDED)
    assert "TimeoutError" in info.note
    assert "推荐表" in info.note


def test_没有密钥时只给推荐表并说明原因():
    info = list_available(None)

    assert info.source == SOURCE_UNCONFIGURED
    assert len(info.recommended) == len(RECOMMENDED)
    assert "密钥" in info.note


def test_清单会缓存_不每次都打上游():
    calls = {"n": 0}

    def counting():
        calls["n"] += 1
        return ("qwen-plus",)

    list_available(SETTINGS, upstream=counting)
    list_available(SETTINGS, upstream=counting)
    assert calls["n"] == 1

    list_available(SETTINGS, upstream=counting, refresh=True)
    assert calls["n"] == 2


def test_过期后会重新拉():
    """缓存过期用注入的时钟判，不 sleep——测试不该跑 10 分钟。"""
    calls = {"n": 0}

    def counting():
        calls["n"] += 1
        return ("qwen-plus",)

    list_available(SETTINGS, upstream=counting, now=1000.0)
    list_available(SETTINGS, upstream=counting, now=1000.0 + catalog.CACHE_TTL_S + 1)
    assert calls["n"] == 2


def test_默认模型不在上游清单时会提示():
    """这是真会遇到的情况：换了服务商，.env 里还留着旧模型名。
    不提示的话用户只看到"提问失败"，看不出是配置过期。"""
    info = list_available(SETTINGS, upstream=lambda: ("qwen-plus",))

    assert info.current == "qwen3.8-flash"
    assert "qwen3.8-flash" in info.note
    assert "不在上游清单" in info.note


def test_ids_先推荐后其余():
    info = list_available(SETTINGS, upstream=lambda: ("aaa-chat", "qwen-plus"))
    # 推荐表顺序固定在前，others 字母序在后
    assert info.ids[0] == "qwen-plus"
    assert info.ids[-1] == "aaa-chat"
