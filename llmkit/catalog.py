"""M7-T1 · 模型目录：把"写死一个模型"变成"可以换一个模型"。

【为什么清单要向上游要，不许写死】
  写死一个模型列表，等于替服务商做了两个决定：哪些模型存在、哪些模型还能用。
  这两件事都会过期，而且是**静默**过期——你写 `qwen-plus`，服务商下架了，
  用户看到的是一句"模型不存在"，而不是"你的清单过期了"。
  所以真实清单从 `GET {base_url}/models` 来，那是唯一权威。写死的只有**推荐**：
  推荐表是"我实测过、知道它快还是慢、知道它要多少输出预算"，这部分没法从接口问出来。

【为什么推荐表里要带 max_tokens】
  这是实测踩出来的，不是设计洁癖。同一句问题、同一把密钥：

    | 模型 | max_tokens=200 | max_tokens=4000 |
    |---|---|---|
    | qwen3.8-max   | 正文 72 token，4.1 s  | — |
    | qwen-max      | 正文 22 token，1.1 s  | — |
    | deepseek-v4-pro | **正文空**，200 全花在 reasoning | 正文 668，reasoning 621，16.9 s |
    | kimi-k3.1-pro | **正文空**，同上 | 正文 862，reasoning 772，31.3 s |
    | glm-5.3       | 正文被截成"信息" | 正文 805，reasoning 727，9.4 s |

  推理型模型先想后答，思考的 token 和正文共用 `max_tokens` 这一个池子。
  问答的 prompt 塞了 6000 字符上下文，思考只会更长，所以这类模型必须给 4000。
  不给的下场是"每个问题都返回空"，而空正文在 llmkit/client.py 里会重试三次再报错——
  用户看到的是"重试三次都失败"，不是"选了个慢模型"。

【推理型是实测标注，不是靠名字猜】
  名字里带 `-think`/`-r1` 只是线索，不可靠：`qwen3.7-plus` 名字看不出来，
  实测 reasoning=439；`qwen3.8-max` 看起来该很重，实测 reasoning=0。
  所以 `reasoning` 字段只对**实测过的模型**置 True，其余一律按 0 处理（给 1600 的默认预算）。
  未知模型给 1600 而不是 800：上限不花钱，模型自己 stop 就停，
  但 800 会让"其实需要想一想"的模型直接吐空。
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

# 输出预算。含义是 `max_tokens`，不是"一定花掉这么多"。
DEFAULT_MAX_TOKENS = 1600
REASONING_MAX_TOKENS = 4000

# 上游清单缓存。600 秒的理由：模型上下架是"周"级变化，
# 而每次切模型都打一次上游会让界面多一次网络往返（实测 0.3~1.2 s）。
CACHE_TTL_S = 600.0

SOURCE_UPSTREAM = "upstream"
SOURCE_FALLBACK = "fallback"
SOURCE_UNCONFIGURED = "unconfigured"


class ModelEntry(BaseModel):
    """一个可选模型。`note` 是给人看的，`reasoning` / `max_tokens` 是给代码用的。"""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    note: str = ""
    reasoning: bool = Field(
        default=False, description="是否先思考再答（实测标注；True 时输出预算要放大）"
    )
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=256, le=32000)


# 推荐表：全部在 2026-11-18 用同一句话实测过，耗时是那次调用的墙钟时间。
# 顺序有意义：第一行是默认值（.env 里的 LLMKIT_MODEL 应与它一致），
# 也是"不知道选什么就选这个"；上游清单拉不到时前端也会退回第一行。
RECOMMENDED: tuple[ModelEntry, ...] = (
    ModelEntry(
        id="deepseek-v4.1-flash",
        note="默认档，非推理型、快且稳（实测 1.1 s）",
    ),
    ModelEntry(id="qwen3.8-flash", note="快且便宜，日常问答备选（实测 1.5 s）"),
    ModelEntry(id="qwen3.8-max", note="旗舰，答案更完整（实测 4.1 s）"),
    ModelEntry(id="qwen-plus", note="通用档，稳定（实测 2.0 s）"),
    ModelEntry(id="qwen-max", note="上一代旗舰，最快（实测 1.1 s）"),
    ModelEntry(
        id="deepseek-v4-pro",
        note="推理型，先想后答，慢但更周全（实测 16.9 s）",
        reasoning=True,
        max_tokens=REASONING_MAX_TOKENS,
    ),
    ModelEntry(
        id="qwen3.7-plus",
        note="推理型，中文表达好（实测 10.3 s）",
        reasoning=True,
        max_tokens=REASONING_MAX_TOKENS,
    ),
    ModelEntry(
        id="glm-5.3",
        note="智谱推理型（实测 9.4 s）",
        reasoning=True,
        max_tokens=REASONING_MAX_TOKENS,
    ),
    ModelEntry(
        id="kimi-k3.1-pro",
        note="推理型，长思考，最慢（实测 31.3 s）",
        reasoning=True,
        max_tokens=REASONING_MAX_TOKENS,
    ),
)

_BY_ID: dict[str, ModelEntry] = {m.id: m for m in RECOMMENDED}

# 上游 184 个模型里绝大多数不是聊天模型。选中 `qwen-image-3.0` 去问"什么是方差"，
# 上游会回一句看不懂的参数错误——这不是用户该遇到的。
# 规则是**黑名单**而不是白名单：新模型没进白名单会被藏起来，黑名单只藏明确不能聊的。
_NON_CHAT_MARKERS: tuple[str, ...] = (
    "image",
    "tts",
    "asr",
    "ocr",
    "embedding",
    "rerank",
    "speech",
    "audio",
    "realtime",
    "livetranslate",
    "omni",
    "s2s",
    "wan2",
    "z-image",
    "character",
    "gui-",
    "mt-",
    "deep-research",
    "deep-search",
    "xiaomi-analysis",
)


def is_chat_model(model_id: str) -> bool:
    """这个名字像不像一个能聊天的模型。大小写不敏感。"""
    low = model_id.lower()
    return not any(marker in low for marker in _NON_CHAT_MARKERS)


def entry_for(model_id: str) -> ModelEntry:
    """查这个模型的参数。没实测过的按默认预算处理，不猜它是不是推理型。"""
    if not model_id:
        return ModelEntry(id="")
    hit = _BY_ID.get(model_id)
    return hit if hit is not None else ModelEntry(id=model_id)


def max_tokens_for(model_id: str) -> int:
    return entry_for(model_id).max_tokens


class ModelList(BaseModel):
    """一次清单查询的结果。`source` 必须如实回传，前端据此决定要不要提示"清单可能过期"。"""

    model_config = ConfigDict(frozen=True)

    source: str
    base_url: str = ""
    current: str = Field(default="", description="服务端 .env 里配的默认模型")
    recommended: tuple[ModelEntry, ...] = ()
    others: tuple[str, ...] = Field(
        default=(), description="上游有、但没进推荐表的聊天模型（已过滤掉非聊天模型）"
    )
    checked_at: str = ""
    note: str = ""

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(m.id for m in self.recommended) + self.others


_CACHE: dict[str, tuple[float, tuple[str, ...]]] = {}
_LOCK = threading.RLock()


def _upstream_ids(settings, client=None, *, timeout: float = 10.0) -> tuple[str, ...]:
    """问上游要清单。失败一律抛，由上层的 fallback 接住——这里不吞异常。"""
    if client is None:
        from llmkit.client import make_client

        client = make_client(settings)
    # openai SDK 的 models.list() 返回的是分页对象，转 list 才能数长度。
    page = client.models.list(timeout=timeout)
    return tuple(str(m.id) for m in page)


def list_available(
    settings,
    *,
    client=None,
    refresh: bool = False,
    ttl: float = CACHE_TTL_S,
    now: float | None = None,
    upstream=None,
) -> ModelList:
    """列出可选模型。

    三种来源，前端要能区分（`source` 字段）：
      upstream      上游真实清单，可信
      fallback      上游问不到，只剩推荐表——仍然能切，但清单可能不是最新的
      unconfigured  连密钥都没有，只能展示推荐表，切了也用不了

    `upstream` 参数只为测试注入：给一个返回 tuple[str, ...] 的函数，
    就不用真联网。
    """
    current = getattr(settings, "model", "") or ""
    base_url = getattr(settings, "base_url", "") or ""
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    if settings is None:
        return ModelList(
            source=SOURCE_UNCONFIGURED,
            current=current,
            recommended=RECOMMENDED,
            checked_at=stamp,
            note="没有可用的密钥，下面只是内置推荐表；先配 .env 再切模型",
        )

    fetch = upstream or (lambda: _upstream_ids(settings, client))
    key = f"{base_url}|{getattr(settings, 'api_key', '')[-6:]}"
    clock = time.monotonic() if now is None else now

    with _LOCK:
        hit = _CACHE.get(key)
    if hit is not None and not refresh and clock - hit[0] < ttl:
        return _build(hit[1], current=current, base_url=base_url, stamp=stamp,
                      source=SOURCE_UPSTREAM, note="")

    try:
        ids = tuple(fetch())
    except Exception as exc:  # noqa: BLE001 - 上游任何失败都该降级，不该让界面崩
        return _build(
            tuple(m.id for m in RECOMMENDED),
            current=current,
            base_url=base_url,
            stamp=stamp,
            source=SOURCE_FALLBACK,
            note=f"拿不到上游模型清单（{type(exc).__name__}: {exc}），只显示内置推荐表",
            only_recommended=True,
        )

    with _LOCK:
        _CACHE[key] = (clock, ids)
    return _build(ids, current=current, base_url=base_url, stamp=stamp,
                  source=SOURCE_UPSTREAM, note="")


def _build(
    ids: tuple[str, ...],
    *,
    current: str,
    base_url: str,
    stamp: str,
    source: str,
    note: str,
    only_recommended: bool = False,
) -> ModelList:
    """把上游 id 列表拆成"推荐（有序）"和"其余聊天模型（字母序）"。"""
    present = set(ids)
    if only_recommended:
        # 拿不到上游清单时不能假装这些推荐"确认存在"，但也不该整页空白：
        # 保留推荐表，让用户至少知道该填什么，source 已经写明是 fallback。
        recommended = RECOMMENDED
        others: tuple[str, ...] = ()
    else:
        recommended = tuple(m for m in RECOMMENDED if m.id in present)
        others = tuple(sorted(i for i in present if i not in _BY_ID and is_chat_model(i)))

    if present and current and current not in present:
        extra = f"；注意 .env 里的默认模型 {current} 不在上游清单里"
        note = (note + extra) if note else extra.lstrip("；")
    return ModelList(
        source=source,
        base_url=base_url,
        current=current,
        recommended=recommended,
        others=others,
        checked_at=stamp,
        note=note,
    )


def clear_cache() -> None:
    """测试用：清掉进程内缓存。"""
    with _LOCK:
        _CACHE.clear()
