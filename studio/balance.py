"""M8 · 代金券余额：把控制台里那张券的余额搬到界面上。

【为什么不能用模型密钥查】
  百炼的 `sk-...` 是**模型调用**的凭据，能证明"这次推理该算在哪个账号头上"，
  但不能读账单。查余额走的是另一个产品：BssOpenApi（费用与成本），
  鉴权用阿里云 AccessKey（AK/SK），并且是 **RAM 用户的另一套权限**。
  所以这里必须另配一对 AK，配不上就明说"没配"，不能拿模型密钥硬试——
  那只会得到一条看不懂的 401，然后让人以为"余额功能坏了"。

【查哪个接口】
  `QueryCashCoupons`（2017-12-14 版）：返回代金券列表，每张带面额、余额、状态、到期时间。
  `QueryAccountBalance` 是**账户可用额度**，官方明说它**不含代金券**，
  所以只调它会出现"界面显示 ¥0，但控制台明明有 300 券"的假象。两个数字必须分开说。

【签名为什么自己实现】
  阿里云官方 SDK（alibabacloud_bssopenapi）会拖进来十几个依赖，
  而这里只需要一个 GET 请求 + HMAC-SHA1。RPC 风格签名（v1.0）的规则是固定的：
  参数按名字排序 → 百分号编码 → `GET&%2F&编码后的串` → HMAC-SHA1(secret + "&") → base64。
  自己写 30 行，换来零依赖，也换来"能测"（见 studio/tests/test_balance.py）。

【验证到哪一步】
  2026-11-18 用一对**假 AK** 打过这两个接口：返回 `InvalidAccessKeyId.NotFound`
  而不是 `SignatureDoesNotMatch`，说明端点、Action、签名格式都被服务端认了，
  剩下的只是密钥真不真。**没有用真实 AK 跑通过**——所以第一次配置时，
  错误信息一律原样回传上游的 Code/Message，不自己编解释。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

BSS_VERSION = "2017-12-14"
BSS_ENDPOINT = "https://business.aliyuncs.com/"

# 余额变化是"分"级的（每问一次几厘钱），30 秒缓存足够挡住连点刷新，
# 又不至于让人以为按钮没生效。
CACHE_TTL_S = 30.0

STATUS_TEXT = {
    "Available": "可用",
    "Expired": "已过期",
    "Cancelled": "已作废",
}


class Coupon(BaseModel):
    """一张代金券。字段名照抄接口，避免"我给它起了个新名字然后对不上文档"。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = ""
    no: str = ""
    nominal: float = Field(default=0.0, ge=0.0, description="面额")
    balance: float = Field(default=0.0, ge=0.0, description="余额")
    status: str = ""
    expiry: str = ""
    products: str = Field(default="", description="适用产品")
    scenarios: str = Field(default="", description="适用场景")
    description: str = ""

    @property
    def status_text(self) -> str:
        return STATUS_TEXT.get(self.status, self.status or "未知")


class Snapshot(BaseModel):
    """一次查询的结果。`configured=False` 时其余字段都无意义。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    configured: bool = False
    ok: bool = False
    coupons: tuple[Coupon, ...] = ()
    total_balance: float = Field(default=0.0, ge=0.0, description="所有可用券余额合计")
    total_nominal: float = Field(default=0.0, ge=0.0, description="所有可用券面额合计")
    checked_at: str = ""
    error_code: str = ""
    error_message: str = ""
    hint: str = ""

    def to_payload(self) -> dict:
        return {
            "configured": self.configured,
            "ok": self.ok,
            "coupons": [c.model_dump() | {"status_text": c.status_text} for c in self.coupons],
            "total_balance": round(self.total_balance, 6),
            "total_nominal": round(self.total_nominal, 6),
            "available_count": sum(1 for c in self.coupons if c.status == "Available"),
            "checked_at": self.checked_at,
            # hint 单独放一份：未配置 AK 时没有 error，但那句"怎么配"必须传下去。
            "hint": self.hint,
            "error": (
                {"code": self.error_code, "message": self.error_message, "hint": self.hint}
                if self.error_code or self.error_message
                else None
            ),
        }


def _pct(value: object) -> str:
    """阿里云要求的百分号编码：只保留 `~`，空格要变成 %20 而不是 +。"""
    return urllib.parse.quote(str(value), safe="~")


def signed_query(
    params: dict[str, str],
    *,
    ak_id: str,
    ak_secret: str,
    nonce: str | None = None,
    timestamp: str | None = None,
) -> str:
    """拼出带签名的查询串（RPC 风格，SignatureVersion=1.0）。"""
    all_params = dict(params)
    all_params.update(
        {
            "Format": "JSON",
            "Version": BSS_VERSION,
            "AccessKeyId": ak_id,
            "SignatureMethod": "HMAC-SHA1",
            "SignatureVersion": "1.0",
            "SignatureNonce": nonce or uuid.uuid4().hex,
            "Timestamp": timestamp
            or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    canonical = "&".join(f"{_pct(k)}={_pct(v)}" for k, v in sorted(all_params.items()))
    string_to_sign = f"GET&{_pct('/')}&{_pct(canonical)}"
    digest = hmac.new(
        (ak_secret + "&").encode(), string_to_sign.encode(), hashlib.sha1
    ).digest()
    signature = base64.b64encode(digest).decode()
    return f"{canonical}&Signature={_pct(signature)}"


def _default_transport(url: str, timeout: float) -> tuple[int, str]:
    """发一个 GET，返回 (状态码, 响应体)。4xx/5xx 也返回，不抛——错误体里有 Code。"""
    # 关掉代理：本机代理常把 CONNECT 弄成 502，而账单接口是直连可达的。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(urllib.request.Request(url), timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _num(value: object) -> float:
    """接口里的金额都是字符串（"100.00"）。转不动就当 0，不要让整页报表崩在这。"""
    try:
        return max(0.0, float(str(value).strip()))
    except (TypeError, ValueError):
        return 0.0


_HINTS = {
    "InvalidAccessKeyId.NotFound": "AK 不存在。检查 ALIYUN_ACCESS_KEY_ID 有没有多空格、是不是别的账号的",
    "SignatureDoesNotMatch": "AK/SK 对不上。常见原因：SK 复制少了几位，或者贴成了另一个 RAM 用户的",
    "NoPermission": "这个 RAM 用户没有查券的权限。给它挂 AliyunBSSReadOnlyAccess（只读，够用）",
    "NotAuthorized": "当前身份不许调账单接口。给 RAM 用户挂 AliyunBSSReadOnlyAccess",
    "NotApplicable": "这种账号类型不能这样调账单接口。改用主账号（root）的 AK",
    "Forbidden.RAM": "RAM 权限不足：需要 bss:DescribeCoupons",
}


def _hint_for(code: str) -> str:
    for key, hint in _HINTS.items():
        if code.startswith(key) or key in code:
            return hint
    return "看 error.message 里的原文；对不上就先去费用与成本控制台确认这张券还在不在"


_CACHE: dict[str, tuple[float, Snapshot]] = {}
_LOCK = threading.RLock()


def query(
    settings,
    *,
    refresh: bool = False,
    transport=None,
    timeout: float = 10.0,
    now: float | None = None,
    ttl: float = CACHE_TTL_S,
) -> Snapshot:
    """查代金券。`transport` 只为测试注入，生产不传。

    三种结果都返回 Snapshot，不抛异常：余额查不到不该让整个页面打不开。
      configured=False  没配 AK（前端显示"怎么配"而不是"出错了"）
      ok=True           查到券
      ok=False          配了 AK 但上游拒绝（原样回传上游 Code/Message）
    """
    ak_id = (getattr(settings, "aliyun_ak_id", "") or "").strip()
    ak_secret = (getattr(settings, "aliyun_ak_secret", "") or "").strip()
    stamp = datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()

    if not ak_id or not ak_secret:
        return Snapshot(
            configured=False,
            checked_at=stamp,
            hint=(
                "查余额要用阿里云 AccessKey（不是模型密钥）。在 RAM 控制台建一个只读用户，"
                "挂 AliyunBSSReadOnlyAccess，把 AK/SK 填进 .env 的 "
                "ALIYUN_ACCESS_KEY_ID / ALIYUN_ACCESS_KEY_SECRET"
            ),
        )

    endpoint = (getattr(settings, "aliyun_bss_endpoint", "") or "").strip() or BSS_ENDPOINT
    key = f"{endpoint}|{ak_id[-6:]}"
    clock = time.monotonic() if now is None else now

    with _LOCK:
        hit = _CACHE.get(key)
    if hit is not None and not refresh and clock - hit[0] < ttl:
        return hit[1]

    url = endpoint.rstrip("/") + "/?" + signed_query(
        {"Action": "QueryCashCoupons"}, ak_id=ak_id, ak_secret=ak_secret
    )
    send = transport or _default_transport
    try:
        status, body = send(url, timeout)
    except Exception as exc:  # noqa: BLE001 - 网络层的任何失败都该变成一句人话
        snap = Snapshot(
            configured=True,
            checked_at=stamp,
            error_code="network_error",
            error_message=f"{type(exc).__name__}: {exc}",
            hint="本机到 business.aliyuncs.com 不通；先确认能不能打开费用与成本控制台",
        )
        with _LOCK:
            _CACHE[key] = (clock, snap)
        return snap

    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        snap = Snapshot(
            configured=True,
            checked_at=stamp,
            error_code=f"http_{status}",
            error_message=body[:300],
            hint="返回的不是 JSON，通常是网关拦截或域名被改写",
        )
        with _LOCK:
            _CACHE[key] = (clock, snap)
        return snap

    code = str(data.get("Code") or "")
    if code and code not in ("200", "Success"):
        snap = Snapshot(
            configured=True,
            checked_at=stamp,
            error_code=code,
            error_message=str(data.get("Message") or ""),
            hint=_hint_for(code),
        )
        with _LOCK:
            _CACHE[key] = (clock, snap)
        return snap

    rows = ((data.get("Data") or {}).get("CashCoupon")) or []
    coupons = tuple(
        Coupon(
            id=str(r.get("CashCouponId") or ""),
            no=str(r.get("CashCouponNo") or ""),
            nominal=_num(r.get("NominalValue")),
            balance=_num(r.get("Balance")),
            status=str(r.get("Status") or ""),
            expiry=str(r.get("ExpiryTime") or ""),
            products=str(r.get("ApplicableProducts") or ""),
            scenarios=str(r.get("ApplicableScenarios") or ""),
            description=str(r.get("Description") or ""),
        )
        for r in rows
        if isinstance(r, dict)
    )
    usable = [c for c in coupons if c.status == "Available"]
    snap = Snapshot(
        configured=True,
        ok=True,
        coupons=coupons,
        total_balance=round(sum(c.balance for c in usable), 6),
        total_nominal=round(sum(c.nominal for c in usable), 6),
        checked_at=stamp,
        hint="" if usable else "没有状态为 Available 的券；过期或作废的券也列在下面，方便对照",
    )
    with _LOCK:
        _CACHE[key] = (clock, snap)
    return snap


def clear_cache() -> None:
    """测试用：清掉进程内缓存。"""
    with _LOCK:
        _CACHE.clear()
