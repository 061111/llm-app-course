"""余额查询的离线测试。

全部用注入的 transport，不联网——余额功能的价值在于"数字对不对"，
而"能不能连上"不该由测试来保证（它取决于部署环境有没有出网）。
"""
from __future__ import annotations

import json

import pytest
import urllib.parse

from llmkit.config import Settings
from studio import balance

AK = Settings(api_key="sk-x", aliyun_ak_id="LTAI-test", aliyun_ak_secret="secret-test")


def _coupon(bid, nominal, bal, status="Available", expiry="2027-10-20T00:00:00Z"):
    return {
        "CashCouponId": bid,
        "CashCouponNo": f"Q-{bid}",
        "NominalValue": str(nominal),
        "Balance": str(bal),
        "Status": status,
        "ExpiryTime": expiry,
        "ApplicableProducts": "通用",
        "ApplicableScenarios": "阿里云按量付费账单",
        "Description": "",
    }


def _ok_body(coupons):
    return json.dumps(
        {"Code": "Success", "Message": "Successful！", "Success": True,
         "Data": {"CashCoupon": coupons}},
        ensure_ascii=False,
    )


def _transport_returning(body, status=200, calls=None):
    def send(url, timeout):
        if calls is not None:
            calls.append(url)
        return status, body

    return send


def test_没配_AK_时说明怎么配而不是报错():
    """未配置不是错误：它跟问答没关系，不该渲染成红色失败。"""
    balance.clear_cache()
    sent = []
    snap = balance.query(Settings(api_key="sk-x"), transport=_transport_returning("{}", calls=sent))

    assert snap.configured is False
    assert snap.ok is False
    assert sent == []  # 没配就不该发请求
    assert "ALIYUN_ACCESS_KEY_ID" in snap.hint
    assert "AliyunBSSReadOnlyAccess" in snap.hint


def test_签名串按参数名排序且带Signature():
    q = balance.signed_query(
        {"Action": "QueryCashCoupons"},
        ak_id="LTAI-test",
        ak_secret="secret-test",
        nonce="n1",
        timestamp="2026-11-18T08:00:00Z",
    )
    # 编码规则：空格/加号必须变成 %20 / %2B，不能出现裸 "+"（那是表单编码）
    assert "%3A" in q
    assert "+" not in q
    # Signature 是最后拼上去的，不算进待签串；只检查它前面那段有序
    canonical, _, signature = q.partition("&Signature=")
    keys = [p.split("=", 1)[0] for p in canonical.split("&")]
    assert keys == sorted(keys), "参数必须按名字排序，否则签名对不上"
    assert signature, "缺 Signature"
    for required in ("Action", "Version", "AccessKeyId", "SignatureMethod",
                     "SignatureVersion", "SignatureNonce", "Timestamp", "Signature"):
        assert required in q, required


def test_同样输入得到同样签名():
    """签名不能带随机性（除了 nonce/timestamp）；这里两者都给定。"""
    kwargs = dict(ak_id="LTAI-test", ak_secret="secret-test",
                  nonce="n1", timestamp="2026-11-18T08:00:00Z")
    assert balance.signed_query({"Action": "QueryCashCoupons"}, **kwargs) == balance.signed_query(
        {"Action": "QueryCashCoupons"}, **kwargs
    )


def test_查到券_只把可用的算进合计():
    balance.clear_cache()
    body = _ok_body([
        _coupon(1, 300, 299.7169655471),
        _coupon(2, 100, 0, status="Expired"),
        _coupon(3, 50, 12.5, status="Available"),
    ])
    snap = balance.query(AK, transport=_transport_returning(body))

    assert snap.ok is True
    assert len(snap.coupons) == 3          # 过期的也列出来，方便对照
    # 合计保留 6 位小数（金额不需要更多）；只加 Available 的那两张
    assert snap.total_balance == pytest.approx(312.2169655471, abs=1e-6)
    assert snap.total_nominal == 350.0

    payload = snap.to_payload()
    assert payload["available_count"] == 2
    assert payload["coupons"][1]["status_text"] == "已过期"
    assert payload["error"] is None


def test_上游报错时把原始_code_与_message_透传():
    """第一次配 AK 时，用户最需要看到的就是阿里云自己那句话。"""
    balance.clear_cache()
    body = json.dumps({"Code": "NoPermission", "Message": "You are not authorized to perform this action."})
    snap = balance.query(AK, transport=_transport_returning(body, status=400))

    assert snap.ok is False
    assert snap.error_code == "NoPermission"
    assert "not authorized" in snap.error_message
    assert "AliyunBSSReadOnlyAccess" in snap.hint


def test_主账号才能调时会提示换主账号():
    balance.clear_cache()
    body = json.dumps({"Code": "NotApplicable", "Message": "This API is not applicable for caller."})
    snap = balance.query(AK, transport=_transport_returning(body, status=400))

    assert "主账号" in snap.hint


def test_网络不通也返回结果不抛异常():
    balance.clear_cache()

    def boom(url, timeout):
        raise TimeoutError("timed out")

    snap = balance.query(AK, transport=boom)
    assert snap.configured is True and snap.ok is False
    assert snap.error_code == "network_error"
    assert "business.aliyuncs.com" in snap.hint


def test_返回不是_JSON_时带上状态码():
    balance.clear_cache()
    snap = balance.query(AK, transport=_transport_returning("<html>502</html>", status=502))

    assert snap.ok is False
    assert snap.error_code == "http_502"


def test_缓存期内不重复打上游_refresh_强制重查():
    balance.clear_cache()
    calls: list[str] = []
    body = _ok_body([_coupon(1, 300, 300)])

    balance.query(AK, transport=_transport_returning(body, calls=calls), now=1000.0)
    balance.query(AK, transport=_transport_returning(body, calls=calls), now=1010.0)
    assert len(calls) == 1

    balance.query(AK, transport=_transport_returning(body, calls=calls), refresh=True, now=1011.0)
    assert len(calls) == 2


def test_金额字段是字符串_也能算():
    """接口返回的是 "100.00" 这种字符串；直接相加会得到 "0100.0050.00" 这种鬼东西。"""
    balance.clear_cache()
    snap = balance.query(AK, transport=_transport_returning(_ok_body([_coupon(1, 100, 99.5)])))
    assert isinstance(snap.total_balance, float)
    assert snap.total_balance == 99.5


def test_没有可用券时给一句解释():
    balance.clear_cache()
    snap = balance.query(AK, transport=_transport_returning(_ok_body([_coupon(1, 100, 0, status="Expired")])))

    assert snap.ok is True
    assert snap.total_balance == 0.0
    assert "Available" in snap.hint


def test_请求里带的是代金券接口而不是账户余额接口():
    """QueryAccountBalance 不含代金券（官方文档明说）。调错了会显示 ¥0，最容易被当成"券没了"。"""
    balance.clear_cache()
    calls: list[str] = []
    balance.query(AK, transport=_transport_returning(_ok_body([]), calls=calls))

    assert len(calls) == 1
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(calls[0]).query)
    assert qs["Action"] == ["QueryCashCoupons"]
    assert qs["Version"] == ["2017-12-14"]
