"""HTTP 层：状态码、错误体、以及"前端能不能只靠 error.hint 就说人话"。

这里不重复 service 的业务断言，只测**线上会坏的地方**：
multipart 解析、状态码、JSON 形状、删除后的 404。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmkit.config import Settings
from llmkit.tests.fakes import FakeClient, make_completion
from studio.api import create_app
from studio.service import Studio

NOTES = "\n\n".join(
    [
        "# 聚类分析",
        "系统聚类法先计算距离矩阵，再按最短距离合并两类。距离矩阵记录样品两两之间的距离。",
        "# 判别分析",
        "距离判别使用马氏距离。马氏距离要考虑协方差阵。",
    ]
)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    studio = Studio(
        tmp_path,
        settings=Settings(api_key="t", base_url="http://fake", model="fake-model"),
        client=FakeClient([make_completion("先算距离矩阵。【notes:p1】")]),
    )
    return TestClient(create_app(studio))


def _upload(c: TestClient, name: str = "notes.md", data: bytes | None = None):
    return c.post(
        "/api/docs",
        files={"file": (name, NOTES.encode("utf-8") if data is None else data, "text/markdown")},
    )


def test_首页能打开(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert "资料问答工作台" in r.text


def test_健康检查(client: TestClient):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["docs"] == 0


def test_上传成功返回_201_和解析摘要(client: TestClient):
    r = _upload(client)
    assert r.status_code == 201
    body = r.json()
    assert body["doc"]["doc_id"] == "notes"
    assert body["doc"]["pages"] >= 1
    assert body["terms"]


def test_重复上传返回_409_且带_doc_id(client: TestClient):
    _upload(client)
    r = _upload(client, "again.md")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "duplicate_doc"


def test_不支持的格式返回_415(client: TestClient):
    r = client.post("/api/docs", files={"file": ("a.docx", b"PK\x03\x04", "application/octet-stream")})
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_type"


def test_空文件返回_400(client: TestClient):
    r = _upload(client, "empty.txt", b"")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_file"


def test_假_PDF_返回_422(client: TestClient):
    r = client.post("/api/docs", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "corrupt_file"


def test_术语接口带出处(client: TestClient):
    _upload(client)
    body = client.get("/api/docs/notes/terms?top_n=10").json()
    assert body["terms"]
    assert all(t["pages"] for t in body["terms"])


def test_未生成画像时返回_null(client: TestClient):
    _upload(client)
    assert client.get("/api/docs/notes/profile").json()["profile"] is None


def test_生成画像接口(client: TestClient):
    _upload(client)
    studio = client.app.state.studio
    studio.llm = lambda _p: json.dumps(
        {
            "domain": "多元统计",
            "summary": "聚类与判别",
            "terms": [{"term": "距离矩阵", "note": ""}],
            "phrasings": [{"question": "系统聚类法第一步", "terms": ["距离矩阵"]}],
        },
        ensure_ascii=False,
    )
    body = client.post("/api/docs/notes/profile").json()["profile"]
    assert body["domain"] == "多元统计"
    assert body["terms"][0]["term"] == "距离矩阵"
    assert body["source"] == "llm"


def test_提问接口的响应形状(client: TestClient):
    _upload(client)
    body = client.post("/api/ask", json={"question": "系统聚类法第一步做什么", "top_k": 3}).json()
    for key in ("answer", "citations", "unsupported", "denied", "evidence", "translated"):
        assert key in body
    assert body["answer"]
    assert body["translated"] is False


def test_空问题返回_400(client: TestClient):
    _upload(client)
    r = client.post("/api/ask", json={"question": "  "})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_question"


def test_top_k_越界返回_400(client: TestClient):
    _upload(client)
    r = client.post("/api/ask", json={"question": "问题", "top_k": 99})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_top_k"


def test_删除后列表为空再问返回_404(client: TestClient):
    _upload(client)
    assert client.delete("/api/docs/notes").status_code == 200
    assert client.get("/api/docs").json()["docs"] == []
    r = client.post("/api/ask", json={"question": "系统聚类法", "doc_ids": ["notes"]})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "doc_not_found"


def test_模型侧失败返回_503_而不是裸_500(tmp_path: Path):
    """实测来源：余额不足时上游 402，没有处理器就是 500 + 空响应体。

    前端 `api()` 只能拿到 HTTP 500，用户看不到任何可行动的提示。
    现在统一翻译成 {"error": {code, message, hint}}，前端照旧只读 error.hint。
    """
    from llmkit.errors import FatalError

    studio = Studio(
        tmp_path,
        settings=Settings(api_key="t", base_url="http://fake", model="fake-model"),
        client=FakeClient([FatalError("请求本身有问题 402，重试无意义", status=402)]),
    )
    c = TestClient(create_app(studio))
    doc_id = _upload(c).json()["doc"]["doc_id"]

    r = c.post(f"/api/docs/{doc_id}/selfcheck?n_terms=1&n_negative=1")

    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "llm_error"
    assert "余额" in err["hint"]


def test_模型清单接口能打开(client: TestClient):
    """假客户端没有 models 接口，等于"上游清单端点不可用"。
    界面必须还能开，所以这里断言的是 fallback 而不是 200 + 真清单。"""
    body = client.get("/api/models").json()

    assert body["source"] == "fallback"
    assert body["current"] == "fake-model"
    assert body["recommended"]
    assert all("id" in m and "max_tokens" in m for m in body["recommended"])
    assert body["note"]


def test_问答接口把模型透传下去(client: TestClient):
    doc_id = _upload(client).json()["doc"]["doc_id"]
    r = client.post(
        "/api/ask",
        json={"question": "系统聚类法第一步", "doc_ids": [doc_id], "model": "qwen3.8-max"},
    )

    assert r.status_code == 200
    assert r.json()["model"] == "qwen3.8-max"


def test_不传模型时回落到默认(client: TestClient):
    _upload(client)
    body = client.post("/api/ask", json={"question": "系统聚类法第一步"}).json()

    assert body["model"] == "fake-model"


def test_模型名传空串等于不指定(client: TestClient):
    """前端 select 没选时容易发出 model=""，不能因为空串就报"模型不存在"。"""
    _upload(client)
    body = client.post(
        "/api/ask", json={"question": "系统聚类法第一步", "model": "   "}
    ).json()

    assert body["model"] == "fake-model"
