"""服务层：上传 → 解析 → 画像 → 问答的串联，以及缓存与缓存失效。

用 .txt 资料而不是 PDF：这条路径测的是"编排"，不是 PDF 解析
（后者在 corpus/tests 和 studio/tests/test_ingest.py 里）。
好处是测试快、零外部文件依赖。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmkit.config import Settings
from llmkit.tests.fakes import FakeClient, make_completion
from studio.errors import DocNotFound, DuplicateDoc, StudioError
from studio.service import Studio

NOTES = "\n\n".join(
    [
        "# 第三章 聚类分析",
        "系统聚类法先计算距离矩阵，再按最短距离合并两类。距离矩阵记录样品两两之间的距离。",
        "# 第四章 判别分析",
        "距离判别使用马氏距离。马氏距离要考虑协方差阵，所以协差阵必须先估出来。",
        "# 第五章 主成分分析",
        "主成分分析用来降维。第一主成分的方差贡献率最大，用它概括原变量的大部分信息。",
    ]
)


def _settings() -> Settings:
    return Settings(api_key="test", base_url="http://fake", model="fake-model")


def _studio(tmp_path: Path, answer: str = "系统聚类法先算距离矩阵。【notes:p1】") -> Studio:
    return Studio(
        tmp_path,
        settings=_settings(),
        client=FakeClient([make_completion(answer)]),
    )


def _last_call(s: Studio) -> dict:
    """最近一次请求的参数。问的是"实际发出去的请求长什么样"，
    而不是"我们以为它会用哪个模型"——后者正是切模型最容易出错的地方。"""
    calls = s.client.calls
    assert calls, "一次模型调用都没发生"
    return calls[-1]


def calls_model(s: Studio) -> str:
    """最近一次请求实际用的模型名。写进 Studio 里会污染被测代码，所以留在测试侧。"""
    return s.client.calls[-1]["model"]


def calls_max_tokens(s: Studio) -> int:
    return s.client.calls[-1]["max_tokens"]


Studio.calls_model = calls_model
Studio.calls_max_tokens = calls_max_tokens


def test_上传后能列出并抽到术语(tmp_path: Path):
    s = _studio(tmp_path)
    r = s.upload("notes.md", NOTES.encode("utf-8"))
    assert r.doc.doc_id == "notes"
    assert r.doc.pages >= 1 and r.doc.chunks >= 1
    assert [d.doc_id for d in s.docs()] == ["notes"]
    assert any("距离" in t for t in r.terms)


def test_重复上传同一份内容被拒(tmp_path: Path):
    s = _studio(tmp_path)
    s.upload("notes.md", NOTES.encode("utf-8"))
    with pytest.raises(DuplicateDoc):
        s.upload("copy.md", NOTES.encode("utf-8"))


def test_空资料库提问给出可操作的提示(tmp_path: Path):
    s = _studio(tmp_path)
    with pytest.raises(StudioError) as exc:
        s.ask("随便问点什么")
    assert exc.value.code == "empty_library"
    assert "上传" in exc.value.hint


def test_中文资料不走翻译(tmp_path: Path):
    s = _studio(tmp_path)
    s.upload("notes.md", NOTES.encode("utf-8"))
    view = s.ask("系统聚类法的第一步是什么", top_k=3)
    assert view.translated is False
    assert view.evidence, "中文问题必须能召回中文块"
    assert view.answer.text


def test_答案里的引用被保留并校验(tmp_path: Path):
    s = _studio(tmp_path, answer="先算距离矩阵。【notes:p1】")
    s.upload("notes.md", NOTES.encode("utf-8"))
    view = s.ask("系统聚类法第一步")
    assert view.answer.cited_sources == ("notes:p1",)
    assert view.answer.unsupported == ()


def test_引用越界会被标出来(tmp_path: Path):
    s = _studio(tmp_path, answer="见【notes:p99】")
    s.upload("notes.md", NOTES.encode("utf-8"))
    view = s.ask("系统聚类法第一步")
    assert view.answer.unsupported == ("notes:p99",)


def test_模型说资料未覆盖时判定为拒答(tmp_path: Path):
    s = _studio(tmp_path, answer="资料未覆盖这个问题。")
    s.upload("notes.md", NOTES.encode("utf-8"))
    view = s.ask("量子纠缠怎么算")
    assert view.denied is True


def test_画像生成后落盘且第二次不再调模型(tmp_path: Path):
    s = _studio(tmp_path)
    s.upload("notes.md", NOTES.encode("utf-8"))

    calls = {"n": 0}

    def fake_llm(_prompt: str) -> str:
        calls["n"] += 1
        return json.dumps(
            {
                "domain": "多元统计",
                "summary": "聚类、判别与主成分",
                "terms": [{"term": "距离矩阵", "note": ""}],
                "phrasings": [{"question": "系统聚类法第一步做什么", "terms": ["距离矩阵"]}],
            },
            ensure_ascii=False,
        )

    s.llm = fake_llm
    first = s.profile("notes")
    assert first.source == "llm" and calls["n"] == 1
    second = s.profile("notes")
    assert calls["n"] == 1, "第二次应读磁盘，不再调模型"
    assert second.term_texts == first.term_texts
    assert s.docs()[0].has_profile is True


def test_刷新画像会重新调模型(tmp_path: Path):
    s = _studio(tmp_path)
    s.upload("notes.md", NOTES.encode("utf-8"))
    calls = {"n": 0}

    def fake_llm(_prompt: str) -> str:
        calls["n"] += 1
        return json.dumps(
            {"domain": "统计", "summary": "s", "terms": [{"term": "距离矩阵", "note": ""}]},
            ensure_ascii=False,
        )

    s.llm = fake_llm
    s.profile("notes")
    s.profile("notes", refresh=True)
    assert calls["n"] == 2


def test_画像会注入到问答_prompt(tmp_path: Path):
    s = _studio(tmp_path)
    s.upload("notes.md", NOTES.encode("utf-8"))
    s.llm = lambda _p: json.dumps(
        {
            "domain": "多元统计",
            "summary": "s",
            "terms": [{"term": "距离矩阵", "note": ""}],
        },
        ensure_ascii=False,
    )
    s.profile("notes")
    view = s.ask("系统聚类法第一步")
    assert view.profile_source == "llm"
    sent = s.client.calls[-1]["messages"]
    joined = " ".join(m["content"] for m in sent)
    assert "多元统计" in joined
    assert "距离矩阵" in joined


def test_删除后检索器缓存失效(tmp_path: Path):
    s = _studio(tmp_path)
    s.upload("notes.md", NOTES.encode("utf-8"))
    s.ask("系统聚类法第一步")
    s.remove("notes")
    assert s.docs() == []
    with pytest.raises(StudioError):
        s.ask("系统聚类法第一步")


def test_指定不存在的_doc_id_提问会报错(tmp_path: Path):
    s = _studio(tmp_path)
    s.upload("notes.md", NOTES.encode("utf-8"))
    with pytest.raises(DocNotFound):
        s.ask("问题", doc_ids=["nope"])


def test_空问题被拒(tmp_path: Path):
    s = _studio(tmp_path)
    s.upload("notes.md", NOTES.encode("utf-8"))
    with pytest.raises(StudioError) as exc:
        s.ask("   ")
    assert exc.value.code == "empty_question"


def test_多份资料可以一起检索(tmp_path: Path):
    s = _studio(tmp_path)
    s.upload("a.md", NOTES.encode("utf-8"))
    s.upload("b.md", (NOTES + "\n\n因子分析用公共因子解释变量间的相关性。").encode("utf-8"))
    view = s.ask("因子分析干什么", doc_ids=["a", "b"], top_k=4)
    assert set(view.doc_ids) == {"a", "b"}
    assert view.evidence


def test_部分覆盖不算拒答(tmp_path: Path):
    """模型答了资料里有的部分、再补一句"某细节未覆盖"，这条答案仍然有效。"""
    s = _studio(
        tmp_path,
        answer="系统聚类法先算距离矩阵。【notes:p1】资料未覆盖 KMO 的具体形式。",
    )
    s.upload("notes.md", NOTES.encode("utf-8"))
    view = s.ask("系统聚类法第一步")
    assert view.denied is False


# ---------------------------------------------------------------- 模型切换


def test_默认不指定模型时用配置里的那个(tmp_path: Path):
    s = _studio(tmp_path)
    s.upload("notes.md", NOTES.encode("utf-8"))
    view = s.ask("系统聚类法第一步")

    assert _last_call(s)["model"] == "fake-model"
    assert view.model == "fake-model"


def test_指定模型时请求真的换了模型(tmp_path: Path):
    """这条测的是"别把 model 参数收下就丢掉"——
    最容易犯的错是签名加了、调用链没透传，界面显示换了模型其实没换。"""
    s = _studio(tmp_path)
    s.upload("notes.md", NOTES.encode("utf-8"))
    view = s.ask("系统聚类法第一步", model="qwen3.8-max")

    assert _last_call(s)["model"] == "qwen3.8-max"
    assert view.model == "qwen3.8-max"


def test_推理型模型自动放大输出预算(tmp_path: Path):
    """实测：deepseek-v4-pro 在 max_tokens=800 下正文为空（reasoning 吃满）。

    如果这里不放大，用户在界面上选它，每个问题都会走"空正文重试三次"再失败，
    看到的是"模型坏了"，而不是"这个模型要想一会儿"。
    """
    s = _studio(tmp_path)
    s.upload("notes.md", NOTES.encode("utf-8"))
    s.ask("系统聚类法第一步", model="deepseek-v4-pro")

    assert _last_call(s)["max_tokens"] == 4000


def test_非推理型模型沿用默认预算(tmp_path: Path):
    s = _studio(tmp_path)
    s.upload("notes.md", NOTES.encode("utf-8"))
    s.ask("系统聚类法第一步", model="qwen3.8-flash")

    assert _last_call(s)["max_tokens"] == 1600


def test_切模型不改密钥和地址(tmp_path: Path):
    """settings_for 必须是"复制一份改 model"，不能是"换掉整个配置"。
    换掉的话密钥就丢了（表现为 401），而且很难看出跟切模型有关系。"""
    s = _studio(tmp_path)
    cfg = s.settings_for("qwen3.8-max")

    assert cfg.model == "qwen3.8-max"
    assert cfg.api_key == "test"
    assert cfg.base_url == "http://fake"
    assert s.settings.model == "fake-model"  # 原配置没被改


def test_没有配置时切模型不会崩(tmp_path: Path):
    s = Studio(tmp_path)
    s.settings = None
    assert s.settings_for("qwen3.8-max") is None


def test_清单拉不到时降级而不是抛异常(tmp_path: Path):
    """假客户端没有 models 接口，正好模拟"上游清单端点不可用"。"""
    s = _studio(tmp_path)
    info = s.models()

    assert info.source == "fallback"
    assert info.recommended


# ---------------------------------------------------------------- 模型切换
