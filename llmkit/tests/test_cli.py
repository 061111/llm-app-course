"""M1-T1.7 验收：命令行。

验收重点不是"能打出字"，而是**退出码**——脚本和 CI 靠退出码判断成败。
    0 成功 / 1 调用失败 / 2 配置缺失
"""
from __future__ import annotations

import json


from llmkit import __main__ as cli
from llmkit.config import ENV_API_KEY
from llmkit.errors import FatalError
from llmkit.types import AskResult, Usage


def test_没有密钥时退出码是_2(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.chdir(tmp_path)          # 切到没有 .env 的空目录

    code = cli.main(["随便问点什么"])

    assert code == cli.EXIT_CONFIG
    assert ENV_API_KEY in capsys.readouterr().err


def test_check_模式只验配置不发请求(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv(ENV_API_KEY, "sk-test12345678")
    monkeypatch.chdir(tmp_path)

    code = cli.main(["--check"])

    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "***5678" in out
    assert "sk-test12345678" not in out      # 完整密钥不能出现在输出里


def test_没给问题时提示用法(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv(ENV_API_KEY, "sk-test12345678")
    monkeypatch.chdir(tmp_path)

    code = cli.main([])

    assert code == cli.EXIT_CONFIG
    assert "请给一个问题" in capsys.readouterr().err


def test_成功时打印答案与用量(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv(ENV_API_KEY, "sk-test12345678")
    monkeypatch.chdir(tmp_path)

    fake_result = AskResult(
        text="贝叶斯定理讲的是条件概率。",
        model="fake-model",
        latency_ms=1834.2,
        usage=Usage(prompt_tokens=128, completion_tokens=96, cost_cny=0.0007),
        attempts=1,
        trace_id="abc123def456",
    )
    monkeypatch.setattr(cli, "ask", lambda *a, **k: fake_result)

    code = cli.main(["贝叶斯定理是什么"])

    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "贝叶斯定理讲的是条件概率。" in out
    assert "128 in / 96 out" in out
    assert "1834ms" in out


def test_json_模式输出可解析的_JSON(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv(ENV_API_KEY, "sk-test12345678")
    monkeypatch.chdir(tmp_path)

    fake_result = AskResult(text="答案", model="m", latency_ms=1.0, trace_id="t")
    monkeypatch.setattr(cli, "ask", lambda *a, **k: fake_result)

    code = cli.main(["--json", "问题"])
    payload = json.loads(capsys.readouterr().out)

    assert code == cli.EXIT_OK
    assert payload["text"] == "答案"
    assert payload["model"] == "m"


def test_调用失败时退出码是_1(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv(ENV_API_KEY, "sk-test12345678")
    monkeypatch.chdir(tmp_path)

    def boom(*args, **kwargs):
        raise FatalError("密钥无效", status=401)

    monkeypatch.setattr(cli, "ask", boom)

    code = cli.main(["问题"])

    assert code == cli.EXIT_CALL_FAILED
    assert "401" in capsys.readouterr().err
