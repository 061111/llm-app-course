# llmkit · 项目 A 的模型调用层（M1）

**一句话**：项目里所有"调模型"的动作都必须经过这里，业务代码永远不 import `openai`。

它在项目 A 里的位置：

```
用户提问
   ↓
[检索层]  ← M3 才做
   ↓ 拼上下文
[llmkit]  ← 现在这一层
   ↓
模型 API
```

M2/M3/M4 加东西时，这一层一个字都不用改——这就是 M1 要先做的原因。

---

## 结构

```
llmkit/
├── types.py     AskResult / AskOptions / Usage / Citation    ← 契约
├── config.py    Settings：密钥、地址、超时、单价             ← 配置
├── errors.py    classify + retry_call + 5 个异常类           ← 异常与重试
├── parsing.py   抽 JSON + 校验 + 错误回灌                    ← 结构化输出
├── obs.py       一行一个 JSON 的日志 + trace id              ← 观测
├── client.py    ask / ask_structured / stream_ask            ← 唯一 import openai 的文件
└── __main__.py  python -m llmkit                             ← 命令行
```

依赖方向是单向的：`client` 依赖其它所有模块，其它模块之间不互相依赖。
所以 `types.py` 可以被任何层引用而不会引入循环。

---

## 快速开始

**1. 配密钥**（只做一次）

```bash
cp .env.example .env
# 编辑 .env，把 LLMKIT_API_KEY 那行改成你的密钥
```

**2. 验证配置**

```bash
python -m llmkit --check
# 配置可用：base_url=https://api.deepseek.com/v1 model=deepseek-flash key=***a571
```

**3. 真跑一次**

```bash
python -m llmkit "用一句话解释什么是过拟合"
# 答案: 过拟合是指模型把训练数据中的噪声和特例也学了进去，导致在训练集上表现很好、在新数据上泛化能力却变差。
# tokens: 50 in / 54 out  |  耗时 3017ms  |  成本 ¥0.000536  |  尝试 1 次
```

**4. 流式**

```bash
python -m llmkit --stream "讲三句话"
```

**5. 看日志**

```bash
python -m llmkit --debug "问题"
# {"ts": "...", "level": "INFO", "logger": "llmkit", "event": "llm.call",
#  "trace_id": "a1b2c3d4e5f6", "status": "ok", "model": "...", "attempts": 1, ...}
```

---

## 代码里怎么用

```python
from llmkit import ask, stream_ask, collect, AskOptions

result = ask("贝叶斯定理是什么")
print(result.text)
print(result.usage.total_tokens, result.latency_ms, result.attempts)
```

结构化输出：

```python
from pydantic import BaseModel
from llmkit import ask_structured

class Answer(BaseModel):
    answer: str
    confidence: float

obj = ask_structured("给出答案和置信度，用 JSON 返回", Answer)
```

流式：

```python
from llmkit import stream_ask, collect, collect_partial

text = collect(stream_ask("讲一段话"))              # 拼成整段
text, error = collect_partial(stream_ask("讲一段话"))  # 断了也保留半截
```

---

## 环境变量

| 变量 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `LLMKIT_API_KEY` | ✅ | — | 缺失时启动即报 `ConfigError` |
| `LLMKIT_BASE_URL` | | `https://api.openai.com/v1` | 任何兼容 OpenAI 协议的服务 |
| `LLMKIT_MODEL` | | `gpt-4o-mini` | 默认模型 |
| `LLMKIT_TIMEOUT_S` | | `60` | 单次请求超时 |
| `LLMKIT_PRICE_IN` / `_OUT` | | `0` | 元/百万 token，用来估算成本；不填就记 0 |

优先级：**进程环境变量 > `.env` 文件 > 代码默认值**。

---

## 异常

业务代码只需要 `except LLMError`。

| 异常 | 什么时候 | 该不该重试 |
|---|---|---|
| `RetryableError` | 超时、连接断、429、5xx | 已经自动重试过了 |
| `FatalError` | 401、400、畸形响应 | 重试无意义 |
| `ConfigError` | 密钥没配、环境变量非法 | 改配置 |
| `LLMParseError` | 模型输出不是合法 JSON / 不合 schema | 带 `raw` 字段看原文 |

`ConfigError` 和 `LLMParseError` 都继承 `FatalError`，所以 `except FatalError` 也能接住。

---

## 测试

```bash
pytest llmkit -v
```

70 条，全部离线，不需要密钥、不联网：

| 文件 | 条数 | 覆盖 |
|---|---|---|
| `test_types.py` | 8 | 契约的必填、越界、序列化往返 |
| `test_config.py` | 12 | 缺 key、优先级、脱敏、成本计算 |
| `test_parsing.py` | 12 | 三种脏输出的抽取、回灌重试 |
| `test_client.py` | 15 | 成功 / 401 不重试 / 超时重试 / 参数透传 |
| `test_stream.py` | 11 | 跳过空块、断流保留半截、close 一定执行 |
| `test_obs.py` | 7 | JSON 日志、幂等 handler、异常入日志 |
| `test_cli.py` | 7 | 退出码 0/1/2 |

---

## M1 验收

| # | 标准 | 状态 |
|---|---|---|
| 1 | `pytest llmkit` 全绿，≥20 条 | ✅ 70 条（全项目 283 条，含 corpus/retrieval/qa） |
| 2 | 真实调用跑通一次 | ✅ 2026-11-17 用 DeepSeek 实测（见下） |
| 3 | 能讲清"一次 ask 经过哪 5 层" | 见下方 |

真实调用实测结果（2026-11-17，`deepseek-flash`，非离线）：

| 场景 | 命令 | 结果 |
|---|---|---|
| 配置检查 | `python -m llmkit --check` | 退出码 0，密钥显示 `***a571` |
| 普通问答 | `python -m llmkit "..."` | 54 in / 145 out，1658ms，1 次尝试 |
| 流式 | `python -m llmkit --stream "..."` | 逐块拼出 `1,2,3,4,5`，退出码 0 |
| 结构化 | `ask_structured(prompt, Event)` | 抽出 `name=张三 date=2026年3月15日`，Pydantic 校验通过 |
| 密钥错误 | 故意填错 key | `FatalError`，203ms，**1 次尝试**（不重试，符合预期） |
| 地址不可达 | 指向 `127.0.0.1:9` | `RetryableError`，退避 0.56s → 1.12s，**3 次尝试**后放弃 |

一次 `ask()` 的完整路径：

1. `config.py` 读环境变量 → `Settings`（缺 key 立刻 `ConfigError`）
2. `client.py` 拼 `messages`（system + user）
3. `errors.py` 的 `retry_call` 包住 `create`，异常经 `classify` 分成可重试/不可重试
4. `obs.py` 打一行 JSON（trace_id、attempts、tokens、耗时）
5. `client.py` 把响应拆成 `AskResult`（含 `Usage` 与 `latency_ms`）

---

## 已知限制

- **真实调用已验证到 DeepSeek**：2026-11-17 用 `deepseek-flash` 跑通了普通/流式/结构化三条路径。
  但 `.env.example` 里列的通义、智谱、月之暗面、OpenAI 四组地址**仍未用真实密钥验证**，
  换服务商后先跑 `--check` 再发请求。
- **不做并发**：`ask()` 是同步的。批量调用要等 W3 的异步层，现在写并发会得到"假并发"。
- **流式不保证重试**：建流失败会重试 3 次；一旦开始遍历，断了就断了——已吐出去的字收不回来。
- **成本是估算**：只按配置的单价乘 token 数，不含缓存折扣、阶梯价、图片计费。
  DeepSeek 的高峰/低谷是两套价（低谷半价），本机 `.env` 填的是高峰价，所以估算偏高。
