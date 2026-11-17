# 讲义知识库问答（RAG v1）

把一批 PDF 讲义变成能问答的知识库，**答案必须能翻回原页**。

输入 5 份数据挖掘讲义（199 页），输出带页码引用的答案；配套 18 道人工核对的标准题，任何改动都能用同一把尺子量出好坏。

```
Q: chap03 第 70 页那个熵 0.65 是怎么算出来的？
A: 按 5/6 算：– (1/6)log2(1/6) – (5/6)log2(5/6) = 0.65
   【chap03:p70】
   注：该页公式第二项印成了 log2(1/6)，照字面算得 0.86，与结果不符。
```

> 最后那句「该页印错了」不是模型猜的——它读到了原文，并指出了原文的矛盾。这正是这个项目想验证的能力：**有据可查，而不是听起来对**。

---

## 这个项目解决什么问题

把讲义直接丢给通用模型是不行的，它没读过这份文件。三个具体障碍，每个对应一层代码：

| 障碍 | 不解决会怎样 | 对应模块 |
|---|---|---|
| 模型会编 | 问它讲义里的公式，它按训练记忆编一个像样的答案 | 检索 + 强制引用 + 越界检测 |
| 199 页塞不进上下文 | 正文 10.2 万字，全塞进去既超限又烧钱 | 切 181 块，每次只取 top-5 |
| 改完不知道有没有变好 | 调参数靠「感觉好像好点」 | 18 道黄金题 + 三张指标表 |

第三条最容易被跳过，但它决定了这个项目能不能说「hit@5 从 72% 提到 94%」，而不是「我觉得还行」。

## 核心功能

四层单向依赖，上层调下层，下层不知道上层存在：

| 层 | 模块 | 能力 |
|---|---|---|
| 调模型 | `llmkit` | 同步问答、结构化输出（Pydantic 校验 + 解析失败自动重问）、流式输出；401 立刻失败不重试，网络抖动指数退避重试 |
| 读文档 | `corpus` | PDF 按页抽文本（按坐标去页码、符号还原成 `∧`/`⇒`/`×`）→ 按页切块（带页码偏移）→ jsonl 落盘 |
| 找内容 | `retrieval` | 中文问题翻成英文关键词，纯 Python BM25 检索，零依赖 |
| 答并自查 | `qa` | 拼带出处头的 prompt，强制标 `【chap03:p70】`，事后核对引用是否越界、资料不足时是否拒答 |

## 实测效果

| 环节 | 指标 | 数值 |
|---|---|---|
| 切分 | 块数 / 中位长度 / 串味 | 181 块 / 507 字 / 0 |
| 检索 | hit@5 / MRR | **94%** / 0.769 |
| 问答 | 判分正确率（top_k=5 / 8） | 17/18 / **18/18** |
| 问答 | 引用越界 | 0 |
| 拒答 | 语料外问题（MySQL / Transformer / 考试日期） | 3/3 拒绝作答，0 编造 |
| 测试 | 全套 | 283 passed |

**这些数字的可信度边界**：18 道题的答案就出自这 199 页，模型看的是原文——这是开卷且答案在手。它证明链路没断、引用没编、该拒答时会拒答，**不证明泛化能力**。换成没见过的文档会掉多少，目前不知道。

评估方法、逐题明细、以及每个数字的复现命令见 [`docs/architecture.md`](docs/architecture.md)。

---

## 快速开始

### 1. 环境要求

| 项 | 要求 |
|---|---|
| Python | 3.12 及以上 |
| 操作系统 | Windows / Linux / macOS 均可（开发在 Windows 11 上完成） |
| 包管理 | 推荐 [uv](https://docs.astral.sh/uv/)，没有也能用 pip |
| 网络 | 需要能访问所选模型服务商的 API |
| 语料 | 5 份 PDF 讲义（**不在仓库里**，见下文） |

### 2. 安装

```bash
git clone <仓库地址>
cd llm-app-course

# 有 uv（推荐）
uv sync --extra dev

# 只有 pip
python -m venv .venv
.venv/Scripts/activate        # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -e ".[dev]"
```

也可以让脚本代劳，它会检查 Python 版本、建虚拟环境、装依赖、准备 `.env`：

```bash
python scripts/bootstrap.py
```

### 3. 配置

```bash
cp .env.example .env          # Windows cmd: copy .env.example .env
```

打开 `.env`，只填这一行就能跑：

```ini
LLMKIT_API_KEY=你的密钥
```

其余变量都有默认值。默认指向 OpenAI；换服务商时同时改 `LLMKIT_BASE_URL` 和 `LLMKIT_MODEL`，完整变量表见下一节。

校验配置是否被正确读到（不会发网络请求）：

```bash
python -m llmkit --check
# 配置可用：base_url=https://api.deepseek.com/v1 model=deepseek-flash key=***a571
```

### 4. 放置语料

项目默认在 `~/Desktop/DataMining pdf/` 下找这 5 个文件：

```
chap03_Classification_Basic ConceptsandTechniques (1).pdf   82 页
chap04a_...pdf                                              37 页
chap04b_...pdf                                              14 页
chap04c_...pdf                                              28 页
chap05_...pdf                                               38 页
```

放在别处就设环境变量：

```bash
export LLMKIT_CORPUS_DIR=/path/to/pdfs       # Linux/macOS
set LLMKIT_CORPUS_DIR=D:\path\to\pdfs        # Windows cmd
```

**没有这批 PDF 也能跑测试**：283 条里有 268 条是纯函数测试，不碰文件；其余 15 条会自动跳过。但建索引和端到端问答需要真实语料。

### 5. 建索引并提问

```bash
python scripts/build_index.py          # 解析 + 切分 + 落盘，约 2 秒
python -m llmkit "贝叶斯分类器的核心假设是什么"    # 只调模型，不用语料
python -m qa.eval_qa 8                 # 端到端跑 18 道黄金题（需语料 + 密钥）
```

索引只建一次，之后所有查询都读 `artifacts/chunks.jsonl`。改切分参数才需要重跑第一步。

---

## 配置说明

所有配置走环境变量，优先级：**进程环境变量 > 同目录 `.env` 文件 > 代码默认值**。

| 变量 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `LLMKIT_API_KEY` | 是 | 无 | 服务商密钥。缺失时调用前抛 `ConfigError`，不会带着空 key 发请求 |
| `LLMKIT_BASE_URL` | 否 | `https://api.openai.com/v1` | OpenAI 兼容的接口地址 |
| `LLMKIT_MODEL` | 否 | `gpt-4o-mini` | 模型名 |
| `LLMKIT_TIMEOUT_S` | 否 | `60` | 单次请求超时（秒） |
| `LLMKIT_PRICE_IN` | 否 | `0` | 输入单价，单位「元/百万 token」，用于成本估算；填 0 表示不算 |
| `LLMKIT_PRICE_OUT` | 否 | `0` | 输出单价，同上 |
| `LLMKIT_CORPUS_DIR` | 否 | `~/Desktop/DataMining pdf` | 语料目录 |

**已实测的服务商**：DeepSeek（`https://api.deepseek.com/v1`，模型 `deepseek-flash`）。`.env.example` 里另外四组地址与模型名来自公开文档，**没有用真实密钥验证过**。

## 命令速查

| 命令 | 作用 |
|---|---|
| `python -m llmkit "问题"` | 一次问答，打印 token / 耗时 / 成本 |
| `python -m llmkit "问题" --stream` | 流式输出 |
| `python -m llmkit "问题" --json` | 强制结构化输出 |
| `python -m llmkit --check` | 校验配置（不发请求） |
| `python scripts/build_index.py` | 建索引；`--dry-run` 只看统计 |
| `python -m corpus.store stats` | 查看已落盘索引的统计 |
| `python -m corpus.eval_chunking` | 切分方案对比矩阵（3 种分组 × 4 个上限） |
| `python -m retrieval.eval_retrieval` | 检索指标 hit@k / MRR |
| `python -m qa.eval_qa [top_k]` | 端到端 18 题评估与判分 |
| `python scripts/verify.py` | 提交前总检查：lint + 测试 + 索引 + 配置 |
| `python scripts/bootstrap.py --check` | 只体检环境，不安装 |

安装为包之后（`pip install -e .`）还可以直接用 `llmkit`、`corpus-store`、`eval-retrieval`、`eval-qa` 四个短命令。

## 测试

```bash
pytest                     # 全套，约 11 秒
pytest corpus -v           # 单层
pytest -k "chunk" -v       # 按名字筛
python scripts/verify.py   # lint + 测试 + 索引 + 配置，四项一起过
```

| 层 | 用例数 | 覆盖什么 |
|---|---|---|
| `w02` | 78 | 教材层：类型校验、异常重试、装饰器、生成器与上下文 |
| `llmkit` | 70 | 配置读取、错误分类、重试退避、JSON 修复、流式、CLI |
| `corpus` | 90 | 页码过滤、符号映射、切分边界、序列化往返 |
| `retrieval` | 25 | 分词归一化、BM25 打分、翻译缓存 |
| `qa` | 20 | prompt 拼装、引用解析、越界检测、双轨判分 |
| **合计** | **283** | 全部离线可跑，不依赖真实 API |

需要真实 API 的只有 `qa/eval_qa.py` 和 `retrieval/eval_retrieval.py` 的翻译步骤；后者带文件缓存，重复跑不重复花钱。

## 目录结构

```
llm-app-course/
├── README.md                 本文件：项目入口
├── pyproject.toml            依赖、打包、pytest 与 ruff 配置
├── uv.lock                   锁定版本，保证换机器装出同一套
├── .env.example              配置模板（.env 已被忽略，不会提交）
├── scripts/
│   ├── bootstrap.py          环境准备：虚拟环境 + 依赖 + .env + 语料检查
│   ├── build_index.py        解析 → 切分 → 落盘
│   └── verify.py             提交前总检查
├── llmkit/                   模型调用层（唯一 import openai 的地方）
│   ├── client.py             ask / ask_structured / stream_ask
│   ├── config.py             .env 读取，Settings 不可变
│   ├── errors.py             classify() 判该不该重试 + retry_call
│   ├── parsing.py            从脏输出里抠 JSON
│   ├── obs.py                一行一 JSON 的日志
│   └── README.md             模块说明书
├── corpus/                   语料层
│   ├── docs.py               5 个 PDF 的登记表
│   ├── gold.py               18 道人工核对的标准题
│   ├── parser.py             PDF → 每页纯文本
│   ├── chunker.py            切块（带页码偏移）
│   ├── store.py              jsonl 落盘与加载
│   ├── eval_chunking.py      切分方案对比
│   └── check_gold_pages.py   核对黄金集引用的页是否真有正文
├── retrieval/                检索层
│   ├── bm25.py               NFKD 归一化 + 分词 + BM25
│   ├── query.py              中文问题 → 英文关键词（带缓存）
│   ├── search.py             索引 + 翻译 + 检索
│   └── eval_retrieval.py     hit@k / MRR
├── qa/                       生成与评估层
│   ├── prompt.py             片段拼装，强制出处格式
│   ├── answer.py             一次完整问答 + 引用越界检测
│   ├── grading.py            数字机械核对 + 模型判卷
│   └── eval_qa.py            端到端 18 题
├── docs/
│   ├── architecture.md       架构、数据模型、八个关键决策、逐项指标
│   └── course-plan.md        24 周学习排期（与代码无关，属学习记录）
├── w01/                      环境体检脚本（W1 产物）
├── w02/                      W2 教材层：6 个模块 + 4 份讲义 + 78 条测试
└── artifacts/                产物目录（已忽略，可由 build_index.py 重建）
```

`w01/`、`w02/` 是学习过程留下的教材，**产品代码对它们零依赖**（`grep` 确认过，只有一句注释提到）。它们留在仓库里是为了保留演进记录，`pyproject.toml` 的 `testpaths` 把 `w02` 一起跑只是顺带。

## 设计要点

四条最关键的决定，展开在 [`docs/architecture.md`](docs/architecture.md)：

1. **重试只包住建请求，不包住流式遍历。** 流到一半断了重试会从头发，屏幕上出现重复的半句话。
2. **页码按坐标删，不按行序删。** PDF 的抽取顺序不等于视觉位置：同一份讲义里，页码有时是最后一行，有时是第二行，但坐标完全一致。
3. **按页切，不按章节合并。** 通用建议「512 token 按章节切」来自散文文档；这批是 PPT，平均 513 字一页，每页本来就是作者切好的概念。按页切单块率 78%、串味 0，按标题合并是 61%、串味 3。
4. **引用越界按「页的并集」判定。** 要求单个块覆盖整个引用区间会误报：模型写 `【p20-21】`，而 p20、p21 是两个独立块且都在结果里。

## 已知限制

1. **跨语言靠翻译**，每次提问多一次模型调用（约 ¥0.002）。接上 embedding 才能知道这条路的天花板。
2. **表格会竖排**。`chap04a` 第 3 页的动物表被抽成 132 行、一格一行，列关系丢失。现有题目不依赖它，但要做「按规则表判断 frog 属于哪类」必须先还原表格。
3. **词法检索分不出「定义页」和「算例页」**。同一批术语在两类页里都密集出现，BM25 只能靠数字（`(0,6)`、`(1,5)`）这类信号区分。其中一道题至今需要 `top_k=8` 才能召回。
4. **单机、同步、无并发**。`ask()` 是同步函数，批量调用是串行的。
5. **没有 Web 服务**。只提供 CLI 与 Python 库两种用法，见下节。

## 部署与使用

当前形态是**本地 CLI + 可导入的 Python 包**，不是常驻服务。

**作为库调用**（比如接到自己的脚本或 FastAPI 里）：

```python
from retrieval.search import Searcher
from retrieval.query import Translator
from qa.answer import answer_question

searcher = Searcher.from_store(translator=Translator())
ans = answer_question("增益率解决了什么问题", searcher=searcher, top_k=5)

print(ans.text)            # 答案正文，含【chap03:p73】这样的引用
print(ans.cited_sources)   # ('chap03:p73',)
print(ans.unsupported)     # () 表示没有编造的引用
```

**作为本地服务部署**：`pyproject.toml` 预留了 `web` extra（FastAPI + Uvicorn），默认不装：

```bash
uv sync --extra web
```

服务层代码尚未实现。若自行封装，注意两点：`Searcher` 初始化较慢（要读 181 块建索引），应在进程启动时建一次并复用；`ask()` 是同步的，放进异步框架要用 `run_in_threadpool` 之类的方式包一层，否则会阻塞事件循环。

## 贡献规范

这个仓库最初是个人学习项目，欢迎提 issue 讨论，提交代码前请确认：

**提交前必须通过**

```bash
python scripts/verify.py     # lint + 283 条测试 + 索引对账 + 配置检查
```

四项有一项红就不要提交。特别地，`pytest` 必须全绿，新增逻辑要配测试。

**代码约定**

- Python 3.12，类型注解用 `from __future__ import annotations`
- 行宽 100，静态检查用 ruff（`select = ["E4","E7","E9","F"]`，只管会出错的问题，不强制格式化）
- 数据模型一律用 Pydantic，不用裸 dict 跨层传递
- 新增依赖前先确认现有库做不到；`pyproject.toml` 的 `dependencies` 只放真正被 import 的包
- 中文注释可以，但**注释要解释「为什么」，不解释「是什么」**——代码本身说明做了什么

**提交信息**

```
<type>: <一句话说明>

feat      新功能
fix       修 bug
refactor  重构，不改行为
docs      只改文档
test      只改测试
chore     配置、依赖、脚本
```

例：`fix: 页码改为按 bbox 过滤，修复 p60 页码夹在标题后的问题`

**分支**

`main` 保持可用。改动走 `feat/xxx` 或 `fix/xxx` 分支，合并前确保 `verify.py` 通过。

## 文档索引

| 文件 | 内容 |
|---|---|
| `README.md` | 项目入口：能做什么、怎么跑、怎么测 |
| [`docs/architecture.md`](docs/architecture.md) | 架构、数据模型、两条流程、八个关键决策、逐项指标与复现命令 |
| [`docs/course-plan.md`](docs/course-plan.md) | 24 周学习排期与取舍（与代码无关） |
| [`llmkit/README.md`](llmkit/README.md) | 模型调用层的模块说明与异常表 |
| [`w02/README.md`](w02/README.md) | W2 教材层的周级索引 |

## 许可证

本仓库未附加开源许可证。代码可自由阅读与学习；如需在其它项目中使用，请先开 issue 联系作者。

讲义 PDF 的版权归课程方所有，不在本仓库内分发。
