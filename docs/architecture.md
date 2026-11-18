# 项目 A · 讲义知识库问答（RAG v1）

> 这份文档只讲项目 A 本身：它解决什么问题、怎么搭起来、指标多少、哪里不行。
> 快速开始与命令索引在根目录 [`README.md`](../README.md)，课程排期在 [`course-plan.md`](course-plan.md)，三份不重叠。
>
> 本文数字均为 **2026-11-17 本机实测**，每条都注明了复现命令。
> 「未重跑」字样表示该数字来自当日记录、本次未再执行。

---

## 1. 一句话

把五份数据挖掘讲义（199 页 PDF）变成一个**能回答课程问题、每个结论都能翻回原页**的问答系统。

不做通用问答，不做多轮对话，只做「这门课的助教」这一件事。

---

## 2. 解决什么问题

学这门课时有三个具体麻烦，项目 A 对应三件事：

| 麻烦 | 系统的回答方式 |
|---|---|
| 想查「拉普拉斯平滑里 n、nc、v、p、m 各是什么」，只能翻 199 页 PPT | 问一句，答案直接标【chap04c:p19】 |
| 大模型会凭训练记忆编一个「看起来对」的答案 | 强制只用讲义原文；片段里没有就回答「讲义未覆盖」 |
| 没法判断「答得准不准」 | 18 条人工核对的标准问答（黄金集）+ 可复现的指标 |

**适用**：单一领域、有明确原文出处的文档问答；需要溯源到页的场景（助教、内部手册、规范查询）。

**不适用**（代码里没有，不是「以后再说」而是当前边界）：
多轮对话、用户体系与权限、并发服务、增量更新、跨文档推理、图表/表格内容问答。

---

## 3. 现在能做什么

| 能力 | 做法 | 现状 |
|---|---|---|
| 回答讲义内的问题 | 检索 5 个片段 → 拼 prompt → 模型作答 | 18/18 判对（top_k=8） |
| 标注出处 | 强制【doc_id:pN】格式，事后核对是否越界 | 越界 0 条 |
| 承认不知道 | 片段没有就答「讲义未覆盖」 | G15 在 top_k=5 时正确拒答 |
| 拒绝超纲问题 | 语料外问题不做答 | 3/3 全部拒绝，0 编造引用 |
| 流式输出 / 结构化输出 | `llmkit` 提供，供上层复用 | 已实测（见 llmkit/README.md） |

负向对照的实测结果（问题问的是 MySQL，语料里没有）：

```
Q: MySQL 的 InnoDB 默认事务隔离级别是什么？
   claims_uncovered=True   citations=()   unsupported=()
```

三个语料外问题（MySQL / Transformer 复杂度 / 期末考试日期）判定一致：全部承认未覆盖、不给出引用、也不编造页码。

---

## 4. 架构

```
                    用户问题（中文）
                          │
┌─────────────────────────▼─────────────────────────┐
│ qa/        生成与评估                              │
│   prompt.py  片段 → 带出处头的 prompt              │
│   answer.py  检索→拼装→调用→解析引用→越界检测       │
│   grading.py 数字机械核对                          │
│   eval_qa.py 18 题批量跑 + 模型判卷                │
└─────────────────────────┬─────────────────────────┘
                          │ search(question, top_k)
┌─────────────────────────▼─────────────────────────┐
│ retrieval/ 检索                                    │
│   query.py   中文问题 → 英文关键词（带缓存）        │
│   bm25.py    归一化 + 分词 + BM25 打分             │
│   search.py  把块/索引/翻译串成 Searcher           │
└─────────────────────────┬─────────────────────────┘
                          │ 读 chunks.jsonl
┌─────────────────────────▼─────────────────────────┐
│ corpus/    语料                                    │
│   docs.py    5 个 PDF 的登记表                     │
│   gold.py    黄金集 18 题                          │
│   parser.py  PDF → 每页文本（去页脚、修符号）       │
│   chunker.py 每页 → 带坐标的块（181 块）            │
│   store.py   jsonl 落盘 / 加载 / 对账              │
└─────────────────────────┬─────────────────────────┘
                          │ ask(prompt)
┌─────────────────────────▼─────────────────────────┐
│ llmkit/    模型调用                                │
│   client.py  唯一 import openai 的文件             │
│   errors.py  异常分类 + 指数退避重试                │
│   config.py  .env → Settings                       │
│   parsing.py 结构化输出 + 解析失败重问              │
│   obs.py     一行一 JSON 日志 + trace_id           │
└─────────────────────────┬─────────────────────────┘
                          ▼
                 DeepSeek API（openai 协议）
```

**依赖是单向的**：`qa → retrieval → corpus → llmkit`。下层不知道上层存在，所以：

- 换模型供应商只改 `.env`；
- 换切分策略只重跑 `corpus.store`，检索与问答代码一行不动；
- 换检索算法（BM25 → 向量）只动 `retrieval/`，`qa/` 不用改。

这不是理论上的好处，而是这条链路的设计目标：M2–M4 的代码都只调用 `llmkit` 已有的三个函数，没有要求它新增接口（`ask` / `ask_structured` / `stream_ask` 的签名至今未变）。
补充一句实话：M1 收尾时确实改过一次 `llmkit/tests/test_stream.py`，但改的是「测试依赖本机没配密钥」这个巧合，不是接口。

---

## 5. 目录与模块

| 路径 | 职责 | 关键文件（行数） | 说明文档 |
|---|---|---|---|
| `llmkit/` | 模型调用层 | `client.py` 284 / `errors.py` 145 / `config.py` 123 | `llmkit/README.md` |
| `corpus/` | 语料层 | `chunker.py` 429 / `parser.py` 282 / `store.py` 225 / `gold.py` 397 | 各文件头 docstring |
| `retrieval/` | 检索层 | `bm25.py` 149 / `query.py` 127 / `search.py` 97 | `retrieval/__init__.py` |
| `qa/` | 生成与评估 | `eval_qa.py` 214 / `answer.py` 180 / `prompt.py` 137 | 本文 §7–§8 |
| `studio/` | 上传服务与界面 | `service.py` 446 / `api.py` 250 / `selfcheck.py` 398 / `balance.py` 306 / `static/index.html` 1202 | `docs/upload-service.md`、`docs/selfcheck.md`、`docs/model-switch.md`、`docs/coupon-balance.md` |
| `artifacts/` | 产物（gitignore） | `chunks.jsonl` 148KB / `qa_eval.json` / `query_cache.json` | — |
| `pyproject.toml` | 依赖与 pytest 配置 | 运行时依赖 8 个，实际用到 3 个（见 §9.1） | — |

生产代码 3810 行（四个包的 `.py`，不含测试），测试 2274 行。

---

## 6. 数据模型

四个模型串起整条链路，字段就是层与层之间的契约。

**`Chunk`（检索单元，frozen）** —— `corpus/chunker.py:111`

| 字段 | 类型 | 含义 |
|---|---|---|
| `chunk_id` | `doc#序号` | 稳定可排序，如 `chap03#051` |
| `doc_id` | str | 属于哪份讲义 |
| `title` | str | 来自 PDF 书签；空页时可能为空 |
| `page_start` / `page_end` | int | 覆盖的页范围（含） |
| `text` | str | 正文，页间用空行分隔 |
| `spans` | `tuple[PageSpan,…]` | **块内字符偏移 → 页码**，引用溯源的依据 |
| `is_empty` | bool | 正文极短，检索层应跳过 |

`spans` 是最容易被当成冗余的字段，但它解决一个具体问题：块内第 137 个字符来自哪一页？

```python
chunk.page_of(137)   # → 70
chunk.header()       # → 【chap03:p70 · Computing Entropy of a Single Node】
```

没有它，跨页块（4 个之一）的引用只能靠猜。

**`SearchResult`** —— `retrieval/search.py:26`：`chunk` + `score` + `rank` + `terms`（本次用的词元，便于排查「为什么召回这个」）。

**`Answer`** —— `qa/answer.py:48`：`text` / `citations` / `retrieved`（本次喂进去的块）/ `unsupported`（越界引用）/ `usage` / `latency_ms` / `trace_id`。

**`GoldItem`** —— `corpus/gold.py:40`：`id` / `question` / `answer` / `doc_id` / `page` / `kind`(fact|calc|trap) / `verified_by_user` / `note`。

---

## 7. 两条流程

### 7.1 离线建库（跑一次，约 2.1 秒）

```
python -m corpus.store save
  ├─ build_all()          按 DOCS 顺序解析 5 个 PDF → 199 个 PageText
  │   ├─ extract_pages()  逐页 read_blocks → 按 bbox 删页脚 → 字符清洗
  │   ├─ read_titles()    读 PDF 书签（199/199 页齐全）
  │   ├─ group_pages()    按页分组（mode=page）
  │   ├─ split_long()     超 1200 字断在换行处，相邻块重叠 150 字
  │   └─ merge_tiny()     不足 200 字的页单向前吞进下一页
  └─ save_chunks()        写 artifacts/chunks.jsonl + chunks.meta.json
```

实测产物（`python -m corpus.store stats`）：

```
块数      181
总字数    102539
字数分布  最小 202 / 中位 507 / 最大 1175
空块      0
各文档    chap03=76  chap04a=31  chap04b=12  chap04c=28  chap05=34
```

`chunks.meta.json` 记住 `mode/max_chars/min_chars/overlap/块数/生成时间`，`load_chunks` 拿块数对账——文件被截断会直接报错，不会把坏数据喂给检索层。

### 7.2 在线问答（每次提问）

`qa/answer.py:123` 的 `answer_question()` 是唯一入口：

| 步 | 函数 | 干什么 |
|---|---|---|
| 1 | `Searcher.from_store()` | 读 `chunks.jsonl`，建内存索引（181 块约 10ms） |
| 2 | `Translator.__call__` | 中文问题 → 英文关键词；命中 `query_cache.json` 就不调模型 |
| 3 | `BM25.search` | 词元去重 → NFKD 归一化 → 打分取 top_k |
| 4 | `build_context` | 每块加 `【doc:pN · 标题】` 出处头，总长硬上限 6000 字 |
| 5 | `llmkit.ask` | `Settings` → `retry_call` 包住 `create` → DeepSeek |
| 6 | `parse_citations` | 正则抽所有【doc:pN】 |
| 7 | `unsupported_citations` | 引用页是否落在「本次检索到的页的并集」里 |

第 5 步的细节在 `llmkit/README.md`：**重试包住建请求，不包住流式遍历**。

---

## 8. 八个关键决策

每条都写「结论 / 为什么 / 证据」，因为这几个地方一开始我都做错了。

**① 按页切，不按章节合并**
PPT 平均 513 字/页，一页本来就是作者切好的一个概念。合并只会把两个概念焊在一起：按标题合并把 chap04c「Issues with Naïve Bayes」5 页并成 2695 字，G14/G15 落进同样 3 块（串味 3）。按页切串味 0、单块率反而更高（78% vs 56%）。

**② 页码按坐标删，不按行序删**
第一版「删最后一行数字」在 4 页上失败：p31 页码在末尾、p60 页码在第二行，但两者坐标完全相同（`y=517.2–530.4`，页高 540）。文本抽取顺序 ≠ 视觉位置，只能按 `y0 >= 页高×0.95 且纯数字` 判。

**③ 私有区字符要映射，不能删**
正文藏着 806 个 PUA 字符。删掉它们，规则的 `∧`、`⇒` 和乘号 `×` 一起消失。实测确定身份：`U+F0D9=∧`、`U+F0AE=⇒`、`U+F0B4=×`（乘号靠 `4/7 × 1/7 × 0.0072 = 0.000588 ≈ 0.0006` 验算确认）。

**④ 检索前必须 NFKD 归一化**
公式是数学斜体（`𝐺𝑖𝑛𝑖𝐼𝑛𝑑𝑒𝑥`、`𝑨𝒄𝒄𝒖𝒓𝒂𝒄𝒚=`），不归一化切不出 `gini`，整页在算 Gini 却搜不到。另外讲义写 `Naïve`、查询写 `naive` 是两个不同词元，也要靠 NFKD 去组合符。

**⑤ 保留单个数字**
G04 问三个节点的 Gini，问题里唯一能区分「算例页」和「定义页」的信号是 `(0,6)/(1,5)/(2,4)`。丢掉单字符数字后 hit@5 从 94% 掉到 89%。

**⑥ 中文问题先翻成英文关键词**
语料英文、问题中文，词法检索对不上。翻译后 hit@5 72% → 94%。「不翻译就是 0」是错的——黄金集近一半题夹着英文术语，字面能蒙中。

**⑦ 引用越界按「页的并集」判**
第一版要求「单个块覆盖整个引用区间」，把 `【chap03:p20-21】` 误判为越界（p20、p21 是两个独立块，但都在结果里）。改成按页集合判定后误报消失。

**⑧ 判卷失败不能中断评估**
首轮第 6 题判卷返回空内容，`LLMParseError` 直接炸掉 18 题评估，前 5 题结果一起丢。改成重试一次、仍失败记 `judge_failed` 继续跑。

---

## 9. 运行手册

### 9.1 环境

```bash
python --version        # 3.12.x
uv sync                 # 或 pip install -e .
python w01/check_env.py # 可选：环境体检
```

运行时真正用到的三方库只有三个：`pydantic`（全部数据模型）、`pymupdf`（PDF 解析）、`openai`（仅 `llmkit/client.py`）。
`fastapi` / `uvicorn` / `tenacity` / `pydantic-settings` / `httpx` 虽在 `pyproject.toml` 里声明，但项目代码一次都没 import——是给后续周次预留的。

### 9.2 配置（`.env`，已被 gitignore）

| 变量 | 必填 | 说明 |
|---|---|---|
| `LLMKIT_API_KEY` | ✅ | 缺失时启动即 `ConfigError` |
| `LLMKIT_BASE_URL` | | 任何兼容 openai 协议的服务，本机为 `https://api.deepseek.com/v1` |
| `LLMKIT_MODEL` | | 本机 `deepseek-flash` |
| `LLMKIT_TIMEOUT_S` | | 单次超时，默认 60 |
| `LLMKIT_PRICE_IN` / `_OUT` | | 元/百万 token，仅用于成本估算 |

优先级：进程环境变量 > `.env` > 代码默认。

### 9.3 六条命令

```bash
python -m llmkit --check                 # 配置自检，不发请求
python -m corpus.store save              # PDF → artifacts/chunks.jsonl（约 2.1s）
python -m corpus.store stats             # 只读文件，不重跑解析
python -m corpus.eval_chunking           # 切分方案对比矩阵
python -m retrieval.eval_retrieval 5     # hit@k / MRR（5 是 top_k）
python -m qa.eval_qa 8                   # 18 题端到端（8 是 top_k，约 1 分钟、¥0.1）
```

外加两个检查工具：

```bash
python -m corpus.check_gold_pages        # 逐条打开 PDF，核对黄金集引用的页是否真有正文
pytest -o addopts="" -q                  # 437 条离线测试
```

---

## 10. 指标

### 10.1 切分（`python -m corpus.eval_chunking`，本次重跑）

| 分组 | 上限 | 块数 | 中位 | 最大 | 单块率 | 串味 | 带题率 |
|---|---|---|---|---|---|---|---|
| page | 600 | 265 | 438 | 599 | 33% | 0 | 69% |
| **page** | **1200** | **181** | **507** | **1175** | **78%** | **0** | **97%** |
| page | 1800 | 178 | 520 | 1402 | 78% | 0 | 99% |
| title | 1200 | 152 | 645 | 1199 | 56% | 3 | 93% |
| title | 3000 | 129 | 645 | 2998 | 78% | 1 | 98% |

三列各自防一种病，必须一起看：**单块率**低 = 切太碎；**串味**高 = 切太大；**带题率**低 = 块里没有自己的标题，模型拿到不知道在讲什么。
（「覆盖」列不列进来了——切分不丢字时它必然 18/18，零区分度。）

选 `page/1200`：串味 0、单块率 78%、带题率 97%，且块长不超过 1175 字。
**未采纳但值得知道**：`page/1800` 的带题率高 2 个点（97%→99%），块数少 3 个。差异太小，没单独做 A/B。

### 10.2 检索（`python -m retrieval.eval_retrieval 5`，本次重跑）

| 模式 | hit@1 | hit@3 | hit@5 | MRR | 未命中 |
|---|---|---|---|---|---|
| 中文原句直接检索 | 56% | 61% | 72% | 0.596 | G07 G12 G13 G17 G18 |
| **翻译成英文关键词** | **67%** | **83%** | **94%** | **0.769** | G15 |

k=8 时 18/18 全中。唯一需要加深的 G15 是「为什么两个类后验都成 0」，答案页 p18 在 k=5 排第 6。

### 10.3 端到端（`python -m qa.eval_qa`，记录于 `artifacts/qa_eval*.json`，未重跑）

| | correct | uncovered | 引用越界 | 平均耗时 | 花费 |
|---|---|---|---|---|---|
| top_k=5 | 17 | 1（G15） | 0 | 2.24s | ¥0.086 |
| top_k=8 | **18** | 0 | 0 | 2.07s | ¥0.104 |

G15 是一次**单变量**因果实验，不是「调了就好了」：

- k=5：p18 没进上下文 → 模型答「讲义未覆盖」，还反过来指出片段与题目矛盾（片段说 `P(Refund=Yes|No)=3/7 ≠ 0`）。
- k=8：拿到 p18 → 给出 `2/6 × 0 × 0.0083 = 0` 的完整算式，与黄金集一致。

---

## 11. 测试

437 条，全部离线（不联网、不需要密钥）。下表是 M5 之前的四层；M5 的 `studio` 133 条见 [`upload-service.md`](upload-service.md)，其中自检 11 条见 [`selfcheck.md`](selfcheck.md)，模型切换 11 条见 [`model-switch.md`](model-switch.md)：

| 包 | 条数 | 覆盖重点 |
|---|---|---|
| `w02` | 78 | 教材层，见 §14 |
| `llmkit` | 84 | 契约、配置优先级、脏 JSON 修复、401 不重试、断流保留半截、空正文重试、模型目录降级 |
| `corpus` | 90 | 解析/切分/落盘的纯函数 + 5 份真实 PDF 的集成用例 |
| `retrieval` | 30 | 分词归一化、BM25 排序、缓存、跨语言对照、中文问句套话剥离 |
| `qa` | 20 | prompt 结构、引用解析、越界误报回归、判卷容错 |

条数用 `pytest <pkg> --collect-only -q` 逐个核过（2026-11-17；M5 新增 studio、M6 新增自检、M7 新增模型切换、M8 新增代金券余额后总数 437，2026-11-18 复核：llmkit 84 / corpus 90 / retrieval 30 / qa 20 / studio 147 / w02 78）。

集成用例用 `skipif(not corpus_available())` 守卫。把语料目录指向空目录实测（`LLMKIT_CORPUS_DIR=/tmp/empty`）：**268 passed, 15 skipped**——15 条依赖真实 PDF 的用例跳过，其余全部照跑，不会一片红。

**测试抓到的三个真问题**（不是覆盖率好看的装饰）：
`@retry` 包生成器不会重试；`SLIDE_PREFIX_RE` 要求冒号导致 `"Slide 38"` 洗不成空串；`stream_ask` 缺配置的测试其实在测「本机恰好没配 key」这个巧合。

---

## 12. 评估怎么做，以及它的可信度边界

**黄金集** 18 条，全部人工核对到页（`verified_by_user=True`），三类：

- `fact` 11 条：概念定义（Gini 的极值、RIPPER 的学类顺序…）
- `calc` 6 条：可复算（G02 的 accuracy 公式、G16 的 support=2/5 与 confidence=2/3、G17 的 2^d−1…）
- `trap` 1 条（G05）：chap03 p70 的熵公式把第二项印成了 `log2(1/6)`，照字面算得 0.86，正确值 0.65 要按 `5/6` 算。答 0.65 = 真读了这页；答 0.86 = 抄了印刷错误；答别的 = 凭记忆编的。

**双轨判分**：数字题做机械核对（`2/3` 与 `0.67` 视为等价写法），其余交给模型判卷（`correct/partial/wrong/uncovered`）。

**必须说清的边界**：黄金集的答案就出在这 199 页里，模型看到的是原文——这是**开卷且答案在手上**。18/18 证明的是「链路没断、引用没编、该拒答时会拒答」，**不证明泛化能力**。换成没见过的文档，或者问题不在语料里，性能会掉，掉多少现在不知道。

---

## 13. 已知限制

1. **表格竖排**：`chap04a` p3 的 22 行动物表被抽成 132 行，一格一行，列关系丢失。要出「按这 5 条规则判断 frog 属于哪类」，必须先做单元格聚类。
2. **个别符号解码错**：chap03 p60 的 Σ 被抽成 `෍`（PDF 用了非标准字体编码）。修它要看 span 的字体名，性价比低，公式类问题已从黄金集避开。
3. **词法检索分不出定义页和算例页**：G04 靠保留单个数字才救回来，G15 至今需要 k=8。
4. **跨语言多一次模型调用**：翻译要调一次模型（有缓存，18 题复跑是 0 次调用）。接 embedding 后才知道这条路的天花板。
5. **不做并发**：`ask()` 是同步的。批量问答要等 W3 的异步层。
6. **成本是估算**：只按配置单价乘 token，不含缓存折扣与阶梯价。

---

## 14. 与课程目录 `w01/`、`w02/` 的关系

**运行时零关系。** 产品代码（`llmkit/` `corpus/` `retrieval/` `qa/`）没有任何一处 `import` w01 或 w02，`grep -rn "w01\|w02" --include="*.py"` 在四个包里只命中 `llmkit/errors.py` 的一句注释。

| | 性质 | 与项目 A 的实际关系 |
|---|---|---|
| `w01/` | W1 环境体检脚本（`check_env.py` + PATH 备份） | **无关**。项目 A 跑起来用不到它 |
| `w02/` | W2 教材：6 个模块 + 4 份讲义，78 条测试 | **血缘关系，不是依赖**。`llmkit/errors.py` 是从 `w02/errors.py` **复制**后改的（新增两个异常、去掉演示函数）。另外 `@retry`、流式切分的思想被用进 `client.py`，但代码是重写的 |

为什么不直接 `from w02.errors import ...`——`llmkit/errors.py` 的文件头自己写了理由：

> `w02/` 是教材目录，会被整目录删改；`llmkit/` 是要被 import 的产品包。产品依赖教材，等于把课堂笔记当运行时依赖。

所以这两个目录是**同一个仓库里放的教材**：`w02` 在 `pyproject.toml` 的 `testpaths` 里，跑「全套 283 条」时会一起跑（78 条就是它的）。如果你按纯工程视角看这个仓库，它们是应该被移出去的。

---

## 15. 下一步（按性价比排序）

| 优先级 | 做什么 | 判据 |
|---|---|---|
| 1 | 接 embedding 做向量检索，与 BM25 对比 | 需要打赢 hit@5 94% / MRR 0.769 这个基线，且说清「为什么用哪种」 |
| 2 | 表格还原（`chap04a` p3 这类） | 能出「按规则表判断 frog 属于哪类」并答对 |
| 3 | rerank（检索 top 20 → 重排取 5） | G15 在 k=5 也能命中，且不拖慢整体 |
| 4 | 服务化（W4 FastAPI） | 有鉴权、限流、流式接口，能被别人调用 |

第 1 项最该先做：面试里最常被追问的就是「你为什么用 BM25 / 为什么用向量」，而你现在手里有一个**可对比的基线**。

---

## 16. 术语

| 术语 | 一句话 |
|---|---|
| RAG | 先检索、再让模型基于检索结果作答，避免它凭记忆编 |
| 块（chunk） | 检索的最小单位，这里是「一页讲义」或它的切分 |
| 黄金集 | 人工核对过答案与出处的题目集，用来量系统好坏 |
| hit@k | 前 k 个结果里有没有包含答案所在页 |
| MRR | 第一个命中结果的排名倒数，越接近 1 越好 |
| 串味 | 一个块同时覆盖两道不同题的页，检索排序会乱 |
| 越界引用 | 模型标的页码不在本次检索结果里 |
