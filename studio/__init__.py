"""上传服务层（M5）：把"用户自己的资料"接进四层问答管线。

依赖方向：`studio` → `qa` / `retrieval` → `corpus`，`llmkit` 只被需要模型的地方调用。
删掉这个包，剩下四层与它们的 283 条测试完全不受影响 —— 这是刻意的：
上传是"应用"，不该反向污染"库"。

模块分工：
    library.py   上传登记表（可写，sha256 去重）
    ingest.py    安检 + 解析 + 切块
    glossary.py  术语抽取（统计层，候选天生可回溯）
    profile.py   领域画像（模型只挑候选，三道机械校验）
    service.py   编排与缓存
    api.py       FastAPI 路由 + 单页界面
"""
