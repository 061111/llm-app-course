"""M3 · 检索层。

模块分工：
  bm25.py           词法检索打分（纯 Python，零依赖）
  query.py          中文问题 → 英文关键词（带缓存）
  search.py         把块、索引、翻译串成 Searcher
  eval_retrieval.py 用黄金集量 hit@k / MRR
"""
