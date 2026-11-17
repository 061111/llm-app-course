"""手动核对工具：黄金集每条引用的页，是否真的有正文。

为什么不是 pytest 用例：这个检查要打开桌面上的 PDF。
换一台机器、或者你把语料挪走，测试就全红了 —— 而代码本来没错。
所以它是一个"有空就跑一次"的体检工具，不是回归测试。

用法（在项目根目录）：
    python -m corpus.check_gold_pages

判读：
    OK          该页字符数 >= 100，有正文可查
    !! 空页/图片页 字符数不足，引到这里等于没有出处
"""
from __future__ import annotations

import pymupdf

from corpus.docs import DOCS
from corpus.gold import GOLD

MIN_CHARS = 100


def main() -> int:
    bad = 0
    for item in GOLD:
        meta = DOCS[item.doc_id]
        path = meta.path()
        if not path.is_file():
            print(f"!! 找不到语料文件：{path}")
            return 2

        doc = pymupdf.open(path)
        text = doc[item.page - 1].get_text("text").strip()
        doc.close()

        ok = len(text) >= MIN_CHARS
        first = text.splitlines()[0][:56] if text else "(整页无文本层)"
        flag = "OK " if ok else "!! "
        print(f"{flag}{item.id} {item.doc_id}:p{item.page:<3} 字符={len(text):<4} {first}")
        if not ok:
            bad += 1

    print()
    if bad:
        print(f"有 {bad} 条引用的页正文过少，需要换页或换题。")
        return 1
    print(f"全部 {len(GOLD)} 条引用的页都有正文。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
