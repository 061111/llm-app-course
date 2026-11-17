"""M2-T2 · 解析层：PDF → 每页纯文本。

这份文件里的每条规则都来自 2026-11-17 对这 5 个 PDF 的实测。

三条实测结论决定了整个设计：

【实测 1】页码的位置不固定，而且和文本抽取顺序无关。
    chap03 p31：页码 "31" 出现在抽取结果的**最后一行**。
    chap03 p60：页码 "60" 出现在抽取结果的**第二行**，夹在标题和正文之间。
    两页的坐标却完全一样：y=517.2~530.4，页高 540，距底边 9.6。
    → 所以"删最后一行"是错的（第一版就错在这，测试立刻抓到）。
      必须按 bbox 删：位于页面底部区域、且内容是纯数字的块。

【实测 2】只有 chap03 有页码页脚（53/82 页），另四份是 0/37、0/14、0/38。
    → 页脚规则必须按文档自适应，不能全局开。
      全局开会误删 chap05 事务表里的末行数字。

【实测 3】每页第一行是**标题**，不是页眉。
    → 绝不能删首行。删了它，T3 切分就没有依据了。

另外 806 个私有区字符（U+E000–U+F8FF）是 Symbol/Wingdings 字体来的，
规则的 ∧ 和 ⇒ 就藏在里面，所以是"映射"不是"删除"。

三条设计原则：
  1. **不静默丢弃**：空页也返回，只标 is_empty。页码必须连续。
  2. **职责分层**：块级过滤（按位置）在 strip_footer_blocks 里，纯函数，
     不碰 PDF，所以单元测试能用假数据覆盖；字符级清洗在 clean_page_text 里。
  3. **留诊断量**：raw_chars 与最终长度之差是 dropped_chars，
     异常增大说明遇到了没见过的版式。

【已知限制，两个，都不修但要知道】

  限制 1：表格会被抽成竖排。
    chap04a p3 那张 22 行的动物表，抽出来是 "human / warm / yes / no / no / mammals"
    一格一行，共 132 行，列关系丢失。要还原成表格行，得按 bbox 做单元格聚类，
    代价不小；而这批语料的黄金集里没有一道题依赖表格的列对齐（G08~G10 都是概念题），
    所以先不修。**如果以后要加"根据这 5 条规则判断 frog 属于哪一类"这类题，
    必须先做表格还原，否则模型看到的是 132 行散装单词。**
    p31 那种表更麻烦：整行是一个 block，block 内部又竖着分成 5 行。

  限制 2：少数数学符号被解码成别的文字。
    chap03 p60 的 Σ 被抽成 "෍"（U+0DCD，僧伽罗字母），因为 PDF 用了非标准字体编码。
    这类错误无法靠规则区分"真符号"和"错解码"，修它需要看字体名（span 的 font），
    属于"性价比很低"的活儿。影响仅限于公式页，而公式本来就已在黄金集里避开
    （G14 问的是文字定义，不是公式符号）。
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf
from pydantic import BaseModel, ConfigDict, Field

from corpus.docs import DOCS, DocMeta

# 少于这个字符数就算空页。取 50 的依据（实测分布）：
#   5 个空页（chap03 p40/p42、chap04a p22/p24、chap04b p7）都低于 50 字符，
#   而有内容的最短页是 chap03 p11 的 53 字符。50 正好落在间隙里。
MIN_CHARS = 50

# 页面底部多少比例算"页脚区域"。实测页码都在 y≈517/540 = 95.8% 处，
# 取 95% 留一点余量；而 p31 表格里最靠下的数字在 y=456（84.5%），不会被误伤。
FOOTER_Y_RATIO = 0.95

NUMERIC_LINE_RE = re.compile(r"^\d{1,3}$")

COPYRIGHT_PREFIXES = (
    "Slides: Introduction to Data Mining",
    "Introduction to Data Mining, 2nd Edition",
)

# 私有区字符映射。码位来自 Adobe Symbol / Wingdings 编码，含义由上下文实测确认：
#   U+F0D9 → ∧   r1: (Give Birth=no) <F0D9> (Can Fly=yes) <F0AE> Birds
#   U+F0AE → ⇒   同上；r5 只有 ⇒（单条件规则），可反证 F0AE 是箭头而非 ∧
#   U+F0B4 → ×   4/7 <F0B4> 1/7 <F0B4> 0.0072 = 0.0006，验算 0.000588，确认是乘号
#   U+F077/U+F0A7/U+F0D8/U+F075 是各级项目符号
PUA_MAP: dict[str, str] = {
    "\uf0d9": "∧",
    "\uf0ae": "⇒",
    "\uf0b4": "×",
    "\uf077": "•",
    "\uf0a7": "•",
    "\uf0d8": "•",
    "\uf075": "•",
}

PUA_RE = re.compile(r"[\ue000-\uf8ff]")

# 逻辑/数学符号两侧统一成一个空格。
# 实测原文是 "(Give Birth = no) <F0D9>(Can Fly = yes)"：符号前有空格、后没有，
# 映射后成了 "∧(Can Fly"，读起来像函数调用。这里把两侧都规范成单空格。
SYMBOL_SPACING_RE = re.compile(r"[ \t]*([∧⇒×])[ \t]*")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


class PageText(BaseModel):
    """一页的文本。不可变：解析结果不应在后续环节被谁改写。"""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    page: int = Field(ge=1, description="页码，从 1 开始，与 PDF 阅读器一致")
    text: str = Field(description="清洗后的正文；空页为空字符串")
    raw_chars: int = Field(ge=0, description="清洗前的字符数")
    is_empty: bool = Field(description="正文少于 MIN_CHARS 时为 True")

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def dropped_chars(self) -> int:
        """被清掉多少字符。异常增大时说明遇到了没见过的版式。"""
        return self.raw_chars - self.char_count

    def source(self) -> str:
        return f"{self.doc_id}:p{self.page}"


# ------------------------------------------------------------------ 字符级


def map_pua(text: str) -> str:
    """把私有区字符映射成可读符号，未登记的码位换成空格。"""
    return PUA_RE.sub(lambda m: PUA_MAP.get(m.group(0), " "), text)


def is_numeric_footer(line: str) -> bool:
    return bool(NUMERIC_LINE_RE.match(line.strip()))


def is_copyright(line: str) -> bool:
    stripped = line.strip()
    return any(stripped.startswith(p) for p in COPYRIGHT_PREFIXES)


def clean_page_text(raw: str) -> str:
    """字符级清洗：映射私有区字符、去掉版权行和空行。

    **不处理页码** —— 页码由 strip_footer_blocks 按坐标处理。
    第一版把页码处理放在这里（"删最后一行"），在 chap03 p60 上直接失效。
    """
    lines = [ln.strip() for ln in map_pua(raw).splitlines()]
    lines = [ln for ln in lines if ln]
    lines = [ln for ln in lines if not is_copyright(ln)]
    # 符号两侧补空格，再压掉由此产生的多空格
    lines = [MULTI_SPACE_RE.sub(" ", SYMBOL_SPACING_RE.sub(r" \1 ", ln)).strip() for ln in lines]

    # 映射后可能出现只剩一个符号的孤立行（原文该行只有一个项目符号），并入上一行
    merged: list[str] = []
    for ln in lines:
        if ln in {"•", "∧", "⇒", "×"} and merged:
            merged[-1] = merged[-1] + " " + ln
        else:
            merged.append(ln)
    return "\n".join(merged).strip()


# ------------------------------------------------------------------ 块级


def strip_footer_blocks(
    blocks: list[tuple[float, float, str]],
    page_height: float,
    *,
    drop_page_number: bool,
) -> list[tuple[float, float, str]]:
    """按坐标删掉页脚块。

    blocks 是 (y0, x0, text)，已经按阅读顺序排好。
    drop_page_number=False 时仍会删版权行 —— 版权行没有语义价值，
    无论它在哪一页、哪个位置都该删。
    """
    out: list[tuple[float, float, str]] = []
    for y0, x0, text in blocks:
        if is_copyright(text):
            continue
        in_footer = y0 >= page_height * FOOTER_Y_RATIO
        if drop_page_number and in_footer and is_numeric_footer(text):
            continue
        out.append((y0, x0, text))
    return out


def read_blocks(page: pymupdf.Page) -> list[tuple[float, float, str]]:
    """取一页的文本块，按 (y0, x0) 排序还原阅读顺序。

    为什么要 blocks 而不是 get_text("text")：text 的先后顺序是 PDF 内容流顺序，
    页码可能被插到页面中间（实测 p60）。blocks 带坐标，能判断它到底在哪。
    图片块（type=1）跳过：这批讲义里的图没有可提取的文字。
    """
    rows: list[tuple[float, float, str]] = []
    for x0, y0, _x1, _y1, text, _bno, btype in page.get_text("blocks"):
        if btype != 0:
            continue
        stripped = text.strip()
        if stripped:
            rows.append((y0, x0, stripped))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def detect_numeric_footer(doc: pymupdf.Document) -> bool:
    """扫整个文档，判断它是否在页脚印页码。

    判据：超过一半的页，底部区域里有一个纯数字块。
    用比例而不是"有一页就算"：正文表格里也可能出现数字块。
    """
    pages = doc.page_count
    if pages == 0:
        return False
    hits = 0
    for i in range(pages):
        page = doc[i]
        h = page.rect.height
        for y0, _x0, text in read_blocks(page):
            if y0 >= h * FOOTER_Y_RATIO and is_numeric_footer(text):
                hits += 1
                break
    return hits > pages / 2


# ------------------------------------------------------------------ 对外


def extract_pages(
    doc_id: str,
    *,
    path: Path | None = None,
    force_footer: bool | None = None,
) -> list[PageText]:
    """解析一份 PDF，返回逐页文本。

    force_footer 是给测试留的口子：None 表示按实测自动判断，
    True/False 表示强制开启或关闭页码删除。
    """
    meta: DocMeta | None = DOCS.get(doc_id)
    if path is None:
        if meta is None:
            raise ValueError(f"未知 doc_id：{doc_id!r}，且没有显式给 path")
        path = meta.path()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 PDF：{path}")

    doc = pymupdf.open(path)
    try:
        drop_footer = detect_numeric_footer(doc) if force_footer is None else force_footer
        result: list[PageText] = []
        for i in range(doc.page_count):
            page = doc[i]
            blocks = strip_footer_blocks(
                read_blocks(page), page.rect.height, drop_page_number=drop_footer
            )
            raw = "\n".join(text for _y0, _x0, text in blocks)
            cleaned = clean_page_text(raw)
            result.append(
                PageText(
                    doc_id=doc_id,
                    page=i + 1,
                    text=cleaned,
                    raw_chars=len(raw),
                    is_empty=len(cleaned) < MIN_CHARS,
                )
            )
        return result
    finally:
        doc.close()


def extract_all() -> list[PageText]:
    """解析全部 5 份讲义，按 DOCS 的顺序拼成一个列表（共 199 页）。"""
    out: list[PageText] = []
    for doc_id in DOCS:
        out.extend(extract_pages(doc_id))
    return out


def corpus_available() -> bool:
    """语料是否在本机。测试用它决定"跑集成用例"还是"跳过"。"""
    return all(meta.path().is_file() for meta in DOCS.values())
