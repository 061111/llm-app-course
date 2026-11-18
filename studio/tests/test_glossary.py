"""术语抽取：证明三件事 —— 抽得到、抽得对、出处点得准。

第三条最容易忽略：术语表上写着 p3，点进去却找不到这个词，
这种"出处错位"比抽不出术语更伤信任。所以每个术语都带
(页, 起始偏移, 结束偏移)，测试直接用它去原文切片核对。
"""
from __future__ import annotations

from corpus.parser import PageText

from studio.glossary import GlossaryBuilder, mine_terms, snippet


def _page(text: str, page: int = 1, doc_id: str = "u001") -> PageText:
    return PageText(doc_id=doc_id, page=page, text=text, raw_chars=len(text), is_empty=False)


# 一段人造资料：术语"距离矩阵"出现 6 次，左右邻字刻意多样化（模拟真实语料）
SAMPLE = "\n".join(
    [
        "系统聚类法的第一步是计算距离矩阵，距离矩阵记录了样品两两之间的距离。",
        "得到距离矩阵之后，我们合并距离最小的两类。距离矩阵会随之更新。",
        "更新后的距离矩阵规模变小，因此距离矩阵的对角线始终为零。",
        "判别分析同样需要距离矩阵，只不过它用马氏距离而不是欧氏距离。",
    ]
)


def test_抽得出领域术语():
    terms = mine_terms([_page(SAMPLE)], top_n=20)
    texts = [t.text for t in terms]
    assert "距离矩阵" in texts
    # 这段人造资料里"距离"几乎只以"X距离"的形式出现，被长词吸收掉是对的；
    # 真实语料里它独立出现 129 次，会作为独立术语保留（见 docs/upload-service.md）
    assert "矩阵" not in texts


def test_术语出处能精确切回原文():
    terms = mine_terms([_page(SAMPLE)], top_n=20)
    term = next(t for t in terms if t.text == "距离矩阵")
    occ = term.occurrences[0]
    page = _page(SAMPLE)
    assert page.text[occ.start : occ.end] == "距离矩阵"
    assert occ.doc_id == "u001" and occ.page == 1


def test_上下文里确实含这个术语():
    terms = mine_terms([_page(SAMPLE)], top_n=20)
    for t in terms:
        for occ in t.occurrences:
            assert t.text in occ.context


def test_频次只数真实出现的次数():
    terms = mine_terms([_page(SAMPLE)], top_n=20)
    term = next(t for t in terms if t.text == "距离矩阵")
    assert term.freq == SAMPLE.count("距离矩阵")


def test_半个词不会被当成术语():
    """'距离' 是 '距离矩阵' 的前缀，但它在原文里也独立出现很多次，
    所以它**应该**保留；而 '离矩' 这种既不是词又总在长词内部的碎片应被丢掉。"""
    terms = mine_terms([_page(SAMPLE)], top_n=20)
    texts = {t.text for t in terms}
    assert "离矩" not in texts


def test_通用停用词被过滤():
    text = "没有 就是 可以 说明 因此 所以 但是 如果 这些词在文中反复出现，" * 8
    terms = mine_terms([_page(text)], top_n=10)
    texts = {t.text for t in terms}
    assert not texts & {"没有", "就是", "可以", "说明"}


def test_文档太短时不硬造术语():
    terms = mine_terms([_page("短")], top_n=10)
    assert terms == []


def test_空输入返回空表():
    assert mine_terms([], top_n=10) == []


def test_多文档联合统计并按页记出处():
    a = _page("先算距离矩阵。再看距离矩阵的规模。把距离矩阵存下来。", page=1, doc_id="a")
    b = _page("按距离矩阵合并两类。更新距离矩阵之后继续。这个距离矩阵很小。", page=2, doc_id="b")
    terms = mine_terms([a, b], top_n=20)
    term = next(t for t in terms if t.text == "距离矩阵")
    assert term.freq == 6
    assert set(term.pages()) == {"a:p1", "b:p2"}


def test_拉丁缩写能被抽到():
    text = "使用 SPSS 做因子分析。SPSS 输出载荷矩阵。SPSS 与 R 都可以。"
    terms = mine_terms([_page(text)], top_n=20)
    assert "SPSS" in {t.text for t in terms}


def test_每类最多保留三处出处():
    text = "先算距离矩阵。再看距离矩阵。按距离矩阵合并。更新距离矩阵。这个距离矩阵很小。"
    page = _page(text)
    term = next(t for t in mine_terms([page], top_n=10) if t.text == "距离矩阵")
    assert term.freq == 5
    assert len(term.occurrences) <= 3


def test_builder_可增量喂入():
    b = GlossaryBuilder()
    b.add_pages([_page(SAMPLE, page=1)])
    b.add_pages([_page(SAMPLE, page=2)])
    terms = b.build(top_n=10)
    term = next(t for t in terms if t.text == "距离矩阵")
    assert term.freq == SAMPLE.count("距离矩阵") * 2


# ---------------------------------------------------------------- 原文片段
#
# 这一组守的是"术语旁边那行字读得懂"。原来的做法是左右各切 18 字再拼接，
# 遇到表格（解析后一个单元格一行）就会把上下两格粘成一句、两头还各断一刀：
#   （旋转是标准步骤） 典型用途 降维、综合评价、去共线性、可视化 寻找潜变量、给指
# 用户看到的第一反应是"这是病句吧"。所以要按行/按句取，不按字数取。


def test_表格单元格取整行并补上表头():
    page = "\n".join(
        [
            "是否有旋转",
            "无",
            "有（旋转是标准步骤）",
            "典型用途",
            "降维、综合评价、去共线性、可视化",
            "寻找潜变量、给指标分类、问卷结构效度",
        ]
    )
    i = page.index("综合评价")

    got = snippet(page, i, i + len("综合评价"))

    assert got == "典型用途 › 降维、综合评价、去共线性、可视化"


def test_长段落按句读切出完整一句():
    page = "第一句。" + "啊" * 70 + "这里有术语存在。" + "哦" * 70
    i = page.index("术语")

    got = snippet(page, i, i + len("术语"))

    assert got == "啊" * 70 + "这里有术语存在。"
    assert got.startswith("啊")   # 没有断头
    assert got.endswith("。")     # 没有断尾


def test_超长无标点行也不会切出一百字以上():
    page = "甲" * 80 + "术语" + "乙" * 80

    got = snippet(page, 80, 82)

    assert "术语" in got
    assert len(got) <= 101


def test_章标题不当表头前缀():
    """带空格的短行是章标题，不是表格列名，拼成前缀只会让人更糊涂。"""
    page = "第 3 章　聚类分析\n3.1 如何测度样本与变量之间的相似性"
    i = page.index("变量")

    got = snippet(page, i, i + len("变量"))

    assert not got.startswith("第 3 章")
    assert "变量" in got
