"""M2-T1 · 黄金集：18 条人工核对的标准问答。

它是什么：一张"评分表"。以后每次改动切分/检索/提示词，都用这 18 条去问系统，
数对了几条。没有这张表，"改完好像好点了"就是你唯一能说的话。

三条纪律（违反任何一条，这张表就废了）：
  1. **答案抄教材口径，不写我的理解**。写我的理解，测出来的就是"系统像不像我"。
  2. **必须带出处页**。答案对不对可以由人争，出处页码是硬事实。
  3. **看过系统输出之后再改答案，等于作弊**。要改只能因为发现原答案抄错了。

三类题（kind）的失败原因完全不同，混在一起就看不出问题在哪：
  fact  概念题。答不出 = 检索没找到，或者找到了但模型没读懂。
  calc  计算题。答案是个数，能按原文公式复算。答错 = 检索到错页，
        或者模型被别的页的相似公式带走了。
  trap  陷阱题。原文本身有瑕疵，或者极易误读。答得"漂亮"反而说明它没读原文。

当前状态：2026-11-17 已核对完毕，18 条 `verified_by_user` 全部为 True。

核对过程中改掉的一条（留个记录，别以后又抄错）：
  G10 原写“二分类时则把样本较少的类当作正类”，把原文的 may choose（可以选）
  强化成了强制规则。已改回带 may 的说法。
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from corpus.docs import DOCS


class GoldKind(str, Enum):
    """题目类型。用 str 枚举，写进 jsonl 时就是普通字符串。"""

    FACT = "fact"
    CALC = "calc"
    TRAP = "trap"


class GoldItem(BaseModel):
    """一条标准问答。

    extra="forbid"：多写一个字段名（比如把 doc_id 写成 docid）会直接报错，
    而不是悄悄多出一个用不上的字段 —— 这是 M1 里 Settings 的同一个套路。
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^G\d{2}$", description="G01 形式，报告里按它点名")
    question: str = Field(min_length=5, description="要问系统的话，直接当 prompt 用")
    answer: str = Field(min_length=10, description="教材口径的标准答案")
    doc_id: str = Field(min_length=1, description="对应 DOCS 的键")
    page: int = Field(ge=1, description="答案所在的起始页，从 1 开始")
    kind: GoldKind = Field(description="fact / calc / trap")
    note: str = Field(default="", description="补充说明：跨页、教材笔误、易错点")
    verified_by_user: bool = Field(default=False, description="人是否已核对过这条答案")

    def source(self) -> str:
        """给人看的出处，形如 chap03:p70。"""
        return f"{self.doc_id}:p{self.page}"

    def check_known_doc(self) -> None:
        """doc_id 必须在登记表里，页号不能超过该 PDF 的总页数。

        单独做成方法而不是 __init__ 里校验，是因为校验要读 DOCS；
        放在构造里会让 gold.py 和 docs.py 循环依赖。
        """
        meta = DOCS.get(self.doc_id)
        if meta is None:
            raise ValueError(f"{self.id}: 未知文档 {self.doc_id!r}，不在 DOCS 里")
        if self.page > meta.pages:
            raise ValueError(
                f"{self.id}: 页码 {self.page} 超过 {self.doc_id} 的总页数 {meta.pages}"
            )


GOLD: list[GoldItem] = [
    # ---------------- chap03 分类基础（82 页，7 题） ----------------
    GoldItem(
        id="G01",
        question="在混淆矩阵里，Type I Error 和 Type II Error 分别对应哪一格？",
        answer=(
            "Type I Error 是 False Positive（FP），即把负例错判为正例；"
            "Type II Error 是 False Negative（FN），即把正例错判为负例。"
            "原文补充：FP 偏高说明模型在“狼来了”式地过度预测正类；"
            "FN 偏高说明漏掉了真正的正例，这在医学诊断中最危险。"
        ),
        doc_id="chap03",
        page=15,
        kind=GoldKind.FACT,
        verified_by_user=True,
        note="TN/FP/FN/TP 四格的定义同在此页。",
    ),
    GoldItem(
        id="G02",
        question="二分类下用 f11、f10、f01、f00 表示时，Accuracy 和 Error Rate 的公式分别是什么？",
        answer=(
            "Accuracy = (f11 + f00) / (f11 + f10 + f01 + f00)，即预测正确的条数除以总条数；"
            "Error Rate = (f10 + f01) / (f11 + f10 + f01 + f00)，即预测错误的条数除以总条数。"
            "两者互为补数：Error Rate 是 Accuracy 的 complement。"
        ),
        doc_id="chap03",
        page=18,
        kind=GoldKind.CALC,
        verified_by_user=True,
        note="Accuracy 公式在 p18，Error Rate 公式在 p19。",
    ),
    GoldItem(
        id="G03",
        question="Gini Index 的定义式是什么？它在哪里取最大值、哪里取最小值？哪些算法用它？",
        answer=(
            "Gini(t) = 1 − Σ(i=0..c−1) p_it²，其中 p_it 是类 i 在节点 t 的频率，c 是类总数。"
            "最大值为 1 − 1/c，出现在记录均匀分布到所有类时，是分类最不利的情形；"
            "最小值为 0，出现在所有记录同属一类时，是分类最有利的情形。"
            "使用 Gini 的算法有 CART、SLIQ、SPRINT。"
        ),
        doc_id="chap03",
        page=57,
        kind=GoldKind.FACT,
        verified_by_user=True,
        note="公式与 CART/SLIQ/SPRINT 同在 p57。",
    ),
    GoldItem(
        id="G04",
        question=(
            "一个节点有 6 条记录、两个类。当 (C1, C2) 分别为 (0,6)、(1,5)、(2,4) 时，"
            "Gini 指数各是多少？"
        ),
        answer=(
            "(0,6)：Gini = 1 − 0² − 1² = 0；"
            "(1,5)：Gini = 1 − (1/6)² − (5/6)² ≈ 0.278；"
            "(2,4)：Gini = 1 − (2/6)² − (4/6)² ≈ 0.444。"
        ),
        doc_id="chap03",
        page=59,
        kind=GoldKind.CALC,
        verified_by_user=True,
        note="三个节点的图与算式同在此页，可直接复算核对。",
    ),
    GoldItem(
        id="G05",
        question="同样三个节点 (0,6)、(1,5)、(2,4)，熵各是多少？",
        answer=(
            "(0,6)：Entropy = 0；"
            "(1,5)：Entropy ≈ 0.65；"
            "(2,4)：Entropy ≈ 0.92。"
            "熵的公式是 Entropy = −Σ p_it log2(p_it)。"
        ),
        doc_id="chap03",
        page=70,
        kind=GoldKind.TRAP,
        verified_by_user=True,
        note=(
            "教材 p70 第二个算式把第二项印成 log2(1/6)，应为 log2(5/6)。"
            "按印刷版字面算会得到 0.86，而给出的 0.65 是按 5/6 算的。"
            "答 0.65 说明读懂了原文；答 0.86 说明照抄了错误公式；"
            "答其他值说明没看这一页。"
        ),
    ),
    GoldItem(
        id="G06",
        question="增益率（Gain Ratio）怎么算？它为了解决信息增益的什么问题？",
        answer=(
            "GainRatio = Gain_split / SplitInfo，"
            "其中 SplitInfo = −Σ(i=1..k) (n_i/n) log2(n_i/n)，n_i 是第 i 个子节点的记录数。"
            "它用“划分本身的熵”作分母，惩罚把数据切成很多小块的分裂 —— "
            "因为信息增益本身偏好取值多的属性。原文：使用该指标的算法是 C4.5，"
            "设计目的正是克服 Information Gain 的缺点。"
        ),
        doc_id="chap03",
        page=73,
        kind=GoldKind.FACT,
        verified_by_user=True,
        note="SplitInfo 公式同在此页。信息增益的公式与“互信息”那句话在 p71。",
    ),
    GoldItem(
        id="G07",
        question=(
            "欠拟合和过拟合在训练误差/测试误差上的表现有什么不同？"
            "原文列出的过拟合原因有哪些，缓解手段有哪些？"
        ),
        answer=(
            "欠拟合：模型太简单，训练误差和测试误差都大。"
            "过拟合：模型太复杂，训练误差小但测试误差大，即在训练集上表现很好、在新数据上表现差。"
            "原因三条：训练数据量有限；模型复杂度过高；算法采用贪心策略（每一步只选当前最优，"
            "可能把无关成分也加进模型）。"
            "缓解五条：增加训练数据、简化模型、交叉验证、剪枝、早停。"
        ),
        doc_id="chap03",
        page=80,
        kind=GoldKind.FACT,
        verified_by_user=True,
        note="定义在 p80，原因在 p81，缓解手段在 p82。",
    ),
    # ---------------- chap04a 规则分类（37 页，3 题） ----------------
    GoldItem(
        id="G08",
        question="一条分类规则 r: A → y 的 coverage 和 accuracy 分别衡量什么？",
        answer=(
            "Coverage：数据集 D 中触发规则 r 的记录比例，也就是满足前件 A 的记录比例，"
            "说明这条规则适用于多大范围的数据。"
            "Accuracy（又叫 confidence factor）：触发规则的记录中类标等于 y 的比例，"
            "也就是同时满足后件的比例，说明在适用范围内有多少条真的属于预测类。"
        ),
        doc_id="chap04a",
        page=5,
        kind=GoldKind.FACT,
        verified_by_user=True,
        note="这一页只给文字定义，没给公式；不要凭空补公式。",
    ),
    GoldItem(
        id="G09",
        question="规则排序的两种方案（rule-based 与 class-based ordering）各自的缺点是什么？",
        answer=(
            "Rule-based ordering：每条规则按各自质量排名，保证每个元组由覆盖它的“最好”规则分类；"
            "缺点是排名靠后的规则难以解释，因为它们隐含地假定了前面所有规则都不成立。"
            "Class-based ordering：同一类的规则放在一起、再按类整体排序，同类内部顺序不重要；"
            "缺点是可能让一条高质量规则被预测了更高排名类别的劣质规则抢先，"
            "即它优先考虑的是类别的整体优先级，而不是单条规则的质量。"
        ),
        doc_id="chap04a",
        page=12,
        kind=GoldKind.FACT,
        verified_by_user=True,
        note="两种方案的缺点各占一段，容易被只记住一种。",
    ),
    GoldItem(
        id="G10",
        question="RIPPER 处理多分类时按什么顺序为各类学规则？最后哪个类成为默认类？",
        answer=(
            "按类别流行度（prevalence，即属于该类的实例所占比例）递增排序："
            "先为最小的类学规则、把其余类都当作负类，然后以次小的类为正类重复。"
            "因此训练集里最大的类最终成为默认类。"
            "二分类时原文只说“可以把样本较少的类选作正类”（may choose the positive class "
            "as the smaller number of cases），正类选谁是可以由人决定的，负类作为默认类。"
        ),
        doc_id="chap04a",
        page=25,
        kind=GoldKind.FACT,
        verified_by_user=True,
        note=(
            "原作用的是 may choose，是建议不是强制；答成“必须选少数类”属于把建议当规则。"
            "递增排序这个方向是硬要求，不可含糊。"
        ),
    ),
    # ---------------- chap04b kNN（14 页，2 题） ----------------
    GoldItem(
        id="G11",
        question="eager learner 和 lazy learner 的根本区别是什么？各举两个原文的例子。",
        answer=(
            "Eager learner 在训练数据一到就完成泛化，立刻构造出把输入属性映射到类标的模型；"
            "例子是决策树、基于规则的分类器。"
            "Lazy learner 把泛化推迟到需要分类测试实例时才做，事先不建模型、只保存训练数据；"
            "例子是 Rote classifier、k-近邻（k-NN）。"
        ),
        doc_id="chap04b",
        page=2,
        kind=GoldKind.FACT,
        verified_by_user=True,
        note="四个例子都在这一页，缺一个就是没读全。",
    ),
    GoldItem(
        id="G12",
        question="k=1 的最近邻分类器决策边界是什么形状？为什么 k 很小时它对噪声敏感？",
        answer=(
            "1-NN 的决策边界构成 Voronoi 图（Voronoi Diagram）。"
            "因为最近邻分类器是局部分类器：分类只依据最近的 k 个训练样本，不使用整个数据集做全局决策；"
            "决策既然是局部的，局部的一个噪声点就能改变一整片区域的归属，所以 k 小时对噪声很敏感。"
        ),
        doc_id="chap04b",
        page=11,
        kind=GoldKind.FACT,
        verified_by_user=True,
        note="该页还列了“能产生任意形状的灵活决策边界”这一性质。",
    ),
    # ---------------- chap04c 朴素贝叶斯（28 页，3 题） ----------------
    GoldItem(
        id="G13",
        question="朴素贝叶斯的“朴素”具体指什么假设？在这个假设下 P(X|Y) 怎么写？",
        answer=(
            "指假设各属性在给定类标 Y 时相互条件独立（conditionally independent）。"
            "于是 P(X1, X2, …, Xd | Yj) = P(X1|Yj) · P(X2|Yj) · … · P(Xd|Yj)，"
            "每个 P(Xi|Yj) 都可以从训练数据单独估计。"
            "新样本被判为使 P(Yj) · Π P(Xi|Yj) 最大的那个类。"
        ),
        doc_id="chap04c",
        page=10,
        kind=GoldKind.FACT,
        verified_by_user=True,
        note="条件独立的定义式 P(X1|X2,Y) = P(X1|Y) 在 p8，配了一个年龄/臂长/阅读能力的例子。",
    ),
    GoldItem(
        id="G14",
        question=(
            "零概率问题为什么会让朴素贝叶斯整个失效？"
            "拉普拉斯估计和 m-estimate 的公式是什么，n、nc、v、p、m 各指什么？"
        ),
        answer=(
            "只要有一个属性的条件概率为 0，整个后验乘积就变成 0，这个类被彻底排除，"
            "哪怕其他属性都强烈支持它。"
            "拉普拉斯估计：P(Xi = c | y) = (nc + 1) / (n + v)；"
            "m-estimate：P(Xi = c | y) = (nc + m·p) / (n + m)。"
            "符号含义：n 是属于类 y 的训练实例数；nc 是属性 Xi 取值为 c 且属于类 y 的实例数；"
            "v 是属性 Xi 可能取值的总数；p 是对 P(Xi = c | y) 的先验估计；"
            "m 是表示对 p 的置信程度的超参数。原文注明：当 nc = 0 时，两种估计都给出非零值。"
        ),
        doc_id="chap04c",
        page=19,
        kind=GoldKind.FACT,
        verified_by_user=True,
        note="这页把 original / Laplace / m-estimate 三个公式并列，五个符号定义同页。",
    ),
    GoldItem(
        id="G15",
        question=(
            "教材例子中（删掉 Tid=7 之后），测试记录 X = (Refund=Yes, Divorced, 120K)，"
            "为什么 P(X|No) 和 P(X|Yes) 都等于 0？"
        ),
        answer=(
            "P(X|No) = P(Refund=Yes|No) × P(Divorced|No) × P(120K|No) = 2/6 × 0 × 0.0083 = 0，"
            "因为 Divorced 在 No 类里出现了 0 次；"
            "P(X|Yes) = P(Refund=Yes|Yes) × P(Divorced|Yes) × P(120K|Yes) = 0 × 1/3 × 1.2×10⁻⁹ = 0，"
            "因为 Refund=Yes 在 Yes 类里出现了 0 次。"
            "两类的后验概率同时为 0，朴素贝叶斯无法判定 X 属于 Yes 还是 No。"
        ),
        doc_id="chap04c",
        page=18,
        kind=GoldKind.CALC,
        verified_by_user=True,
        note="同页给出删除 Tid=7 之后的全套条件概率（分母变成 6）。问题描述见 p17。",
    ),
    # ---------------- chap05 关联分析（38 页，3 题） ----------------
    GoldItem(
        id="G16",
        question=(
            "用教材的 5 条事务表：{Milk, Diaper, Beer} 的 support count 和 support 各是多少？"
            "规则 {Milk, Diaper} → {Beer} 的 support 和 confidence 各是多少？"
        ),
        answer=(
            "support count σ({Milk, Diaper, Beer}) = 2（TID 3 与 TID 4 同时含这三样），"
            "support S = σ/N = 2/5 = 0.4。"
            "规则 {Milk, Diaper} → {Beer} 的 support = σ(X∪Y)/N = 2/5 = 0.4；"
            "confidence = σ(X∪Y)/σ(X) = 2/3 ≈ 0.67，"
            "因为含 Milk 和 Diaper 的事务有 TID 1、4、5 共 3 条。"
        ),
        doc_id="chap05",
        page=3,
        kind=GoldKind.CALC,
        verified_by_user=True,
        note="事务表在 p3（定义页），support/confidence 的公式与同一算例在 p4。",
    ),
    GoldItem(
        id="G17",
        question=(
            "不剪枝直接穷举时，d 个不同 item 能产生多少个候选频繁项集？"
            "原文给出哪三种降低计算量的策略？"
        ),
        answer=(
            "完全搜索（complete search）有 2^d − 1 个候选。"
            "三种策略：① 减少候选数 M —— 用剪枝技术，在不计算支持度的情况下剔除候选；"
            "② 减少事务数 N —— 项集越大，能支持它的事务越少，DHP 与垂直格式算法用这一招；"
            "③ 减少比较次数 N×M —— 用高效数据结构存候选或事务，不必让每个候选与每条事务逐一比对。"
        ),
        doc_id="chap05",
        page=11,
        kind=GoldKind.CALC,
        verified_by_user=True,
        note="2^d − 1 在 p11 的 Reduce the number of candidates 一条里。",
    ),
    GoldItem(
        id="G18",
        question="若频繁项集 L 的大小为 k，能生成多少条候选关联规则？",
        answer=(
            "2^k − 2 条。做法是枚举 L 的所有非空真子集 f，"
            "对每个 f 构造规则 f → L − f，并丢掉 L → ∅ 与 ∅ → L 这两种平凡情况。"
        ),
        doc_id="chap05",
        page=35,
        kind=GoldKind.CALC,
        verified_by_user=True,
        note="同页用 {A,B,C,D} 举了全部 14 条候选规则（2^4 − 2 = 14）的例子。",
    ),
]


def validate_all() -> None:
    """跑一遍全部校验：doc_id 存在、页号不越界、id 不重复。

    供测试和 T4 落盘前调用。发现问题立刻抛出，错误消息里带 id，
    这样 18 条里哪一条写错了能一眼看到。
    """
    seen: set[str] = set()
    for item in GOLD:
        if item.id in seen:
            raise ValueError(f"id 重复：{item.id}")
        seen.add(item.id)
        item.check_known_doc()
