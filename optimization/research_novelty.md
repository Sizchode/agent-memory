# 当前算法的贡献边界与理论解释

## 当前写作摘要

暂名 **Source-Supported Graph Indexing（来源支持驱动的图索引）**。
主线是 consolidation 之后如何派生索引，而不是给 HippoRAG 换一个超图名称：
原文和 OpenIE 保留为基础记录，既有支持选择决定传播结构；最新独立实验进一步
让相同支持决定哪些原始事实进入 recognition 候选，不重抽取、不改向量。
传播图使用事实成员的已有无自环投影，加上来源直连；在线继续使用原 recognition/PPR
和已冻结的 RRF、读出、QA。这是一套离线图索引方法，不是在线更新系统或新的检索器。

系统的“基础记录与派生视图”最贴近实现，图论描述具体投影；目前不以 PL 理论为贡献。
候选筛选已完整达到双 reader 六任务 6/6；它改善部分任务，但两 reader 的 2Wiki 小幅回退。
“旧 refinement 图 + 同样新候选索引”已完整完成，为 9B 4/6、4B 6/6，不能替代新图的双 6/6。
默认采用新图加保留事实索引；新图的 SH 和 9B FCMH 回退，不是逐任务支配。
来源顺序选择仍是经验政策，不能因使用物化视图术语就改称无 heuristic 的理论算法。

### 论文主张与 novelty

标题方向：**Consolidate Before You Propagate: Source-Supported Indexing for Graph Memory**。

问题起点是“保留什么知识”与“沿什么结构访问知识”并非同一件事：
图侧已经移除的事实支持，仍可能在识别候选中竞争；只筛候选也没有改变传播连接。
因此研究支持选择应如何同时体现在传播图和事实识别候选中，而不是只调一组边权。
本文研究同一保留支持如何派生这两个索引，并在固定抽取、向量与下游算子的条件下，
分别检验事实来源投影和候选筛选的作用。这里的“同一支持”只指这两个派生索引，
不包含仍可访问全部历史的原文库、BM25、来源窗口和原实体来源映射。

可用于 introduction 的英文主张：We study source-supported graph indexing, in which
the same retained fact support determines both propagation structure and the fact candidates
exposed to recognition. Reusing existing OpenIE outputs and embeddings, we jointly examine
source-aware graph construction and candidate-index restriction under fixed downstream operators.

机制解释假设是减少派生访问结构中的历史干扰，同时保留原文作为完整证据库；
并非全部答案收益的已证实原因。候选筛选也改变原 min-max 分数及查询种子，
现有采样差异限制纯图归因。“最后来源优先”是既有经验政策，不等于语义有效性判断。

贡献是这一具体设计及其分离对照，不是新的超图投影定理、首次 memory update，
或全局索引一致性保证。旧图配同一候选索引仍有部分任务更强，因此应写成整体配置的
经验取舍，不能把双 6/6 的覆盖率当成新图逐任务支配或统计显著性。

## Experiments: Dataset Selection and Coverage

以下是当前实验范围的英文稿，可放入实验设置；不是全部 MemoryAgentBench 的评测。
六项是来自三个 benchmark suite 的 task settings，不是六个独立数据集。

**Dataset selection and coverage.** Our evaluation comprises six task settings from three benchmark suites, covering three requirements relevant to offline graph-based memory: access to updated facts, single- and multi-hop evidence retrieval, and long-term conversational recall. From [MemoryAgentBench](https://arxiv.org/html/2507.05257v4), we use SH-Doc QA, MH-Doc QA, FactConsolidation-SH, and FactConsolidation-MH, with 100 questions in each selected released configuration. The consolidation tasks directly test updated-fact access, while the document-QA tasks assess retrieval utility beyond explicit updates. [LoCoMo](https://snap-research.github.io/locomo/) contributes all 1,986 questions from locomo10 across single-hop, multi-hop, temporal, open-domain, and adversarial categories, extending evaluation to multi-session conversational histories. For [2WikiMultiHopQA](https://aclanthology.org/2020.coling-main.580/), we use the [HippoRAG 2](https://arxiv.org/html/2502.14802#S4.SS2) released 1,000-question subset and associated corpus, enabling comparison on multi-document reasoning with supporting-evidence annotations. Together, these 3,386 questions assess whether the selected graph and fact index support updated-fact access while retaining utility across document and conversational QA. We evaluate every question in these selected configurations and retain task-specific scoring procedures. This combination provides complementary coverage of the proposed offline indexing scope, not exhaustive coverage of agent memory; procedural learning, tool-use policies, and online adaptation are not evaluated.

**Development protocol.** The evaluation sets were also used for method development and configuration selection; we therefore report in-distribution benchmark results rather than held-out generalization estimates. Each graph is constructed from source material without evaluation questions, answers, or supporting-evidence labels, and is frozen before retrieval. We retain the benchmark-specific scoring procedures across ablations and report each task separately.

本地范围核对：dataset_loader/loader.py 的 _TASK_SOURCES 分别为 ruler_qa1_197K、
ruler_qa2_421K、factconsolidation_sh_262k、factconsolidation_mh_262k；不混入长度消融。
LoCoMo 保留五类 QA 的现有评分与 adversarial 处理，不把它扩写成已经评测摘要或多模态生成。
2Wiki 的 1000 题来自发布子集，不是原始数据全集。每个 reader 共 3386 题，4B/9B 共 6772 次 QA，
不能把双 reader 的回答数称为独立题目数。四项 MAB 各一个长上下文，LoCoMo 十段对话，
2Wiki 一个共享语料库；题目共享记忆库，不以题数证明独立样本或统计充分性。

范围的理由是覆盖相关能力与输入形态，而不是由任务数量证明“足够”。特别是本文不评测
MAB 的 test-time learning 和完整 long-range understanding 任务组，也不复现其整套在线
增量交互能力评测。FactConsolidation 是直接的更新测试；其余任务检查一般检索/推理的取舍。
若论文扩展为通用或在线 agent memory，当前任务组合不能单独支持该扩大后的主张。

## 问题主线：更新与 consolidation

按最新讨论，以历史事实如何参与当前记忆为问题起点，而不是以超图或系统术语起笔。
核心 refinement 是来源支持 consolidation：按主体/关系槽选择最后来源的支持，
再由新构图将选中支持物化为传播权重。canonicalization 是可消融的分槽组件，
事实关联/无自环投影是表示机制，记录与索引分离是解释框架。
待验证主张是 consolidation 与图构建的联合效果，不是 latest-only 或超图本身的新颖性。

[MemoryAgentBench 的 FactConsolidation](https://arxiv.org/html/2507.05257v4#S3)
提供按编号排序的更新任务；本地 Conflict_Resolution 两项保留其新编号优先指令和 SubEM。
更新能力的直接证据应来自这两项及其消融，其他四项不能被重新命名成冲突解决任务。
当前是对既有历史语料的一次离线整合，不是在线增量更新实现。
选择按段落/来源进行：同一最终来源中的多个值全部保留，并未实现事实编号级的消歧；
普通百科语料的 loader 顺序也不能解释为事实时间。被移除的是相应图支持，原文仍可经
BM25、PPR 种子或来源窗口进入 reader，因此不是“撤销后永不可检索”的 active-only memory。

## 可执行的结构探索

按新增目标，概念必须对应可实现的操作或能区分机制的实验，不只用于给已获选配置命名。
目前优先探索两个与代码直接对应的结构，不引入新的权重公式或任务规则：

| 结构 | 与现有实现的对应 | 下一项可检验问题 | 不能预称的能力 |
|---|---|---|---|
| 来源支持上的多个派生视图 | 保留事实支持分别生成传播图和识别候选索引 | 新图同索引双 6/6，旧 refinement 图同索引为 4/6、6/6；支持两者联合选择，但非逐任务支配或显著性证明 | 所有索引全局一致、所有旧事实不可访问、语义冲突已经解决 |
| 同一节点集上的两类加权连接 | 事实成员的无自环投影，加上额外来源直连；运行时求和后交给原 PPR | 已有删除直连的完整负结果；后续在获选索引下复查必要性，而不是扫描混合系数 | 新的高阶推理理论、层间扩散算法或纯超图投影单独胜出 |

系统线可进一步参考 [DBSP，PVLDB 2023](https://www.vldb.org/pvldb/vol16/p1601-budiu.pdf)
的 Z-set 与增量视图维护：记录变化可以表达为插入/撤销，再更新派生视图。
负的变更计数用于撤销记录，不是向现有 PPR 注入负权。我们尚未实现该引擎或增量算子，
不能继承其 Lean 证明、复杂度或性能主张，也不能把已有 latest 选择说成由 DBSP 推出的语义。
只有在静态方法确定后，才考虑用既有增量框架检验“更新后的索引是否等于同数据全量重建”，
测量维护成本；这将是单独的系统实验，不替代当前六任务完整 QA。

图论线参考 [Kivela 等的 Multilayer Networks](https://arxiv.org/abs/1309.7233)
作为区分连接类型的表示语言。当前仍是两类贡献相加的普通加权图，不运行保留层状态的游走。
事实层继续采用 [Kumar 等式 3](https://link.springer.com/article/10.1007/s41109-020-00300-3)
的已有度数保持约简；其性质属于引用算子，不能改名当作我们的新定理。
支持成员变化会改变该事实的投影归一项，因而不能仅删一条来源边就假装完成更新维护。

当前暂不转向 hyperbolic embedding、PL 语义保持或 signed PageRank：前者没有已验证的
层次结构动机且需改变冻结向量，后两者分别缺少程序语义对象或需要改动在线传播算子。
不是这些方向无价值，而是目前没有比多视图索引和连接分层更直接的证据支撑。

## 新构图研究状态

单轮 QA 分差的归因边界：同 H100 的旧 refinement 与新主图在 FCSH 有 73/100 题的
完整 prompt 与解码设置相同，但其中 9B/4B 分别有 9/6 题的答案分数不同；LoCoMo 也存在
这种现象。官方采样协议不变时，统一设置 seed 不等于逐题输出一致，故四格 QA 差值
不能全部解释为构图的因果贡献。保留完整实际分数，不通过择优重跑制造稳定性。
已有单例还显示目标事实首条检索命中后 reader 仍答旧值，以及参考长人名被原 10-token
上限截断；改图并不能直接消除这些已定位的 reader 错误。详细输入核验见 record.md。

当前另一个可操作的问题是派生图与事实识别索引使用不同支持范围：图按选中支持构建，
recognition 候选仍来自全部历史三元组。单例已显示新事实进入候选后仍被 recognition 排除。
这为独立检验离线候选索引提供动机，不证明筛选索引一定有效；旧事实索引组合曾有完整负结果。
新索引对照必须与已完成的固定识别候选四格分开，不把索引变化算成旧图实验的收益。
当前完整索引对照为双 6/6，旧 refinement 图同索引格为 4/6、6/6；单轮采样差异的归因限制仍然适用。

2026-09-14：已完成独立事实节点及标准 incidence 随机游走投影的两轮对照，
分别有无 graph refinement。两轮完整结果均已收齐：
有 refinement 的两种表示均为 9B/4B 双 5/6，收益与回退并存，不统一支配旧方案。
去掉原 passage-entity 直连贡献的全量消融已完成，联合 refinement 后为 9B 4/6、4B 5/6；
该失败分支已归档并从代码移除，保留直连。后续无自环事实投影已完成六任务双 reader：
单独新构图为双 4/6，联合 refinement 为 9B 5/6、4B 6/6。
9B FCSH 65 未超过最佳 baseline 66；相对冻结 refinement 参照也不是逐任务无损改进。
同 H100 的旧图两格完整复现与核验现已完成：旧 refinement 为 9B 4/6、4B 6/6，
无自环联合方案为 5/6、6/6。不能用历史旧分的 4B 5/6 来证明新构图提升其胜出覆盖率。
对旧 refinement 的 9B MH/FCSH/FCMH/2Wiki 有提高，SH/LoCoMo 回退，非逐任务支配。
全部支持删除消融已完整完成，为 9B 5/6、4B 4/6，不能采用。
默认现改为无自环联合方案；显式事实节点和含自环投影的独立入口退役，代码归档、结果保留。
清理后全量核验状态、完整同硬件表与历史表均见 record.md。
依据分别是 [HyperGraphRAG 的关联表示](https://arxiv.org/html/2503.21322#S4.SS1) 和
[Zhou 等的超图随机游走](https://papers.nips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf)。
这些结构和公式是已有工作，不是新理论；我们未采用其整套抽取、训练或检索方法。

可检验的系统视角是：来源支持是逻辑记录，传播图是其物化索引；支持选择发生在度数与投影
之前。删掉一个来源会改变事实成员数，因此先投影再删连接不等于重新物化选中支持。
这是现有操作之间的依赖，不是新的优化器正确性定理。显式事实节点上的 PPR 和投影后的 PPR
在相同 damping 下不等价，不能把前者的两跳未经验证地写成后者的一跳压缩。
目前不增加 Kron reduction、MemorySSA、bisimulation 等名称来暗示尚未实现的保证。

### 写作主线与待验证问题

暂用描述性名称 Source-Supported Graph Indexing（来源支持的图索引构建）。
以 update/consolidation 组织问题，以系统的“基础记录与派生索引分离”解释设计，
以事实关联和已有图约简描述算法；
定位仍是图记忆方法研究，不是仅凭命名成立的系统性能或 PL 理论贡献。
原文/OpenIE 是基础记录，选中支持是派生关系，传播图是供固定在线流程使用的物化索引。
已有 HippoRAG 也包含来源节点；区别必须落在支持如何共同贡献传播权重，而不是有无来源信息。

待验证的核心假设是来源支持选择与传播图构造的联合效果。原图/新图分别有无 refinement
组成四格主对照；已完成“全部支持但不恢复 synonym”的删除消融进一步区分支持选择与同义边。
这不是完整因子分解：旧 refinement 仍包含来源连接筛选与去 synonym，reader 附录仍依赖
canonical/latest。不得用图侧删除成功推断整个 reader 不需要这些组件。
比较采用预先指定的 H100 旧图复现，不把历史低分作为新构图收益的唯一参照。

全部支持消融因 4B SH/FCMH 不达标而失败，暂保留图侧关系归一/latest 链条的条件作用，
但不能因此把 loader 顺序提升为事件时间语义或正确性保证。不追加新公式
来解释分差；没有建图内存/体积/时延实测，就不称为压缩索引或更高效的系统。

后续仅删除图侧 canonical 分槽的完整消融仍为 9B 5/6、4B 6/6：归一不是维持当前胜出
覆盖率的必要条件，不能将其独立贡献写成已证实。它使 FCMH 从 11/9 变为 9/6，
9B LoCoMo 从 57.05 提高到 57.58，其他项有持平和小幅变化；本轮没有多种子显著性证据。
默认保留 canonical 版本，完整 raw 版本作为取舍对照，非失败达标实验。
reader 仍使用 canonical/latest 附录，故这也不是整个系统取消归一的实验。

在当前无自环算子下单独删除额外 passage-entity 直连的全量结果为 9B 4/6、4B 5/6，
低于主图的 5/6、6/6；MH 从 63/62 降至 59/59，而 LoCoMo 从 57.05/50.81 提高到
57.82/51.15。故获选配置是事实投影与已有来源直连的组合，不是纯事实投影单独胜出。
这组仍保留来源作为事实成员，不能用它证明有无 provenance 的整体效应。
2Wiki 来源 Recall@5 从 87.30 到 87.55，但答案 F1 未形成两 reader 一致提高，
再次不能以证据召回替代 QA，也不能将组合中的全部收益归于“高阶表示”。

### 新表示的具体边界

当前新增的操作是把一个规范化三元组及其实际来源共同作为事实成员，再物化传播权重。
这与仅按实体对累计出现次数不同，但不意味着所有事实语义都进入了在线传播：
谓词只决定事实身份，PPR 不读取谓词标签，也不执行有向关系组合。
显式节点保留独立事实身份；投影矩阵则不保证能反推出原来的 incidence 分解。
不同谓词若具有完全相同成员，其投影贡献会叠加，不能称为无损的关系语义索引。

[HyperGraphRAG 4.1](https://arxiv.org/html/2503.21322#S4.SS1) 已有超边及二部存储，
并生成自然语言 n-ary 事实及置信度；我们的事实来自冻结的 OpenIE 三元组，来源节点也参与成员。
这只是具体构造和实验边界的区别，不据此宣称超图或来源关联本身具有 novelty。
[HippoRAG2](https://arxiv.org/html/2502.14802) 本来就有 passage 节点、来源连接及实体/段落 PPR 种子；
原候选索引下的四格对照保持查询识别不变，比较离线事实支持如何形成图权重；
新增候选索引对照则改变候选及实际查询种子，不能混为相同的冻结条件。
当前可检验的贡献仍是这一构造与已有支持筛选的组合及其效果/成本取舍，
不是新的抽取器、检索算法、可逆编码或语义保持编译器。

全量成员计数还显示：refined 的 167074 个活跃事实中，167072 个只有一个来源，
只有 2 个包含两个来源。因此不能把多来源超边的聚合作为该联合方案的主要收益解释。
对通常由主体、客体、单一来源构成的事实，当前投影直接改变的是这几个节点之间的相对连接
权重；在线仍在普通加权图上执行 PPR。高阶结构的表示形式本身不证明高阶推理收益。
当前新增的无自环投影对照采用 [Kumar 等 2020](https://link.springer.com/article/10.1007/s41109-020-00300-3)
的既有度数保持图约简；
该度数性质属于引用算子，不是我们证明的新定理，且不蕴含 QA 或 PPR 排名保持。

## 历史与复现

refinement-only 阶段的理论解释和旧对照已移入归档，避免以“固定拓扑仅改权重”描述当前
事实来源投影。原文保存在
`/oscar/home/zliu328/agent-memory-archives/construction_experiments_20260914/research_notes_before_current_method_cleanup.tar`，
归档与原文逐文件比较通过后整理；不删除实验结果或缓存。

当前算法入口和普通查询接口见 [README.md](README.md)，新旧结构、单模块删减、失败运行、
完整 QA 与成本见 [record.md](record.md)。最新近邻方法比较见
[related_work_analysis.md](related_work_analysis.md#当前贡献定位2026-09-14)。
不从已归档阶段继承未经新索引验证的必要性结论；旧图-only 消融也不等于整条流水线的删除。
