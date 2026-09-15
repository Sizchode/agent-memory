# 相关工作核对与当前贡献定位

2026-09-14 状态说明：下文的“下一步”是当时的研究计划，gist、incidence 等候选现已退役。
保留文献核对及负结果背景；当前实现和贡献解释以 [research_novelty.md](research_novelty.md) 为准。

更新：2026-09-13。范围是离线建图及固定检索，不扩展 agentic retrieval。
本文区分论文机制、本地实现事实、待验证判断；不把方向差异直接当作新颖性或效果证据。
本轮不新增经验权重、阈值、数据切分、转换器或评分公式，也不提交新的参数扫描。

## 当前贡献定位（2026-09-14）

当前候选为 Source-Supported Graph Indexing：保留原文与 OpenIE，把选中的来源支持
同时用于构造传播图和筛选事实识别候选。图侧使用事实成员的已有无自环投影并保留来源直连，
索引侧仅选择原事实 ID/向量；不重写原抽取，不增加查询规划，不把已有图算子当作新定理。
新图同索引完整结果为双 reader 6/6；旧 refinement 图同索引为 9B 4/6、4B 6/6，
在相同索引下不能替代新图的目标覆盖率，但新图在 SH 和 9B FCMH 有回退。
这支持当前组合的选择，不证明投影普遍必要或全部 QA 分差都由图引起。

| 近邻工作 | 论文已有机制 | 本文可以明确区分的操作 |
|---|---|---|
| [HippoRAG 2](https://arxiv.org/html/2502.14802#S3) | phrase/passage 图、三元组候选与 recognition/PPR；已有来源连接 | 支持选择先于传播图物化，选中事实的主客体和实际来源共同贡献权重；候选索引也使用相同的选中支持 |
| [CatRAG](https://aclanthology.org/2026.findings-acl.290.pdf) | 针对当前问题调整种子和局部边权 | 改动在离线构图/候选索引，多个问题复用同一索引；不是新的 query-adaptive traversal |
| [HyperGraphRAG](https://arxiv.org/html/2503.21322#S4.SS1) | 新的自然语言 n-ary 事实抽取、二部存储及实体/超边检索 | 复用既有三元组和向量，以实际来源参与事实关联并投影到原节点；不提出新的 n-ary 抽取或超图检索 |
| [A-MEM](https://arxiv.org/html/2502.12110#S3) | LLM 构造 note、建立链接，新记忆触发旧 note 的上下文/属性演化 | 不生成和演化 note 内容，研究既有抽取上的支持选择如何改变两个索引；其写入机制仍是直接相关工作 |
| [Zep](https://arxiv.org/html/2501.13956v1) | 双时间轴事实及基于 LLM 矛盾识别的边失效 | 不实现有效期推断或语义矛盾检测；目前按来源位置选择支持，不能声称更强的时序更新语义 |

可主张的是具体的支持选择、事实来源投影与候选筛选组合及其受控实验证据，不是这些组件
各自首次出现，也不是“多个视图”或“更新”这两个概念的新颖性。上述差异来自对已核对工作的
比较，不是对全部文献的首创性证明。尤其 HippoRAG 2 附录 E 已讨论 recognition 过滤丢失
支持实体的情况，不能把“识别过滤会漏证据”本身作为我们的新发现。

系统的基础记录/派生视图分离是组织设计的语言；目前只能说传播图和候选使用同一份支持，
不能称全部访问路径一致：原历史文本、BM25、来源窗口以及实体来源频次归一仍保留。
原 get_fact_scores 在候选集合内做 min-max，故筛选索引也会改变分数尺度和查询种子；
不是只删除 recognition 的若干输出，也不能证明只靠“减少旧事实干扰”造成全部收益。
若旧图同索引可达到相同目标，应据效果与复杂度考虑进一步删薄，而不是保住图论命名。

## 历史结论（2026-09-13）

我们的工作与近邻方法确实有不同的优化对象：目前主要改变的是查询前的图连接支持，
而不是查询时的路径规划。但更值得推进的主线，是让来源上下文真正参与索引和传播，
不能只在检索完以后把更多上下文拼给 reader。

目前代码仍以 HippoRAG 2 的节点和边为底座，筛选来源后重算权重；
因此可以称为离线图支持优化，暂不能称为已经实现了一套新的上下文图表示。
“离线一次构建”指每份语料构建后供全部问题和 reader 复用，不指一次 LLM 调用。

## 方法差异

| 工作 | 建图或记忆对象 | 在线阶段 | 与当前实现的具体区别 |
| --- | --- | --- | --- |
| HippoRAG 2 | phrase、passage；关系、同义和来源包含边 | query-to-triple、recognition、PPR | 我们继承其基础表示与读取，改来源支持；原文节点、离线图与 PPR 均不是新增机制 |
| CatRAG | HippoRAG 2 底图；另有实体上下文摘要 | 查询实体弱锚定、LLM 评估局部边、关键事实来源增强、PPR | 它按当前问题调整传播，我们在问题到来前冻结支持；它不是必须多轮调用工具的 agentic RAG |
| A-MEM | 带原文、时间、关键词、标签和 context 的 note，note 间链接 | 向量检索相关 note | 新记忆进入时建立链接并演化旧 note；它的 agentic 主要发生在写入和维护，不能以不做 agentic retrieval 为由忽略其建图相关性 |
| Zep / Graphiti | episode、entity、community；带时间和来源的事实 | 混合检索、重排、上下文构造 | 它逐事实判断时序矛盾并记录失效，我们按关系标签和来源顺序筛选支持；不是同一种更新判断 |
| REMem | gist、带时间限定的 fact、phrase 及来源关联 | REMem-I 为 agentic；REMem-S 为单步检索 | 它首先丰富事件表示，我们主要处理已抽取的三元组；其单步版也属于直接相关工作 |
| PropRAG | 上下文完整的 proposition 及其实体关联 | PPR、命题路径 beam search、再次 PPR，无在线生成式 LLM | 它同时改变表示与检索；可以研究其离线表示，但不能把采用其表示的实验称为完整 PropRAG |
| LinearRAG | entity、sentence、passage 的 relation-free Tri-Graph | 句子语义桥接激活实体，再聚合段落重要性 | 它绕开关系抽取，我们依赖关系语义筛选；这提供了应当检验的替代解释：是否根本不需要复杂关系归一化 |

论文依据：[HippoRAG 2 第 3 节](https://arxiv.org/html/2502.14802)、
[CatRAG 第 3 节](https://aclanthology.org/2026.findings-acl.290.pdf)、
[A-MEM 第 3 节](https://arxiv.org/html/2502.12110)、
[Zep 第 2、3 节](https://arxiv.org/html/2501.13956v1)、
[REMem 第 3、4.4 节](https://arxiv.org/html/2602.13530v1)、
[PropRAG 第 5 节和附录 A.1](https://arxiv.org/html/2504.18070)、
[LinearRAG 第 3 节](https://arxiv.org/html/2510.10114)。

两处容易误写的地方：

- CatRAG 的动态是 query-conditioned transition，不等于持续改写长期记忆；其代码在 PPR 后恢复原边权。
- PropRAG 的命题是重要的索引和路径单元，但第 5.1 节及附录中的物理图节点是实体与段落，
  同一命题中的实体形成 clique / implicit hyper-edge；不能只凭摘要写成“所有命题都是图节点”。

## 本地代码核对

### 我们：语义判断在投影成图权重时进一步丢失

[source_consolidation.py](graph_construction/source_consolidation.py) 的
`retained_statements` 使用主体、关系、来源顺序及 schema 决定保留支持。
但同文件 `latest_relation_weights` 将保留记录汇总为排序后的实体对计数，
并将 passage-entity 支持转成是否存在的二值连接。

这意味着关系标签会影响哪些记录留下，却不作为独立关系通道进入最终传播权重；
同一实体对的不同关系、方向和多次来源贡献在这一汇总中无法分别表达。
原始三元组和 passage 仍在存储里，不等于这些区别在 PPR 传播时仍然可用。
这是代码中的表示限制，不是已经证明导致某类 QA 错误的因果结论。

旧 `relation_identity.py`（实现已退役并归档）的消融仅取消后续 canonical 映射，
并未改变上述实体对汇总。因此“去掉错误合并还不够好”不能排除更细粒度表示的价值。

### REMem：有真实的中间上下文单元，但本地关联仍偏粗

[episodic_gist_strategy.py](../baseline_algorithms/REMem/src/remem/rag_strategies/episodic_gist_strategy.py#L239)
实际建立 verbatim、gist、entity、fact 的关联；gist 有独立 embedding，
并在 gist 之间添加相似连接。它不只是给最终答案增加一段摘要。

同文件的构图循环把一个 chunk 中的每个 gist 连接到该 chunk 的各个 fact 及其主客体。
这是 chunk 级共同来源关联，并非逐条确认的 gist-fact 对齐。
因此它也不能直接保证同段多事件不会串联；但这只是结构上的可能性，尚未量化其影响。
此观察针对当前检出的本地实现，不外推到作者所有版本或配置。

抽取接口还存在需要明确披露的区别：本地
[episodic_gist_extraction_openai.py](../baseline_algorithms/REMem/src/remem/information_extraction/episodic_gist_extraction_openai.py#L104)
按 json_mode 选择无 response_format 或 json_object，并在 length 时调用已有 JSON 修复函数。
我们的组件试验只复用了模板，使用严格 json_schema，且拒绝 length 和内容修复。
目前两条来源的重复生成失败首先是这套抽取接入未完成，不能据此声称 REMem 表示无效。
格式诊断与完整图/QA 比较分开记录；两个失败来源的诊断通过也不是新图方法验证。

### CatRAG：确实改变在线边权，本地仓库自述为复现

[CatRAG.py](../baseline_algorithms/CatRAG/src/catrag/CatRAG.py#L1740) 中
`graph_search_with_fact_entities` 调整种子实体到关键事实来源的权重，
调用 `call_llm_score` 评估查询相关邻居，再运行 PPR，最后恢复被改动的边。
这与我们的 source-only 冻结支持不同，不能统称为同一种图剪枝。

[README](../baseline_algorithms/CatRAG/README.md) 明确使用 reproduced implementation 的表述。
现有 CatRAG 实测成绩应标注所用复现代码与配置，不写成已经核验作者官方实现完全一致。

### LinearRAG：上下文在检索中使用，而不只在末端展示

[LinearRAG.py](../baseline_algorithms/LinearRAG/src/LinearRAG.py#L218) 的实体激活读取关联句子及其
query embedding 相似度；句子关联另存为映射或稀疏矩阵，最终再运行段落 PPR。
因而“使用句子层”与“把 sentence facts 拼给 reader”不是同一机制。
本轮只核对其相关路径，不将本地整个实现自动认定为论文的逐项忠实复现。
A-MEM 和 Zep 本轮核对到论文机制，未完成其作者代码的逐函数审计。

### Dense X Retrieval：索引粒度与读取粒度可以不同

[Dense X Retrieval 第 4.3 节](https://aclanthology.org/2024.emnlp-main.845.pdf)
给出了现成的细粒度索引对照：逐 proposition 编码，一个 passage 的分数取其内部
proposition 与 query 相似度的最大值，返回去重后的 top-k 原段落。
该节的 Passage Recall 是答案是否出现在返回段落中的题目比例，不是本项目 2Wiki
使用的 gold supporting-passage 集合召回；两者不能直接拼表。
论文采用五个开放域 QA 数据集、不同 retriever 和 reader，未复现为本地 baseline。

我们当前 `ordered_gist_texts` 把同一来源的 gist 换行拼接后编码，一个来源仍只有一个向量。
REMem 本地构建既有逐 gist 单独索引，也有 `concatenate_gists_per_chunk` 分支
（[代码](../baseline_algorithms/REMem/src/remem/rag_strategies/episodic_gist_strategy.py#L117)）。
因此“采用 gist 文本”和“保留细粒度索引单元”是两个不同轴，拼接表示失败不能否定逐单元索引。

下一步判断：先完成已冻结的四候选；若继续检验粒度，复用完整 gist 及已发表的 max-to-source
聚合做独立对照，不新造融合分数，不重新切分原始数据，不同时改变 reader 读取单位。
这属于已有方法的索引组件，不是我们的新数学结构；图贡献仍须与相同表示的纯向量检索比较。

## 据此调整改进顺序

### 第一优先：先检验上下文化表示，不继续扩展标签级政策

先按 REMem 或 PropRAG 已公开的离线抽取定义分别建立表示对照，
而不是把几篇方法的抽取、连边、打分和过滤同时拼成一个候选。
选定一个已有表示后，保持输入语料及原始来源映射不变，不从 QA 或 gold evidence 生成记忆。
模型和数据适配与论文不同的部分必须单列；没有证据时不自行补全人物、时间或事实。

同一套已抽取表示至少需要区分：仅向量检索该表示，以及通过其来源关联进行图检索。
先验证表示本身，再检验关联结构的独立收益。若使用新的索引单元，不能继续声称
旧 triple recognition / reset 完全不变；涉及新单元到 seed 的适配也要单独列出。
固定 reader、QA prompt、最终原文读取方式和评分器，不靠多轮查询补救。

这一阶段是已有方法组件的受控对照，不预先命名为我们的新算法。
其结果用于决定是否值得进一步研究更精确的上下文与来源关联。

### 第二优先：把关联粒度当成待回答的问题

当前有两个具体限制：我们的实体对汇总，以及 REMem 本地实现的 chunk 级 gist-fact 关联。
可以据已有来源记录检查哪些关联有直接来源支撑，哪些仅是同段共现。
没有细粒度标注时不把共现边自动标成错误，也不凭问题答案人工挑选保留边。
在确认现有抽取输出能支持什么粒度之前，不新造对齐阈值或权重公式。

若后续实验确证更细关联有独立收益，贡献可以落在“上下文关联如何进入检索图”，
而不是泛称“首次上下文化”“首次离线记忆”或“首次保留来源”。

### 第三优先：时间更新独立处理，不再用最后出现替代事实有效期

Zep 区分事实有效时间与系统摄入时间；REMem 保留带时间限定的历史事实。
这两种已有机制都比按关系标签统一保留最后来源更接近具体事实级处理。
需要时先做已有机制的对照，不追加新的 state/event 标签级覆盖规则。
尤其不能对没有时间语义的百科段落，把 loader 顺序解释成事实新旧。
按最新讨论，以更新/consolidation 为问题动机；但时间线不是所有任务共同瓶颈，
普通检索任务用于检查适用范围，不能作为同等强度的冲突解决证据。

### 更新主线的补充近邻（2026-09-14）

重新核对原论文方法节，而非合并其不同模型或不同数据范围的成绩：

| 工作 | 与更新相关的实际机制 | 与当前实现的边界 |
|---|---|---|
| [TEPA](https://arxiv.org/html/2608.07429) | 带 Hypothesis/Active/Revoked 生命周期的记忆；用支持/冲突证据更新状态，限制检索到 active 记录 | 我们离线选择图支持，不采用其后验、阈值或验证试验；历史原文仍可被检索，不是相同的撤销语义 |
| [StateFuse](https://arxiv.org/html/2607.05844) | 不可变操作与针对 claim 的撤销，显式 ConflictSet，查询投影不改基础状态 | “保留历史、派生视图”已有直接先例；我们没有复制合并、显式冲突接口或语义撤销句柄，不能借用其系统保证 |
| [Reliable Post-Retrieval Assembly](https://arxiv.org/html/2606.01435) | 检索后抽取候选和版本编号，执行确定的新版本选择；多跳逐步处理 | 改动位于检索后/推理流程；当前按用户范围冻结此阶段，研究来源支持如何进入离线图，不移植其 query decomposition |

更新/consolidation 本身不能作为 novelty。当前可区分的具体操作是选中来源支持如何形成
图传播权重；四格主实验和支持选择消融需要支持这一主张。
本地 reader 已含新编号优先指令（utils/prompts.py），但图只按来源段落取最后支持，
同段多个值仍共存。[MAB 方法与附录](https://arxiv.org/html/2507.05257v4#S3)
不能替当前实现提供事实级冲突处理的保证。保持已有划分、loader 和指标，不为故事另造冲突子集。

## 评测与可比较性

本轮已检查 [loader.py](../dataset_loader/loader.py#L73)、
[runner.py](../experiments/runner.py#L421) 和 [report_results.py](report_results.py#L11)。
保持既定协议，不为上述研究方向重写切分、转换或评分：

- 四个 MemoryAgentBench 任务各 100 题，来自当前 loader 指定的官方主实验 source；
  SH/MH 使用 Accurate_Retrieval，两个 FactConsolidation 使用 Conflict_Resolution。
  主指标是 substring exact match，不改成 token F1。
- LoCoMo 使用 locomo10 的全部 1986 题，保留原类别评分及 adversarial 处理。
- 2Wiki 使用 HippoRAG 发布的 1000 题及配套 corpus，不称为整个原始 2Wiki 数据集全量。
- 当前同一候选覆盖上述六任务全部 3386 题和 4B/9B 两个 reader，共 6772 次回答；2B 已退役。
  不能拿部分题完成或逐任务挑配置的结果代替整个方法。
- 沿用已有 gold passage 指标检查检索，仅在有官方支持证据的任务上使用；
  不把答案字符串命中率包装成证据正确率，不新增自定义综合分数。
- 保持实际 token、离线 LLM、在线 recognition / retrieval 和 QA 成本分列；
  五个记忆单元不代表等 token，现有九 baseline 也包含较大上下文的 AnchorMem 设置。

论文原始排行榜不能直接并入我们的分数表：HippoRAG 2 使用不同 reader / embedding；
CatRAG 的构建、检索模型及 QA 表设置应分开看；A-MEM 主表包含类别 F1 / BLEU；
REMem 另报 BLEU、LLM judge 等，且有不同推理模式。
上述论文采用的检索预算、语料构造和评测范围并不相同。
本地 test 已用于开发，继续明确披露；九个完整 baseline 的胜出不等于全领域 SOTA。

## 本轮执行边界

首次文献核对只更新研究路线，未新增算法实现或提交新实验。
随后继续推进的来源事实 incidence 结构对照及其完整作业矩阵，
见 [fact_incidence_experiment.md](https://github.com/Sizchode/agent-memory/blob/c443de4790e619d43c5bd9dcce9ccea707da56bc/optimization/fact_incidence_experiment.md)。
该组现已完成，9B/4B/2B 分别严格胜出 1/6、2/6、1/6，未取代旧最佳候选。
据此启动 [上下文化索引表示对照](https://github.com/Sizchode/agent-memory/blob/c443de4790e619d43c5bd9dcce9ccea707da56bc/optimization/contextual_gists_experiment.md)：
只复用 REMem 公开抽取模板，先完成来源表示，再区分向量匹配与图传播。
当前运行的是抽取阶段，不能声称新的图检索或 QA 已有效。
已有状态/事件实验不取消。此前 rank-window 的 FactConsolidation-SH 检索作业
6339634 已以 0:0 完成，两候选各 100 题的旧 top5 前缀校验通过；
它尚不是六任务三 reader 的完成结果，也不属于纯建图贡献，不据此声称提升。
