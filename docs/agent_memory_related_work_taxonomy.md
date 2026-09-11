# Agent Memory 相关工作分类与论文定位

更新日期：2026-09-11

## 核心结论

现有文献并不支持将 agent memory 简单地平铺为“图、实体、向量或流式”四类。这些词描述的是 memory 系统的不同属性：

- **图（graph）**是一种表示形式；
- **实体（entity）**是一种可能的 memory 单元或节点类型；
- **向量（embedding）**通常是一种索引和检索信号；
- **流式（streaming）**描述 memory 写入或更新的时机。

因此，同一个系统可以同时以实体为中心、采用图结构、用向量建立索引，并以流式方式更新。论文中最清晰的分类方式是多维分类：系统存储什么、如何表示、实现了哪些 memory 操作、如何访问，以及何时发生变化。这一归纳与 Du 等人的“表示—操作”分类，以及较新的“形式—功能—动态”和“写入—管理—读取”综述一致 [1–3]。

我们的 related work 正文应采用简洁的历史演进叙事，对比表则采用多维分类。除非新方法确实存储了功能不同的多类 memory，否则没有必要展开冗长的认知科学分类。

## 范围：本文所说的 agent memory

本项目研究**外部、非参数化 memory（external, non-parametric memory）**：从文档、对话或交互历史中得到的信息被持久化到 Evaluation Backbone 之外，之后作为证据提供给下游答案模型。该范围不包括常规模型预训练、循环架构中的隐状态 memory 和模型编辑；但包括最基本的普通检索，因为它也实现了信息的持久保存和后续访问，即使它没有学习得到的整合或更新策略。

全文应始终区分三个角色：

- **Generator Backbone：**读取原始历史，生成、抽取、压缩或更新 memory。
- **Memory mechanism：**决定存储单元、表示、维护方式、索引和检索策略。
- **Evaluation Backbone：**消费检索出的证据并生成最终答案。

这种区分可以避免把更强答案模型带来的提升错误归因于 memory。

## 可辩护的多维分类体系

### 1. Memory 功能：保留哪类知识？

认知科学词汇在该领域很常见，但只有当方法真正实现了相应区别时才应使用：

| 功能 | 在 agent 中的可操作含义 | 典型内容 |
|---|---|---|
| 工作 memory（working memory） | 当前任务或交互内所需的临时状态 | 当前目标、中间结果、最近几轮对话 |
| 情景 memory（episodic memory） | 带有上下文的“发生过什么”的记录 | 对话轮次、事件、轨迹、工具结果 |
| 语义或事实 memory（semantic/factual memory） | 从记录中抽象出的、与具体上下文相对独立的知识 | 用户属性、命题、实体事实 |
| 程序或经验 memory（procedural/experiential memory） | 关于“如何行动”的可复用知识 | 计划、技能、成功或失败的策略 |

当前 benchmark 主要测试情景和事实 memory，不能支持程序 memory 或持续学习方面的主张。ACL 2026 关于“遵循经验”行为的实证研究与程序/经验 memory 有关，但它也表明：当系统检索到表面相似但不适用的记录时，存储经验可能传播错误 [17]。

### 2. Memory 单元与粒度：可寻址对象是什么？

常见单元包括 token 或 chunk、对话轮次、session、主题一致的片段、原子事实、事件或轨迹，以及多粒度混合。粒度不是无关紧要的 data loader 选择：在固定 top-k 证据预算下，它直接改变系统能够检索到什么。

SECOM 将其作为核心研究问题：turn、session 和 summary 级 memory 具有不同的完整性与噪声权衡，因此提出主题一致的分段与压缩 [4]。RMM 和 MemGAS 同样认为单一固定粒度无法适应所有 query，因而采用多粒度 memory 构建或选择 [10, 14]。

### 3. 表示：memory 单元以什么形式存储？

| 表示形式 | 优势 | 典型局限 |
|---|---|---|
| 原始文本或 passage | 忠实于来源；写入便宜；证据可审查 | 冗余且碎片化；没有显式更新或关系语义 |
| 摘要或 profile | 紧凑、连贯；节省下游上下文 | 不可逆的信息遗漏；过度概括；摘要可能过时 |
| 原子事实或命题 | 信息经过筛选，可独立检索，也更易更新 | 抽取错误；丢失篇章、角色、时间和来源上下文 |
| 实体/事件图 | 显式表达关系、路径和时间结构 | 开放信息抽取与实体链接可能出错；构建昂贵；缺边会阻断遍历 |
| 层级或混合 memory | 保留多种粒度，或结合互补检索信号 | 组件更多，难以归因；需要额外路由与融合决策 |

Du 等人采用更宽泛的分类：上下文化非结构表示、上下文化结构表示和参数化表示 [1]。“实体 memory”属于结构化事实或情景 memory，而不是“向量 memory”的同级类别。同样，一个图方法也常常会对节点或 passage 做向量化。

### 4. Memory 操作：观察到信息后，系统能够做什么？

Du 等人总结了六种反复出现的操作：**整合、更新、索引、遗忘、检索和压缩**（consolidation, updating, indexing, forgetting, retrieval, compression）[1]。与仅按存储后端命名相比，这套分类更适合比较方法。

- **整合**把多个观察合并为稳定的 memory。
- **更新**修订或取代已存储的信念。
- **索引**为访问建立地址、向量、链接或元数据。
- **遗忘**删除低价值或过时内容，或降低其权重。
- **检索**为当前 query 选择证据。
- **压缩**在尽量保留效用的前提下减少 memory 或上下文长度。

Mem0 强调显著事实抽取，以及显式的 ADD/UPDATE/DELETE/NOOP 式维护；图版本进一步加入关系结构 [7]。LightMem 结合预压缩、主题分段、短期整合和离线“睡眠期”长期更新 [8]。A-MEM 的重点是动态创建、链接和演化 note，而不是使用固定图 schema [9]。这些方法不能笼统地都称为“LLM 生成的 memory”：它们的 novelty 位于不同的操作上。

### 5. 访问机制：系统如何选择相关 memory？

| 访问机制 | 有充分依据的优势 | 主要失败模式 |
|---|---|---|
| 稀疏词法检索 | 精确名称、罕见字符串、标识符和强词面重合 | 改写表达和隐含关联 |
| 稠密向量检索 | 语义匹配和改写匹配 | 细微实体区分、版本冲突和关系组合 |
| 图遍历或传播 | 多跳关联和证据整合 | 依赖正确的抽取、链接和初始检索节点 |
| 时间或元数据过滤 | 显式处理日期、顺序、身份和有效性约束 | 元数据必须被正确捕获；刚性 schema 可能不完整 |
| 混合或 agentic 检索 | 可以结合互补证据信号和工具 | 融合/路由增加成本，也使提升更难归因 |

HippoRAG 的贡献并不只是“使用知识图谱”。它指出相互孤立的 passage 检索会妨碍跨文档证据整合，并结合 LLM OpenIE、图、稠密 query 种子和 Personalized PageRank，实现单步多跳检索 [5]。APEX-MEM 结合以实体为中心的时序属性图、仅追加历史和检索时冲突消解 [13]。RMM 则利用答案所引用证据的反馈来学习改进检索 [10]。

### 6. 更新时机与控制：memory 在何时、由谁改变？

重要区别包括：

- 在线或流式写入，与批量/离线整合；
- 仅追加历史，与破坏性替换；
- 固定规则，与学习得到或由 LLM 控制的决策；
- 立即更新，与延迟到“睡眠期”的更新。

当新信息必须立即可用时，流式写入很有吸引力，但会增加每轮延迟，并对信息顺序敏感。离线整合能够高效处理更长上下文并比较多个观察，但在整合完成前 memory 会保持过时。仅追加设计保留冲突历史，有利于时间推理；替换设计更紧凑，却可能擦除审查或撤销错误更新所需的证据。LightMem 明确主张使用离线整合，将昂贵计算与在线交互解耦 [8]；APEX-MEM 则保留所有版本，并在检索时解决冲突 [13]。

## 受控 baseline 的分类映射

2026-09-11 更正：论文机制与本地执行路径必须区分。只读审计确认 LightMem 尚未执行最终离线合并，当前 Mem0 官方 SDK 使用 ADD-only 流程。下表描述实际产物；前文对论文的介绍不表示对应流程在本项目中都已执行。[证据和三轮研究记录](memory_research_iterations.md)

| 方法 | 单元与表示 | 主要操作 | 访问方式 | 更新时机 |
|---|---|---|---|---|
| BM25 | 原始 chunk/turn/passage | 索引、检索 | 稀疏词法 | 建立索引后静态不变 |
| Dense retrieval | 原始 chunk/turn/passage | 索引、检索 | 稠密向量 | 建立索引后静态不变 |
| HippoRAG 2 | 原始 passage，加上图中的抽取实体/关系 | 图构建、索引、检索 | 稠密种子加图传播 | 在本 benchmark 中批量构建 |
| Mem0 SDK `dae67f74` | 生成事实及上下文；使用向量库 | 抽取、追加、去重、检索；本轮无 UPDATE/DELETE | 官方 SDK search | 顺序增量摄取 |
| LightMem，离线合并前 | 预压缩、主题分段后抽取的事实 | 压缩、分段、抽取、插入、检索；本轮无最终离线合并 | 对已插入 memory 做稠密检索 | 缓冲式摄取 |

这张表说明，图、向量和流式不能作为同一个分类体系的并列类别。HippoRAG 同时使用图结构和向量种子；Mem0 SDK 基于事实、使用向量索引并做增量追加；LightMem 本轮执行了缓冲式摄取，但不能称为已完成官方离线更新。

## 高水平论文如何组织 related work

简洁的 related work 可以按机制演进来组织，而不是罗列产品名称：

1. **检索支持的外部 memory。** 使用稀疏或稠密检索为原始 turn 或 passage 建立索引。这是保留原文忠实度的最低复杂度 baseline，但它把 memory 单元彼此独立地处理。
2. **构建和压缩后的 memory。** 摘要、命题、主题片段和 profile 用来降噪或改变粒度。SECOM、Mem0 和 LightMem 均属于这一方向，但采用不同的构建和维护操作 [4, 7, 8]。
3. **关系式和层级式组织。** HippoRAG/A-MEM 加入关联；MemoryOS/LightMem 加入层级；APEX-MEM 加入显式时序事件结构 [5, 6, 8, 9, 12, 13]。
4. **自适应 memory 控制。** RMM 从回答证据中学习检索；较新的工作研究自适应粒度和 memory policy，而不是固定的写入/读取规则 [10, 14, 17]。
5. **评测。** 分开衡量检索覆盖率、确定性的下游 QA、构建成本，以及更新/冲突处理行为。只看端到端答案准确率，无法判断错误来自构建、检索还是 Evaluation Backbone。

“Storage → Reflection → Experience（存储→反思→经验）”三阶段视角提供了另一种历史叙事：系统从保留轨迹，发展到精炼轨迹，再到跨轨迹抽象可复用知识 [2]。它适合综述段落，但过于粗糙，不足以比较我们五个受控 baseline。

## ACL/ICLR/NeurIPS 论文如何定位 novelty

最有说服力的论文通常采用同一个论证模式：

| 论文 | 诊断出的局限 | 与局限对应的机制 | 支撑主张的证据 |
|---|---|---|---|
| HippoRAG，NeurIPS 2024 [5] | 独立 passage 向量无法整合跨 passage 证据 | OpenIE 图加 Personalized PageRank | 多跳检索/QA、与迭代检索的效率对比、消融 |
| SECOM，ICLR 2025 [4] | 固定 turn/session/summary 粒度会产生碎片或噪声 | 主题分段加压缩降噪 | 不同粒度上的检索与 QA；组件消融 |
| RMM，ACL 2025 [10] | 固定 memory 粒度与检索无法适应对话 | 前瞻式多粒度反思加回顾式检索学习 | 多 benchmark/metric 和机制消融 |
| A-MEM，NeurIPS 2025 [9] | 固定 memory 操作和静态组织限制适应性 | agent 生成的 note 属性、动态链接和 memory 演化 | 多 backbone、任务和组件消融 |
| MemoryOS，EMNLP 2025 [12] | 扁平存储无法管理长期个性化交互 | 短期、中期和长期三层 memory 及显式迁移 | LoCoMo 效果与效率分析 |
| LightMem，ICLR 2026 [8] | Memory 构建和在线更新造成过多调用、token 与延迟 | 早期过滤/分段加离线整合 | 准确率、调用数、token、运行时间和消融 |
| APEX-MEM，ACL 2026 [13] | 相似度检索无法可靠处理演化或冲突事实 | 仅追加时序属性图和检索时冲突消解 | LoCoMo/LongMemEval 和时序分析 |
| MemGAS，ICLR 2026 [14] | 单一粒度要么证据不完整，要么噪声过多 | 多粒度关联和 query 自适应选择 | 多任务、query 类型和 top-k 下的 QA 与检索 |

共同教训是：认知科学或系统比喻可以阐明动机，但不能独自支撑 novelty。除非论文明确指出改变了哪一种表示或操作，并衡量其后果，否则“类人 memory”“图 memory”和“更好的组织”都过于宽泛。

### 经得住审稿的定位模板

1. 准确说明最近邻方法的具体失败。
2. 在受控的 Generator Backbone、Evaluation Backbone、证据预算和官方确定性 metric 下展示该失败。
3. 引入直接针对该失败的最小机制。
4. 衡量该机制理应改善的中间量，例如 gold evidence coverage、正确版本选择率或保留事实覆盖率。
5. 单独报告下游 QA，因为即使 memory 和 retrieval 正确，答案模型仍可能失败。
6. 对每项声称的操作做消融；如果效率属于论文主张，还要报告 memory 构建成本、检索成本和存储规模。
7. 明确方法边界，例如只主张事实/情景式文档和对话 memory，而不主张程序性技能学习。

### 应当避免的主张

- “首个基于图的 memory”：HippoRAG、A-MEM、Mem0 的图版本和时序属性图系统都是直接先行工作。
- “首个层级式 memory”：MemoryOS 和 LightMem 都是直接先行工作。
- “原子事实可以改善检索”：命题级检索和事实增强检索已经建立这一方向 [7, 11, 16]。
- “返回原始证据可以避免信息损失”：这是合理设计，但相关检索和长程 memory 工作已经使用保留来源的 key/value 检索 [11, 16]。
- “更好的端到端 QA 证明更好的 memory”：结果也可能来自 Evaluation Backbone 推理更好，或输入 token 更多。
- “Prompt engineering 解决 memory 更新”：只改变 prompt 措辞并不会产生显式版本、有效期或冲突消解机制。

## 我们当前设计方向的定位

本项目明确研究 **training-free agent memory**。Generator Backbone 可以执行推理来形成、整合、压缩、关联与更新记忆；所有模型权重保持固定。学习型 memory control 是 related work 的一类，不是本项目采用的路线。

方法贡献必须在持久化 memory 中体现。BM25、Dense 与既有检索流程提供配套索引及对照，Retrieval 和 QA 检验记忆质量。训练 retriever、训练 reranker，或仅改变查询时证据排序，均偏离本项目目标。

当前候选方向是保留事实条件与依赖的记忆整合。核心问题是：压缩和更新之后，陈述的主体、关系、时间、角色与限定条件是否仍然成立；某条陈述被修订时，依赖它的既有记忆能否一起更新，并复用未受影响的内容。

完整提案见 [Training-free memory 提案](training_free_memory_proposal.md)。该机制尚未实现，不能把它写成已有原型或声称取得效果。

## 与主要 memory 算法的区别应怎样论证

主要对标 Mem0、LightMem 和 HippoRAG 2，并将 A-MEM 作为动态关联及记忆演化的重要近邻。

- Mem0 已有事实整合和更新，因此“让模型决定更新”本身不构成创新。
- LightMem 已有主题组织和离线整合，因此批处理、压缩或减少在线调用本身不构成创新。
- HippoRAG 已有跨来源关系结构，因此建图和保留关联本身不构成创新。
- A-MEM 已有链接生成与记忆演化，因此需要具体核对新提案的条件表示、整合依赖和更新范围是否产生额外价值。

候选主张只能是：同一个 memory 整合原理改善了条件保留与后续修订，并在固定下游流程中带来效果和效率收益。这需要官方实现对比、真实错例和机制消融，不能由组件组合或论文措辞成立。

最新讨论优先考察由完整陈述、适用条件和支持组合约束的建图，而不是相似度连边。但最近邻已经很接近：[StructMem](https://aclanthology.org/2026.acl-short.12/) 做事件绑定，[MemForest](https://arxiv.org/abs/2605.23986) 做局部更新，[ContextWeaver](https://arxiv.org/abs/2604.23069) 做依赖图，[Dependency-Guided Rollback Repair](https://arxiv.org/abs/2608.10502) 做局部恢复。它们收窄而不是自动证明本项目的创新空间。完整比较、实际错例与尚未解决的问题见研究迭代记录。

## Evaluation Backbone 推理的研究边界

固定 Generator 在记忆写入时判断陈述关系和修订语义，属于 training-free memory 操作。Evaluation Backbone 消费记忆并产生答案；证据充分但答题模型仍推理错误的情况，不是本项目优先优化对象。

答案字符串已经出现不能作为“证据充分”的判据。只有官方证据标注或逐题检查支持时，才将失败归因于 memory、取用或回答阶段。不通过增加查询时推理模块，把 reader 改进重新命名为 memory 改进。

## 方法实验应当证明什么

1. 在相同 Generator 与输入下，记忆构建是否保留此前丢失的条件或关系。
2. 在官方支持更新评测的任务上，新事实到来后，既有记忆是否仍存在错误覆盖或过时内容。
3. 固定检索与答题流程后，官方确定性 QA 是否改善。
4. 新增依赖维护的成本计入之后，是否减少总构建或更新开销。

只在官方标注可直接支持时报告中间指标，不发明近似 evidence 标签或“严谨性”总分。静态最终状态的 QA 结果不足以证明完整的在线持续更新能力。

消融应改变记忆构建或维护操作，例如条件保留、依赖保存和更新范围。涉及构建的消融必须生成对应记忆；已有 baseline memory 继续复用。不能在完整记忆上只换一种检索方式，就声称验证了构建模块。

## 所需对比与消融结构

对于 ACL/ICLR 水平的投稿，当前实验设置可以支持如下清晰对比：

- 原始稀疏和稠密检索提供保留来源、低复杂度的对照；
- 当前 Mem0 SDK 测试追加式生成 memory，不能代表原论文的增删改操作；
- 当前 LightMem 测试压缩、分段和抽取后的 memory，不能代表已完成最终离线合并；
- HippoRAG 测试关系图检索；
- 新方法必须使用相同 Generator Backbone、三个固定且关闭 thinking 的 Evaluation Backbone、seed 42、top-5 证据预算、官方 split 和确定性 metric。

新方法的消融实验每次只应移除一个声称的操作。只有在 gold source evidence 能够直接映射、无需发明 heuristic 时，才报告 retrieval metric。无法映射回 source passage 的生成式 summary/fact，应将 passage precision/recall 标为不适用。对所有方法报告端到端答案 metric，然后通过检查必要证据是缺失、已检索但未正确使用，还是在 memory 构建中丢失，对失败进行分类。

## 一手文献

1. Du et al. [*Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions*](https://arxiv.org/abs/2505.00675), 2025.
2. Luo et al. [*From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms*](https://aclanthology.org/2026.findings-acl.2069/), Findings of ACL 2026.
3. Hu et al. [*Memory in the Age of AI Agents*](https://arxiv.org/abs/2512.13564), 2025.
4. Pan et al. [*On Memory Construction and Retrieval for Personalized Conversational Agents*](https://openreview.net/forum?id=xKDZAW0He3), ICLR 2025.
5. Gutiérrez et al. [*HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models*](https://papers.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html), NeurIPS 2024.
6. Gutiérrez et al. [*From RAG to Memory: Non-Parametric Continual Learning for Large Language Models*](https://openreview.net/forum?id=LWH8yn4HS2), HippoRAG 2.
7. Chhikara et al. [*Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*](https://arxiv.org/abs/2504.19413), 2025.
8. Fang et al. [*LightMem: Lightweight and Efficient Memory-Augmented Generation*](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a05b72653ec5b473732129829ae04195-Abstract-Conference.html), ICLR 2026.
9. Xu et al. [*A-MEM: Agentic Memory for LLM Agents*](https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html), NeurIPS 2025.
10. Tan et al. [*In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents*](https://aclanthology.org/2025.acl-long.413/), ACL 2025.
11. Chen et al. [*Dense X Retrieval: What Retrieval Granularity Should We Use?*](https://aclanthology.org/2024.emnlp-main.845/), EMNLP 2024.
12. Kang et al. [*Memory OS of AI Agent*](https://aclanthology.org/2025.emnlp-main.1318/), EMNLP 2025.
13. Banerjee et al. [*APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI*](https://aclanthology.org/2026.acl-long.749/), ACL 2026.
14. Xu et al. [*From Single to Multi-Granularity: Toward Long-Term Memory Association and Selection of Conversational Agents*](https://openreview.net/forum?id=i2yIvZARnG), ICLR 2026.
15. Maharana et al. [*LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents*](https://aclanthology.org/2024.acl-long.747/), ACL 2024.
16. Wu et al. [*LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*](https://openreview.net/forum?id=pZiyCaVuti), ICLR 2025.
17. Xiong et al. [*How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior*](https://aclanthology.org/2026.acl-long.27/), ACL 2026.
