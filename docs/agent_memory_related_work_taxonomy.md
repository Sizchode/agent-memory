# Agent Memory 相关工作分类与论文定位

更新日期：2026-09-09

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

| 方法 | 单元与表示 | 主要操作 | 访问方式 | 更新时机 |
|---|---|---|---|---|
| BM25 | 原始 chunk/turn/passage | 索引、检索 | 稀疏词法 | 建立索引后静态不变 |
| Dense retrieval | 原始 chunk/turn/passage | 索引、检索 | 稠密向量 | 建立索引后静态不变 |
| HippoRAG 2 | 原始 passage，加上图中的抽取实体/关系 | 图构建、索引、检索 | 稠密种子加图传播 | 在本 benchmark 中批量构建 |
| Mem0 | 生成的显著事实；基础版本使用向量库 | 抽取、整合、更新、删除、检索 | 稠密向量 | 增量摄取 |
| LightMem | 压缩后的主题片段，以及分层的感知/短期/长期存储 | 压缩、分段、整合、索引、检索 | 对整合后 memory 做稠密检索 | 缓冲式摄取加离线更新 |

这张表说明，图、向量和流式不能作为同一个分类体系的并列类别。HippoRAG 同时使用图结构和向量种子；Mem0 基于事实、使用向量索引并做增量更新；LightMem 使用层级结构、向量索引和缓冲/离线更新。

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

当前“事实增强 key（fact-augmented key）”原型把原始来源作为 value，生成去上下文化事实作为增强检索 key，并在稠密检索后返回原始来源。该设计有原则依据，因为它把面向检索的表示与可回答的 payload 分离；但**仅凭这一点不足以构成 novelty**。它与命题级检索和 LongMemEval 的事实增强 key 设计有显著重合 [11, 16]。

已经完成的 baseline 结果指出两个更具体的开放问题：

- **跨 memory 证据覆盖（cross-memory evidence coverage）：**HippoRAG 提高了 2Wiki 的 Recall@5，但图抽取和遍历仍会漏掉 supporting passage；
- **演化事实选择（evolving-fact selection）：**所有方法在多跳事实整合上都较弱，因为语义相似度无法表达哪个取值当前有效。

一个可辩护的新方法必须明确加入事实增强 key 尚未提供的操作。这里可以先定义问题，例如：在不损失原始来源忠实度的情况下保留跨来源单元的联系，或依据 benchmark 已发布的顺序显式表示并选择事实间的取代关系；这些是**问题定义，不是在没有错误分析证据前预设的算法**。只有在错误分析确认主要失败来源后才能确定具体机制，之后还必须与 HippoRAG/A-MEM 的关联机制和 APEX-MEM/Mem0 的更新机制对比。

最强的主张形式应当是：

> 在固定 Generator Backbone 和 Evaluation Backbone 的条件下，所提出的 memory 操作改善了一个明确的 memory 层失败（例如证据覆盖或正确版本选择），并且该改善在不增加证据预算的情况下传递到了使用确定性 metric 的端到端 QA。

这种表述使贡献明确落在 memory mechanism 上，而不是语言模型能力上。

## Memory 是否应当负责“推理”？

应当，但必须把两类推理严格分开。

- **Memory-side reasoning** 是 memory mechanism 的组成部分：系统在写入、整合、更新、建立关联和检索时，判断哪些信息应被保存、哪些事实互相支持或冲突、哪个版本有效，以及回答当前 query 需要联合返回哪些证据。
- **Answer-side reasoning** 属于 Evaluation Backbone：模型读取给定证据，完成关系组合、计算、答案类型转换和最终文本生成。

二者的分界不由“是否调用 LLM”决定，而由操作的输入和输出决定。如果一个操作读取历史并输出持久化 memory、版本关系、结构链接或被选中的 evidence set，它属于 memory；如果操作输出最终答案，它属于 Evaluation Backbone。因此，图传播、冲突消解、多跳 evidence selection 都可以是 agent memory 的一部分，而不是只能留给最终答案模型。

这个边界对本文尤其重要。观察到“gold evidence 已完整但三个 Evaluation Backbone 都答错”时，不能直接声称 storage 或 retrieval 失败，也不应通过更换更强 evaluator 来制造 memory 提升。我们有两种合法处理：

1. 将其标为 reader-limited failure，不让它驱动 memory 算法；
2. 如果错误来自证据呈现仍然碎片化、版本关系未解析或必要链条未组织，则设计一个与具体 evaluator 无关的 memory read operation，把相同证据预算编排成更明确的 evidence bundle。

第二种仍然是 memory 改进，因为它改变的是 memory 的选择、消歧和组织，而不是 Evaluation Backbone 的参数或推理能力。但它必须改善可直接测量的 memory 中间量，不能只用端到端 QA 倒推其有效。

## 面向本文的原则化框架

### 核心问题

现有方法通常只优化 memory 生命周期中的一个局部目标，因此产生相互不同但可以统一解释的失败：

| 现有方法 | 主要优势 | 暴露的结构性缺点 | 本文需要的原则 |
|---|---|---|---|
| BM25 | 保留原文，精确词面匹配强 | 无法稳定处理改写和隐式关系 | 检索表示应支持语义匹配 |
| Dense retrieval | 语义匹配强，原文仍可返回 | passage 相互独立，相似度不能表达证据组合或版本有效性 | 检索必须显式覆盖关系链和状态约束 |
| HippoRAG 2 | 图传播提升多跳 evidence coverage | OpenIE 漏掉的事实或边无法由图遍历恢复 | 结构化表示不能替代原始来源，链接必须能回到 source evidence |
| Mem0 | 原子事实和增量更新适合单跳个性化信息 | 抽取会丢上下文；相似新旧事实仍可能同时被召回 | 更新必须显式表示版本、顺序和有效性 |
| LightMem | 分段、压缩和离线整合降低长期构建成本 | 压缩或结构化生成会不可逆地丢失未来问题所需细节 | 压缩应服务于寻址，而不应成为唯一 answerable payload |

这些缺点指向的不是“再换一种 embedding”，而是一个统一矛盾：**memory 为了可检索性而进行抽取、压缩和结构化，但这些变换又可能破坏回答所需的来源忠实度、跨单元联系和时间有效性。**

### 工作中的算法框架：source-grounded、multi-view、state-aware memory

本文可以围绕四个相互约束的 memory 层展开。这里先固定原则和可验证接口，不在错误分析完成前臆造评分公式或数据规则。

1. **不可变来源层（immutable source layer）。** 每个原始 turn、chunk 或 passage 始终作为最终可回答的 payload 保留。任何 summary、fact 或 graph node 都必须能够回链到该来源，生成式变换不能覆盖或替代原文。
2. **多视图寻址层（multi-view addressing layer）。** 从同一来源建立词法、语义事实和关系三种互补访问视图。原子事实用于去上下文化和语义匹配，结构链接用于跨单元关联，但两者只充当 key/index，不充当唯一证据。
3. **显式状态层（explicit state layer）。** 对 benchmark 已提供的 timestamp、serial order、speaker/entity identity 建立可审查的关系，例如 `supports`、`same_entity`、`supersedes` 和 `valid_at`。状态关系只能来自发布数据中的明确字段或可验证抽取，不能由答案或临时 heuristic 倒推。
4. **受预算约束的 evidence composition。** 检索不再独立选择五个最高相似度单元，而是在相同 top-5 预算内选择能够共同覆盖 query 所需实体、关系和有效版本的 source evidence bundle。返回给 Evaluation Backbone 的仍是原始证据，并附带必要的来源顺序或关系，而不是提前生成最终答案。

这个框架将多跳关联、冲突消解和 evidence organization 放在 memory read operation 中，因此可以减少小型 Evaluation Backbone 必须自行完成的隐含推理；同时，不触碰 evaluator 权重、thinking 设置或官方答案 metric。

### 从问题到论文主张的因果链

论文故事不应写成“我们结合了事实、图和向量”，而应写成以下可检验的因果链：

> 现有 memory 在把历史转换成可检索表示时，无法同时保证来源忠实度、跨单元证据完整性和演化事实有效性。我们将 answerable source 与 retrieval-oriented views 分离，并在检索时显式组合关系链和有效版本。该设计应先提高 gold evidence coverage 与正确版本选择，再在相同 Generator Backbone、Evaluation Backbone 和 top-5 预算下提高确定性端到端 QA。

这里真正需要推销的 novelty 不是“混合检索”，而是三个约束的共同满足：

- **source-grounded：**所有生成式索引都有可逆 source link，最终答案上下文不依赖有损 summary；
- **state-aware：**冲突不是交给余弦相似度或答案模型猜测，而是在 memory 层显式保留并解析有效性；
- **coverage-oriented composition：**优化对象从单条相关性变为固定预算内的完整 evidence set。

这三个部分仍分别邻近 Dense X Retrieval/LongMemEval、APEX-MEM/TReMu 和 HippoRAG/A-MEM。最终 novelty 必须落在它们之间尚未解决的具体接口或联合约束上，并通过消融证明，而不能声称这些组成部分本身首次出现。

### 实验必须形成的证据链

为避免把弱 evaluator 的失败错归因给 memory，实验应依次回答四个问题：

1. **写入忠实度：**gold fact/source 是否在构建后仍然存在？仅在 benchmark 提供直接映射时测量，不发明近似匹配规则。
2. **状态正确性：**对于有官方 serial/timestamp 的冲突样本，memory 是否保留并选择正确的最新版本？
3. **检索完整性：**在 top-5 内是否召回全部 gold supporting passages，或 benchmark 明确定义的必要证据？
4. **答案可用性：**在前三项成立时，三个固定 Evaluation Backbone 的确定性 QA 是否同步改善？

若前三项失败，这是本文算法应优先解决的 memory failure。若前三项均通过而答案仍错，应报告为 reader-limited failure；除非能够证明固定、与 evaluator 无关的 evidence composition 操作改善了它，否则不把该题用于支撑 memory novelty。

### 建议的核心消融

所有消融应复用同一批生成 memory，仅移除一个操作：

- 仅返回 raw source 的 dense retrieval；
- 加入 fact key，但仍返回 raw source；
- 再加入跨来源 structural expansion；
- 再加入显式 version/validity resolution；
- 完整方法在相同 top-5 下进行 evidence composition。

这条消融顺序分别检验语义寻址、关系覆盖和状态消解的边际贡献。不能把更大的 context、更多 retrieved items、更强 Generator 或更强 Evaluation Backbone 混进同一项消融。

### 本文不应主张解决的范围

- gold evidence 已完整且顺序清晰，但小模型仍不会做常识推断或复杂计算；
- 官方确定性 metric 不接受语义等价 alias；
- 依赖参数化世界知识、而非外部 memory 的 open-domain 问题；
- procedural skill learning 或跨任务策略改进。

明确这些边界不会削弱论文，反而能证明贡献针对的是可隔离、可测量的 agent-memory 问题。

## 所需对比与消融结构

对于 ACL/ICLR 水平的投稿，当前实验设置可以支持如下清晰对比：

- 原始稀疏和稠密检索提供保留来源、低复杂度的对照；
- Mem0 测试带更新的生成式原子 memory；
- LightMem 测试经过压缩、分段的层级 memory；
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
