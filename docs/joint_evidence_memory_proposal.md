# Joint Evidence Memory：基于联合证据结构预测的 Agent Memory

状态：研究提案，尚未实现，尚无方法结果。

## 一句话定义

Joint Evidence Memory（JEM）是一种只追加原始 memory、在读取时联合选择固定预算证据集的外部 memory。它不在写入时用 LLM 重写历史，也不预先对全语料构建 OpenIE 图；它先用词法与稠密检索得到候选，再用一个从官方 supporting evidence 学到的结构化模型，在相同 top-5 预算内一次选择互相连接、时间状态一致的原始证据。

核心研究问题不是“如何让 Evaluation Backbone 更会推理”，而是：

> 在不丢失原文、不进行全语料 LLM 图抽取、也不逐条调用 LLM 更新 memory 的前提下，memory read operation 能否找齐一个问题所需的跨来源证据，并在存在演化事实时选对有效版本？

## 1. 提案来自哪些实证失败

最终基线包含 BM25、Dense、LightMem、HippoRAG 2、Mem0，覆盖六个保留任务和 Qwen3.5-9B/4B/2B 三个固定、关闭 thinking 的 Evaluation Backbone。所有数字均来自官方确定性 metric。

### 1.1 独立相似度没有直接优化多跳证据集合

2WikiMultiHopQA 的结果如下：

| 方法 | Recall@5 | Precision@5 | QA F1：9B / 4B / 2B |
| --- | ---: | ---: | ---: |
| BM25 | 0.6583 | 0.3106 | 0.451 / 0.426 / 0.339 |
| Dense | 0.6883 | 0.3228 | 0.488 / 0.460 / 0.365 |
| HippoRAG 2 | **0.7498** | **0.3544** | **0.519 / 0.498 / 0.370** |

在 BM25、Dense 和 HippoRAG 2 都召回全部 gold passages 的同一批 286 题上，9B QA F1 分别为 0.768、0.783、0.778；当 HippoRAG 2 召回完整证据、但至少一个普通检索方法未召回完整证据时，三者分别为 0.220、0.352、0.590。HippoRAG 2 的主要收益因此来自 evidence-set coverage，而不是替 Evaluation Backbone 完成答案推理。

### 1.2 生成式 memory 的更新能力不等于跨文档证据能力

Mem0 在 LoCoMo 上最好，9B/4B/2B F1 为 0.537/0.497/0.444；但在 2Wiki 上只有 0.436/0.407/0.348，三个 evaluator 都低于 Dense 和 HippoRAG 2。LightMem 在 2Wiki 上更低，为 0.343/0.328/0.292。

这说明原子事实、压缩与更新可以帮助个性化对话，却可能删掉未来多跳问题需要的关系或上下文。JEM 因此不让生成的 fact 或 summary 成为唯一 answerable payload。

### 1.3 词面和语义信号互补

Dense 在 LoCoMo 和 2Wiki 上优于 BM25，却在 SH-Doc QA 和 FactConsolidation-SH 上明显落后。精确名称、数值和关系仍需要词面匹配，而改写和隐含语义需要稠密匹配。JEM 的候选生成保留两种信号，但不把简单分数加权当成核心贡献。

### 1.4 多跳事实整合首先是 memory failure，其次才是 reader failure

FactConsolidation-MH 中，三个 evaluator 全错且答案字符串不在 top-5 的题数为：BM25 76、Dense 76、LightMem 94、HippoRAG 2 77、Mem0 89。答案已经出现在 top-5、但三个 evaluator 仍全错的题数为 15、19、1、18、2。

第一组远大于第二组，说明优先级应是找齐正确事实。第二组仍然存在，所以 JEM 输出的不只是五个独立片段，而是带有可检查连接关系和顺序的 evidence bundle；但它不修改 Evaluation Backbone，也不把 reader-only error 计为 memory 改进。

### 1.5 现有强方法的成本不可忽略

| 方法 | 六任务构建时间 | 构建期 Generator 调用 | 构建期 Generator tokens |
| --- | ---: | ---: | ---: |
| LightMem | 10,821.23 秒 | 3,482 | 4,833,992 |
| HippoRAG 2 | 8,212.09 秒 | 28,724 | 20,617,866 |
| Mem0 | 97,783.49 秒 | 14,362 | 139,416,826 |

HippoRAG 2 在读取时还使用 3,374 次 LLM 调用和 10,120,992 tokens。Mem0 的顺序更新最慢。JEM 的核心版本不使用 Generator Backbone，目标是把主要学习成本放到一次训练、把新 memory 写入降为普通索引更新。

## 2. 与最接近工作的边界

JEM 不能把“保留原文”“联合证据检索”或“时间状态”单独声称为首次提出。

| 先行工作 | 已经解决的部分 | JEM 必须与之区分的部分 |
| --- | --- | --- |
| [HippoRAG 2](https://openreview.net/forum?id=LWH8yn4HS2) | 全局 OpenIE 图、passage/phrase integration、LLM recognition memory、PPR 多跳检索 | JEM 不构建全局生成图；它只在 query 候选中学习并预测 evidence set，返回原始 memory |
| [AnchorMem](https://aclanthology.org/2026.findings-acl.1736/) | fact anchor 与 immutable context 解耦，使用 associative event graph 恢复上下文 | JEM 不以生成 fact anchor 或 event graph 为核心；它直接对原始候选做受 top-k 约束的结构化集合预测 |
| [APEX-MEM](https://aclanthology.org/2026.acl-long.749/) | append-only 时序属性图，查询时由多工具 agent 解决演化事实 | JEM 使用一个固定的判别式 read model 和精确约束推断，不生成检索摘要，不依赖 query-time LLM agent |
| [RMM](https://aclanthology.org/2025.acl-long.413/) | 多粒度反思，并从答案引用的证据反馈中用 RL 改进检索 | JEM 直接以官方 supporting set 为监督，训练目标是固定预算下的集合级结构预测，不从 Evaluation Backbone 的自由文本反馈学习 |
| [Joint Candidate Evidence Retrieval](https://aclanthology.org/2021.naacl-main.363/) | 已经明确提出多条候选 evidence 的联合检索与重排 | JEM 的对象是可持续追加、带角色和时间状态的外部 agent memory；“联合检索”本身不是 novelty |
| [ParaSet / SetCE](https://arxiv.org/abs/2607.05712) | 已经直接学习 query-set compatibility，并证明集合级 scorer 可改善多跳检索与下游 QA | JEM 不能把 set-level learning 作为单独贡献；必须验证 append-only memory 中的状态一致性、增量写入和固定预算结构约束带来额外收益 |
| [ARM](https://arxiv.org/abs/2501.18539) | 已经联合建模 query-object relevance 与 object-object compatibility，并用混合整数规划选择相互兼容的数据对象 | JEM 的 ILP 和 pairwise compatibility 都不是首次提出；差异只能来自原始流式 memory、时间版本约束及不依赖查询时 LLM 的训练方式 |

因此，可辩护的 novelty 不是组件清单，而只能是这个统一问题及其经验证的解法：**append-only source memory + query-conditioned set structure + state-aware exact top-k read**。其中 raw source、set scoring、pairwise compatibility、时间图和 ILP 均已有先行工作，不能分别包装成创新。

### 2.1 当前 novelty 风险与 go/no-go 条件

当前版本是值得实现的研究假设，但还不是已经成立的论文贡献。尤其是，如果 JEM 只在静态 2Wiki 上复现 ParaSet/SetCE 的集合级收益，或只是把 ARM 的对象选择从 table 换成 passage，就不足以支撑 ACL/ARR 主方法。

继续投入的最低条件是同时观察到以下结果：

1. 在相同候选池和相同 top-5 下，结构化选择相对已发表的 set-level retriever 提高官方 joint evidence recall；
2. 在含官方顺序、时间或冲突监督的数据上，state-aware 约束相对去掉状态的同模型提高正确版本 evidence selection；
3. 新 memory 的写入只需追加和更新索引，不重跑已有 memory 的生成或全局图抽取；
4. 上述 retrieval 提升跨三个固定 Evaluation Backbone 传递到官方 QA metric，并且不依赖 test-specific threshold。

如果现有六任务的官方 train/development split 不能为第 2 项提供合法监督，就不训练、也不声称 state-aware 模块；必须把 JEM 缩小为检索研究原型，而不是用答案匹配或人工规则补标签。

## 3. 问题定义

给定不断增长的外部 memory：

\[
\mathcal{M}=\{m_i\}_{i=1}^{N},\qquad
m_i=(x_i,t_i,r_i,g_i),
\]

其中 `x_i` 是官方数据转换得到的原始 turn、chunk 或 passage，`t_i` 是数据原生时间或顺序，`r_i` 是原生 speaker role，`g_i` 是 session/document group。不存在的字段保持为空，不由模型猜测。

对于 query `q`，memory mechanism 必须返回不超过官方预算 `k=5` 的原始 memory 集合 `S`，以及只描述这些 memory 之间读取关系的结构 `T`：

\[
(S^*,T^*)=\arg\max_{|S|\le k} p_\theta(S,T\mid q,\mathcal{M}).
\]

Evaluation Backbone 只消费 `S*` 中的原始文本及其原生顺序，不接收 gold answer、训练标签或由 JEM 生成的最终答案。

## 4. 算法

### 4.1 写入：Append-only Source Memory

每个新的 `m_i` 执行三件事：

1. 原文按 benchmark 官方转换单元原样保存；
2. 建立 BM25 posting；
3. 用固定 embedding backbone 计算一次向量并加入 ANN index。

旧 memory 不被重写、摘要、合并或删除。显式冲突和演化状态留到 read operation 处理。核心版本的写入没有 Generator Backbone 调用，因此不会出现 LightMem 的压缩遗漏、Mem0 的抽取 JSON 失败和逐条 LLM update 延迟。

JEM 不自行重新切 chunk。不同 benchmark 的单元仍来自各自官方 adapter；相同数据在所有方法间使用相同转换。

### 4.2 候选生成：Lexical-Semantic Recall Union

BM25 和 dense retriever 分别产生候选排名，取两者前 `M` 的并集 `C(q)`。这里的并集只负责提高候选 recall；它不是把两个不可比较的原始分数人工相加。

`M` 是一个全局检索预算，只能在官方 training/development data 上选一次，然后对所有 test task 锁定。候选生成单独报告 official evidence recall，确保后续 set model 的改善不是偷偷增加候选范围得到的。

### 4.3 查询条件下的证据结构模型

模型产生两类 learned potential：

- `uθ(q,mi)`：query 与单条原始 memory 的相关性；
- `vθ(q,mi,mj,Δtij,ri,rj,gi,gj)`：两条 memory 对当前 query 是否构成有用的有向证据连接。

时间差、角色和 group 只在数据原生字段存在时输入；模型不把“最新一定正确”或“同实体一定相关”写成规则。关系 potential 由训练数据学习，因此同一 read operation 可以表达跨 passage 连接、同一事件的补充信息，以及演化事实之间的有效性。

结构得分为：

\[
s_\theta(q,S,T)=
\sum_{i\in S}u_\theta(q,m_i)
+\sum_{(i,j)\in T}v_\theta(q,m_i,m_j,\Delta t_{ij},r_i,r_j,g_i,g_j).
\]

这里没有手工 relevance/recency/novelty 权重。所有 potential 由同一个训练目标学习。

### 4.4 精确的固定预算推断

将 query 看作 root，在候选 `C(q)` 上选择至多五个节点和连接它们的有向树。二元变量 `z_i` 表示是否选择 memory，`y_ij` 表示是否使用有向连接。约束包括：

- `Σ z_i ≤ 5`；
- 只有两个端点被选择时才能使用边；
- 每个被选择的非 root 节点恰有一个入边；
- 单商品 flow 约束保证所选证据与 query root 连通并消除环。

在候选集上用整数线性规划求精确 MAP，而不是使用临时 threshold、贪心扩图或测试集特例。候选预算 `M` 控制 `O(M²)` pair scoring；ILP 的实际延迟必须在开发集和全量测试中报告，若不满足效率目标则该方法假设被否证，不能通过减少困难样本掩盖。

### 4.5 Evidence Bundle 输出

返回给 Evaluation Backbone 的仍是 `x_i` 原文。顺序由所选树的 root-to-leaf 关系决定；同一路径内若存在原生 timestamp/serial，则保留该顺序。结构只提供如下最小连接信息：

```text
[Evidence 1]
原始文本

[Evidence 2; follows Evidence 1]
原始文本
```

JEM 不总结这五条证据、不补写缺失事实、不生成答案。这样可以把“memory 是否找齐并组织证据”与“Evaluation Backbone 是否能回答”分开评测。

## 5. 训练方法

### 5.1 监督来源

只使用 benchmark 官方 training/development split 中明确提供的 supporting passage、supporting fact 或 evidence annotation。没有官方 evidence label 的任务只用于 zero-shot transfer evaluation，不用 test answer 产生伪标签。

如果某个数据集没有公开 train/dev evidence，JEM 不为它发明 evidence mapping。时间或冲突能力也只在官方数据明确给出 timestamp/serial 和正确 evidence 时训练、评测。

### 5.2 Retriever 训练

Dense retriever 用每个官方 gold evidence 作为正例、同一候选池中的非 gold source 作为负例，使用标准 contrastive likelihood。BM25 保持非参数化。训练只改变 dense encoder，不改变原文和 evidence budget。

### 5.3 Set-structure 训练

官方 gold evidence set 为 `G`。当数据没有标注 evidence 之间的树结构时，树是 latent variable，对所有连接 `G` 的合法树求和：

\[
\mathcal{L}_{set}=-\log
\frac{\sum_{T\in\mathcal{T}(G)}\exp s_\theta(q,G,T)}
{\sum_{S,T}\exp s_\theta(q,S,T)}.
\]

分母与训练时的候选池和 `k=5` 约束一致。若精确配分函数代价过高，应采用已有、可引用的 structured-learning approximation，并在方法中固定；不能根据测试表现临时切换推断方式。

整个训练过程不调用 Evaluation Backbone，也不以最终答案 F1 作为 reward，因此不会把某个 evaluator 的推理偏好学进 memory mechanism。

## 6. 为什么它可能同时改善效果和效率

| 基线缺点 | JEM 对应机制 | 可直接验证的结果 |
| --- | --- | --- |
| BM25/Dense 独立排序漏掉证据链 | set-level directed structure | official joint Recall@5、QA metric |
| LightMem/Mem0 抽取或压缩丢原文 | append-only raw payload | gold source retention 恒为 100%，前提是 adapter 正确 |
| Mem0 每条输入都做 LLM update | O(1) append + embedding/index update | 构建时间、Generator calls/tokens |
| HippoRAG 全语料 OpenIE 和 query-time LLM 昂贵 | 只在 `M` 个 query candidates 上判别式 pair scoring | 构建成本、单题 retrieval latency |
| 演化事实由相似度或 reader 猜测 | learned state-aware pair potential | 有官方标注时的正确版本 evidence selection |

“原文保留 100%”只指 JEM 不删除 adapter 交给它的 source units，不等于 retrieval 一定召回，也不替代官方 metric。

## 7. 必须预注册的实验与消融

所有实验继续使用当前固定 Generator/Evaluation Backbone、seed 42、top-5、官方 split 和官方确定性 metric。JEM 核心版本没有 Generator，因此 Generator Backbone 记为不适用，而不是补一个生成步骤保持形式一致。

### 7.1 主对比

- 五个现有 baseline：BM25、Dense、LightMem、HippoRAG 2、Mem0；
- 同一六任务套件；
- 三个 Evaluation Backbone 分别完整报告，不取跨 evaluator 平均；
- 只在有官方 gold passage/evidence 的任务报告 Recall@5、Precision@5；
- 所有任务报告官方 QA metric、构建时间、检索时间、Generator 调用/tokens 和 index 大小。

### 7.2 最小消融链

1. Dense only，独立 top-5；
2. BM25+dense candidate union，但仍按单条 relevance 选 top-5；
3. 相同 candidate union，加 set-structure model；
4. 完整 JEM，但移除原生 time/order/role features；
5. 完整 JEM。

第 2 与第 3 项必须使用完全相同的候选池，隔离联合结构模型的贡献。第 4 与第 5 项只用于确实含官方时间/顺序字段的任务。不能把更大 `M`、更多最终证据、更强 embedding 或更强 evaluator 混入某项消融。

### 7.3 否证标准

下列任一结果都应缩小或否定论文主张：

- candidate union 已包含 gold set，但 structured selector 的 joint Recall@5 不优于 independent reranker；
- retrieval 改善只出现在一个 Evaluation Backbone，或没有传递到官方 QA metric；
- 时间特征不能提高官方标注的有效版本选择；
- query-time pair scoring/ILP 的成本接近或超过 HippoRAG 的 query-time LLM filtering；
- 提升必须依赖 test-specific threshold、答案泄漏或额外 evidence budget。

## 8. 建议的论文主张

如果上述实验成立，可以主张：

> 现有 agent memory 在可检索性、原文忠实性、多跳证据完整性和更新成本之间存在可测量冲突。JEM 将写入简化为只追加原始 memory，并将读取建模为带时间状态的固定预算结构化 evidence-set prediction。它在不使用写入时或查询时 Generator LLM 的情况下，提高官方 gold evidence coverage，并将提升传递到多个固定 Evaluation Backbone 的确定性 QA。

不应主张：

- 首次保留 raw context；
- 首次联合检索多条 evidence；
- 首次使用 temporal memory；
- 首次把 lexical 与 dense retrieval 结合；
- 解决了 gold evidence 已完整但 Evaluation Backbone 不会推理的问题。

## 9. 实现顺序

1. 在当前 frozen retrieval/QA scaffold 中加入 JEM baseline 名称和 append-only raw index；
2. 先实现相同候选池上的 independent reranker 与 exact ILP selector，验证 selector 接口而不训练 Generator；
3. 只用官方 train/dev supporting evidence 训练 unary/pair potentials；
4. 先在 2Wiki 验证 joint Recall@5，再运行全部六任务端到端 QA；
5. 最后加入原生时间/角色特征，并只在有相应官方标注的任务上做状态消融。

如果第 3–4 步不能稳定优于相同候选池的独立 reranker，就停止增加模块；不通过更复杂图、额外 LLM 或测试集规则挽救故事。
