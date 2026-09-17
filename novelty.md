# 算法与研究主张

更新：2026-09-17。实验、消融、成本和运行状态只在 [results.md](results.md) 维护。

## 最新故事与状态

**在不重新抽取知识、不更换在线推理算法的条件下，把“事实及其来源”作为构图单元，将实体间的关联与指向原始证据的连接共同投影为检索图。** 工作名称仍为 Source-Supported Graph Indexing；这是当前最值得验证的构图主张，不是已经证明成立的机制结论。

问题不是图中有没有 passage 节点，而是事实与证据之间的关联如何决定传播权重。只保存实体对关系、再另加段落包含边，与按每条事实的主体、客体和实际来源共同构图，是不同的建模选择。我们的研究对象是这项离线构图选择能否改善证据访问，并能否迁移到不同 reader 和已有 agentic memory algorithm。

当前默认仍为 raw-relation 版本，9B 5/6、4B 6/6；旧关系归一完整版本为双 6/6。这里均指严格超过九项完整本地 baseline 的逐任务最佳，而非全领域 SOTA。性能优先：保留强版本及缓存作参照，只删除证据表明不重要的部分，不为减少模块数直接采用更弱配置。

**本轮消融改变了故事：共同筛选图和候选并未稳定优于只筛一侧。** 因此旧标题方向 Consolidate Before You Propagate 暂不作为核心贡献，来源顺序筛选也不再被预设为不可缺少。固定 all-support 读出、完整候选和 RRF 后，原图为 9B 3/6、4B 2/6，新图为双 5/6；这已支持构图组合有独立于读出规则的贡献，但不是投影算子单独有效或全任务都提高的证明。

最贴合实现的是图表示与证据索引视角。“记忆编译”可以描述原始记录到派生索引的离线流程，但没有新增 PL 理论、语义保持证明或在线增量维护保证。不以换成 system/PL 名称代替机制证据。

## 问题与贡献逻辑

1. **为什么重要：** 抽取与 embedding 已经完成后，仍可能因索引组织不当而取不到回答所需的原始证据。若仅改变构图即可改善多个 reader，便可复用既有昂贵抽取，并保留现有查询接口。
2. **为什么难：** 构图、候选识别、种子归一化和读出共同影响 QA。删掉一个组件后下降，只能说明它在该组合中有用，不能单独证明新图有效；有来源节点也不保证证据路径有用。
3. **与已有工作的具体区别：** 不是首次使用来源、超图或 PPR，而是复用已有二元 OpenIE，把事实的实体与实际来源放在同一关联单元中，再用已有投影返回原实体/段落节点，检验这项替换是否有收益。
4. **证据如何闭合：** 构图贡献由固定读出的原图/新图对照回答；支持选择由图侧/候选侧四格回答；来源路径互补性由单独和联合删除回答；reader 与 agentic 接入验证可迁移性。未完成的项目不能写成结果。

完整流程：已有 OpenIE/embedding → 可选来源支持筛选 → 事实-实体-来源关联与无自环投影 → 原 recognition/PPR 与 BM25/RRF → 原文、事实附录和局部来源窗口 → 原 QA。当前实现中的“可选”表示正在消融的设计，不表示已从默认路径删除。

## 1. 信息抽取：继承部分

沿用 HippoRAG OpenIE：Qwen3-30B-A3B-Instruct-2507 先做段落 NER，再输入原文和实体列表抽取 `(subject, relation, object)`。保存每条三元组的实际来源；原实体、事实、段落向量由 Qwen3-Embedding-0.6B 生成。本轮复用这些缓存，不重做 IE 或微调 generator。

代码：[openie_openai.py](baseline_algorithms/HippoRAG/src/hipporag/information_extraction/openie_openai.py)。构图只使用来源材料，不使用问题、答案或 gold evidence。loader 会加载原评测对象，不代表问答字段参与构图决策。

## 2. Refinement：选择事实支持

当前主方法直接使用底座 OpenIE 的规范化关系文本，不生成额外关系 schema，不做关系聚类或语义归一。
2026-09-17 已采用完成双 reader 全六任务评测的 raw-relation 版本：9B 5/6、4B 6/6。
两个关系归一模块及 schema 作业入口已删除；旧完整版本及其产物保留为消融参考。

[retained_statements](optimization/graph_construction/source_consolidation.py#L7) 的关键代码：

```python
normalized = tuple(normalize(list(triple)))
subject, relation, _ = normalized
slot = (subject, relation)
latest[slot] = max(latest.get(slot, -1), positions[key])
```

原函数还保存各条记录，随后只保留 `positions[key] == latest[slot]` 的支持。同一末来源的多值全部保留，所有关系使用同一政策。关系别名不会额外合并，三元组/向量身份沿用底座。上面省略记录收集与合法性检查，完整实现见链接。

这是 loader 顺序政策，不等于事件时间或语义冲突检测。关系归一同时从图、候选和附录删除，
不是只把图侧开关关掉；删去它有性能代价，9B FCSH 从 68 降至 64，因此不再称双 6/6。
简化实现已通过 27 项测试、15 组图/候选/读出的独立精确重建，以及 3386 题普通接口检索匹配。raw 四格、模块删除和种子归一化对照已完成；预算控制不再推进。
四格不支持共同筛选稳定优于单侧：只筛图为 9B 6/6、4B 5/6，共同筛为 9B 5/6、4B 6/6。
整链取消支持选择仍双 5/6，因此共享支持选择不能继续写作已证明必要的核心机制。更薄候选尚在联合消融，冻结后再确定最终主张。

## 3. 构图：事实与来源共同参与

[statement_incidence_graph](optimization/graph_construction/statement_incidence.py#L6) 将每个不同规范化三元组作为内部事实单元，成员为主体、客体及保留的实际来源。成员关系是二值的，不按重复次数累加：

```python
subject, _, obj = triple
members = {positions[entity_keys[subject]], positions[entity_keys[obj]]}
members.update(positions[source] for source in sources[triple])
edges.extend((len(names) + index, member) for member in sorted(members))
```

[project_statement_graph](optimization/graph_construction/statement_incidence.py#L52) 使用 Kumar 等的已有无自环约简。`incidence` 的行是原实体/段落节点、列是内部事实单元：

```python
degrees = np.asarray(incidence.sum(axis=0), dtype=np.float64).ravel()
if np.any(degrees == 1):
    raise ValueError("A loop-free fact transition requires at least two members")
degrees = degrees - 1
inverse = np.divide(1.0, degrees, out=np.zeros_like(degrees), where=degrees > 0)
facts = incidence @ sparse.diags(inverse) @ incidence.T
facts.setdiag(0)
facts.eliminate_zeros()
projected = adjacency[:original_vertices, :original_vertices] + facts
```

每个有 m 个成员的事实对不同成员贡献 `1/(m-1)`；额外保留有支持的原 passage-entity 直连，去掉 synonym 贡献。最后运行普通实体/段落加权图，不保留新的在线事实节点或事实 embedding。

投影及其度数性质属于 [Kumar 等 2020，式 3](https://link.springer.com/article/10.1007/s41109-020-00300-3)，不是我们的新定理。谓词区分事实身份，但 PPR 不读谓词标签/方向；投影不可保证反推出原事实，不是无损编码。当前活跃事实几乎都只有一个来源，不能用“多来源超边融合”解释主要收益。

### 理论解释与实验责任

该已有算子对应两步随机游走：从当前节点选择一个关联事实，再在该事实的其他成员中均匀选择下一节点。当前事实等权，大小为 m 的事实对每个成员贡献的总出边权为 `(m-1)/(m-1)=1`，避免未经归一化的 clique expansion 随成员数放大传播质量。实际代码另加的来源直连属于额外传播通道，不能从这个性质推出它有必要或可以删除。原论文已有此随机游走解释；我们贡献的候选是将具体事实来源表示接入这套算子，而非新游走理论。

这给出的是传播机制解释，不是 QA 正确性、冲突处理或最优索引证明。支持筛选决定哪些成员进入关联结构，其语义适用性仍需单独检验。全任务统一、由节点/事实度数计算权重，也不等于已经学习了 self-adaptive memory。

| 需要解释的主张 | 已有理论或代码性质 | 必须由实验回答的部分 |
|---|---|---|
| 为什么这样定投影权重 | 已有无自环、度数保持的超图约简；现有单元测试覆盖无自环、节点加权度及小图 PPR | 在我们的事实来源表示上是否改善实际 QA，不能由度数性质推出 |
| 为什么需要来源连接 | 保留从事实实体到原始证据的传播通道 | 联合删除双 1/6、单删双 5/6 支持通道可替代，而非两条通道均必要 |
| 为什么筛图与筛候选不等价 | 前者改变传播边，后者还能改变识别、种子及 min-max 范围 | 四格和固定识别的归一化对照；当前结果不支持共同筛选稳定最优 |
| 为什么固定通用参数 | RRF 在当前 top-5 条件下，已检查的一组常数排序等价 | 来源窗口、候选窗口尚需敏感性验证；格式选择不能用投影理论解释 |

## 4. 候选索引：与图共用支持

[build_graph.py](optimization/build_graph.py) 用保留支持选择原 fact 行，保留原相对顺序。在线加载由 [restrict_fact_index](optimization/retriever/hipporag.py#L10) 完成：

```python
positions = {key: i for i, key in enumerate(hippo.fact_node_keys)}
vectors = hippo.fact_embeddings[[positions[key] for key in keys]]
hippo.fact_node_keys = list(keys)
hippo.fact_embeddings = vectors
```

没有重算向量，也不修改实体/事实到原来源的映射。候选集合改变会影响 HippoRAG 的 min-max 分数与查询种子，不能说识别输入完全不变。原文库、BM25、来源窗口仍能访问历史，所以不是所有访问路径的全局一致性或 active-only memory。

## 5. 检索、读出与 QA

原 recognition LLM 从向量召回的事实中筛选相关项，原实体种子及 dense passage 种子进入 PPR。图检索与同语料 BM25 各前五，通过固定等权 RRF（60）合并为五中心；没有新的 agentic 查询循环。

对应 [hybrid_graph.py](optimization/retriever/hybrid_graph.py#L11)：

```python
scores[item.text] = scores.get(item.text, 0.0) + 1.0 / (rank_constant + rank)
order = sorted(candidates, key=lambda text: -scores[text])[:top_k]
```

[compiled_sources.py](optimization/graph_construction/compiled_sources.py) 返回中心原文、来源位置/时间及保留事实 JSON；[source_window.py](optimization/graph_construction/source_window.py) 在连续相同非空 timestamp 内补前后三条原文，不跨中心去重。原 QA prompt、生成设置及评分保留。五中心不是五段文本或固定 token 预算。

## Novelty 来源与相关工作

潜在创新落在**具体的事实来源构图方式及其可替换性验证**，不是现成算子的首创。支持选择与候选筛选属于已检验的设计选项，不能在四格结果不支持时继续声称二者协同是贡献来源。

| 近邻工作 | 已有机制 | 当前区别 |
|---|---|---|
| [HippoRAG 2](https://arxiv.org/html/2502.14802) | 已有 phrase/passage 节点、事实边、包含边、synonym、recognition 与 PPR | 我们按每条事实的实体及实际来源建立关联，再投影回原节点；继承在线查询，不以“加 provenance”主张首创 |
| [CatRAG](https://aclanthology.org/2026.findings-acl.290.pdf) | query-adaptive 导航及边权 | 我们在查询前构建共享索引，不增加查询时路径规划 |
| [HyperGraphRAG](https://arxiv.org/html/2503.21322) | 抽取自然语言 n-ary 事实，以实体-超边二部图存储，分别检索实体与超边并结合文本块 | 我们不重新抽取 n-ary 事实或训练超边向量，而以已有三元组及实际来源构造关联并投影，在线仍用原 recognition/PPR；不是“首次把超图用于 RAG” |
| [A-MEM](https://arxiv.org/html/2502.12110) | note 构造、链接、历史 note 演化 | 我们不改写记忆内容；它的写入机制仍是直接相关工作 |
| [Zep](https://arxiv.org/html/2501.13956v1) | 时间有效性及矛盾识别/失效 | 我们的来源顺序选择没有同等语义保证 |

“减少历史干扰”是解释假设，不是全部 QA 分差的已证实原因。系统视角是原始记录与派生索引分离；图论视角是事实关联与已有投影。未实现 DBSP 增量维护、MemorySSA、语义保持编译或 hyperbolic embedding，不将这些名称写成贡献。

投影权重由事实成员数计算，属于已有度数归一化，不是逐数据集调参，也不是学到了自适应策略。固定 RRF、窗口和附录属于组合设计，不能因全任务统一就宣称它们本身具有 novelty。来源顺序筛选仍是经验规则，不能写成语义冲突检测。

**新增的归因证据：** 固定 all-support 读出、候选和 RRF，原图与新图的 2Wiki 为 49.52→57.17（9B）、47.42→55.81（4B）；FCMH 为 2→9、4→6。4B 的 MH、FCSH 和 LoCoMo 并非同步提高，实际输入长度也未匹配。构图替换包含事实投影、来源直连权重及 synonym 处理，不能把差值全部归给投影公式。

同时删除事实来源成员和额外来源直连使两个 reader 均降至 1/6，而单独删除任一路径仍双 5/6。这支持来源连通路径具有替代性，反对“两条路径都不可缺少”的叙事。完整候选、保留图支持这一更强候选的同类消融仍在继续；不同 reader 和 agentic 接入的迁移验证尚未完成。

旧完整版本的新 reader 迁移结果进一步限定主张：对 BM25、Dense、HippoRAG 2，Gemma-3-4B 为 6/6，Llama-3.1-8B 为 4/6；这些不是简化主方法的迁移成绩。该完整版本中，删窗口使 LoCoMo 下降 5.66/3.13 点，删事实附录使 FCSH 下降 24/16 点（9B/4B）；端到端收益不能全归给传播图，结构与读出的贡献须分别报告。简化版冻结后另做第三 reader 与九 baseline 的完整比较。

## 仍保留的人工选择

撤回此前“六组、18 项”的精确计数。该数字来自旧分组清单减去关系归一组，并非完整的逐项代码审计；其中混合了数值、模块开关、由定义推出的行为和表示格式，且遗漏了 RRF 的零分过滤与同分排序等实现规则。不能据此宣称只有 18 个 heuristic 或需要扫 18 个参数。

按当前默认路径，实际设置如下。此表用于定位实验维度，不再通过主观拆分宣称一个总数。

| 部分 | 当前代码中的选择 | 应怎样检验 |
|---|---|---|
| 支持选择 | 规范化 `(subject, relation)` 槽；按 loader 来源顺序选最后位置；保留同位置全部值；所有关系使用同一政策 | 结构消融；不是可连续扫描的数值。保留多值是规则语义，不独立发明新冲突策略 |
| 事实关联与传播 | 事实成员含主体、客体及实际来源；二值成员关系；已有无自环投影；额外二值 passage-entity 直连；去掉 synonym 贡献 | 来源成员、直连、synonym 已分别/联合测试；标准投影不新增混合系数来调分 |
| 候选索引 | 仅保留有选中支持的原 fact 行，保持原相对顺序及向量 | 完整原索引对照，正在验证删除此模块后的剩余机制 |
| 融合 | 图与 BM25 两路等权 RRF；常数 60；每路前 5；最终前 5；跳过非正 BM25 得分；同分时图路顺序优先 | 数值敏感性针对常数及每路窗口；最终返回数属于当前固定评测设置；其他行为不能冒充独立数值参数 |
| 来源窗口 | 前后最多各 3 条；非空且相同 timestamp；连续扫描遇到不同 timestamp 即停止 | 数值敏感性针对半径；去整个窗口已有消融；边界定义是结构选择 |
| 读出 | 保留中心原文；附来源位置及可用时间；按源顺序附选中三元组 JSON，并在单来源内去重；不同中心的窗口不做联合去重 | 附录删除已有消融；格式、元数据和去重行为尚无逐项必要性证据，不称已全部消融 |

对应实现：`source_consolidation.py`、`statement_incidence.py`、`compiled_sources.py`、`source_window.py`（均在 `optimization/graph_construction/`），以及 `optimization/retriever/hipporag.py`、`optimization/retriever/hybrid_graph.py`。
本方法显式数值设置包括 RRF 常数、每路候选窗口和来源窗口半径。RRF 常数已完成下述排序等价性检查；候选窗口与来源窗口半径尚未系统扫描。底座 recognition/PPR、embedding、chunking 和 QA 解码设置固定，不将改变它们的结果混作构图消融。

RRF 采用 [Cormack 等 2009](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) 的默认常数 60。按其表 1 的扫描值检查：当前两路等权、每路前五时，`10,20,30,40,50,60,70,80,90,100,500` 的所有单路/双路排名模式均有相同的相对顺序与平局关系；实际融合函数在全部 1546 种两路重合映射下也返回相同的有序文本与 metadata。QA 只读取文本，不读取 RRF 分数，因此这些常数下无需重复 QA。`k=0` 不等价；增加候选窗口、改变路数或权重后结论不自动成立。这是现有算子的范围内性质，不是新的融合算法，也不是从 QA 分数挑出的最优常数。
标准算子/继承参数、跨数据集统一设置、经验性策略需要区分；当前核心路径没有按任务名选择参数的分支。
通用参数可以是 principled 方法的组成部分，不要求发明新数学或把每个常数都删掉。
全局固定也不自动证明来源顺序等经验策略正确，或证明方法具备 self-adaptive 能力。
新目标仍力争两个 reader 各 6/6；各至少 5/6 是可接受下限，不是停止优化或继续强删的理由。raw 版本已达到此前采用门槛，已实际移除归一代码；保留旧双 6/6 参照，不按任务拼接版本。
其余模块不由“删了降分”推断内部每个常数必要，也不宣称已经最小化；进行中的消融见 results.md。

## 其他 Reader 与 Agentic 验证

- 本地 Hugging Face 模型为 Llama-3.1-8B-Instruct、Gemma-3-4B-it；与 Qwen 合计三个模型家族、四个 reader 配置。不是 Google API 或 Gemma-4。
- 先完成当前消融取舍并冻结统一方法，再做九 baseline 加最终方法的六任务全量比较。已有 raw 预检只代表输入对齐，撤销的 QA 队列不能计为已跑；旧 canonical 的三个 baseline 迁移成绩不替代此次主实验。
- 扩展 reader 时保留各任务原 prompt、指标和解码协议，记录实际模型版本、输入长度和失败；不根据新 reader 成绩反向选择构图参数。
- Agentic 实验是既有 memory algorithm 原版与接入我们图的配对比较，保留其写入/演化机制。当前仅检查 A-MEM 官方实现，尚无接入或正式结果；六任务不兼容之处须明确报告，不临时改造数据来制造覆盖。

## 实验段落

**Dataset selection and coverage.** We evaluate six task settings from three benchmark suites, covering updated-fact access, single- and multi-hop evidence retrieval, and long-term conversational recall. Four selected MemoryAgentBench configurations contribute 100 questions each: SH-Doc QA, MH-Doc QA, FactConsolidation-SH, and FactConsolidation-MH. LoCoMo contributes all 1,986 questions from locomo10 across five question categories. For 2WikiMultiHopQA, we use HippoRAG 2's released 1,000-question subset and associated corpus. These 3,386 questions connect focused consolidation tests with document and conversational QA. We evaluate all questions in the selected configurations using task-specific scoring. This provides complementary coverage of offline indexing, not exhaustive coverage of agent memory; procedural learning, tool-use policies, and online adaptation are outside scope.

**Development protocol.** The evaluation sets were also used for method development and configuration selection; results are therefore in-distribution benchmark results rather than held-out generalization estimates. Index construction uses source material, not evaluation questions, answers, or evidence labels. Each index is frozen before retrieval.

依据：[MemoryAgentBench](https://arxiv.org/html/2507.05257v4)、[LoCoMo](https://snap-research.github.io/locomo/)、[2Wiki](https://aclanthology.org/2020.coling-main.580/)。未覆盖完整 MAB 在线交互/能力组，题目共享记忆库，不把题数当独立样本数。

## 代码与复现

- 构建：`python -m optimization.build_graph --task "SH-Doc QA" --output-root <fresh-directory>`。CPU 用 batch 分区；默认构建当前获胜配置。
- 原索引四格：加 `--no-retained-fact-index`，分别使用两种 `--construction`；旧图同保留索引使用 `--construction projected`。
- raw-relation 已是正式构建入口的默认路径，不再依赖临时消融脚本。旧 canonical、删直连及读出删除保留归档结果。
- 检索/验证/QA：`python -m optimization.run_graph --phase retrieve|verify|evaluate --task ... --output-root ...`。新 reader 用 `--evaluation-backbone` 指定。
- 识别缓存未命中时需同源 generator 地址及 `--allow-generator-calls`；guard 阻断缓存缺失或 provider 异常后的静默 fallback，不改变上游事实解析失败的处理。不要把缓存验证当成任意新问题无需 LLM。
- 前置产物：原 HippoRAG 索引及 OpenIE，`--source-root` 可指定。本入口不负责首次抽取，不再需要 schema 或冻结读出；可选 `--readout-root` 仅用于与既有产物核对，新读出由源材料直接构建。
- 环境：Oscar 的 `agent-memory-envs/hipporag` 用于构图/检索，`runner` 用于 QA，`vllm_cu129` 用于原 generator；CUDA 12.9。沿用现有环境，不重新安装或下载既有模型。
- 环境定义见 `requirements/`；可移植安装入口为 `requirements/create_environments.sh`，仅新环境需要。各 baseline 隔离安装；兼容补丁位于 `baseline_patches/`，由 `requirements/apply_baseline_patches.sh` 应用。补丁细节与旧运行适配保存在文档归档，不冒充原版算法改进。

仓库自有研究 MD 仅保留本文件与 results.md。旧全文、负结果、代码与缓存的位置见 results.md；第三方 submodule 文档不删。
