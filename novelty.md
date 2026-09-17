# 算法与研究主张

更新：2026-09-16。实验、消融、成本和运行状态只在 [results.md](results.md) 维护。

## 一句话

**从带来源的事实中选择支持，再用同一份支持生成传播图与事实识别索引。** 暂名 Source-Supported Graph Indexing，标题方向 Consolidate Before You Propagate。

问题不是怎样抽取更多事实，而是同一批事实如何组织成可用的检索结构。事实选择只作用于图时，候选索引仍可能让被排除的支持参与识别；只筛候选则没有改变传播连接。我们联合检验这两个索引，不增加 agentic 查询规划。

“记忆编译”是可用的写作框架：原文是历史档案，带来源的事实是中间表示，支持选择是整理步骤，图与候选是派生索引。它不是新的 PL 正确性理论，当前没有语义保持或在线增量维护保证。

## 1. 信息抽取：继承部分

沿用 HippoRAG OpenIE：Qwen3-30B-A3B-Instruct-2507 先做段落 NER，再输入原文和实体列表抽取 `(subject, relation, object)`。保存每条三元组的实际来源；原实体、事实、段落向量由 Qwen3-Embedding-0.6B 生成。本轮复用这些缓存，不重做 IE 或微调 generator。

代码：[openie_openai.py](baseline_algorithms/HippoRAG/src/hipporag/information_extraction/openie_openai.py)。构图只使用来源材料，不使用问题、答案或 gold evidence。loader 会加载原评测对象，不代表问答字段参与构图决策。

## 2. Refinement：选择事实支持

[relation_schema.py](optimization/graph_construction/relation_schema.py) 取每关系前两个不同三元组作为例子，用 LLM 生成 canonical 标签。[canonicalize_schema.py](optimization/graph_construction/canonicalize_schema.py) 按关系 embedding 分组后联合归一；同 canonical 再合并，取最小输入 ID 的代表。role/cardinality 仍在旧生成输出中，但当前算法不使用它们。

[retained_statements](optimization/graph_construction/source_consolidation.py#L7) 的关键代码：

```python
normalized = tuple(normalize(list(triple)))
subject, relation, _ = normalized
if schema is not None:
    record = schema[triple[1]]
    relation = normalize(record["canonical"])
slot = (subject, relation)
latest[slot] = max(latest.get(slot, -1), positions[key])
```

原函数还保存各条记录，随后只保留 `positions[key] == latest[slot]` 的支持。同一末来源的多值全部保留，所有关系使用同一政策。canonical 只用于分槽，不重写原三元组/向量身份。上面省略记录收集与合法性检查，完整实现见链接。

这是 loader 顺序政策，不等于事件时间或语义冲突检测。本轮整链 raw-relation 消融调用已有 `schema=None` 路径，图、候选和附录同步不用 canonical，不是只把图侧开关关掉。该候选已出现目标任务回退，未替换默认算法；临时构建入口已归档退役。

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

## 贡献与相关工作

可主张的是支持选择、事实来源投影和候选筛选的具体联合设计及其分离消融，而非组件首创。

| 近邻工作 | 已有机制 | 当前区别 |
|---|---|---|
| [HippoRAG 2](https://arxiv.org/html/2502.14802) | 已有来源节点、recognition、PPR | 我们改变支持如何派生传播权重与候选范围，不以“加 provenance”主张首创 |
| [CatRAG](https://aclanthology.org/2026.findings-acl.290.pdf) | query-adaptive 导航及边权 | 我们在查询前构建共享索引，不增加查询时路径规划 |
| [HyperGraphRAG](https://arxiv.org/html/2503.21322) | 新 n-ary 抽取、实体/超边检索 | 我们复用 OpenIE，将事实来源关联投影回原节点 |
| [A-MEM](https://arxiv.org/html/2502.12110) | note 构造、链接、历史 note 演化 | 我们不改写记忆内容；它的写入机制仍是直接相关工作 |
| [Zep](https://arxiv.org/html/2501.13956v1) | 时间有效性及矛盾识别/失效 | 我们的来源顺序选择没有同等语义保证 |

“减少历史干扰”是解释假设，不是全部 QA 分差的已证实原因。系统视角是原始记录与派生索引分离；图论视角是事实关联与已有投影。未实现 DBSP 增量维护、MemorySSA、语义保持编译或 hyperbolic embedding，不将这些名称写成贡献。

新 reader 的完整迁移结果进一步限定主张：对 BM25、Dense、HippoRAG 2，Gemma-3-4B 为 6/6，Llama-3.1-8B 为 4/6；后者两个事实整合任务并未超过 baseline 最佳值。不能写成与 reader 无关的普遍改善。最终索引条件下，删窗口已使 LoCoMo 下降 5.66/3.13 点，删事实附录使 FCSH 下降 24/16 点（9B/4B）；端到端收益不能全归给传播图，结构与读出的贡献须分别报告。

## 仍保留的人工选择

当前清单仍有七组、23 项明确的规则/设置，并非 23 个独立算法：关系归一 5、支持选择 4、来源直连/去 synonym 2、候选筛选 1、RRF 5、窗口 3、附录 3。另有底座参数与投影选择，不包含在该计数中。用户已确认保留 4B/9B 六项全胜约束；本轮四项删除的双 reader 全六任务评测均已完成，均不满足该约束，未采用。移除临时实验入口不算减少主算法规则；模块级消融也不能证明内部每个常数必要，或证明当前算法已最小化。

## 实验段落

**Dataset selection and coverage.** We evaluate six task settings from three benchmark suites, covering updated-fact access, single- and multi-hop evidence retrieval, and long-term conversational recall. Four selected MemoryAgentBench configurations contribute 100 questions each: SH-Doc QA, MH-Doc QA, FactConsolidation-SH, and FactConsolidation-MH. LoCoMo contributes all 1,986 questions from locomo10 across five question categories. For 2WikiMultiHopQA, we use HippoRAG 2's released 1,000-question subset and associated corpus. These 3,386 questions connect focused consolidation tests with document and conversational QA. We evaluate all questions in the selected configurations using task-specific scoring. This provides complementary coverage of offline indexing, not exhaustive coverage of agent memory; procedural learning, tool-use policies, and online adaptation are outside scope.

**Development protocol.** The evaluation sets were also used for method development and configuration selection; results are therefore in-distribution benchmark results rather than held-out generalization estimates. Index construction uses source material, not evaluation questions, answers, or evidence labels. Each index is frozen before retrieval.

依据：[MemoryAgentBench](https://arxiv.org/html/2507.05257v4)、[LoCoMo](https://snap-research.github.io/locomo/)、[2Wiki](https://aclanthology.org/2020.coling-main.580/)。未覆盖完整 MAB 在线交互/能力组，题目共享记忆库，不把题数当独立样本数。

## 代码与复现

- 构建：`python -m optimization.build_graph --task "SH-Doc QA" --output-root <fresh-directory>`。CPU 用 batch 分区；默认构建当前获胜配置。
- 原索引四格：加 `--no-retained-fact-index`，分别使用两种 `--construction`；旧图同保留索引使用 `--construction projected`。
- 本轮 raw-relation、删直连及读出删除的临时实现/作业定义已归档，不作为常驻 CLI 开关；结果与归档位置见 results.md。
- 检索/验证/QA：`python -m optimization.run_graph --phase retrieve|verify|evaluate --task ... --output-root ...`。新 reader 用 `--evaluation-backbone` 指定。
- 识别缓存未命中时需同源 generator 地址及 `--allow-generator-calls`；guard 阻断缓存缺失或 provider 异常后的静默 fallback，不改变上游事实解析失败的处理。不要把缓存验证当成任意新问题无需 LLM。
- 前置产物：原 HippoRAG 索引、OpenIE、schema、冻结读出；`--source-root`、`--schema-root`、`--readout-root` 可指定。本入口不负责首次抽取；无 schema 的 raw 消融仅在归档实现中复现。
- 环境：Oscar 的 `agent-memory-envs/hipporag` 用于构图/检索，`runner` 用于 QA，`vllm_cu129` 用于原 generator；CUDA 12.9。沿用现有环境，不重新安装或下载既有模型。
- 环境定义见 `requirements/`；可移植安装入口为 `requirements/create_environments.sh`，仅新环境需要。各 baseline 隔离安装；兼容补丁位于 `baseline_patches/`，由 `requirements/apply_baseline_patches.sh` 应用。补丁细节与旧运行适配保存在文档归档，不冒充原版算法改进。

仓库自有研究 MD 仅保留本文件与 results.md。旧全文、负结果、代码与缓存的位置见 results.md；第三方 submodule 文档不删。
