# 算法总结：以实际代码为准

更新：2026-09-22。实验见 [results.md](results.md)，论文叙事见 [story.md](story.md)。

## 1. 当前方法是什么

固定语料经过 OpenIE 抽取后，用事实及其原文出处重新构造检索图；查询时结合图检索和 BM25，最后提供原文、相邻对话和最多十条相关事实给回答模型。不训练 retriever，不按 reader 或任务切换配置。

完整流程：

```text
原文 + 原有 OpenIE/embedding 缓存
  -> 关系名称归一 + 按原文顺序选择事实支持
  -> 实体、事实和出处的关联结构
  -> 标准无自环投影 + 保留的实体到出处连接
  -> 事实候选索引 + 问题相关事实识别 + personalized PageRank
  -> 与 BM25 做 reciprocal rank fusion
  -> 原文 + 同时间戳的相邻记录 + 问题相关的最多十条事实
  -> 原生任务提示词 + reader
```

这里的“当前方法”是 canonical 关系版本的投影图，加十条相关事实的最终 QA 上下文。不是后来尝试但未采用的显式事实节点检索、图模式匹配或独立 reranker。

## 2. 构图与索引

### 信息抽取与关系名称

原始抽取由 Qwen3-30B-A3B-Instruct-2507 完成，embedding 为 Qwen3-Embedding-0.6B。现在构图复用缓存，不重新生成实体或事实。

[relation_schema.py](optimization/graph_construction/relation_schema.py) 从关系标签及例子生成 schema；[canonicalize_schema.py](optimization/graph_construction/canonicalize_schema.py) 用 embedding 分组后让 LLM 合并同义关系。保留这些代码是为了重建当前 canonical 缓存。schema 中的 cardinality/role 字段是历史输出，当前支持选择不使用它们。

### 哪些事实用于图和候选索引

[source_consolidation.py](optimization/graph_construction/source_consolidation.py) 的 `retained_statements`：

1. 将主体和关系名称归一，用二者共同确定一组记录。
2. 找到这组记录在数据加载顺序中最后出现的原文。
3. 保留该原文中对应的全部三元组，不进一步挑选某一个客体。
4. 用保留结果生成图权重和候选事实 ID；原始全文及原谓词文本仍保留。

这是按来源位置筛选，不是根据事件发生时间判断真假；同一最后原文内的冲突仍可能并存。该规则不能被描述为完整的历史记忆或冲突解决算法。

### 怎样形成检索图

[statement_incidence.py](optimization/graph_construction/statement_incidence.py)：

- 每个不同的规范化三元组对应一个内部事实节点。
- 将它与主体实体、客体实体及被选中的出处连接，成员去重，关联权重为 1。
- 当前完整配置去掉不属于原文成员连接的残余边，包括继承的相似实体边。
- 对事实关联应用 Kumar 等人的标准无自环投影，消去内部事实节点。
- 再保留经过支持筛选的实体到原文的二值连接。

设 H 是实体/原文到事实的二值关联矩阵，d 是每条事实的成员数，C 是保留的实体到原文连接。代码对应：

```text
F = offdiag(H diag(1 / (d - 1)) H^T)
A = C + F
```

没有支持的事实列贡献为零；一个成员的异常事实会报错。投影公式不是我们提出的新数学方法。

例如，事实 (Alice, works at, Lab X) 来自原文 p，三名成员为 Alice、Lab X、p。F 为三对连接各增加 1/2；若 C 含两条实体到 p 的成员连接，则最终权重分别为 1.5、1.5、0.5。多个不同事实的贡献相加。

最终在线图只保留原实体和原文节点。关系文本仍在事实索引中，但 PPR 的边不编码谓词方向；因此不能称为无损高阶推理。当前“最后出处”策略通常只为一个三元组保留一个出处，不能把跨出处的同事实连接宣传为主配置的核心。

[build_graph.py](optimization/build_graph.py) 负责组装上述操作，产出：

| 文件 | 用途 |
|---|---|
| `graph.pickle` | 当前构造的图，实际位置以 graph.json 为准 |
| `edge_weights.npy` | 与加载图边序一致的权重 |
| `retained_fact_keys.json` | 允许进入事实识别的候选 ID |
| `lexical_source_keys.json` | BM25 使用的完整原文顺序 |
| `graph.json` | 原始图、构造图、事实索引、上下文文件的实际路径 |
| `compiled_sources/<task>/<group>/contents.json` | 原文、保留事实、位置、时间戳和邻接记录 |

## 3. 查询与上下文

[retriever/hipporag.py](optimization/retriever/hipporag.py) 加载图及候选索引。问题编码、候选事实相似度、LLM fact recognition 和 PPR 来自现有 HippoRAG 实现；构图、支持选择、索引限制及上下文适配在本项目代码中实现。保留这个依赖事实，不把已有算子认领为新贡献。

[hybrid_graph.py](optimization/retriever/hybrid_graph.py) 使用 RRF 融合图与 BM25 的排序，常数 60。one-shot 每路取五段、最终选五个中心原文；BM25 非正分不参与融合。同分保持稳定顺序，不使用独立神经 reranker。

[source_window.py](optimization/graph_construction/source_window.py) 只为有时间戳的记录加入同一连续时间戳区间内前后各至多三条邻居，不跨时间戳边界；没有时间戳的原文不扩展。

[query_fact_context.py](optimization/retriever/query_fact_context.py) 的 `QueryFactContext.render`：

1. 保留中心原文、位置、时间戳和已经冻结的邻接原文。
2. 从这些出处的保留三元组中收集候选，按事实 ID 去重。
3. 用归一化 query/fact embedding 的内积选至多十条事实。
4. 以相似度从低到高的顺序将所选事实放在原文之后，最相关事实最后展示。
5. 无候选时只返回原文，不补造事实，也不改变来源检索。

这是查询相关的事实展示，不是额外来源检索。它复用了 KAPING 组件试验中的选择和排序设置，不是训练得到的上下文效用预测器。

## 4. 实际入口与默认值

**清理没有修改算法默认值。直接运行旧入口，不会自动获得最新十事实结果。**

| 入口 | 实际行为 |
|---|---|
| `optimization.build_graph` | 默认构建 canonical 投影图及保留候选索引 |
| `optimization.run_graph --phase retrieve` | 仍经 CompiledSourceMemory 返回旧完整事实附录 |
| `QueryFactContext.render` | 将同一批来源转换为当前十事实 QA 上下文 |
| `optimization.ircot --graph-retrieval hybrid` | 图 + Elasticsearch BM25 的多轮检索；默认值 graph 不含融合 |
| `optimization.ircot` 的 CAPS | 保留原轨迹生成设置 1/3/5/7；当前论文主表只报告 1/3/5 |
| `optimization.report_results` | 通用原生预测核对与六任务报告，不自动发现最新批次 |

最新 one-shot 上下文的调用关系如下。config 来自本项目配置加载器，artifact_directory 指向一个任务/来源组的 graph.json 所在目录：

```python
from optimization.retriever.hipporag import load_optimized_memory
from optimization.retriever.query_fact_context import QueryFactContext

memory = load_optimized_memory(config, artifact_directory, writable_runtime)
try:
    sources = memory.retrieve(question, top_k=5)
    context = QueryFactContext(memory._memory, memory.contents).render(question, sources)
    # Pass context to the existing task-specific QA evaluator.
finally:
    memory.close()
```

重放旧问题时，应使用对应优化图的 recognition 缓存和 CacheMissGuard；新的问题需要可用的抽取/识别模型服务。不能把 recognition 调用失败后退回 dense 当作成功重现。缓存、graph.json 含 Oscar 绝对路径，换账户需显式调整路径，写入位置应使用自己的 scratch。

## 5. One-shot 与 IRCoT 的区别

One-shot：原问题检索一次，五个中心原文加相关事实交给 reader。

IRCoT：[ircot.py](optimization/ircot.py) 使用已发表 IRCoT 的参与组件和控制流程，批量调用 vLLM；每轮由当前生成的推理文本触发检索，图始终固定。融合模式的每路检索数量来自官方配置，当前为六段，累计原文上限 15 段。轨迹沿用官方段落过滤及推理提示处理。

**当前主结果复用已有轨迹，仅在最终 QA 使用十事实和邻接原文。** 不是每轮都给增强上下文重新生成轨迹。旧完整附录的在线增强另有实验，但没有被选为当前主方法。最终 QA 仍使用原生任务提示、评分和 reader。

## 6. 保留哪些代码

| 目录/文件 | 保留原因 |
|---|---|
| `optimization/graph_construction/` | 当前关系缓存重建、支持筛选、构图和原文上下文 |
| `optimization/retriever/` | 当前图加载、RRF、来源身份、十事实展示 |
| `optimization/build_graph.py`、`run_graph.py` | 主方法及构图/索引必要消融 |
| `optimization/ircot.py`、`report_results.py` | 多轮运行、批量 QA、原生结果报告 |
| `optimization/test_*.py` | 保留模块的回归测试和完整缓存检查 |
| `baseline/`、`baseline_algorithms/`、`baseline_patches/` | 对照方法、已有本地改动和官方依赖 |
| `main.py`、`experiments/`、`dataset_loader/`、`utils/`、`requirements/` | 仍用到的实验入口、数据、评分及运行依赖 |

已删除的失败分支及一次性脚本列在 [results.md](results.md)。没有为了清理删除主方法的 schema 或改写 prompt、指标、数据划分；也没有把所有非主算法代码一概视为垃圾。

## 7. 验证命令

```bash
cd /oscar/home/zliu328/agent-memory
PYTHONDONTWRITEBYTECODE=1 /oscar/scratch/zliu328/agent-memory-envs/hipporag/bin/python -B -m unittest optimization.test_graph_construction optimization.test_report_results
```

默认测试不启动模型或 Slurm 作业。依赖完整图/检索缓存的三项检查需要显式配置 MODULE_ABLATION_ROOT 等环境变量，普通单元测试不会冒充全量 QA 重跑。
