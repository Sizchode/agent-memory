# 图记忆：当前实现与复现入口

当前方法暂名 **Source-Supported Graph Indexing（来源支持驱动的图索引）**：
先选择事实支持，再从支持构造传播图；最新独立消融还让同一支持决定事实识别候选。
问题主线是 consolidation 后的索引构造，不是新抽取器、agentic retrieval 或 PL 正确性理论。
论文标题方向为 **Consolidate Before You Propagate**；具体设计与证据边界见
[research_novelty.md](research_novelty.md)。

更新：2026-09-14。默认为 `statement_projection_loop_free_retained_index_rrf_window`：
同一份保留支持生成无自环事实投影图，并筛选原事实识别候选；保留来源直连。
OpenIE、原向量、recognition/PPR 算子、RRF、读出和 QA 协议不变；候选变化会改变查询种子。

六任务完整结果及同索引构图对照均已核验，原指标按百分制列出：

| reader / 构图 | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9B 新构图 + 保留事实索引 | 91 | 63 | 68 | 11 | 57.5679 | 57.7539 | 6/6 |
| 9B 旧 refinement 图 + 同索引 | 94 | 59 | 64 | 15 | 57.4063 | 56.6379 | 4/6 |
| 4B 新构图 + 保留事实索引 | 90 | 62 | 61 | 12 | 51.0135 | 56.3973 | 6/6 |
| 4B 旧 refinement 图 + 同索引 | 91 | 59 | 61 | 12 | 50.4071 | 55.9104 | 6/6 |

这是对九项完整本地 baseline 的逐任务严格胜出，不是全领域 SOTA 或独立测试泛化。
两种图并非逐任务支配：新图的 SH 和 9B FCMH 回退，保留完整取舍结果。
原候选索引下的新构图为 9B 5/6、4B 6/6；新候选改善目标覆盖，但其 2Wiki 小幅回退。
完整四格、同索引对照、删减实验和原始成本见 [record.md](record.md)。

新索引正式检索新增 recognition 2168 次，输入 6323456、输出 102681 token，pilot 另计。
旧图同索引复用相同识别缓存，新增调用为零；不等于方法整体无需识别成本。
获胜配置已独立重建：15 组图、14362 来源、167074 候选完全匹配，3386 题普通接口验证通过。
显式事实节点、旧自环投影、raw-relation 和删直连等独立实验入口已归档退役，结果和缓存保留。

## 方法

本轮优化范围是「构图与索引 + 图表示 refinement」，不是只调旧图权重：

| 层次 | 当前改动 | 保持不变 |
|---|---|---|
| 构图与索引 | 事实及其实际来源的无自环投影；同一支持选择筛选原事实候选 | OpenIE、原实体/段落节点、embedding、原实体/事实来源映射 |
| 图表示 refinement | 关系键归一、按最后来源选择事实支持、筛选来源直连并去掉 synonym 贡献 | 原始历史文本 |
| 检索、读出与 QA | 算子冻结，不作为新检索贡献；候选改变会影响识别输入和查询种子 | recognition/PPR、BM25/RRF、五中心、来源窗口、事实附录、QA 协议 |

实现中先选择支持，再构造事实关联和投影；refinement 是功能划分，不表示必须在新图建完后
再执行一遍后处理。目前是离线整合，没有在线增量更新接口。

1. 复用 HippoRAG2 的 OpenIE、实体/段落节点和向量索引。
2. 从来源关系标签与例子生成 schema，再联合归一关系别名；不读取问答。
3. 对每个规范化主体和 canonical relation，
   保留 loader 顺序中最后来源的三元组支持，同一末来源的多个值全部保留。
   当前对所有关系采用此政策，不读取 role/cardinality 或按 state/event 决定是否覆盖。
4. 将每个不同的规范化三元组表示为事实关联，成员是主体、客体及其被保留的来源。
   采用 Kumar 等（2020，式 3）的已有无自环度数保持投影，物化回原实体/段落节点；
   保留有支持的原来源直连，不保留 synonym 贡献。由同一支持选择保留原事实识别候选，
   不更新原向量，不修改实体/事实到原来源的映射。
5. 查询走原 HippoRAG recognition/PPR，以及同语料 BM25；
   两者各取前五，以等权 RRF（常数 60）合并为五个中心来源。
6. 读出完整中心原文、来源位置/已有时间与保留三元组 JSON。
   同一非空时间 metadata 的连续来源段内附加前后三条原文；不同中心间不去重。

最后来源不是事件时间。保留原文不保证历史问答无损；五个中心不等于五段文本或等 token。
这是一次离线语料构建，不是一次 LLM 调用；没有查询时建图或 agentic retrieval。
原始抽取和 schema 生成成本不能因为复用落盘结果而记成零。

## 核心文件

| 目录 | 文件与职责 |
|---|---|
| `graph_construction/` | `relation_schema.py`：来源 schema；`canonicalize_schema.py`：关系归一；`source_consolidation.py`：支持选择/旧图对照；`statement_incidence.py`：事实成员与无自环投影；`compiled_sources.py`：原文与事实表示；`source_window.py`：来源窗口 |
| `retriever/` | `hipporag.py`：原生查询接口；`hybrid_graph.py`：BM25/RRF；`compiled_sources.py`：来源读出 |

不计 `__init__.py`，核心是六个建图文件、三个检索文件。
主入口已解除对旧 `run_anchormem.py`、`run_gap_query_memory.py` 实验脚本的导入依赖；
QA 计量留在 `run_graph.py`，真实 recognition 调用计量复用 `baseline/graph_usage.py`。
`statement_incidence_graph` 构建事实成员，`project_statement_graph` 使用唯一保留的无自环算子。
显式事实节点仅为内部构建步骤，不保留为单独在线算法或新增向量索引。
谓词用于区分事实，不由 PPR 在线解释，投影也不是无损的事实编码。
冻结精简快照为 19 个代码/脚本文件、1447 行；当前构图实验新增一个模块并扩展已有入口和测试，
不把快照行数当成当前工作树统计。已用完的旧消融执行文件不再保留在工作树。

## 实验入口

Python 入口保留在仓库；以下三个 Oscar sbatch 启动器仅保留本地副本，不再跟踪，
新 checkout 可在文末的清理前版本中查看，或直接调用对应 Python 模块。

- `build_schema.sbatch`：来源 schema 与联合归一。提示、例子选择和生成协议保持原设置。
- `build_graph.py` / `build_graph.sbatch`：从原索引、既有 loader 和 schema 直接物化主图与原图对照，CPU 走 batch 分区。
- `run_graph.py --phase retrieve`：在已构建图上通过普通接口执行完整任务，默认不允许新增 recognition 调用。
- `run_graph.py --phase verify`：通过普通查询接口核对已有完整检索记录。
- `run_graph.py --phase evaluate`：对完整检索记录运行原 QA/评分流程，当前仅 9B、4B。
- `report_results.py`：已有完整实验的原指标、QA usage 与 baseline 对照。

`load_optimized_memory(config, artifact_directory, runtime)` 接受普通新问题，
不查题号或保存的 query reset。识别缓存未命中时须配置可用的同源生成服务；
冻结结果验证默认禁止新增生成调用，缓存缺失明确失败。

旧 `replay_graph.py`、`compile_graph_context.py`、`prepare_hybrid_graph.py` 及对应 sbatch
已由 `build_graph` 替代并删除，不再依赖保存的 query seed、历史检索排名或 adaptive 中间产物。
cardinality 分支、纯三元组读出及旧候选列表已删除。旧主配置入口曾通过六任务两种图的
完整重建与普通检索验证；本次去 discourse 后另做全量校验，不把旧验证算作新验证。
主图默认 `statement_projection_loop_free_retained_index_rrf_window`。
默认只构建获胜配置；`--construction projected` 构建旧 refinement 图与同一保留事实索引。
`--no-retained-fact-index` 恢复原完整候选索引，并分别构建有无 refinement 两格；
结合两种 construction 可复现原候选索引下的四格对照。读出规则相同，不能与历史旧读出混用。
实际配置选择与各阶段完整范围见 record.md。

在具备上述来源索引、schema 和冻结读出的环境中，直接使用仓库内的 Python 入口。
例如重建一个任务，六任务分别使用前述原任务名和同一新输出根目录：

```bash
python -m optimization.build_graph \
  --task "SH-Doc QA" \
  --output-root /oscar/scratch/zliu328/agent-memory-outputs/<fresh_name>
```

Oscar 上 CPU 构建提交到 batch 分区；本地保留的 sbatch 只是调度包装，不是 Git checkout 的前置文件。
随后用 `python -m optimization.run_graph --phase retrieve` 执行检索，`--phase evaluate` 执行 QA，
同时指定相同的 `--task` 与 `--output-root`。识别缓存不全时需提供同源生成服务地址，
并显式启用 `--allow-generator-calls`；默认验证路径不允许缓存未命中后静默降级。
`--task` 使用 `SH-Doc QA` / `MH-Doc QA` 的空格拼写；
输出目录中的任务名使用下划线。新语料索引与 schema 可用 `build_graph.py` 的 root 参数指定，
数据仍走已有 loader；这一步不负责重新抽取 OpenIE 或重新生成 schema。
默认新构图会与 `--readout-root` 下的冻结读出逐条核对；迁移语料需提供对应来源索引、schema
及读出参照，不是无需前置产物的端到端抽取入口。

## 消融与评测

已完成六项单消融：RRF、来源窗口、事实附录、关系归一、latest-only、discourse filter。
这是模块粒度计数，不是全部人工选择的总数；具体选择与未消融部分见 [ablation_plan.md](ablation_plan.md)。
后三项只改变图，保留旧 reader 事实表示，不等于整条流水线删除相应机制。
另有两项联合消融：同时删除图筛选，分别保留旧事实读出或同时删除附录。
已完成的两项联合方案均不能替代两个 reader 的统一主配置。
依赖检查后另设 `without_discourse_selection`：图与附录同时取消 discourse 过滤，
该组也已全量完成，报告 6359682 审核通过，6372 新 QA / 400 整任务复用。
按双 reader 至少 5/6 的精简目标采用该组，不与原图-only 消融混用。
去 RRF、窗口、canonicalization、latest 至少使一个 reader 降到 4/6，暂保留；
去附录虽有双 5/6，但 FCSH 降 19/13 点，未采用。

固定 9B/4B、六任务全量共 3386 题：SH/MH Doc QA 与两项 FactConsolidation
各 100，LoCoMo 1986，2Wiki 1000（使用的已发布 HippoRAG 子集，并非原始全集）。
沿用原 loader、划分、QA prompt、解码和评分：MAB 为 substring_exact_match，
LoCoMo 为 f1，2Wiki 为 answer_f1。不把答案指标称为 retrieval F1。

test set 已用于开发与配置选择，不能报告成独立测试泛化。
九项既定 baseline 的比较不等于全领域 SOTA；上下文相同但 QA 分数变化不归因于图。
目前没有一项主机制已被完整消融证明可全局无损删除。
discourse 的删除是已披露的效果/简化取舍。原 schema 提示仍输出 role/cardinality，
为保持生成协议未删除这些字段；图与附录已不使用它们，不声称消除其历史生成成本。
最新正负结果与成本见 [record.md](record.md)；历史调度记录见
[ablation_jobs.json](https://github.com/Sizchode/agent-memory/blob/f2121c11cc6d0c36bf931674152db0ca07669573/optimization/ablation_jobs.json)。

以下是已冻结的旧 refinement 对照，不是当前无自环构图或同 H100 四格结果。
该轮固定检索/读出规则，精简主图相对原图的 2Wiki 分差为 9B +6.79、4B +7.71 点，
LoCoMo 则为 -0.71 / -0.20 点；不声称所有任务更好。原图对照对九项 baseline 为双 2/6，
精简主图为双 5/6。这是图物化阶段的整体条件差异，不是某个子规则的独立因果效应或显著性结论。
两图选择不同来源，全任务输入 token 每 reader 相差 136764，固定读出规则不等于相同 token 预算。
完整 `comparison.json` 与 `results.md` 位于
`/oscar/scratch/zliu328/agent-memory-outputs/optimization_canonical_latest_cleanup_seed42_20260914/`。

## 历史与产物

24 个历史作业 JSON 和 3 个本地 sbatch 已取消 Git 跟踪，GitHub 当前版本不再包含它们；
本地副本保留，代码不依赖这些 JSON 调度记录。需要核对时可从
[清理前版本](https://github.com/Sizchode/agent-memory/tree/f2121c11cc6d0c36bf931674152db0ca07669573/optimization)
查看。算法 Python 文件及构建、评测入口仍保留在仓库。
本次清理不删除 scratch 中的完整预测、报告、索引或代码归档。

2026-09-14 后续整理：24 个历史作业 JSON、7 个退役说明 MD、2 个历史环境 TXT
已打包到 `/oscar/home/zliu328/agent-memory-archives/refinement_only_20260914/legacy_notes.zip`，
通过 ZIP 完整性及逐文件内容比对后移出本地目录。当前三个 sbatch 本地入口仍保留。
README、record、ablation_plan、related_work_analysis、research_novelty 保留为当前查阅入口；
历史说明链接指向 Git 旧版本。算法、当前环境及实验结果不变。

已删除旧权重搜索、adaptive synonyms、gist、fact incidence/index、rank window、
context packing、state/event 读出分支及其专用入口/测试。旧代码不搬到新的算法目录。
清理前包含未跟踪文件的快照：
`/oscar/scratch/zliu328/agent-memory-outputs/optimization_code_before_legacy_cleanup_20260914.tar.gz`。

本轮完成后删除 `ablation.py`、`prepare_ablation.py`、`evaluate_ablation.py`、
`report_ablation.py`、`run_ablation.sbatch`，并去掉主构造器的消融开关和去附录分支。
完整旧实现与消融复现入口保存在
`/oscar/scratch/zliu328/agent-memory-outputs/optimization_code_before_discourse_removal_20260914.tar.gz`。
三组完整消融的 JSON/Markdown 报告、预测及成本均保留，不需要重新生成报告才能阅读结果。

原图、embedding、schema、QA、负结果和旧实验记录均保留在 scratch 或历史记录中。
旧结构专用 artifact 不再保证由当前 loader 加载，需要归档实现；不能静默换成当前算法。
一次性测试脚本完成验证后删除，不增加替代文件。
当前贡献解释及其限制见 [research_novelty.md](research_novelty.md)。
