# 图记忆：当前实现与复现入口

当前方法暂名 **Source-Supported Graph Indexing（来源支持驱动的图索引）**：
先选择事实支持，再从支持构造传播图；最新独立消融还让同一支持决定事实识别候选。
问题主线是 consolidation 后的索引构造，不是新抽取器、agentic retrieval 或 PL 正确性理论。
具体设计与尚缺的对照见 [research_novelty.md](research_novelty.md)。

更新：2026-09-14。默认改为 `statement_projection_loop_free_refined_rrf_window`：
来源支持选择后执行已有无自环事实投影，保留来源直连；不重新抽取或调在线检索参数。
六任务完整结果为 9B 5/6、4B 6/6；同 H100 的旧 refinement 对照为 4/6、6/6。
9B FCSH 为 65，低于最佳 baseline 66；有任务回退，不宣称双 reader 全面胜出。
显式事实节点、含自环投影的完整对照已归档并移除实验入口。
删直连为 9B 4/6、4B 5/6；全部支持但去 synonym 为 5/6、4/6，均不采用。
失败开关已删除，所有图、缓存与完整结果保留。清理后重建与普通检索核验状态见 record.md。
比较使用预先指定的同 H100 旧图复现；历史成绩另存，不择高、不按题拼接。
默认切换后的六任务四格普通检索核验已全部通过。删除图侧 canonical 标签的完整消融
仍为 9B 5/6、4B 6/6，但 FCMH 从 11/9 降至 9/6；保留为取舍结果，不替换默认。
该轮 reader 事实附录冻结，不能据此取消整条流水线的 schema。实验入口已归档退役，见 record.md。
无自环 refined 构图下删除额外来源直连的完整消融为 9B 4/6、4B 5/6，未采用；
该轮仍保留事实投影中的来源关联。实验开关已归档退役，默认保留两类来源连接。
当前新增 `--retained-fact-index` 独立消融：只让已有保留事实进入识别候选，图和 reader 冻结。
候选变化可能触发新 recognition，原提示/模型/算子不改；这轮不再声称识别候选完全冻结，
生成调用须实测，尚无完整 QA 结果，不改默认。旧事实索引负结果及本轮门槛见 record.md。
截至发布核验，候选 9B 六任务已完成，分数均超过本地基线阈值；4B 的四项 MAB 与
2Wiki 已完成并严格胜出，LoCoMo 及双 reader 总报告尚未完成，不报双 6/6。

## 方法

本轮优化范围是「构图与索引 + 图表示 refinement」，不是只调旧图权重：

| 层次 | 当前改动 | 保持不变 |
|---|---|---|
| 构图与索引 | 用事实及其实际来源的成员关联生成无自环投影，替换原实体对事实计数贡献 | OpenIE、原实体/段落节点、embedding、事实识别候选索引 |
| 图表示 refinement | 关系键归一、按最后来源选择事实支持、筛选来源直连并去掉 synonym 贡献 | 原始历史文本 |
| 检索、读出与 QA | 本轮冻结，不作为新增构图贡献 | recognition/PPR、BM25/RRF、五中心、来源窗口、事实附录、QA 协议 |

实现中先选择支持，再构造事实关联和投影；refinement 是功能划分，不表示必须在新图建完后
再执行一遍后处理。目前是离线整合，没有在线增量更新接口。

1. 复用 HippoRAG2 的 OpenIE、实体/段落节点和向量索引。
2. 从来源关系标签与例子生成 schema，再联合归一关系别名；不读取问答。
3. 对每个规范化主体和 canonical relation，
   保留 loader 顺序中最后来源的三元组支持，同一末来源的多个值全部保留。
   当前对所有关系采用此政策，不读取 role/cardinality 或按 state/event 决定是否覆盖。
4. 将每个不同的规范化三元组表示为事实关联，成员是主体、客体及其被保留的来源。
   采用 Kumar 等（2020，式 3）的已有无自环度数保持投影，物化回原实体/段落节点；
   保留有支持的原来源直连，不保留 synonym 贡献。不更新原向量或事实识别候选索引。
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
主图默认 `statement_projection_loop_free_refined_rrf_window`。
构图默认同时产出有无 refinement 两格；`--construction projected` 产出旧 refinement 与原图两格。
四格读出规则相同，不能与历史目录中同名但旧读出的对照混用。具体作业和范围见 record.md。

在 Oscar 重建已有来源索引的六任务图：

```bash
sbatch --array=0-5 --export=ALL,EXPERIMENT_ID=<fresh_name> optimization/build_graph.sbatch
```

随后用 `run_graph.sbatch` 的 `PHASE=retrieve` 执行检索，`PHASE=evaluate` 执行 QA，
或直接调用 Python CLI。`TASK` 使用 `SH-Doc QA` / `MH-Doc QA` 的空格拼写；
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
最新正负结果、成本与等待项见 [record.md](record.md) 和 [ablation_jobs.json](https://github.com/Sizchode/agent-memory/blob/f2121c11cc6d0c36bf931674152db0ca07669573/optimization/ablation_jobs.json)。

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
