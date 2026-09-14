# 图记忆：当前实现与复现入口

更新：2026-09-14。精简主配置为 `canonical_latest_rrf_window`，删除 discourse 过滤。
对应已完成的整项消融 `without_discourse_selection`：9B/4B 均严格胜出 5/6。
旧主配置 `canonical_graph_rrf_window` 为 5/6、6/6；新配置牺牲 4B SH 的 3 个百分点，非无损替代。
删代码后的完整重建与六任务普通检索已验证一致，20 项现有测试通过；
新读出下的原图 QA 对照亦已完成，最终报告与全量核验通过，不继承旧读出的分差。

## 方法

1. 复用 HippoRAG2 的 OpenIE、实体/段落节点和向量索引。
2. 从来源关系标签与例子生成 schema，再联合归一关系别名；不读取问答。
3. 对每个规范化主体和 canonical relation，
   保留 loader 顺序中最后来源的三元组支持，同一末来源的多个值全部保留。
   当前对所有关系采用此政策，不读取 role/cardinality 或按 state/event 决定是否覆盖。
4. 将保留三元组累计为无序实体对边权，段落与实体之间用保留支持是否存在决定边权。
   原节点和边序不变；无支持的边权为零，不更新原向量或事实识别候选索引。
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
| `graph_construction/` | `relation_schema.py`：来源 schema；`canonicalize_schema.py`：关系归一；`source_consolidation.py`：支持选择/投影；`compiled_sources.py`：原文与事实表示；`source_window.py`：来源窗口 |
| `retriever/` | `hipporag.py`：原生查询接口；`hybrid_graph.py`：BM25/RRF；`compiled_sources.py`：来源读出 |

不计 `__init__.py`，核心是五个建图文件、三个检索文件。
完整目录现为 19 个代码/脚本文件、1447 行；已用完的消融执行文件不再保留在工作树。

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
主图默认 `canonical_latest_rrf_window`；`original_graph_rrf_window` 使用本次构建的相同新读出，
其结果不能与历史目录中同名但旧读出的对照混用。具体作业和范围记录在 `record.md`。

在 Oscar 重建已有来源索引的六任务图：

```bash
sbatch --array=0-5 --export=ALL,EXPERIMENT_ID=<fresh_name> optimization/build_graph.sbatch
```

随后用 `run_graph.sbatch` 的 `PHASE=retrieve` 执行检索，`PHASE=evaluate` 执行 QA，
或直接调用 Python CLI。`TASK` 使用 `SH-Doc QA` / `MH-Doc QA` 的空格拼写；
输出目录中的任务名使用下划线。新语料索引与 schema 可用 `build_graph.py` 的 root 参数指定，
数据仍走已有 loader；这一步不负责重新抽取 OpenIE 或重新生成 schema。

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

最终固定检索/读出规则的对照中，精简主图相对原图的 2Wiki 分差为 9B +6.79、4B +7.71 点，
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
