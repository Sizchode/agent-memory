# LightMem 离线整合对照

状态说明（2026-09-12 UTC）：本页保留已执行的 LightMem 对照及其历史作业记录，不表示仍在运行。当前实验队列为空，不再自动提交实验；下一步见 [Generator 与 memory 质量检查计划](generator_memory_quality_review.md)。

本轮在已经完成的 LightMem memory 上增加官方离线整合，再运行同一 retrieval 和 QA。原来的“建立 memory 后直接评测”是保留的实验设置，不是无效结果；这次新增的是经过授权的对照，不重新抽取原文。Mem0 继续使用原官方 SDK 和原结果，不换版本、不重跑。

## 输入、输出与作业

- 输入：`/oscar/scratch/zliu328/agent-memory-outputs/final_qwen3_30b_seed42_clean_20260910/lightmem/`
- 新实验：`/oscar/scratch/zliu328/agent-memory-outputs/lightmem_offline_20260911T184409Z/`
- 离线整合及 retrieval：数组 `6242062`，六任务最多同时使用六张 `gpu-he` B200。
- QA：`6242063` 至 `6242068`，每个任务对应三个 `gpu` L40S evaluator 作业，依赖该任务自身的离线阶段成功，而不是等六任务全部结束。
- 日志：`/oscar/home/zliu328/agent_memory_<array job id>_<array index>.out`。例如 `agent_memory_6242062_0.out`。
- 已提交的具体依赖见新实验目录的 `launch.json`。

提交不表示已经完成；实时状态应查看 Slurm 和下述结果文件。12 小时是分配的作业时限，不是耗时预测。

### 首批实际计时与 token

以下来自已完成 group 的 `consolidation.jsonl`，不是按输出上限估算。时间单位为秒，token 为服务端实际 usage。

| 任务 | 队列构建 | 官方更新阶段 | 输入 token | 输出 token | 更新调用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SH-Doc_QA | 147.44 | 9.14 | 28,655 | 2,317 | 44 |
| MH-Doc_QA | 159.25 | 9.08 | 30,392 | 1,498 | 47 |
| FactConsolidation-SH | 4,037.54 | 264.83 | 768,725 | 23,511 | 1,264 |
| FactConsolidation-MH | 4,050.48 | 261.73 | 768,723 | 23,583 | 1,264 |
| LoCoMo（十段对话累计） | 336.52 | 95.46 | 1,027,431 | 67,954 | 1,642 |
| 2WikiMultiHopQA | 591.51 | 36.18 | 45,192 | 2,733 | 66 |

后续更新：六任务全部离线整合及检索作业已成功结束，15 个会话/语料组全部 completed、没有 failed 记录；QA 完成情况另查其依赖作业。上表为各任务内部累计时间，不能相加后称为并行作业的实际等待时长。

更新阶段包含候选扫描、模型请求和写回，不能等同于纯 GPU 推理时间。两阶段之和分别为 156.58、168.34 秒，不包含复制索引、模型加载、检索和 QA。对应离线加检索 Slurm 作业实际占用为 355、368 秒，包含启动等开销；这些口径不能混用。原 memory 的抽取成本依然属于完整方法成本，本轮只测新增整合成本。

SH-Doc_QA 的 2B QA 作业 `6242063_38` 在 CUDA 初始化时失败，预测文件为零行；已按同一参数提交 `6243155_38`，排除当次故障节点 `gpu3001`。没有修改 memory、模型或生成预算，也不把失败作业算成成功评测。

## 保持不变的设置

| 项目 | 设置 |
| --- | --- |
| 数据范围 | 六个保留任务，全部原始问题；不恢复 EventQA，不取子集 |
| Generator | 本地 Qwen3-30B-A3B-Instruct-2507，BF16，non-thinking |
| 服务 context 上限 | 32,768 |
| LightMem 输出上限 | 16,000 |
| 队列构建 | 官方 `construct_update_queue_all_entries()`；top_k=20，keep_top_n=10，max_workers=8 |
| 离线整合 | 官方 `offline_update_all_entries(score_threshold=0.9)`；max_workers=5 |
| Prompt | 官方 UPDATE_PROMPT，未修改 |
| Embedding | Qwen3-Embedding-0.6B，1,024 维 |
| Retrieval | 原流程，top-5 |
| QA | Qwen3.5-9B / 4B / 2B，原官方任务 prompt、生成预算与确定性 metric |
| Seed | 42 |

阈值 0.9 来自所用官方 LoCoMo 构建脚本 Phase 3 和方法默认值，不是根据当前结果调出来的。六任务统一调用这条官方入口。离线方法如何修改文本和存储向量也继续沿用原实现，没有加入重嵌入或其他新的整理规则。

## 只补执行入口和兼容性

本地 `experiments/consolidate_lightmem.py` 先复制每个 group 的已存索引，再以 `on_disk=True` 打开副本。该程序不调用 `build()` 或 `add_memory()`，原目录始终是复制来源。本次只发布研究文档，不包含该脚本的未提交改动。

官方 vLLM manager 缺少离线调用函数，因此适配器直接复用官方 `OpenaiManager._call_update_llm` 的函数体，仍由原本的本地 vLLM client 发请求。没有新写一套更新 prompt 或 JSON 修复规则。

另外两行修复让官方线程池的异常传回主程序：原先没有读取 `executor.map` 的结果，工作线程报错可能被静默丢弃。现在读取 iterator，遇到错误令任务失败；不会继续生成看似完成的结果。该修改不改变成功请求的输入、候选集合或更新操作。

这三个兼容性动作与启用离线整合本身分开记录，前者是调用/异常传播修复，后者是本轮实验变量。源码补丁保存在 `baseline_patches/`。

## 可检查的产物

每个任务的新目录包含：

- `lightmem_indices/`：离线整合后的副本。
- `consolidation.jsonl`：每组的开始、完成或失败状态，整合前后条数、删除/更新条数、候选队列时间、LLM 更新时间和实际 tokens/calls。
- `memory_changes.jsonl`：真正变化的 memory ID、操作、修改前后内容；不使用 LLM judge 或自动语义失败标签。
- `retrieval.jsonl`、`efficiency.jsonl`：既有 runner 产生的检索及检索计时。离线成本单独记录，不能把检索计时当成全构建成本。
- `evaluations/<model>/predictions.jsonl` 和 `summary.json`：三个模型的完整 QA。

## 对算法研究的用途

这轮给出一个真正改变 memory 的对照：相同初始记忆，经过或不经过官方离线整合，然后让相同下游流程作答。可以先检查哪些更新带来答对、哪些删除后原本答对的题变错，再追到具体变化的陈述。没有变化到完成 QA 之前，不预先称合并一定有利或一定有害。

需要特别区分：一次目标 memory 合并多个候选后，某个细节究竟被保留、删掉，还是被错误绑定到另一个时间、主体或事件。这个检查直接服务于记忆构建和组织，不增加查询时的推理模块，也不训练检索器。

题目分数继续使用全部原题的官方指标；上述逐题解释不是新的 benchmark 分数，也不能自动当作单个操作的因果贡献——一次离线整合可能同时改了多条 memory。

## 复用入口

提交新对照使用：

```bash
cd /oscar/home/zliu328/agent-memory
/oscar/scratch/zliu328/agent-memory-envs/runner/bin/python \
  experiments/launch_lightmem_offline.py \
  --memory-experiment final_qwen3_30b_seed42_clean_20260910
```

每次默认创建新的实验目录，不覆盖或继续修改原 baseline。不需要额外 API key。
