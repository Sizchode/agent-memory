# 最终基线实验：运行状态与故障记录

更新时间：2026-09-11 00:38 EDT

## 实验范围

- 方法：BM25、Dense、LightMem、HippoRAG 2、Mem0。
- 最终保留的数据集任务：SH-Doc QA、MH-Doc QA、FactConsolidation-SH、FactConsolidation-MH、LoCoMo、2WikiMultiHopQA。EventQA 在 2026-09-10 03:10 EDT 应用户决定从本轮论文实验中移除。
- 答题模型：Qwen3.5-9B、Qwen3.5-4B、Qwen3.5-2B。
- Generator Backbone：Qwen3-30B-A3B-Instruct-2507，BF16，关闭 thinking。
- 检索返回 5 条证据，随机种子为 42。
- LightMem 单次生成上限为官方的 16,000 tokens；Mem0 为官方的 2,000 tokens；没有动态修改任何任务的上限。

## 最终进度

- BM25、Dense、LightMem、HippoRAG 2 和 Mem0 在最终保留的 6 个任务上均已完成 retrieval 与三个 evaluator QA。
- 最终实验共有 30 个 retrieval cell 和 90 个 QA cell；全量 validator 已全部通过。
- Mem0 2WikiMultiHopQA 的完整备用作业 `6191128_34` 于 2026-09-11 00:23 EDT 正常完成，退出码为 0，写出 1,000 条唯一 case 的 top-5 retrieval 和一条 completed efficiency 记录。
- 对应三个 QA 于 00:35 EDT 前完成。Qwen3.5-9B、4B、2B 的 answer F1 分别为 0.435900、0.407367、0.347934。
- 6,119 次 Mem0 extraction 中有 1 次响应为无法解析的 JSON。官方 Mem0 路径将该次记为空抽取后继续；本轮没有添加 retry、fallback 或 JSON 修补，也没有因此改动参数。该事件作为方法失败保留在最终记录中，作业本身没有 runtime error。
- 本轮最终结果没有使用 EventQA；其失败原因作为已排除数据集的诊断记录保留在本文档。

## 评测节点 ECC 故障

BM25 的 SH-Doc QA 和 MH-Doc QA 的 9B evaluator 最初各运行 13 秒后失败。两个作业都落在 `gpu3006` 的同一张 L40S，日志在模型载入时报告 `CUDA error: uncorrectable ECC error encountered`。故障发生在第一个问题之前，原文件均为空，不是实验结果。

待执行的短、长 QA 及重试作业均已排除 `gpu3006`。两个 cell 在其他 L40S 上原参数重跑成功，各写出 100 条预测，得分分别为 0.81 和 0.47。因此这是已解决的节点硬件错误，没有改变 evaluator 或实验协议。

## 2WikiMultiHopQA 的阶段性 retrieval 结果

2WikiMultiHopQA 有官方 gold supporting passages，因此可以对返回原文 passage 的方法直接报告 Recall@5 和 Precision@5：

| 方法 | Recall@5 | Precision@5 | 构建时间 | Generator 调用 | Generator tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.6583 | 0.3106 | 0.15 秒 | 0 | 0 |
| Dense | 0.6883 | 0.3228 | 25.85 秒 | 0 | 0 |
| HippoRAG 2 | 0.7498 | 0.3544 | 3,115.91 秒 | 13,238 | 10,799,457 |

HippoRAG 的证据覆盖最高，但成本也显著高于 Dense。LightMem 和 Mem0 返回的是生成式 memory 而非原文 passage，不能将字符串无法精确匹配 gold passage 所得的零分当作可比 retrieval 结果。最终端到端 QA 也由 HippoRAG 在 2Wiki 的三个 evaluator 上全部领先。

## HippoRAG 在 B200 上的故障

最初将全部需要生成模型的方法放在 B200 上。Qwen 生成服务能够正常使用 B200，但是 HippoRAG 官方环境固定使用 PyTorch 2.5.1。这个 PyTorch 构建只包含到 `sm_90` 的 CUDA 内核，不包含 B200 所需的 `sm_100` 内核。

HippoRAG 在对第一批文本运行 Qwen3-Embedding-0.6B 时因此报错：

```text
NVIDIA B200 with CUDA capability sm_100 is not compatible with the current PyTorch installation.
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

故障发生在 embedding 的第一次前向计算，尚未调用 Generator Backbone，也没有生成可用的 memory 或 retrieval 结果。因此这不是数据、prompt、16K 输出上限、检索设置或 HippoRAG 算法本身的问题。

## 处理方式及公平性

没有升级 HippoRAG 的 PyTorch，也没有修改模型、prompt、上下文长度、输出长度、chunk、top-k、随机种子或数据。处理方式只是把 HippoRAG 移到其官方环境原生支持、并且能够容纳 BF16 30B Generator Backbone 的 H100：

- SH-Doc QA 使用 `gpu-debug` 的 H100，作业号 `6185256_21`；其余四个较短任务使用作业 `6185373`。
- LoCoMo 和 2WikiMultiHopQA 使用 `gpu-he` 的 H100，作业号 `6185257`。
- B200 上失败或取消的空 HippoRAG 产物已经删除；H100 从每个任务的原始输入重新运行。
- LightMem、Mem0、BM25 和 Dense 的已有结果没有删除或重建。

第一次提交短任务时，每个任务申请了 64GB CPU 内存。`gpu-debug` 的用户上限是 96GB，因此只能同时启动一个任务。SH-Doc QA 运行时测得 CPU 峰值约为 6.8GB；尚未启动的四个任务据此改为每个 24GB、2 CPU，使该 partition 最多可以同时运行四张 H100。这只修改 Slurm 的 CPU/内存资源申请，不改变 GPU、模型或实验计算。

`gpu-he` 对该用户最多允许同时运行 6 个 GPU job。原 B200 数组的后续并发已从 6 调整为 4，为两个长 HippoRAG H100 任务保留 2 个槽位。最终并发构成为 4 张 B200 加 2 张 H100，仍使用满 6 个允许的 job。

这次改动只解决硬件与官方软件环境不兼容，不改变任何实验方法。后续 launcher 也已按同一规则固定：LightMem 和 Mem0 使用 B200，HippoRAG 使用 H100。

H100 上的首个完整任务验证了修复：HippoRAG SH-Doc QA 用时 13 分 10 秒，作业退出码为 0，没有 CUDA、生成或解析错误。它写出了预期的 100 条 retrieval 记录，问题编号无重复且每条正好返回 top-5。

HippoRAG EventQA 随后暴露了另一个纯运行时问题。H100 共有约 79.2GB 显存；vLLM 在默认的 0.85 显存占比下实际占用约 68.2GB，而使用 4096-token 文本的 embedding 进程已经占用约 10.7GB，并需要再申请 522MB，因而 OOM。该任务在第一次 embedding 前向计算时失败，Generator Backbone 调用数仍为 0，没有产生可用结果。

首先根据实测显存尝试将 vLLM 的占比从 0.85 降到 0.82。vLLM 报告的 KV cache 从 67,600 tokens 降至 41,664 tokens，仍高于 32K context；但 embedding 进程使用了新释放的显存后仍在同一处 OOM。因此不能继续猜测更低比例：再降低可能使 KV cache 小于规定的 32K context，而且不能解决两个模型峰值同时出现的问题。该次失败仍发生在第一次 embedding 前向计算，Generator Backbone 调用数为 0，失败目录已经删除。

最终的运行时修复是为 HippoRAG EventQA 的同一个 Slurm job 申请两张 H100：GPU 0 运行不变的 BF16 30B Generator Backbone，GPU 1 运行不变的 Qwen3-Embedding-0.6B。模型、精度、batch、32K context、生成上限、prompt、chunk、top-k 和随机种子均不变，只把原本互相争用显存的两个现有组件放到不同设备。该处理只用于使用官方 4096-token chunk、并实际复现 OOM 的 EventQA。

双 H100 作业已经验证设备分离：启动时 GPU 0 上的 vLLM 占用约 69.8GB，GPU 1 仅有显示服务的约 83MB；随后 9/9 个长文本 embedding batch 全部完成，越过了前两次的固定 OOM 点，并进入 NER 和 triples 抽取阶段。

## HippoRAG EventQA 的官方输出上限

双 H100 解决了 OOM，但完整运行第一个 EventQA group 后，HippoRAG 在 OpenIE 阶段按官方逻辑退出。这一组共有 131 次 NER 和 131 次 triple extraction，262 次 API 调用本身全部成功，但缓存中的 `finish_reason` 显示：

- 41/131 个 NER 输出用满官方 512-token 上限；
- 68/131 个 triple 输出用满官方 2,048-token 上限；
- 共 109/262 个输出因长度停止；官方 `fix_broken_generated_json` 能修复大部分被截断 JSON，但两个 triple 输出在字符串中间截断，仍无法解析。

最终错误是 `Triple extraction failed for 2 chunk(s)`，而不是 CUDA OOM、API 故障、32K context 溢出或 metric 错误。EventQA 使用官方 4,096-token chunk，单个 chunk 产生的实体和关系显著更多，因此将 HippoRAG 自身的 512/2,048 输出上限变成了真正瓶颈。

截断现象不是 EventQA 绝对独有：已完成的 HippoRAG SH-Doc QA、MH-Doc QA 和两个 FactConsolidation 任务也有少量输出用满 512 或 2,048 tokens，但它们都被官方修复器成功处理，memory 构建和 retrieval 完整结束。所以，当前的致命问题是 **HippoRAG 与 EventQA 的方法-数据集组合**，不是整个 benchmark 套件的 16K 或 32K 问题。

本轮没有提高 512/2,048 上限，没有改小 4,096-token chunk，也没有跳过失败 chunk。按已确定的“保留 algorithm default”规则，这个 cell 目前应记为官方方法设置下无法完成，而不是修改参数制造一个结果。

## 关于 LightMem 的 16K

16K 是 LightMem 每次 extraction 调用的最大输出长度，不是把数据集裁成 16K。此前观察到的截断集中在 LightMem 与 EventQA 的组合，因为该任务的单批抽取输出较长；并非所有数据集的共同问题。

本轮不会提高或动态改变 16K。任务完成后将检查生成失败次数、无 token 统计的响应、最终 group 数和问题数。若 EventQA 在官方上限下仍有批次失败，将把它作为方法在长结构化输出场景中的限制报告，而不是改变上限重跑。

## 最终作业状态

- 原 12 小时 Mem0 2Wiki 作业 `6182678_34` 因预计无法在时限内完成而取消；它的 partial state 未用于最终结果。
- 从原始输入完整运行的 24 小时作业 `6191128_34` 正常完成；完整目录在只读语义校验通过后替换了旧的 0 行 canonical 目录。
- 最后三个 QA 使用作业 `6224952_102`、`6224952_103` 和 `6224966_104`，分别在 `gpu` 与 `gpu-he` 的 L40S 上正常完成。只迁移了尚未启动的 2B task；没有取消或复制已运行的 evaluator。
- 当前没有仍需等待的 agent-memory 实验作业。

Mem0 MH-Doc QA 已于 07:34 EDT 成功完成，对应三路 QA `6189076` 也已全部完成并通过验证。Mem0 2WikiMultiHopQA 因本地 memory store 增大而逐渐减速，原 12 小时作业号为 `6182678_34`。11:32 EDT 时其进度为 4,530/6,119，最近 60 分钟约 5.63 次调用/分钟，而按 15:23 EDT 时限完成需要约 6.87 次/分钟。这是资源时限风险，不是程序错误。

为避免原作业只差少量输入却因 Slurm 时限丢失全部进度，当时在另一张 B200 上启动了完全相同协议的 24 小时备用作业 `6191128_34`，使用独立目录 `2WikiMultiHopQA_backup_20260910`。它的模型、BF16 精度、prompt、32K context、2K Mem0 输出上限、top-5 和 seed 42 均不变。原作业随后取消；最终只采用备用作业从原始输入完整运行的结果，没有混合两次运行的 memory，也没有做部分 state 拼接或启发式 resume。完整产物通过只读校验后被重命名为 canonical 目录，旧的 0 行 partial 目录已经删除。

EventQA 的 LightMem 和 Mem0 运行已取消，五个方法的 EventQA 结果目录均已删除。这两个作业释放的 B200 槽位已被 Mem0 FactConsolidation-MH 和 LoCoMo 接管。HippoRAG LoCoMo 完成后，`gpu-he` 只剩一个 HippoRAG H100 作业；因此将原 memory array 的 B200 并发从 4 恢复为 5，使最大的 Mem0 2WikiMultiHopQA 在 03:23 EDT 立即开始。总并发仍严格遵守 `gpu-he` 的 6-GPU 用户上限。

由于本轮存在部分取消后重新提交的 Slurm array，最终判断不依赖 array 的聚合退出码，而依赖每个 cell 的产物。每个 retrieval cell 均已确定性检查官方问题数、group 数、JSON 可读性、唯一 case，以及每条是否正好返回 top-5；每个 QA cell 均已检查预测数和 summary。最终 30 个 retrieval cell 和 90 个 QA cell 已全部通过全量验证。
