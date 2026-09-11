# Agent Memory 基线结果与失败分析

更新时间：2026-09-11 00:38 EDT

## 如何阅读这份文档

这份文档面向没有参与代码开发的合作者。当前保留 6 个任务：SH-Doc QA、MH-Doc QA、FactConsolidation-SH、FactConsolidation-MH、LoCoMo 和 2WikiMultiHopQA。EventQA 已因 HippoRAG 与该数据集的官方输出上限不兼容而从最终实验中排除。

表格中每个数字都是官方确定性指标，没有使用 LLM judge。每个 cell 按顺序报告 Qwen3.5-9B、Qwen3.5-4B 和 Qwen3.5-2B 三个 Evaluation Backbone。MemoryAgentBench 任务报告 Substring Exact Match，LoCoMo 报告官方 F1，2WikiMultiHopQA 报告 answer F1。这三类指标的含义不同，因此不构造一个新的总分。

## 当前完整性

- 五个方法在六个保留任务上的 30 个 retrieval cell 和 90 个 QA cell 已全部完成。
- 全量 validator 已检查每个预期文件、JSON 结构、官方问题数、唯一 case ID、固定 top-5、三个 Evaluation Backbone 的预测数和 summary，结果为通过。
- Mem0 的 2WikiMultiHopQA 从原始输入完整运行，写出 1,000 条 retrieval 和一条 completed efficiency 记录；未使用被取消作业的部分 state，也未合并两次运行结果。

## 端到端 QA 结果

| 任务 | BM25 | Dense | LightMem | HippoRAG 2 | Mem0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SH-Doc QA | 0.81 / 0.82 / 0.72 | 0.53 / 0.50 / 0.47 | 0.51 / 0.47 / 0.45 | **0.88** / **0.89** / 0.75 | 0.85 / 0.86 / **0.77** |
| MH-Doc QA | 0.47 / 0.43 / 0.34 | 0.43 / 0.37 / 0.28 | 0.41 / 0.40 / 0.30 | **0.59 / 0.54** / 0.47 | 0.50 / 0.51 / **0.49** |
| FactConsolidation-SH | 0.48 / 0.48 / 0.45 | 0.19 / 0.22 / 0.15 | 0.49 / **0.51** / 0.44 | 0.36 / 0.39 / 0.40 | **0.60** / 0.47 / **0.54** |
| FactConsolidation-MH | 0.03 / 0.04 / 0.04 | 0.03 / 0.00 / 0.03 | 0.03 / 0.04 / 0.03 | **0.04** / 0.02 / 0.02 | **0.04** / **0.05** / **0.08** |
| LoCoMo | 0.474 / 0.436 / 0.370 | 0.498 / 0.476 / 0.435 | 0.439 / 0.406 / 0.368 | 0.504 / 0.484 / 0.436 | **0.537 / 0.497 / 0.444** |
| 2WikiMultiHopQA | 0.451 / 0.426 / 0.339 | 0.488 / 0.460 / 0.365 | 0.343 / 0.328 / 0.292 | **0.519 / 0.498 / 0.370** | 0.436 / 0.407 / 0.348 |

最终结果呈现两个互补的强方法：HippoRAG 2 在 MH-Doc QA 的 9B/4B evaluator 和 2WikiMultiHopQA 的三个 evaluator 上领先；Mem0 在 LoCoMo、FactConsolidation-SH 的 9B/2B evaluator、MH-Doc QA 的 2B evaluator 上领先，并将 FactConsolidation-MH 的最高分提高到 0.08。按 18 个“任务 × evaluator”cell 分别比较，Mem0 单独领先 9 个，HippoRAG 2 单独领先 7 个，LightMem 单独领先 1 个，另有 1 个 HippoRAG 2 与 Mem0 并列。这个计数仅描述各 cell 的领先者；由于三类任务使用不同官方指标，不把它当成新的总体分数。

## 2WikiMultiHopQA 检索结果

2WikiMultiHopQA 提供官方 supporting passages，可以在固定 top-5 下检查证据是否被召回。

| 方法 | Recall@5 | Precision@5 |
| --- | ---: | ---: |
| BM25 | 0.6583 | 0.3106 |
| Dense | 0.6883 | 0.3228 |
| HippoRAG 2 | **0.7498** | **0.3544** |

LightMem 和 Mem0 返回的是生成式 memory，而不是原文 passage。它们与 gold passage 无法字符串精确匹配，因此这两个方法不报告 passage recall/precision，不把程序产生的零分解读为检索失败。

为确定 HippoRAG 2 的改善来自哪一环，我们进一步用官方 Recall@5 和 Qwen3.5-9B 的官方 answer F1 做同题比较：

- 在三个原文检索方法都找齐全部 gold supporting passages 的同一批 286 题上，BM25、Dense 和 HippoRAG 2 的 answer F1 分别为 0.768、0.783 和 0.778。当必要证据都已出现时，三者的答题结果非常接近。
- 在 HippoRAG 2 找齐全部 gold passages、但 BM25 或 Dense 至少一者没有找齐的同一批 183 题上，三者 answer F1 分别为 0.220、0.352 和 0.590。

这两个配对分解支持一个比“图结构让 evaluator 更会推理”更精确的结论：HippoRAG 2 在 2Wiki 上的主要优势是把同一问题需要的多个 passage 一起召回。当三种方法都已经找齐证据时，没有观察到 HippoRAG 2 额外提升 Evaluation Backbone 的答案产生。因此新算法应优先学习“联合召回必要证据集”，而不是把 evaluator 推理能力作为主要优化目标。

## 效率结果

下表汇总 6 个保留任务的完整 per-group wall-clock 记录。“检索时间”是所有 3,386 个问题的单题检索时间之和。这些数字用于比较方法开销，不是官方效果 metric。

| 方法 | Memory/index 构建时间 | 检索时间 | Generator 调用 | Generator tokens |
| --- | ---: | ---: | ---: | ---: |
| BM25 | **0.48 秒** | **38.49 秒** | **0** | **0** |
| Dense | 125.09 秒 | 765.51 秒 | **0** | **0** |
| LightMem | 10,821.23 秒 | 197.17 秒 | 3,482 | 4,833,992 |
| HippoRAG 2 | 8,212.09 秒 | 1,733.38 秒 | 28,724 | 20,617,866 |
| Mem0 | 97,783.49 秒 | 3,209.77 秒 | 14,362 | 139,416,826 |

表中的 Generator 调用和 tokens 只计算 memory/index 构建阶段。HippoRAG 2 的图结构确实改善了多跳证据覆盖，而不只是让答题模型更会猜；但它为这个改善付出了大量全语料 OpenIE 生成成本，query-time fact filtering 还额外使用 3,374 次调用和 10,120,992 tokens。Mem0 不在 retrieval 阶段调用 Generator，但顺序更新造成约 27.2 小时的累计构建时间，并消费 1.394 亿 tokens，是五个方法中构建成本最高的。

## 已经可确定的失败模式

### 1. 多跳证据不能只靠独立相似度

HippoRAG 2 在 2Wiki 上同时提高 Recall@5 和三个 evaluator 的 answer F1，并且在 MH-Doc QA 上也一致领先。这是直接证据：跨 passage 关联和联合证据选择是值得保留的 memory 能力。

### 2. 语义 embedding 不能替代词面定位

Dense 在 LoCoMo 和 2Wiki 上胜过 BM25，但在 SH-Doc QA 和 FactConsolidation-SH 上明显落后。这些任务的答案常依赖精确名称、数值或关系。因此新方法不应把稀疏检索整体替换掉，而应保留词面与语义两种寻址视图。

### 3. 有损生成式 memory 会丢失未来问题需要的细节

LightMem 在 FactConsolidation-SH 上有竞争力，但在 2Wiki 和 LoCoMo 上落后 BM25、Dense 和 HippoRAG 2。最终 Mem0 的 2Wiki answer F1 为 0.436 / 0.407 / 0.348，也在三个 evaluator 上都低于 Dense 和 HippoRAG 2；其中 423/1,000 题在三个 evaluator 上同时得到 0，另外四种方法对应数量为 BM25 403、Dense 379、LightMem 453、HippoRAG 2 335。生成的事实 memory 适合直接问答和持续更新，但一旦抽取或压缩删掉关系链中的细节，后续检索无法恢复原文。Mem0 在 LoCoMo 上领先却在 2Wiki 多跳上落后，说明“可更新的事实记忆”与“保留跨文档证据链”是两个不同能力，不能用一个替代另一个。

Mem0 2Wiki 构建的 6,119 次 extraction 中有 1 次返回了无法解析的 JSON。Mem0 官方代码路径将该次视为没有抽取出 memory 并继续运行；本轮没有增加 retry、修补 JSON 或重跑来掩盖该失败。最终 1,000 个 top-5 中有 3 题各包含一条完全重复的 memory 文本。这个重复比例很小，不足以解释整体分数差距，但它与单次解析失败一起说明生成式 memory 还存在可测量的抽取可靠性问题。

### 4. FactConsolidation-MH 是共同失败，不是单一 baseline 的 bug

五个方法在 FactConsolidation-MH 上的 Exact Match 几乎都在 0.00–0.08。同一评测程序和同一批 evaluator 在其他任务上能产生正常分数，因此没有证据表明这是 metric 或 evaluator 代码错误。

为区分 memory/retrieval 失败与 reader 失败，这里将每题的 top-5 evidence 拼接后，直接复用官方 Substring Exact Match 检查 gold answer 是否出现。这只是保守诊断，不作为新 benchmark metric；对生成式 memory，语义等价但文字不同的事实会被计为未出现。

| 方法 | 答案字符串在 top-5 中 | 三个 evaluator 全为 0 | 答案在 evidence 中但三者全错 | 答案不在 evidence 中且三者全错 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 24/100 | 91/100 | 15 | 76 |
| Dense | 24/100 | 95/100 | 19 | 76 |
| LightMem | 5/100 | 95/100 | 1 | 94 |
| HippoRAG 2 | 23/100 | 95/100 | 18 | 77 |
| Mem0 | 10/100 | 91/100 | 2 | 89 |

这个分解表明，首要问题确实属于 memory/retrieval：对大多数错题，固定 top-5 里根本没有可直接支持官方答案的文字。但 BM25、Dense 和 HippoRAG 2 也分别有 15、19 和 18 题在答案已明确出现时三个 evaluator 仍然全错，因此还有第二个、规模更小但可复现的 evidence organization/reader 问题。新方法应优先提高必要事实覆盖，然后在相同 top-5 中检验关系组织是否减少“证据已有但三者全错”。

这一模式不只存在于 FactConsolidation-MH。下表将同一个确定性诊断扩展到四个 MemoryAgentBench 任务；每格为“答案在 top-5 中且三个 evaluator 全错 / 答案不在 top-5 中且三个 evaluator 全错”。每个任务均有 100 题。

| 方法 | SH-Doc QA | MH-Doc QA | FactConsolidation-SH | FactConsolidation-MH |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 3 / 12 | 10 / 34 | 23 / 5 | 15 / 76 |
| Dense | 4 / 36 | 8 / 44 | 27 / 47 | 19 / 76 |
| LightMem | 4 / 39 | 7 / 47 | 21 / 14 | 1 / 94 |
| HippoRAG 2 | 3 / 5 | 15 / 20 | 26 / 22 | 18 / 77 |
| Mem0 | 3 / 9 | 5 / 35 | 17 / 2 | 2 / 89 |

SH-Doc QA 中 HippoRAG 2 和 Mem0 的答案文字分别出现在 94/100 和 90/100 个 top-5 中，而 Dense 和 LightMem 只有 61/100 和 59/100，与端到端得分差距一致。在 FactConsolidation-SH 中，Mem0 覆盖 98/100 个答案，但仍有 17 题在答案已出现时三个 evaluator 全错；这说明“召回到一个答案字符串”不等于“提供了足够清晰的事实关系”。

### 5. 更大的 Evaluation Backbone 通常更强，但不是每个 cell 都单调

9B 在 LoCoMo 和 2Wiki 上一般领先 4B/2B，但 SH-Doc QA 中偶尔出现 4B 略高于 9B。因此报告三个 evaluator 的完整结果比只选一个模型更稳妥，也能检查 memory 改进是否能跨答题模型成立。

### 6. LoCoMo 的普通问答与对抗拒答需要分开解读

下表是 Qwen3.5-9B 在 LoCoMo 官方五个类别上的 F1，不对 evaluator 或类别再做新的聚合。

| 方法 | Multi-hop | Temporal | Open-domain | Single-hop | Adversarial |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.166 | 0.374 | 0.116 | 0.463 | 0.841 |
| Dense | 0.294 | 0.397 | 0.136 | 0.479 | 0.812 |
| LightMem | 0.265 | 0.128 | 0.163 | 0.403 | **0.899** |
| HippoRAG 2 | **0.328** | 0.412 | 0.153 | 0.492 | 0.780 |
| Mem0 | 0.327 | **0.419** | **0.164** | **0.514** | 0.879 |

Single-hop 明显容易于 multi-hop 和 open-domain，进一步支持“必要事实覆盖与跨证据关系组织是主要瓶颈”的结论。Temporal 上 LightMem 的降幅尤其明显，表明生成式压缩不仅会丢失事实，还可能弱化事件与时间的绑定。

Adversarial 类别的官方规则不是将预测与题目中的诱饵事实做普通 token F1，而是检查模型是否回答 `No information available` 或 `not mentioned`。因此该列更多反映 Evaluation Backbone 能否在证据不充分时拒答，不能与前四类一样直接解读为 memory 召回越高越好。这也解释了为什么某些预测与诱饵答案字面完全相同却依然得 0：这是官方协议，不是 metric 或 adapter 故障。

## 新方法的直接优化目标

当前最值得保留的效果机制来自 HippoRAG 2，最值得保留的效率特性来自 BM25/Dense。因此不应简单对 Generator 或 Evaluation Backbone 做微调，而应将研究问题定义为：

> 如何在不对全语料进行高成本 LLM OpenIE 的情况下，保留 HippoRAG 的跨来源关联和多跳证据覆盖，同时保留 BM25/Dense 的原文忠实性和低成本。

当前有证据支持的框架是：

1. 始终保留原始 chunk/turn 作为可回答内容，避免 LightMem 式不可逆信息损失。
2. 同时保留词面和语义寻址，因为现有结果证明两者在不同任务上互补。
3. 将关系学习限制在候选 memory 的联合选择中，而不是先对整个 corpus 生成所有 triples。具体学习目标必须来自官方 training split 或已发布的 supervision，不用测试答案或临时启发式规则。
4. 最终仍在相同 top-5 预算内返回原始证据，用官方 retrieval/QA metric 和实测构建时间、检索延迟、LLM 调用及 tokens 同时验证效果与效率。

这一方向的关键不是“把 BM25、Dense 和 graph 拼起来”，而是将 HippoRAG 的多跳优势改造成一个只在查询需要的局部 memory 上运作的、可学习的 evidence-set selection 问题。最终配对分析已经表明 HippoRAG 在 2Wiki 的主要增益来自找齐必要 evidence set；下一步算法应在不破坏原文证据和持续更新能力的前提下，降低全语料 OpenIE 与顺序 LLM 更新的成本。

具体算法、训练目标、精确推断、先行工作边界、消融和否证标准见 [Joint Evidence Memory 提案](joint_evidence_memory_proposal.md)。该文档明确区分已经验证的 failure analysis 与尚未实现的方法假设。
