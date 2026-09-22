# 实验与消融结果

更新：2026-09-22。算法见 [algorithm.md](algorithm.md)，论文定位见 [story.md](story.md)。

## 1. 评测范围与读表方式

| 任务 | 题数 | 答案指标 |
|---|---:|---|
| SH-Doc QA | 100 | Substring Exact Match |
| MH-Doc QA | 100 | Substring Exact Match |
| FactConsolidation-SH | 100 | Substring Exact Match |
| FactConsolidation-MH | 100 | Substring Exact Match |
| LoCoMo | 1986 | 原生类别评分后汇总的 F1 |
| 2WikiMultiHopQA | 1000 | Answer F1；另报 passage Recall@5 |

每条件共 3386 题。前四项为 MemoryAgentBench 既定发布配置；LoCoMo 使用 locomo10 的全部五类；2Wiki 使用 HippoRAG 发布的 1000 题子集及 6119 段语料，不是原始完整数据集。数据加载在 `dataset_loader/loader.py`，任务评分在 `experiments/runner.py` 和 `utils/*metrics.py`。

四个 reader 为 Qwen3.5-4B、Qwen3.5-9B、Gemma-3-4B-it、Llama-3.1-8B-Instruct。下表为百分制；不跨不同指标平均。胜/平/负按未舍入原生分数判断，不是回答正确率。

九项 baseline 配置：BM25、dense、HippoRAG 2、Mem0、LightMem、LightMem offline、AnchorMem dense、AnchorMem official、CatRAG。它们不是九个完全独立的算法。原生配置对照和添加相同上下文的补充对照分开报告。

**数据曾反复用于方法选择，seed=42。以下是本地 test-as-dev 结果，不是独立 held-out 验证、显著性结论或全领域 SOTA。** 四 reader 共用 one-shot 检索；这不是四次独立检索实验。当前整理没有改变数据、指标、预测、图或缓存。

## 2. 当前 One-shot 主结果

同一个 canonical 投影图、RRF、来源窗口及十事实配置用于四 reader。每格为“我们 / 九项原生 baseline 最佳”。

| 任务 | Qwen3.5-4B | Qwen3.5-9B | Gemma-3-4B | Llama-3.1-8B |
|---|---:|---:|---:|---:|
| SH-Doc QA | 93 / 85 | 92 / 88 | 89 / 84 | 87 / 89 |
| MH-Doc QA | 59 / 55 | 63 / 62 | 54 / 50 | 60 / 56 |
| FactConsolidation-SH | 74 / 55 | 81 / 66 | 68 / 60 | 39 / 34 |
| FactConsolidation-MH | 15 / 5 | 17 / 5 | 11 / 6 | 5 / 2 |
| LoCoMo | 49.74 / 50.55 | 56.04 / 54.65 | 38.41 / 40.87 | 49.46 / 48.00 |
| 2WikiMultiHopQA | 59.34 / 49.81 | 59.11 / 52.35 | 49.19 / 44.35 | 50.63 / 42.07 |
| 严格胜出任务数 | 5/6 | 6/6 | 5/6 | 5/6 |

相对旧完整附录版本，四 reader 的 LoCoMo 都下降：Qwen4 50.15 -> 49.74，Qwen9 57.62 -> 56.04，Gemma 40.00 -> 38.41，Llama 49.57 -> 49.46。不按任务更换版本。

Llama 的同组件补充对照中，BM25 加事实补充后 FC-SH 为 42，高于我们的 39；HippoRAG 2 加事实补充后 SH 为 88，高于我们的 87。对这两项增强对照的逐任务最佳，我们为 4 胜 2 负。不能将原生主表的 5/6 解读成对所有增强版本都领先。

### QA 输入成本

以下是每 reader 全部 3386 题的 QA token 总数，不含离线抽取、构图和在线检索成本；不同 tokenizer 的绝对数不横向比较。

| Reader | 旧版输入 tokens | 新版输入 tokens | 输入减少 | 旧版输出 tokens | 新版输出 tokens |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 11283658 | 9136162 | 19.03% | 23745 | 22501 |
| Qwen3.5-9B | 11283658 | 9136162 | 19.03% | 28677 | 27849 |
| Gemma-3-4B | 11102210 | 9072131 | 18.29% | 19781 | 20148 |
| Llama-3.1-8B | 10491829 | 8172137 | 22.11% | 32081 | 32648 |

普通 `QueryFactContext` 接口已在全部 3386 题、15 个来源组精确重放检索结果与上下文，零新增编码/识别调用。这是实现重现检查，不是额外效果实验。

## 3. 当前 IRCoT 主结果

复用同环境的固定轨迹，比较 BM25 与图 + BM25；十事实及相邻原文只加入最终 QA，不重新生成中间推理。1/3/5 是最大轮数，允许原停止规则提前结束，不逐任务挑轮数。

### 对原生 IRCoT + BM25

| Reader | 1 轮 | 3 轮 | 5 轮 |
|---|---|---|---|
| Qwen3.5-4B | 6 / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 |
| Qwen3.5-9B | 6 / 0 / 0 | 5 / 0 / 1 | 4 / 1 / 1 |
| Gemma-3-4B | 6 / 0 / 0 | 3 / 1 / 2 | 3 / 1 / 2 |
| Llama-3.1-8B | 6 / 0 / 0 | 5 / 0 / 1 | 5 / 0 / 1 |

合计 61 胜、3 平、8 负，共 72 个 reader/任务/轮数条件。Qwen9/Llama 的 2Wiki 三轮和五轮仍落后；Gemma 三轮/五轮的 LoCoMo 与 2Wiki 仍落后。

![四个 reader、六任务、完整 1/3/5 轮曲线](figures/ircot_context_curves.png)

[矢量图](figures/ircot_context_curves.pdf)。曲线同时保留 BM25、图原文、旧完整事实附录和当前相关事实，不拼接不同配置。

### 对同上下文组件的 BM25

BM25 使用自身轨迹，补同一保留事实库、十事实选择、相邻原文及最终 QA 协议。

| Reader | 1 轮 | 3 轮 | 5 轮 |
|---|---|---|---|
| Qwen4 | 6 / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 |
| Qwen9 | 5 / 0 / 1 | 5 / 0 / 1 | 5 / 0 / 1 |
| Gemma | 4 / 0 / 2 | 3 / 1 / 2 | 4 / 0 / 2 |
| Llama | 5 / 0 / 1 | 5 / 0 / 1 | 5 / 0 / 1 |

合计 59 胜、1 平、12 负。这说明差异不只是 baseline 没有附加上下文；它不是只改变图拓扑的对照，也不保证 token 相同。三轮总 QA 输入如下：

| Reader | 原生 BM25 | 同上下文 BM25 | 我们 |
|---|---:|---:|---:|
| Qwen4 | 6217525 | 17195131 | 17798720 |
| Qwen9 | 6375742 | 17502653 | 18135638 |
| Gemma | 5844631 | 16467131 | 17004248 |
| Llama | 5779040 | 14772286 | 15518335 |

当前增强上下文的输入约为图原文版本的 2.4--2.6 倍，不能声称无额外成本或端到端压缩。

### 未采用的每轮在线增强

另测过 Llama 在每轮及最终 QA 加入旧完整事实附录，重新生成轨迹。对 BM25 的 1/3/5 轮分别为 4/3/4 项胜出，不优于当前统一配置。三轮 FC-SH 为 17、FC-MH 为 0、2Wiki 为 40.87；当前最终 QA 十事实配置分别为 40、9、52.46。在线版本 LoCoMo 为 50.89，当前为 49.93，存在局部收益。

该负结果不代表所有在线增强无效，也不能被当前固定轨迹结果覆盖。产物在下方目录索引中保留。

## 4. 消融：完成范围

四 reader、六任务、one-shot 与 IRCoT 1/3/5 的计划条件已经完成。汇总共 96 个 reader/任务/流程单元、672 条条件记录，缺项为 0；包括复用条件，不能称为 672 次新运行。

完整矩阵及成本：
- [ablation_summary.json](/oscar/scratch/zliu328/agent-memory-outputs/optimization_method_analysis_seed42_20260921/ablation_summary.json)
- [全部条件表](/oscar/scratch/zliu328/agent-memory-outputs/optimization_method_analysis_seed42_20260921/ablation_summary.md)

| 对照 | 固定内容 | 改变内容 |
|---|---|---|
| 原图替换 | 候选索引、PPR、BM25、上下文政策、reader | 整个构图模块的连接及权重 |
| 图/候选筛选四格 | 投影与下游政策 | 两侧分别采用全部事实支持或筛选支持 |
| 无 RRF | 图、事实候选、上下文政策 | 删除 BM25 排名融合 |
| 原文/事实/邻居四格 | 检索来源、顺序、prompt、reader | 最终 QA 展示内容 |
| IRCoT 同上下文 BM25 | 各自已有轨迹、统一增强规则与 reader | 比较不同检索流程选出的材料 |

构图替换会改变取回的文本；上下文消融才固定同一份来源。不要混用两种因果归因。

### 主要结果

#### 按 LoCoMo 官方题型拆分

以下为原生 LightMem 对照与完整方法，不是同组件消融。每格为 LightMem / 我们，所有题均保留，不改变官方类别或评分：

| Reader | 多跳，282 题 | 时间，321 题 | 开放域，96 题 | 单跳，841 题 | 对抗/拒答，446 题 |
|---|---:|---:|---:|---:|---:|
| Qwen4 | 24.61 / 36.67 | 9.21 / 34.43 | 14.97 / 13.60 | 39.04 / 66.22 | 83.18 / 45.74 |
| Qwen9 | 27.47 / 36.71 | 12.55 / 39.20 | 14.97 / 16.44 | 40.15 / 66.67 | 90.13 / 68.83 |
| Gemma | 23.66 / 31.60 | 8.55 / 34.04 | 8.83 / 7.50 | 36.06 / 56.98 | 71.30 / 17.49 |
| Llama | 24.54 / 40.32 | 14.63 / 35.75 | 16.51 / 16.15 | 38.23 / 65.44 | 79.15 / 42.15 |

四个 reader 的单跳、时间和多跳均提高，拒答题均下降。这支持具体的收益和边界，不能概括成改善全部长期记忆能力。拒答分数仍采用既有原生规则，不把标注的干扰答案当作正确答案。此表本身不隔离组件作用；下方固定检索的相邻对话消融进一步检验了其中一个来源。

#### 固定检索，只移除事实附录

固定来源检索、顺序、原生 prompt 和解码，只删除相关事实附录：

| FC-SH，全部 100 题 | 完整方法 | 无事实附录 |
|---|---:|---:|
| Qwen4 | 74 | 49 |
| Qwen9 | 81 | 48 |
| Llama | 39 | 31 |
| Gemma | 68 | 49 |

该任务所有问题的相邻记录集合为空，因此此处没有同时删除额外原文。事实选择不重新计算；这是事实展示的介入，而非新检索或新构图。输入长度会改变，不能进一步声称排除了 token 数量或展示位置的作用。完整六任务及多轮结果均保留于 `ablation_summary.json`，此处只展示 FC-SH。

#### 固定当前上下文政策，取消图和候选两侧筛选

FC-SH 的完整方法 / graph_all_index_all：Qwen4 为 74 / 69，Qwen9 为 81 / 81，Gemma 为 68 / 71，Llama 为 39 / 39。四组均为完整 100 题。该对照保持投影、事实附录和其他组件，只改变图及识别候选采用全部支持还是筛选后支持。

这组联合介入不是全面有益的证据，也不说明单个筛选无作用；其余两格和六任务结果现已齐全。它限制了写作：不能声称两侧共同筛选对每个 reader 都必要，更不能把 FC-SH 的完整提升全部解释为筛掉旧事实。

#### 当前构图与原图：四个 reader 的完整任务结果

下面保持保留事实候选、PPR、BM25 融合、上下文组织政策、reader 和评分不变；仅换回原图。实际检索结果及其附录会随图改变，并非固定同一份 QA 输入。

| Reader | 2Wiki 原图 | 2Wiki 当前图 | FC-MH 原图 | FC-MH 当前图 |
|---|---:|---:|---:|---:|
| Qwen4 | 49.86 | 59.34 | 6 | 15 |
| Qwen9 | 49.72 | 59.11 | 8 | 17 |
| Gemma | 40.32 | 49.19 | 5 | 11 |
| Llama | 43.56 | 50.63 | 3 | 5 |

2Wiki 为全部 1000 题 answer F1，FC-MH 为全部 100 题 SubEM。四 reader 的 2Wiki 提升为 7.07–9.47 个点；相同检索经不同 reader 得到一致方向的答案变化。该对照是构图模块的受控介入，包含完整事实与出处的连接、相似边处理和相应边权，不能把整组效果单独归因于投影公式或任一边类型。共享检索 Recall@5 从 74.825% 到 87.425%，全部支持段落齐全的题数从 468 到 691。它与原生 HippoRAG 2 的 74.975% / 469 题是两个不同控制。

相反，在相同投影下取消两侧支持筛选，2Wiki 相对完整方法的差值只有 -0.35、+0.23、-0.01、-0.04 个点（依次 Qwen4/Qwen9/Gemma/Llama），LoCoMo 四个 reader 均略高于完整方法；FC-MH 则由 15/17/11/5 降至 6/11/9/4。筛选收益明显依赖任务，不能把所有图收益归于筛选。无 BM25 融合也存在取舍：Llama FC-SH 从 39 提高到 51，但 SH-Doc 从 87 降至 84。完整六任务结果保留，不逐任务重新选配置。

#### 相邻对话的收益和拒答代价

固定同一份来源检索、不提供事实附录，只增加既有相邻对话。LoCoMo 全部原生单跳题（841 题）和对抗/拒答题（446 题）的结果如下：

| Reader | 单跳：来源 / 加相邻对话 | 拒答：来源 / 加相邻对话 |
|---|---:|---:|
| Qwen4 | 51.97 / 67.29 | 62.78 / 42.38 |
| Qwen9 | 52.64 / 67.21 | 70.85 / 66.14 |
| Gemma | 44.15 / 55.26 | 41.03 / 18.83 |
| Llama | 50.88 / 65.61 | 57.17 / 39.24 |

这比完整方法对 LightMem 的观察更接近组件归因：来源和 reader 不变，展示相邻记录会同时产生收益和代价。仍不能拆分新增语义内容、token 长度与位置的独立影响，也不能声称所有拒答错误都来自这个组件。

#### BM25 获得相同上下文增强后的 IRCoT

BM25 复用自身原始轨迹，与我们采用完全相同的保留事实库、十事实选择、相邻原文及最终 QA 协议；没有改变其中任一方法的中间推理。四 reader、六任务、1/3/5 轮全部 72 个条件中，我们为 59 胜、1 平、12 负；原生 BM25 对照是 61 胜、3 平、8 负。完整矩阵和 token 量见 本文件第 3 节。这排除了“只是对方没有同一个上下文组件”的解释，但不意味着相同 token 预算或只改变拓扑，也不证明每个多轮任务改善。Qwen9/Llama 的 2Wiki 在三轮、五轮依然落后。

## 5. 定性分析

全量配对 `failures/summary.json` 包含 288 个任务/reader/流程/baseline 对照，`paired_scores.jsonl` 保存逐题分数，`case_index.json` 索引 278 个实际输入案例。案例按原题序取各结果类别的首例，不是新的评测子集，也不是总体发生率估计。

原生 HippoRAG 2 与我们在 2Wiki 分别有 469 和 691 题找齐支持段落，Recall@5 为 74.975% 和 87.425%。这是原生对照，勿与第 4 节保留我们候选索引的原图控制 468 题 / 74.825% 混淆。

### 案例与干预结果

### 找到更新记录后，如何呈现仍然重要

`case_015.json`，Qwen4，FC-SH，题目 `factconsolidation_sh_262k_no0`：询问 Adam Putnam 的大学。以任务提供的记录为准，后续记录给出的答案是 Federal University of Rio de Janeiro。

HippoRAG 2 返回包含 University of Florida 的早先记录，回答 Florida。我们的来源检索同时返回旧记录和后续的 Rio de Janeiro 记录；完整方法回答 Rio de Janeiro，而在相同来源上删除事实附录后，仍回答 Florida。

这给出两个可分开的观察：完整检索找到了 baseline 未返回的后续材料；相同来源下，事实展示也改变了答案。不能将成功简写为“图删除了旧事实”：实际事实附录仍包含两个大学，关系字符串分别为 educated at 与 was educated at。语义归一和最终判断并非绝对可靠。此次四格对照中，只有两侧均筛选的格子答对；但保留候选和上下文、换回原图也答对。因此该例不能证明新的图结构独自必要。

### 事实附录也可能强化抽取遗漏

`case_017.json`，Qwen4，FC-SH，题目 `factconsolidation_sh_262k_no2`：原始同一段落依次含 Sigur Ros 来自 Iceland 和 Israel 两条记录，任务答案为 Israel。

实际 OpenIE 对这个出处只抽取了 Iceland；保留事实和最终附录也只含 Iceland。两种方法都返回了这段原文，但 HippoRAG 2 回答 Israel，完整方法回答 Iceland。固定我们的检索，仅删除事实附录后，答案恢复为 Israel。

这是可核查的局部负例：错误并非未检索到正确原文；不完整的抽取表示和事实展示共同构成需要检查的失败路径。不能由此宣称所有失败都由抽取造成，也不能把图说成可以修复已经缺失的事实。

### 多跳支持材料改善与 reader 失败并存

`case_040.json`，Qwen4，2Wiki，题目 `a1cdb240085811ebbd5bac1f6bf848b6`：比较两部影片导演的年龄。

HippoRAG 2 的五段原文缺 Michael Curtiz 的出生信息，覆盖 3/4 标注段落；我们返回该导演原文，覆盖 4/4，并回答正确。已有 gold-only 条件在这题仍答错。当前受控结果进一步分清了两个环节：

| 条件 | 标注段落覆盖 | Qwen4 答案得分 |
|---|---:|---:|
| 完整方法 | 4/4 | 1 |
| 换回原图，其他政策不变 | 3/4 | 0 |
| 保留同一来源，只删除事实附录 | 4/4 | 0 |
| 同一投影，取消两侧筛选 | 4/4 | 1 |
| 删除 BM25 融合 | 4/4 | 1 |

该题在新构图下找齐证据；固定这些证据，事实展示又影响最终回答。它是两个具体介入均影响结果的正例，而不是“找齐就必然答对”或“筛选总是必要”的例子。

### FC-MH 答案得分提高不等于验证更新机制

`case_024.json`，Qwen4，`factconsolidation_mh_262k_no1`：询问 Karl Lueger 所属国家的宗教，发布标注是 atheism。我们的原文同时包含第 1219 条 Austria-Hungary 国籍、第 9973 条 Lordship of Ireland 国籍，以及第 11423 条 Austria-Hungary 与 atheism 的关系；事实附录也保留两个国籍。HippoRAG 2 回答 Lordship of Ireland，原生得分为 0；我们回答 atheism，得分为 1。

这个例子确实找到了组成标注答案的两条事实，但较晚的国籍记录并不是 Austria-Hungary。因此，单凭本题得分不能证明系统遵循了更新规则，更不能用它说明旧事实已被删除。此处保留发布标注和原生分数，不重标、不剔除；它只用于说明答案指标和具体机制需要分别验证。[官方任务及指标说明](https://github.com/HUST-AI-HYZ/MemoryAgentBench#-clarification-on-evaluation-metrics)

### Memory baseline：检索到了相关词，但不是对应事件

`case_036.json`，Qwen4，LoCoMo，`conv-26-3`：问题是 Caroline 研究了什么。LightMem 返回“之后会做研究”和“正在探索职业选项”等记忆，回答 career options；我们返回明确记载 Researching adoption agencies 的原始对话并答对。

这是实际返回内容的差异，不足以断言 LightMem 在构建时删除了 adoption 信息：尚未检查其整个记忆库。该例可以说明查询词相关不等于事件相关，不能用它攻击所有摘要式 memory 的信息完整性。当前全部构图和上下文消融条件在这题仍答对，故它不是某个新模块必要性的案例。

`case_014.json`，同 reader 的 MH-Doc QA，`ruler_qa2_421K_no6`：LightMem 返回其他人物和 consultant 的零散记忆，没有 Aladin 的对应人物；我们返回 Eenasul Fateh 的原文，以及 management consulting 的解释，答出名字。它说明本次检索支持的差异，不是对所有抽取式或压缩式 memory 的普遍反例。

### 我们也会遗漏多条历史事件

`case_037.json`，Qwen4，LoCoMo，`conv-26-18`：问题询问 Melanie 去过哪些地方露营。LightMem 返回山地和森林两类记忆，分数 2/3；我们的五个中心集中在森林露营及无关对话，回答只有 forest，分数 1/3。当前介入中，取消两侧筛选恢复 mountains, forest，得分 2/3；仅取消任一侧仍是 forest，得分 1/3。它支持联合筛选在此题有代价，但还没有把候选变化、PPR 起点、材料选择的中间效应单独分开，不能据此声称单一“最新事实”规则解释全部历史事件遗漏。

### 多轮答对不代表轨迹找齐了证据

`case_210.json`，Gemma，IRCoT 三轮，2Wiki `dcbee4b608b011ebbd85ac1f6bf848b6`：比较两部影片导演的去世先后。BM25 与图轨迹都没有找齐两名导演的标注材料；BM25 最终答对，而图方案和已有 gold-only 都答错。

逐轮缓存为 `optimization_ircot_hybrid_seed42_20260918/main/google_gemma-3-4b-it/traces/{optimized_graph,bm25}/2WikiMultiHopQA/hipporag-2wikimultihopqa.json`。图轨迹第三轮累计的 15 个 source index 与本例最终输入的 source_position 顺序完全一致。首轮两篇影片原文已说明导演分别是 Clarence Brown 和 Robert A. Stemmle，但混入的 John G. Adolfi 传记被模型用作推理首句；第二轮实际查询是 John G. Adolfi died in 1933，第三轮变为 Karl Gilg died in 1981。它说明错误生成会改变后续检索目标，不能解释为图缺少影片与导演的全部关联。

BM25 轨迹首轮同样错误生成 Fritz Genschow 为导演，后续查询又转向 Fritz Hippler；最终独立 QA 却答对。因此本例既不能解释成“BM25 检索更完整”，也不能证明最终答对来自正确的逐轮推理。这里只记录可见轨迹；未做替换中间查询的介入，不能给出这种偏移的总体发生率或独立因果效应。QA 结果、支持证据覆盖和轨迹行为分别报告。

## 6. 产物位置

以下路径相对于 `/oscar/scratch/zliu328/agent-memory-outputs/`。机器可读报告和逐题预测是完整结果来源，本文件不再保留过期排队日志。

| 内容 | 路径 |
|---|---|
| 原始 OpenIE、embedding、HippoRAG 图和基础 baseline | `final_qwen3_30b_seed42_clean_20260910/` |
| 当前关系归一缓存 | `optimization_canonical_schema_seed42_20260913/` |
| 当前构造图和保留候选索引 | `optimization_retained_fact_index_seed42_20260914/statement_projection_loop_free_retained_index_rrf_window/` |
| Llama 当前 one-shot | `optimization_fact_context_seed42_20260920/source_and_fact_context/main/meta-llama_Llama-3.1-8B-Instruct/` |
| 其他三 reader 当前 one-shot | `optimization_source_fact_transfer_seed42_20260920/main/<reader>/comparison.json` |
| 三 reader 同环境原生 baseline | `optimization_native_readers_seed42_20260920/` |
| Llama 原生 baseline 与旧完整版本 | `optimization_context_reranking_seed42_20260920/` 中原生条件，见各 comparison.json |
| IRCoT 当前 QA 主结果 | `optimization_ircot_fact_context_seed42_20260920/main/<reader>/comparison.json` |
| IRCoT 冻结检索轨迹 | `optimization_ircot_hybrid_seed42_20260918/main/<reader>/traces/` |
| IRCoT 旧附录在线增强 | `optimization_ircot_online_context_seed42_20260920/main/meta-llama_Llama-3.1-8B-Instruct/` |
| 完整机制消融、同上下文 BM25、案例 | `optimization_method_analysis_seed42_20260921/` |
| 当前消融运行代码归档 | 上一目录的 `report/code_6562268.zip` |
| Gold-only 诊断 | `optimization_gold_evidence_seed42_20260919/` |
| 历史尝试、旧文档及本次删除源码 | `code_cleanup_20260922/legacy.zip` |

baseline 的具体检索缓存路径还可查 `optimization/report_results.py:BASELINES`。历史事实图、查询模式搜索、reranker、压缩等正负结果仍在原 scratch 目录；没有为了让表格好看而删除失败预测。

### Peer 共享目录

入口：`/oscar/scratch/zliu328/agent-memory-share-zli532-20260919`，对 Oscar 用户只读，不仅限于 zli532。目录为 `baselines/`、`our_graph/`、`one_shot/`、`ircot/`；基础数据多为链接，不是独立可搬走的数据包。修改 SQLite/Qdrant/图前，应复制到接收者自己的 scratch。

**这是 9 月 19 日的旧事实图快照，不是当前十事实主方法。** 其中 our graph 为 `fact_graph_without_synonyms`；其旧 `load_fact_memory` 实现本次移出活动代码，可从上述 legacy.zip 或 Git 提交 `4ede17a91ac633d3b99feeedafbb990ad35145b5` 恢复。此次没有更新共享包或改动它依赖的数据，不将该包分数冒充最新结果。

## 7. 本次代码清理

删除 9 个已退役文件：

- `optimization/retriever/fact_incidence.py`
- `optimization/retriever/fact_join.py`
- `optimization/retriever/query_pattern.py`
- `optimization/retriever/reranker.py`
- `optimization/graph_construction/fact_index.py`
- `experiments/ablate_recognition.py`
- `experiments/run_fact_ircot.py`
- `experiments/report_fact_graph.py`
- `experiments/evaluate_gold_evidence.py`

同步移除这些分支的专用测试；保留当前图、RRF、来源上下文、错误处理、结果报告及完整缓存核对测试。baseline 的既有修改、未提交适配器和 launcher 未删，第三方子模块及其文档未改。项目自有文档仅保留 algorithm.md、results.md、story.md。

清理前完整源码/文档小归档已通过 ZIP 完整性检查，未复制模型、环境、大缓存或原始结果。清理只改变活动代码和文档，不代表新增效果实验。

清理后验证：35 项单元测试通过，3 项需要显式配置完整缓存的测试跳过；48 个自有 Python 文件通过语法解析，7 个 shell/Slurm 文件通过 bash 语法检查。构图、one-shot 和 IRCoT 的 CLI 帮助入口均正常；21 个文档本地链接有效；没有遗留对已删除模块的活动代码引用。未重跑 GPU QA。
