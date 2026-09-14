# 固定候选窗口的完整对照

日期：2026-09-13。补齐已经实现的 `RANK_WINDOW = 20` 两候选实验，不新增扫描点。
此前只有 FCSH 的 100 题检索落盘，本轮复用该结果，补齐其他五任务与所有 QA。

## 改了什么

候选为 `original_rrf20_sentence_facts` 和 `canonical_rrf20_sentence_facts`。
分别使用已有原图和 canonical 图，图本身不重新构建或改变权重。
每次取图排名与 BM25 各前 20 个候选，调用现有 RRF，再保留五个来源。
RRF 常数仍为 60，来源窗口、去重与逐行事实读出都复用既有代码和冻结产物。
不新增 query LLM 调用、重排模型、QA prompt 或任务路由。

代码：[retrieve_rank_window.py](retrieve_rank_window.py)、
[rank_window.py](retriever/rank_window.py)。RRF 依据仍是
[原方法](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)，20 是本地固定候选预算，不是论文保证的最优值。

## 直接参照与解释边界

- `canonical_rrf20_sentence_facts` 对照已有 `canonical_rrf_sentence_facts`：
  相同图、融合规则和读出规则，候选窗口从 5 改为 20。
- 本轮的 original 与 canonical 两组互为图对照：相同的 20 候选和相同的 canonical 来源事实读出。
- 旧 `original_graph_rrf_window` 没有使用当前相同的去重和逐行事实格式，
  不能将它与本轮 original 的全部差值归因于候选窗口。

待检验的问题是：扩大融合前的候选覆盖，是否能保留两种检索中排名稍后但一致相关的来源？
这是检索预算组件，不是新的离线图表示。更大的候选池也可能使词面偶合影响最终排名，
不能提前断言更多候选必然提高召回或 QA。
最终五来源及固定窗口规则不保证 token 相同，逐题实际用量照常记录。

## 执行范围

两候选各六任务全部 3386 题、三个 reader，共 20316 次 QA。
不改加载、切分、评分和既有解码，test 继续作为开发集。
构建先冻结当前任务所有语料图与来源表示，随后才查询；不读取 gold answers 来建立索引。

每题检查原图或 canonical 图返回的前五名与既有原始排名一致，
以排除扩大结果返回数量时意外改变了原检索行为。
候选检索完成后，还通过普通 `load_optimized_memory` 接口完整复核该任务的两组结果，
只有验证成功才放行三个 reader 的 QA。验证不使用保存的 query resets。
复用 recognition 缓存；任何生成器 cache miss 都报错，不接受错误被吞掉后的 dense fallback。

作业清单：[rank_window_jobs.json](https://github.com/Sizchode/agent-memory/blob/f2121c11cc6d0c36bf931674152db0ca07669573/optimization/rank_window_jobs.json)。
输出目录：`/oscar/scratch/zliu328/agent-memory-outputs/optimization_rank_window_seed42_20260913`。
完整结果最终由 6342884 汇总，两个候选必须一起传给报告程序。

## 当前状态

17:33 EDT：FCSH 普通接口校验已完成，两组均为 100/100，三个 reader QA 已启动。
SH、MH、FCMH 的新检索已正常完成；LoCoMo 与 2Wiki 正在检索。
这不是完整 QA 结果，也不根据短任务先出的分数取消其余任务。
完整分数、收益和退步将在 [record.md](record.md) 继续记录。

17:37 EDT 阶段结果：FCSH 三 reader 的两候选 QA 全部完成，每项均为原 100 题。
报告程序已核对完整题目集合、逐题分数与 summary 一致，并保留另外五任务的未完成状态。

| reader | original 20 | canonical 20 | 既有 canonical 5 |
| --- | ---: | ---: | ---: |
| 9B | 58 | 55 | 63 |
| 4B | 56 | 63 | 64 |
| 2B | 61 | 62 | 61 |

相对直接预算参照，canonical 20 的 9B FCSH 明显退步，2B 略有提高，尚不支持普遍收益。
不据此取消剩余任务，不将不同硬件下的单次解码结果解释为已证明的因果机制。

### 2Wiki 支持段落检索

17:48 EDT：两候选的完整 1000 题检索已完成。使用现有
`utils.hipporag_metrics.gold_passage_recall_at_k` 计算逐题 Recall@5 后取均值，
不新增指标或筛选题目。四组题目 ID 唯一且完整，整个 case 对象（含 gold passages）一致。
每题均为五个原始来源，读取 `original_source_text`，不把事实附录当成 gold passage；
四组在本任务的 `window_sources` 均为空，没有额外窗口来源。

| 检索配置 | Recall@5 (%) |
| --- | ---: |
| 原 HippoRAG 2 | 74.975 |
| canonical RRF 5 + sentence facts | 85.800 |
| original RRF 20 + sentence facts | 73.500 |
| canonical RRF 20 + sentence facts | 80.150 |

同一 canonical 图下扩大窗口使支持段落召回下降 5.65 个百分点，
不支持“更多融合候选就能改善证据覆盖”的动机。固定窗口 20 时 canonical 图比原图高
6.65 个百分点，但这一结果只涉及 2Wiki 的检索，不能代替完整 QA 或六任务结论。
这里的检索差异不经过 reader 生成；具体为何改排仍需逐题分析，不能直接归因于词面噪声。
数据分别保存在原基线、`optimization_context_packing_seed42_20260913` 和本实验目录的
对应候选 `2WikiMultiHopQA/retrieval.jsonl` 中。

逐题比较：169 题 Recall@5 下降、20 题上升、811 题不变；按题累计丢失 189 个
原已召回的 gold passage，新增 25 个。以下是文件顺序中的首个退步与首个改善例，
用于展示已发生的排名变化，不代替全量统计，也不用于添加题目规则。

- 退步题 `83bf3b5a0bd911eba7f7acde48001122` 问 Lothair II 母亲的去世时间。
  `Ermengarde of Tours` 原图排名第 3，原窗口融合后保留；扩大窗口后被挤出最终五条。
  新进入的非 gold 来源包括 `Bertha, daughter of Lothair II`（graph 4 / lexical 6）
  与 `Teutberga`（graph 6 / lexical 5），该题 gold 覆盖从 2/2 降到 1/2。
- 改善题 `076288460bde11eba7f7acde48001122` 问 Henry I of Ziebice 父亲的出生日期。
  gold 来源 `Nicholas the Small` 在 graph 13 / lexical 10，扩大窗口后进入最终五条，
  该题覆盖从 1/2 提高到 2/2。

这两例显示固定 RRF 对两路共同出现的候选会产生不同取舍；不能概括成所有新增候选都是噪声，
也不能据此宣称需要新增查询路由。现阶段保留旧窗口作为较强参照，等待完整 QA。

17:49 EDT：四个 MAB 任务的两候选、三 reader 均已完成并通过报告程序审计。
以下仅为四任务结果，LoCoMo 与 2Wiki QA 尚未完成。

| 候选 / reader | SH | MH | FCSH | FCMH |
| --- | ---: | ---: | ---: | ---: |
| original 20 / 9B | 89 | 56 | 58 | 6 |
| original 20 / 4B | 89 | 54 | 56 | 5 |
| original 20 / 2B | 80 | 49 | 61 | 4 |
| canonical 20 / 9B | 89 | 58 | 55 | 6 |
| canonical 20 / 4B | 90 | 57 | 63 | 8 |
| canonical 20 / 2B | 76 | 47 | 62 | 7 |

18:06 EDT：2B 已完成两候选全部六任务，均为每候选 3386 题、无缺失 usage。

| 2B 配置 | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original 20 | 80 | 49 | 61 | 4 | 43.72 | 33.76 | 2/6 |
| canonical 20 | 76 | 47 | 62 | 7 | 42.86 | 37.73 | 2/6 |
| canonical 5 直接参照 | 79 | 51 | 61 | 8 | 43.64 | 42.15 | 3/6 |

窗口 20 没有改善 2B 的完整任务覆盖，不替换旧配置。它的 canonical 2Wiki 支持召回与
QA 都低于窗口 5，但这仅是同向变化，不是逐题证明 QA 下降全部由召回下降导致。
original 20 / 4B 也已全量完成（1/6 胜出）；余下结果仍在运行，等待完整两候选三 reader 矩阵。

18:13 EDT：original 20 已完成三 reader 全量 QA，严格胜出为 2/6、1/6、2/6。
每个 reader 均为 3386 题，输入 9412110 token；完整向量及取舍已更新 [record.md](record.md)。
canonical 20 的 4B 2Wiki 已完成为 52.87，但 LoCoMo 尚未完成，不补齐或猜测剩余成绩。

## 完整结论

6342884 已 COMPLETED 0:0。两候选、六任务、三 reader 全部 20316 次 QA 完成，
报告核对完整题目集合、逐题分数与 summary、全部九个 baseline 和 QA usage。

| 候选 | 9B 严格胜出 | 4B 严格胜出 | 2B 严格胜出 |
| --- | ---: | ---: | ---: |
| original 20 | 2/6 | 1/6 | 2/6 |
| canonical 20 | 3/6 | 6/6 | 2/6 |
| canonical 5 直接参照 | 5/6 | 6/6 | 3/6 |

canonical 20 的 9B 六任务均低于直接参照；4B FCMH 和 LoCoMo 略升，其余下降；
2B 仅 FCSH 略升。未达到三个 reader 的共同覆盖目标，不替换旧配置，不继续扫描窗口。
固定窗口 20 的原图/canonical 图差异仍是有效图对照，但不能把扩大窗口本身写成构图改进。
完整逐任务矩阵、成本和分析保存在 scratch 报告及 [record.md](record.md) 第 12 节。
