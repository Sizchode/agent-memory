# 方法尝试记录

更新：2026-09-14。记录 `optimization/` 图优化阶段实际做过的尝试，包括负结果、未完成及无效运行。
本文按方法整理；逐次作业历史见 [progress.md](https://github.com/Sizchode/agent-memory/blob/c443de4790e619d43c5bd9dcce9ccea707da56bc/optimization/progress.md)，文献对照见
[related_work_analysis.md](related_work_analysis.md)。不把事后机制解释当作已证明的理论。

2026-09-14 仓库清理：24 个历史作业 JSON 和 3 个 sbatch 取消 Git 跟踪，本地副本保留；
算法、评测结果与 scratch 产物不变。下文历史 JSON 文件名对应
[清理前版本](https://github.com/Sizchode/agent-memory/tree/f2121c11cc6d0c36bf931674152db0ca07669573/optimization)，
不再表示当前 Git 版本跟踪的文件。

## 口径与当前结论

当前默认为 `statement_projection_loop_free_retained_index_rrf_window`，六任务、9B/4B
均为 6/6 严格胜出。原候选索引四格、候选索引消融、旧 refinement 图同索引对照均已完成；
获胜配置已独立重建并通过全部 3386 题普通接口验证。以下发布条目与作业状态保留时间顺序，
最终配置选择见“单独筛选事实候选索引”节。

### 本次发布整理

2026-09-14 早期发布：提交无自环构图及 retained-fact-index 独立实验，当时默认仍为已完整验证的
`statement_projection_loop_free_refined_rrf_window`，未根据部分结果提前切换。
retained-index 的 2Wiki 已完成：9B answer F1 57.7539、4B 56.3973；9B LoCoMo
6387723_0 随后完成，F1 57.5679，9B 六任务完整分数已全部超过本地基线阈值。
随后 4B LoCoMo 与最终报告 6387725 完成，统一候选已达到双 6/6；完整表与限制见下文。
四项 MAB 数值与完整调用成本见下文该实验节。

本次移除 Git 中九份已归档的退役 MD/TXT，共 1367 行；归档中保留原内容。
raw-relation、删直连、显式事实节点及旧自环投影的独立实验开关已退役，原始图两格
仍是必要对照，不作为 legacy 删除。当前候选、永久回归测试、全部缓存和结果保留。
这次同时增加了新构图和必要运行代码，不将文档减少冒充算法代码压缩。

解除主入口对未跟踪旧实验 run_anchormem / run_gap_query_memory 的依赖：沿用原六任务
列表和原 QA 计量实现，不改变问题顺序、提示、生成或评分。提交所需的现有
baseline/graph_usage.py 与 utils/models.py 两行真实 token 计量，不提交其余 baseline 改动。
清理前完整源码位于 construction_experiments_20260914/implementation_before_publication_cleanup.tar，
已逐文件 tar --compare 通过。batch 6389057 的 28 项回归测试通过；git diff --check 通过。
从提交 592b6ce 导出干净目录后，batch 6389081 再次通过全部 28 项测试，以及 build_graph
和 GraphUsageRecorder 导入检查；不依赖工作树中未提交的旧实验脚本。
该检查不等于在新机器重跑六任务 GPU 实验；数据、模型与既有上游环境仍需按原协议准备。

- 当前按用户新要求只推进 **4B、9B 的全六任务消融与代码删减**；2B 不再参与当前目标，旧结果保留。
  以下三个 reader 的提分表及早期作业快照是历史记录，不代表仍继续优化 2B。
- 下文 `9B / 4B / 2B` 数字为分别严格超过九个完整 baseline 配置中每任务最高成绩的任务数，分母均为 6；平局不算胜出。
- 标记“完整”的候选覆盖 SH、MH、FCSH、FCMH 各 100 题、LoCoMo 1986 题、2Wiki 发布集 1000 题，三个 reader 共 10158 次 QA。
- 四个 MAB 任务为 substring exact match，LoCoMo 使用原类别评分，2Wiki 为 answer F1；表中的任务分数均为百分数。
- test 用作开发集。不拼接不同候选的最好任务成绩，不称为独立测试泛化或全领域 SOTA。
- 基线池是 BM25、dense、HippoRAG 2、Mem0、LightMem、LightMem offline、AnchorMem dense、AnchorMem official、CatRAG；有些上下文预算不同。
- 历史三 reader 比较尚没有全部六任务严格胜出的统一配置。`canonical_graph_rrf_window` 与 `canonical_rrf_sentence_facts` 均为 **5 / 6 / 3**。
- 9B 目标不只差“同一个任务”：前者 MH 为 58，未超过 59；后者 FCSH 为 63，未超过 66。不能把两组拼成 6/6。

结果来源根目录：`/oscar/scratch/zliu328/agent-memory-outputs/`。
各节给出实验目录简称；完整名称统一为 `optimization_<简称>_seed42_20260913`。
每个完整目录的 `comparison.json` 包含逐任务三 reader 分数与基线阈值，`results.md` 包含分数和 QA 实际 token。
本次已读取这些报告核对完整状态；这不等于本次重新运行了全部 QA。
此前已检查 46 个完整候选运行的名称均出现在本记录中；同名 schema 的不同轮次分别列出。
本轮新增完成两个 RRF 20 候选，完整候选数增至 48，其余未完成状态单列。

## 1. 图边权与上下文校正

实验目录：`context_graph`。固定原 recognition、reset、PPR 与 top5 原文。

| 候选 | 改动 | 胜出数 9B / 4B / 2B | 状态 |
| --- | --- | --- | --- |
| relation_only | 排除同义边贡献，保留原关系和来源支持 | 2 / 2 / 2 | 完整 |
| balanced | 对称节点强度归一化 | 2 / 1 / 1 | 完整 |
| contextual | 用实体来源段落向量校正同义连接 | 3 / 2 / 2 | 完整 |
| contextual_balanced | 上下文校正加强度归一化 | 2 / 2 / 1 | 完整 |

有效处：排除同义贡献后，2Wiki Recall@5 从原 HippoRAG 的 74.975 提高到 87.225。
不足：归一化没有形成跨 reader 的整体优势；上下文均值校正也未突破 2B。
判断：保留 `relation_only` 作为后续结构对照，不把所有同义边视为错误，也不把局部召回提高泛化到所有 QA。

## 2. 来源顺序与关系 schema

实验目录：`source_consolidation`、`schema_graph`、`canonical_graph`、`canonical_latest`。

| 候选 | 改动 | 胜出数 9B / 4B / 2B | 状态 |
| --- | --- | --- | --- |
| latest_relation | 每个主体、关系保留最后来源支持 | 3 / 4 / 3 | 完整 |
| latest_relation_balanced | 上述图加节点强度归一化 | 4 / 4 / 2 | 完整 |
| schema_latest（首版） | LLM 关系分类，单值取末来源、多值保留 | 3 / 4 / 2 | 完整 |
| schema_latest_synonyms（首版） | 上述图恢复同义支持 | 1 / 2 / 0 | 完整 |
| schema_latest（联合归一） | 对语义近邻标签联合判断 canonical label | 3 / 3 / 2 | 完整 |
| schema_latest_synonyms（联合归一） | 联合归一后恢复同义支持 | 1 / 1 / 0 | 完整 |
| canonical_latest | 所有 canonical 内容关系都取最后来源 | 3 / 4 / 3 | 完整 |
| canonical_latest_balanced | 上述图加强度归一化 | 4 / 4 / 3 | 完整 |

有效处：最新来源政策提供了后来较强组合的图底座；canonical 图的 9B 2Wiki 为 61.04。
不足：来源顺序不是事件时间；单值不等于可覆盖状态。LLM 归一化还出现过把 `has children`、
`are sitting on`、`is a type of` 放在同组的错误。恢复同义边在这两轮没有改善共同覆盖。
判断：是开发集上有效的来源支持 heuristic，不是正确时序更新或语义消歧的证明。

## 3. 统一权重扫描、来源比例与自适应同义边

实验目录：`canonical_weight_grid`、`provenance_graph`、`adaptive_synonyms`。

| 候选 | 改动 | 胜出数 9B / 4B / 2B | 状态 |
| --- | --- | --- | --- |
| canonical_soft025 | 强度归一化指数 0.25 | 2 / 4 / 3 | 完整 |
| canonical_syn005 | 同义支持比例 0.05 | 4 / 4 / 3 | 完整 |
| canonical_syn020 | 同义支持比例 0.2 | 3 / 2 / 1 | 完整 |
| canonical_syn005_soft025 | 0.05 同义支持加 0.25 归一化 | 3 / 4 / 1 | 完整 |
| canonical_syn020_soft025 | 0.2 同义支持加 0.25 归一化 | 2 / 2 / 2 | 完整 |
| canonical_provenance | 来源边乘以保留断言占比 | 2 / 4 / 3 | 完整 |
| canonical_provenance_balanced | 来源比例加强度归一化 | 4 / 4 / 3 | 完整 |
| canonical_provenance_syn005 | 来源比例加 0.05 同义支持 | 3 / 4 / 2 | 完整 |
| adaptive_syn020 | 用来源断言兼容性调节 0.2 同义支持 | 4 / 5 / 3 | 完整 |
| adaptive_syn020_balanced | 自适应同义支持加强度归一化 | 2 / 2 / 1 | 完整 |
| adaptive_provenance_syn020 | 自适应同义支持加来源比例 | 4 / 5 / 2 | 完整 |

有效处：`adaptive_syn020` 使 4B 达到 5/6；9B MH 为 67。
不足：其 LoCoMo 为 51.34 / 47.75 / 44.53，三者都未越过目标；2B SH 为 71。
多个看似合理的校正叠加后反而退步，不能由“约束更多”推出更好。
判断：保留全部扫描结果，不只报告最佳点；这些是显式经验规则，未建立概率校准或理论最优性。

## 4. 修改事实索引

主实验目录：`fact_index_source_runtime`。改变用于 recognition 的事实索引，重新检索。

| 候选 | 改动 | 胜出数 9B / 4B / 2B | 状态 |
| --- | --- | --- | --- |
| index_latest | 最新来源的事实进入索引 | 3 / 4 / 2 | 完整 |
| index_schema | schema 保留政策的事实进入索引 | 3 / 3 / 2 | 完整 |

有效处：原 HippoRAG 环境下完成六任务原图一致性检查，可独立运行，不只依赖保存的 resets。
不足：改变识别入口未比后续图与读出组合更好；新环境试跑曾改变分句与原图检索，不能混为同一变量。
判断：以 `source_runtime` 为主结果，其他失败及部分运行列在末节。

## 5. 纯事实读出与保留原文

实验目录：`compiled_graph_context`、`source_context`。

| 候选 | 改动 | 胜出数 9B / 4B / 2B | 状态 |
| --- | --- | --- | --- |
| original_graph_compiled | 原图排名读取保留三元组表示 | 0 / 0 / 0 | 完整 |
| canonical_graph_compiled | canonical 图读取同一事实表示 | 2 / 3 / 2 | 完整 |
| canonical_graph_with_source | 原文加来源信息和保留事实 | 5 / 4 / 3 | 完整 |
| adaptive_graph_with_source | adaptive 图读取同一原文加事实表示 | 4 / 4 / 1 | 完整 |

有效处：保留原文后，canonical 的 9B FCSH 为 69、FCMH 为 12；纯事实版只有 59、11。
不足：纯事实版 LoCoMo 仅 42.09 / 40.35 / 37.73，丢失上下文严重。
原文加事实也未解决 LoCoMo，分数为 52.64 / 48.80 / 43.78。
代价：9B FCSH 输入 token 从原文版 335758 增至 560654，分数 63 到 69，不是压缩收益。
判断：原文对上下文完整性重要，事实附录的收益与成本要独立报告。

## 6. 来源窗口、BM25 与图排名融合

实验目录：`source_window`、`hybrid_graph`。窗口不重新切分，限制在同一非空时间 metadata 内；
RRF 使用现成排名融合，常数 60、两个候选列表各前五，最终五来源。

| 候选 | 改动 | 胜出数 9B / 4B / 2B | 状态 |
| --- | --- | --- | --- |
| canonical_graph_source_window | canonical 图加邻接来源窗口 | 5 / 4 / 3 | 完整 |
| adaptive_graph_source_window | adaptive 图加同一窗口 | 5 / 4 / 1 | 完整 |
| original_graph_rrf | 原图与 BM25 融合，原文读出 | 3 / 1 / 1 | 完整 |
| canonical_graph_rrf | canonical 图与 BM25 融合，原文读出 | 4 / 3 / 1 | 完整 |
| original_graph_rrf_window | 原图融合加窗口表示 | 2 / 3 / 2 | 完整 |
| canonical_graph_rrf_window | canonical 图融合加窗口表示 | 5 / 6 / 3 | 完整 |

有效处：图、词面检索和来源上下文组合后，4B 达到完整 6/6。
同一 RRF 和原文下，2Wiki 原图到 canonical 图为 52.05→60.10、48.81→57.63、37.36→46.19。
不足：同一对照的 FCSH 从 52→44、52→50、53→52，图收益不是普遍正向。
窗口加 RRF 时，LoCoMo 原图到 canonical 图为 57.40→56.97、51.23→51.46、44.21→43.34。
判断：强组合成立，但不能把融合与窗口的全部收益归给建图；2B 仍未解决。
无时间 metadata 的任务中窗口没有改变 reader 输入，重复 QA 的波动不算窗口效应。

## 7. 窗口去重与事实序列化

实验目录：`context_packing`。

| 候选 | 改动 | 胜出数 9B / 4B / 2B | 状态 |
| --- | --- | --- | --- |
| canonical_rrf_deduplicated | 同一排名与窗口，来源只出现一次 | 5 / 5 / 3 | 完整 |
| canonical_rrf_sentence_facts | 去重加逐行自然语言事实格式 | 5 / 6 / 3 | 完整 |

有效处：LoCoMo 实际输入 token 从 5672028 降到去重版 4949417、逐行版 4814003，
分别减少约 12.7%、15.1%，保留全部唯一来源。逐行版恢复 4B 的 6/6。
不足：4B LoCoMo 从原窗口的 51.46 降为 50.20、50.73，节省成本并没有带来更高分。
逐行版 2B LoCoMo 43.64，仍低于 44.80；9B FCSH 63，低于 66。
判断：这是有实际 token 证据的上下文压缩与格式收益，不是新图结构。

## 8. 历史保留、state/event 与取消关系合并

实验目录：`history_graph`、`event_graph`、`identity_readout`。

| 候选 | 改动 | 胜出数 9B / 4B / 2B | 状态 |
| --- | --- | --- | --- |
| canonical_rrf_history_facts | 图不变，读出保留多值历史 | 5 / 5 / 3 | 完整 |
| schema_rrf_sentence_facts | 图保留多值历史，读出不变 | 4 / 5 / 3 | 完整 |
| schema_rrf_history_facts | 图与读出都保留多值历史 | 5 / 6 / 3 | 完整 |
| canonical_rrf_event_facts | 图不变，读出采用 state/event 政策 | 4 / 5 / 2 | 完整 |
| event_graph_rrf_sentence_facts | 图采用 state/event，读出不变 | 4 / 5 / 3 | 完整 |
| event_graph_rrf_event_facts | 图与读出都采用 state/event | 3 / 5 / 2 | 完整 |
| identity_graph_rrf_sentence_facts | state/event 图不应用 canonical 合并 | 4 / 6 / 3 | 完整 |

有效处：保留历史恢复了一些被覆盖的来源归属；只改 state/event 图的 4B FCSH 达到 68。
取消 canonical 合并使 4B FCMH 从对应 event 图的 4 到 6，达到该配置完整 6/6。
不足：更多历史未改善共同覆盖；只改历史读出时 9B FCSH 从逐行参照的 63 降至 56。
取消合并后 2B LoCoMo 从对应 event 图的 44.47 降至 43.10。
判断：语义上更细的分类不自动等于更高分；标签级 state/event 和最后来源仍不是真实时间推理。
identity 消融只取消映射的应用，分类 prompt 仍见过旧 canonical 标签，不是完整 canonical-free 抽取。

## 9. 删除事实附录，只读原文

实验目录：`source_only`，候选 `canonical_rrf_source_only`，完整结果 **4 / 4 / 3**。
固定图、排名、窗口和来源去重，仅删除额外事实附录。

有效处：每个 reader 总输入从逐行事实版的 9584490 降到 7181935 token，减少约 25.1%；
2B 2Wiki 从 42.15 提高到 46.07。
不足：9B FCSH 从 63 降到 51，4B 从 64 降到 47，2B LoCoMo 从 43.64 降到 42.86。
判断：事实附录存在任务取舍，不能据 2Wiki 一项就全局移除，也不按任务开关路由。

## 10. 显式来源事实 incidence 图

实验目录：`fact_incidence`，候选 `source_fact_incidence`，完整结果 **1 / 2 / 1**。
每次三元组出现建独立节点，连接原主体、客体和来源；采用标准二部表示及原 PPR。
不同时加 RRF、窗口、关系归一化或新摘要。详见 [实验说明](https://github.com/Sizchode/agent-memory/blob/c443de4790e619d43c5bd9dcce9ccea707da56bc/optimization/fact_incidence_experiment.md)。

有效处：来源事实出现及重复贡献可独立追踪；六任务 3386/3386 普通检索接口验证通过。
相对 `relation_only`，9B FCSH 从 41 到 50。
不足：同一参照下 SH 从 90 降至 84；2Wiki Recall@5 为 86.05，低于参照的 87.225。
谓词与角色只是 metadata，原无向 PPR 不读取其语义；共享实体仍可能把不同事件连接起来。
判断：增加结构表达能力不等于检索收益；该组不是已成功的 hypergraph 新算法。

## 11. Gist 表示与已有强组合

原格式和紧凑格式都没有完成全语料，输出与成本保留；json_object 诊断也未全通过。
当前抽取版本为 `contextual_gists_format_retry`，采用统一、有界的格式重试，并核对后复用原格式有效日志。
没有 gist QA 分数，详情见第 13 节末尾和 [生成协议](https://github.com/Sizchode/agent-memory/blob/c443de4790e619d43c5bd9dcce9ccea707da56bc/optimization/gist_generation_protocol.md)。
四个候选为 `gist_hipporag`、`gist_dense`、`raw_dense_control`、`canonical_rrf_gist_index`。
前三者区分现成向量匹配与原 HippoRAG；第四个只替换已有强组合 `canonical_rrf_sentence_facts`
的 passage embedding，保留其图权重、RRF、BM25、窗口和事实读出，不把 gist 再附加给 reader。

待检验的好处：同一来源的上下文化表示可能改善匹配；若对原图和强组合都有效，证据比只在弱参照上有效更强。
当前限制：gist 的内容由 LLM 生成，格式合法不保证忠实；在原分块之外的指代无法凭空解决。
更长的 gist 不自动构成压缩。索引表示有效也不等于新图结构有效。
暂不写“有效/无效”结论，待完整结果后补充。新增组合共享同一份抽取和 embedding，不新增离线生成调用。
78 项实现测试通过；真实 QA 完成情况以 [gist_index_jobs.json](https://github.com/Sizchode/agent-memory/blob/f2121c11cc6d0c36bf931674152db0ca07669573/optimization/gist_index_jobs.json) 和 scratch 报告为准。

17:50 EDT 对四个已完成任务的全部 7368 来源，使用原 Qwen3-Embedding-0.6B tokenizer
测量原文与换行拼接 gist：FCSH/FCMH 的长度比为 0.708 / 0.707，LoCoMo 为 1.455，
SH 为 1.003；这些来源均无空 gist。详细总 token 和最长表示见实验说明。
已知好处仅限部分任务的表示变短，已知不足是对话来源整体变长；均不是 QA 收益证据。
据此将本轮机制表述为“索引表示改写”，不预称统一 context compression，仍保持六任务统一操作。

## 12. 补齐 RRF 候选窗口对照

实验目录：`rank_window`，候选为 `original_rrf20_sentence_facts`、`canonical_rrf20_sentence_facts`。
此前只完成 FCSH 检索，本轮继续原先固定的 20 候选设置，不增加新的参数扫描点。
图和 BM25 各取前 20、RRF 常数 60、最终仍五来源；其余来源窗口和逐行事实读出不变。

可能的好处：融合前保留更多可供交叉匹配的来源，而不是只在两个 top5 的并集中选。
可能的不足：更多词面偶合候选会改变最终排名，排名一致不保证事实相关；这不是新图构建贡献。
当前没有完整 QA 分数，不把上述动机写成实测结论。
canonical 组的直接预算参照是旧 `canonical_rrf_sentence_facts`；本轮 original/canonical 则是同预算图对照。
两组完整六任务三 reader 已提交，共 20316 次 QA，FCSH 两组普通接口验证均为 100/100。
协议和执行见 [rank_window_experiment.md](https://github.com/Sizchode/agent-memory/blob/c443de4790e619d43c5bd9dcce9ccea707da56bc/optimization/rank_window_experiment.md)、[rank_window_jobs.json](https://github.com/Sizchode/agent-memory/blob/f2121c11cc6d0c36bf931674152db0ca07669573/optimization/rank_window_jobs.json)。
首个完整短任务 FCSH：original 20 为 58 / 56 / 61，canonical 20 为 55 / 63 / 62。
直接参照 canonical 5 为 63 / 64 / 61；9B 明显退步、2B 略升，记录为混合结果，
不把这一任务的 2B 提高写成全局改善。尚需其余五任务完成。

2Wiki 完整 1000 题的支持段落 Recall@5 已核对：原 HippoRAG 2 为 74.975%，
canonical 5 为 85.800%，original 20 为 73.500%，canonical 20 为 80.150%。
使用原有 gold-passage 指标、相同完整 case 和每题五个原始来源，不包含事实附录或额外窗口。
实测不足：直接扩大 canonical 候选窗口使召回下降 5.65 个百分点，而非扩大最终证据覆盖。
有效处：固定 20 窗口时 canonical 相比原图仍高 6.65 个百分点。
这是一项完整检索结果，不是完整 QA，也未证明下降由词面噪声造成。
逐题为 169 题召回下降、20 题上升、811 题不变：既有关键证据被双路共同候选挤出，
也有低位 gold 被找回的实例；两类实例及来源排名已记录在实验说明，不根据这些题添加特例。
17:49 EDT 四个 MAB 任务已完整完成：canonical 20 的 9B 为 89 / 58 / 55 / 6，
4B 为 90 / 57 / 63 / 8，2B 为 76 / 47 / 62 / 7（依次 SH、MH、FCSH、FCMH）。
相较直接参照 canonical 5，4B FCMH 从 7 到 8，但 2B MH 从 51 到 47；窗口扩大仍呈取舍。

18:06 EDT：2B 两候选的六任务全部 3386 题 QA 已完成并审计。
original 20 为 80 / 49 / 61 / 4 / 43.72 / 33.76，严格胜出 2/6；
canonical 20 为 76 / 47 / 62 / 7 / 42.86 / 37.73，严格胜出 2/6。
直接参照 canonical 5 为 79 / 51 / 61 / 8 / 43.64 / 42.15，严格胜出 3/6。
因此扩大窗口在 2B 上没有提高整体覆盖：FCSH 略升，但 MH、LoCoMo、2Wiki 等退步。
original 20 的 4B 也已全量完成，胜出 1/6；其余 reader 状态以报告为准，尚不列两候选全矩阵结论。

18:13 EDT：original 20 三 reader 全六任务均完成，严格胜出 **2 / 1 / 2**。
9B 为 89 / 56 / 58 / 6 / 56.95 / 48.91，4B 为 89 / 54 / 56 / 5 / 51.16 / 47.31，
2B 为 80 / 49 / 61 / 4 / 43.72 / 33.76。每个 reader 3386 次 QA、输入 9412110 token。
它在 9B/4B LoCoMo 高于基线池阈值，但没有提高三个 reader 的共同覆盖。
canonical 20 仍未全矩阵完成，不能以 original 的完成替代它的剩余 QA。

窗口对照现已完整收齐：汇总作业 6342884 COMPLETED 0:0，两候选共 20316 次 QA。
canonical 20 三 reader 严格胜出 **3 / 6 / 2**，其全矩阵为：

| reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9B | 89 | 58 | 55 | 6 | 55.92 | 54.05 |
| 4B | 90 | 57 | 63 | 8 | 51.15 | 52.87 |
| 2B | 76 | 47 | 62 | 7 | 42.86 | 37.73 |

每个 reader 输入 9518885 token。相对 canonical 5 的 **5 / 6 / 3**，9B 六任务分数均下降；
4B 的 FCMH（7 到 8）与 LoCoMo（50.73 到 51.15）提高，其余下降；2B 仅 FCSH（61 到 62）提高。
结论：扩大候选窗口未提高共同覆盖，保留完整负结果，不替换旧最佳配置、不追加窗口扫描。
这些是当前协议单次运行的取舍，未声称跨硬件差异已排除或达到统计显著。

## 13. 未完成与无效尝试

| 尝试 / 目录 | 当前事实 | 判断 |
| --- | --- | --- |
| fact_index（首轮） | recognition 认证失败，上游吞错误后可能退回 dense；已有 INVALID.json | 无效，不并入成绩 |
| fact_index_authenticated | index_latest 与 index_schema 仅三个任务完成；新环境原图检索不一致 | 部分记录，不替代原环境全量结果 |
| rank_window | 已补交全六任务依赖链；FCSH 两组普通接口验证通过，QA 已启动 | 无完整 QA 分数，不宣称扩大候选有效 |
| contextual_gists | 复用 REMem passage prompt，目标全部 14362 来源；续跑后 FCSH、FCMH 各 537/537，LoCoMo 已开始生成 | 来源表示抽取未完成，不是新方法 QA 结果 |
| gist_hipporag / gist_dense / raw_dense_control | 接入代码已实现，已提交完整索引、六任务检索与三 reader QA 依赖链，等待抽取完成 | 保持原图、原文读取，不预称图创新 |
| canonical_rrf_gist_index | 同一 gist 索引接入旧强组合，完整 QA 已提交 | 只改变索引表示，尚无效果证据 |
| relation_schema / canonical_schema / update_policy_schema / relation_identity_schema 等中间目录 | 对应上述图实验的离线分类或派生构建产物 | 不是额外独立的 QA 候选 |

gist 作业 6341669 因两个 HTTP 连接错误失败，已保存 1072 条有效抽取，失败请求 usage 未知而非零。
续跑 6341980 未启动生成：共享 vLLM 环境的启动入口消失，退出码 127。
搜索后未找到其他完整环境，已用缓存及必要的缺失依赖恢复项目专用 `agent-memory-envs/vllm_cu129`，
不覆盖残留目录，不删除已有输出，不把环境失败当成方法失败。
恢复后的 vLLM 0.26.0+cu129、Torch 2.11.0+cu129、Transformers 5.14.1、xgrammar 0.2.3
均已核对，190 个包通过依赖兼容性检查，CUDA 模块为 12.9.0-cinr。
续跑作业为 6342535；完整索引 6342552、六任务检索数组 6342553、
六个三 reader QA 数组 6342554 至 6342559、汇总 6342560。
完整记录见 [gist_index_jobs.json](https://github.com/Sizchode/agent-memory/blob/f2121c11cc6d0c36bf931674152db0ca07669573/optimization/gist_index_jobs.json)；提交不等于已完成，暂无新 QA 分数。
加入第四候选后，新增 QA 数组为 6342728 至 6342733；6342560 在 pending 时取消，
统一汇总改为 6342734，等待全部四候选共 40632 次 QA。
17:22 EDT 已核对新服务 health 与真实生成请求均返回 200；FCMH 两个缺失来源已补齐，
FCSH 已有完整结果直接复用，LoCoMo 抽取正在进行。原两条连接错误记录仍保留。

17:48 EDT：FCSH、FCMH、LoCoMo 十组和 SH 均完成，共 7368 个来源。
MH 抽取继续运行，已记录一条新的 HTTP 连接错误（usage 未知），后续请求仍正常返回；
不重启正在运行的作业，也不把该来源静默丢弃。待本组结束后按实际作业状态续跑缺失来源。

同一续跑的 MH 另有一条 `finish_reason=length`：来源
`chunk-53a79c07fca298a488eb49c531ad04af` 的输出在若干 gist 后产生大量重复空白，
消耗 1039 输入、8192 输出 token，耗时约 209.8 秒。保留原始响应和实际成本，
不把非空前缀解析为成功，不删空白或补闭合 JSON 来修复，不混入索引。
这与无 usage 的连接错误分别记账；后续只允许同配置续跑未完成来源，已有有效结果不重抽。

17:58 EDT：6342535 实际 FAILED 1:0，运行 40:07；MH 最终为 872/875 有效来源。
第三条未完成来源 `chunk-e0f7dba25da8b4c736306ebbf76ec01d` 同样为 length，
输入 991、输出 8192 token、约 141.4 秒。两条 length 输出均不进入索引，连接错误 usage 仍未知。
同配置续跑 6343507 已启动，只补三个缺失来源，之后继续完整 2Wiki；总计已有 8240 个有效来源可复用。

Slurm 将原索引 6342552 及全部未启动下游链取消为 DependencyNeverSatisfied，运行时间均为零。
修改旧索引依赖被拒绝（job already finished），因此重新提交：索引 6343525，检索 6343527，
六个三 reader QA 数组 6343528 至 6343533，报告 6343534。每个新 QA 数组统一跑全部四候选，
总范围仍为 40632 次 QA，未重新计算任何已完成 QA。旧链完整保存在作业清单的 superseded_pipelines。

18:01 EDT：6343507 再次 FAILED 1:0，用时 3:03。连接错误来源恢复，MH 为 873/875，
但两条 length 来源再次各用满 8192 输出 token（输入分别 1039 / 991，约 81.4 / 81.7 秒）。
重复内容分别为 JSON 字段间空白、字符串中的括号文本，并非两条都是纯空白。
6343525 至 6343534 对应的整条下游链再次自动取消，均未运行，不把提交次数记成评测次数。

不再原样重试这两个来源。格式诊断 6343703 使用 vLLM 自带紧凑 JSON 模式，
按保存的失败状态自动选择未完成来源，保留相同 prompt/schema/模型/预算，但逐条请求。
这不是新的 benchmark 候选，诊断输出不自动并入正式索引；完整实验尚未完成。
新增测试验证诊断排除已补齐、有效空列表和从未尝试的来源；相关 7 项测试通过。

18:07 EDT：诊断 6343703 COMPLETED 0:0，用时 2:08。两来源均 stop 且通过原内容 schema，
输入仍为 1039 / 991 token，输出为 360 / 498 token，约 5.77 / 3.65 秒；未修复或截断输出。
这证明该运行设置能完成两个失败来源，不证明 gist 的检索效果；诊断逐条运行，不能排除批次差异。

据此提交正式新目录 `contextual_gists_compact`，全部 14362 来源从头采用同一紧凑 JSON 配置，
仍用原 prompt、内容 schema、模型、T0、seed42、8192 预算及并发 16。
不把诊断结果或旧格式的有效结果混入新语料，旧缓存不删除，重复抽取属于配置改变后的全量实验成本。
生成器 6344538，索引 6344548，六任务检索 6344549，三 reader QA 数组分别为
6344550 / 6344551 / 6344553 / 6344554 / 6344555 / 6344556，报告 6344557。
全部四候选仍需六任务全量 40632 次 QA；此前两条取消链和失败成本完整保留。
16 项相关测试及 shell 语法检查通过；新正式抽取已提交不等于来源或 QA 已完成。

紧凑格式全量运行未通过：6344538 在 FCSH 出现新的 length 输出，包含对反事实材料的说明及重复文本。
18:18 EDT 因已经记录的生成失败主动停止，Slurm 状态为 CANCELLED，用时 9:14，不伪称自然 FAILED。
停止后客户端清理将 FCSH 的 537 个来源全部留有请求记录：399 个 stop 且格式有效、15 个 length，
另有 16 个 shutdown abort 和 107 个 shutdown 期间的客户端连接异常；后两类不能算作独立方法失败。
有 usage 的记录合计输入 481209、输出 277437 token，含已返回的 abort 用量；107 条异常 usage 未知，不补零。
未修复、删除失败内容或缩小正式六任务范围。6344548 等下游整链均被 Slurm 自动取消，运行时间为零。

新诊断 6344799 对照 REMem 默认的 json_object 请求接口，仍用同一 passage prompt、模型和解码预算。
自动合并两个未完成根目录的来源，共 140 条：原格式 2 条及紧凑格式 138 条，其中包含停止影响的请求。
因此 140 不是“140 条模型格式错误”；后续必须区分原本 length 与 shutdown 影响的来源。
诊断仍逐条请求，不修复 JSON，不进入正式图或 QA。18 项相关测试与 shell 检查通过。

6344799 最终 FAILED 1:0，用时 11:22：139/140 来源有效，唯一未完成为原 MH 的
`chunk-53a79c07fca298a488eb49c531ad04af`（length）。诊断合计输入 156909、输出 82182 token。
该结果不支持把 json_object 当作全语料已验证的替代，也不以其来源子集表现报告检索分数。

新生成器 6345102 使用 `contextual_gists_retry`：每个来源先原格式，格式失败才允许一次紧凑请求，
不修补内容、不比较下游质量；两格式均失败则停止，跨续跑保留已失败格式，不无限重试。
这是此前无格式重试协议的修订，不称为研究 novelty。HTTP 读取错误与 abort 不触发格式切换。
核对原生成设置与来源后复用 8241 个有效来源及原日志，复制日志不计新增调用，不导入诊断结果。
完整协议、成本和缓存边界见生成协议文档；90 项测试通过，六任务三 reader 范围不变。

6345102 的真实运行暴露接口错误：请求级 disable_any_whitespace 被接收但未进入 xgrammar 的
JSON 编译参数，响应仍含原缩进；作业 FAILED 1:0，用时 3:24。新增两次请求输入 2030、输出 8774 token，
一条 stop、一条 length，但均不能标为已生效的 compact 请求。该目录已写 INVALID.json，不进入正式索引。

进一步真实 matcher 核对发现，带 minLength 的字符串规则拒绝合法的 JSON 引号、反斜杠和换行转义。
修正为用 xgrammar 从结构 schema 生成紧凑 EBNF，通过已有 grammar 请求接口提交；
只移除 decoder 的 minLength 限制，原输出校验仍拒绝空白/空字符串，不修补内容。
生成 grammar 的一致性、标准转义、结构与紧凑格式均通过真实 compiler/matcher 的 3 项测试。
新作业 6345810 使用 `contextual_gists_format_retry`，仍仅复用原单格式根目录，
不复用无效请求级版本或诊断输出。实际生成与后续 QA 仍须验证，单测通过不代表全语料已成功。

6345810 的真实生成已验证修正后的 grammar 接口：原 MH 两个未完成来源均 stop 且通过原校验，
输出分别为 552 / 564 token，输入仍为 1039 / 991 token。MH 已达到 875/875，14/15 来源组完成；
19:04 EDT 核对时 2Wiki 已有 2191/6119 个有效来源，2191 条记录均 stop，无已记录失败。
这只是进行中的覆盖检查，不是全语料成功或 QA 提升。所有新调用与导入旧调用分别追踪。

正式后续链已提交：索引 6346491、检索数组 6346492、六任务三 reader QA 数组
6346493 / 6346494 / 6346495 / 6346496 / 6346497 / 6346498，报告 6346499。
输出目录 `optimization_gist_index_format_retry_seed42_20260913`，四候选和 40632 次 QA 范围不变。
依赖要求全部来源和索引完成后，再执行全任务原始/packed 控制与候选检索；每任务检索通过才开始 QA。
作业清单保留此前取消链及无效 6345102，不把排队状态写成已完成结果。

本轮运行中的成本审计：逐组解析旧、新 JSONL 并验证导入部分与原文件记录逐条相等，
共导入 8248 条调用（8241 stop、4 length、3 连接异常），已知输入 5728793、输出 1927344 token；
3 条异常 usage 仍未知。这个导入成本只计一次，不再算成本轮新增成本。
检查时新调用为 2772 条（MH 2 条和 2Wiki 2770 条），全部 stop 且有效；
新增已知输入 1696047、输出 457441 token。这是生成进行中的快照，不是最终成本。
请求时间之和受并发影响，不能当作阶段 wall time；后者另取作业和阶段计时。
本轮重新执行 93 项测试：普通环境 90 项通过、3 项 xgrammar 测试跳过；
再在 vllm_cu129 环境单独执行这 3 项真实 compiler/matcher 测试，全部通过。

19:20 EDT，生成 6345810 正常 COMPLETED 0:0，用时 29:27。逐组核对输入 ID、有效调用、
gists.json、组完成标记和根完成标记：全部 15 组、14362 来源、123023 条 gist，空列表来源为 0。
本轮新增 6122 次调用：6121 stop、1 length；输入 3748150、输出 1018448 token，均有 usage。
其中 6119 次原格式、3 次紧凑 grammar（旧 MH 2 个来源，以及新 2Wiki 1 个来源），不是每题额外调用。
2Wiki 的 `chunk-ae8b08c63257896f41b4bbdd5261063a` 原格式输入 633、输出 8192，
触发同一格式重试后输出 156 token 并 stop；保留第一次失败输出，不修补或丢弃来源。
导入旧日志的成本沿用上段单独计数；此前独立诊断和无效实验成本也不能从研究总成本中删去。
索引 6346491 已自动开始；截至本次记录仍未有 gist QA 分数，生成完整不等于方法优于 baseline。

索引 6346491 正常完成（5:04），15 组来源全部建好，embedding 输入共 2493586 token。
逐组检查全部向量数组及四候选元数据，来源键完整唯一、向量有限；packed 候选的边权、fusion、
readout 均保持冻结控制原样。这轮改变索引表示，不把它描述为新图拓扑。
四个 MAB 任务检索已完成，每任务 100 题的原始和 packed 控制全部匹配；LoCoMo / 2Wiki 仍在检索。
QA 已启动。为利用空闲普通 GPU 配额，将尚未运行的 FCMH QA 数组 6346496 改到 gpu / norm-gpu，
保持原作业 ID、配置、输出路径及三个 reader，不重启已运行任务，也不挑选复跑分数。

拼接 gist 的阶段报告由原 report_results 审计器生成，核对完整任务的题号、预测均值、usage
及九组 baseline。四个 MAB 任务已显示 2B 的 packed 配置在 SH 78 / FCMH 7，不能达到全面胜出；
LoCoMo / 2Wiki 的全量评测继续，不取消或隐藏失败候选。

据此提前准备下一轮单轴粒度对照（不等待长任务结束后再闲置建索引）：完全复用现有 gist，
逐 gist 编码并采用 Dense X Retrieval 第 4.3 节已有的 max-to-source 聚合，随后沿用 HippoRAG
原归一化、PPR 与 readout。原有拼接实验不改配置。完整方案见 gist_units_experiment.md。
三个新候选仍全六任务、全三个 reader，计划 30474 次 QA；原文 dense 控制沿用上一轮完整 QA，
不重复跑后取最高。15 项针对性测试通过，包括真实上游单单位等价、重复单位、负分数、
原始查询编码路径，以及保存/加载来源映射；尚无本轮完整检索或 QA 成绩。

拼接索引的六任务检索现已全部正常完成；所有 3386 题的原始/packed 控制均匹配。
2Wiki 完整 1000 题使用既有 gold_passage_recall_at_k，来源原文还原后均为 5 个唯一原段落：

| 配置 | gold supporting-passage Recall@5 (%) |
|---|---:|
| 原 HippoRAG | 74.975 |
| 原 packed 强配置 | 85.800 |
| gist_hipporag | 73.450 |
| gist_dense | 66.975 |
| raw_dense_control | 68.700 |
| canonical_rrf_gist_index | 85.375 |

因此在这个完整检索指标上，拼接 gist 没有超过对应原文索引控制；不把生成完成等同于表示有效。
QA 仍须完整运行，检索 recall 不能直接替代回答分数。
逐 gist 构建作业 6347152 已在 gpu / norm-gpu 运行；新协议及路径见 gist_units_experiment.md，
作业清单见 gist_units_jobs.json。全 optimization 测试共 99 项，96 通过、3 项 xgrammar 跳过；
此前该 3 项已在专用 vLLM 环境真实通过，此轮未改 grammar。

逐 gist 的真实构建已落盘多个组，实际向量为每条 gist 1024 维；来源数、gist 数与映射逐项相符，
数值有限。后续链已提交：检索 6347323；六任务 QA 6347324 / 6347330 / 6347331 / 6347332 /
6347333 / 6347334；报告 6347335。仍须全源索引完成后启动检索，控制通过后开始对应任务 QA。
运行中的拼接与逐条索引实验各自保留完整结果，不以这个阶段的胜负取消其余任务。

拼接版本的一个正结果：gist_hipporag 在 2B 的完整 LoCoMo 上 F1 为 45.73194%，
超过九组完整 baseline 最佳 CatRAG 的 44.80081%（+0.93113 个百分点），原 HippoRAG 为 43.64696%。
已用原 audited_score 检查全部 1986 题 ID 与预测均值；QA 输入 924593、输出 15794 token，
调用时间之和 437.907 秒。这个单任务结果支持继续检查表示作用，不代表该候选六任务都好，
也不把不同硬件的时间或单次分数差当作显著性结论。

逐 gist 索引构建 6347152 正常完成（9:41）：15 组、14362 来源、123023 个单位向量。
全部单位顺序和来源映射与未修改的 gist 一致，向量维度与有限值通过检查。
本轮 embedding 输入 2602135 token、编码调用 427.724 秒，无新生成调用；检索数组 6347323 已启动。
尚无逐 gist 的完整 QA 结果，因此不预判其是否强于拼接配置。

## 优先级调整：旧主配置消融

用户确认优先消融与简化，暂停新增提分组合。第一阶段固定 canonical_graph_rrf_window，
移除 RRF、邻居窗口、triples 附录、关系归一化、latest-only、discourse 过滤，分别做单模块对照。
图消融保持旧 reader 表示，避免同时改变事实附录；其结论仅涉及图部分的条件作用。
精确还原全部旧边权和旧来源字段后才允许检索；每任务普通接口还须完整复现旧配置上下文。
已有 no-RRF QA 只在完整上下文等价后复用；其他全任务与主配置完全等价的结果也不重复生成。
这不是按分数选择结果，所有新/复用成本分开。完整设计与第二阶段边界见 ablation_plan.md。

消融构建 6353716 正常完成（1:29），15 组、14362 来源全部精确还原旧边权和旧上下文字段。
旧元数据缺少后来增加的 window_base_text / retained_triples 等字段，只允许重建结果增加字段，
不放宽任何旧字段或 reader 文本相等检查。新消融均在独立根目录 optimization_ablation_seed42_20260913。
107 项测试中 104 项已执行通过，3 项独立 xgrammar 测试跳过；其中包括原构图默认行为、
单模块开关、旧元数据一致性和报告正负差值/复用成本检查。
检索数组：6353757（四个 MAB，gpu-he）与 6353758（LoCoMo / 2Wiki，gpu）；
六任务三 reader QA 为 6353759 / 6353760 / 6353761 / 6353762 / 6353763 / 6353764，报告 6353765。
目前只完成构建，尚未得出消融贡献结论。完整作业与复用边界见 ablation_jobs.json。

首轮六任务检索均因 case 校验实现错误退出，依赖 QA / 报告被 Slurm 自动取消，没有新 QA 分数。
逐字段检查全部 3386 题：差异仅为原 dataclass 的 answers / gold_passages 元组与 JSON 数组读回的
list 类型；按实际 JSON 序列化比较后，原 loader 与两个既有参考的全部字段完全一致。
修复仅统一比较表示，并将全题字段校验移至 GPU 加载之前，不改变 loader、答案、指标或检索规则。
增加测试保证题目、答案、gold、category 等任一字段改变，以及字段缺失/新增，仍会失败。
首轮输出和日志保留；重跑使用 optimization_ablation_case_serialization_seed42_20260913 独立根目录。
重跑前已检查两个参考的六任务、三个 reader 共 36 份 QA 均完整，预测 ID、评分与 usage 通过原审核函数。
复用校验同时要求题目遍历顺序一致，保留 LoCoMo 原选项随机序列。108 项测试中 105 通过、3 项跳过。

重建后的结构检查如下，仅为与完整配置不同的边权数量，不是 QA 贡献或检索质量：

| 任务 | 去 canonicalization | 去 latest-only | 去 discourse filter |
|---|---:|---:|---:|
| SH-Doc_QA | 135 | 712 | 306 |
| MH-Doc_QA | 47 | 410 | 76 |
| FactConsolidation-SH | 4491 | 12744 | 0 |
| FactConsolidation-MH | 5042 | 13269 | 0 |
| LoCoMo | 2114 | 19021 | 1214 |
| 2WikiMultiHopQA | 375 | 2515 | 82 |

两个 FactConsolidation 图的 discourse 消融边权完全不变，仍须完整普通检索校验才复用 QA。
实现层面的耦合：canonical 标签只影响 latest-only 分槽；最终边权累积无向实体对和 passage-entity
支持，不使用 predicate embedding。固定 reader、关闭 latest-only 后去 canonicalization 的边权等价，
已加入针对性测试；不能把二者描述成独立可加的两个图语义模块。

重跑构建 6354005 完成（1:25），四个 MAB 检索 6354028_0..3 全部通过完整旧控制和 no-RRF 校验。
各消融与旧主配置有序 reader 文本相同的题数如下（每任务均 100 题），不代表回答正确率：

| 任务 | 去 RRF | 去窗口 | 去附录 | 去 canonicalization | 去 latest-only | 去 discourse |
|---|---:|---:|---:|---:|---:|---:|
| SH-Doc_QA | 1 | 100 | 0 | 99 | 96 | 99 |
| MH-Doc_QA | 0 | 100 | 0 | 99 | 91 | 100 |
| FactConsolidation-SH | 0 | 100 | 0 | 75 | 45 | 100 |
| FactConsolidation-MH | 0 | 100 | 0 | 69 | 32 | 100 |

四个任务的窗口均未改变实际文本，因此窗口不能解释这些任务的成绩；discourse 对 MH 和两个
FactConsolidation 的完整检索也没有作用。本轮按预定全任务等价规则复用这些 QA，不逐题拼接。
LoCoMo / 2Wiki 检索 6354029_4..5 继续；QA 数组为 6354102 / 6354116 / 6354117 / 6354123 /
6354124 / 6354125，报告 6354131。2Wiki QA 使用 gpu / norm-gpu，其他使用 gpu-he，避免单分区闲置。
此时 SH 三 reader 已启动，还没有完整新 QA 分数；不能把检索相同率解读为最终模块收益。

### 已有整体图对照复核

`optimization_hybrid_graph_seed42_20260913/original_graph_rrf_window` 已有完整 QA，可用于整体换回原图的
对照。本轮实际核对 15 组：边权等于原 HippoRAG pickle；source_graph、RRF 参数、词面来源顺序、
完整 compiled source 字段均与主配置一致。全 3386 题的 case 字段与遍历顺序相同，18 份 QA 的预测
ID、原评分和 usage 均完整。此处是既有结果复核，不声称本轮重新运行了该对照的普通检索或 QA。

下表为旧主配置减原图对照的差值，单位百分点；正值表示新边权配置更高，不表示统计显著：

| reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki |
|---|---:|---:|---:|---:|---:|---:|
| 9B | -3.00 | 0.00 | +3.00 | +3.00 | -0.429 | +6.501 |
| 4B | +3.00 | +3.00 | +1.00 | +5.00 | +0.225 | +7.666 |
| 2B | +4.00 | -4.00 | -3.00 | +6.00 | -0.873 | +7.462 |

4B 是六任务描述性正差值；9B 为 3 胜 / 1 平 / 2 负，2B 为 3 胜 / 3 负。
2Wiki 和 FCMH 的三个 reader 均为正，LoCoMo 的 9B / 2B 为负；不能把完整 pipeline 的胜出数
直接写成“建图在所有模型任务上都更好”。readout 仍含完整策略筛选的事实，结论是固定 reader 的图贡献。

### 已通过完整检索校验的去 RRF 对照

四个 MAB 任务可按预定规则复用既有 no-RRF QA。以下为去 RRF 减旧主配置的百分点差值：

| reader | SH | MH | FCSH | FCMH |
|---|---:|---:|---:|---:|
| 9B | -2 | +6 | +3 | +5 |
| 4B | -2 | -2 | +3 | +5 |
| 2B | -5 | -4 | +1 | +5 |

RRF 并非处处受益：这两个 FactConsolidation 任务中，去掉 RRF 的三个 reader 都更高。
此时 LoCoMo / 2Wiki 的新检索校验尚未结束，不据此选定全六任务的删减配置。

### 第一份三 reader 完整新 QA 与波动检查

FCSH 六项消融的三个 reader 均已完成；每格为消融减旧主配置的百分点差值：

| reader | 去 RRF | 去窗口 | 去附录 | 去 canonicalization | 去 latest-only | 去 discourse |
|---|---:|---:|---:|---:|---:|---:|
| 9B | +3 | 0 | -19 | -5 | -4 | 0 |
| 4B | +3 | 0 | -13 | 0 | +1 | 0 |
| 2B | +1 | 0 | -9 | -8 | -4 | 0 |

去附录在这个完整任务上三个 reader 均明显下降；这是 reader 表示作用，不是纯图增益。
去窗口和去 discourse 的零差值来自预先声明的全任务上下文等价复用，并非独立随机生成恰好相同。

具体波动反例：4B 的 SH 去 canonicalization 后是 92%，主配置 90%；完整检索只有 1/100 题上下文
改变，其余 99 题中 34 题回答字符串改变、4 题官方评分改变（3 道从错到对，1 道从对到错）。
这部分净变化已足以解释 +2 分，因此不能把该涨分归因于移除了 canonicalization。
这里只记录原始预测差异，不更改评分，不排除这些题，不按题拼接控制答案，也不选更高的复跑。
单次执行的采样或数值差异必须作为解释限制；必要时采用事先确定的完整配对复现实验，不能凭小分差
决定删减。自动汇总也明确加入这一限制。当前尚未有全六任务、三个 reader 的完整消融表。

进度快照：18 个任务-reader 组合中 10 个已完成全部六项消融，分别为 MH / FCSH 的三个 reader，
以及 SH / FCMH 的 4B 和 2B。SH / FCMH 的 9B 继续运行；LoCoMo / 2Wiki 仍做完整检索，
依赖 QA 与最终报告尚未开始。新目录当前无失败；不得将该快照写成全任务消融已经完成。
最终代码验证：109 项测试中 106 通过、3 项 xgrammar 跳过，报告文案变更后其针对性测试再次通过；
作业脚本 bash -n 与 git diff --check 通过。未提交或推送 Git，未修改既有结果。

### 四个 MAB 全部完成

SH / MH / FCSH / FCMH 的三个 reader 均已完成全部六项消融，共 12/18 个任务-reader 组合。
再次使用原审核函数核对各结果的完整 100 题 ID、评分和 usage。下表为消融减旧主配置，单位百分点：

| 任务 | reader | 去 RRF | 去窗口 | 去附录 | 去 canonicalization | 去 latest-only | 去 discourse |
|---|---|---:|---:|---:|---:|---:|---:|
| SH | 9B | -2 | 0 | 0 | +1 | +1 | +1 |
| SH | 4B | -2 | 0 | +1 | +2 | 0 | +2 |
| SH | 2B | -5 | 0 | -4 | 0 | 0 | -1 |
| MH | 9B | +6 | 0 | +4 | 0 | -1 | 0 |
| MH | 4B | -2 | 0 | +3 | -1 | +2 | 0 |
| MH | 2B | -4 | 0 | 0 | +1 | -1 | 0 |
| FCSH | 9B | +3 | 0 | -19 | -5 | -4 | 0 |
| FCSH | 4B | +3 | 0 | -13 | 0 | +1 | 0 |
| FCSH | 2B | +1 | 0 | -9 | -8 | -4 | 0 |
| FCMH | 9B | +5 | 0 | +1 | -2 | 0 | 0 |
| FCMH | 4B | +5 | 0 | 0 | -2 | -1 | 0 |
| FCMH | 2B | +5 | 0 | -5 | 0 | 0 | 0 |

这不是统一删减的证据：去 RRF 对 FCSH / FCMH 是正差值，对 SH 是负差值；去附录对 FCSH
为负，而 MH 9B / 4B 为正。尚不按任务选择不同配置，不据此提前取消两个长任务。

最终报告增加对既有预测的只读诊断：逐任务列出有序 reader 上下文不变但官方评分变化的题数，
JSON 另保留回答字符串变化数。QA 上下文相同题数必须与完整检索记录一致，否则报告失败。
不修改官方分数，不排除题目，不做 per-case 预测替换，也不把该计数称为新指标或显著性检验。
已在当时完成的 11 个组合上真实通过一致性检查；其中 FCSH 去 canonicalization 的不变上下文评分
变化题数为 9B:8、4B:7、2B:16，再次说明图消融的小差值不能直接归因于图。
新增针对性测试覆盖文本顺序、回答变化但分数不变、缺失/重复 case、QA/检索计数不一致拒绝。
最新全测试 111 项，其中 108 通过、3 项 xgrammar 跳过；不需要改变或重跑任何 QA。

### 六任务检索全部完成

6354029_4（LoCoMo，21:57）与 6354029_5（2Wiki，24:10）均正常完成。全部 3386 题的主配置
普通检索与 no-RRF 参考有序文本精确相同，各 case 字段和遍历顺序一致。长任务上下文相同题数如下：

| 任务 | 总题数 | 去 RRF | 去窗口 | 去附录 | 去 canonicalization | 去 latest-only | 去 discourse |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | 1986 | 0 | 0 | 0 | 1539 | 810 | 1702 |
| 2Wiki | 1000 | 18 | 1000 | 0 | 979 | 899 | 998 |

六任务完整校验后，预定新增 QA 数为 45690，整任务复用 QA 数为 15258，合计六消融 x 三 reader x
3386 题 = 60948。这里只是调用计划与复用审计，不是宣称所有新 QA 已执行。
LoCoMo 的 6354124_0..2 和 2Wiki 的 6354125_0..2 已全部启动。为使用空余配额，2Wiki 的 2B
子作业在 pending 时转到 gpu-he，数组并发上限改为 3；其余 2Wiki reader 留在 gpu / norm-gpu。
scontrol 已确认 2B 的 Restarts=0，模型/输入/解码未改变，未取消或重复任何 QA。

去 RRF 的完整长任务 QA 已按预定规则复用，下面补全去 RRF 减主配置的百分点差值：

| reader | LoCoMo | 2Wiki |
|---|---:|---:|
| 9B | -1.662 | +0.690 |
| 4B | -1.771 | +0.345 |
| 2B | +0.014 | +0.133 |

结合四个 MAB，去 RRF 不是统一改进：对 SH 的三个 reader、MH 的 4B/2B、LoCoMo 的 9B/4B 为负。
不因 FactConsolidation 的正结果就全局删除 RRF，更不采用按任务切换策略。长任务其他五项消融 QA
仍在运行，完整报告尚未生成。

### 长任务首批完整单项结果

已按原函数核对下列完整任务的预测 ID、官方评分和 usage；不是未完成题目的临时平均：

| 任务 | reader | 消融 | 消融分数 (%) | 相对旧主配置 (百分点) | 输入 token：原 / 消融 |
|---|---|---|---:|---:|---:|
| LoCoMo | 2B | 去窗口 | 40.9488 | -2.3924 | 5672028 / 1841379 |
| 2Wiki | 2B | 去事实附录 | 45.8799 | +7.6753 | 3004624 / 1562155 |
| 2Wiki | 4B | 去事实附录 | 57.9823 | +1.9453 | 3004624 / 1562155 |

窗口在这个 LoCoMo 2B 完整运行中有正向差值；附录在 2Wiki 2B/4B 则有负向差值，与 FCSH 的
结果相反。上下文预算同时变化，不能称为等 token 比较，也不能按任务选择不同开关后声称统一算法。
其余长任务消融继续执行；这些单项结果不足以完成全局删减决策。

2Wiki 2B 作业 6354125_2 正常完成（16:07），六消融的全部 1000 题已审核，共 4000 次新 QA 和
2000 次预定复用 QA；总体进度为 13/18 个任务-reader 组合。图消融结果如下：

| 2Wiki 2B 消融 | 分数 (%) | 相对旧主配置 (百分点) | 不变上下文题数 | 其中评分改变题数 |
|---|---:|---:|---:|---:|
| 去 canonicalization | 38.6957 | +0.4911 | 979 | 8 |
| 去 latest-only | 38.9037 | +0.6991 | 899 | 8 |
| 去 discourse | 38.4957 | +0.2911 | 998 | 8 |

QA 中的上下文相同题数均与检索完成记录一致。原 2Wiki temperature=0 设置未改变，这里仍出现了
输入不变时的评分差异，因此不能把以上小幅正差值直接解释为删除模块的收益；该检查不改变原成绩。
LoCoMo 三个 reader、2Wiki 9B/4B 仍在运行，完整报告继续等待所有依赖。

### 用户调整范围：只推进 4B / 9B，并立即清理代码

用户明确停止关注 2B，当前目标改为 4B / 9B 的完整六任务，dataset、split、指标和任务数量不变。
只取消仍运行的 LoCoMo 2B 作业 6354124_2（32:52），其余 2B 已完成作业不重跑，任何产物都不删除。
停止后 LoCoMo 2B 保留：去窗口 1986 题、去附录 1986 题完整 QA；去 canonicalization 344 题及对应
344 条 usage，为未完成产物，没有伪造 summary。此为用户调整研究范围，不是把运行失败或负结果藏掉。

旧报告 6354131 在 pending 时取消；新报告 6355178 只依赖剩余四个 4B/9B 长任务子作业，并仍逐项
检查两个 reader 的全部六任务。第一次带旧 MAB 已完成 job ID 的提交被 Slurm 拒绝，未产生作业；
改为对剩余作业设依赖，已完成 MAB 结果仍由报告程序完整检查。报告 JSON / Markdown 显式写出 reader
和 task 范围，不再把缺少 2B 当作失败。新模型选择测试验证缺少被排除 reader 不妨碍报告，但目标
reader 的缺失结果仍失败；空、重复、未知 reader 选择拒绝。调度脚本不再提供 2B 数组项。

当前两个 reader 口径为 30460 次新 QA、10172 次复用，总计 40632 个任务内问题-消融-reader 组合；
这些是完整目标调用数，不是当前已执行量。早期 2B 研究成本和旧三 reader 调用计划保留在历史字段中。

按用户要求，第一批直接删除三个非主配置实验实现及其构建入口：

| 删除模块 | 退役配置数 | 删除模块源码行数 |
|---|---:|---:|
| graph_construction/weight_grid.py | 5 | 32 |
| graph_construction/provenance_weights.py | 3 | 42 |
| graph_construction/adaptive_synonyms.py | 3 | 47 |

三个文件原本是干净的已跟踪文件，历史实现可从 Git 提交 218a06a 取回。另移除 replay_graph.py 的
构建分支、参数与相关 metadata，run_graph.py 的对应导入 / CLI 候选入口，以及六个仅测试退役实现的
测试函数；测试文件中其他已有修改保留。旧 positional 构建开关明确拒绝，不静默运行另一套配置。
三个实现文件加 replay 构建入口共 163 行删除、6 行新增；这不是整个工作树的改动统计。
历史实验结果、冻结图、QA、负结果说明都保留，artifact loader 仍能读取已保存图。

这批是 **11 个非主路径实验配置退役**，不是主算法已经删除 11 个 heuristic。主配置六项消融的行为
没有改变。验证：105 项测试中 102 通过、3 项 xgrammar 跳过；保留的 CLI 全新进程 import/help 通过，
旧 positional 构建参数在执行前拒绝，源码无残留的退役模块 import。bash -n 与 git diff --check 通过。

额外提交 CPU 重建 6355268（1:44 正常完成），独立目录 optimization_ablation_cleanup_check_seed42_20260913。
全部 15 组、14362 来源的旧控制边权与 reader 字段精确重建；再逐项比较清理前后的六消融，共 90 组
variant-group，边权数组、完整来源表示、RRF 配置与 ablation statistics 均相等。不新增 QA，不改变正在
运行的 4B/9B 数据或预测。此时目标范围 9/12 组合完整：八个 MAB 加 2Wiki 4B；其余三作业继续。

### 继续压缩：删除主构造器内的闲置计算

第二批移除 `latest_relation_balanced`、`schema_latest_synonyms`、`canonical_latest_balanced`
三个历史输出和对应 balanced strength / synonym 恢复计算；每次构建只返回实际选择的一个权重数组。
来源三元组的规范化从重复调用改为一次计算后复用，不改变旧选择规则。累计退役 14 个历史配置，
仍不是从当前算法删除了 14 项独立 heuristic。

CPU 重建 6355420 正常完成（1:24），目录 optimization_ablation_constructor_cleanup_seed42_20260913。
15 组、14362 来源全部重建精确；90 组单消融边权、完整来源表示、RRF 配置及统计均与清理前相等。
清理后 106 项测试中 103 通过、3 项 xgrammar 跳过；随后更新报告默认只选择 9B/4B，
图构建、消融、报告的 59 项针对性测试再次全部通过。未改写任何既有图或 QA。

### 下一组：联合移除图筛选依赖链

用户继续要求压缩算法。固定 `without_graph_selection` 与 `without_selection_and_fact_appendix`
两个对照，详细定义见 `ablation_plan.md`。前三个图筛选步骤一起删除，来源支持权重计算不变；
后一组同时移除旧事实附录，使 reader 不再间接使用这些筛选。RRF、窗口与所有原评测协议固定。
这不是新增算法公式，不宣称联合删减的差值能分摊给每个模块，也不预设会更好。
两候选独立冻结、检索及报告，不将新变体插入仍在执行的第一阶段 QA；4B/9B 全六任务均保留。
实现复用现有消融入口，以 protocol 显式区分完整六单项与完整两联合项，拒绝混合或缺项配置。
109 项测试中 106 通过、3 项 xgrammar 跳过；新增测试验证无 schema 时仍保留旧事实与 discourse
支持、两个候选边权一致、原文与窗口仍在且无附录，以及旧/新实验范围互不混合。
联合构建提交为 6355727，目录 optimization_ablation_joint_removal_seed42_20260913，尚无 QA 结论。
6355727 已正常完成（1:24），全部 15 组 / 14362 来源主对照精确重建。逐组核对两个候选的边权
完全相同、所有三元组来源支持保留、RRF 不变；第二组原文 / 位置 / 时间 / 窗口来源与旧配置相同，
中心文本以完整原文结束，不再附加事实。第一组仍直接引用旧完整 reader 表示。SH 普通检索 pilot 为 6355810_0。
SH pilot 正常完成（0:58），100 题主对照检索与原字段、顺序一致。两个候选与主配置相同上下文题数
分别为 95 和 0，均不复用主配置的整任务 QA。其余五任务检索数组 6355843（索引 1..5，最多两并发）；
QA 数组按 SH / MH / FCSH / FCMH / LoCoMo / 2Wiki 为 6355844 / 6355858 / 6355859 / 6355860 /
6355861 / 6355862，每组仅索引 0=9B、1=4B。后五任务 QA 同时依赖自身完整检索和 SH 两个 reader
pilot 成功，报告 6355868 依赖全部六个 QA 数组。任务失败时不跳过依赖、不缩减任务范围。
补测联合报告会读入两个联合变体并明确标注其归因边界，三个报告测试通过。

SH 两个 reader pilot 已正常完成（9B 3:26、4B 3:02），逐题预测 ID、官方 summary 均值、完整 usage
与上下文诊断均通过审核。四个 MAB 的联合检索已完成，长任务检索继续，后续 QA 依赖已正常放行。

| SH 联合删减 | 9B 分数 / 差值百分点 | 4B 分数 / 差值百分点 | 每个 reader 输入 token |
|---|---:|---:|---:|
| without_graph_selection | 93 / +2 | 88 / -2 | 616406 |
| without_selection_and_fact_appendix | 89 / -2 | 92 / +2 | 276277 |

仅删除图筛选时，两个 reader 都有 95 题上下文不变，其中均有 4 题评分变化；因此 +2 / -2 的
差值不能直接归因为三项图筛选的作用。删除附录的候选全部 100 题上下文改变，且明显降低 token，
结果是 mixed，不提前称为统一改进；继续按既定范围执行，不按 reader 或任务挑不同方法。

继续按原函数审核长任务完整单项，以下均非未完成题目的临时均值：

| 任务 | reader | 删除项 | 相对旧主配置 (百分点) |
|---|---|---|---:|
| LoCoMo | 9B | 窗口 | -5.2647 |
| LoCoMo | 4B | 窗口 | -3.4363 |
| 2Wiki | 9B | 事实附录 | +1.6476 |
| 2Wiki | 9B | canonicalization | -0.1618 |
| 2Wiki | 4B | canonicalization | -0.1152 |
| 2Wiki | 4B | latest-only | -0.0802 |
| 2Wiki | 4B | discourse filter | -0.0152 |

窗口删减在目标 reader 的 LoCoMo 明显下降，不作为当前直接全局删除项；2Wiki 的图筛选微小差值
仍需结合不变上下文的 QA 波动判断。所有分数与完整预测 ID / usage 已核对，不作显著性声明。

### 消融报告保留完整 baseline 比较

继续阶段再次用完整预测核对既定九项 baseline，而不是只比较 HippoRAG2 或旧主配置。范围仍为
BM25、dense、HippoRAG2、Mem0、LightMem、LightMem offline、AnchorMem dense、AnchorMem official、CatRAG。
4B/9B 的每个任务均核对完整 case ID 集合、逐题官方分数与 summary 一致，缺失结果不补零。

| 任务 | 9B baseline 最佳 (%) | 9B 旧主配置 (%) | 4B baseline 最佳 (%) | 4B 旧主配置 (%) |
|---|---:|---:|---:|---:|
| SH-Doc_QA | 88 | 91 | 89 | 90 |
| MH-Doc_QA | 59 | 58 | 54 | 59 |
| FactConsolidation-SH | 66 | 67 | 56 | 60 |
| FactConsolidation-MH | 6 | 9 | 5 | 8 |
| LoCoMo | 55.3736 | 56.9728 | 50.3363 | 51.4586 |
| 2WikiMultiHopQA | 51.8735 | 56.5287 | 49.7646 | 56.0370 |

旧主配置仍是 9B 严格胜出 5/6、4B 6/6。报告入口现同时给出每个消融相对旧主配置的差值、相对
baseline 最佳值的差值及严格胜出任务数；JSON 保留每项 baseline 原分数与产物根目录。平分不算胜出，
不跨不同指标求平均，不将既定九项对照称为整个领域 SOTA，也不称为等 token 预算比较。
三个报告测试通过，覆盖正负差值、平分不胜出、缺失 baseline 拒绝以及单项/联合消融范围。
这次只扩充已有成绩的报告，不修改图、QA、官方指标或任何预测。

### 删除用完的一次性测试脚本

按用户要求，删除本轮新增且已完成验证的 `test_ablation.py`（150 行）和
`test_ablation_reporting.py`（126 行），共 276 行；不新增替代测试文件。两者没有被运行脚本或算法导入。
删除前最近一次完整验证为 109 项：106 通过、3 项 xgrammar 跳过；相关结果保留为历史验证记录，
不再声称当前目录仍包含这 109 项测试。原仓库已有的 `test_graph_construction.py` 保留不动。
运行中的建图、检索、QA 和报告不依赖这两个文件，实验产物与负结果不删除。

### 联合删减的短任务完成，暂不能替代主方法

联合组全部六任务普通检索已完成：LoCoMo 6355843_4（7:19），2Wiki 6355843_5（12:29）。
再次逐项核对全部 3386 case 字段与遍历顺序、主配置普通检索一致性和候选有序上下文相同计数，均通过。
两候选无任何整任务与主配置完全相同，因此全部 13544 次 QA 都是新调用；不是当前已执行量。
四个短任务的两个 reader QA 全部完成，完整预测、官方指标、usage 和上下文诊断均已审核。

| 任务 | 9B 仅删图筛选 | 9B 同时删附录 | 4B 仅删图筛选 | 4B 同时删附录 |
|---|---:|---:|---:|---:|
| SH-Doc_QA | 93 (+2) | 89 (-2) | 88 (-2) | 92 (+2) |
| MH-Doc_QA | 57 (-1) | 61 (+3) | 61 (+2) | 60 (+1) |
| FactConsolidation-SH | 63 (-4) | 43 (-24) | 61 (+1) | 46 (-14) |
| FactConsolidation-MH | 6 (-3) | 7 (-2) | 7 (-1) | 5 (-3) |

括号为相对旧主配置的百分点差值，列内固定同一方法，不按任务选择开关。仅删图筛选的版本，9B
在这四任务仅严格胜出既定 baseline 1/4，即使长任务全胜也最多 3/6；同时删附录的版本，4B 为 2/4，
最多 4/6。两者均不能达到当前两个 reader 同一配置的目标，不据此取消长任务或隐去负结果。
FCSH 的附录联合删除有大幅回退，不应称作无用模块；仅图删减的小差值仍有上下文不变时 QA 变化的干扰。

新增完整单项审计：LoCoMo 4B 去附录为 50.3371%，较旧配置 -1.1215 点，1986 次 QA / 5096058 输入 token；
2Wiki 9B 去 latest-only 为 56.3904%，差 -0.1383 点，1000 次 QA / 3014485 输入 token。后者 899 题上下文
不变，其中 14 题评分改变，不将微小差值直接归为图效果。

2Wiki 联合 4B 作业 6355862_1 原因 norm-gpu 配额待机，现仅将此未启动子作业转到空闲 gpu-he 配额；
scontrol 确认 Restarts=0，4 CPU / 64G / 原模型与原解码设置不变。其他已运行作业不动。
本次只审核结果与更新记录，没有新增算法、实验变体或测试脚本。

### 去事实附录：两个 reader 的六任务完整结果

该单项已在 4B/9B 全六任务完成，全部 6772 题预测、官方分数、usage 与九项 baseline 再次审核。
不是等待其他消融期间的部分题目均值，也没有把不同方法的任务结果拼接。

| 任务 | 9B 去附录分数 (%) | 相对主配置 (点) | 4B 去附录分数 (%) | 相对主配置 (点) |
|---|---:|---:|---:|---:|
| SH-Doc_QA | 91 | 0 | 91 | +1 |
| MH-Doc_QA | 62 | +4 | 62 | +3 |
| FactConsolidation-SH | 48 | -19 | 47 | -13 |
| FactConsolidation-MH | 10 | +1 | 8 | 0 |
| LoCoMo | 55.9022 | -1.0706 | 50.3371 | -1.1215 |
| 2WikiMultiHopQA | 58.1763 | +1.6476 | 57.9823 | +1.9453 |

两个 reader 按既定严格大于口径都为 5/6；4B 的 LoCoMo 优势仅 0.000823710455 个百分点，
不称为稳健优势或显著提升。每个 reader 全六任务 QA 输入 token 从 11092376 降为 7904546，约减少 29%。
这是输入预算和分数的取舍，不是纯图收益或等 token 对照。FCSH 存在大幅回退，不能把事实附录认定为无用，
也不能为了简化只保留 MH/2Wiki 的正结果。暂不替换完整主配置，其他单项继续按原计划收齐。

### 联合消融完整收尾，单项仅剩 LoCoMo

联合组 LoCoMo 9B/4B 作业分别正常完成于 38:27 / 29:21，2Wiki 为 47:05 / 36:04。
报告 6355868 正常完成（0:17），输出位于 optimization_ablation_joint_removal_seed42_20260913/ablation_results.md。
完整 JSON 再次核对六任务、两个 reader、九项 baseline 及 13544 次新 QA；无复用、缺项或按题拼接。

| 配置 | 9B 严格胜出 | 4B 严格胜出 | 每个 reader 全六任务输入 token |
|---|---:|---:|---:|
| 旧主配置 | 5/6 | 6/6 | 11092376 |
| without_graph_selection | 3/6 | 5/6 | 11059916 |
| without_selection_and_fact_appendix | 5/6 | 4/6 | 7916055 |

仅删图筛选的 LoCoMo 分数为 9B 57.2317%、4B 51.7968%；同时删附录为 56.6341%、50.7726%。
2Wiki 对应为 56.3904% / 55.8208%，以及 58.4120% / 57.9889%。这些长任务结果未扭转短任务上的
负结果，两套联合方案均不作为两个 reader 的统一替代。不将某个 reader 的优势推广到另一 reader。

单项 2Wiki 9B 已正常完成（6354125_0，1:38:57），目标范围因此为 10/12 组合完整，仅剩 LoCoMo。
去 discourse 的 2Wiki 分数为 9B 56.4669%、4B 56.0218%；差值分别 -0.0618 / -0.0152 点。
998 题 reader 上下文不变，其中 9B 有 14 题、4B 有 10 题评分改变，微小负差值不能直接归因为该过滤。

LoCoMo 去 canonicalization 的完整结果为 9B 57.3076%（+0.3348 点）、4B 51.4742%（+0.0156 点）。
1539 题上下文不变，其中分别有 336 / 298 题评分变化，因此不把这两个微小正差值作为模块无用的证据。
所有新增完整单项均已核对原指标、完整 ID、usage 与检索/QA 上下文计数；原预测保持不变。
继续等待 LoCoMo 的 latest-only / discourse 结果，不新增代码或变体。

### 主路径 legacy 清理与全量等价验证（2026-09-14）

清理前已将整个 optimization 代码目录（含未跟踪源码，不含 pyc）保存到仓库外：
`/oscar/scratch/zliu328/agent-memory-outputs/optimization_code_before_legacy_cleanup_20260914.tar.gz`。
这不是图或预测的备份替代品；原图、embedding、QA、schema 和失败产物均未删除。

以该快照为基准，统计 `.py` / `.sbatch` / `.ebnf` 的实际文本行：
61 文件 / 5751 行 -> 28 文件 / 2408 行，净减 3343 行（58.1%）。
该口径包含实验入口和已有测试，不包含 Markdown、结果、缓存；
此前删除的一次性测试已不在快照中，不重复累加。核心目录目前为 5 个建图文件、3 个检索文件，
不计 `__init__.py`；消融辅助从 `graph_construction/ablation.py` 移到根目录 `ablation.py`。

本轮实际移除 gist、fact incidence/index、passage/dense 索引、rank-window、packing、
关系 identity/update-policy 的历史实现与专用入口/测试，以及旧 context weighting 路径。
主 loader 移除对应分派；旧专用 artifact 明确拒绝，而非悄悄改用当前图。
继续删除未被主配置使用的 state/event 选择和 schema/state-event/逐行句子读出选项。
对应已退役测试及 30 个失去源文件的 pyc 删除；不新增测试脚本。
这不是删除了同样数量的有效主机制，六项主消融机制仍待证据决定。

普通查询验证：SH 6358089、MH 6358241、FCSH 6358242、FCMH 6358243、
LoCoMo 6358244、2Wiki 6358245 全部 COMPLETED 0:0。
分别为 100/100/100/100/1986/1000 题，每题有序 reader 文本与旧主配置相同。
此前 6358054 因传入 `SH-Doc_QA` 而非 CLI 要求的 `SH-Doc QA` 在 2 秒内失败；
纠正命令参数后重提，没有改变 loader 或运行失败任务的 QA。

CPU 6358294 完成（0:0，1:27），现有 build 入口重新构建全部 15 组 / 14362 来源，
主图权重和旧读出字段完全匹配。随后终端内联逐项核对全部 90 个消融 variant/group：
权重、完整 contents、来源顺序、RRF、图 metadata（输出 contents 路径除外）及消融统计全部一致。
输出为 `optimization_ablation_legacy_cleanup_seed42_20260914`；新增 QA 为零。
剩余已有测试 26 项全部通过，report CLI 新进程导入与三个 sbatch 语法检查通过。
原报告的 109 项是历史计数，不再作为当前测试数量。

schema 启动脚本仍引用已消失的共享 vLLM 入口，现修正为已存在的
`/oscar/scratch/zliu328/agent-memory-envs/vllm_cu129/bin/vllm`，保持原生成参数不变。
本轮只检查路径与脚本语法，没有重新启动生成器，不宣称新完成了 GPU 生成验证。

README 已改成当前方法/入口说明；novelty 文档删除把 state/event、gist、incidence 当主线的叙述，
转为来源支持物化视图、键归一与覆盖顺序、计数投影的解释。引用 MemorySSA、DBSP、
provenance semirings 与 spectral sparsification 时逐项说明未实现或不成立的保证，
不增补公式、方法、数据规则或理论声明。

仍未结束：三段旧构建入口 replay/compile/hybrid 需要合并并全量验证，
其纯三元组/adaptive/cardinality 历史选项仍待一并退役；单项 LoCoMo 尚未全部完成。
因此不能宣称 legacy 已全部清理，不能把 3343 行净减解释为已证明主机制可删。

新增完整结果：LoCoMo 4B 去 latest 为 51.7274%，相对主配置 +0.2687 个百分点，
1986 次新 QA / 5652728 输入 token。810 题上下文不变，其中 162 题评分变化；
已检查完整 ID、原评分、usage 和上下文计数，不据小幅正差值直接认定 latest 无用。

### 合并历史构建链，保留单一主入口（2026-09-14）

删除 replay_graph、compile_graph_context、prepare_hybrid_graph 及三个 sbatch，
改为 `build_graph.py` / `build_graph.sbatch`。直接使用原 Hippo 索引、原 loader 的来源顺序与 schema，
生成主图和固定 reader 的原图对照；不再需要旧 query resets、检索排名、adaptive 图或已编译读出。
仍复用原 loader，不新增数据转换；构建不使用 case 的问题、答案或指标来选择来源支持。

构造器删除 cardinality 选择分支及候选名称字典，直接返回当前权重向量；
读出只保留原文加 JSON 附录与消融所需的无附录表示，纯三元组路径删除。
schema 提示及输出中的 cardinality 暂不改，以免代码清理同时改变 LLM 输出。
`run_graph.py` 补上普通 retrieve 阶段接到原 QA 入口；不是新检索策略。
默认缓存未命中明确失败，显式允许在线生成时也保留原 provider 失败保护。

当前 `.py` / `.sbatch` / `.ebnf` 为 24 文件 / 1973 行，相对清理前快照净减 3778 行（65.7%）；
比上轮 2408 行继续净减 435 行。这些计数包含新构建入口、检索入口及保留测试，
不靠把旧实现移动到其他源码目录减小分母。三个旧驱动的 pyc 同步删除。
当前已有测试 22 项全通过，全部 Python 模块语法、四个 sbatch 语法及 CLI 导入检查通过。

CPU 全部在 batch：pilot 6358497_0 完成 0:37；其余五任务 6358518_1..5
分别完成 0:28、0:35、0:32、0:39、0:29。全六任务 15 个语料组的两种图共 30 组，
边权、完整来源顺序、原 graph/RRF metadata 和旧读出字段均逐项与历史冻结结果相同。
独立输出 `optimization_graph_build_cleanup_seed42_20260914`；图验证没有生成新 QA。

单项消融重建 6358504 完成 1:17，终端内联核对 90 组权重、完整 contents、来源顺序、
RRF 与消融统计完全相同。联合消融重建 6358593 完成 1:12，随后逐项核对 30 组对应产物也完全相同。
对应输出为 `optimization_ablation_consolidated_builder_seed42_20260914` 和
`optimization_joint_consolidated_builder_seed42_20260914`。没有新增测试文件或实验配置。

普通检索同时验证主图与原图对照：SH 6358519 完成 0:36；MH 6358576 完成 0:58；
FCSH 6358577 完成 0:45；FCMH 6358578 完成 0:43。四任务每个图均完整 100 题，
case 全字段与遍历顺序、有序 reader 文本和旧结果匹配。LoCoMo 6358579 完成 3:59，
两种图各 1986 题完整顺序与读出均核对相同。2Wiki 6358580 完成 8:02，
两种图各 1000 题的完整顺序与读出均核对相同。至此六任务、两种图各 3386 题普通检索
全部通过；没有复用旧排名来生成这些新检索记录，也没有重跑 QA 后挑分数。

当前不存在旧三段构建链的源码引用；核心仍为 5 个建图文件、3 个检索文件。
六组消融机制内部的例子选择、canonical 合并、支持投影、RRF 与窗口规则已列入 ablation_plan，
不把“六组模块”当成全部 heuristic 的精确计数，也不把构建入口简化算作提分。
单项 LoCoMo QA 仍等待原作业，不因等待时间长而重启。

### 去 latest：两个目标 reader 的全六任务已收齐

完整核对 6772 题预测的原指标与 ID、每份 QA usage，以及九项 baseline 的任务分数。
9B / 4B 的严格胜出分别为 4/6、6/6；该统一配置未维持两个 reader 都至少 5/6。

| Reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki |
|---|---:|---:|---:|---:|---:|---:|
| 9B 去 latest 分数 (%) | 92 | 57 | 63 | 9 | 57.1284 | 56.3904 |
| 相对主配置 (百分点) | +1 | -1 | -4 | 0 | +0.1555 | -0.1383 |
| 4B 去 latest 分数 (%) | 90 | 61 | 61 | 7 | 51.7274 | 55.9568 |
| 相对主配置 (百分点) | 0 | +2 | +1 | -1 | +0.2687 | -0.0802 |

LoCoMo 两 reader 各 1986 新 QA / 5652728 输入 token；9B 输出 22570 token，
QA 调用 3088.60 秒；4B 输出 19504 token，调用 2545.11 秒。
两 reader 都有 810 题上下文不变，其中各有 162 题评分变化；不把微小涨分视为因果改进。
单项矩阵当前仅剩 LoCoMo 的 discourse；原 6354124_0/1 继续执行，不重启。

代码精简与性能取舍不是同一问题。“无损”不是本项目新添加的硬性评测指标。
去附录已有两 reader 5/6 的完整结果，但 FCSH 大幅回退且 4B LoCoMo 优势极小；
已向用户询问是否接受这种取舍，未回答时保留旧主配置，继续核对现有消融，不另开参数搜索。

### 当前报告范围与 discourse 的双重用途

旧 `report_results.py` 默认仍遍历三个模型，现与单消融报告统一默认 9B/4B；
历史 2B 可通过 `--models` 显式读取，不启动新的 2B QA。既有报告测试同时检查默认范围和显式历史范围，
22 项测试通过。此次报告与测试共增加 8 行，无算法变化；当前 24 个代码/脚本文件共 1981 行，
相对归档净减 3770 行。之前 1973 行是构建链清理完成时的快照，不再作为当前行数。

CPU `6359164` 在 batch 完成（0:40）。使用现有选择/序列化/窗口函数，先逐来源重建并验证旧读出，
再检查对附录也设置 `drop_discourse=False` 的影响；不写新图、QA 或实验变体，不读取答案进行选择。
只使用已完成的图-only discourse 消融排名，核对每个 source_passage 与旧读出，再替换其离线表示作比较。

| 任务 | 来源数 | 改变的来源读出 | 若附录也取消过滤，改变上下文的题数 |
|---|---:|---:|---:|
| SH-Doc_QA | 412 | 85 | 35 / 100 |
| MH-Doc_QA | 875 | 27 | 13 / 100 |
| FactConsolidation-SH | 537 | 0 | 0 / 100 |
| FactConsolidation-MH | 537 | 0 | 0 / 100 |
| LoCoMo | 5882 | 519 | 849 / 1986 |
| 2WikiMultiHopQA | 6119 | 36 | 33 / 1000 |

共 667 条来源读出、930 题上下文改变。这是输入等价性检查，不是新评分指标或 QA 效果。
即使当前只删图过滤的分数保持，也不能据此把整个 discourse 机制从构建器中删除：
原附录仍使用该规则。若后续选择整项删除，需要图与附录同时取消过滤的完整验证。
两个 FactConsolidation 任务与现有图-only候选的完整输入相同，其余四任务不相同；
按既定整任务复用口径，后续该候选需 6372 次新 QA，而不是只补 930 题或按题拼结果。
这是依赖范围推导，不是已经提交的 QA 计划；当前原 discourse 单项仍在运行，配置不变。

### 4B 的六项单消融全矩阵完成

LoCoMo 4B 作业 6354124_1 正常完成（3:20:46）。CPU 审核 6359282 在 batch 完成（0:13），
使用现有 report_ablation 的完整检查，保留真实预测、指标、usage、baseline 和上下文读取，
只在内存中接收报告输出，不用单 reader 报告覆盖等待中的双 reader 最终报告。

| 4B 配置 | 严格超过九项既定 baseline 的任务数 | 本轮新 QA | 全六任务输入 token |
|---|---:|---:|---:|
| 完整主配置 | 6/6 | 既有控制 | 11092376 |
| without_rrf | 4/6 | 0，完整已有对照复用 | 11606137 |
| without_window | 5/6 | 1986 | 7261727 |
| without_fact_appendix | 5/6 | 3386 | 7904546 |
| without_canonicalization | 6/6 | 3386 | 11088940 |
| without_latest | 6/6 | 3386 | 11076037 |
| without_discourse_filter | 6/6 | 3086 | 11087765 |

“新 QA”是这一轮消融新调用量，不是方法本身的总成本；复用记录仍计入方法的输入 token。
去 discourse 的 LoCoMo 4B 为 50.9553%，比主配置低 0.5033 个百分点，仍高于最佳 baseline 0.6190 点。
1986 次新 QA / 5667705 输入 / 19516 输出 token，调用 2551.30 秒。
1702 题上下文不变，其中 692 题回答字符串改变、361 题评分改变，不据单次差值声称显著有害或无效。

三个图单项各自 6/6 不意味着可以同时删除：已有联合删图筛选的 4B 只有 5/6，9B 只有 3/6。
9B 单独去 canonicalization 或 latest 均只有 4/6；不按 reader 选择不同算法拼成统一成绩。
当前完整范围为 11/12 个 reader/task 组合，唯一未完成的是 LoCoMo 9B 的 discourse。
最终报告 6355178 仍等待该原作业，未重启、取消或改变运行配置。

### 补齐 discourse 的整项删除对照

依赖检查改变了后续动作：现有图-only 单项无论最后分数如何，都不能单独证明整个规则可删。
因此不再仅等待原 9B 作业，而是新增唯一的整项删除 `without_discourse_selection`，
同时取消图与附录的 discourse 过滤。此操作复用原选择函数的 `drop_discourse=False`，
不改变当前主算法、既有两个消融组、指标、数据范围、生成提示或运行中的 QA。
现有三个消融辅助/入口文件共增加 15 行，不新增模块或测试文件；当前代码/脚本 24 文件、1996 行，
相对归档净减 3755 行。原 22 项测试通过，终端额外检查新旧组不能混用、图权重与既有图-only移除相同。

输出为 `optimization_ablation_without_discourse_seed42_20260914`。
batch 构建 6359366 完成 1:14，完整旧主图与旧读出重建一致；batch 审核 6359472 完成 0:44。
审核在全部 15 组 / 14362 来源上仅提供 canonical 字段，确认图和附录不需要读取 role 或 cardinality，
边权与现有 graph-only discourse 消融完全一致；新 contents 全字段与关闭过滤的原函数输出相同。
各任务改变的来源读出与此前审计一致：85、27、0、0、519、36，共 667 条。

SH 普通检索 pilot 6359473_0 完成 0:58。旧主配置 100/100 题匹配；新图排名与旧 graph-only
候选相同，全部 case 字段和顺序一致，reader 文本有 35 题变化，与 CPU 预期一致。
新候选与完整主配置也有 35 题不同，因此 SH 必须完整重评 100 题，不能复用其中 65 题的 QA。

两次后续提交因引用已从 Slurm 实时表移除的完成 pilot 而被拒绝，未创建任何作业或 QA。
scontrol 确认该实时条目不存在，sacct 和完整产物确认成功后，直接提交后续，不重跑 pilot。
其余任务普通检索为 6359528_1..5；SH QA pilot 为 6359529_0/1，分别对应 9B/4B。
后续 QA 必须通过其完整检索与 SH reader pilot 检查。预计整任务复用 400 条、新 QA 6372 条，
最终以完整输入比较为准；不是已执行调用量。原 6354124_0 与最终报告 6355178 保持原状。

另核对 QA 随机性来源：原 `_official_generation` 为 LoCoMo 使用 temperature=0.4、top_k=10、
top_p=0.9；MAB 使用 temperature=0.7；2Wiki 使用 temperature=0。`HuggingFaceChatModel.answer`
在温度为正时启用采样，当前每个候选开头重置 seed，不在每题前重置。
因此不能假定相同上下文的题会进入相同的随机状态；前面题目的采样与生成长度变化可能传播到后续题。
这是代码结构支持的解释，不证明所有评分变化都来自它，更不全部归咎于硬件。
没有改动温度、seed、采样、评分或增加按题种子规则，保留原协议与全部已有预测。

### Discourse 整项删除：全量检索通过，剩余 QA 已提交

6359528 的其余五任务检索全部完成，LoCoMo 为 4:15、2Wiki 为 5:59。
预定的检索汇总审计未留下 Slurm 作业或输出；确认后提交 batch 6359667，完成 0:03。
该内联审计不新增脚本文件：核对全 3386 题字段与顺序、top5、与图-only 版本相同的
中心来源排名及分数、与已有主配置的文本等价性，以及两个 SH reader pilot 的完整预测和 usage。
与图-only 版本相比，上下文变化仍为 35、13、0、0、849、33 题，与此前源表示审计一致。
与主配置的相同上下文题数则为 65、87、100、100、1079、965，两个口径不混用。

SH pilot 6359529_0/1 均正常完成：

| Reader | 分数 | 相对主配置 | QA 输入 / 输出 token | QA 调用秒数 |
|---|---:|---:|---:|---:|
| 9B | 91% | 0 点 | 617418 / 897 | 137.05 |
| 4B | 87% | -3 点 | 617418 / 1255 | 117.19 |

4B SH 低于既定最佳 baseline 的 89%，不能把整项删除宣称为无损简化。
各有 65 题输入上下文不变，其中 9B/4B 分别有 3/5 题评分变化；仍不把单次差值当因果证明。
该负结果保留，继续完成预定六任务，不按 pilot 分数丢弃任务或选择复跑。

剩余 QA 已提交，统一依赖审计 6359667 成功：

| 任务 | 9B/4B 数组 | 分区 | 工作 |
|---|---|---|---|
| MH-Doc_QA | 6359675_0/1 | gpu-he | 全任务新 QA |
| LoCoMo | 6359676_0/1 | gpu-he | 全任务新 QA |
| 2WikiMultiHopQA | 6359679_0/1 | gpu | 全任务新 QA |
| FactConsolidation-SH | 6359680_0/1 | batch | 完整旧 QA 核验与整任务引用，不加载模型 |
| FactConsolidation-MH | 6359681_0/1 | batch | 完整旧 QA 核验与整任务引用，不加载模型 |

输入等价检查确认本候选总计划为 6372 新 QA、400 复用，非当前已完成调用量。
最终完整报告 6359682 在 batch 等待以上五个数组，SH pilot 已由先行审计覆盖。
随后四个 CPU 复用子作业均以 0:0 完成（各 1-2 秒），四份 qa_complete 已核验：
共 400 条整任务引用、零新调用。MH、LoCoMo、2Wiki 的六个 GPU 子作业均已进入 RUNNING。
原单项 9B 6354124_0 仍正常运行，当前已写 1838/1986 条 discourse LoCoMo 预测；
原汇总 6355178 继续等待，不重启。主算法、参数、提示、采样和评分均未改变。
此次仅更新记录与作业清单，代码仍为 24 文件 / 1996 行，相对归档净减 3755 行。

### 六项单消融的双 reader 全量报告完成

最后的 LoCoMo 9B 作业 6354124_0 正常完成（3:56:50），没有重启或挑选复跑。
batch 报告 6355178 完成（0:29）：所有六项删减、六任务、4B/9B，逐份核对预测覆盖、
原指标、九项 baseline、QA usage 和不变上下文诊断。报告根目录为
`optimization_ablation_case_serialization_seed42_20260913/ablation_results.json`，同目录有 Markdown。
结构与成本总量再次核对通过：30460 次新 QA、10172 次已声明的整任务复用。

| 配置 | 9B 严格胜出 | 4B 严格胜出 |
|---|---:|---:|
| 主配置 | 5/6 | 6/6 |
| 去 RRF | 5/6 | 4/6 |
| 去窗口 | 4/6 | 5/6 |
| 去事实附录 | 5/6 | 5/6 |
| 图侧去 canonicalization | 4/6 | 6/6 |
| 图侧去 latest | 4/6 | 6/6 |
| 图侧去 discourse | 5/6 | 6/6 |

最后收齐的图侧 discourse LoCoMo 9B 为 57.1745%，相对主配置 +0.2016 点，
比最佳既定 baseline 高 1.8009 点。1986 次 QA、5667705 输入 / 21948 输出 token，
调用 3063.13 秒；1702 题上下文不变，其中 688 题回答改变、384 题评分改变。
4B 对应分差为 -0.5033 点。因此保持胜出任务数不等于所有题或任务分数不变，也不是显著性结论。

删减决定已写入 ablation_plan：RRF、窗口、canonicalization、latest 暂保留；
去附录保留为成本/效果取舍，不称为冗余。discourse 是当前最直接的整项删除候选，
但图-only 结果不能删除附录中的同一逻辑，继续等待已启动的图与附录整项移除。
该组 MH 的两个作业已正常完成（2:48 / 2:25），完整报告仍等待 LoCoMo 与 2Wiki。

当前理论解释随证据收窄：关系键归一与来源覆盖是相互依赖的支持选择，
RRF、窗口、附录是额外系统组件，不写成六个独立创新；不把图侧 discourse 宣传为必要核心。
没有新增算法、参数、测试文件或实验组；现存代码再次统计为 24 文件 / 1996 行。

### 等待整项删除期间：核对近邻机制，收窄贡献描述

重新阅读 HippoRAG2、A-MEM、GraphRAG 原论文的方法节，并对照本地
`add_fact_edges`、`add_passage_edges` 与当前 `latest_relation_weights`。
结论与原文链接已补入 research_novelty：离线图、来源关联和重复关系计数都不是独有创新。
当前差异应写为计数前基于关系键的来源支持选择，不将 A-MEM 错写成只关心 agentic retrieval。
该文献核对不增加 baseline 成绩，也不把不同论文的任务/reader/指标拼入当前九项比较。

另确认实现只保留原拓扑并替换等长边权数组，零权重不等于物理删除边。
因此不宣称已实现图存储压缩、索引内存下降或 PPR 加速；代码清理行数与这些成本无关。
不为该结论新增公式、数据规则或测试文件，正在运行的 QA 与原主配置不变。

### 整项 discourse 删除完成，采用精简配置

完整报告 6359682 在 batch 正常完成（0:14）。两 reader、六任务的预测、原指标、
九项 baseline、usage 和上下文诊断全部通过，确认 6372 次新 QA / 400 次整任务复用。
完整报告位于 `optimization_ablation_without_discourse_seed42_20260914/ablation_results.json`，
同目录有 Markdown；不替换或删除旧主配置的 5/6、6/6 结果。

| Reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9B 去整项 discourse (%) | 91 | 58 | 67 | 9 | 57.0747 | 56.4669 | 5/6 |
| 相对旧主配置 (百分点) | 0 | 0 | 0 | 0 | +0.1019 | -0.0618 | |
| 4B 去整项 discourse (%) | 87 | 59 | 60 | 8 | 51.2541 | 55.9857 | 5/6 |
| 相对旧主配置 (百分点) | -3 | 0 | 0 | 0 | -0.2045 | -0.0513 | |

每 reader 全六任务输入 11107895 token，比旧主配置增加 15519；不是上下文压缩。
LoCoMo 9B/4B 作业耗时 25:14 / 19:45，2Wiki 为 27:21 / 20:16；
运行节点不同，不能把这些墙钟差异归因于方法加速。
按当前“做薄且双 reader 至少 5/6”目标选择这一统一配置，接受披露的回退；
既不是全领域 SOTA、全面更好，也不是证明分类规则无损冗余。

### 删除规则与已完成的消融执行代码

先将完整现存 optimization 目录（含未跟踪文件，不含 Python 缓存）归档并核对清单：
`/oscar/scratch/zliu328/agent-memory-outputs/optimization_code_before_discourse_removal_20260914.tar.gz`。
三组消融均完成后，删除五个执行文件：ablation、prepare_ablation、evaluate_ablation、
report_ablation 及 run_ablation.sbatch。旧指标实现、QA 和报告不变，归档可复现旧实验。
不将这些退役逻辑搬入别的算法文件。

主支持选择删除 discourse 过滤和相应统计，删除仅供消融的 latest_only 开关，
保留对所有关系的 canonical-key/latest 选择。读出删除未采用的 facts_format=none 分支。
两个退役分支的测试删除；既有 canonical 测试改用仅 canonical 的 schema，
已有类别测试确认 role/cardinality 不再影响支持选择，没有新增测试文件。
原 schema 生成提示和验证协议保留，避免代码清理同时改变 canonical 输出；不宣称省去历史生成成本。

当前生产配置改名为 `canonical_latest_rrf_window`，与旧主配置区分。
代码由 24 文件 / 1996 行降为 19 文件 / 1447 行，本轮净减 549 行；
相对最初 61 文件 / 5751 行归档，累计净减 4304 行。
按原六组可消融机制的口径，删除了 discourse 一组；仍有归一、latest、RRF、窗口、附录，
不把这五组误称为所有人工选择的精确总数，支持投影等选择继续披露。

batch 测试 6360030 完成（0:02），20 项现有测试全部通过。
重建 pilot 6360031_0 完成 0:26，其余 6360034_1..5 均正常完成（25-44 秒）。
batch 全量审计 6360107 完成 0:09：15 组、14362 来源，主图权重和全部 contents 与获选消融精确相同；
原图对照权重与原图 pickle 相同，两图的源顺序、RRF、冻结/边序元数据均一致。
SH 普通检索 pilot 6360126 已启动，同时验证新主图与相同新读出的原图入口。
后续需完整验证新主图的普通检索，并为新读出下原图补齐 QA；
不把旧事实读出下的纯图分差自动转移到新方法。这是固定读出对照，不新增方法或 heuristic。

SH 检索 6360126 完成（1:30），batch 审计 6360166 完成（0:01）。
主图 100 题全字段、源排名/分数、reader 文本与获选消融一致；原图排名/分数与旧原图一致，
但有 36 题新读出不同。故提交原图 SH 完整 QA pilot 6360175_0/1，不按题补 36 条。
其余五任务检索为 6360169、6360170、6360171、6360172、6360173，均依赖 pilot 审核成功后启动。
batch 全量检索审计 6360181 依赖以上作业：逐题核对新主图与获选版本，以及原图与旧原图的
完整输入/源排名；只有全任务输入与既有 QA 实际上下文相同才建立整任务引用。
该审计还核对原指标、usage 和预测 ID/顺序；不按分数挑选，不创建新的测试脚本。
新原图的非等价任务需完整 QA，具体范围以这次检索核验为准。

### 精简实现的全任务普通检索验证完成

其余五任务全部完成：MH 2:18、FCSH 0:36、FCMH 0:38、LoCoMo 3:53、2Wiki 7:02。
batch 全量核验 6360181 正常完成（0:07）：3386 题的精简主图源排名、分数、
完整 case 字段及顺序、reader 文本均与获选整项消融一致。
原图的新旧源排名/分数亦一致；新读出改变的题数依任务顺序为 36、13、0、0、780、38。

所有整任务引用均在核对其源 QA 的预测 ID/顺序、实际上下文、原指标和 usage 后建立：
新主图的 6772 条既有 QA 全部引用，原图两项 FactConsolidation 的 400 条 QA 引用。
引用为完整任务/reader 目录链接，并有 qa_reuse.json，未按题拼接或按分数选择。
因此删除代码没有引入新主图的额外 QA；原图固定读出对照需要 6372 次新 QA。

原图 SH pilot 两 reader 正常完成（2:34 / 2:08），batch 审计 6360240 完成（0:02）。
9B/4B 分数为 94% / 86%，输入各 616958 token，输出 933 / 1101，调用 137.71 / 113.88 秒。
相对新主图的 91% / 87%，这部分图对照分别为 -3 / +1 点；负差值不隐去。

全量检索审计和两个 reader pilot 都通过后，提交原图剩余完整 QA：
MH 6360275_0/1、LoCoMo 6360276_0/1、2Wiki 6360277_0/1。
最终报告 6360278 在 batch 依赖这三个数组，使用现有 report_results 对两图、两 reader、
六任务与九项 baseline 审核；不创建新报告脚本。剩余工作是原图 QA 与最终贡献边界汇总。

### 本轮收尾：新读出的原图对照完成

原图最后三个任务全部正常完成：MH 6360275_0/1 为 2:58 / 2:25，
LoCoMo 6360276_0/1 为 25:11 / 19:37，2Wiki 6360277_0/1 为 27:47 / 20:34。
最终报告 6360278 在 batch 完成（0:10），输出到
`optimization_canonical_latest_cleanup_seed42_20260914/comparison.json` 和 `results.md`。
batch 追加核验 6360474 完成（0:08）：全部 24 个任务/reader/图组合，13544 条预测的
ID、顺序、实际 reader 上下文、原指标与 usage 对应一致；所有整任务引用路径正确。
共 6372 次新原图 QA、7172 次整任务复用（6772 次主图和 400 次原图），没有按题拼接。

| 固定检索/读出规则，主图减原图 (百分点) | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki |
|---|---:|---:|---:|---:|---:|---:|
| 9B | -3 | 0 | +3 | +3 | -0.7070 | +6.7925 |
| 4B | +1 | +6 | +1 | +5 | -0.1976 | +7.7070 |

原图对九项既定 baseline 为双 2/6，精简主图为双 5/6。
原图全任务输入各 10971131 token，主图各 11107895，差 136764；
两图排序不同带来不同长度来源，不将固定读出规则称为严格等 token 比较。
9B SH 和两个 reader 的 LoCoMo 退步保留，不宣称全任务优于原图或统计显著。
这些是支持选择/投影阶段的整体条件差异，不能单独归因于某个子规则。

本轮完成范围：六项单消融、两项联合消融、整项 discourse 删除；据此采用统一精简配置，
删除规则的使用和退役执行分支；保留 19 个代码/脚本文件、1447 行，20 项现有测试通过。
全部 15 组图/14362 来源和 3386 题普通检索验证一致，最终两个 reader 的完整 QA/对照通过。
记录与负结果、两份代码归档保留；没有新的测试文件、数据切分、指标、权重公式或任务路由。
理论解释收敛为来源支持物化、关系键归一与覆盖顺序、计数/存在性投影及其限制，
不宣称新图空间、程序语义保持、图物理压缩或独立测试泛化。
这是当前消融证据下完成的精简版，不是证明所有可能组合中全局最简，也不是全面 SOTA。

## 后续如何记录与选择

### 2026-09-14：新增目标下的结构探索

用户将目标扩展为可用 fancy concept/数学结构探索。当前完整索引消融继续，不在运行中叠加变量。
重新核对 DBSP 的 Z-set/增量视图维护、Kivela 等的多层网络，以及当前引用的 Kumar 约简。
research_novelty.md 新增具体操作、下一项实验与不能主张的能力：优先研究共同事实支持下的
传播图/候选索引两个派生视图，以及事实投影/额外来源连接两种贡献的条件作用。
若当前索引结果完整有效，下一项应固定新候选索引比较原/新图，避免把候选变化的收益全归给投影。
增量维护仅作为后续可实现方向，目前仍是一次离线构建；不引入负 PPR 权重、混合系数扫描、
新的冲突检测规则或未实现的 PL 正确性保证。文献引用及边界见 research_novelty.md。

### 2026-09-14：单独筛选事实候选索引

后续固定索引的构图对照：`canonical_latest_retained_index_rrf_window`，输出根
`optimization_original_retained_fact_index_seed42_20260914`。只将传播图换回同 H100 旧 refinement
版本，候选文字/ID/顺序/向量、来源映射、reader 与新构图 retained-index 组相同。
复用该组已完成检索的 recognition 缓存，不新增 LLM 调用；缓存缺失明确失败。
先核对全部 15 组旧图权重、候选和读出，再执行每任务首题 pilot 与完整普通检索、双 reader QA。
这一格用于区分候选筛选和事实投影作用，不改变默认、不按结果选择题或改 QA 协议。
复用 build_graph 的旧图路径，仅开放既有 retained-index 开关；不新增独立执行脚本。

6389173 的 28 项回归测试通过，0:0（2 秒）。构建 6389189_0-5 全部完成（31-48 秒）。
全量核验 6389227 为 0:0（13 秒）：15 组 / 14362 来源 / 167074 候选，旧图权重、
新旧两组读出及候选顺序均一致；用 SQLite backup 复制关闭的 recognition 缓存并逐行比对。
首批检索 6389238 在发现 pilot 的 get_all_id_to_rows 不包含 embedding 后取消：
五个子作业已分配 9 秒，一个未启动，尚未进行正式检索或 QA；不是按分数取消。
改用原 get_embeddings(keys) 接口后重新提交 6389256_0-5，不改变图、向量或任何查询参数。
SH/MH/FCSH/FCMH 均已完成原顺序首题 pilot 和完整检索（53-59 秒），无新增识别调用。
LoCoMo 与 2Wiki 尚在执行。全量检索核验为 6389275；QA 数组依次为
6389283 / 6389284 / 6389285 / 6389286 / 6389287 / 6389288（0=9B、1=4B），
全部等待核验通过。最终报告 6389297 同时等待本轮 QA 和新图 retained-index 报告 6387725，
原四格与新图索引结果只整目录引用，不逐题拼接。所有一次性检查仍通过 stdin 提交。
随后 LoCoMo / 2Wiki 普通检索分别完成（3:01 / 3:50）；6389275 为 0:0（3 秒），
全 3386 题的原 case 顺序、五中心、冻结读出与零新增识别调用均通过。
旧图同索引 2Wiki 来源 Recall@5 为 0.859，Precision@5 为 0.416；新图同索引为
0.87425 / 0.4252。此差异是证据检索指标，不提前替代尚待完成的 QA 比较。
旧图同索引的 9B MH / FCSH 已完整完成，分别为 59 / 64，未严格超过对应基线 59 / 66；
因此该配置最多为 4/6，不可能替代新图同索引的双 6/6。剩余任务继续原样运行，
不取消长任务，也不把这两个分数当作完整六任务对照结论。

为最终收敛准备，使用获胜配置的显式参数在
`optimization_retained_index_selection_cleanup_seed42_20260914` 独立重建：6389786_0-5。
batch 6389835 等待构建完成，逐组比较图节点/边序/权重、候选 ID/顺序、全部来源读出；
复制完整识别缓存及原检索记录供普通接口验证，不复制或重采样 QA。
随后六任务 verify 为 6389842 / 6389843 / 6389845 / 6389846 / 6389847 / 6389848，
全部依赖重建核验成功，使用 CacheMissGuard 禁止新生成。当前默认仍不变。
重建六子作业均完成（25-58 秒）；6389835 完成，0:0（13 秒），15 组 / 14362 来源 /
167074 候选及图结构、权重、读出全部匹配。普通接口全量 verify 尚待完成。
四个短任务 verify（6389842 / 6389843 / 6389845 / 6389846）已完成，400 题逐题匹配。
等待长任务期间，将 research_novelty.md 中 refinement-only 的旧理论说明归档，
由 331 行整理为 186 行，保留当前新构图、索引、证据限制和英文实验段落。
原说明保存在 construction_experiments_20260914/research_notes_before_current_method_cleanup.tar，
tar --compare 通过；此次仅整理文档，不把文档减行计为算法压缩。
LoCoMo / 2Wiki verify（6389847 / 6389848）随后完成，分别为 0:0（3:01 / 4:06），
1986/1000 题均匹配。至此获胜配置独立重建后的全 3386 题普通接口验证通过，
不读取保存的 query resets、不新增 recognition、不重采样 QA。

完整旧图同索引对照随后收齐：6389287_0/1 为 0:0（17:43 / 26:25），6389288_0/1
为 0:0（14:32 / 12:06）；报告 6389297 为 0:0（20 秒）。6772 条新 QA 的完整 case、
实际检索输入、原主指标与 usage 通过核验，旧四格及新图同索引结果整目录引用。

| reader / 同一保留事实索引下的构图 | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9B 旧 refinement 图 | 94 | 59 | 64 | 15 | 57.4063 | 56.6379 | 4/6 |
| 9B 新无自环图 | 91 | 63 | 68 | 11 | 57.5679 | 57.7539 | 6/6 |
| 4B 旧 refinement 图 | 91 | 59 | 61 | 12 | 50.4071 | 55.9104 | 6/6 |
| 4B 新无自环图 | 90 | 62 | 61 | 12 | 51.0135 | 56.3973 | 6/6 |

旧图同索引每 reader QA 输入 11131086 token，9B/4B 输出 36486/31289，QA 秒数
2251.4/2624.5；新图每 reader 输入多 152572 token，不是等 token 预算。
旧图复用同候选识别缓存，无新增 recognition，但方法本身继承原识别成本。
新图在 MH、LoCoMo、2Wiki 更高，在 SH 和 9B FCMH 回退；旧图不能替代双 6/6 目标。
同索引对照提供新构图仍有用的条件证据，不将全部采样 QA 分差解释为纯图因果效应。

完整对照与重建验证后，默认选择新图 + 保留事实索引。build_graph 默认启用原有
retained-fact-index，显式 --no-retained-fact-index 可复现原候选索引四格；run_graph 默认
与之匹配。只改入口默认值，不改变已经验证的投影、支持选择、向量、读出或 QA。
清理前源码存于 construction_experiments_20260914/implementation_before_retained_index_default.tar，
tar --compare 通过。既有测试文件增加默认构建/查询入口一致性及原索引开关检查，
未增加新的算法参数、独立脚本或调度 JSON。
默认入口回归作业 6390450 完成，0:0（6 秒），29 项测试全部通过；git diff --check 通过。

基于下节真实失败追踪，本轮候选为 `statement_projection_loop_free_retained_index_rrf_window`，
输出 `optimization_retained_fact_index_seed42_20260914`。这是新的离线索引消融，重新开放
此前四格冻结的事实候选集合；不把它混写成“识别输入完全不变”的构图实验。
按现有 canonical/latest retained_statements 选择原事实行，保留其文字、ID、向量和相对行序。
不修改实体到来源、三元组到来源映射，不重新 embedding、不改 recognition 提示/模型/解码、
PPR 算子、图权重、RRF、五中心、读出或 QA。候选减少会影响原 get_fact_scores 的 min-max
归一和实际查询种子；算子不变不等于输入分数不变，不宣称纯粹只删去若干过滤器输出。
图与 reader 必须逐组等于主配置，实验构建仍不读问题或答案。

旧 fact_index 完整负结果保留，其同时改变来源映射，不能直接替代本轮单变量对照。
这里只复用已有支持选择删除候选，不新造评分公式、阈值、冲突分类或任务路由。
现有 build_graph / run_graph / retriever/hipporag 增加临时入口，测试仍在既有文件内。
本地可用 vLLM 为 agent-memory-envs/vllm_cu129，CUDA 12.9，原 Qwen3-30B-A3B-Instruct-2507
权重已缓存；不使用已缺入口的 llm_tool_ckpt/venvs/vllm，不安装或重下载环境。
候选变化可能触发新 recognition 调用，复用 GraphUsageRecorder 记录真实 provider token/耗时；
原 OpenIE、schema 和 embedding 的既有成本不记成零，缓存命中不计成新调用。

单测作业 6387166，六任务构建 6387180；构建成功后独立检查
15 组事实行选择和主图/读出一致性，再启动同源服务，按每任务至少一个样本验证缓存/新调用
及原图控制路径。全流程通过才进行六任务 3386 题的完整检索与双 reader 6772 次 QA。
不根据 pilot 答案或单例表现选择任务、规则或输出；pilot 不进入正式分数。

6387166 的 27 项测试通过，0:0（4 秒）。6387180_0-5 全部构建成功（26-54 秒）。
独立全量索引核验为 batch 6387240，检查原事实行序/向量、支持选择、主图权重及冻结读出；
GPU smoke 6387255 依赖它成功，使用现存 vllm_cu129/CUDA 12.9 和原 BF16 30B-A3B 模型。
每任务按原顺序第一题检查默认控制读出、新索引行/向量及来源映射，记录新 provider 调用，
不读取答案选择候选。新代码同时把 settings 的 additional_generator_calls 更新为实测新调用数，
避免构建阶段的零调用字段被误当作整轮识别成本。pilot 的调用单独记录，不混入正式 QA。

6387240 全量核验成功，0:0（1:37）：15 组、14362 来源，原事实候选总数 190345，
保留 167074；主图节点/边序/权重、读出及选中向量逐项一致。原完整向量存储仍加载并保留，
不由候选数量下降声称总索引磁盘或峰值内存压缩。
6387255 六任务 pilot 成功，0:0（6:04）：SH/MH 首题 recognition 缓存命中，其余四任务
各一次真实新调用，合计输入 11708、输出 135 token；这是 pilot 额外成本。
控制读出均匹配，实体/事实来源映射不变。2Wiki pilot 的新 recognition 返回空列表，
按原流程合法退回 dense；不是 provider 错误，也不据此提前宣称索引改进。

完整检索 6387416 使用同一已验证的服务配置串行完成六任务，单次服务启动后分别运行
现有 run_graph；不将整个实验改为 agentic retrieval，不重做抽取或向量。
检索/调用成本核验 6387429 等待其完成，核对完整 case/来源与实际调用数、token。
以下 QA 均同时依赖完整检索和该核验成功，先不基于短任务分数调整后续任务：

| 任务 | 9B/4B QA 数组 |
|---|---|
| SH-Doc_QA | 6387417_0-1 |
| MH-Doc_QA | 6387418_0-1 |
| FactConsolidation-SH | 6387419_0-1 |
| FactConsolidation-MH | 6387420_0-1 |
| LoCoMo | 6387421_0-1 |
| 2WikiMultiHopQA | 6387422_0-1 |

最终报告 6387430 等待全部成功后，逐题核对 QA 的实际检索输入并使用原 report_results。
原四格控制以整目录链接引用，旧分数不重采样、不逐题选择。本轮尚未产生完整 QA 结论。

完整检索 6387416 在 LoCoMo 第五组模型加载时 CUDA OOM，1:0（10:53），不是观察超时。
已完成四个短任务各 100 题和 LoCoMo 四个完整组共 757 题，2Wiki 尚未开始；依赖 QA/报告
均被调度器取消，尚无 QA 发生。vLLM 占约 68.17 GiB，检索进程已占约 10.91 GiB。
补上组间关闭后清除对象引用、gc.collect 和 CUDA 空闲缓存释放，沿用 pilot/旧 fact_index
已经使用的释放方式；不改变模型、候选、图、检索规则或生成参数。
从已归档 run_fact_index 复用完整组前缀续跑检查，只接受原问题顺序和完整组边界，
拒绝从半组或不匹配问题继续；已有完整任务直接保留，未完成文件只追加剩余组。
LoCoMo 原 757 题另保留在同实验根的 locomo_retrieval_before_resume.jsonl，后续按原字节核对。

新增续跑边界测试作业 6387690；恢复检索 6387717 依赖测试成功。新成本/输入核验 6387718
依赖恢复完成，汇总恢复前后全部调用并核对原 757 题字节未变。
新 QA 数组按 SH/MH/FCSH/FCMH/LoCoMo/2Wiki 为
6387719 / 6387720 / 6387721 / 6387722 / 6387723 / 6387724，均为 0=9B、1=4B，
同时等待恢复检索与核验；新最终报告为 6387725。旧取消依赖不计为已运行 QA，
也没有删除已完成检索或重新抽样已有问题。全部脚本仍通过 stdin 提交，不新增仓库执行文件。

6387690 的 28 项测试通过，0:0（3 秒）。恢复检索 6387717 完成，0:0（13:56），
组间释放后通过剩余全部来源组。6387718 完成，0:0（3 秒）：六任务原 case、五个中心、
冻结读出与真实 provider 调用数均核验通过；LoCoMo 原 757 题前缀逐字节一致。
原来源 2Wiki Recall@5 / Precision@5 为 0.87425 / 0.4252。
正式检索新增 recognition 共 2168 次，输入 6323456、输出 102681 token；另有上述 pilot
4 次输入 11708、输出 135 token，不混入 QA 成本。按任务的新调用数为 2/8/85/86/1911/76，
主要成本在 LoCoMo；不能把“沿用既有向量”写成没有新增 LLM 成本。
四个短任务 QA 已完整完成：SH/MH/FCSH/FCMH 的 9B 为 91/63/68/11，4B 为 90/62/61/12，
均为各自短任务 4/4 严格胜出。该时点 LoCoMo 和 2Wiki 仍在执行，没有提前采用或报双 6/6。

完整结果：6387723_0/1 分别完成（17:39 / 26:32），6387724_0/1 完成（14:37 / 12:20）。
最终报告 6387725 为 0:0（18 秒），两 reader 六任务 6772 条预测与实际检索输入一致，
原指标、完整 case 集、QA usage 和九项 baseline 阈值核验通过。同一配置达到双 6/6：

| reader / 配置 | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9B 原候选索引 | 91 | 63 | 65 | 11 | 57.0535 | 57.9152 | 5/6 |
| 9B 保留事实索引 | 91 | 63 | 68 | 11 | 57.5679 | 57.7539 | 6/6 |
| 4B 原候选索引 | 90 | 62 | 61 | 9 | 50.8067 | 56.6039 | 6/6 |
| 4B 保留事实索引 | 90 | 62 | 61 | 12 | 51.0135 | 56.3973 | 6/6 |

QA 输入为每 reader 11283658 token，比原候选索引增加 7926；9B/4B 输出为 36061/31682，
QA 调用秒数为 2256.3/2652.2。这些不是构建、识别或总 wall time，不能省略上文新增识别成本。
9B FCSH、4B FCMH 各提高 3 点，LoCoMo 小幅提高，2Wiki 分别回退约 0.1613/0.2067 点。
因此是目标覆盖率提升，不是逐任务支配；仍是 test-as-dev、单轮原采样协议结果。
暂不以结果达标结束研究：正在运行旧图同索引对照，检验能否删除新投影而保留目标效果。

### 2026-09-14：先定位错误来源，不继续盲目删除组件

连续两轮删减未改善覆盖率后，暂停新增构图候选，先检查同 H100 旧 refinement 与新主图
的实际 reader 输入和既有失败预测。6386265 为 0:0（6 秒），复用 analyze_failures 的
原评分与失败列表，不新增指标、数据划分或自动失败归因规则。
6386834 为 0:0（46 秒），进一步调用现有 _read_retrieval_records / _answer_prompt /
_official_generation 重建全部 prompt，包含 LoCoMo 的既有选项随机顺序；逐项核对保存的
QA usage generation_settings。以下相同指完整 messages、答案选项映射和解码设置一致：

| 任务 | 完全相同输入题数 | 这些题中答案分数变化数 9B / 4B |
|---|---:|---:|
| SH-Doc_QA | 81/100 | 0 / 3 |
| MH-Doc_QA | 72/100 | 5 / 5 |
| FactConsolidation-SH | 73/100 | 9 / 6 |
| FactConsolidation-MH | 83/100 | 2 / 2 |
| LoCoMo | 1120/1986 | 242 / 246 |
| 2WikiMultiHopQA | 604/1000 | 0 / 0 |

MAB 使用原温度 0.7，LoCoMo 为 0.4，2Wiki 为 0。当前 seed 在一次完整评测起点设置，
不是每题独立复位；前面生成长度不同可能改变后续采样状态。这是一个可能来源，不将所有
变化唯一归因于它或 GPU。相同输入仍存在分数变化，足以否定“全部 QA 分差都是图效应”。
保留单轮 5/6、6/6 的已发生结果，但不解释为稳定胜率、显著提高或纯图因果效应。
不为消除波动而改成 greedy、逐题 seed、重新抽取或择优重跑；这些会改变冻结协议。

6386327 为 0:0（1 秒），对 FCSH 的三个既有失败例手工追踪原文与结构化 OpenIE。
这里只用文字查找辅助阅读，不生成 gold passage 标签，不将三个例子当作总体错误比例，
题号/实体不进入算法或数据 loader：

- no14：Bart Simpson 的新事实 Ray Bradbury 位于来源 520、编号 17816，已在 OpenIE
  和保留事实附录中，且是第一条检索证据；旧值 Matt Groening 所在来源 270 也进入 reader。
  两 reader 都答旧值。它不是目标事实未被抽取或完全未被检索到的例子。
- no17：Queen Victoria I speaks language Dutch 位于来源 379、编号 12988，OpenIE 已抽出，
  但该来源不在当前五个中心里；不能先判为抽取失败。继续核对保留支持及原 recognition。
- no91：目标长人名已在首条证据和保留附录中，两 reader 都输出到 Alb 后停止。
  6386834 用各自现有 tokenizer 核对参考文本需要 12 token，实际 usage 为 10 token，
  原协议 max_tokens=10。该案例的截断不是缺图边；不修改原输出上限或 SubEM 来消除错误。

目前只能分别定位到候选访问和 reader/输出约束，不能把 FCSH 的全部缺口归结为新构图不够强。
单例追踪作业 6386852 只调用现有缓存 recognition 和普通检索，不新增生成或 QA；
继续核对 no17 的具体入口，再决定是否有在冻结范围内值得实施的结构变化。

6386852 读取清理快照 settings 时因缺少 config 退出（1:0，15 秒），在加载检索模型前失败，
没有新 recognition 或 QA。改读获选方案原始完整运行 settings，未改变其配置，
重试 6387084 为 0:0（30 秒）；两个例子的普通接口读出均与保存结果一致，缓存守卫无 miss。
该诊断只重放已有识别和检索，不重新生成答案；临时脚本经 sbatch stdin 执行，未进入仓库。

- no17 的 Dutch 三元组已被 canonical/latest 保留。原事实索引候选第一、二名分别是
  English、Dutch，但 recognition 只输出 English；图前五和 BM25 前五都不包含 Dutch 来源。
  因而已定位到候选识别与后续证据访问，不能归因于 OpenIE 漏抽或来源选择删错。
- no14 的候选第一名是 Ray Bradbury，但 recognition 输出旧 Bart/Matt 及重复的 Homer/Matt。
  即便如此，新图与 BM25 都把新事实来源排第一；旧值来源是 BM25 第二名，仍进入合并上下文。
  它同时说明 recognition 不等于更新判断，以及正确首条证据不保证 reader 遵循更新指令。

下一步优先检查离线事实候选索引与图支持的一致性，而不是追加边权规则。
已读归档 fact_index.py / run_fact_index.py 及完整旧结果：index_latest 为 9B/4B 3/6、4/6，
index_schema 为 3/6、3/6，不能隐去这些负结果。旧实现同时修改事实候选、entity-to-source
和 triple-to-source 映射，使用旧图和原文读出，不是当前新构图下单独事实候选索引的对照。
若实施新轮，只复用现有 retained_statements 选择既有事实行/向量，不发明筛选规则；
保持 graph、实体来源映射、PPR、RRF、reader 不变，显式记录这是重新开放离线候选索引的
单独消融，不再声称“识别候选完全冻结”。候选变化可能触发新 recognition，须先检查同源
Qwen3-30B-A3B 服务、缓存命中及成本，再做每任务 smoke 和完整六任务双 reader QA。
目前尚未修改事实候选索引，也未提交该轮生成或 QA，不能预称解决了这些案例。

### 2026-09-14：无自环投影下删除额外来源直连

raw relation 消融清理后 6385219 的 26 项测试通过，0:0（5 秒）。
下一轮只删除获选无自环 refined 图中额外保留的原 passage-entity 直连贡献；
来源仍是事实成员，投影产生的来源连接完整保留，不是删除 provenance 或原文。
候选 `statement_projection_loop_free_no_direct_refined_rrf_window`，入口 `--no-direct-source`，
输出根目录 `optimization_loop_free_no_direct_seed42_20260914`。
保留 canonical/latest、去 synonym、事实成员、Kumar 无自环算子、原节点/向量、
识别/PPR、BM25/RRF、五中心与冻结来源读出，loader、指标、QA prompt 和解码不变。
只扩展现有入口与一项既有测试，不创建新算法、公式、阈值或仓库执行脚本。

过去 no-direct 负结果使用含自环投影，不能代替本轮。这里检验额外直连对当前主图的
条件作用，不把消融差值称为来源信息的整体贡献。六任务 3386 题、双 reader 6772 次新 QA，
固定原四格整项参照，不按短任务分数筛选是否完成长任务，不用跨候选拼接成绩。
先单测，再完整构图与原文核验：新图必须等于当前主图减去已有选中来源直连矩阵，
同时保留原节点、事实来源和冻结 reader 内容。核验成功才进行 GPU 检索与 QA。

6385284 的 26 项测试通过，0:0（4 秒）；构图数组 6385301_0-5 全部成功（20-49 秒）。
全图核验 6385304 为 0:0（1:36）：15 张新图符合纯事实无自环投影，恰好等于主图减去
已有选中来源直连；30 张默认对照图重算一致，14362 条来源读出冻结不变。
该核验成功后提交下列完整检索与双 reader QA，QA 依赖对应任务完整检索成功。

| 任务 | 检索 | 9B/4B QA 数组 |
|---|---|---|
| SH-Doc_QA | 6385389 | 6385397_0-1 |
| MH-Doc_QA | 6385390 | 6385398_0-1 |
| FactConsolidation-SH | 6385391 | 6385407_0-1 |
| FactConsolidation-MH | 6385392 | 6385408_0-1 |
| LoCoMo | 6385393 | 6385409_0-1 |
| 2WikiMultiHopQA | 6385394 | 6385410_0-1 |

batch 检索核验 6385437 依赖全部六任务，核对完整原 case、五个来源中心与冻结 reader 文本，
使用原有 2Wiki 来源 Recall@5 / Precision@5。最终报告 6385438 依赖全部 QA 与该核验，
再逐题检查真实 QA 输入和完整预测，用现有 report_results 生成五配置全量对照。
四个既有原图/新图参照仅整目录链接，既有 QA 不重新采样，也不作为本轮新增调用。

完整检索及核验均成功，6385437 为 0:0（3 秒），沿用原来源指标的 2Wiki
Recall@5 / Precision@5 为 0.8755 / 0.4272，高于主图 0.8730 / 0.4248。
LoCoMo 6385409_0/1 为 0:0（28:11 / 26:28），2Wiki 6385410_0/1 为 0:0（9:08 / 7:32）。
6385438 为 0:0（17 秒），完整核对六任务双 reader 共 6772 条新 QA 及其实际来源输入。

| 无额外直连的 refined 无自环投影 | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9B | 92 | 59 | 63 | 10 | 57.8210 | 57.8522 | 4/6 |
| 4B | 89 | 59 | 60 | 9 | 51.1541 | 56.7158 | 5/6 |

相对主图，MH 从 63/62 降至 59/59，FCSH 从 65/61 降至 63/60；
LoCoMo 两 reader 都提高，但未维持共同阶段目标，不采用。每 reader 输入 11183065 token，
比主图少 92667，仍是不同检索文本而非等 token 比较；调用秒数不当作建图收益。
这支持在当前配置下保留额外来源连接，不证明该连接在所有任务上都有益或理论必需。
尤其不能把这组负结果描述成“没有 provenance”，因为来源仍完整参与事实成员及投影。

完整代码归档 `implementation_before_loop_free_no_direct_removal.tar`，tar --compare 通过，
位于 `/oscar/home/zliu328/agent-memory-archives/construction_experiments_20260914/`。
随后删除此轮构图/运行开关及专用测试断言，主图、原四格与全部实验数据保持不变。
这次减少的是实验分支，主机制未被证明可以再删一项，不把入口行数删减冒充算法简化。
清理核验 6386152 为 0:0（2:12）：26 项测试通过，30 张默认对照图按当前算子重算一致，
15 张完整消融图仍符合纯事实投影，14362 条来源读出不变；两轮临时开关均已移除。
该核验没有新增 QA，也不冒充新一轮普通查询全量复验。

### 2026-09-14：更新主线与相关工作边界

根据最新讨论，把 update/consolidation 作为问题起点，核心 refinement 定位为来源支持整合；
超图投影是表示机制，系统的基础记录/派生索引分离用于解释设计，不作为独立新理论。
重新核对 MemoryAgentBench 论文、本地 loader 和 reader prompt：两项 FactConsolidation
来自 Conflict_Resolution，保留官方新编号优先指令与原 SubEM；其余四项不改成更新任务。
当前按来源段落取最后支持，同段多个值仍保留，不具备事实编号级冲突消歧或在线增量更新保证。
补读 TEPA、StateFuse 与 Reliable Post-Retrieval Assembly 的原方法节，差异记录在
related_work_analysis.md；这些工作尚未成为本地实测 baseline，不把其论文成绩并入九 baseline 池。
不增加数据切分、冲突检测打分或新的后处理规则，raw-relation 消融保持提交时协议。

### 2026-09-14：仅删除图侧 canonical relation 键

主配置清理后的全部普通检索复验已通过，再进行比“全部支持”更小的删除消融。
保持 latest 来源选择，但分槽使用规范化的原 OpenIE 关系标签，不应用 LLM canonical 映射。
复用 retained_statements / latest_relation_weights 已有 schema=None 行为，不新增规范化规则、
投影公式、阈值或 source-order 策略；只是删除一层关系别名合并。

候选 `statement_projection_loop_free_raw_relations_rrf_window`，入口 `build_graph --raw-relations`，
输出 `optimization_statement_projection_raw_relations_seed42_20260914`。
主方案及全部支持失败消融作为整项参照。保持去 synonym、来源直连政策、事实单位权重、
无自环投影、原节点/向量、识别/PPR、BM25/RRF 和来源读出规则。
reader 的事实附录仍使用冻结 canonical/latest，因此本轮不代表完整 canonical-free 流水线。
不重做 OpenIE/schema，不改变 loader、六任务数据或 QA 协议；4B/9B 各 3386 题，共 6772 次新 QA。
完整原图四格是上一轮已完成的固定参照，不为本消融重新采样。

先在已有关系别名单测中核对不合并时两个 raw slot 都保留，再进行六任务构图和全图核验。
核验默认图未变、raw slot 仍只取最后来源、投影使用既有算子、全部 reader 文本仍等于冻结版本。
之后运行完整检索与 QA；无论短任务是否达标，长任务和完整报告都保留。

6383190 的 26 项测试通过，0:0（4 秒）；构图 6383196_0-5 均 0:0（28-39 秒）。
全图核验 6383223 检查 15 张新图、30 张默认对照图及全部来源读出；GPU 作业依赖该核验成功。
该核验已完成，0:0（1:59）：15 张 raw 图符合原关系分槽的最后来源支持与既有投影，
30 张默认图节点/边序/权重完全不变，14362 条来源读出与冻结版本一致。

| 任务 | 检索 | 9B/4B QA 数组 |
|---|---|---|
| SH-Doc_QA | 6383232 | 6383233_0-1 |
| MH-Doc_QA | 6383234 | 6383235_0-1 |
| FactConsolidation-SH | 6383236 | 6383237_0-1 |
| FactConsolidation-MH | 6383238 | 6383239_0-1 |
| LoCoMo | 6383240 | 6383241_0-1 |
| 2WikiMultiHopQA | 6383242 | 6383243_0-1 |

6383269 对完整检索核对原 case、五个中心来源和冻结文本，沿用 2Wiki 原来源证据指标。
6383286 等待全部 QA 与检索核验后，检查实际 reader 输入并生成六配置报告。
最初报告提交因引用已完成的旧报告 6382421 出现 dependency problem，没有产生新作业；
sacct 再确认旧报告为 COMPLETED 0:0 后，仅移除该已满足依赖再提交，未重启任何实验。
旧四格与全部支持失败消融以整目录链接引用，不覆盖或重新采样。

该轮已全部完成：LoCoMo 6383241_0/1 为 0:0（27:49 / 15:38），
2Wiki 6383243_0/1 为 0:0（9:16 / 7:33）；最终报告 6383286 为 0:0（20 秒）。
报告逐题核对六任务两 reader 的实际输入与完整检索一致，6772 条新 QA 均完整落盘。

| 原关系键 + latest | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9B | 91 | 63 | 65 | 9 | 57.5809 | 57.9112 | 5/6 |
| 4B | 90 | 62 | 62 | 6 | 50.6439 | 56.5773 | 6/6 |

相对当前 canonical 无自环主图，9B FCMH -2、LoCoMo +0.5274，4B FCSH +1、
FCMH -3、LoCoMo -0.1628；2Wiki 两 reader 分别 -0.0040 / -0.0267 点。
胜出任务数不变，不能声称图侧 canonical 是达到 5/6、6/6 的必要条件。
这不是未达标的失败候选，但没有解决 9B FCSH 缺口，且当前更新任务 FCMH 回退，
故暂不替换默认。单次解码结果不证明小分差显著，不能将 FCMH 的全部差值当成纯图效应。
每 reader 输入 token 11272253，比主图少 3479；reader 仍依赖原 canonical/schema，
没有删除生成阶段或实测其成本收益。检索核验 6383269 为 0:0，原来源 2Wiki
Recall@5 / Precision@5 为 0.8730 / 0.4248，与主图聚合指标相同，不代表逐题排名相同。

归档 `implementation_before_raw_relation_removal.tar` 后 tar --compare 通过，
路径为 `/oscar/home/zliu328/agent-memory-archives/construction_experiments_20260914/`。
删除已用完的 raw-relations 构图/运行入口和该轮专用断言，保留全部图、缓存、预测和报告。
原四格与主算子保持不变；归档是依赖既有环境的源码覆盖层，不是独立运行镜像。

### 2026-09-14：采用无自环联合方案并精简构图入口

同 H100 四格对照与全部支持删除消融完整收齐后，默认采用
`statement_projection_loop_free_refined_rrf_window`（9B 5/6、4B 6/6）。
旧 refinement 同 H100 为 4/6、6/6；新方案满足双 reader 阶段目标，但仍非逐任务支配，
9B FCSH 65 未超过最佳 baseline 66，不能宣布双 reader 全面胜出。

保留原图/新图各有无 refinement 的四个执行入口。退役显式事实节点的独立在线候选、
含自环投影选项及失败的全部支持开关；内部事实 incidence 仍用于构建无自环投影。
主方案的来源支持筛选并未删除；减少的是未采用的实验分支，不虚报主机制数量下降。
新构图成为 build_graph Python/Oscar 入口默认，run_graph 默认选无自环 refined；
原图两格可用 `--construction projected` 显式构建。

清理前整个 optimization 目录保存到
`/oscar/home/zliu328/agent-memory-archives/construction_experiments_20260914/implementation_before_projection_selection.tar`，
tar --compare 通过。它是依赖既有基础实现/环境的源码覆盖层；全部图、缓存、QA 与报告均未删除。
build_graph.py、run_graph.py、statement_incidence.py、test_graph_construction.py、本地 build_graph.sbatch
五文件从归档时 678 行减到 644 行，净删 34 行；不将文档变化计入算法压缩。

6382311 的 26 项既有构图/报告测试通过，0:0（4 秒）。
默认新构图重建 6382335_0-5 全部 0:0（26-61 秒），旧构图重建 6382336_0-5 全部 0:0（39-48 秒）。
6382366 全图核验为 0:0（38 秒）：四格共 60 张图的原节点、边顺序、权重、来源顺序与完整读出
逐项等于获选结果及同 H100 原图对照，清理没有改变实际图算子。

新构图输出 `optimization_graph_selection_cleanup_seed42_20260914`，
旧构图输出 `optimization_original_graph_selection_cleanup_seed42_20260914`。
前者整目录链接后者两格，组成完整四格报告；读取已完成的检索和 QA，不重新采样答案。
普通接口全量 verify 作业按 SH/MH/FCSH/FCMH/LoCoMo/2Wiki 为
6382412 / 6382415 / 6382417 / 6382418 / 6382419 / 6382420，均包含四格。
共核对 13544 次检索，禁止新增 recognition 调用；6382421 依赖六项全部成功再生成四格报告。
此轮新增 QA 为零，不能将引用的历史 QA 成本当作本轮重新发生的调用。

普通检索复验全部完成：SH/MH/FCSH/FCMH 为 0:0（1:09 / 2:05 / 1:14 / 1:12），
LoCoMo 6382419 为 0:0（9:58），2Wiki 6382420 为 0:0（20:35）。
四格六任务共 13544 次普通查询的中心来源、读出和分数均匹配保存结果。
最终报告 6382421 为 0:0（11 秒），四格完整预测/原指标仍为 9B 3/4/4/5、4B 2/6/4/6 个胜出任务；
主配置确认为无自环 refined 的 5/6、6/6。至此该轮默认切换与清理已完成验证。

### 2026-09-14：无自环构图的全部来源支持消融

无自环联合方案全量完成后，补一个图侧删除实验，而非新增投影、阈值或权重。
既有 unrefined 对照同时恢复全部支持和 synonym 边，不能把该对照的退步全部归因于缺少
canonical/latest。新增 `statement_projection_loop_free_all_support_rrf_window`：
取消图侧的 canonical/latest 支持筛选，保留全部抽取事实及原 passage-entity 连接，
仍删除 synonym 贡献；事实单位权重和已有无自环投影不变。
直接参照为同轮已完成的无自环 refined/unrefined 两格，不按模型或任务选择不同方案。
这是现有组件的删除消融，没有新算法公式或专门的数据筛选。

入口 `build_graph --construction statement_projection_loop_free --all-support`，
只构建一个新候选，不重复构建/评估已完成的 unrefined 对照。
输出目录 `optimization_statement_projection_all_support_seed42_20260914`。
冻结 OpenIE、向量、识别/PPR、BM25/RRF、读出/窗口/事实附录及原 QA 协议；
六任务完整 3386 题、4B/9B 共 6772 次新 QA，不取消阴性任务。
事实附录仍依赖既有 canonical/latest，故成功也只能支持删除图侧依赖，不能宣称整条流水线
不需要 schema 或 latest。实际 reader token 数仍可能随检索中心变化。

复用既有测试中的共享事实场景，核对全部来源成员和直连不丢失、仅去 synonym 贡献，
并在完整构图上独立核对其相对 unrefined 图的矩阵差只包含原 synonym 权重。
全部来源读出逐条核对冻结版本；测试、六任务构图和核验成功后才启动 GPU 检索与 QA。
旧图同 H100 复现继续独立运行，不覆盖历史结果，不作为启动新候选的成绩筛选门槛。

测试 6379666 为 0:0（3 秒），24 项 graph_construction 测试通过；
本次未修改 report_results，也未把未运行的两项报告测试计入该数字。
构图 6379670_0-5 均 0:0（35-49 秒）；全图核验 6379696 为 0:0（23 秒），
15 张新图均等于已有 unrefined 无自环投影减去原 synonym 贡献，
14362 条来源读出及原节点身份保持不变。

| 任务 | 检索 | 9B/4B QA 数组 |
|---|---|---|
| SH-Doc_QA | 6379723 | 6379729_0-1 |
| MH-Doc_QA | 6379724 | 6379730_0-1 |
| FactConsolidation-SH | 6379725 | 6379731_0-1 |
| FactConsolidation-MH | 6379726 | 6379732_0-1 |
| LoCoMo | 6379727 | 6379733_0-1 |
| 2WikiMultiHopQA | 6379728 | 6379734_0-1 |

检索依赖完整图核验，QA 依赖对应完整检索。6379761 核对完整 case、中心来源映射与读出，
沿用原来源 2Wiki Recall@5/Precision@5。6379777 等待该核验、六任务双 reader QA 和
同 H100 旧图报告 6378444，核对实际 QA 输入后生成五配置报告。
旧两格引用本次 H100 固定复现，另外两格引用已有无自环完整结果；不重评这些对照。

完整 QA 和报告现已完成：LoCoMo 6379733_0/1 为 0:0（17:33 / 26:00），
2Wiki 6379734_0/1 为 0:0（14:40 / 12:25），报告 6379777 为 0:0（18 秒）。
每 reader 3386 题，共 6772 次新 QA，完整预测逐条对应检索输入；九项 baseline 原指标核验通过。

| reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9B | 91 | 63 | 66 | 8 | 57.53 | 57.49 | 5/6 |
| 4B | 88 | 61 | 62 | 4 | 51.47 | 56.53 | 4/6 |

不采用：相对无自环 refined，4B SH -2、MH -1、FCMH -5 点，不能满足双 reader 至少 5/6。
但 FCSH 与 LoCoMo 提高，因此不能说支持筛选对所有任务都有益。
9B FCSH 66 仅与最佳 baseline 持平，不计胜出。
6379761 检索审计为 0:0（3 秒），全部 case 和读出映射通过；
2Wiki 原来源 Recall@5 / Precision@5 为 0.87325 / 0.4250，refined 为 0.8730 / 0.4248。
这再次说明证据指标相近不等于两个 reader 的六任务 QA 等价。
每 reader 输入 11221170 tokens，略少于 refined 的 11275732，不据此推断图结构压缩。
结果、图和缓存保留；归档代码后删除该开关，不按 reader 分别选择有无支持筛选。

### 2026-09-14：旧图两格的同 H100 对照复现

旧对照日志确认存在硬件差异：原图 SH 6360175_0 使用 L40S，2Wiki 6360277_0 使用 RTX A6000；
本轮新构图使用 H100。新增的是旧两格的固定协议复现，不是新候选或选择较高分的重复运行。
输出目录 `optimization_construction_h100_controls_seed42_20260914`，
重新评估 original_graph_rrf_window / canonical_latest_rrf_window 的完整六任务、4B/9B。
使用冻结检索记录与原 reader 文本，原 seed42、prompt、解码、指标和环境不变；
不重新检索、抽取、构图或改向量。13544 次新 QA 单独计入研究成本。
既有历史 QA 不覆盖；无论高低，两套旧图成绩均保留，同设备构图比较使用本次预先指定复现。
九项 baseline 池仍是原始既定结果，不因本次复现而宣称整个 baseline 池同硬件。

| 任务 | 9B/4B QA 数组 |
|---|---|
| SH-Doc_QA pilot | 6378374_0-1 |
| MH-Doc_QA | 6378411_0-1 |
| FactConsolidation-SH | 6378412_0-1 |
| FactConsolidation-MH | 6378413_0-1 |
| LoCoMo | 6378414_0-1 |
| 2WikiMultiHopQA | 6378415_0-1 |

后五项依赖 SH 双 reader pilot 完成。batch 6378444 等待全部复现与无自环轮完整报告，
逐条核对实际 reader 文本、case 顺序及预测覆盖，再用现有 report_results 汇总十配置。
其他八种新构图的 H100 结果整目录引用，不按题拼接，也不为换一张对照表重评新候选。

4B 六任务复现已完成，独立核验 6380907 为 0:0（12 秒）：三种配置的完整预测覆盖、
逐题原指标重算、实际 reader 输入及九项完整 baseline 均通过检查。
以下为 4B 的同 H100 完整结果；9B LoCoMo 尚在运行，双 reader 总报告仍等待 6378444。

| 配置 | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原图，H100 复现 | 87 | 54 | 62 | 3 | 51.5705 | 48.4706 | 2/6 |
| 旧 refinement，H100 复现 | 91 | 59 | 60 | 9 | 50.7813 | 56.1370 | 6/6 |
| 无自环新构图 + refinement | 90 | 62 | 61 | 9 | 50.8067 | 56.6039 | 6/6 |

因此，历史参照中的“4B 从 5/6 到 6/6”不能作为新构图独有收益；统一硬件后旧 refinement
也为 6/6。新构图对旧 refinement 的差为 SH -1、MH +3、FCSH +1、FCMH 0、
LoCoMo +0.0254、2Wiki +0.4669 点，仍不是逐任务支配，也不把微小单次分差解释成稳定收益。
原历史结果保留，不覆盖、不择高。相同 seed 不保证跨环境重跑逐题输出相同；
本次对照变化本身也不能单独证明所有差异均由 GPU 型号造成。

随后 9B LoCoMo 6378414_0 完成，0:0（55:14）；总核验/报告 6378444 完成，0:0（30 秒）。
旧两格六任务、双 reader 的实际读出输入逐条保持冻结版本，完整预测和原指标重算通过。
同 H100 的四格对照现已齐全；以下为后续构图比较的主参照，历史表不覆盖：

| 构图 | refinement | 9B：SH / MH / FCSH / FCMH / LoCoMo / 2Wiki | 9B 胜出 | 4B 胜出 |
|---|---|---|---:|---:|
| 原图 | 无 | 93 / 61 / 62 / 5 / 57.2890 / 50.0232 | 3/6 | 2/6 |
| 原图 | 有 | 94 / 58 / 63 / 9 / 57.1394 / 56.6652 | 4/6 | 6/6 |
| 无自环事实投影 | 无 | 92 / 60 / 64 / 4 / 57.2492 / 52.4801 | 4/6 | 4/6 |
| 无自环事实投影 | 有 | 91 / 63 / 65 / 11 / 57.0535 / 57.9152 | 5/6 | 6/6 |

新联合方案相对同 H100 的旧 refinement：9B MH +5、FCSH +2、FCMH +2、2Wiki +1.2500 点，
SH -3、LoCoMo -0.0859 点。因此之前相对历史分数的“9B FCSH 回退 2 点”不适用于同硬件参照，
但该项仍未超过最佳 baseline 66。旧方案这次 9B 仅 4/6，新联合方案恢复双 reader 至少 5/6。
不能把四格胜出数量视为统计交互检验，也不把 baseline 池说成全部同硬件或等上下文。
完整输出目录中的 comparison.json / results.md 保留十配置，未对任何任务择高拼接。

### 2026-09-14：已有无自环、度数保持投影对照

删直连轮的 9B 四个短任务已完整完成，其中 MH 57、FCSH 61 未超过既定最佳 baseline 的
59、66，因此即使两个长任务全部胜出，也最多 4/6，不能作为双 5/6 的替代配置。
仍将该轮所有长任务与完整报告跑完，不取消、不删阴性结果，也不改变其中任何运行设置。
下一对照只改变事实层的投影算子，保留原直连贡献，避免将两种删除混在一起。

候选 statement_projection_loop_free_rrf_window / statement_projection_loop_free_refined_rrf_window，
新目录 `optimization_statement_projection_loop_free_seed42_20260914`。
采用 [Kumar 等 2020，式 3 及其随机游走解释](https://link.springer.com/article/10.1007/s41109-020-00300-3)：
选择一个相邻事实后，均匀选择其中不同于当前位置的成员；分母为成员数减一，事实投影对角为零。
这是已有的 H (D_e - I)^{-1} H^T 的无自环度数保持投影，不是只清零旧矩阵对角线，
也不是我们的新公式。仅采用其图约简，不采用聚类、modularity 或迭代调权；不是完整 IRMM 复现。
“不在当前事实内原地返回”不等于 non-backtracking walk，跨步返回原节点仍允许。

每个活跃事实至少有实体与来源两个不同成员；空事实保持无贡献，单成员输入明确报错。
单位事实权重、原 passage/synonym 残余贡献、来源筛选、原节点/向量、PPR/识别、
RRF、读出、六任务与 4B/9B 的原 QA 协议均不变，不搜索新的权重或 damping。
测试在已有文件增加混合成员数、度数/残余边保持和 singleton 前置条件检查；batch 6375942。
通过后完整构图并核验旧默认投影未变、新算子与已有公式逐项一致，再运行检索与 QA。

6375942 的 27 项测试通过，0:0（6 秒）；构图 6376005_0-5 全部 0:0（41-70 秒）。
全量核验 6376048 为 0:0（1:02）：30 张旧默认投影重建后边序/权重完全一致，
30 张无自环图符合引用算子，事实度数贡献与残余连接保持，14362 条来源读出不变。
该作业也重新核对删直连轮 9B 四项完整预测及对应九项 baseline，确认其最多 4/6 的资格上限。
“度数保持”指相对同一 incidence 的加权度数，不是恢复原 HippoRAG 的度数或保证排名不变。

| 任务 | 普通检索 | 9B/4B QA 数组 |
|---|---|---|
| SH-Doc_QA | 6376072 | 6376096_0-1 |
| MH-Doc_QA | 6376073 | 6376097_0-1 |
| FactConsolidation-SH | 6376074 | 6376099_0-1 |
| FactConsolidation-MH | 6376075 | 6376100_0-1 |
| LoCoMo | 6376076 | 6376101_0-1 |
| 2WikiMultiHopQA | 6376078 | 6376102_0-1 |

检索依赖 6376048；每项 QA 依赖对应完整检索与构图核验。
batch 6376134 核对十配置的完整 case/读出和已有 2Wiki 证据指标。
6376135 等待本轮全量 QA、检索核验及删直连轮完整报告 6375025，才汇总十配置。
历史参照整目录链接，本轮最多 13544 次新 QA；不把未完成成绩计为零或只保留胜出任务。

本轮现已全量完成：LoCoMo 6376101_0/1 均 0:0（55:41 / 52:03），
2Wiki 6376102_0/1 均 0:0（18:23 / 14:34）；完整报告 6376135 为 0:0（27 秒）。
两种新配置各 reader 均为 3386 题，合计 13544 次新 QA。
以下按 SH / MH / FCSH / FCMH / LoCoMo / 2Wiki 排列，严格胜出对照九项完整 baseline 的逐任务最佳值：

| 无自环配置 | reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 新构图，不加 refinement | 9B | 92 | 60 | 64 | 4 | 57.25 | 52.48 | 4/6 |
| 新构图，不加 refinement | 4B | 86 | 56 | 62 | 2 | 51.02 | 51.87 | 4/6 |
| 新构图 + refinement | 9B | 91 | 63 | 65 | 11 | 57.05 | 57.92 | 5/6 |
| 新构图 + refinement | 4B | 90 | 62 | 61 | 9 | 50.81 | 56.60 | 6/6 |

联合方案相对冻结 refinement 历史参照：9B MH +5、FCMH +2、2Wiki +1.4483 点，
但 FCSH -2、LoCoMo -0.0212 点，SH 不变；4B SH +3、MH +3、FCSH +1、FCMH +1、
2Wiki +0.6182 点，但 LoCoMo -0.4474 点。不能称为逐任务支配。
相对含自环投影联合方案，9B FCMH +2、2Wiki 提高，而 4B FCMH、LoCoMo、2Wiki 回退；
去自环并非在所有任务均有益。4B 对最佳 baseline 的 LoCoMo 优势仅 0.4704 点，
9B FCSH 仍比最佳 baseline 低 1 点。单 seed 分差不作为显著性结论。

6376134 检索审计为 0:0（12 秒），十配置六任务均为相同完整 case 集及原来源映射读出。
2Wiki 无自环联合方案的原来源 Recall@5 / Precision@5 为 0.8730 / 0.4248，
冻结 refinement 为 0.8580 / 0.4154，含自环联合方案为 0.86675 / 0.4216。
这是五个中心来源的证据可达性，不是全图保真或整个扩展 reader 上下文的召回率。
联合方案每 reader QA 输入 11275732 tokens，冻结参照为 11107895，约增加 1.51%；
固定读出规则不等于实际 token 数相等，也不作压缩或等硬件加速主张。

当前判断：保留无自环联合方案作为候选，默认仍为 canonical_latest_rrf_window。
继续完成已预先提交的 H100 旧图复现后再判断同硬件构图收益，不按结果择高拼接。
test set 用作开发集的边界不变；4B 6/6 是本地既定 baseline 池内的单次完整胜出，
不是未见数据上的泛化结果，也不是全部文献方法的 SOTA 证明。

### 2026-09-14：删减原 passage-entity 直连贡献

前两轮均已完整完成后，启动单项结构消融，不添加第三套打分或阈值。
直接对照为 statement_projection_rrf_window / statement_projection_refined_rrf_window；
新候选为 statement_projection_no_direct_rrf_window / statement_projection_no_direct_refined_rrf_window，
输出目录 `optimization_statement_projection_no_direct_seed42_20260914`。

唯一变化：在事实 incidence 投影前，不再保留原 passage-entity 边的残余贡献。
每个事实的主体、客体和实际来源成员不变；unrefined 组的原 synonym 贡献仍保留，
refined 组仍按原 refinement 不保留 synonym。投影公式、对角项、事实单位权重不变。
这不是删除来源信息，也不是无权图拓扑的等价变换；直连权重改变会改变 PPR。
refined 组由此只保留既有标准事实 incidence 投影，不新增图论公式。

完整六任务、4B/9B、原 OpenIE/向量/schema/QA 协议不变；逐来源读出与冻结参照完全相同。
复用已有构图、检索、评测与报告入口，仅增加一个删除开关及相应两种配置名。
25 项现有测试由 batch 6374888 执行；完整构图需依赖测试成功，随后全量检查旧默认构图
未变、新图仅移除直连贡献，再进行普通检索和全任务 QA。不是依据单一任务选择配置。
结果收齐前不删除默认连接或替换主方案，当前修改是可独立评估的删减开关。

25 项测试通过，6374888 为 0:0（4 秒）。构图数组 6374917_0-5 全部 0:0（28-57 秒），
全量核验 6374956 为 0:0（2:07）：重建的 30 张默认 incidence 图与上一轮节点、事实身份、
边序和权重完全相同；30 张删减投影图等于原投影仅减去原直连贡献，保留自环计数约定。
全部 14362 条来源读出与冻结参照一致，未改默认方案或抽取缓存。

| 任务 | 普通检索 | 9B/4B QA 数组 |
|---|---|---|
| SH-Doc_QA | 6374964 | 6374991_0-1 |
| MH-Doc_QA | 6374965 | 6374992_0-1 |
| FactConsolidation-SH | 6374966 | 6374993_0-1 |
| FactConsolidation-MH | 6374968 | 6374994_0-1 |
| LoCoMo | 6374969 | 6374995_0-1 |
| 2WikiMultiHopQA | 6374970 | 6374996_0-1 |

所有检索依赖全量构图核验；每个 QA 数组依赖对应完整检索与构图核验，仍使用 H100。
batch 6375024 等待六任务检索，核对八配置的完整 case、五中心来源及冻结读出文本，
并使用已有指标报告 2Wiki Recall@5/Precision@5。6375025 等待完整 QA 和检索核验汇总八配置。
六份历史参照只作整目录链接；本轮最多 13544 次新 QA，不逐题复用或选择较高分回答。

6375024 完成，0:0（11 秒）：八配置全六任务 case、顺序、五中心原来源和对应冻结读出一致。
2Wiki 的无 refinement Recall@5 / Precision@5 为 0.78500 / 0.3734，
有 refinement 为 0.87075 / 0.4248；原投影分别为 0.78925 / 0.3756、0.86675 / 0.4216。
该删除改善 refined 证据召回，但已完成的 9B MH/FCSH 明显回退，不能把召回增益当成 QA 增益。

来源成员计数 batch 6375594 完成，0:0（8 秒）：refined 活跃事实 167074 个，
其中 167072 个只有一个来源，2 个有两个来源；成员数 3 的 166899 个、成员数 2 的 173 个、
成员数 4 的 2 个，另有 23271 个被保留为孤立节点的无活跃支持事实。
因此该联合方法主要是来源支持的实体/来源小型事实投影，不以大量多来源超边作为机制解释。
成员数 4 的例外只在 2Wiki；未据此截断成员、改 schema、改图或修改任何数据。

删直连完整 QA 已完成：9B LoCoMo 6374995_0 为 0:0（56:09），全八配置报告 6375025
为 0:0（23 秒），两个新候选共 13544 条预测及 usage 核验通过。

| 删除直连 | Reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 无 graph refinement | 9B | 90 | 61 | 57 | 4 | 57.55 | 51.11 | 3/6 |
| 无 graph refinement | 4B | 87 | 61 | 54 | 3 | 51.08 | 49.46 | 2/6 |
| 有 graph refinement | 9B | 92 | 57 | 61 | 8 | 57.70 | 56.73 | 4/6 |
| 有 graph refinement | 4B | 89 | 57 | 60 | 9 | 50.91 | 56.12 | 5/6 |

该删除未达到双 5/6，不采用；refined 相比保留直连的投影，9B MH/FCSH 分别 -6/-4 点，
4B MH -5 点。refined 每 reader 输入 token 为 10967090，虽比保留直连少 210190，
仍不足以支持用此取舍替代原投影。4B SH 与最佳 baseline 同为 89，不计严格胜出。
实现将归档后退役删除开关及专用测试，图、完整预测、usage、原缓存和报告保留。

源码归档完成，tar --compare 对照通过：
`/oscar/home/zliu328/agent-memory-archives/construction_experiments_20260914/implementation_before_no_direct_removal.tar`。
该归档为 optimization 覆盖层，依赖此前 refinement_only 基础实现归档；旁边 README 说明恢复范围。
工作树已删除 direct_sources 参数/分支、两项 no_direct CLI 名称和专用测试；无新执行文件。
首次补丁因测试行匹配失败而整体未应用，核对实际文件后修正，未发生部分清理或实验重跑。
26 项余下测试 6378261 为 0:0（4 秒）；batch 6378276 为 0:0（2:51），重新检查全部默认
incidence、投影及无自环算子与已保存图一致，全部来源读出不变。
退役失败候选不计作主方法删掉一个 heuristic，原直连贡献保留。

### 2026-09-14：已有随机游走投影的结构对照

上一轮新事实节点图仍在完整 QA，四个短任务已完成且效果混合，未更换主方案。
2Wiki 的原来源 Recall@5：原图 0.7465、旧 refinement 0.8580、事实节点 0.7660、
事实节点 + refinement 0.8545；由 6371419 用已有指标和显式原来源映射核对。
固定 PPR damping 时插入事实节点增加传播步数，因此不能把星形表示与投影表示的差值
仅解释成信息是否保留。新增一个已有方法对照，不进行参数扫描。

候选 statement_projection_rrf_window / statement_projection_refined_rrf_window，
目录 `optimization_statement_projection_seed42_20260914`。采用
[Zhou 等 NIPS 2006 第 4 节式 3](https://papers.nips.cc/paper_files/paper/2006/file/dff8e9c2ac33381546d96deea9922999-Paper.pdf)
的 incidence 随机游走：先选择相邻事实，再在该事实成员中均匀转移。
用既有标准 H D_e^{-1} H^T 投影物化事实层，保留当前设定的 passage/synonym 残余连接；
将结果交给同一 PPR。事实超边权重为 1，不新增置信度、关系打分、阈值或温度。
保留投影对角项，使用 igraph Weighted_Adjacency 的 loops="twice" 对应其无向边计数约定。
这不是与显式事实节点 PPR 等价的压缩：两者在相同 damping 下的有效传播步不同。
也不是我们的新图论公式；现阶段只是检验表示与传播步粒度的已有算子对照。

CPU 预检 6371949 已读取全部 30 张事实图：最大事实成员数 192，逐任务投影项规模可处理，
不因度数截断成员。复用同一事实身份/来源选择、OpenIE、embedding、读出和六任务协议。
新增投影函数仍放在 statement_incidence.py，不新建执行文件；现有测试增加两项矩阵/自环语义检查，
测试作业为 batch 6371973，构图与后续 GPU 任务依赖其成功。

6371973 的整数权重测试触发 NumPy 除法输出类型错误，尚未进入构图；依赖数组
6371997 被调度器取消，0 秒、无产物。将度数数组显式转为 float64 后，24 项测试
在 6372014 全部通过（4 秒）。替换构图数组 6372016_0-5 全部 0:0（33-63 秒）。
6372075 在 batch 全量核对 30 张图的矩阵、无向自环计数约定、原节点和读出，0:0（28 秒）。

| 任务 | 普通检索 | 9B/4B QA 数组 |
|---|---|---|
| SH-Doc_QA | 6372041 | 6372095_0-1 |
| MH-Doc_QA | 6372042 | 6372096_0-1 |
| FactConsolidation-SH | 6372043 | 6372097_0-1 |
| FactConsolidation-MH | 6372045 | 6372098_0-1 |
| LoCoMo | 6372046 | 6372099_0-1 |
| 2WikiMultiHopQA | 6372047 | 6372100_0-1 |

四个短任务两候选普通检索已完成，各 100 题，0:0；两长任务及 QA 继续执行。
batch 6372187 核对六种图的完整 case/顺序和 2Wiki 原来源 Recall@5；已完成的短任务由
sacct 验证成功，仅将仍运行的两长任务设为依赖（把已退出调度器的旧 ID 加入依赖的首次提交被拒绝，
没有重跑检索）。6372188 等待两轮完整 QA、检索核验和上一轮报告后汇总六种配置。
归档与阴性结果不修改；当前两轮均未达到完整矩阵，不根据短任务成绩宣称改进或提前筛掉候选。

6372187 已完成，0:0（6 秒）：六种配置均覆盖相同的 3386 个 case，题目与顺序不变，
每题五个中心均有原来源映射。两轮全部普通检索均已完成。2Wiki 的 1000 题证据结果如下，
调用原有指标函数，不使用扩展读出文本做 gold passage 精确匹配：

| 构图 | 无 graph refinement Recall@5 | 有 graph refinement Recall@5 | 无 graph refinement Precision@5 | 有 graph refinement Precision@5 |
|---|---:|---:|---:|---:|
| 原实体对图 | 0.74650 | 0.85800 | 0.3528 | 0.4154 |
| 显式事实节点 | 0.76600 | 0.85450 | 0.3636 | 0.4156 |
| 标准事实投影 | 0.78925 | 0.86675 | 0.3756 | 0.4216 |

本表只支持该任务上投影的证据可访问性更好，不证明全局图质量、QA 改进或独立测试集泛化。
无 refinement 的显式事实节点也改善召回，但加 refinement 后比原实体对图略低；
因此不能用“保留更多结构必然更好”解释结果。QA 继续按既定全六任务、双 reader 跑完，
主配置不变，不按任务选择不同图。batch 6372366 另测实际节点、边、图文件和权重文件大小；
该测量不计原始 embedding/cache，也不把节点数减少等同于端到端索引压缩。

6372366 完成，0:0（13 秒）。下表累计六任务的 15 组图，字节数为实际文件大小：

| 表示 | graph refinement | 节点数 | 边数 | graph.pickle 字节 | edge_weights.npy 字节 |
|---|---|---:|---:|---:|---:|
| 显式事实节点 | 无 | 350330 | 3320861 | 197961406 | 26568808 |
| 显式事实节点 | 有 | 350330 | 708031 | 81978739 | 5666168 |
| 标准事实投影 | 无 | 159985 | 3188398 | 140373296 | 25509104 |
| 标准事实投影 | 有 | 159985 | 515702 | 52203169 | 4127536 |

未 refinement 的 LoCoMo 边数从显式事实节点的 153076 增至投影的 231456，
投影不保证每个任务的边数更少。显式图保留事实节点和原边属性，投影只保留原节点属性及权重，
故 pickle 差值也包含属性差异，不是仅靠拓扑压缩的收益。运行仍先加载原始图与向量索引，
这些新增产物不是独立替代索引；此表不是总磁盘、峰值内存或 PPR 加速测量。

该轮完整 QA 与六配置总报告均已完成：9B LoCoMo 6372099_0 为 0:0（55:35），
6372188 为 0:0（17 秒）。报告重新核对各配置六任务、双 reader 的主指标、预测覆盖及 usage。
新投影两候选共 13544 次新 QA；前一轮和原图参照整目录复用，不逐题拼接。

| 投影配置 | Reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 无 graph refinement | 9B | 93 | 60 | 64 | 4 | 58.43 | 51.80 | 3/6 |
| 无 graph refinement | 4B | 86 | 52 | 64 | 2 | 51.88 | 50.65 | 3/6 |
| 有 graph refinement | 9B | 91 | 63 | 65 | 9 | 57.08 | 57.20 | 5/6 |
| 有 graph refinement | 4B | 88 | 62 | 61 | 10 | 51.02 | 56.72 | 5/6 |

投影 + refinement 相对旧 refinement 改善 MH（+5/+3 点）和 2Wiki，但 9B FCSH -2 点，
4B LoCoMo 下降；没有达到全面胜出。refined 每 reader 输入 token 11177280，比旧方案
11107895 多 69385，也高于显式节点的 10934922，不能用“更小上下文”解释此轮效果。
两种新表示均维持双 5/6，但没有统一支配旧方案。下一步仅删减重复直连贡献，保留这些完整对照。


### 2026-09-14：固定抽取，开始构图与 refinement 的 2x2 对照

本轮仅两个新候选：statement_incidence_rrf_window 和 statement_incidence_refined_rrf_window。
输出根目录：`optimization_statement_incidence_seed42_20260914`，不覆盖已冻结参照。
沿用已有 OpenIE、实体/段落/事实 embedding、query recognition、原实体/段落种子、PPR、
RRF、top5、来源读出、QA prompt/解码、loader/划分和评分；不新增生成、embedding、阈值或调参扫描。
两组读出均逐来源比对已冻结 compiled_sources，完全一致后才允许写入新实验。

构图依据是已有的事实 incidence 表示，参见
[HyperGraphRAG 4.1 式 5](https://arxiv.org/html/2503.21322#S4.SS1)。本轮不是完整复现该方法，
不使用其 n-ary 抽取、置信度、向量双路检索或生成策略，也不把已有表示称为新理论。
每个不同的原始规范化 (subject, relation, object) 一个事实节点；完全相同三元组共享节点，
关联其主客体及实际支持来源，所有成员连接为二值 incidence。不同谓词不合并。
原 passage-entity 连接保留，实体对 fact 贡献改由事实节点表达；原同义连接在无 refinement
组保留，在 refinement 组按当前 refinement 的既有行为移除。后者是既有模块的一部分，
不能将 refinement 的差值单独归因为 latest。支持选择仍调用原 retained_statements。
新节点 reset 为零，不新增 query-to-fact seeding，不重新解释谓词语义或声称方向推理。

与已退役 source_fact_incidence 不同：旧试验按每次事实出现建节点，移除了直接 passage-entity
和同义连接，且没有当前 RRF/window/附录。本轮使用共享事实身份并固定当前外围协议，
因此旧负结果保留，但不能直接填本轮下排两格。新表示改变路径长度和度数，不声称无损 PPR。

检查顺序：现有测试扩展两项结构检查（batch 6371267）→ 六任务构图与读出一致性
→ 各任务两候选普通检索成功 → 4B/9B 完整 QA → 原指标完整对照。所有阶段未完成前不报收益。
核心新增文件仅 graph_construction/statement_incidence.py，复用已有 build_graph/run_graph/report_results。

22 项测试通过。构图数组 6371286_0-5 全部 0:0（每任务 32-48 秒）；
batch 全量核验 6371357 为 0:0（18 秒），确认 15 组 / 14362 来源读出与冻结版本完全相同，
30 张新图的原节点顺序、独立事实身份、边权文件和图结构一致。
首次 GPU 提交同时指定两个 partition 被集群拒绝，无作业落地；改为 gpu-he。
L40S 排队预计次日，六个尚未启动的检索作业只改 GPU 资源为 H100，未取消或重跑任何题。

| 任务 | 普通检索 | 9B/4B QA 数组 |
|---|---|---|
| SH-Doc_QA | 6371303 | 6371360_0-1 |
| MH-Doc_QA | 6371304 | 6371361_0-1 |
| FactConsolidation-SH | 6371305 | 6371362_0-1 |
| FactConsolidation-MH | 6371306 | 6371363_0-1 |
| LoCoMo | 6371314 | 6371364_0-1 |
| 2WikiMultiHopQA | 6371315 | 6371365_0-1 |

QA 依赖对应任务完整检索和全量构图核验成功；六数组全部成功后 batch 6371379
使用现有 report_results 汇总四格完整对照。两份原图/旧 refinement 参照仅整目录引用，
不按题拼接。SH/FCSH/FCMH 的两候选普通检索已经各 100 题通过；其余与 QA 待完成。
MH 两候选普通检索也完成（6371304，0:0，1:09），4B/9B QA 已实际启动。
另提交 batch 6371419，依赖六任务检索完成，核对四格完整题目顺序和五个原始来源映射，
用既有 gold_passage_recall_at_k / precision 函数对 2Wiki 的 original_source_text 计算证据指标。
旧 QA runner 附带的 passage 指标对扩展读出文本做精确匹配，不作为本轮证据召回；
答案主指标、QA 调用和原评分协议不改。总报告 6371379 也等待这一核验成功。
首个端到端任务已完成：SH 9B 的两候选各 100 条预测落盘，6371360_0 为 0:0（2:53）。
该结果只证明本轮新构图到 QA 的完整执行路径已跑通，仍等待六任务、双 reader 的统一报告。

4B 的两候选已完成六任务，batch 6372865 为 0:0（6 秒）：复用 audited_score，
对照原始 case 集逐条核对预测数量、主指标均值与 summary，并重新核对九项 baseline。
以下为百分制；LoCoMo 使用 F1，2Wiki 使用 answer F1，四项 MAB 使用 substring EM。

| 4B 配置 | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 refinement | 87 | 59 | 60 | 8 | 51.2541 | 55.9857 | 5/6 |
| 显式事实节点 | 87 | 60 | 62 | 5 | 51.6924 | 49.3027 | 3/6 |
| 显式事实节点 + refinement | 88 | 57 | 61 | 8 | 51.1234 | 56.4728 | 5/6 |

显式节点加 refinement 相对旧方案有升有降，没有扩大 4B 胜出范围。
随后 9B LoCoMo 6371364_0 完成，0:0（55:13）；完整四格报告 6371379 为 0:0（14 秒），
两候选各六任务、双 reader 共 13544 条新预测均完成，原图两组参照整目录复用。
完整结果位于该输出根目录的 results.md / comparison.json，主指标、case 覆盖和 usage 核对通过。

| 9B 配置 | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 refinement | 91 | 58 | 67 | 9 | 57.0747 | 56.4669 | 5/6 |
| 显式事实节点 | 93 | 59 | 64 | 4 | 57.5441 | 51.0319 | 2/6 |
| 显式事实节点 + refinement | 93 | 61 | 64 | 9 | 57.4277 | 57.0537 | 5/6 |

结论：显式事实节点加 refinement 维持双 5/6，但没有增加胜出任务数；9B 的 MH 改善同时
伴随 FCSH 回退 3 点，4B 的 MH 回退 2 点。它不能统一替代旧配置，也不证明显式节点必需。
两 reader 各自输入 token 从旧 refinement 的 11107895 降到 10934922，仍不是等 token 比较；
跨 H100/L40S 的 QA 秒数不用于加速主张。该轮作为结构消融保留，等待投影轮完整结果。

第二轮后续任务一度因 H100 后续资源安排等待；只将尚未启动的短任务 6372097/6372098
时限由 2 小时改为 15 分钟，2Wiki 6372100 改为 1 小时，LoCoMo 6372099 改为 1.5 小时。
依据是同环境、同规模第一轮的实测时长；不改变运行中的作业、GPU 型号或实验协议。
短任务已全部完成，第二轮长任务已全部启动，无因观察超时而取消或重跑。

### 2026-09-14：历史文本 ZIP 归档

24 个历史作业 JSON、7 个退役实验/进度说明 MD、2 个历史环境 TXT 共 33 个文件，
原始内容合计 146296 字节，归档为
`/oscar/home/zliu328/agent-memory-archives/refinement_only_20260914/legacy_notes.zip`。
ZIP 完整性检查和解压流逐文件 cmp 均通过，随后移出工作目录；不新增替代脚本或 JSON。
当前算法文档、消融定义、方法记录和相关工作仍保留，历史文档链接改指 Git 旧版本。
不删除当前运行环境、模型权重、结果或缓存；这次主要减少目录杂项，不是大容量存储清理。

### 2026-09-14：冻结为原始抽取 + refinement 消融参照

用户决定保留当前结果与 cache，供后续抽取/构图研究比较；尚未运行新的抽取方法。
主配置为 canonical_latest_rrf_window，对照为 original_graph_rrf_window；保留完整六任务、
9B/4B、检索上下文、QA、usage、OpenIE、底图/embedding、两阶段 schema 和 LLM 缓存。
独立归档目录：`/oscar/home/zliu328/agent-memory-archives/refinement_only_20260914/`。
目录内 README 记录范围、原路径和恢复边界；数据 tar 解引用历史 QA 软链接，源码另存 tar。
后续新方法必须使用新的输出目录，不覆盖当前结果或共享原始缓存；大型产物不上传 GitHub。
这组是未来比较的“原始抽取 + refinement”参照，不预称已完成 generator 优化消融。
归档作业 6369944 在 batch 完成（0:0，1:04），两个 tar 与原文件逐一对照通过；
数据归档约 2.3 GiB，含最终 24 份 QA 预测、45 份 SQLite、15 份底图和 15 份 OpenIE。

每一轮补充：模块变化、直接参照、完整范围、4B/9B 结果、实际成本、正负例证、状态与产物路径。
当前只做既有机制删减与消融，不新增 heuristic、公式、数据规则或任务路由。保留实际实现和负结果，
再提炼能解释结果的机制；multigraph、hypergraph、hyperbolic space 或系统/PL 理论都不能只作为名称装饰。
旧主配置消融、本轮删减、新读出原图对照与最终核验均已完成。
gist 等退役实验保留历史结果，不再扩展。
