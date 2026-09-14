# 方法尝试记录

更新：2026-09-13。记录 `optimization/` 图优化阶段实际做过的尝试，包括负结果、未完成及无效运行。
本文按方法整理；逐次作业历史见 [progress.md](progress.md)，文献对照见
[related_work_analysis.md](related_work_analysis.md)。不把事后机制解释当作已证明的理论。

## 口径与当前结论

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
不同时加 RRF、窗口、关系归一化或新摘要。详见 [实验说明](fact_incidence_experiment.md)。

有效处：来源事实出现及重复贡献可独立追踪；六任务 3386/3386 普通检索接口验证通过。
相对 `relation_only`，9B FCSH 从 41 到 50。
不足：同一参照下 SH 从 90 降至 84；2Wiki Recall@5 为 86.05，低于参照的 87.225。
谓词与角色只是 metadata，原无向 PPR 不读取其语义；共享实体仍可能把不同事件连接起来。
判断：增加结构表达能力不等于检索收益；该组不是已成功的 hypergraph 新算法。

## 11. Gist 表示与已有强组合

原格式和紧凑格式都没有完成全语料，输出与成本保留；json_object 诊断也未全通过。
当前抽取版本为 `contextual_gists_format_retry`，采用统一、有界的格式重试，并核对后复用原格式有效日志。
没有 gist QA 分数，详情见第 13 节末尾和 [生成协议](gist_generation_protocol.md)。
四个候选为 `gist_hipporag`、`gist_dense`、`raw_dense_control`、`canonical_rrf_gist_index`。
前三者区分现成向量匹配与原 HippoRAG；第四个只替换已有强组合 `canonical_rrf_sentence_facts`
的 passage embedding，保留其图权重、RRF、BM25、窗口和事实读出，不把 gist 再附加给 reader。

待检验的好处：同一来源的上下文化表示可能改善匹配；若对原图和强组合都有效，证据比只在弱参照上有效更强。
当前限制：gist 的内容由 LLM 生成，格式合法不保证忠实；在原分块之外的指代无法凭空解决。
更长的 gist 不自动构成压缩。索引表示有效也不等于新图结构有效。
暂不写“有效/无效”结论，待完整结果后补充。新增组合共享同一份抽取和 embedding，不新增离线生成调用。
78 项实现测试通过；真实 QA 完成情况以 [gist_index_jobs.json](gist_index_jobs.json) 和 scratch 报告为准。

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
协议和执行见 [rank_window_experiment.md](rank_window_experiment.md)、[rank_window_jobs.json](rank_window_jobs.json)。
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
完整记录见 [gist_index_jobs.json](gist_index_jobs.json)；提交不等于已完成，暂无新 QA 分数。
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

每一轮补充：模块变化、直接参照、完整范围、4B/9B 结果、实际成本、正负例证、状态与产物路径。
当前只做既有机制删减与消融，不新增 heuristic、公式、数据规则或任务路由。保留实际实现和负结果，
再提炼能解释结果的机制；multigraph、hypergraph、hyperbolic space 或系统/PL 理论都不能只作为名称装饰。
旧主配置消融、本轮删减、新读出原图对照与最终核验均已完成。
gist 等退役实验保留历史结果，不再扩展。
