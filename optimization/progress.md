# 优化进展

核对时间：2026-09-13 07:09 EDT。所有已提交评测完成，4B 有两个候选严格胜出 6/6；三模型总目标仍未达成。

## 固定口径

- 六任务、全部 3386 题；每个候选用 Qwen3.5-9B/4B/2B 完整评测。
- 用户将目标提高为全面胜出：同一算法与配置必须让三个 reader 分别严格胜出 6/6，
  5/6 仅作阶段里程碑，不拼接任务或模型的最佳值。
- test set 明确用作开发集；不称为独立测试泛化。不改原评分、QA prompt 或解码。
- 图权重与事实索引轮固定五段原文；另列的 compiled context 轮改变来源节点的读取表示，
  不是等 token 对照，其收益不能全部归因于图拓扑。
- 图权重轮复用已验证的原始 recognition/reset；事实索引轮重新运行相同 recognition/PPR。

## 已完整的候选

表中每格为严格超过完整 baseline 最佳成绩的任务数，分母均为 6。

| 候选 | 9B | 4B | 2B |
|---|---:|---:|---:|
| relation_only | 2 | 2 | 2 |
| balanced | 2 | 1 | 1 |
| contextual | 3 | 2 | 2 |
| contextual_balanced | 2 | 2 | 1 |
| latest_relation | 3 | 4 | 3 |
| latest_relation_balanced | 4 | 4 | 2 |
| 第一版 schema_latest | 3 | 4 | 2 |
| 联合归一 schema_latest | 3 | 3 | 2 |
| canonical_latest | 3 | 4 | 3 |
| canonical_latest_balanced | 4 | 4 | 3 |
| adaptive_syn020 | 4 | 5 | 3 |
| index_latest（原环境） | 3 | 4 | 2 |
| canonical_graph_with_source | 5 | 4 | 3 |
| adaptive_graph_with_source | 4 | 4 | 1 |
| canonical_graph_source_window | 5 | 4 | 3 |
| adaptive_graph_source_window | 5 | 4 | 1 |
| original_graph_rrf | 3 | 1 | 1 |
| canonical_graph_rrf | 4 | 3 | 1 |
| original_graph_rrf_window | 2 | 3 | 2 |
| canonical_graph_rrf_window | 5 | 6 | 3 |
| canonical_rrf_deduplicated | 5 | 5 | 3 |
| canonical_rrf_sentence_facts | 5 | 6 | 3 |

`adaptive_syn020` 的 4B 首次完成单模型 5/6 阶段里程碑；同一候选的 9B/2B 未到 5/6，
不能宣布三个 reader 的阶段目标或全面目标达成。

`canonical_rrf_sentence_facts` 的 4B 首次完整胜出 6/6，分数按上述六任务顺序为
92 / 63 / 64 / 7 / 50.73494384731588 / 56.47278538511298。
2026-09-13 06:31 再次使用 `experiments.runner._read_retrieval_records` 与 `_score`
对全部 3386 条预测重新计分，每题指标与已保存值完全相同；各任务未舍入均值都超过
现有九项完整 baseline 的最佳值。LoCoMo 优势约 0.40 分，是开发集上的单次实测结果。
同一候选 2B 完整结果仅 3/6，9B 完整结果为 5/6（FCSH 未胜出），不能宣布总目标完成。

完整分数与 baseline 明细在各 scratch 实验根目录的 `results.md`、`comparison.json`。
使用 `python -m optimization.report_results --output-root <root> --variants <names...>` 刷新。
该报告检查所有预测的数量、唯一 case key 与汇总值；不补零，不合成任务最佳配置。

## 各轮状态

- `canonical_latest_jobs.json`：六任务、三 reader 全部完成，仍未达到目标。
- `fact_index_source_runtime_jobs.json`：主事实索引对照。四个短任务原图检索均为 100/100；
  6324316 正常完成，LoCoMo 与 2Wiki 原图一致性分别为 1986/1986 和 1000/1000。
  全部检索与 QA 已完成；index_latest 三模型胜出数为 3/4/2，index_schema 为 3/3/2。
  当前 42 项本地测试通过。
- `weight_grid_jobs.json`：五个统一权重配置，完整六任务 CPU PPR 与三模型 QA 均已完成。
- `provenance_jobs.json`：来源边按保留断言占比加权，完整矩阵已完成。
- `adaptive_synonyms_jobs.json`：根据来源断言兼容性调节同义边，完整六任务均已提交。
  adaptive_syn020 的 SH/MH/FCSH/FCMH：9B 为 91/67/59/7，4B 为 90/57/61/6，
  2B 为 71/50/57/4。adaptive_syn020 的三模型六任务已全，胜出计数为 4/5/3。
  LoCoMo 为 51.34/47.75/44.53，均未超过目标；2Wiki 为 56.81/54.67/41.59。
  本轮三个变体完整矩阵均已完成；adaptive_provenance_syn020 三模型胜出数为 4/5/2。
- `compiled_context_jobs.json`：全部六任务的来源事实表示已冻结，原图和 canonical 图使用
  同一表示的 QA 对照已提交。来源顺序不等于事件时间；原有 timestamp metadata 保留。
  普通图加载检索接口的完整 FCSH 验证作业 6324843 已完成，两图均为 100/100。
  四个短任务 QA 全部完成：canonical_graph_compiled 的 9B 为 80/52/59/11、
  4B 为 78/52/66/13、2B 为 67/44/63/8，事实整合局部有收益，SH/MH 明显退步。
  全矩阵完成；canonical_graph_compiled 三模型胜出数为 2/3/2。
  LoCoMo 只有 42.09/40.35/37.73，纯三元组表示没有解决短板。
- `source_context_jobs.json`：保留原文并补充来源 metadata、图保留事实的两个对照。
  canonical 与 adaptive 图使用同一表示，全六任务 CPU 构建已正常完成，三 reader QA 已提交；
  普通检索接口的完整 FCSH 验证 6325253 已正常完成。
  canonical_graph_with_source 的四个短任务：9B 90/60/69/12，4B 88/59/63/13，
  2B 76/45/65/13。9B 四项均胜出，2B 的 SH/MH 仍不足；长任务 QA 已启动。
  FCSH-9B 原文版与原文加事实版的输入 token 分别为 335758 与 560654，分数为 63 与 69；
  增加约 67% 输入，不能称为压缩收益。逐任务 QA usage 已加入报告并检查完整性。
  两个候选三模型均已全量完成，最后的 2Wiki QA 数组 6325252 正常退出。
  canonical_graph_with_source 胜出数为 5/4/3，LoCoMo 为 52.64/48.80/43.78；
  adaptive_graph_with_source 胜出数为 4/4/1。
- `source_window_jobs.json`：同一非空时间 metadata 下的来源邻接窗口，前后三条，查询前冻结。
  不重新分句，不跨时间边界；无时间 metadata 时表示不变，仍全任务评测并标为重复对照。
  六任务构建已全部完成，QA 6325561/63/65/67/69/71；
  完整 LoCoMo 普通接口验证 6325574 已完成，两种图均为 1986/1986。
  canonical_graph_source_window 的 9B 已完整达到 5/6，LoCoMo 为 55.31，仍低于目标 55.37；
  4B/2B LoCoMo 为 49.69/43.36，尚未解决三个 reader 的共同短板。
  两候选全矩阵已完成，canonical 胜出数为 5/4/3，adaptive 为 5/4/1。
  adaptive 的 9B LoCoMo 达到 55.68，但 FCSH 为 58，仍低于目标 66；
  不能将其 LoCoMo 与 canonical 的 FCSH 拼成同一候选的 6/6。
- `hybrid_graph_jobs.json`：固定 RRF 词面与图排名融合，原图/新图和原文/窗口共四组交叉对照。
  BM25 在同一份来源 corpus 上构建，不导入其他方法的预测。常数 60，两个排名各前五，
  最终返回五来源；无新增查询 LLM，改变了检索设置，不能称为纯图构建贡献。
  六任务 CPU 准备已正常完成（6325836/45/47/49/51/54）；QA 为 6325841/46/48/50/52/55。
  完整 MH 普通接口验证 6325856 已完成，四组均为 100/100。
  canonical_graph_rrf_window 四个短任务：9B 91/58/67/9，4B 90/59/60/8，2B 81/49/64/8。
  2B 首次超过 SH 目标，但 MH 与 FCMH 仅并列，不能计胜出；该版本 2B 全量为 3/6。
  同一 RRF 与原文表示下，2Wiki 原图为 52.05/48.81/37.36，新图为 60.10/57.63/46.19；
  FCSH 原图为 52/52/53，新图为 44/50/52。不能只报告 2Wiki 的正向图效应。
  6325855_2 的 2Wiki QA 转到 gpu-debug 并完成；_1 已在 gpu-he 启动，未迁移正在运行的作业。
  全部 2Wiki QA 已完成；LoCoMo 6325852 三个 reader 也已正常退出，四组全矩阵完成。
  original_graph_rrf_window 全矩阵完成，胜出数 2/3/2；其 9B LoCoMo 为 57.40。
  canonical_graph_rrf_window 胜出数为 5/6/3，LoCoMo 为 56.97/51.46/43.34。
  其 4B 六项严格胜出，9B 的 MH 为 58，仍低于目标 59；2B 的 MH/FCMH 仅并列。
  同一窗口表示与 RRF 下，LoCoMo 原图/新图分别为 57.40/56.97、51.23/51.46、
  44.21/43.34，不能将窗口版高分全部归因于新图。
- `context_packing_jobs.json`：固定 canonical_graph_rrf，新增来源去重与去重加三元组逐行文本两组。
  六任务构建 6326035/37/39/41/43/45 均已正常完成；QA 6326036/38/40/42/44/46，
  完整 LoCoMo 普通接口验证 6326047 正常完成，两组均为 1986/1986，未使用保存的 query resets。
  2Wiki QA 数组 6326046 转到 gpu/norm-gpu。
  后续仅将尚未启动的 6326044_2、6326046_2 调至 gpu-debug/L40S，并已分别正常完成
  20:03 与 10:23；没有重启正在运行的 QA。全部 2Wiki 和 4B/2B LoCoMo 均已完成。
  真实输出已逐题核对：两组均覆盖 3386 题，保留全部唯一来源 ID，并且无重复来源 ID。
  LoCoMo 来源字符数从 18410697 降为去重版的 15881654（约减 13.7%），逐行事实版为
  15577114。此为字符数，不等于 token 降幅。只去重时，非 LoCoMo 的 1400 题文本完全不变。
  去重版三模型全矩阵完成，胜出数为 5/5/3，LoCoMo 为 55.84/50.20/42.87。
  逐行事实版全矩阵完成，胜出数 5/6/3，LoCoMo 为 56.31/50.73/43.64。
  最后作业 6326044_0 正常完成，用时 01:34:10；07:09 的 squeue 无用户作业。
  实际 LoCoMo QA 输入 token 从未压缩窗口版的 5672028 降为去重版 4949417、
  逐行事实版 4814003，分别约减少 12.7% 和 15.1%，不是仅用字符数推测。
  未压缩窗口版 4B LoCoMo 为 51.46，去重版为 50.20，逐行事实版为 50.73。
  因此本轮成本确有下降，但没有观察到 4B LoCoMo 分数提升；不同硬件执行也是比较限制。
  尚未达到同一候选三个模型全部胜出的目标。

## 执行问题

- 两个 SH CPU 作业因 Slurm 用户环境读取失败未启动。改用 `--export=NIL` 和位置参数，
  替换作业 6322942/6322944 已成功，不覆盖原图。
- `optimization_fact_index_seed42_20260913` 的首次索引试跑认证失败，已停止并标记 `INVALID.json`。
  上游会吞掉 recognition 调用错误；新增 provider failure guard，禁止把这种 dense fallback 当结果。
- `optimization_fact_index_authenticated_seed42_20260913` 的三个短任务保留为独立记录；
  新环境的分句及 MH 原图检索不一致，未将它混入主结果。
- 主索引轮固定原 HippoRAG 环境与 L40S，生成器 TP=2。6324227 已完成四个短任务和
  LoCoMo 首个完整组，随后在加载下一组时 OOM。修复模型引用清理、降低 vLLM 显存预留后，
  6324316 只续跑剩余完整组；已超过此前停止位置，原图一致性检查继续通过。
- 来源窗口与原文提示版本已逐题核对：四个短任务各 100/100、2Wiki 1000/1000 的
  问题、答案字段、top_k 和 reader 所见文本完全一致；LoCoMo 0/1986 一致，即全部发生变化。
  非 LoCoMo 的重复 QA 出现分数波动，不能将这些差值解释为窗口收益，也不能挑有利重跑。
  这与跨硬件采样执行可能产生差异一致，但仅凭该检查不能断言波动的唯一原因。
- 06:31 前 quota 检查：scratch 已用 489.53/512 GB，home 86.13/100 GB；没有删除已有产物。

## 小模型短板

沿用现有 LoCoMo 分类指标，不新增评分方法。2B 的 HippoRAG2 / canonical RRF 窗口版：
category 4 为 46.88 / 61.35，category 5 为 54.26 / 27.58。
逐行事实版 category 4 / 5 为 59.01 / 31.84，总 F1 为 43.64，仍低于目标 44.80。
这些数值来自完整 1986 题的既有分类汇总，显示扩充上下文的明显取舍；
不能仅凭它们断言是检索、上下文干扰还是 reader 的哪一个因素造成。
下一步需核对无信息问题的错误输出与来源事实，不按 category 路由算法或改 QA prompt。

06:35 逐题对照：LoCoMo category 5 共 446 题，HippoRAG2 与逐行事实版在 2B 上
共同正确 114、共同错误 176、从正确变错误 128、从错误变正确 28。
按原有输出顺序查看退步例子，可见模型把另一人的经历或相邻但不同的活动作为答案，
例如把 Caroline 的画展动机回答给询问 Melanie 画展动机的问题。
这里保存的是原评分器处理后的预测，不能据此区分模型原始输出了选项字母还是答案全文。

检查 `locomo-conv-26` 来源位置 185/187/188/189 的原始 OpenIE 与冻结表示：
原抽取包含 `Caroline is the creator of paintings`、`Caroline created painting`、
`Caroline is the artist of painting`，这些均未进入 canonical 最新来源保留集合；
位置 189 最后仅保留 `this represents unity and strength`。
证据文件为原 HippoRAG2 索引的 `openie_results_ner_Qwen_Qwen3-30B-A3B-Instruct-2507.json`
及逐行事实版 `LoCoMo/memory/locomo-conv-26/contents.json`。
这证明该裁边规则会删除此处历史归属支持，不能证明它单独造成全部 QA 退步，原文仍完整存在。
下一轮应优先核对历史事件归属的保留机制，不以题目、正确答案或 category 决定保留规则。

最新提供的 AGENTS.md 禁止 heuristic，与早期优化轮的授权口径冲突，已发出异步澄清。
回复前继续现有作业、核验和错误分析，不新增 heuristic 或修改原评分、数据入口。

## 下一步

当前候选全部收齐，不再称为等待长任务结果。两个 4B 全胜候选的 9B/2B 都未全面胜出：
融合窗口版 9B 差 MH；逐行事实版 9B 差 FCSH。2B 前者 MH/FCMH 并列、LoCoMo 不足，
后者 SH/FCMH 并列、LoCoMo 不足。不得按任务或 reader 拼接这两个候选。
来源内部的细粒度冲突消解仍未实现，也不应把历史事件都解释为需要覆盖的冲突。

已核对 [REMem 原文](https://arxiv.org/html/2602.13530) 第 3.1、4.3、4.4 节及仓库实现：
- 索引阶段保留带时间的 gist 和 fact，包括潜在矛盾的历史记录，而非最新来源覆盖。
- gist 关联同来源的主体/客体 phrase，事实连接 phrase；这是已有建图方法，不作为新颖性声明。
- REMem-S 是单步检索设置，REMem-I 才使用迭代工具检索；不采用 agentic 部分仍需单独验证。
- 论文的 F1、LLM judge、top-10 检索与当前分类 F1、固定 reader 设置有差异，不能直接比较分数。
- 本地 `EpisodicGistExtraction._extract_chunk` 按 dataset 选择抽取模板，不能未经说明当作
  当前六任务同一配置的直接替代；尚未接入或启动 REMem 构建，不宣称已复现或有性能收益。

在 heuristic 授权澄清前，不新增来源裁边或拼接规则；现有结果和上述实现核对均保留。
报告最终 achieved 为同一候选三个 reader 全部 6/6，另记 5/6 里程碑。
所有新代码仍在 `optimization/`，没有提交或推送，也没有恢复 HyperMem。
