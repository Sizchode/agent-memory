# 优化进展（历史阶段）

2026-09-14：下文保留早期探索日志，已退役模块和三 reader 计划不是当前运行状态。
当前精简实现见 [README.md](README.md)，消融、删除代码与最新作业见 [record.md](record.md)。

按方法整理的正负结果与未完成尝试见 [record.md](record.md)，后续每轮持续补充。
最新用户明确允许逐模块组合试验；不将模块组合自动当作理论贡献，不硬编码任务或答案。

17:19 EDT：已完成 gist passage 索引与现成 dense/PPR 接入，77 项测试通过。
原抽取作业两次失败分别为连接错误和共享环境入口缺失；保留 1072 条有效输出。
已恢复独立 CUDA 12.9 vLLM 环境并续跑 6342535；完整建索引、检索、QA 和汇总依赖链
记录于 [gist_index_jobs.json](gist_index_jobs.json)，尚无新 QA 分数。
17:22 EDT：续跑已实际生成，FCMH 补齐到 537/537，FCSH 完整缓存复用，LoCoMo 抽取继续。
17:28 EDT：增加单模块组合 `canonical_rrf_gist_index`，只替换旧强配置的 passage embedding，
共享本轮 gist 抽取与索引；78 项测试通过，完整六任务三 reader QA 已接入同一依赖链。
统一汇总替换为 6342734，原三个候选的 QA 未取消。原图和旧强组合都将进行完整普通接口控制验证。
此时抽取已有 3263 条有效响应、无新增错误；总日志含原来的 2 次连接错误，未知 usage 不计为零。
17:33 EDT：补齐既有 RRF 20 候选对照的完整六任务三 reader 范围，复用此前 FCSH 检索，
不新增扫描点。FCSH 两组普通接口验证均为 100/100，三个 reader QA 已启动；
其他任务已进入检索或校验。作业与解释边界见 [rank_window_experiment.md](rank_window_experiment.md)。

## 2026-09-13 16:49 EDT 相关工作驱动的表示对照

来源事实 incidence 图完成全部六任务三 reader，9B/4B/2B 严格胜出 1/6、2/6、1/6，
没有取代此前 5/6、6/6、3/6 的完整候选。详见 [完整结果](fact_incidence_experiment.md)。
状态/事件三组也全部完成并重汇总：只改读出 4/6、5/6、2/6；只改图 4/6、5/6、3/6；
两者都改 3/6、5/6、2/6。没有一个配置达成三 reader 共同目标。

新作业 6341669 正在对全部 14362 个原始来源抽取 REMem prompt 定义的 gist；
不看 QA、不改切分、不增加在线轮次，当前只有抽取，尚无新索引或 QA 结果。
范围、来源、阶段成本和下一步组件对照见 [contextual_gists_experiment.md](contextual_gists_experiment.md)。
最新约束下不继续新增经验规则或参数搜索；下文旧 heuristic 授权与实验仅为历史记录。

## 2026-09-13 来源事实关联图

相关工作核对后实现标准 incidence 结构对照，详见
[fact_incidence_experiment.md](fact_incidence_experiment.md)。
不新增打分公式或语义剪枝；保留全部事实出现和来源关联，固定原检索与 top5 原文。
全部六任务已提交构图、普通接口全量验证、三个 reader 的 QA 依赖链。
63 项测试通过；FCSH 100 题普通接口与 CPU 回放全部一致，71 题排名相对原图变化。
尚无完整 QA 成绩，不把排名变化当作提升。作业见 `fact_incidence_jobs.json`。
16:32 EDT：六任务全部 3386 题构图与回放完成，普通接口验证已完成五任务 2386 题；
2Wiki 验证仍在运行。四短任务三个 reader 已完成，结果与限制见实验笔记。
2Wiki 完整 Recall@5 为 86.05%，低于直接表示参照 `relation_only` 的 87.225%；
不据相对原 HippoRAG 的提高宣称 incidence 的独立优势。完整 QA 汇总由 6340424 等待依赖生成。
16:33 EDT：2Wiki 普通接口验证完成，六任务合计 3386/3386 一致；剩余 QA 继续。
下面各节保留此前实验的时间点和原始记录，不代表当前作业状态。

核对时间：2026-09-13 13:54 EDT。用户澄清 heuristic 授权，实验已恢复；三模型总目标仍未达成。

## 历史保留对照

用户明确允许能够解释、统一应用的 heuristic，禁止 hardcode；不再等待授权澄清。
本轮复用已有来源 schema：单值关系保留最新来源支持，多值关系保留历史支持。
不是新的分类器，可能继承旧 schema 的误分类，不把来源顺序视为事件时间。

三组新候选与已有 `canonical_rrf_sentence_facts` 构成交叉对照：

| 候选 | 图连接保留政策 | 事实读出政策 |
|---|---|---|
| 已有 canonical_rrf_sentence_facts | 所有关系取最新来源 | 所有关系取最新来源 |
| canonical_rrf_history_facts | 所有关系取最新来源 | 单值取最新，多值保留历史 |
| schema_rrf_sentence_facts | 单值取最新，多值保留历史 | 所有关系取最新来源 |
| schema_rrf_history_facts | 单值取最新，多值保留历史 | 单值取最新，多值保留历史 |

RRF、窗口、去重、逐行文本格式、QA prompt、解码和评分不变，不追加生成器调用。
只改图的候选直接引用旧事实表示，避免同时改变读出政策。全部候选评测六任务三模型。

- 实验：`optimization_history_graph_seed42_20260913`，作业记录 `history_graph_jobs.json`。
- 六任务 CPU 构建 6335130 至 6335135 全部正常完成。
- QA：6335136/6335139/6335140/6335141（四个短任务），6335144（LoCoMo）、6335145（2Wiki）。
- 普通加载接口验证：6335146（完整 LoCoMo）、6335147（完整 FCSH）。
  两个作业均已正常完成，三个候选分别为 LoCoMo 1986/1986、FCSH 100/100，
  未读取保存的 query resets，实际接口输出与回放完全一致。
- 45 项本地测试通过；3386 题逐题核对评测内容、固定图的来源排名及唯一来源集合，全部通过。
- 真实来源检查已恢复 `Caroline created painting` 等历史归属，但其他创作者关系仍受旧 schema
  单值分类影响，没有宣称完整解决归属问题。
- 当前 QA 正在运行，没有新完整矩阵；不重跑挑选旧控制的分数。
- 首批 9B 四个短任务结果（SH/MH/FCSH/FCMH）：只改读出为 90/62/56/8，
  只改图为 92/59/63/9，图与读出都改为 90/60/61/8。尚未解决 9B 的全面胜出。
  同一 canonical 图下，保留更多历史将 FCSH 从旧逐行事实版的 63 降至 56，
  提示单值/多值未必对应可更新状态/历史事件；不能把更多历史一概视为正确。

以下为此前已完成实验及其结论。

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

此前的 heuristic 授权冲突已由用户本轮明确澄清。原评分、数据入口及禁止 hardcode 的约束继续保留。

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

历史支持交叉对照已启动，优先用实测判断图与事实读出各自的影响；不提前宣称机制成立。
报告最终 achieved 为同一候选三个 reader 全部 6/6，另记 5/6 里程碑。
此前代码及总结已以 218a06a 提交并推送；本轮新改动仍在 `optimization/`，尚未提交或推送。
没有恢复 HyperMem。

## 2026-09-13 14:43：更新语义对照与贡献定位

用户要求边优化边思考 novelty，新增 `research_novelty.md`，将已有组件、具体差异、
未实现能力和可证伪的实验分开。近邻核对包括 HippoRAG 2、Zep、REMem 和 A-MEM；
不把已有的时间图、事件表示或单步读取重新称为首次提出。

新增 source-only 的 state / event / discourse 关系政策分类，复用旧 canonical label，
每个预测保存简短理由，不将模型理由当作人工真值。状态最新来源与事件历史保留
独立于 cardinality。原 QA、数据入口和 scorer 不变；49 项测试通过，shell 语法通过。

schema 作业 `6337210` 在 gpu 的两张 L40S 上启动，固定 Qwen3-30B-A3B bf16、TP=2。
六任务 replay、三候选全量 QA 与两个完整任务普通接口验证均已提交依赖链，
作业清单见 `event_graph_jobs.json`。该轮尚无 QA 成绩，不提前宣称收益。
完整 schema 生成有新增离线 LLM 成本，后续图组装零新增调用仅指组装阶段。
scratch 当前 490.59/512 GB，继续保留已有结果、环境和缓存。

首个 schema 作业 `6337210` 已确认 FAILED：模型反复返回单条 JSON，而非完整 relations
列表；校验拒绝后退出，所有下游依赖作业自动取消，没有部分图进入 QA。
新增 JSON Schema 输出约束，固定列表字段及批次条目数，仍逐项校验 id、政策和理由，
不修改语义分类规则。50 项测试通过；普通 schema 构建模式仍使用原先输出设置。
参考 [vLLM 结构化输出接口](https://docs.vllm.ai/en/latest/features/structured_outputs/)。
新作业 `6337337` 使用独立的 `optimization_update_policy_schema_structured_seed42_20260913`
目录，六任务下游与普通接口验证已重新提交，最新编号见 `event_graph_jobs.json`。
旧失败输出和 usage 全部保留，完整成本报告应计入这些调用，不只汇总成功输出。

历史 cardinality 对照新收齐 4B：`schema_rrf_history_facts` 为 90/57/62/6/51.78/55.93，
相对九个完整 baseline 配置严格胜出 6/6。其 2B 为 74/50/64/7/44.01/42.26，仅 3/6；
9B 的最后一项 LoCoMo 仍在执行。不能将另一 reader 的成功当成三 reader 共同胜出。

## 2026-09-13 14:50：只读原文的组件消融

新增 `canonical_rrf_source_only`，冻结 canonical RRF 图与排名，沿用来源去重窗口，
仅省略三元组附录。完整原文、位置与时间仍提供给 reader，未改变任何 QA prompt。
它检验事实读出的独立作用，不作为新的图算法。52 项测试通过、shell 语法检查通过。
六任务全量三 reader 与 LoCoMo/FCSH 普通接口验证已提交，见 `source_only_jobs.json`。
新输出根为 `optimization_source_only_seed42_20260913`，旧结果与缓存保持不变。

首个失败的政策分类作业保存了 140 次调用，输入 123127、输出 6491 token，
调用时间之和 211.94 秒；该并发调用总时长不是实际 wall time。
修复后的 `6337337` 已通过真实批次校验并继续生成，未因前次失败重启旧 QA 作业。

14:51 历史保留三候选全矩阵完成。`schema_rrf_history_facts` 的 9B LoCoMo 为 56.69，
最终严格胜出 5/6、6/6、3/6；另两组分别为 5/5/3 和 4/5/3。
这仍未超过此前同一候选的胜出任务数量，不能称目标已完成。

只读原文版六项 CPU 构建全部完成。逐题核对 3386 条记录的 case、排名、分数、
中心来源和窗口来源均与逐行事实参照一致；文本字符总数从 30770000 降到 21435649，
约少 30.3%，不是 token 节省的测量值。GPU QA 正在运行，普通接口验证仍排队。

新 schema 已完成前六组时，3734 个关系标签中有 1472 state、2211 event、51 discourse，
68 个 canonical 组含不同更新政策。查看 `locomo-conv-26` 发现旧 canonical 归一模型
直接将 `has children`、`are sitting on`、`is a type of` 等放入同一组，原独立 schema
中这些标签仍各自不同。证据是 `optimization_canonical_schema_seed42_20260913/LoCoMo/`
`locomo-conv-26/batches.jsonl` 的实际 groups 输出，不是数据 loader 或 id 校验错误。
这提示应单独消融关系归一化，而不是手工修补特定标签或把不同政策强制统一。

15:04 只读原文的 2B 全矩阵为 81/47/57/5/42.86/46.07，严格胜出仍为 3/6。
实际输入 token 为 7181935，低于逐行事实版 9584490，但不是质量全面改善。
9B/4B 的 FCSH 分别降至 51/47；两者 LoCoMo 尚在执行，不提前填完整结果。

政策分类共 42484 个来源关系标签。已对 FCSH、FCMH 和 LoCoMo 的 12 组完整 schema
逐项检查覆盖、id、政策、canonical 保持一致，随后解除对应 replay 作业
`6337357/6337360/6337377` 对全局分类作业的等待。其余三任务仍等待完整分类，
所有六任务均继续运行，未用部分组构建 LoCoMo。只读原文的两个普通接口验证作业
`6337599/6337600` 从待启动的 gpu 队列移到空闲 gpu-he，未重启任何运行作业。

15:16 只读原文版完整结果为：

| Reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9B | 91 | 64 | 51 | 10 | 55.14 | 58.06 | 4/6 |
| 4B | 90 | 62 | 47 | 8 | 49.62 | 57.98 | 4/6 |
| 2B | 81 | 47 | 57 | 5 | 42.86 | 46.07 | 3/6 |

三个 reader 各自完整 3386 次 QA，输入均为 7181935 token。
原文版的普通检索接口验证覆盖 FCSH 100/100、LoCoMo 1986/1986，均完成。
这组减少输入但降低整体表现，不替换当前最好候选，也不把单项提升拼接成新算法。

状态/事件图的 FCSH、FCMH、LoCoMo 构建全部完成并进入 QA；原图 CPU 回放检查通过。
SH schema 的 7524 个标签完整校验后释放 replay `6337363`，其余两项仍待分类结束。
状态/事件图普通接口验证 `6337384/6337385` 在尚未启动时移至 gpu-he，保持 L40S 环境。

## 2026-09-13 15:20：原关系标签的图消融

新增 `graph_construction/relation_identity.py`，从完整更新政策 schema 派生独立副本，
保持 state/event/discourse 分类、理由不变，只恢复原关系标签用于来源支持整理。
原 canonical label 保存在 previous_canonical，不改变旧图或旧 schema。
候选为 `identity_graph_rrf_sentence_facts`，冻结逐行事实读出、RRF、PPR 和 QA 协议，
与当前 canonical 状态/事件图进行图侧对照。它可能避免错误覆盖，也可能遗漏真实别名。
53 项测试通过，shell 语法检查与 diff 检查通过；没有新增分类 LLM 调用。

六任务全量三 reader 已提交，见 `identity_graph_jobs.json`。
FCSH、FCMH、SH、LoCoMo 已有完整输入，先启动；MH、2Wiki 等待其政策 schema 完成。
输出分为 `optimization_relation_identity_schema_seed42_20260913`、
`optimization_relation_identity_graph_seed42_20260913`、
`optimization_identity_readout_seed42_20260913`，保持各阶段来源与成本边界。
15:20 scratch 为 491.10/512 GB，未删除任何数据或缓存。

15:34 更新政策 schema 作业 `6337337` COMPLETED，退出码 0，用时 49:57。
15 组、42484 个标签完整覆盖并逐项验证；event 28052、state 14029、discourse 403。
本次完成作业记录 892 次调用，输入 2589911、输出 1552122 token；包含过程中保存的调用，
不是仅统计最终采用的输出。加上前次失败作业的 140 次调用，本轮政策分类共 1032 次调用，
输入 2713038、输出 1558613 token，仍不包含更早的 OpenIE、向量和 canonical schema 成本。
EngineDeadError 出现在完整输出后的 SIGTERM 清理阶段；已结合退出码和完整 schema 验证，
不是数据生成中途失败，不重跑已完成分类。

原标签图的已完成四任务（FCSH、FCMH、SH、LoCoMo）共 2286 题、13 组已核对：
与 canonical 状态/事件图共享相同冻结读出文件、原图节点来源和 RRF 配置，153 题排名改变。
其 FCSH 9B 为 64，相比状态/事件合并图 61 有提升，但 FCMH 从 6 降到 5；两者仍未过目标。
这只说明改变合并会产生取舍，不支持“去掉合并就全面更好”。完整矩阵尚未收齐。

MH schema 完整校验后释放 `6337366` 和 `6338634`，两条图构建链继续。
2Wiki 分类结束后，其两条构建链也已启动。尚未启动的 `6337382_0/1` 移到 gpu/norm-gpu
并约束 L40S，使用新释放的两张卡；未重启任何运行中的 QA。

15:46 原标签图与 canonical 状态/事件图的六任务 3386 题、15 组全部核对通过：
同一冻结读出文件、原始图来源和 RRF 设置，case 完全一致，156 题的来源排名改变。
两组普通接口验证均已完成，每组覆盖 FCSH 100/100 与 LoCoMo 1986/1986。
因此当前差异不是仅缓存回放可以工作；但尚不能由结构变化直接推导 QA 改进。

另查原始检索的 dense_fallback 标记：2Wiki 55/1000、FCSH 2/100、FCMH 12/100、
LoCoMo 235/1986、MH 6/100、SH 8/100，总计 318/3386。
绝大多数查询没有走纯 DPR fallback，不能把图侧改动收益有限解释为“大部分查询绕过图”。
两组新图均已完整落盘，余下工作为在运行的完整 QA 与最终汇总，不重跑已完成项目。

## 2026-09-13 15:57：原标签对照全矩阵完成

| Reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9B | 90 | 60 | 64 | 5 | 56.68 | 57.61 | 4/6 |
| 4B | 90 | 61 | 65 | 6 | 51.66 | 56.10 | 6/6 |
| 2B | 79 | 50 | 62 | 7 | 43.10 | 42.31 | 3/6 |

每个 reader 均完成 3386 次 QA，输入 token 为 9550192。
原标签对照没有改善三 reader 的共同胜出覆盖，不替换此前 5/6、6/6、3/6 的最好候选。
在发现旧 schema 过度合并后，本组给出了真实图侧对照；语义区别更合理并不自动带来
全面 QA 增益。不能将取消合并包装成已经成立的通用改进。
canonical 状态/事件三组仍有 2Wiki 和少量 MH QA 在运行，保留原作业继续收齐。

下一项可检查的现有设置是 RRF 候选池：当前两路都只取前五，融合后仍取五。
原 HippoRAG 的 num_to_retrieve 只切最终排序，不改变 recognition/PPR 算法；扩展候选池
可以作为固定单轮检索的参数对照，但不能归为纯建图贡献。尚未实现或提交该对照。
原缓存的 dense_fallback 查询 reset 全零，只有前五结果，不能假造后续排名；若扩池，
必须经普通检索接口取得完整候选并核验旧前五，不能仅从现有五条文件补候选。
