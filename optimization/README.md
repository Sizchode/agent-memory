# 图构建优化

本轮工作概览与最终结果见 [summary.md](summary.md)，运行进度与证据见 [progress.md](progress.md)。
本次目录提交的运行依赖边界见总结末节，不能视作独立可复现的干净克隆版本。

目标：同一算法与配置在 Qwen3.5-9B、4B、2B 上均严格超过既定六项完整任务的完整 baseline 最佳成绩。
三个 reader 各自 5/6 仅为阶段里程碑，最终争取全部 6/6；比较范围不是未经验证的整个领域。
当前 benchmark test set 明确作为开发集。下述 heuristic 记录早期已执行实验；
最新提供的 AGENTS.md 禁止 heuristic，已请求澄清，回复前不新增此类规则。
不将开发集成绩称为独立测试泛化，不硬编码题号、答案、任务路由或评测结果。

## 首轮：固定读取，改变图权重

复用 HippoRAG2 的完整抽取、向量和图。构图函数只接收图、实体来源和原文向量。
原有 recognition memory、reset、PPR 参数和最终五段原文保持不变。
`original` 回放检查与旧检索的逐题一致性；不把回放当成新方法。

- `relation_only`：去掉同义连接的权重贡献，保留来源支持的事实边与段落边。
- `balanced`：对原始边权进行对称节点强度归一化，指数为 1/2。
- `contextual`：每个实体的上下文为其来源段落向量均值；同义边权乘以两端上下文的非负 cosine，相同来源支持的事实边不受此乘法影响。
- `contextual_balanced`：先上下文校正，再做同样的强度归一化。

这些是待验证的构图 heuristic，不是已经成立的新颖性或提升结论。
主要底座和固定读取机制见 [HippoRAG2 原文](https://arxiv.org/html/2502.14802)。
不改已有数据加载、划分、转换、QA prompt、解码规则和评分器。
不恢复 HyperMem，不增加多轮查询，不扩大最终上下文条数。

## 第二组：按来源顺序整理关系支持

`latest_relation` 对规范化后的同一主体与关系，只保留最后一个来源段落的图连接支持；
同一末段中的多个值全部保留。关闭同义边，所有原文段落仍存在。
`latest_relation_balanced` 再加对称强度归一化。原图 query reset、recognition 与 PPR
参数不变。来源顺序来自既有 loader，不使用 OpenIE 导出的随机排列，也不把它称为
真实事件时间。这是可能损害历史问答或多值关系的 heuristic，必须完整报告负结果。

`replay_graph.py` 使用已验证的原图检索 seeds，先冻结新图，再读取问题进行 CPU PPR。
因此这一轮不需要新的生成或 embedding 调用，但旧调用的成本仍计入来源成本。

## 第三组：来源驱动的关系 schema

固定 Qwen3-30B-A3B-Instruct-2507 为每组语料的全部关系标签生成 canonical label、
单值/多值属性及内容/话语结构分类。每个标签提供前两个不同来源三元组作例子；
不读取问题或答案，不用外部常识纠正输入事实。48 项一批，最多 16 并发；
完整记录超长时分批，不截断关系例子。模型调用、失败输出及 usage 单独保存。

`schema_latest` 只对分类为单值的 canonical relation 按来源顺序整理，保留多值关系
全部支持，去掉话语记账连接和同义边。`schema_latest_synonyms` 保留同样的关系
整理，但恢复原有同义边作为对照。两者仍使用同一套原始查询 seed 与固定 PPR。
分类本身可能出错，canonical label 也可能跨批次不一致，不能当作人工真值。

联合归一对照：`canonicalize_schema.py` 用既有 embedding 和 sklearn BisectingKMeans
按语义分组，再让同一个生成器对每组标签联合判断等价关系；每个输入 id 必须恰好
出现一次，保留原标签作代表，不生成新事实。分类版与联合归一版独立运行全矩阵。

`canonical_latest` 与 `canonical_latest_balanced` 使用联合归一后的关系标签，但不依据
模型的单值/多值判断决定是否整理：所有内容关系都按最后来源处理，同一末段多值仍保留。
原 schema 分类不被覆盖；这是显式的最新来源 heuristic，并不宣称所有关系在语义上
都是单值。仍用原始 query reset，对比不整理多值关系的版本，并完整检查历史信息损失。

统一权重搜索以 `canonical_latest` 为底图，补充五个固定配置：同义边权重比例
0.05/0.2，分别配合强度归一化指数 0/0.25，另加比例 0、指数 0.25 的对照。
来源支持与缩放后的同义支持仍取最大值，再按两端强度乘积的指定幂做归一化。
这是显式的开发集参数搜索；所有配置都跑六任务、三 reader，不按任务挑不同参数。

来源支持比例对照：`canonical_provenance` 在 canonical 最新来源图上，将每条实体与
段落连接乘以该实体在此段落中的保留三元组支持数/原始三元组支持数。事实边不改变；
原文仍完整保留。另测强度归一化和 0.05 同义连接两个对照。其意图是减少仅因少量
有效事实而仍被强连接的旧段落，不将答案字符串出现率当成证据正确率或新评测指标。

自适应同义连接：统计实体参与的原始事实中，有多少 canonical 主体-关系-客体断言
仍存在于当前图。重复出现但仍有效的同一断言全部算兼容，不把去重当成事实冲突。
原同义边乘以 0.2 及两端的兼容支持比例，再与来源事实支持取最大值。另测强度
归一化和段落支持比例两个对照。这是来源驱动的统一权重 heuristic，不使用任务名、
reader 名称或题目决定连接强度，也不把该比例解释为校准概率。

## 单列对照：来源节点的事实表示

`compile_graph_context.py` 在检索前，把 canonical 最新来源图保留的原始三元组按
所属来源节点序列化，保留三元组原始表面文字、来源位置及已有时间 metadata，不新增
事实或生成调用。历史原文仍保留可追溯链接。随后用原图与 canonical 图各自的原生
top5 来源排名读取同一套新节点表示，形成 `original_graph_compiled` 与
`canonical_graph_compiled`。两者不进行查询时改写或多轮检索。

这一组改变 reader 所见上下文，必须与五段原文的图权重实验分开报告。五个节点不代表
等 token 预算，也不保证每个节点表示都更短；实际 QA token 记录用于成本比较。
QA prompt、解码和 answer 指标不变；对生成表示直接做原文字符串 gold-passage
匹配不等于来源级 Recall，因此不能用这一数值证明图检索退步或提升。
节点表示同时支持 `load_optimized_memory` 的普通新问题接口，不依赖预存题号。

保留原文的后续对照：`compile_graph_context.py --with-source` 不删除或改写原文，
把来源位置、已有 timestamp、完整原文和图保留的原始三元组分别标注。
`canonical_graph_with_source` 与 `adaptive_graph_with_source` 分别沿用 canonical 最新来源图
和自适应同义边图的排名，两者读取同一份离线冻结表示。没有查询时摘要或附加生成调用。
这是原文加来源事实提示，不是压缩，也不能称为新的语言模型 contextualization 算法。
QA 输入 token 可能增加；与各自五段原文结果比较时必须同时报告表示变化和实际成本。
该对照由纯三元组表示在 SH/MH 上的退步触发；抽查同时发现措辞变化造成的固定指标扣分，
以及关系语境变化，尚不能把全部退步归因于信息丢失。原评分器保持不变。

来源邻接窗口：`--with-source --source-window 3` 在离线阶段为节点附加同一非空时间
metadata 下连续的前后三条来源原文，并保存 `window_sources`。遇到时间变化立即停止，
不把之后又出现的相同时间字符串跨段连接；缺失时间 metadata 时不扩展，也不按任务名
判断是否扩展。仍按中央节点原有排名返回五个节点，但实际 reader 上下文会增加。
`canonical_graph_source_window` 与 `adaptive_graph_source_window` 分别沿用两种图。
该机制借鉴 [LlamaIndex sentence-window](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/sentence_window.py)
的离线邻接上下文表示及默认三项窗口，但使用既有来源记录而非重新分句，且新增时间边界，
不是官方 LlamaIndex 方法复现。窗口大小和边界是显式 heuristic，不声称等 token 或纯拓扑收益。
全六任务仍评测；无时间 metadata 的任务属于输入表示不变的重复对照，不算独立机制增益。

## 单列对照：固定词面与图排名融合

`prepare_hybrid_graph.py` / `retriever/hybrid_graph.py` 在同一图的完整原文 corpus 上复用
已有 `BM25Baseline`（k1=1.5、b=0.75），将图检索与 BM25 各自的前五来源以
[RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) 合并，固定 rank constant=60，
最终仍返回五个来源。使用截断前五而不是原论文完整排名，属于显式配置选择；
BM25 零分项不提供排序支持，同分按图排名优先、随后词面排名的新来源稳定排列。
没有查询规划、追加 LLM 或多轮检索，不读取其他 baseline 的预测或答案。

`original_graph_rrf` 与 `canonical_graph_rrf` 比较相同融合下的原图和新图；
`original_graph_rrf_window` 与 `canonical_graph_rrf_window` 再使用同一套来源事实/窗口表示。
四组全六任务、三 reader 评测。RRF 本身不是新颖性或图构建贡献；与没有 RRF 的旧结果
比较时必须说明检索改变，与同一 RRF 设置的原图对照才可分析图的额外收益。
构建时只写入原图引用、边权、来源顺序和冻结表示引用；普通新问题接口直接在这些
来源上执行原有一次 HippoRAG 检索与一次 BM25 查询，不依赖预存问题。

## 单列对照：来源窗口压缩

`prepare_context_packing.py` 固定 `canonical_graph_rrf` 的来源排名，对窗口读出做两组
对照。`canonical_rrf_deduplicated` 预留所有中央来源，再将邻接来源按原排名首次出现
的位置附加一次，保留中央原文、保留事实与全部唯一邻接原文，不按问题相关性删来源。
`canonical_rrf_sentence_facts` 在相同去重上，把原三元组字段用空格连接并逐条换行，
替代 JSON 数组序列化；不调用 LLM、不合并不同断言、不改实体、关系、否定或客体字段。
这是读取表示与去重 heuristic，不是新增图拓扑机制。完整事实字段仍以结构化列表保存。
无邻接窗口的任务中，仅去重版本的输入保持不变，重复 QA 的波动不能算作压缩收益。

动机来自已构建的融合窗口输入：LoCoMo 1986 题中 1314 题包含重复来源，64522 次
来源出现对应 54236 个逐题唯一来源，共 10286 次可去重出现。它只是输入冗余统计，
不是新评分指标或证据正确性标注。实际 QA token、分数及所有唯一来源保留检查分开报告。

## 加载与核验

事实索引一致性对照：`index_latest` 和 `index_schema` 分别沿用来源合并与联合 schema
的裁图规则，同时只允许有保留来源支持的事实进入候选索引，并同步实体来源计数。
保留事实的文字与向量均直接取原索引，不重写事实。该轮改变了检索候选，因此不复用
旧 query reset；重新运行原有一次 recognition 与 PPR，缓存未命中调用同一生成器。
每题同时检查原图检索与旧基线完全一致，防止新 GPU/运行环境引入未核对的差异。
这与前几轮仅改变边权、固定 reset 的消融应分开报告，最终仍为五段原文。

`run_fact_index.sbatch` 在 B200 上使用既有 lightmem 环境的 Torch 2.8/cu128，补装与
HippoRAG 环境一致的 igraph 0.11.8、litellm 1.73.1 和 boto3 1.43.89；不更换原基线环境。
数据读取仍在原 HippoRAG 环境调用同一个 `_load_groups`，通过进程间传递原有数据对象，
避免不同 NLTK/tiktoken 版本改变分句；不新增划分、转换或 chunking 规则。
跨环境检查中，MH 原图检索仍出现差异，因此完整主对照切回原 HippoRAG 环境与 L40S；
同一个生成器用两张 L40S、TP=2 承载。此前通过检查的短任务保留为独立运行结果，
不将未完成的跨环境任务混入新运行。每题原图一致性检查继续生效。

`retriever.hipporag.load_optimized_memory(config, artifact_directory, runtime)` 能直接加载
落盘权重，用原生 HippoRAG 处理新问题，不读取预存 query seeds、问题列表或预测。
新问题的 recognition cache 未命中时，需要 config 中可用的本地生成服务。
`run_graph.py --phase verify` 对完整任务重新走该接口，核对输出与快速回放一致。

## 文件与产物

- `graph_construction/`：查询无关的离线图变换。
- `retriever/`：原生 HippoRAG 适配及识别缓存保护。
- `run_graph.py`：复用原数据入口，全量构图、检索和三模型评测。
- `run_graph.sbatch`：GPU 作业入口，命令行可覆盖分区和 GPU 型号。

大产物放在原 scratch 输出根目录的新实验目录。图保存为原图引用和按原边序排列的
权重向量，不复制所有 embedding 或覆盖原图。识别缓存复制到独立运行目录；
命中复用与新调用成本分开。默认缓存缺失即失败，不能悄悄变成 dense fallback。
抽取历史成本仍引用原 `efficiency.jsonl`，不宣称构建零成本。
`report_results.py` 另汇总每个完整任务的实际 QA 输入/输出 token 与调用秒数；缺失 usage
不补零，记录数量须与完整预测一致。这不含建图、检索、模型加载或排队时间，GPU 型号
不同的调用秒数不当作等硬件速度比较。
