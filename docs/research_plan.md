# Agent Memory 图构建研究计划与相关工作

## 当前计划：分布内图优化，三个模型各达到 5/6 最优

更新：2026-09-12。本节记录本次讨论确定的目标与执行顺序，优先于下文历史计划。
用户已授权围绕既定 benchmark 优化构图；最新指令要求本轮只更新并推送本文，
因此本轮不修改算法、不提交 GPU 作业。下文旧的“先批准诊断、再申请改动”是
历史阶段安排，不作为今后执行本次已授权优化的重复审批要求。

### 1. 目标与达标口径

核心贡献定位为更好的图构建：完整摄入允许的原始材料，完成一次构建阶段后冻结图，
再做检索和 QA。构建阶段内部可以有多次抽取、归一和整理调用；one-shot 不指一次
LLM 调用。不开发 agentic retrieval/RAG、查询规划或多轮搜索，不要求提出通用记忆
理论或证明分布外泛化。方法属于 benchmark-informed development，如实记录开发
使用的数据和结果，不把同一批成绩包装成独立泛化验证。

**达标要求：Qwen3.5-9B、4B、2B 各自在至少五个任务上，严格超过当前已完成 baseline
的该任务最佳成绩。** 使用未舍入分数判断；并列单列，不计作严格胜出。同时报告三个
模型共同胜出的任务，但不额外要求三个模型必须赢相同五项。六任务始终全量运行，
不预先丢掉第六项。最终使用同一构图方法和配置，不能按任务、模型或题号挑选不同
候选拼成 Ours；保留原有 benchmark 各自的输入和 QA 协议。

比较组：BM25、Dense、HippoRAG 2、Mem0 ADD-only、LightMem 整理前后、AnchorMem
两种设置及 CatRAG。旧多来源 ensemble 探索结果单列。
以下目标已从保存的 `summary.json` 核对，百分制，非全领域 SOTA 排名：

| 完整任务 | 题数 | Qwen3.5-9B | Qwen3.5-4B | Qwen3.5-2B |
|---|---:|---:|---:|---:|
| SH-Doc_QA | 100 | 88.00 | 89.00 | 79.00 |
| MH-Doc_QA | 100 | 59.00 | 54.00 | 49.00 |
| FactConsolidation-SH | 100 | 66.00 | 56.00 | 54.00 |
| FactConsolidation-MH | 100 | 6.00 | 5.00 | 8.00 |
| LoCoMo | 1,986 | 55.37 | 50.34 | 44.80 |
| 2WikiMultiHopQA | 1,000 | 51.87 | 49.76 | 37.02 |

这张目标表包含输出预算较大的 AnchorMem 官方检索；另报固定 top5 对照，不把两者
混称等上下文比较。前四任务沿用 substring exact match，LoCoMo 沿用类别评分规则，
2Wiki 沿用 answer F1，不把不同指标平均成新总分。完整输入范围仍是四个既定
MemoryAgentBench source、全部 locomo10 五类题目、HippoRAG 官方 2Wiki 复现文件
全部 1,000 题。划分、转换、加载和原有指标定位见本文历史记录及 `results.md`。

### 2. 首轮保持项与允许的改变

首轮只改变构图：复用原始输入、Qwen3-30B-A3B-Instruct-2507 Generator、
Qwen3-Embedding-0.6B、HippoRAG 的检索过程与参数、最终五段原文，以及三个回答
模型的现有 prompt、解码设置和评分器。HippoRAG 原有 recognition-memory 调用保留，
不另加查询推理。每个候选的同一份检索由三个 reader 复用。

构图只读取允许的 source 内容和来源 metadata，不读取评测问题、答案、支持证据
标注或人工诊断标签。可以用完整结果判断下一轮改动，但不能把题号、答案、手写
别名表、任务分支或按题型定制的规则写进方法。不得编造重要性公式、剪枝阈值或
结果；使用已有方法也须检查其实际规则，不能认为“论文用过”就符合无 heuristic
要求。已有底座参数继续明确记录，新增操作与参数必须有具体方法依据。

复用已有抽取仅限内容及设置兼容的候选。抽取原始成本与新增转换成本分开报告。
原图、原文、缓存和历史结果保留；新产物独立保存，不融合不同 baseline 的 memory。

### 3. 分阶段执行

| 阶段 | 具体工作 | 完成产物与下一步判断 |
|---|---|---|
| 0：接口和机制核查 | 检查 HippoRAG 抽取、phrase identity、同义边、段落边、图组装和查询 seed；核对数据划分、转换、完整覆盖与评分；追溯已有同名实体和条件丢失案例 | 明确改动如何影响拓扑或索引，写出操作、来源方法、设置与基于真实候选数量的成本估算；不先要求完成全量语义标注 |
| 1：候选 A，实体归一 | 从 HippoRAG 原抽取和来源原文出发，核查 KGGen 的实体聚类/归一算法及代码；建立来源可追溯的实体身份映射，按原有段落关联和边语义重建图 | 全六任务、三个模型共 18 格；检验是否改善关联及减少错误连接，而非仅统计节点变少或连通性变高 |
| 2：候选 B，上下文感知构图 | 独立于 A，参考自包含 proposition 与 contextual retrieval 的原方法，消解实体和关系中的上下文依赖，保留原文实际给出的参与者、时间、否定及适用条件 | 同样完整 18 格；首轮仍返回原文，区分构图收益与换成生成事实作为 QA 输入的收益 |
| 3：有证据再组合 | 对照 A/B 的逐任务逐模型结果和改善、退步案例；确有互补机制时运行 A+B | 补齐 baseline/A/B/A+B 消融矩阵；若不互补，不为凑模块强行组合 |
| 4：按失败机制追加 | 只有错误关联或重复表示确实限制结果时，才进入来源支持的关联修正或结构压缩；先选定并核查已有算法 | 每个新增候选仍完整 18 格；不能按频率、度数、答案命中或自造 confidence 阈值删事实 |
| 5：定稿和验证 | 选择一套全任务、全 reader 通用的构图配置，复核覆盖、原评分、成本和三个 5/6 计数 | 代码、完整结果、必要消融、成本表、具体反例和达标状态；达标前不宣布完成 |

候选 A 需要与 HippoRAG 已有同义连接明确对照。实体合并会改变多跳可达性，也可能
把不同人物混为一谈，因此保留所有来源映射。先审计 KGGen 的具体算法；遇到与
无 heuristic 约束不兼容的步骤，应明确说明并另选已有机制，不能悄悄替换成新规则。

候选 B 必须先确定来源上下文协议，沿用已有 source 单元及其可用上下文；不把整个
增长语料反复塞进每次调用，不静默截断，也不为适配方法重造 dataset loader。
若结构必须依赖新 retriever 才能工作，应明确列为不同实验，而不混入构图单因素
对照。其失败可能来自生成失真，不能因为保留更多字段就假定信息一定更完整。

Self-adaptive 在这里指由输入语料的语义决定实体/关系如何组织，不是按 benchmark
名称分支。关系 schema 自归一可参考 EDC，但要先确认它实际改变图或候选匹配；
仅重命名 PPR 不使用的关系标签，不能预期必然改善检索。压缩和 pruning 是条件性
后续路线，不在第一轮同时堆叠。

### 4. 调度、迭代和交付

- 每个候选包含 3,386 次问题检索、三个模型共 10,158 个回答。A/B 合计 36 格、
  20,316 个回答；若执行 A+B，再加 18 格、10,158 个回答。设置相同时复用已有
  HippoRAG 完整结果，不将复用计成新实验。
- 使用现有环境和模型缓存，gpu-he 最多同时六个单卡作业。按完整任务并行，QA
  依赖对应构图和检索完整成功。此计划不要求代理分工。长作业沿用 15–20 分钟
  检查间隔；异常或用户消息可提前检查。
- 接口检查可使用少量输入，但不把它当 benchmark 成绩或替代全量。先基于实际
  source/candidate 数量估调用量，再依据真实吞吐估时间，不承诺未测的完成时间。
- 输入/输出 token、模型调用耗时、图处理、embedding、检索、QA、加载排队及
  总 wall time 分开记录；并发调用时间之和不等于阶段时间。失败和取消尝试单列，
  缺失 usage 不补零。
- 每个完整候选完成后更新 18 格表，列出改动机制、改善和退步，再选下一项构图
  改动。接口错误修复后重跑；有效负结果保留。连续负证据促使更换假设，不靠增加
  无关模块、缩减任务或硬编码挽救成绩。
- 最终核对问题唯一性和全覆盖、逐题均值与 summary 一致、QA 设置未变，以及三个
  模型各自的严格胜出数。开发阶段沿用 seed 42；额外 seed 应在选定配置后用于
  验证并分别报告，不挑有利重跑。未达标交真实结果及下一步，不把预算耗尽当成功。
- 产物用 JSON/JSONL，脚本按功能命名，不新增 hash、实验数据库或锁文件框架。
  保留已有用户改动。提交和推送范围按当次用户指令执行，不加 co-authorship。

### 5. 具体方法依据

[HippoRAG 2](https://arxiv.org/html/2502.14802) 是固定读取流程与主要构图参照；
[KGGen](https://proceedings.neurips.cc/paper_files/paper/2025/file/2b368455e832d2b1a60bcad8c4c6481f-Paper-Conference.pdf)
用于核查实体归一；[EDC](https://aclanthology.org/2024.emnlp-main.548/) 用于关系
定义及归一参照，其训练型 schema retriever 不自动纳入当前范围。
[Dense X Retrieval](https://aclanthology.org/2024.emnlp-main.845/) 与
[Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
提供自包含事实和上下文索引依据；
[SiReRAG](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f9668d223e713943634dce9c66e8f2c1-Abstract-Conference.html)
的跨段 proposition 组织可作后续参照，但原版微调抽取器须单独说明；
[MemGraphRAG](https://arxiv.org/html/2606.00610v1) 的构图处理可作对照，其频率过滤
和阈值不得未经检查直接采用。这些工作的原始数据、指标及成绩不替换本项目协议，
引用它们也不自动证明我们的新颖性或效果。

## 上下文图记忆调研与已完成对照

### 上下文图记忆：定向 survey（2026-09-12）

结论：值得研究，但不能把“时间/来源字段＋超图”作为尚无人研究的贡献。
以下区分事实上下文（谁、何时、在哪个事件中成立）与查询上下文（当前问题
需要什么）。前者影响记忆构建，后者常用于检索调权；两者不能混称。
此次是机制定向检索，不是穷尽所有论文或 Google Scholar 引用数量审计。

| 2026 工作及发表依据 | 已有机制 | 对我们意味着什么 |
|---|---|---|
| [HyperMem，ACL](https://aclanthology.org/2026.acl-long.1627/) | 主题—事件片段—事实三级；同主题片段和同片段事实由超边组织，保留原始对话与时间线索 | 已是上下文中的事实抽取，不能说它完全没有上下文。仅保留为相关工作，不再安排复现或后续实验 |
| [APEX-MEM，ACL，§3–6](https://aclanthology.org/2026.acl-long.749.pdf) | 事实绑定有时间的事件；只追加历史，查询时通过工具解决冲突 | “事件化事实＋保留版本＋读取时消歧”已有直接先例；其多工具 QA 不能不加说明压成固定 top5 后宣称原版 |
| [MemORAI，Findings ACL，§3–4](https://aclanthology.org/2026.findings-acl.1408.pdf) | 实体—话轮—片段异构图，实体上下文描述与话轮来源；按查询调节图传播 | 与“上下文增强图”非常接近。明确 inference-only；但只保留用户相关信息的入口不一定适合文档任务，不能擅自改掉后称官方默认 |
| [GAM，ACL，§3–4](https://aclanthology.org/2026.acl-long.1600.pdf) | 局部事件推进图与全局主题网络分开，在语义边界整理；检索结合时间、角色、置信度 | “缓冲隔离、完整事件再合并”及系统式命名已有先例；不能单凭借用系统概念主张新颖性 |
| [CatRAG，Findings ACL，代码入口](https://github.com/kwunhang/CatRAG) | 查询相关边权、符号锚定及关键事实到原文的连接增强 | 主要是查询时导航，不是专门保存事实有效范围；适合作为区别于建图改动的对照 |
| [Does Memory Need Graphs?，ACL](https://aclanthology.org/2026.acl-long.1232.pdf) | 分开分析记忆表示、索引、检索与回答，在 LongMemEval/HaluMem 做阶段对照 | 图方法的总分不等于图结构的单独贡献；这是本项目归因实验的重要方法学参考，不是另一种必加算法 |

邻近工作：[MemGraphRAG](https://arxiv.org/html/2606.00610v1) 的
[项目页标注 KDD 2026](https://github.com/XMUDeepLIT/MemGraphRAG)，本次未独立核对出版方记录。
它已对逻辑、时间、粒度冲突分类，回原文裁决，并补时间或细化关系；
因此“根据原文修复冲突图”也不是空白。它主要面向文档 GraphRAG，不能
因名字含 memory 就等同于长期对话记忆。仍未接入，不额外申请 GPU。

不计入严格 2026 已发表清单的参考：
[Zep，2025 预印本](https://arxiv.org/abs/2501.13956) 已研究时间知识图；
[T-Mem，2026 预印本](https://arxiv.org/html/2606.15405v1) 把检索线索与事实/场景分开；
[EdgeMem，2026-09-03 预印本](https://arxiv.org/html/2609.05553v1)
以原始话轮为证据，用内容、时间、事件锚点建超图，不调用生成模型管理记忆。
EdgeMem 尤其提示：保留原文、减少生成调用也已有非常近的工作，尚不能据此许诺新颖性。

评测边界：HyperMem、APEX-MEM 的 headline accuracy 使用 LLM judge，
不与本项目确定性分数横比。GAM 报告 LoCoMo 的 F1/BLEU-1；
MemORAI 在 LoCoMo-10、LongMemEval-s 报告 F1 等及话轮/会话级
Recall@3/5/10，同时另有 judge 分数。采用其指标需要核对答案归一化、
证据单位与任务覆盖，不能照搬表格数字。各方法完整代码的训练依赖与六任务
适配尚未全部审计，不能把“论文没有强调训练”当作已验证 training-free。

下一步只验证已有错误的机制，不先发明评分公式或修改 baseline：对照原文、
生成记忆、检索结果，区分限定条件丢失、实体/事件绑定错误、相互不适用事实
被组合，以及证据完整但回答失败。观察到前两类才支持改写入，第三类还须
分清建图与检索责任，最后一类不直接支持重建 memory。需要人工语义判断的
案例应标注为人工诊断，不包装成确定性 gold 指标或因果结论。

值得继续核实的窄问题是“同主题关联是否混淆了事实的适用范围”，不是宣称
所有旧图都没有上下文。是否存在可复现的改进空间，要等逐例证据和固定读取
流程的对照；不承诺新颖性或提升。保持既定六任务全量与三模型，旧结果不动。
用户本次要求作业轮询间隔 15–20 分钟；异常或新用户消息可提前检查。

可执行性补查：[GAM §4.1](https://arxiv.org/html/2604.12285v1) 明确
training-free，使用现成指令模型，原设定检索 k=10、MiniLM embedding/reranker，
实验硬件 RTX 4090。其目标函数由语义边界触发策略近似，并非训练求解或
给出了全局最优保证。[MemORAI §4.1](https://arxiv.org/html/2605.01386v1)
同样明确不额外训练。两篇本次检查的正文及定向搜索未定位到可确认的作者
代码仓库；同名 MemorAI 应用不可冒认。故先作 related work/机制参照，不能
承诺立即按官方代码接入；若以后运行，统一 top5 等与其原设置的差异须单列。

### 建图研究边界与新增对照候选

补充实现核查：MemGraphRAG 的[官方索引入口](https://github.com/XMUDeepLIT/MemGraphRAG/blob/main/code/index.py)调用 `index_with_memory`；[该流程](https://github.com/XMUDeepLIT/MemGraphRAG/blob/main/code/src/MemGraphRAG.py)依次抽取三元组、归纳 schema、过滤、检测/解决冲突，再安装最终图。公开入口没有要求先训练新权重，但尚未本地运行验证全部依赖。它不是可以直接宣称为严格流式的入口：先读取整个 corpus；`conflict_streaming_only_previous` 只是冲突比较方向设置。因此可作为全局建图/冲突处理的对照，不能不加核实地说它已经实现我们讨论的局部增量更新。仍未接入或修改现有实验。

检索成绩衡量的是图与检索器共同作用，不能单独等同图质量。先沿用有官方证据标注任务的 Recall@5/Precision@5；没有对应标注的任务不靠答案字符串命中或自造标签冒充 evidence recall。错题诊断分别核对原文证据、生成的事实/关系及实际返回内容，保留无法判定项，不使用 LLM judge。若实验只改变建图，应固定检索器、模型及评测设置；不同图格式需要适配时必须披露，不能把适配差异当建图收益。

最相关的新候选是 [MemGraphRAG（作者代码标注 KDD 2026）](https://github.com/XMUDeepLIT/MemGraphRAG)。它维护 schema、fact、passage 三层记忆，通过抽取、冲突检测与解决协调建图；[论文表 3](https://arxiv.org/html/2606.00610v1) 已专门测试更换建图器并保留不同检索框架。因此“全局一致性建图”或“可插拔建图”本身不能作为未有人做过的新意。论文同时使用字符串指标和 LLM judge；我们若接入仍只沿用当前确定性评分，不搬用其 judge 成绩。当前仅调查，未下载新环境、未提交此方法任务；需继续检查实现是否完全无需训练及其默认配置。

### CatRAG 已完成结果与接入记录

CatRAG 六任务 × 三个回答模型全部完成（18/18）。LoCoMo 9B/4B/2B 全类别
汇总为 51.56%/48.53%/44.80%，各 1,986 题；唯一题号、缓存检索、分类均值及
usage 已核对。2Wiki 三个模型的 Answer F1 为 51.52%/48.04%/36.56%，各
1,000 题；检索 Recall@5/Precision@5 为 72.33%/33.94%。完整分数见
`results.md`，不再保留已被最终结果覆盖的排队和中间完成数快照。

CatRAG 使用原生动态边权、锚点、事实增强及图传播，最终返回 top5；没有替换
为 dense 检索。原文字符串列表作为输入，LoCoMo 保留真实说话者和时间。
查询边权评分显式传入的输出预算为 9192，不能统一改成 2048。
`experiments/run_catrag.py` 提供 build-retrieve、retrieve-existing、evaluate，
复用既有六任务加载器、seed 42 和确定性评分器。

既有实现记录保留以下成本边界：同步与异步调用均需等待真实响应后记录 usage；
CatRAG SH 的早期计量有 328 条 access 调用缺失 usage，不能补零或当作完整成本。
缓存复用与本次调用分开，构建、检索、回答的 token 和耗时分别报告。
B200 与旧 Torch 的兼容性失败、H100 embedding OOM 等历史失败产物仍保留；
最终 embedding batch 为 8。节点摘要使用各任务独立的 cache_dir，不跨任务共享。
本次仅整理文档，未修改这些实现、配置或产物。

REMem 保留为相关工作，不启动构建或查询；其原生多轮工具访问不能冒充固定一次
检索与统一 QA。不同方法的 memory 不直接互用，已有模型缓存和环境继续保留。

## 范围与结论

本综述收录截至 2026 年 9 月 12 日能够核实正式发表信息的 **24 篇 2026 年论文**：ACL 主会 6 篇、ACL Findings 8 篇、ACL System Demonstrations 1 篇、ICLR 主会 4 篇、ICML 主会 2 篇、EACL 主会及 Findings 各 1 篇、KDD 1 篇。年份按正式会议发表年份，不按预印本首次上传年份。未把只有预印本或投稿记录的工作计入，也没有用 2024–2025 年论文凑数。这是有明确边界的文献综述，不是全年论文数量的穷尽统计。

研究对象分三层：14 篇会话或 agent memory 系统，包含层级记忆和系统展示；9 篇相关的文档图检索、语义记忆或知识图谱构建工作；1 篇记忆架构实证分析。后两层具有机制参考价值，**但不能全部称为“图式 agent memory 的直接竞争算法”**。同样，论文标题中的 agentic 不代表已经评测了长期交互中的规划、工具使用或行动成功率；本综述多数证据仍来自对话历史或文档上的问答。

最重要的结论是：**“把 memory 建成图”已经不是一个足够具体的新方向。** 事件节点、原文关联、时间限定、高阶关联、分层索引以及查询条件化的图遍历都有 2026 年已发表工作。研究需要定位究竟是哪一种信息在生成、组织或访问过程中被破坏，而不是先选择图、超图或编译器术语，再为它寻找问题。与此同时，已有对照研究发现图的收益取决于记忆表示和访问设置，错误的建图或检索策略也会降低成绩。[^s24]

以下逐篇介绍区分“论文采用的机制与评测”以及“对本项目的适用性判断”。没有把文献成绩当成本项目复现结果，没有据此宣布任何候选已解决现有错题，也没有修改或启动实验。

## 怎样分类才不会把不同工作混在一起

图的分类至少要同时回答两个问题：**节点究竟保存什么；边究竟表示什么。** “使用 embedding”不能与“建图”并列为互斥类别，因为许多图方法仍靠向量相似度选择入口。时间索引也可以与事件图、实体图并存。下表是对文献机制的整理，不是新算法。

| 结构类型 | 节点与边的主要含义 | 主要用途 | 代表工作 |
|---|---|---|---|
| 实体与关系图 | 人物、地点等实体；抽取的关系及其来源 | 通过关系连接跨段信息 | MemORAI、AutoSchemaKG |
| 事件与事实图 | 带参与者及语境的事件、事实；事件间关联 | 保留谁在何时做了什么，避免只剩实体名字 | REMem、CompassMem、GAM |
| 多视图和时序图 | 同一记忆上的实体、时间、因果等不同联系 | 区分语义相关、时间相邻和因果支持 | MAGMA、APEX-MEM、Hindsight |
| 关联线索图 | 对话、线索、标签、原文；线索与内容的指向关系 | 从容易匹配的线索找到具体记录 | AssoMem、AnchorMem、MRAgent |
| 超图 | 一条关联同时连接多个事实或事件 | 表示不能简单拆成两两关系的共同语境 | HyperMem |
| 原文索引图 | 段落、句子、实体；共现或语义连接 | 保留原文，用图定位证据，不抽取全部关系 | LinearRAG、BrowseNet、ZoomRAG |
| 层级记忆 | 不同抽象层的摘要与具体记录；上下层指针 | 从粗到细缩小查找范围 | H-MEM、LiCoMemory |

另一条必须分开的轴是**写入时做什么、查询时做什么**。Generator Backbone 读取原始输入，生成、抽取、压缩或更新 memory；记忆访问机制读取已存产物，选择或组合证据；Evaluation Backbone 最后消费证据并回答。有些论文让模型反复规划查询、探索图或判断证据充分性，这属于访问阶段的额外计算，不能计作一次普通检索，也不能把全部问答增益归因于生成记忆更好。

## 一、会话与 Agent Memory 系统

### 01. REMem — ICLR 2026 主会

**机制。** 将经历写成带语境的事件概要，并与短语及事实关系连接；事实携带时间限定。读取时，agent 调用检索、图访问和时间操作工具，逐步组合证据，而不是只返回一组相似片段。[^s01]

**评测。** LoCoMo、REALTALK、Complex-TR 主要使用模型裁判；Test of Time 使用 Exact Match。文中还分析词面指标及拒答行为，拒答 F1 与答案 token F1 是不同指标。

**价值与边界。** 很适合参考参与者、事件时间和跨记录组合，但完整系统改变了查询过程。论文自己的错误分析仍包含属性选错、时间范围不符，以及证据已找到却拒答。因此“事件图存在”不等于所有必要绑定都被正确使用。与 HippoRAG2 有机制联系，不据此计为原代码上的直接补丁。

### 02. AssoMem — ICLR 2026 主会

**机制。** 建立对话 utterance 与抽取线索的关联图，将相关性、图重要性和时间匹配等信号结合排序。这里 utterance 指一条发言，不是整场会话。[^s02]

**评测。** LongMemEval-s、-m、作者扩展的 -l，以及合成 MeetingQA；前三者不是三个独立数据集家族。报告发言级 Recall@k、nDCG@k，答案评估还包含模型裁判及其他文本质量分数。

**关键限制。** 不能整体标作 training-free：论文包含利用标注有用性的信号融合，以及小模型的去噪问答微调；大模型未微调不意味着完整方法不使用监督信息。适合相关工作对照，不应直接放进“无任务训练、无标签参与”的同设置主表。

### 03. MemORAI — ACL 2026 Findings

**机制。** 先筛选记忆、分段及压缩，再连接实体、对话轮次和片段；事实关系关联其原始发言。查询构造局部子图，并用查询相关的边权执行 Personalized PageRank，即从相关入口传播重要性，最后返回发言及支持三元组。[^s03]

**评测。** LoCoMo-10、LongMemEval-s；会话级及发言级 Recall@3/5/10，F1、BLEU、ROUGE、BERTScore 和 GPT-4o 裁判分。论文明确无额外微调。

**价值与边界。** 同时覆盖“准确定位哪一轮说过”与“避免传播到无关实体”。但筛选、分段、表示和访问共同改变，不能只把端到端提升归于图。对于输入并非真实对话的任务，个人信息筛选的适配尤其需要核对。

### 04. AnchorMem — ACL 2026 Findings

**机制。** 以抽取的原子事实作为检索锚点，保留与原始内容的连接，再构建关联事件语境；读取事实后可展开相应上下文。它不是把多个 baseline 的输出拼接起来。[^s04]

**评测。** LoCoMo；报告 F1、BLEU-1，也有模型裁判准确率及成本分析。不同访问设置是否展开原文和事件，会改变最后提供给答案模型的内容量。

**价值与边界。** 是本项目已有的直接比较对象。“事实加来源再加关联语境”不能作为新的贡献。原文仍在也不能证明抽取事实正确；生成层错误、事件层错误、实际返回是否包含原文要分别检查。

### 05. CompassMem — ACL 2026 Findings

**机制。** 以事件为中心建立逻辑关联图，访问时先形成目标，再沿图探索、跳过无关节点并改写查询。论文名称为 *Memory Matters More*，CompassMem 是方法名。[^s05]

**评测。** LoCoMo 和 NarrativeQA，主表报告 F1、BLEU-1；同时分析构建与回答的成本。论文还比较强模型构建记忆、较小模型访问与回答的分工。

**价值与边界。** 与“生成记忆好坏、后续如何组合”直接相关。但生成阶段较快不代表系统总成本低，多步访问仍需额外模型调用。只保留它的图却去掉访问流程，不能称完整官方方法；其默认流程也不能直接等同当前一次 top-5 检索。

### 06. Synapse — ACL 2026 Findings

**机制。** 对话轮次作为经历节点，抽取概念作为语义节点，通过时间、抽象和关联边连接。查询激活相关节点并沿边传播，同时抑制竞争节点、处理衰减和不确定性。[^s06]

**评测。** LoCoMo，主表包含 F1、BLEU；附录的主要汇总排除 adversarial 类。拒答相关成绩不能与普通问答 F1 混算。

**价值与边界。** 是“传播而非独立相似度排序”的直接参考。论文也讨论过度聚焦导致细节被压制的问题；抑制噪声并不必然保留问题需要的少数线索。其稀疏化、衰减和激活参数属于方法定义，不能只搬名称而自行重设。

### 07. MAGMA — ACL 2026 主会

**机制。** 同一份记忆具有语义、时间、因果、实体四种图视图；根据查询意图选择访问策略。把不同关系拆开建图并按问题选择，已经是明确的已发表设计。[^s07]

**评测。** LoCoMo、LongMemEval；主要结论采用模型裁判，同时提供词面分数。尤其要注意附录表 9：LoCoMo 总体 F1 为 MAGMA 0.467、Nemori 0.502。裁判分领先不能换写成 F1 领先。

**价值与边界。** 可用于比较时间与因果关系是否需要不同访问方式。但是因果边仍是模型生成的解释，不等于因果关系已被实验验证；也不能因四图结构复杂就认定比单图更好。

### 08. GAM — ACL 2026 主会

**机制。** 用局部事件演进图记录正在发展的对话，到语义边界时整理到长期主题关联网络；访问兼顾角色、时间和语义。这里的 GAM 特指 *Hierarchical Graph-based Agentic Memory*，不与其他同名方法混用。[^s08]

**评测。** LoCoMo 前四类问题和 LongDialQA，使用 F1、BLEU-1；采用多个回答模型，论文明确 training-free。

**价值与边界。** 与多人对话中的主体混淆、事件切分和长期组织有关，并且确定性指标与本项目较接近。其对话语义边界设计是否适合文档和事实更新任务尚无本项目证据；不能从 LoCoMo 的增益推断六项任务都会改善。

### 09. HyperMem — ACL 2026 主会

**机制。** 将主题、经历、事实组织成多层超图，用超边保留多个记录的共同语境；通过词面与语义索引从粗到细访问。[^s09]

**评测。** LoCoMo 的主要成绩 92.73% 是模型裁判准确率，不是 token F1。论文分别列出离线构建与在线回答成本；在线检索不调用模型，并不代表前期构建便宜。

**价值与边界。** 仅作高阶关联的相关工作，不再安排复现、适配或后续实验。现有论文评测不能证明它对长文档或反事实更新任务有效。

### 10. APEX-MEM — ACL 2026 主会

**机制。** 用实体和事件的属性图表达事实及时间，保留历史，在查询时处理冲突；agent 通过实体查找、图查询、搜索等工具读取记忆。[^s10]

**评测。** LoCoMo、LongMemEval、SealQA-Hard；主要问答成绩使用模型裁判，构建质量分析也使用裁判。后者不是人工 gold graph 上的确定性抽取正确率。

**价值与边界。** 历史保留和时间表达与版本问题相关，但工具型查询能力也是系统的一部分。对于给定反事实且按序号更新的任务，不能擅自把现实时间或“更可信的事实”作为替代规则。

### 11. LiCoMemory — ACL 2026 Findings

**机制。** CogniGraph 分为会话摘要、实体关系、原始片段三个层次，通过连接定位来源；图主要负责索引，避免把全部内容塞进图节点。查询自上而下，再结合层级、语义和时间重排。[^s11]

**评测。** LoCoMo、LongMemEval；模型裁判准确率及 Recall@15，还分析查询 token、延迟和构建开销。

**价值与边界。** 已明确提出“内容保存与图的组织职责分开”，因此这句话本身也不能作为新颖性。时间衰减与正确执行历史查询不是同一件事，不能默认近期记录总比旧记录更有用。

### 12. Hindsight — ACL 2026 System Demonstrations

**机制。** 区分外部事实、自身经历、观察总结与主观看法；事实之间建立实体、时间、语义和因果联系。读取结合向量、关键词、图传播和时间过滤，再融合排序；另有反思操作。[^s12]

**评测。** 展示论文汇总 LoCoMo、LongMemEval 准确率，并指向较完整技术报告。该展示稿没有给出足以复核为 token F1/EM 的完整评分定义，因此本综述不把其 headline accuracy 当作确定性分数。

**价值与边界。** 可参考事实与意见分离，但它是系统展示，不列作 ACL 主会长文。原生实现依赖 PostgreSQL 及相应索引；survey 不意味着需要为本项目新增数据库。公开预训练组件与任务特定训练也须区分。

### 13. H-MEM — EACL 2026 主会

**机制。** 将具体记录逐层抽象为层级记忆，保存到下一层子记忆的位置指针，按层路由读取；论文说明更新外部记忆而不更新语言模型参数。[^s13]

**评测。** LoCoMo 五类任务，F1、BLEU-1，并分析效率。五种题型不是五个独立 benchmark。

**价值与边界。** 这是层级组织对照，**不是抽取实体关系的知识图谱**。能回答“是否需要复杂关系图，还是层级定位已经足够”，但不能提供对任意多跳关系完整性的保证。

### 14. MRAgent — ICML 2026 主会

**机制。** 用线索、标签和内容构成记忆图，标签连接相关经历；读取时反复探索、筛选和重构所需证据。重要的关系组合被推迟到查询时，而不是全部提前生成。[^s14]

**评测。** LoCoMo、LongMemEval，包含模型裁判及证据召回。**附录 D.3 将 F1 定义为基于裁判判定的 precision/recall，而非答案 token 重叠**，不能因表头写 F1 就放入本项目的同指标横向比较。

**价值与边界。** 是“组合发生在写入还是读取”的直接参考，但不属于只替换 Generator 的方法。ICML 官方目录及论文会议页脚用于确认发表归属，不沿用不同版本搜索记录中的会议信息。

## 二、相关语义记忆、图检索与建图工作

这一组主要研究文档上的问答或知识图谱构建。它们可帮助定位机制和成本，不自动证明在持续对话 memory 中有效。

### 15. Panini — ICML 2026 主会

**机制。** 将文档写成生成式语义工作区：组织实体、事件和问答对；查询拆解后按中间答案连接推理链，返回结构化证据。方法使用外部记忆实现非参数更新，而非必须微调回答模型。[^s15]

**评测。** NQ、PopQA、MuSiQue、2Wiki、HotpotQA、LV-Eval，报告 EM/F1；采用论文指定的 HippoRAG2 子集配置。另造的 Platinum 评测涉及可回答性重新标注，不等于原 benchmark。

**价值与边界。** 很接近“Generator 生成可组合记忆”的研究对象。不过反向关系遗漏、生成节点遗漏和查询拆解错误仍出现在其失败分析中。不能把其重新整理过的数据或参考标注直接搬入我们的原始任务。

### 16. CatRAG — ACL 2026 Findings

**机制。** 明确基于 HippoRAG2：加入查询实体锚点、查询相关动态边权和关键事实所在段落的增强，减少图传播偏向高连接度但无关的节点。[^s16]

**评测。** MuSiQue、2Wiki、HotpotQA、HoVer；Recall@5，前三者 QA F1、HoVer 准确率；另报告完整证据链及答案联合成功。使用约千题级论文子集，不是各 benchmark 的全部发布数据。

**价值与边界。** 是本综述最明确的 HippoRAG2 直接扩展，但主要改读取，不解决抽取内容已经失真。动态边权需要查询时模型调用。论文限制节注明原始代码无法公开，不能宣称已有可直接运行的官方完整实现。

### 17. AutoSchemaKG — ACL 2026 主会

**机制。** 从文本动态归纳结构，联合组织实体、事件和概念，不依赖固定人工关系清单；实验将构建出的图交给 HippoRAG1/2 等访问方法。[^s17]

**评测。** 建图研究及 MuSiQue、2Wiki、HotpotQA 的 EM/F1。附录 D 同时调整 HippoRAG2 过滤候选及传播设置，所以不能把整组成绩视为严格“只换生成记忆”的实验。

**价值与边界。** 是生成侧最直接的参考之一，但不支持“图更丰富必然全面提高”：同为 70B 构建器的表 11 中，两个数据集 F1 上升、2Wiki F1 下降，三个 EM 均下降。需要解释具体证据变化，而不是只强调结构复杂度。

### 18. LinearRAG — ICLR 2026 主会

**机制。** 建立实体、句子和段落之间的图，靠实体识别及语义关联连接信息，不要求生成模型抽取全部关系；先局部激活，再传播到相关段落。[^s18]

**评测。** HotpotQA、2Wiki、MuSiQue 及 Medical；主表使用答案包含匹配准确率和模型裁判准确率，不能把前者称为 EM。

**价值与边界。** 是检验“能否绕开不可靠且昂贵的关系抽取”的重要对照。保留句子减少压缩环节，但仍可能没有连到需要的证据；语义连接也不直接表达关系方向、否定或版本约束。

### 19. BrowseNet — ICLR 2026 主会

**机制。** 节点直接是带标题的原文片段，共有或同义实体建立连接；用实体识别模型等构图，查询则被拆成有依赖关系的子问题图，再寻找相应证据。[^s19]

**评测。** HotpotQA、2Wiki、MuSiQue 各千题验证集配置及合并候选语料；Recall@2/5、答案 EM 和 token F1，并检查所需图边。不是只在每题自带的少量候选中检索。

**价值与边界。** 避免把原文全部压成三元组，但查询拆解仍可能错。论文对失败题的阶段分析同时标记缺边、查询结构、语义访问和答案生成；一题可有多个错误，比例不能当互斥分区相加。

### 20. MemGraphRAG — KDD 2026

**机制。** 多个构建 agent 共享全局的结构规范、事实和原文记忆，检测跨文档重复与冲突，再生成分层索引图；另配记忆引导的访问机制。[^s20]

**评测。** HotpotQA、2Wiki、MuSiQue，以及 GraphRAG-Bench 的 Medical/Novel。主要使用字符串包含准确率、模型裁判准确率及 benchmark 的证据相关性指标，**不是 EM/F1 主表**。有替换其他方法建图模块的实验，包括 HippoRAG2。

**价值与边界。** 是“局部抽取缺少全局一致性”的直接参考。但冲突裁决是否忠实遵守输入，而非用常识挑选所谓正确事实，对反事实更新任务至关重要。它首先是文档 GraphRAG，不是已在本项目六任务上验证的统一记忆模块。

### 21. ZoomRAG — ACL 2026 Findings

**机制。** 用实体识别构建文档及片段尺度的索引图，查询从全局文档层走向局部片段层，避免完整关系抽取；以跨层随机游走选取内容。[^s21]

**评测。** 2Wiki、HotpotQA、MuSiQue，EM/F1、证据召回和时间分析；主要对照统一 top-10。原文使用的时间指标名称与描述包含回答生成，不能未经拆分就当作纯检索耗时。

**价值与边界。** 是结构化定位兼顾效率的参考，而不是生成丰富语义记忆的路线。其 top-10 成绩不能直接当作我们的 top-5 成绩；多尺度结构是否改善版本或主体错误仍需分别验证。

### 22. TopoRAG — ACL 2026 Findings

**机制。** 在近邻查找时加入返回子图的直径约束，直接选择结构上较连贯的实体集合，而不是先选独立向量近邻、再随意补邻居；也讨论共现增强和查询拆解。[^s22]

**评测。** MultiHop-RAG、HotpotQA、NarrativeQA；前两者使用论文定义的包含式 Accuracy/Recall，后者用 BLEU-1、METEOR、ROUGE-L F1。这里的 Recall 不能自动解释成 passage Recall@k。

**价值与边界。** 这是语义图上的拓扑约束检索，不只是向量数据库内部的近邻索引图。结构连通不等于事实正确，也可能限制需要较远证据的查询；主要贡献位于访问阶段。

### 23. ATOM — EACL 2026 Findings

**机制。** 先把原文分为自足的原子事实，再抽取带起止有效时间的关系；区分事实何时有效与何时被观察到，并合并局部时间图。[^s23]

**评测。** 作者从 NYT 构建并人工核验的 2020-COVID-NYT 数据，包含 1,076 篇文章；报告事实和时间事实的 precision、recall、F1、重复运行一致性、实体关系消歧和构建时间。它不是现有 agent QA 数据集上的端到端成绩。

**价值与边界。** 与“记忆生成保存了多少信息”直接相关，但其金标准图不能假定其他 benchmark 也具备。表 2 中原子化提高覆盖同时降低部分 precision，说明拆细仍有取舍，而不是可以无代价修复生成遗漏。

## 三、必须纳入讨论的实证论文

### 24. Does Memory Need Graphs? — ACL 2026 主会

**研究。** 将长期对话记忆拆为检索键、返回内容、查询、索引、访问和回答，分阶段比较图与非图设置。它是一篇实验与系统分析论文，不计作第 24 个新 baseline 算法。[^s24]

**评测。** LongMemEval-s/-m、HaluMem-Medium；包括检索 Recall@5/10、记忆相关指标及模型裁判问答。不能把它的问答准确率直接与本项目 F1 比。

**结论与边界。** 记忆单元、原文是否返回、实体名字还是实体描述作索引都会影响效果，图只在某些配置有优势。对本项目最重要的是：**有信息、可定位、能返回和能答对不是同一件事。** 这也要求我们不能仅凭错题就宣布“本质是建图失败”。

## 直接改进 HippoRAG2 的到底有多少

在这 24 篇中，能够明确确认的关系如下；这是本清单内的分类，不是对 2026 全年所有发表工作的上界判断。

| 关系 | 论文 | 应如何描述 |
|---|---|---|
| 明确在 HippoRAG2 架构上提出改动 | CatRAG | 直接扩展，主要调整查询时的图遍历 |
| 把自己的图交给 HippoRAG2 | AutoSchemaKG | 替换建图方案，但实验还调整访问设置 |
| 包含给 HippoRAG2 替换构建模块的实验 | MemGraphRAG | 构建模块的可迁移性对照；完整系统另有自己的读取机制 |
| 独立结构或系统，与 HippoRAG2 比较或借鉴其机制 | REMem、LinearRAG、BrowseNet 等 | 相关工作或竞争方案，不应统称直接扩展 |

因此可确定说：**本次找到 1 篇明确的直接扩展，以及 2 篇明确测试与 HippoRAG2 组合的建图工作。** 不能说“只有一篇相关工作，所以空间很大”；独立竞争方案也会限制一个新方法的贡献空间。[^s16][^s17][^s20]

## 指标、数据范围与成本：哪些结论能用于本项目

### 指标同名不代表同一件事

本项目坚持采用既定确定性评估，不因为相关论文常用模型裁判就更换指标。词面 F1 衡量答案 token 重叠；实体或时间事实 F1 衡量结构记录是否匹配；拒答 F1 衡量识别不可回答问题；基于模型裁判的 F1 又是另一种统计。它们不能合成一列。

| 文献设置 | 能从论文读取什么证据 | 不应做的转换 |
|---|---|---|
| GAM、CompassMem、H-MEM、AnchorMem 的词面指标 | 相应对话任务上的 F1/BLEU 结果 | 不同题型范围和回答模型的绝对分数直接排名 |
| Panini、BrowseNet、ZoomRAG 的 QA 结果 | 指定文档语料和题目范围下的 EM/F1 | 千题子集成绩冒充全数据成绩 |
| LinearRAG、MemGraphRAG 的字符串准确率 | 生成答案是否包含 gold 表达 | 包含匹配写成 Exact Match |
| MAGMA、HyperMem 等裁判分 | 其裁判协议下的语义答案判断 | 裁判 SOTA 写成 token F1 SOTA |
| MRAgent 的 F1 | 按其附录定义理解，不能默认词面评分 | 只看列名就与 token F1 合并 |
| ATOM 的事实 F1 | 已有人为核验事实图时的抽取质量 | 在无 gold memory 的任务上直接套用或伪造标注 |

对于 retrieval，也必须明确 k 数的是会话、发言、事实、段落还是事件。命中一个事实后展开原文可能增加大量内容；相同 top-5 不必然意味着相同最终输入。论文采用自己的默认流程可以忠实复现，但不能静默裁切后继续称完整官方方法，也不能为了跑分扩大本项目已批准的预算。

### 数据集名称不够，候选语料与题目范围同样重要

多篇图检索论文采用每数据集千题左右的既有研究配置，并将候选原文汇集为语料；它们不等同官方完整验证集，也不等同每题仅检索自带文档。Panini 还另设重新标注的 Platinum 评测。这里记录这些差异是为了正确解释文献，**不采用它们来缩小本项目已经约定的六项完整任务**。后续若接入候选，先核对其官方输入与任务语义；不能只在 LoCoMo 做出增益就宣称通用于全部任务。

### 时间和 token 必须按发生阶段解释

构建成本是读取原始材料、生成和组织记忆的成本；更新或整理是额外的写入成本；查询时的图探索、问题拆解、证据筛选是访问成本；最后一次回答是 Evaluation Backbone 成本。对于不调用模型的图访问，其 token 可以为零，但 elapsed time 并不为零。对于多步 agentic 访问，把全部模型调用都藏在最终 QA 一栏，会掩盖方法本身的开销。

因此，评价 memory 的资源收益需要保留“构建／更新、访问、最终回答”边界。统一最终答案模型并不会消除不同方法的查询时调用成本；另一方面，也不需要把完全相同的最终答案调用误写成生成记忆算法本身的成本。论文展示的 token 节省有的仅指回答上下文，不能直接解释为全生命周期节省。

## 对现有错题的启发：文献覆盖了什么，尚未证明什么

下表引用下方既有逐题审核中的现象，只作机制对应，不报告全量失分比例，也不将尚未确认的缺失计为丢失。现有原文、ground truth、生成产物和实际返回记录继续分开保存。

| 已观察现象 | 已有相关机制 | 为什么还不能说被解决 |
|---|---|---|
| 第二跳事实已存，但实际返回只围绕第一实体 | BrowseNet 的问题依赖结构；CatRAG 的动态遍历；Panini 的链式访问 | 都依赖定位到正确中间实体，且可能新增查询计算 |
| 日期被编成过度精确值，交流日被当成事件日 | REMem、APEX-MEM、ATOM 的事件时间表达 | 时间字段仍由模型解析，错误值也能被结构化保存 |
| 整理删除必要事实，旧事实与新事实混杂 | APEX-MEM 的历史保留；GAM 的分阶段组织 | 留下历史不等于能按该任务规定选对版本 |
| 生成器用现实常识否定题目给定关系 | MemGraphRAG 的冲突处理；Hindsight 的事实与意见区分 | 冲突裁决自身可能再引入常识偏差，不保证服从反事实输入 |
| 标题、主体或关系方向在压缩中失真 | AnchorMem、LiCoMemory 的原文连接；LinearRAG 的原文索引 | 原文可恢复与生成表示忠实是两个不同结论 |
| 所需事实已返回，但答案模型选错对象或答反 | REMem、BrowseNet 的阶段错误分析；统一记忆实证研究 | 不能把最后阶段的错误自动归因于生成遗漏 |

**判断：** 图结构值得研究，但“图不够复杂”不是现有证据支持的共同根因。更加具体的候选问题是：关系所属的主体、事件、时间和更新条件是否在压缩后仍然可区分，以及这些区分是否真的影响后续定位。该判断与多篇已有工作重叠，尚不是新算法或已证实的新颖性。

## 下一步应优先读什么，而不是立即再堆 baseline

如果研究重点是 **training-free 的 memory generation / organization**，优先精读三组已有方案：AutoSchemaKG 与 MemGraphRAG 对应跨记录建图；GAM 与 REMem 对应事件和语境组织；LinearRAG、BrowseNet 与 ZoomRAG 提供不依赖完整关系生成的反面参照。ATOM 可以参考如何检验“原子化是否真的提高保留”，但不能因此新造本项目的 gold graph 或修改 benchmark。

如果研究重点改成 **已有记忆的访问机制**，CatRAG、MRAgent、CompassMem 更直接。两条路线都能研究，但论文必须明确自己的贡献在哪一阶段。不能实际只改变查询循环，却把故事写成更好的生成记忆；也不能把多个 baseline 的独立产物合并后的增益说成对任意单一方法即插即用。

进入实验前需要核对官方代码是否可获得、是否训练或使用监督标注、是否必须具有对话角色和真实时间、图访问是否会改变已约定预算。任何被选为实验方法的候选仍需覆盖既定六项完整任务；官方机制无法忠实适配的情况先说明，不静默添加规则。当前没有授权变更参数、构建流程、评分或数据范围。

**对此前建议的修正：** 仅凭已有少量错题就优先推荐 A-MEM，或先决定围绕 HippoRAG2 加图、时间、来源，是不充分的。这些机制已有密集工作。本综述不再以它们本身作为 novelty，而要求后续逐题证据先区分生成失真、组织丢失、访问遗漏和回答失败，再判断哪一个已有机制值得做完整任务对照。基于已见 benchmark 的选择应如实称探索性分析，不改写成独立未见测试上的泛化结论。

## 来源

以下论文均为 2026 年正式发表项。发表归属使用会议论文集或官方会议目录；方法和评测细节优先引用正文及附录。ICML 两项同时给出官方目录与论文全文，避免只用 arXiv 年份判断。脚注中的节号和表号指出上述关键限制的核验位置。

[^s01]: Shu 等，*REMem: Reasoning with Episodic Memory in Language Agent*，ICLR 2026。[正式论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b93eda3b396a7f665a26fbf53655adc-Abstract-Conference.html)；[全文，尤其评测与 §6 错误分析](https://arxiv.org/html/2602.13530v1)。正式目录标题为单数 Agent，预印本标题为 Agents。
[^s02]: *AssoMem: Scalable Memory QA with Multi-Signal Associative Retrieval*，ICLR 2026。[正式论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a921f335253add9996d5175ad30896ec-Abstract-Conference.html)；[全文 §3.2–3.3、§4](https://arxiv.org/html/2510.10397)。
[^s03]: *MemORAI: Memory Organization and Retrieval via Adaptive Graph Intelligence for LLM Conversational Agents*，ACL Findings 2026。[发表页](https://aclanthology.org/2026.findings-acl.1408/)；[论文 §3–4](https://aclanthology.org/2026.findings-acl.1408.pdf)。
[^s04]: *Anchored Facts with Associative Contexts for Building Memory in Large Language Models*，ACL Findings 2026。[发表页](https://aclanthology.org/2026.findings-acl.1736/)；[论文](https://aclanthology.org/2026.findings-acl.1736.pdf)。
[^s05]: *Memory Matters More: Event-Centric Memory as a Logic Map for Agent Searching and Reasoning*，ACL Findings 2026。[发表页](https://aclanthology.org/2026.findings-acl.1123/)；[论文 §3–5、图 3 和附录 C](https://aclanthology.org/2026.findings-acl.1123.pdf)。
[^s06]: *Synapse: Empowering LLM Agents with Episodic-Semantic Memory via Spreading Activation*，ACL Findings 2026。[发表页](https://aclanthology.org/2026.findings-acl.1108/)；[论文 §3、附录 A.3](https://aclanthology.org/2026.findings-acl.1108.pdf)。
[^s07]: *MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents*，ACL 2026 主会。[发表页](https://aclanthology.org/2026.acl-long.1709/)；[论文，特别是表 4 和附录表 9](https://aclanthology.org/2026.acl-long.1709.pdf)。
[^s08]: *GAM: Hierarchical Graph-based Agentic Memory for LLM Agents*，ACL 2026 主会。[发表页](https://aclanthology.org/2026.acl-long.1600/)；[论文 §3–4、附录算法 2–3](https://aclanthology.org/2026.acl-long.1600.pdf)。
[^s09]: *HyperMem: Hypergraph Memory for Long-Term Conversations*，ACL 2026 主会。[发表页](https://aclanthology.org/2026.acl-long.1627/)；[论文 §3–4、表 1 和表 4](https://aclanthology.org/2026.acl-long.1627.pdf)。
[^s10]: *APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI*，ACL 2026 主会。[发表页](https://aclanthology.org/2026.acl-long.749/)；[论文 §6、表 2](https://aclanthology.org/2026.acl-long.749.pdf)。
[^s11]: Huang 等，*LiCoMemory: Lightweight and Cognitive Agentic Memory for Efficient Long-Term Reasoning*，ACL Findings 2026。[发表页](https://aclanthology.org/2026.findings-acl.1835/)；[论文 §3、§4.1](https://aclanthology.org/2026.findings-acl.1835.pdf)。
[^s12]: *Hindsight: Structured Agent Memory that Retains, Recalls, and Reflects*，ACL System Demonstrations 2026。[发表页](https://aclanthology.org/2026.acl-demo.27/)；[展示论文 §3、§6 及 Limitations](https://aclanthology.org/2026.acl-demo.27.pdf)。
[^s13]: *H-MEM: Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents*，EACL 2026 主会。[发表页](https://aclanthology.org/2026.eacl-long.15/)；[论文 §3–4](https://aclanthology.org/2026.eacl-long.15.pdf)。
[^s14]: *Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents*，ICML 2026。[官方会议目录](https://icml.cc/Downloads/2026)；[会议版论文，尤其附录 D.3–D.4](https://arxiv.org/pdf/2606.06036)。
[^s15]: *Panini: Continual Learning in Token Space via Structured Memory*，ICML 2026。[官方会议目录](https://icml.cc/Downloads/2026)；[全文 §3、附录 F.2](https://arxiv.org/html/2602.15156)。
[^s16]: *Breaking the Static Graph: Context-Aware Traversal for Graph-Based RAG*，ACL Findings 2026。[发表页](https://aclanthology.org/2026.findings-acl.290/)；[论文 §4.2–4.4、表 2–4 和代码公开限制](https://aclanthology.org/2026.findings-acl.290.pdf)。
[^s17]: *AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora*，ACL 2026 主会。[发表页](https://aclanthology.org/2026.acl-long.942/)；[论文，特别是附录 D 和表 11](https://aclanthology.org/2026.acl-long.942.pdf)。
[^s18]: *LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora*，ICLR 2026。[正式论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ee1955739b91db042e659b6f782a5e79-Abstract-Conference.html)；[全文 §3–4、表 1](https://arxiv.org/html/2510.10114)。
[^s19]: *BrowseNet: Graph-Based Associative Memory for Contextual Information Retrieval*，ICLR 2026。[正式论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7314e20a73542bbfff25030d1185ce88-Abstract-Conference.html)；[论文 §3–5、表 6](https://proceedings.iclr.cc/paper_files/paper/2026/file/7314e20a73542bbfff25030d1185ce88-Paper-Conference.pdf)。
[^s20]: Wu 等，*MemGraphRAG: Memory-based Multi-Agent System for Graph Retrieval-Augmented Generation*，KDD 2026。[正式 DOI](https://doi.org/10.1145/3770855.3818074)；[会议版论文 §4–5、表 3 和附录冲突处理](https://arxiv.org/pdf/2606.00610)。
[^s21]: *ZoomRAG: Hierarchical Random-walk Zooming across Multi-scale Information Graphs for Fast and Accurate RAG*，ACL Findings 2026。[发表页](https://aclanthology.org/2026.findings-acl.1643/)；[论文 §3–5、表 1–4](https://aclanthology.org/2026.findings-acl.1643.pdf)。
[^s22]: Wu、Luo，*TopoRAG: Graph-based RAG via Topology-aware Approximate Nearest Neighbor Search*，ACL Findings 2026。[发表页](https://aclanthology.org/2026.findings-acl.1703/)；[论文 §3–4.1](https://aclanthology.org/2026.findings-acl.1703.pdf)。
[^s23]: Lairgi 等，*ATOM: AdapTive and OptiMized dynamic temporal knowledge graph construction using LLMs*，EACL Findings 2026。[发表页](https://aclanthology.org/2026.findings-eacl.49/)；[论文 §3–4、表 2](https://aclanthology.org/2026.findings-eacl.49.pdf)。
[^s24]: Hu 等，*Does Memory Need Graphs? A Unified Framework and Empirical Analysis for Long-Term Dialog Memory*，ACL 2026 主会。[发表页](https://aclanthology.org/2026.acl-long.1232/)；[论文 §3–5、附录个案分析](https://aclanthology.org/2026.acl-long.1232.pdf)。

---

## 既有逐题审核与历史研究记录

以下保留原有审核材料及历史计划，不属于上述 24 篇文献清单。旧的候选优先级与新颖性判断以上方综述为准；其中审核数量是记录当时的进度，不应当作本次文献工作新增的全量诊断结果。

## 当前实际审核进度：先看生成记忆是否保留信息

当前审核主线：**对应方法的错题 → 原文所需信息 → 该方法cached generated memory 是否保留**。只审实际失分的方法，不因为同题另一个方法出错就审核答对的方法。同一份memory跨回答模型去重，不加入答对对照。用户最新同意同步查阅已有方法，适用则试验、根据结果继续迭代；这不授权修改旧baseline结果或任意改变预算。当前新增工作仍只有审核与文献/官方代码核对，尚未启动新算法试验；下方旧计划中的干预不是已完成实验。

当前 **49 道有失分的题**导出 **312 条失分方法×题目记录**：SH-Doc 6、MH-Doc 6、FactConsolidation-SH 6、FactConsolidation-MH 6、LoCoMo 11、2Wiki 14。此前误加入的31条全模型答对的方法记录已从审核导出排除；后续不加入答对对照，原始实验结果未动。各任务按原始顺序跳过全方法均满分的题。这是审核进度，不是代表性抽样或新benchmark subset。

每条记录明确分开：

- `ground_truth`：原参考答案，不用于冒充支持证据。
- `supporting_passages` / `additional_original_passages_read`：官方标注原文，以及实际补充核对的原文；没有修改官方标注。
- `generated_memory_evidence`：真正生成的三元组、事实或事件，带原生ID、原文及必要payload。**禁止用保留的原文passage充当生成保留证据**。
- 实际返回文本、失分模型的原答案和原分数、审核说明及未确定事项。答对模型不纳入审核记录。

已审范围内，按“该记忆产物对应至少一个模型/查询设置有失分”计数；AnchorMem两个查询设置合并为一份生成产物，LightMem整理前后分别列出。F1部分分也包括在内，不能将“失分”直接等同事实错误。

| 生成产物 | 已审失分题 | 生成信息确认完整保留 | 尚未确定完整性 |
|---|---:|---:|---:|
| HippoRAG2 | 33 | 19 | 14 |
| Mem0 | 35 | 20 | 15 |
| LightMem整理前 | 43 | 21 | 22 |
| LightMem整理后 | 45 | 21 | 24 |
| AnchorMem（共用生成产物） | 41 | 22 | 19 |

合计197条失分题目×生成产物记录中，103条已找到所需信息，94条尚未确定完整性。它们不是197道独立题，也不是全量比例；LightMem整理前后分别计数。当前证据说明很多失分不能用“所需事实根本没有生成”解释，但尚不能据此给出summarizer与composition的因果比例：信息已保存但未返回、信息已返回但答错、词面评分失分、保存同时带有冲突都需要分开。

这些是**初步逐条阅读结果，非独立人工复核标签**。BM25/Dense无Generator，单列为不适用，不计入生成保留率。当前未确定项中既有目标关系明确失真的条目，也有未找到完整时间/版本绑定的情况；尚不能将它们整体算作确认丢失。因此不报告全量kept/lost比例，也不把“确认丢失计数尚无”说成“没有丢失”。

已出现的实质区别：

1. **原文与ground truth冲突**：LoCoMo `conv-26-5` 的原始JSON对话为last Saturday，答案为The sunday before 25 May 2023。Mem0生成Saturday May 20符合原文，不能据官方失分判生成失败。原始答案及分数均不改。
2. **保存不等于无冲突**：AnchorMem `conv-26-0` 的事件 `event-f213d0c886d8c7e7e59d6fdb984fc4ec` 保留May 8交流时说前一天参加；另一个事件却写May 8参加。正确信息在，但错误陈述也在，不能用一个kept比例隐去冲突问题。
3. **别名定位不能当缺失判断**：MH-Doc中的Mem0实际保存了Edward Davis Wood Jr.的American身份；只搜索Ed Wood会漏掉。
4. **保留关系不一定保留版本条件**：两个FactConsolidation例子中，HippoRAG/Mem0/AnchorMem保存了目标关系，但生成层的版本约束尚未核实。原文序号仍在，不能拿原文给生成层补齐。
5. **生成前提与最终答案不同**：LoCoMo教育题的原文前提是继续教育并对counseling/mental health感兴趣，而参考答案是Psychology/counseling certification。前提保存了，不要求Generator额外写出参考答案的推断表达。
6. **第二步事实已经保存，但没有被返回**：MH-Doc的`ruler_qa2_421K_no1`问电影演员担任什么政府职位。Mem0、LightMem及AnchorMem dense的返回内容围着电影、续集、广播剧打转，而原生记忆实际保存了Shirley Temple Black担任Chief of Protocol。2Wiki的`914b452c0bdc11eba7f7acde48001122`也类似：Luis Mandoki出生于Mexico City已经保存，Mem0与LightMem的返回中却缺了这一条。
7. **把干扰对象的信息答给目标对象**：上一道2Wiki题中，LightMem返回了其他导演；4B回答的Milan属于Sergio Gobbi，2B回答的Watertown属于Charles Giblyn。SH-Doc的`ruler_qa1_197K_no2`则已经返回Norse的三国来源，但失分模型取了另一条Germanic tribes的泛化来源。前者缺目标第二步信息，后者目标信息已经在回答上下文中，不能混为一种问题。
8. **生成日期失真与词面评分失分必须分开**：LoCoMo `conv-26-6`的原文是2023年5月25日说下个月露营。Mem0三条生成记录写October 2026或October 2023，实际返回后4B/2B答October 2026；这是可指认的生成失真，但尚不等于全库已证实丢失。LightMem保留并返回了May 25与next month，9B答Next month around June 2023，仍只有0.5714 F1；不能因此判它丢了日期。
9. **电影标题被误写成人物发言**：2Wiki的Gaby: A True Story在AnchorMem中被写成Gaby learned about the film titled A True Story。导演出生地仍在，但标题及主体关系已出现明确失真。记录具体失真，不以未找到其他记录为由宣称全库丢失。
10. **完整比较信息已返回仍答反**：2Wiki `d6898f78089511ebbd75ac1f6bf848b6`中两个岛都在Russia，HippoRAG和LightMem生成层保存了对应信息，BM25、Dense、HippoRAG、LightMem实际返回也均可读到两岛的Russia关系，失分的2B仍答No。这是信息已给出的回答错误，不支持重建这两条事实。
11. **问题问婚恋状态，返回内容却在讲朋友关系**：LoCoMo `conv-26-7`所需的single parent和breakup在各生成方法中都能定位；已逐条读取的HippoRAG、Mem0、LightMem及AnchorMem dense返回没有给这些前提，而给出领养意向或relationships更真诚。答案随之变成准备当母亲、社交关系质量，说明保存了相关词不等于返回了正确属性。
12. **给定更新规则与参考答案的疑点**：FactConsolidation-MH `factconsolidation_mh_262k_no2`参考答案Oceania，但原文10793作者F. Scott Fitzgerald→16800配偶Carl Lumbly→9097国籍Colombia→7238所在洲Antarctica。已核对本地缓存的原始Conflict_Resolution.arrow及当前较大序号更新规则，而非只看转换结果。先单列疑点，不改分，不要求生成器保存与已核新事实不一致的答案链。

13. **整理确实删除过所需事实记录**：FactConsolidation-SH `no3`的AppleWorks→Boeing及旧Apple记录，在LightMem整理后的原生库中ID不再存在，`memory_changes.jsonl`第201、375行明确记录delete。FactConsolidation-MH `no3`的NBC→New York City/Paris记录亦在第254、754行删除。这比搜索没命中更强，但仍未排除其他合并记录保存等价含义；不直接宣布全库语义丢失，也未证明删除独自造成错答。
14. **生成器以现实常识反驳任务给定事实**：AnchorMem在AppleWorks题保存Boeing关系，却又生成`event-3b5c137e5884b4efb5bb307453424d79`称Boeing归属错误、历史确认Apple为唯一开发者；该事件进入dense实际返回。NBC题也有类似事件，但所读dense返回没有它，不能归因成同一条错误路径。问题是任务要求的事实更新与生成判断冲突，不是要求算法背诵参考答案。
15. **日期被过度具体化**：LoCoMo `conv-26-8`只给June9交流时的last week，Mem0生成June2精确日，另有条目把交流日June9写成演讲日。LightMem保留last week及June9元数据，不能因同题失分就判所有方法都丢了时间。
16. **词形与称呼失分仍需排除**：2Wiki `006d81bc0bde11eba7f7acde48001122`来源人物国籍是American、参考答案是America；SH-Doc `no5`来源分别称William the Conqueror和Duke William II of Normandy。不能把采用来源支持称呼的失分，与选择错误歌手或生成错误日期混在一起。

17. **生成响应遗漏一篇输入文档，并非该响应解析丢失或length截断**：MH-Doc `no4`的AnchorMem输入段包含Big Stone Gap电影及另外三篇人物文档，输出19条事实仅覆盖后三篇。已在原生响应缓存找到唯一包含Rice与Houllier的有效JSON列表，与保存的19条逐项完全一致；metadata为输入1115、输出537 token、finish_reason=stop。可定位这一响应未生成电影事实，但不能推断缺失发生在模型内部的具体注意力机制。HippoRAG/Mem0保存了电影→导演→Greenwich Village两跳；Mem0实际只返回电影及其他纽约电影，另属访问问题。没有改prompt或重跑。
18. **标注关系不一定写在提供给Generator的段落中**：2Wiki `f5e3b9ca0bdb11eba7f7acde48001122`问歌手的孩子，参考Dean Miller。发布样例的`evidences`确实标注Roger Miller→child→Dean Miller，但`supporting_facts`指向的人物第0句只有Roger Dean Miller本人生卒及职业，整个人物支持段也未写孩子。不能拿这个标注补生成输入，或者把父亲全名中的Dean Miller子串当父子关系。各方法第一跳保存不等于两跳完整，此题暂不算Generator丢弃已给出的父子事实。
19. **同一错误日期不代表同一原因**：LoCoMo `conv-26-9`的LightMem整理前后都保留June9与family meeting last week的记录，实际前五条却都围绕长期支持关系，而非聚会。HippoRAG把返回重点放到Caroline辅导青少年的另一段经历，不能把mentor词相关当成目标事件相同。这里存在保存/访问分离；LightMem完整参与者绑定仍待确认。
20. **多跳失分答案可以追到另一作品的属性**：FactConsolidation-MH `no4`的Mem0答Finnish，对应实际返回Pelit的语言；AnchorMem dense答Tamil/Sanskrit，对应另两部作品。目标著作的语言记录已存但未返回。同时，原始Arrow中旧著作Ramayana的新语言为English，较晚著作关系却为Life and Fate；题目所指著作与更新规则的解释须单列疑点，不因答案关键词存在就认定整条证据链正确。

21. **输出格式与语义遗漏可以在同一道题中同时出现**：2Wiki `265daf200bdc11eba7f7acde48001122`中，HippoRAG原始响应包含两元素列表`[Assassination took place in, Delft]`，不是合法三元组。重放仓库未修改的过滤函数，将28条变为27条，与保存结果完全一致；响应507 token、正常stop。另一来源的30条原始三元组只写Maurice继任William，未生成father关系，过滤前后不变。前者是输出格式不合要求，后者是语义遗漏；不手工补主体，不把格式修复单独包装成新的memory机制。Mem0、LightMem和AnchorMem则保存了本题父子及死亡地关系，但所读dense类返回全部围绕儿子Maurice。
22. **基础事实抽取也会主动改写给定关系**：FactConsolidation-MH `no5`原文12849写Les Moonves与Louis Aragon结婚，AnchorMem基础事实却否定它并改为输入段没有的Lisa Vinnecour。唯一匹配的缓存响应与保存事实列表完全一致，输出778 token、stop；不是后续事件层才发生的改写，也不是解析器改写。其他方法保留了该配偶关系，但仍有版本与访问问题；不能把某条关系正确直接等同整个方法更好。
23. **漏掉具体名称会削弱记忆的可定位性**：MH-Doc `no5`中LightMem记录了WINNER出道专辑的日期及YG Entertainment，却在该条省略标题2014 S/S，且发行关系不能替代乐队组建关系。实际返回混入Seventeen由Pledis组建，9B答Pledis；尚未排除全库其他等价记录，不宣称仅恢复标题就一定答对。
24. **不要用词面失分制造生成问题**：LoCoMo `conv-26-10`的四年时长在所有生成方法中均保存，且所读生成方法返回中也有。LightMem整理后2B答Four years，原F1为0.5；另一些模型答交流日June9，或取了另一段问吉他交流的August28。数字词形失分与选错时间不能混算。SH-Doc宗教题同样保存并返回Catholic orthodoxy，回答更宽泛的Christianity不等于未生成宗教关系。

可检查文件：`/oscar/scratch/zliu328/agent-memory-analysis/memory_diagnosis_20260912/retention_review/sample_reviews.jsonl`；汇总为同目录`summary.json`；逐条阅读笔记为上一级`retention_review_notes.json`。删除记录路径与行号、生成事件原文、匹配原始响应的缓存位置及元数据均在逐题记录中。`experiments/summarize_memory_retention.py`只校验原生记录引用并汇总笔记，不做语义自动分类、模型调用或新评分。

**尚未完成全量审核，goal继续保持active。** 后续沿原始题目顺序推进；已读取但未解决的时间、版本、别名问题继续保留未确定，不用猜测补齐表格。

## 错题与已有方法的对应：先验证别人能否解决（2026-09-12）

以下是根据已读错题提出的适用性判断，不是新方法的实测结果，也不是“已有论文已解决我们的错误”。当前优先核对两个有官方实现的候选，不先堆更多baseline。

| 已观察的问题 | 已有机制与候选 | 仍须验证的关键点 |
|---|---|---|
| 第二跳已存但没返回；主体关联不足 | A-MEM在创建记忆时生成描述、建立记忆链接，并更新相邻记忆的描述 | 是否真的连到所需另一实体，而不只是多返回同主题内容；增加上下文的作用要与记忆机制区分 |
| 生成改写或整理删除了必要条件 | A-MEM笔记保留原始content，更新主要作用于context/tags | 原文保留能否让答案免受错误生成描述影响；保留原文不等于生成描述本身忠实 |
| 交流日与事件日混淆；新旧事实并存 | Zep/Graphiti区分事实有效时间与记录时间，保留来源，并使被替代事实失效而非直接抹去历史 | 能否忠实解析相对日期、保留不确定性；MAB事实序号不是现实日期，不能擅自编日期映射 |
| 以常识否认任务反事实 | 上述方法只能列为候选，尚无本项目证据证明它们防止该行为 | 阅读其抽取/更新prompt并检查真实产物，不能把“有来源链接”当作忠实性保证 |

A-MEM依据为[论文](https://arxiv.org/abs/2502.12110)及[论文复现仓库](https://github.com/WujiangXu/A-mem)，不是另一个同名SDK。Zep依据为[论文](https://arxiv.org/abs/2501.13956)及[Graphiti官方仓库](https://github.com/getzep/graphiti)。这些机制已有公开工作支持，因此“给事实加时间、来源和链接”本身不能宣称新颖。

**先试A-MEM的理由与边界：** 它同时涉及创建、链接和更新，不只是训练或替换retriever；论文复现代码不要求新增图数据库。官方LoCoMo输入逐条保留speaker和会话时间，与我们已核对的日期/主体问题直接相关。当前未安装、未接入、未提交试验，不能报告任何增益。

- 核对到的官方默认：`test_advanced.py`默认检索种子数10；`memory_layer.py`建立关联时取5个邻近记忆，`find_related_memories_raw`还返回种子的关联记忆。因此“官方k=10”不是最终10条，“改成k=5”也不是我们的最终top5；不静默裁掉链接展开后称官方算法。见[官方实现](https://github.com/WujiangXu/A-mem/blob/main/memory_layer.py)与[实验入口](https://github.com/WujiangXu/A-mem/blob/main/test_advanced.py)。
- 官方OpenAI调用路径设置输出上限1000；不同后端代码并不完全相同。接入本地Generator时须明确选定路径，保留其生成设置，统计输入/输出token及时间，不通过放宽上限处理失败。统一Generator及三个Evaluation Backbone属于现有约定，不能因此声称复现原论文绝对分数。
- 官方实验入口对LoCoMo第5类构造候选答案时会接收参考答案；不将该入口接到我们的开放式QA。只接memory生成和官方访问机制，继续使用本项目既定任务范围、原问题、QA入口与确定性评分，不把ground truth交给Generator或答案模型。
- 优先适配有真实会话时间的完整LoCoMo任务，不挑错题组成新的算法成绩子集。其他五任务须先核对文档入口与时间语义；不发明事实序号到日期的映射。统一backbone评估与原论文不同，表格明确标注适配。
- Graphiti另需原生图存储，且当前SDK不等于Zep论文中的完整系统；不为快速跑分重写其存储或时序规则。Zep论文使用模型裁判的实验分数不能直接与我们的F1/EM比较。

**迭代判断：** 对候选同样检查原文、生成产物和实际返回，保留改善、未变和退步。若只提高答案分、未修复预期的事实/关联问题，不能宣称机制得到验证；若失败，则定位发生在创建、更新还是访问，再选择已有机制的官方实现，不临时发明答案导向规则。所有基于已见benchmark错题作出的选择都如实标为探索性分析，不声称独立测试集验证或已经达到SOTA。

## 历史执行计划（仅供参考，2026-09-12）

本节是此前提出的计划，不是已完成实验，也不覆盖上方最新授权与实际进度。下文旧个案及官方证据诊断保留为已有材料，不能替代逐题记录。

### 1. 先回答什么，而不是先设计什么

分开回答三个问题：

1. 哪些问题所需的信息，在记忆生成或整理时被改错、遗漏或失去必要上下文？
2. 哪些问题的信息已经正确保存，只是没有被返回或组合？
3. 已确认的记忆问题被局部修正后，哪些原本的错误实际改善，哪些没有改善？

不预设建图失败，不预设换更强 Generator，也不以发明新公式、新标签分数或刷榜作为这阶段目标。最后才决定值得改生成、整理、组织还是访问机制。发现事实失真不等于该失真造成当前错题；一次补证据后答对也不等于证明生成是唯一原因。

### 2. 范围、顺序和不可修改项

- 六个既定任务全部保留；八种 baseline 设置全部覆盖；三个 Evaluation Backbone 全部保留。按原始题目顺序，以任务为单位轮流推进，不按期待的结论挑题。
- 仅检查对应方法实际失分的题目，不保留答对对照；同一份生成memory在多个回答模型下失分只审核一次。首轮每任务前若干题只是审查进度，不建立新 benchmark subset，不用它估计总体比例。
- 一个方法的同一份记忆不因三个回答模型重复计算三次生成问题。生成及整理的记录按实际产物区分；LightMem 整理前后、AnchorMem 两种查询设置明确标记共享关系。
- 原有数据转换、划分、chunk、预算、prompt、seed 42、thinking 设置、检索数量、评分器及结果均不修改。不混用不同方法的记忆。诊断只读原有产物，新增内容在独立 JSON/JSONL 中。
- 先复核现有评分入口与当前采用的官方版本：MemoryAgentBench substring EM、LoCoMo 官方分类评分、2Wiki 答案 F1。分数不满分不自动等于事实错误；答案合理但词面未匹配、问题歧义、证据标注不足另行说明，不悄悄改分。[MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench)、[LoCoMo](https://github.com/snap-research/locomo)。

### 3. 每一道题怎样审，而不是只统计答案是否出现

每题依次记录：

| 检查环节 | 必须留下的内容 | 不允许的推断 |
|---|---|---|
| 题目与评分 | 题号、原问题、参考答案、任务规则、各模型原答案及原指标 | F1 不满分就判推理错 |
| 原始证据 | 支持结论的原文位置；多跳的每一步；主体、关系、时间、否定及版本条件 | 仅找到答案字符串就说证据充分 |
| 生成输入 | 能取得的实际输入及压缩/分段中间产物；缺失的环节明确写出 | 原文有就假定 Generator 实际看到了 |
| 原生记忆 | 逐条对应的原生记录 ID 和原文，必要时包括原文备份、事实、图关联、metadata | 只检查检索 top5 就说整个 memory 没存 |
| 整理变化 | 有 before/after 时逐条对照，区分初始错误与后续删除、合并、改写 | 整理后有错就归因于整理 |
| 实际返回 | QA 实际可见的完整上下文，以及哪些条件没有被传过去 | 数据库有时间戳就说 QA 看到了时间戳 |
| 判读 | 直接观察、待证假设、反例、尚缺证据、是否适合干预 | 用方法间成绩差直接证明某阶段的因果作用 |

使用官方支持证据作为定位起点，而不是假定标注一定完整。无官方逐条证据的任务，按原始输入核对并明确说明定位方式；不建立一个自称官方的新证据集。FactConsolidation 以其事实序号更新规则为准，不用现实世界常识替换反事实。

关键词、实体名、来源 ID 可以加速找记录，但不能自动决定语义标签。找到忠实的完整证据可以证明“已保存”；关键词搜索失败不能证明“已丢失”。没有完整来源对应或足够的原生记忆审查时，缺失结论必须保持未确定。

审查笔记是可核对的分析，不是新的模型裁判分数。不调用另一个 LLM 给全量问答打语义分或让批量自动标签直接成为论文真值。需要对外作为正式语义标注统计使用的部分，保留证据供研究者复核，明确标注来源和未复核范围。

### 4. 原因怎样判：允许共存，不强行单选

以下是工作中的描述，不是新 benchmark 指标。

- **生成/压缩问题**：原始证据与对应生成记录有具体矛盾，或在可完整追踪的输出中确实遗漏所需信息。若拿不到压缩后的实际输入，只能定位到“生成流程”，不能再细分为压缩器还是抽取器。
- **整理问题**：所需信息初始存在，在已保存的整理前后记录中被删改。需要证明与当前问题有关；非相关事实删改不算这道题的原因。
- **信息存在但组织/访问不足**：原生记忆有完整、忠实的所需事实，实际上下文没有覆盖；再检查实体关联、图路径、候选过滤和排序，不能把它们统称建图失败。
- **传递/呈现问题**：必要条件在 metadata 或源记录中，却没有进入回答输入；与“生成根本没保存”分开。
- **证据返回后的回答问题**：实际上下文已经支持回答，但输出错误或不符合评分形式。保留对问题、标注、词面评分和采样的解释，不直接称模型太弱。
- **未确定**：缺中间记录、别名对应不清、官方证据不足，或者多个解释无法区分。必须保留，不用排除这些题来提高归因完成率。

同题可有多个问题。统计列出重叠，不能把各类百分比相加当作 100%。至少分别报告：已审数量、未审数量、未确定数量、有直接证据的问题数量，以及有对照实验支持的改善数量。每项给分母，不用个案外推全量。

### 5. 先交一份逐题诊断，再申请做干预

第一阶段只读审查，交付：

- `sample_reviews.jsonl`：每题每方法的证据和诊断；模型原始得分沿用现有记录。
- `review_summary.json`：按任务和方法汇总真实覆盖、直接观察及未确定项；尚未完成的格子不填 0 当成没有问题。
- 本文：跨方法的共性、方法特有问题、支持与反对各假设的例子。

有了上述证据后，先给用户看“拟干预清单”：具体方法、题目范围、证据、改动前后文本或操作、保持项、预期反例、调用数和 GPU 时间估算。未批准该清单，不启动新的修复性实验。

### 6. 因果诊断怎样做

共同原则：先在同一方法、同一题内部做对照；不把不同 baseline 的 memory 互换。每个实验只针对已证实存在的一个具体差异，其他操作尽量一致。干预只能支持其实际改变的因素，不能越级声称整个算法机制成立。

| 待验证问题 | 对照与干预 | 能回答什么；不能回答什么 |
|---|---|---|
| 某条生成陈述改错，影响当前回答吗？ | 对照重放原实际上下文；干预在相同位置仅修正有原文依据的那处关系/时间/主体，其他条目不动 | 检查这处内容差异是否影响回答；这是答案可见的诊断修复，不是可部署算法，也不是整条生成链的因果证明 |
| 事实已存，只是没有返回吗？ | 从**该方法自己的**记忆中取已确认的所需记录，和原返回上下文分别交给同一回答模型 | 检查该方法已有产物能否支持回答；选择用了标注且上下文变了，不能当成检索算法成绩或单一排序因素的证明 |
| 信息保存在 metadata，但 QA 没看到吗？ | 只在对应记录处呈现其已有时间/说话者/版本信息，不修事实、不借用其他方法 | 检查呈现差异；新增文本长度是变化的一部分，不能宣称纯语义、等 token 效果 |
| 离线整理删改是否造成退步？ | 根据真实 before/after 记录，在相同其他上下文下替换相关条目；确认后才考虑在独立索引中做完整回放 | 局部对照测试文本影响；索引回放才包含排序变化，二者不能混作一个结论 |
| HippoRAG 是实体链接、图结构还是遍历问题？ | 先只读跟踪正确实体/关系是否存在、是否成为候选、在哪一步被排除。找到具体步骤后，再申请只改变该步骤的候选或关联 | 没有原生中间记录就先报告缺证，不直接换图算法；强制正确候选属于诊断，不是部署方案 |
| 预压缩还是初次抽取导致问题？ | 有保存中间输入则直接对照；没有则先申请原设置的隔离回放。之后若需开关消融，再单独批准 | 新回放不是历史中间记录；不能用一次新输出假装精确恢复旧运行原因 |

“只给官方证据”的对照已有公开先例，例如 LongMemEval 的 oracle 输入只保留证据会话；本项目不引入它的模型裁判评分。[LongMemEval 官方说明](https://github.com/xiaowu0162/LongMemEval)。已有两任务官方证据实验只作为诊断材料，不替代上述同方法局部实验，也不自动算严格上界。

**控制随机性和输入差异：**

- 复用同一 evaluator、官方评分器及该任务的 QA prompt、temperature、输出上限和 thinking 设置，不为了对照好看而设更低 temperature 或更长输出。
- 原始结果保留；对照臂与干预臂均在同一个诊断入口重放。不拿新干预直接与旧运行比较后宣布因果。
- 拟采用每题两臂均重置 seed 42 的配对诊断；这是随机状态分配方式的显式诊断设置，须随实验清单批准，不追称它与旧运行逐 token 等同。单 seed 的改善只报告为本次配对观察；不报告“稳定因果效应”。额外 seed 先申请。
- 尽量保持原有条目位置和其他文本。无法保持的条数、输入 token 数、顺序、上下文截断分别列出。保持相同预算上限不代表实际输入等长；不靠填充或新截断 heuristic 伪造等预算。
- 同样保存改善、不变和退步的题。不只重跑有利个案，不静默丢弃超时、OOM 或失败。修复一次出现改善，也不能证明原问题只有这一个原因。

### 7. 怎样回答“多少需要改生成”

最终不只给一个混合百分比，而是每个任务×生成方法分别给：

1. **已确认的题目相关生成问题**：有原文及对应记忆证据，区别于记忆中存在无关错误。
2. **事实已完整保存**：有逐步所需事实的正向证据；这排除“必须补存这些事实”，不排除组织方式仍可改善。
3. **仍未确定**：缺少中间数据或不足以证明缺失，保留到分母中。
4. **干预后实际改善**：在批准的配对实验范围内，逐模型列出原指标改善、不变和退步，明确不是全量可部署收益。

同一题的生成问题只计一次，三个 evaluator 的改善分别统计。共享记忆的两个查询配置不重复宣称两次生成缺陷。全量审查未完成前，只报告已审数量和范围，不报总体归因比例。

### 8. 什么时候才设计新算法

诊断完成到足以支持机制选择后，再交一个单独提案，而不是直接实现：

- 若题目相关的生成失真反复出现且修复对照有效，考虑忠实生成/保留必要上下文；不是直接重建全部记忆。
- 若主要是事实已存但未组合，优先研究记忆组织与访问，不以换 Generator 掩盖问题。
- 若官方证据也不能支持原评分，先解释评测边界；不把该部分包装成 memory 算法收益。
- 每个提案必须写明针对哪类已观察问题、适用哪些表示、不适用哪些方法、与相关论文的真实差异及额外成本。不先套 compiler/system 故事。
- 候选机制只读原始输入和运行时允许的信息；不读参考答案、题号对应答案或人工诊断标签。若用评测集分析并调过方法，论文明确是 test-informed exploratory development，不把同一批成绩称作独立泛化验证。新增独立验证集需要另行决定。

### 9. 权限、资源和停止条件

批准此计划后，第一阶段可做：读已有产物、逐题审查、生成分析 JSON/JSONL、更新现有研究文档、运行不调用模型的汇总检查。方法实现、数据、memory 和已发表表格保持不变。

**仍须单独确认：**新的干预作业及其清单、重新调用 Generator、重建或改写索引、修改生成/检索/QA 参数或 prompt、增加 seed/模型/dataset、训练或 soft prompt、新环境/大下载、删除旧文件、git push。训练型 latent memory 不在当前 training-free 范围内。

诊断 GPU 作业若获批，只用 gpu-he，最多同时六个单卡作业，复用环境和模型；每个实验先给调用数与实测吞吐估算，再安排任务，不以“GPU 空着”扩张实验。记录各阶段调用 token、运行时间、排队/加载时间；普通 QA 与算法额外调用分开。JSON/JSONL，不新建实验数据库、hash 或锁文件。

goal 的完成条件应是“批准范围内的逐题证据、诊断汇总及批准的配对实验全部交付”，不是“必达 SOTA/必发论文”。时间用尽交真实覆盖和未解决项；资料不足时先报告，不用猜测补齐。不能承诺一夜完成所有语义归因：最费时的是证据审查，不是六张 GPU 的吞吐。

本计划建议分两次批准：**先批准全量只读诊断；看过证据汇总和具体干预清单，再批准 GPU 因果实验。** 用户不需要预先把所有改动权限交出去。

## 目标和边界

目标是 training-free 的记忆算法，或能分别接到多种记忆方法上的模块，不把不同 baseline 的输出拼成 ensemble。先检查生成质量和信息组织，再决定是否需要图。训练 soft prompt 会改变研究范围，本轮不做。不能保证一晚上得到可发表的新算法。

不改原有 memory、baseline 参数、数据划分、答案和评分器。诊断使用官方标注证据时，单独存放，不能算作新算法成绩。所有新增调用记录时间和 token。不用 LLM-as-judge，不用关键词命中代替语义审查；关键词只定位待阅读的记录。观察、原因猜想和干预结论分开写。

## 全部 baseline：按实际运行版本分类

不能只按“embedding / graph”分组：生成什么、是否保留原文、如何更新、如何返回上下文是不同维度。

| 本实验设置 | 保存的内容及生成过程 | 组织与查询 | 必须区分的限制 |
|---|---|---|---|
| BM25 | 原文，无生成 | 词项索引，返回 5 段原文 | 没取回不能归因于生成 |
| Dense | 原文，无生成 | 向量索引，返回 5 段原文 | 相似并不保证关系或实体正确 |
| HippoRAG2 | 保留原文，抽取实体与三元组 | 实体、段落和关联图；查询关联、过滤及图传播，返回 5 段原文 | 抽取失败与已存事实未被返回不同；查询时的模型调用属于算法成本 |
| Mem0 | 官方 SDK 抽取事实；本实验是 ADD-only | 事实向量检索，返回 5 条文本 | 不是论文完整 ADD/UPDATE/DELETE 流程；存储 metadata 不等于回答模型能看到 |
| LightMem，整理前 | 预压缩、主题分段、抽取记忆 | 记忆向量检索，返回 5 条带时间文本 | 预压缩损失与抽取损失，缺中间产物时不能分开判定 |
| LightMem，整理后 | 复用上述记忆，再运行离线整理 | 相同查询入口 | 需要依据 before/after 记录判断删改影响，不能把原有错误归因于整理 |
| AnchorMem，dense top5 | 抽取事实、聚合事件，并保留原文链接 | 直接取 5 条事实或事件 | 不展开原文，不等于官方查询流程 |
| AnchorMem，官方查询 | 与上一行共用生成产物 | 5 条事实展开来源，加 5 条事件 | 最终上下文不是统一的 5 条；不能宣称等预算比较 |

方法依据：[HippoRAG2](https://arxiv.org/abs/2502.14802)、[Mem0](https://arxiv.org/abs/2504.19413)、[LightMem](https://arxiv.org/abs/2510.18866)、[AnchorMem](https://arxiv.org/abs/2604.17377)。表格描述本仓库运行设置，不把 adapter 设置说成论文默认。

## 材料覆盖不等于完成分析

2026-09-12 已对齐全部 6 个任务、8 种设置、3 个回答模型：3,386 道题、27,088 个方法×题目记录、81,264 个回答。包含原生记忆文本、来源定位、实际返回内容和逐模型答案。所有返回文本均能精确定位到导出的原生记录；这只证明记录对应，不证明证据正确或充分。

六任务为 SH-Doc_QA、MH-Doc_QA、FactConsolidation-SH、FactConsolidation-MH、LoCoMo、2WikiMultiHopQA，来自三个 benchmark 家族。前四项各 100 题，后两项为 1,986 和 1,000 题。

材料目录：`/oscar/scratch/zliu328/agent-memory-analysis/memory_diagnosis_20260912/evidence/`。`cases/` 按题列出全部方法；`native_memories/` 保存完整原生文本而非只保存检索前五。`coverage.json` 的状态是 `assembled_not_reviewed`：没有把未读记录自动填成某类原因。

## 已逐段检查的具体证据

以下是个案，不是总体失败比例；记录 ID 可在上述原生文本目录定位。

### 事实存在，查询没返回：不是生成没存

`MH-Doc_QA / ruler_qa2_421K_no0` 问 Scott Derrickson 与 Ed Wood 是否同国籍。

HippoRAG 的 `chunk-de2da7b552418c38ed8464dcd649b172/triple/38` 明确保存 Scott 是美国导演；`chunk-c7f80cb4074fa055dee7866c582891d9/triple/38` 保存 Ed Wood 是美国电影人。但实际第一段讲的是种植园主 Ed Wood Sr，其他四段也没有返回这两个所需国籍陈述，三个模型都答错。

这支持“正确事实已存，返回证据没有覆盖问题”，并提示同名人物、电影与人物混淆。尚未证明是哪条图边或哪个过滤步骤造成。Mem0、LightMem 的部分模型在没有返回 Ed Wood 导演国籍的情况下仍答对，说明答对也不能证明记忆完整。

### 冲突事实：生成损失和回答失败同时存在

`FactConsolidation-SH / factconsolidation_sh_262k_no0` 问 Adam Putnam 的大学。任务规则要求采用较大序号的事实，而不是现实世界知识：8054 是 University of Florida，14907 是 Federal University of Rio de Janeiro。

- HippoRAG 保存了新旧两个三元组，但只返回包含旧版本的段落。新事实定位：`chunk-bee99a4e800d9d1ffe095a5585a50c54/triple/8`。
- Mem0 返回两个正确的就读关系，但回答文本里没有原始版本序号。9B 选旧，4B/2B 选新，不能把偶然选新视为解决了版本关系。
- LightMem 把两个“educated at”抽成“associated with”，且返回文本中没有事实序号。这是可直接观察到的关系弱化和版本信息缺失；尚不能区分预压缩与抽取各自责任。新记录为 `8dc8063d-707f-4214-a49b-ccec819f1bbe`。
- BM25 已返回两条原文及其序号，9B/4B 仍选旧；AnchorMem 官方查询也返回两条带序号原文，三个模型仍选旧。这些不能全部归因于记忆缺失。

因此，补充版本信息是待测因素，不是已经验证的通用解法；增加检索数量也不能自动解决此例。

### 原文中的相对时间：保持事实不等于易于使用

`LoCoMo / conv-26-0` 原文是 2023 年 5 月 8 日说“昨天去了 LGBTQ support group”，答案为 5 月 7 日。BM25、Dense、HippoRAG 已返回这条原文，但模型多把说话日期当成事件日期。LightMem 保存“昨天”和对应的说话时间，不能说事件被删除；Mem0 保存明确的 May 7，三个模型均答对。

这是“保留相对时间”和“解析成绝对时间”的生成差异，但不同方法同时改变其他上下文，不能据此直接宣称绝对化导致提升。新增只给官方原文证据的诊断中，9B 仍回答 Yesterday，官方 F1 为 0；这也提醒我们区分事实错误、未消解时间与评分形式。

### 细节被弱化：需要追溯生成输入

`SH-Doc_QA / ruler_qa1_197K_no1` 问 Normans 在 Normandy 的时期，参考答案为 10th and 11th centuries。Mem0 与 AnchorMem 保存明确陈述；LightMem 返回的是“身份在十世纪上半叶形成，之后继续演化”的长句，不能直接回答原问题。整理前后该条内容相同，不能归咎于离线整理。尚未完成所有语义等价记忆的排查，也缺相应预压缩中间输入，因此不宣布整个记忆库彻底丢失这条事实。

### 多跳链：每一步都存在，不等于能够组合

`FactConsolidation-MH / factconsolidation_mh_262k_no0` 问 Christian Abbiati 所从事运动的起源国。原文更新后的链是：10906 指定 cornerback；7164 将 cornerback 关联到 field hockey；7774 指定 field hockey 起源于 Philippines。旧事实分别是 goalkeeper、American football 和 England。这里必须服从任务给定的反事实，不用现实常识纠正它们。

HippoRAG 原生记录保存这三个新关系，分别可定位到 `chunk-b68a3408a5ff02897483bff671df8b6e/triple/2`、`chunk-90f9a47ded745931cf0f29dcf0b4fa7e/triple/15`、`chunk-b42852e8bcdb765098146b074ae8e3a1/triple/15`。返回内容包含人物的新旧位置，却没有组成完整的新链，三个模型均失败。不能因此宣布需要重新生成这些已经存在的事实；下一步应检查版本约束和查询如何走到后两步。

另一个值得追查的生成现象：LightMem 把序号 1923 的“field hockey 起源于 England”写成“England has a field hockey team established in 1923”。记录 `51bd0e4c-f428-47a0-b483-e9c1724e3eca` 同时出现关系改变和把序号当年份。它不是上述失败题当前链的充分原因，但是真实的生成忠实性问题，不能用“图不够好”解释。

### 同一道多跳题中，生成忠实性和后续组合是两回事

`2WikiMultiHopQA / 83bf3b5a0bd911eba7f7acde48001122` 问 Lothair II 的母亲何时去世。所需两步是母亲 Ermengarde of Tours，以及其死亡日期 20 March 851。

Mem0 的 `ed75abec-519f-42a5-8aff-16c58acb2e4d` 和 AnchorMem 的 `chunk-0a62e6c19abb4d9175a31d5937650821/fact/2` 都忠实保存死亡日期，但各自的实际返回内容没有包含这条记忆。Dense、HippoRAG 返回两段所需原文，三个模型都答对；BM25、Mem0、AnchorMem 的部分回答把妻子的死亡年份 875 当成母亲答案。

LightMem 返回的 `4d4e6532-e57a-4e2d-b337-a0a089fc03d1` 则把 Teutberga 原文的“died 11 November 875”改成“born on 11 November 875”。这证明某条生成关系不忠实，但不是对本题全部错误的归因：本题问母亲而非妻子，仍需检查母亲记忆是否保留。这一错误在整理前后都存在。

## 已完成的诊断与下一步

作业 `6272665` 用 gpu-he 六张 GPU 分别运行两任务×三个模型，已全部完成，合计 8,958 个回答。直接提供官方支持证据，保持原有 QA prompt、生成设置、seed 42 和确定性指标。它改变了证据长度、内容与顺序，是诊断对照，不是同预算 baseline，也不是性能上界的证明。

官方证据对照的原有 F1（百分制）：

| 完整任务 | Qwen3.5-9B | Qwen3.5-4B | Qwen3.5-2B |
|---|---:|---:|---:|
| 2WikiMultiHopQA | 72.31 | 70.57 | 58.43 |
| LoCoMo，包含对抗类与标注异常 | 63.71 | 54.85 | 49.51 |

不能把剩余错误直接归因于模型能力：还需逐题检查标注、上下文和评分。LoCoMo 中有 4 题无证据 ID、9 题 ID 不能解析；保留并标记，不修补或排除出完整结果。对抗类问题不能当作正向充分证据对照。[LoCoMo 官方资料](https://github.com/snap-research/locomo)、[2Wiki 官方资料](https://github.com/Alab-NII/2wikimultihop)。

完整逐题配对保存在分析目录的 `supporting_evidence_comparison/`，包括原有八种设置的答案和分数、官方证据诊断答案和标注状态；没有自动赋予语义或因果标签。诊断输入和调用成本在 `official_supporting_evidence/` 中。现有 baseline 结果未覆盖。

接下来按全部题目继续记录实际证据和未解决疑点，并对已完成诊断作逐题配对。优先区分：关系是否在生成时弱化、时间或版本是否失去对应关系、事实是否已存但组织方式使它未被返回。需要干预时一次只改一个有原始证据支持的因素，单独保存结果；不先盲试 latent memory、soft prompt 和 graph，再事后包装故事。不将人工补入正确答案的诊断输入当作可部署算法。

目前没有完成全部题目的语义归因，也没有证据支持一个已经优于所有 baseline 的新方法。论文中的分析范围必须与实际审查和实验覆盖一致。
