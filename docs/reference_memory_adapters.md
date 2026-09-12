# AnchorMem 与 StructMem 接入状态

状态说明（2026-09-12 UTC）：本页保留接入决策与当时的作业进度，不表示这些作业仍在运行。当前队列为空，已有结果保留；下一步先按 [生成产物检查计划](generator_memory_quality_review.md) 分析 memory，不启动新 baseline 或 ensemble。

## 当前 QA 决定与运行

用户已授权两种 AnchorMem 对照，并将 top-5 的具体接入交由实现方按现有设置处理。核对后必须明确：旧实验统一的是最终返回五个内容项，而不是所有方法都使用同一个 retriever；之前并不存在已实现的 AnchorMem top-5 默认值。

两版共用 `anchormem_full_seed42_20260911` 的已完成 memory，不重新生成：

| 标识 | 检索与送入 QA 的内容 | 定位 |
| --- | --- | --- |
| `anchormem_dense_top5` | 对全部已存事实与事件，按项目现有 Dense 余弦相似度规则选五条，直接送入 QA；不展开来源原文 | AnchorMem memory + 固定五条的 Dense 检索对照，不称为官方算法默认 |
| `anchormem_official` | 原样调用官方 retrieve：五个事实展开来源原文，再追加五个事件；不裁剪、不新增排序 | 官方检索 + 项目统一 QA |

没有发明事实/事件配额分配、跨类型分数加权、重排 prompt 或迭代检索。top-5 对照使用原有 embedding 的已保存向量；余弦计算向量化，稳定的同分处理沿用 Dense 的索引顺序。已与现有 Dense 的标量余弦实现做存储向量上的数值及 top-5 一致性检查。两版没有相同 token 预算的承诺；每题保存内容项数、文本长度和真实 QA 输入 token 数，以便报告差异。

两版均沿用三个 Qwen3.5 Evaluation Backbone（9B、4B、2B）、BF16、seed 42、non-thinking，以及当前 benchmark 的 QA prompt、输出上限和确定性答案指标。每个版本开始 QA 前独立重置 seed，保留同一题目顺序。官方版不使用 AnchorMem 自带的另一套 QA prompt 或 LLM judge。

对于 AnchorMem，这轮不汇报 gold 原文精确匹配的 passage Recall@5/Precision@5：事实/事件不是 gold 原文，官方展开也不是有序的最终五段。这不改变任何既有 baseline 的指标，也没有发明替代指标。检索输出保留内容类型和原生 ID，可另行检查来源；当前端到端对照以既定 QA 指标为主。

提交数组 `6249530`，六任务最多六张 `gpu-he` B200；每个任务先完成两版 retrieval，再依次用三个 evaluator 跑两版 QA，共 36 个组合。内存 32 GB、CPU 4 核；12 小时为作业上限，不是耗时预测。启动入口为 `experiments/evaluate_anchormem.sbatch`，功能实现为 `experiments/run_anchormem.py` 和 `baseline/anchormem_retrieval.py`。

新结果目录：`/oscar/scratch/zliu328/agent-memory-outputs/anchormem_qa_controls_seed42_20260911/<variant>/<task>/`。包含 `retrieval.jsonl`、`retrieval_efficiency.jsonl`、`settings.json`，以及 `evaluations/<model>/` 下的预测、汇总和 `qa_usage.jsonl`。usage 与预测按行对应；`output_characters` 是字符数，不能当成输出 token。日志在 home：`anchormem_qa_6249530_<index>.out`。提交并不等于 QA 完成。

整个实验计划为六类方法、八个设置：BM25、Dense、LightMem 无/有离线整合、AnchorMem 上述两版、HippoRAG 2、Mem0。旧五方法已有 90 个 QA 汇总，LightMem 离线版另有 18 个；不重跑、不覆盖这些结果。加上本轮 36 个，完整计划共 144 个 QA 组合。

运行更新：数组 0–3（四个 MemoryAgentBench 任务）均成功，24 个 QA 组合完成；LoCoMo、2Wiki 的数组 4、5 继续运行。已完成组合的逐题对照与研究解释见 [失败挖掘与算法方向](memory_failure_mining_and_algorithm_direction.md)。本轮只额外修复了 QA 输入 token 的计数错误：使用 `input_ids` 长度替代 tokenizer 字典的字段数，并从原 prompt 重算已完成日志；不改输入、预测、得分或生成参数。旧进程未完成日志中的错误计数不用于效率结论。

## 最新状态：全量构建与事件解析修复（2026-09-11）

用户已批准修复 AnchorMem 的事件解析错误，并运行全部保留数据。当前实验为 `anchormem_full_seed42_20260911`，使用 `gpu-he` B200：恢复数组 `6248049`（0–9，最多同时 1 张卡），文档构建数组 `6248050`（10–14，最多同时 5 张卡），合计不超过 6 张卡。StructMem 不运行。最终 retrieval/QA 的上下文预算仍需明确，本轮提交均为 memory 构建任务。

| 任务 | 完整输入规模 | 后续问题数 | 当前任务 |
| --- | ---: | ---: | --- |
| LoCoMo | 10 段对话，3,011 个官方窗口 | 1,986 | 复用生成缓存，恢复事件和索引 |
| SH-Doc QA | 412 个 benchmark 分块 | 100 | 新构建 |
| MH-Doc QA | 875 个 benchmark 分块 | 100 | 新构建 |
| FactConsolidation-SH | 537 个 benchmark 分块 | 100 | 新构建 |
| FactConsolidation-MH | 537 个 benchmark 分块 | 100 | 新构建 |
| 2WikiMultiHopQA | 6,119 段官方语料 | 1,000 | 新构建 |

“全部”是当前保留的六任务，沿用此前的数据版本与评测范围，不表示扩展到 2Wiki 原始训练集或其他已排除的数据集。EventQA 不恢复，没有通过删减语料或抽取问题子集提速。

新产物在 `/oscar/scratch/zliu328/agent-memory-outputs/anchormem_full_seed42_20260911/anchormem/<task>/build_<index>/`。日志在 home：`anchormem_build_6248049_<index>.out` 和 `anchormem_build_6248050_<index>.out`。文档任务申请 64 GB CPU 内存，供原生事实相似度矩阵和索引使用，不是增加模型 context。12 小时是调度上限，不是预计耗时。

2Wiki 的首个作业 `6248050_14` 在原文 embedding 阶段 OOM，新增生成调用为 0。原服务预留约 153 GiB 显存，embedding 峰值超过剩余空间。按允许的 OOM 修复，仅将这个重试的 vLLM 显存预留比例从 0.85 降到 0.70；embedding batch 64、32,768 context、生成输出上限、精度与算法参数不变。重试作业为 `6248226_14`，输出在 `2WikiMultiHopQA/build_oom_retry/`，日志在 home 的 `anchormem_build_6248226_14.out`。失败的 `build_0` 仅保留作错误记录，不作为完成结果。

文档任务使用官方 README 展示的 `AnchorMem.index(List[str])` 接口，直接传入项目已有 benchmark loader 输出。MemoryAgentBench 沿用 512-token 分块；2Wiki 使用官方发布语料的原段落。没有注入虚构时间、说话人或问答标签，事实与事件 prompt 均原样保留。官方主实验基于 LoCoMo，因此文档任务是统一 benchmark 上的跨数据集适配实验，不是官方已报告的文档成绩复现。

### 事件解析错误与缓存恢复

原数组 `6246117` 的十个作业均正常退出，但官方事件 prompt 要求字符串 JSON 列表，`extract_event_list` 却用圆括号正则读取内容。2,630 次事件生成最终只保存了 5 条圆括号内容，原 memory 不能直接用于正式评测。生成本身的 5,641 次调用均成功，全部 `finish_reason=stop`；输入 5,860,806 tokens，输出 967,999 tokens。原始回答已保留。

获批修复只改变解析：复用同文件事实解析已有的 `json.loads` / `ast.literal_eval`，读取字符串列表，不新增内容重写或启发式修复。缓存中 2,629 条事件回答可直接按 JSON 读取；另 1 条带反斜线换行，可由已有字面量解析读取。2,630 条回答合计包含 2,681 个事件字符串。内联检查覆盖全部回答，以及括号保留、代码围栏和非列表拒绝，未留下 test 文件。

事件 worker 的异常从静默跳过改为显式失败，避免再次将缺失的事件标为完成。补丁为 `baseline_patches/anchormem_event_json.patch`。prompt、事实分组阈值、相关事实数、模型与 token 预算不变。

恢复时将原事实、原文向量、事实向量和生成缓存复制到新目录，只重建派生的事件文件与事件向量。旧产物不覆盖、不删除。恢复客户端禁止新的生成请求，任何缓存 miss 都使作业报错。恢复时间单独记录，旧 token 成本不重复计入新调用。2,681 是缓存解析检查的数量，最终索引完整性仍须由恢复作业确认。

恢复结果：数组 `6248049` 十个任务全部成功，最终文件含 2,681 条事件，新增生成请求与 tokens 均为 0；十段对话的 index 阶段时间累计 100.49 秒（不是整个数组的端到端耗时）。逐组核对通过：原文和原事实与旧文件完全一致，每个事件存在于事件索引，所有来源事实标识存在于事实索引。SH-Doc QA 的新 memory 构建 `6248050_10` 也已完成；其他文档任务继续运行。

截至 16:34 EDT，LoCoMo 与四个 MemoryAgentBench 任务均完成。新构建的存储数量为：SH-Doc QA 7,397 个事实、11 条事件；MH-Doc QA 18,091 个事实、47 条事件；FactConsolidation-SH 18,507 个事实、1,184 条事件；FactConsolidation-MH 18,538 个事实、1,187 条事件。不同文档集合的事实相似度不同，不能为了凑更多事件而降低官方分组阈值。

需要保留的限制：MH-Doc QA 有 1 次回答的 `finish_reason=length`，其余三项的生成回答均为 `stop`。它沿用已有 32K 服务 context 和官方未显式指定输出上限的请求设置，走官方事实抽取的截断处理；没有增大预算、重试生成或更换 prompt。不能将这一次回答称为未截断。

2Wiki 重试已通过原 OOM 位置，原文 embedding 达到 60/96 批次时未再报错，之后仍需继续监测构建结果。本轮尚无 AnchorMem QA 成绩。

环境继续复用：memory 构建为 `agent-memory-envs/lightmem`；生成服务为 `/oscar/scratch/zliu328/llm_tool_ckpt/venvs/vllm/bin/vllm`，不新增环境或权重。本次配额为 home 85.95/100 GB、scratch 475.94/512 GB（软限额）。

## 初次接入记录（以下为历史状态，以上述最新状态为准）

最新运行决定：StructMem 暂不纳入，不启动。AnchorMem 已获准先生成 memory，将最终检索预算与 QA 对照留到 memory 完成之后。2026-09-11 当前作业为 `6246117`，`gpu-he` B200 数组 `0-9`，起初最多并行 4 个；另外两个 LightMem 作业完成后已提高为最多 6 个。覆盖完整 LoCoMo 十段对话的 3,011 个官方格式窗口；对应评测仍保留全部 1,986 题，但本作业不运行 QA。

运行目录：`/oscar/scratch/zliu328/agent-memory-outputs/anchormem_qwen3_30b_seed42_20260911T195832Z/anchormem/LoCoMo/build_<index>/`。每个目录包含该对话的官方 memory 文件、`settings.json` 与 `construction.jsonl`。日志为 `/oscar/home/zliu328/anchormem_build_6246117_<index>.out`。提交入口为 `experiments/build_anchormem.sbatch`，复用现有 Generator 服务参数：BF16、32,768 context、seed 42；AnchorMem 请求保留官方 `max_new_tokens=None`。六小时是 Slurm 时限，不是完成时间预测。

首次提交 `6246096` 在模型加载前因专用环境没有 `bin/vllm` 而退出，没有生成 memory。启动器现沿用既有主实验脚本的路径回退，使用 `/oscar/scratch/zliu328/llm_tool_ckpt/venvs/vllm/bin/vllm`；未安装或升级环境。失败作业日志保留，不与当前生成成本混为成功调用。

已接入两个官方 memory 构建接口，见 `baseline/reference_memory.py` 和 `experiments/build_reference_memories.py`。入口目前只支持真实时间戳的 LoCoMo，明确为 build-only；尚未接入最终 retrieval/QA 预算。AnchorMem 使用独立构建数组，没有被加入旧五方法的 Slurm 数组，不会改变正在运行的作业或旧结果。

Python 语法、CLI help、官方模块导入和补丁反向检查通过。GPU 检查作业 `6243997` 在 `gpu-he` B200 上成功完成，用现有 LightMem 环境的官方 AnchorMem Transformers embedding 后端编码了一条检查文本，得到 `(1, 1024)`。日志为 `/oscar/home/zliu328/memory_backend_check_6243997.out`。这不是 baseline QA 结果，也不是抽取或整合已经端到端通过的证据。

## 官方来源与已保留设置

AnchorMem 来自 [官方仓库](https://github.com/RayNeo-AI-2025/AnchorMem)，当前 submodule commit 为 `48bee9fbff5b32d0f4e6b2e8260e0ff5cdfcb4dd`。构建参数取其 `main.py` 的 LoCoMo 入口：事实相似度阈值 0.85，相关事实数 3，关联检索参数 5，embedding store batch 8，生成上限 `None`，不擅自改为 2K 或 16K。`None` 表示请求没有指定输出上限，不意味着服务器具有无限 context。正式启动前仍需核对本地服务的实际 context 和默认输出限制。

AnchorMem 输入复用官方 formatter 的三轮对话窗口、重叠一轮。使用 `only_referenced=False`，不根据 QA gold evidence 挑选 session；构建函数只收到 `sample_id` 和 `conversation`，不接收问题、答案或 gold evidence。对当前十组 LoCoMo，开启和关闭原筛选得到的文本列表逐一相等，共 3,011 个窗口。保留完整原始输入并不改变这批数据的实际构建内容。

StructMem 已包含在现有 [LightMem 官方仓库](https://github.com/zjunlp/LightMem/blob/main/StructMem.md) 中，不需要第二份源码或新环境。使用 `extraction_mode=event`、官方 factual/relational prompts、现有 LoCoMo 时间及说话者转换、16,000 输出上限、原压缩和主题分割设置；跨事件摘要使用 global 范围、3,600 秒窗口、15 个种子和 `process_all=True`。按照官方入口顺序，在更新前索引的副本上生成摘要，再对原索引执行 0.9 阈值的离线更新。副本均属于新方法自身的输出，不打开或修改旧 LightMem memory。

统一 Generator 为本地 Qwen3-30B-A3B-Instruct-2507、embedding 为 Qwen3-Embedding-0.6B、seed 42，继续 non-thinking。没有新增抽取 prompt、JSON 修复、失败重试策略或裁剪规则。当前只是接口准备，正式生成服务尚未启动。

## 环境与磁盘

复用 `/oscar/scratch/zliu328/agent-memory-envs/lightmem`，其 PyTorch 为 2.8.0+cu128；后续生成服务复用 `qwen_generator` 环境与 `/oscar/scratch/zliu328/hf_output` 的模型缓存。

本次配额检查：home 85.68/100 GB，scratch 475.78/512 GB，均在软限额以内。新源码放 home，实验产物放 scratch，Slurm 日志放 home。不复制权重、不删除缓存或历史结果，不创建依赖 lock 文件，也不对文件做哈希。官方依赖导入下载了缺少的 NLTK stopwords 等小型语言资源，保存在已有 scratch NLTK 目录；没有安装新模型或整套 Python 环境。

AnchorMem 原先在包导入时加载所有模型提供方，包括未使用的 GritLM/Cohere/Bedrock；现改为选择该提供方时再导入。当前 Transformers embedding 与 OpenAI-compatible 本地推理的实现不变，补丁保存在 `baseline_patches/anchormem_optional_imports.patch`。官方内部缓存实现没有另行改造。

## 必须确认后才能继续的差异

1. **检索单位与数量。** StructMem 官方示例用 60 条详细记忆和 5 条摘要；AnchorMem 将命中的事实展开为原文及事件，最终数量不是固定 top-5。直接裁成五条会改变方法；把展开内容全部塞进“五条”也不能称为与旧 top-5 等预算。需决定采用官方多层检索、报告实际内容量，还是另行批准共同预算及其分配办法。
2. **文档任务中的时间语义。** StructMem 跨事件整合依赖时间。MemoryAgentBench 文档和 2Wiki 没有 LoCoMo 式真实时间；不能为了运行而用当前时间、文档序号或人工间隔伪造事件时间。当前入口明确拒绝没有对话时间与轮次的 StructMem 输入，等待迁移方案确认。

此外，两个方法自带的 QA prompt 和 LLM judge 不自动取代现有确定性评测。后续继续使用完整既有问题和既定 Evaluation Backbone；任何必要的输入格式或预算差异先列明，不默默调整旧指标、split、问题范围或答案。

## 时间和 token 的记录口径

每个会话向 `construction.jsonl` 写入 started/completed/failed 状态、阶段时间和实际生成 usage。StructMem 的 extraction/update 客户端与 summary 客户端分开记录，汇总完整构建成本时两者都要计入。AnchorMem 同时保留原生 cost statistics 和真实 client responses 计数，不能把缓存命中或尝试数当作新成功调用。

包含模型初始化的 elapsed 与算法阶段计时分开。构建入口不复用其他方法已生成的 memory 冒充本方法结果；环境、权重、原始数据可以复用，算法产物是否可复用取决于生成协议是否一致。
