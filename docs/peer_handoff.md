# Agent Memory 交接给 zli532

更新日期：2026-09-19。代码在 https://github.com/Sizchode/agent-memory ，数据不进入 Git。

## TLDR

Oscar 共享入口：`/oscar/scratch/zliu328/agent-memory-share-zli532-20260919`。

访问方式：用自己的 Oscar 账号登录，执行 `cd /oscar/scratch/zliu328/agent-memory-share-zli532-20260919`，再读 `README.md`。已按用户确认开放该目录的只读访问，不使用逐用户 ACL；其他知道路径的 Oscar 用户也可读取。原始 one-shot 与 IRCoT 目录分别为 `/oscar/scratch/zliu328/agent-memory-outputs/optimization_fact_graph_main_qa_seed42_20260919` 和 `/oscar/scratch/zliu328/agent-memory-outputs/optimization_ircot_fact_graph_seed42_20260919`。最终文件清单以 `manifest.json` 为准，整理完成标记见 `verification.json`。

| 目录 | 内容 | 优先看什么 |
|---|---|---|
| `datasets/native_inputs/` | 六任务的原加载器输出，含题目、答案、原文和现有证据标注 | 按原任务加载，不重新筛题 |
| `caches/hipporag2/` | 原始 OpenIE、embedding、HippoRAG 图与 LLM 缓存 | 最昂贵的可复用资产，不必重新抽取 |
| `caches/` 其他目录 | BM25、dense、Mem0、LightMem、AnchorMem、CatRAG 的既有产物 | 每项已有检索输出和对应记忆存储 |
| `graphs/old_complete/` | 旧完整方法及其对照、整理后的上下文 | 不要和新事实图的结果混用 |
| `graphs/fact_graph/` | 新事实图、候选索引的构图/检索对照 | `graph/` 内是六任务图文件 |
| `graphs/synonym_edge_ablation/` | 新事实图保留/删除相似实体边的对照 | 按配置与原始来源路径定位实际图 |
| `results/one_shot/` | 四 reader、九项 baseline 加新图、六任务全量结果 | 先读 `results.md`，再看 `comparison.json` |
| `results/ircot/` | 四 reader、BM25 与两种事实图、1/3/5 轮 | `results.md`、`comparison.json`、`figures/` |
| `results/old_complete_readers/`、`results/raw_readers/` | 旧完整版本与简化版本的模型迁移结果 | 属于历史对照，不是新图主结果 |
| `ablations/` | 模块、支持筛选、归一化、窗口与排序敏感性实验 | 以完成标记和逐题预测为准 |
| `diagnostics/` | gold-only、去事实筛选、固定查询回放、上下文与 PCST 对照 | 失败与不完整产物保留，不作为有效成绩 |
| `progress/` | 交接前的研究日志与历史说明 | 历史状态可能落后于这里的最新说明 |
| `manifest.json` | 逐文件原路径、大小、修改时间、链接或副本类型、排除清单 | 查找出处与复用缓存 |
| `verification.json` | 共享文件统计与读取权限检查 | 确认交接是否完整 |

## 当前进展

目前还没有确定最终算法，也不能称为通用 memory SOTA。当前方向是研究图构建对多跳支持证据检索的影响，保留其他任务作为适用范围和副作用检查，不用一种能力解释所有任务。

- 新事实图的 one-shot 主实验已完成。四个 reader 对九项 baseline 中每任务最佳成绩，均为六任务中的两项严格领先；相对 BM25 则均为五胜一负。这里是本地已列对照，不是全领域 SOTA。
- 新事实图的四 reader IRCoT 已完整汇总。1/3/5 轮均保留；图的优势随任务、reader、轮数变化，没有一致全面胜出。不要按任务选最优轮数后隐藏其余结果。
- gold-only 已完成：2Wiki 的 1000 题，四 reader，每个比较 gold-only、BM25、新事实图。它使用标注证据，只用于诊断，不是可部署方法，也不是严格理论上限。
- 去 LLM 事实筛选的实验放在 `diagnostics/recognition/exact_context/`；旧失败运行也保留。不能仅凭目录存在就称全量成功。
- 固定查询回放只验证原轨迹可重放；不代表已证明替换图后的效果。

Gold-only 同次运行的答案 F1（百分制）：

| Reader | Gold-only | BM25 | Fact graph |
|---|---:|---:|---:|
| Qwen3.5-4B | 70.68 | 42.64 | 58.14 |
| Qwen3.5-9B | 72.15 | 45.31 | 57.38 |
| Gemma-3-4B | 61.66 | 39.36 | 50.11 |
| Llama-3.1-8B | 68.14 | 38.58 | 50.53 |

原始记录在 `diagnostics/gold_evidence/main/complete.json` 和各模型的预测中。这里的 QA 重新生成，不与旧主实验分数拼接。段落数量不同，不能据此单独归因于噪声、顺序或 reader 能力。

## 数据和协议

六项固定任务共 3386 题：SH-Doc QA 100、MH-Doc QA 100、FactConsolidation-SH 100、FactConsolidation-MH 100、LoCoMo QA 1986、2WikiMultiHopQA 1000。2Wiki 是 HippoRAG 发布的子集，不是原始整个 benchmark；LoCoMo 这里只测 QA。

四项 MemoryAgentBench 用 `substring_exact_match`，LoCoMo 用原分类别 QA 规则，2Wiki 主指标为答案 token F1。检索指标和答案指标分开解释，不对不同原生指标求一个总平均分。具体数据、提示和评分看 `dataset_loader/loader.py`、`experiments/runner.py` 和 `utils/`。

现有开发反复使用了测试题，主要为 seed 42；这些是 test-as-dev 结果，不是独立测试泛化证明。下一阶段冻结设计后，需要在未参与开发的官方数据上验证。

抽取模型是 `Qwen/Qwen3-30B-A3B-Instruct-2507`，embedding 是 `Qwen/Qwen3-Embedding-0.6B`。Reader 为 Qwen3.5-4B/9B、Gemma-3-4B-it、Llama-3.1-8B-Instruct；精确 revision 和生成设置见各运行的 `reader.json`、`settings.json`、`protocol.json`。不把 2B 历史成绩并入这次四 reader 汇总。

## 如何复用

1. 先读两个最新 `results.md`，再定位 `retrieval.jsonl`、`evaluations/.../predictions.jsonl`、`summary.json`、`qa_usage.jsonl` 和完成标记。图目录则查 `graph.pickle`、OpenIE JSON、embedding parquet、事实索引与 `llm_cache`。
2. 共享资产只读。文件大多链接到既有 scratch 产物，并非可搬走的独立归档。不要删除原目录；如果需要可迁移副本，须解引用链接复制。
3. 要改图、续写缓存或运行 SQLite/Qdrant，先把所需目录复制到自己的 scratch。只复制自己使用的那一项，避免重复整个缓存。
4. 从 GitHub clone 代码，在自己的 batch/GPU 作业和环境内运行。现有实验入口保留历史绝对路径；运行前必须将输出位置改到自己的 scratch，输入用本共享目录或清单中的可读原路径。
5. 源图配置可能引用其他历史目录。共享清单保留原路径；没有对应文件时先确认依赖，不能静默重新抽取或替换方法。`CacheMissGuard` 用于显式发现缓存缺失。
6. 不共享模型权重、HF token、API key、虚拟环境、服务地址文件或运行中的数据库。需要的 gated 模型应由接收者按自己的授权获取。不要依赖交接时仍运行的个人推理/Elasticsearch 服务。
7. 输入 pickle 依赖项目中的类定义，需先 clone 代码并设置 `PYTHONPATH`。只加载可信来源的 pickle。

共享不更改原始数据和模型的授权条款。本目录开放 Oscar 只读访问，并非仅限 zli532；接收者不获得修改原实验资产的权限。不改 home 权限，不放个人凭据。

## 代码入口

- `optimization/graph_construction/`、`optimization/build_graph.py`：构图。
- `optimization/retriever/fact_incidence.py`、`optimization/retriever/hipporag.py`：新事实图和既有检索接口。
- `optimization/ircot.py`、`experiments/run_fact_ircot.py`：批量 IRCoT 与事实图接入。
- `experiments/report_fact_graph.py`：当前 one-shot/IRCoT 结果核对与汇总。
- `experiments/evaluate_gold_evidence.py`、`experiments/ablate_recognition.py`：两项诊断入口。

本次只迁移实际使用的 Python 入口到 GitHub，不加入旧 launcher；去掉了代码归档步骤对相邻 `.sbatch` 文件的依赖，未改实验方法。上游 IRCoT 代码版本和配置依赖见 `optimization/ircot.py` 及运行的 protocol；这不是开箱即用的跨账号环境包。

## 希望一起判断的问题

优先利用现成证据检查：抽取是否遗漏必要事实；事实仍在但图检索是否没有返回对应原文；原文已返回但 reader 是否回答失败。再判断哪些构图差异能解释这些失败。暂不添加数据集专属规则、不重新抽取全库，也不把事实节点或 hypergraph 名称本身当作 novelty。
