# Agent Memory 共享数据

给 zli532 的 Oscar 入口：

```bash
cd /oscar/scratch/zliu328/agent-memory-share-zli532-20260919
less README.md
```

| 目录 | 只包含这些内容 |
|---|---|
| `baselines/` | 九项 baseline 的既有缓存、存储与历史结果；HippoRAG 中含 OpenIE、embedding 和原图，AnchorMem 的实际存储另在 `anchormem_store/` |
| `our_graph/graph/` | 当前新事实图的六任务构图产物 |
| `our_graph/recognition_cache/` | 已有事实筛选 LLM 缓存；基础 OpenIE 和 embedding 复用 `baselines/hipporag2/` |
| `one_shot/` | 四 reader、九项 baseline 加 our graph 的六任务全量 QA 结果、逐题预测和检索证据 |
| `ircot/` | IRCoT BM25 vs IRCoT + our graph 的 1/3/5 轮结果、轨迹与运行缓存；`task_inputs/` 是复用轨迹所需的原任务输入缓存 |

先看 `one_shot/results.md` 和 `ircot/results.md`；`comparison.json` 保留机器可读分数，模型子目录保留逐题预测和检索原文。

**不包含独立消融、gold-only、PCST、其他诊断、旧完整图、pilot 或研究日志。** 原始实验目录没有删除。本共享目录只保留当前主实验需要的内容。

IRCoT 中的 our graph 固定为 `fact_graph_without_synonyms`，与 one-shot 主实验相同，不按任务或 reader 挑配置。IRCoT 汇总仅保留 BM25 和这一列，数字直接取原始报告，未重跑或改分。原运行协议和共享 LLM 缓存可能记录其他配置的历史调用，不作为本次共享的消融结果。

`our_graph/graph/` 存放基础事实图；主实验在加载时使用 `load_fact_memory(..., keep_synonym_edges=False)` 禁用继承的相似边，不要只凭 pickle 文件名判断配置。抽取模型为 Qwen3-30B-A3B-Instruct-2507，embedding 为 Qwen3-Embedding-0.6B。

四 reader：Qwen3.5-4B、Qwen3.5-9B、Gemma-3-4B-it、Llama-3.1-8B-Instruct。六任务：SH-Doc QA、MH-Doc QA、FactConsolidation-SH、FactConsolidation-MH、LoCoMo QA、2WikiMultiHopQA；每条件共 3386 题。2Wiki 使用 HippoRAG 的 1000 题子集。沿用各任务原生指标，不跨指标求平均。这些是 seed 42、test-as-dev 开发结果，不是独立测试或全面 SOTA 证明。

大文件均优先链接到现有 scratch 产物，不重复复制大缓存；因此不是可搬走的独立数据包。**如需修改图或续写 SQLite/Qdrant，请先复制所需目录到自己的 scratch。** 不要删除链接所依赖的原始实验目录。只加载可信 pickle，并先 clone 项目以提供原类定义。

本目录对 Oscar 用户只读，不限于 zli532；不改 home 权限，不共享个人凭据、模型权重或环境。基于文件系统权限检查可读，未登录 zli532 的账号代测。

代码与实验入口： https://github.com/Sizchode/agent-memory 。源码不打包进本目录。模型访问用接收者自己的授权；现有入口含历史绝对路径，运行前需把输出位置改到自己的 scratch。
