# 当前状态与约定

2026-09-12：实验及后继任务已全部取消；memory、结果、日志保留。暂停新实验，先分析生成记忆质量。多方法拼接不再作为研究主线。

目标：training-free agent memory；做单方法可插拔模块，或从原始输入开始的独立完整算法，不训练模型、不预设建图。

Generator：Qwen3-30B-A3B-Instruct-2507，负责抽取、压缩与更新。Evaluator：Qwen3.5-9B／4B／2B，负责QA。seed=42，关闭thinking；原baseline及回答预算不变。

范围：MemoryAgentBench四任务各100题、LoCoMo 1986题、HippoRAG官方2Wiki复现范围1000题。评分分别为Substring Exact Match、官方类别F1、answer F1，不用LLM judge、不改答案、不挑子集。

[结果](results.md)｜[下一步](research_plan.md)
