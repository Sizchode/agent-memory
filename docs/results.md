# 已有结果及边界

BM25、Dense、HippoRAG、Mem0、LightMem整理前后、AnchorMem两种设置，六任务×三个回答模型共144个baseline评测已完成。Mem0是指定ADD-only SDK；LightMem两阶段分开报告，不能直接等同论文全部流程。

9B分数（百分制）：

|任务|最高baseline|多方法拼接＋图|
|---|---:|---:|
|单跳文档|88|94|
|多跳文档|59|66|
|单跳事实更新|66|75|
|多跳事实更新|6|5|
|LoCoMo|55.37|59.40|
|2Wiki|51.87|58.04|

拼接版在9B／4B共12格中领先11格，但依赖三套memory、成本不同，不是独立算法或SOTA证明。第二轮新增评测仅完成16格后取消，部分输出不报完整成绩。

产物保留在 `/oscar/scratch/zliu328/agent-memory-outputs/`；分析在同级 `agent-memory-analysis/`。旧文档可从Git提交 `5d1f5f2` 恢复。
