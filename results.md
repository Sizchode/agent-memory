# 实验结果与消融

更新：2026-09-16。算法、代码与贡献解释见 [novelty.md](novelty.md)。本文件只维护实验事实，不追加逐次调度日志。

## 评测协议

- 六个 task settings，来自三个 benchmark suite：SH/MH Doc QA、FactConsolidation-SH/MH 各 100 题；LoCoMo 1986 题；2Wiki 1000 题。每 reader 共 3386 题。
- MAB 使用既定四个发布配置；LoCoMo 使用 locomo10 全五类；2Wiki 使用 HippoRAG 发布子集及 6119 段语料，不是原始数据全集。
- 原 loader、分片、问题顺序及指标不变：MAB substring exact match；LoCoMo 原类别评分/F1；2Wiki answer F1。各指标单列，不求混合总分。
- seed=42；MAB temperature=0.7，Doc 最大 50、FC 最大 10 token；LoCoMo temperature=0.4、top_k=10、top_p=0.9、最大 50；2Wiki temperature=0、最大 2048。
- test set 用于开发与配置选择。结果不是独立测试泛化或统计显著性证明；同上下文也出现采样答案差异。
- Qwen 主目标为 4B、9B，统一方法与配置覆盖全部六任务。2B 旧结果保留，不继续优化。
- “6/6”指严格超过下列九项完整本地 baseline 的逐任务最高值，平局不计；不是全领域 SOTA。不同方法的证据预算不完全相同。

## 当前完整结果

默认：`statement_projection_loop_free_retained_index_rrf_window`。分数为百分制。

| Reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 91 | 63 | 68 | 11 | 57.5679 | 57.7539 | 6/6 |
| Qwen3.5-4B | 90 | 62 | 61 | 12 | 51.0135 | 56.3973 | 6/6 |

九项 baseline，单元格为 9B / 4B；严格比较使用未舍入原值：

| 方法 | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 81 / 82 | 47 / 43 | 48 / 48 | 3 / 4 | 47.4265 / 43.5583 | 45.1314 / 42.5527 |
| Dense | 53 / 50 | 43 / 37 | 19 / 22 | 3 / 0 | 49.7524 / 47.5976 | 48.8011 / 46.0157 |
| HippoRAG 2 | 88 / 89 | 59 / 54 | 36 / 39 | 4 / 2 | 50.4127 / 48.4493 | 51.8735 / 49.7646 |
| Mem0 | 85 / 86 | 50 / 51 | 60 / 47 | 4 / 5 | 53.7076 / 49.7135 | 43.5900 / 40.7367 |
| LightMem | 51 / 47 | 41 / 40 | 49 / 51 | 3 / 4 | 43.8833 / 40.6484 | 34.3407 / 32.8354 |
| LightMem offline | 51 / 47 | 40 / 38 | 44 / 48 | 3 / 4 | 43.7277 / 40.5892 | 34.3975 / 32.9132 |
| AnchorMem dense | 84 / 83 | 47 / 44 | 66 / 51 | 3 / 4 | 45.4302 / 43.6594 | 40.9879 / 39.2416 |
| AnchorMem official | 87 / 88 | 54 / 48 | 51 / 56 | 6 / 3 | 55.3736 / 50.3363 | 42.3081 / 41.9666 |
| CatRAG | 76 / 77 | 54 / 48 | 28 / 28 | 2 / 4 | 51.5556 / 48.5319 | 51.5200 / 48.0361 |

## 构图与索引对照

以下六组固定读出规则；前四组使用原完整事实候选，后两组使用保留事实候选。不是把各任务最优结果拼接为 Ours。

| 构图 / 候选 | Reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 胜出 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 原图 / 原候选 | 9B | 93 | 61 | 62 | 5 | 57.2890 | 50.0232 | 3/6 |
| 原图 / 原候选 | 4B | 87 | 54 | 62 | 3 | 51.5705 | 48.4706 | 2/6 |
| 旧 refinement / 原候选 | 9B | 94 | 58 | 63 | 9 | 57.1394 | 56.6652 | 4/6 |
| 旧 refinement / 原候选 | 4B | 91 | 59 | 60 | 9 | 50.7813 | 56.1370 | 6/6 |
| 新投影无 refinement / 原候选 | 9B | 92 | 60 | 64 | 4 | 57.2492 | 52.4801 | 4/6 |
| 新投影无 refinement / 原候选 | 4B | 86 | 56 | 62 | 2 | 51.0249 | 51.8742 | 4/6 |
| 新投影有 refinement / 原候选 | 9B | 91 | 63 | 65 | 11 | 57.0535 | 57.9152 | 5/6 |
| 新投影有 refinement / 原候选 | 4B | 90 | 62 | 61 | 9 | 50.8067 | 56.6039 | 6/6 |
| 旧 refinement / 保留候选 | 9B | 94 | 59 | 64 | 15 | 57.4063 | 56.6379 | 4/6 |
| 旧 refinement / 保留候选 | 4B | 91 | 59 | 61 | 12 | 50.4071 | 55.9104 | 6/6 |
| 新投影有 refinement / 保留候选 | 9B | 91 | 63 | 68 | 11 | 57.5679 | 57.7539 | 6/6 |
| 新投影有 refinement / 保留候选 | 4B | 90 | 62 | 61 | 12 | 51.0135 | 56.3973 | 6/6 |

新图在同索引对照的 MH、LoCoMo、2Wiki 更高，但 SH 和 9B FCMH 回退。候选筛选提高部分任务，但两 reader 的 2Wiki 小幅下降。不能写成逐任务支配或将所有采样分差归因于图。

## 已有删减证据

| 删除 / 替换 | 所属实验条件 | 9B / 4B 胜出 | 决定及局限 |
|---|---|---|---|
| 图与附录的 discourse 过滤 | 旧 refinement-only | 5/6、5/6 | 已删除；4B SH 回退，不是无损 |
| BM25/RRF | 旧 refinement-only | 5/6、4/6 | 当时保留；不是最终索引下的必要性证据 |
| 来源窗口 | 旧 refinement-only | 4/6、5/6 | 当时保留；LoCoMo 回退 5.26/3.44 点 |
| 事实附录 | 旧 refinement-only | 5/6、5/6 | 未采用；FCSH 回退 19/13 点 |
| 图侧 canonical | 旧 refinement-only | 4/6、6/6 | 只改图，reader 未删归一 |
| 图侧 latest | 旧 refinement-only | 4/6、6/6 | 只改图，不是完整删除支持政策 |
| 全部支持，仍去 synonym | 新无自环图、原候选 | 5/6、4/6 | 未采用 |
| 图侧 raw relation | 新无自环图、原候选 | 5/6、6/6 | 达到旧里程碑，但 FCMH 回退；reader 仍有 canonical |
| 删除额外来源直连 | 新无自环图、原候选 | 4/6、5/6 | 未采用；仍保留事实来源成员，不能叫无 provenance |
| 显式事实节点 / 含自环投影 | 各自有 refinement、原候选 | 均双 5/6 | 未采用；专用入口已归档删除 |

旧阶段还完成两项联合删除及多轮构图探索，完整预测和负结果未删除。不能把移除实验分支算成主算法删除 heuristic。

## 新 Reader 迁移

固定现有抽取、图、检索和原评测协议，仅更换 reader。每个模型完成四方法各 3386 题，共 **27088 次正式 QA**；另有 48 次 pilot。6458395 已逐条重评分，并核对输入来源、覆盖、生成设置、token 和汇总。比较仅含 BM25、Dense、HippoRAG 2，不等于 Qwen 的九 baseline 范围。单元格为 **Llama-3.1-8B / Gemma-3-4B**：

| 方法 | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 78 / 79 | 47 / 44 | 34 / 52 | 0 / 4 | 41.4861 / 34.5995 | 38.9383 / 39.1871 |
| Dense | 54 / 54 | 38 / 35 | 12 / 28 | 0 / 3 | 45.7518 / 39.2283 | 39.7607 / 40.3471 |
| HippoRAG 2 | 87 / 82 | 53 / 49 | 26 / 46 | 2 / 3 | 46.4510 / 39.9002 | 42.0355 / 44.7807 |
| 当前方法 | 89 / 85 | 55 / 54 | 31 / 63 | 1 / 7 | 50.0758 / 40.1672 | 43.5955 / 45.9012 |

严格胜出为 **Llama 4/6、Gemma 6/6**。Llama 的 FCSH 低于 BM25，FCMH 低于 HippoRAG 2；Gemma 的 LoCoMo 优势只有 0.2670 点，不作显著性主张。没有用新 reader 的成绩反向调图或修改 prompt、输出上限。完整模型 revision、逐题输入/输出成本及原始预测保存在 reader-transfer 产物目录。

## 2026-09-16 删除消融

四项均在最终保留候选索引条件下完成全部六任务、两个 reader，共 **27088 次 QA**；6459909 已逐条重评分并核对覆盖、实际读出、生成设置和 usage。严格胜出仍对九 baseline 的逐任务最佳值计算，平局不计。

| 删除项 | Reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 胜出 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 关系归一 | 9B | 92 | 63 | 64 | 12 | 57.2349 | 57.4395 | 5/6 |
| 关系归一 | 4B | 91 | 61 | 59 | 8 | 50.9295 | 56.0573 | 6/6 |
| 额外直连 | 9B | 92 | 59 | 63 | 16 | 57.2786 | 57.8022 | 4/6 |
| 额外直连 | 4B | 89 | 54 | 60 | 12 | 51.1831 | 56.5592 | 4/6 |
| 窗口 | 9B | 91 | 63 | 68 | 11 | 51.9118 | 57.7539 | 5/6 |
| 窗口 | 4B | 90 | 62 | 61 | 12 | 47.8834 | 56.3973 | 5/6 |
| 事实附录 | 9B | 90 | 62 | 44 | 10 | 55.9071 | 58.7431 | 5/6 |
| 事实附录 | 4B | 87 | 61 | 45 | 12 | 50.2552 | 58.3660 | 3/6 |

关系归一同时从图、候选及附录删除，不再读取 schema；其 15 组构建及 3386 题来源映射已独立核对。删额外直连保留事实来源成员、原候选与读出，只删除额外 passage-entity 直连，新增 recognition 为零。两项读出删除保持五中心、顺序、分数不变；窗口仅影响 LoCoMo。没有修改指标、数据、QA 设置或添加补救规则。

**决定：四项均不采用，默认仍为双 6/6。** 用户确认只删除不破坏 4B/9B 各六项全胜的机制；本轮主机制删除数为零，不证明其他删除或内部常数都不可简化。临时 raw 构建入口及临时测试已归档移除，不把实验代码清理计作主机制删减。完整负结果保留；不能只报告删附录提升 2Wiki、删直连提升 9B FCMH，而省略其他任务回退。

运行问题均留痕：首次 reader 长度预检误计返回字段数，正式 QA 前已按 `input_ids.shape[-1]` 重验全部输入并核对 pilot 实际 usage；首次 raw 检索因本地鉴权变量不匹配被 guard 终止，补齐 `OPENAI_API_KEY` 后重跑。旧无效预检和失败作业均保留。删直连复现了旧运行 6387717 的同一条缓存事实解析警告，沿用原解析行为；缓存/provider guard 不代表所有返回文本都能解析。

## 成本与验证

- 当前候选正式检索新增 recognition 2168 次：输入 6323456、输出 102681 token；pilot 另有 4 次，输入 11708、输出 135。
- 每个 reader 的 QA 输入 11283658 token；9B/4B 输出 36061/31682，QA 调用秒数 2256.3/2652.2。不是总 wall time，也不含继承 OpenIE/schema 成本。
- 原候选 190345 条，保留 167074；原完整向量库仍加载，不称为等比例磁盘/内存压缩。
- 独立重建：15 组图、14362 来源、167074 候选匹配；3386 题普通接口检索一致，无保存 query resets。
- 发布代码 8624980、记录 bd0c866；29 项测试通过。干净提交审计 6390831 重评分 60948 条 baseline 与 40632 条构图/索引预测，核对实际输入、生成协议、usage 与原汇总。
- 本轮清理后测试 6459831 再次通过 29 项；reader 迁移审计 6458395 与删除消融审计 6459909 合计重评分 **54176 条新增正式预测**。运行时间和全部 token 成本在各实验 `comparison.json`，pilot 单列；不将它们混入默认配置成本。
- 单轮采样未提供多 seed 显著性证据。五中心加窗口不等于五段原文或等 token 比较；历史文本仍可能进入 reader。

## 原始产物

统一根目录：`/oscar/scratch/zliu328/agent-memory-outputs/`。各完整实验目录含 `comparison.json`、`results.md`、检索、预测、usage；这些生成报告不是仓库研究文档。

| 实验 | 根目录下的相对路径 |
|---|---|
| 当前六组完整对照 | `optimization_original_retained_fact_index_seed42_20260914` |
| 获胜索引原始运行 | `optimization_retained_fact_index_seed42_20260914` |
| 原候选同 H100 四格 | `optimization_construction_h100_controls_seed42_20260914` |
| 获胜配置独立重建 | `optimization_retained_index_selection_cleanup_seed42_20260914` |
| 旧阶段单项/联合消融 | `optimization_ablation_case_serialization_seed42_20260913` |
| 原候选下 raw relation | `optimization_statement_projection_raw_relations_seed42_20260914` |
| 原候选下删直连 | `optimization_loop_free_no_direct_seed42_20260914` |
| 原候选下全部支持 | `optimization_statement_projection_all_support_seed42_20260914` |
| 本轮整链 raw relation | `optimization_retained_index_raw_relations_seed42_20260916` |
| 最终索引下读出删除 | `optimization_retained_index_readout_deletions_seed42_20260916` |
| 最终索引下删除额外直连 | `optimization_retained_index_no_direct_seed42_20260916` |
| Llama / Gemma reader 迁移 | `optimization_reader_transfer_seed42_20260916` |

原 BM25/Dense/HippoRAG2/Mem0/LightMem 来自 `final_qwen3_30b_seed42_clean_20260910`；其余 baseline 路径在 [report_results.py](optimization/report_results.py)。HyperMem 未完成且已停止，不填零、不再纳入新实验。2B 的完整历史和失败尝试在归档中保留。

旧 10 份 MD（含未提交内容）完整保存在 `/oscar/home/zliu328/agent-memory-archives/retained_index_ablation_20260916/research_documents.zip`，已通过 ZIP 校验；改动前代码为同目录 `implementation_before_ablation.tar`，已逐文件比较。更早退役实现位于 `agent-memory-archives/construction_experiments_20260914/` 和 `refinement_only_20260914/`。不删除环境、数据、模型、缓存或实验结果。
本轮临时实现保存在同目录 `implementation_with_deletion_ablations.tar`，已逐文件比较；18 份实际 Slurm 作业定义（含失败作业）保存在 `submitted_experiments.zip`，已校验并逐字节比对。仓库不增加一次性实验脚本。
