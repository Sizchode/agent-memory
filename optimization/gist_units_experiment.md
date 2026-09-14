# 逐 gist 索引粒度对照

日期：2026-09-13。完整索引已构建，检索正在运行，尚无本轮完整 QA 结果。

## 动机与边界

拼接 gist 对照的四个 MAB 任务已显示不能达到三个 reader 全面胜出：packed 配置的
2B 在 SH 为 78、FCMH 为 7，均低于已有 baseline 最佳值。因此继续完整跑完该轮
LoCoMo / 2Wiki，不删负结果；同时只改变索引粒度，检验逐条表示是否不同于拼接表示。
这是基于现有 test-as-development 结果作出的下一轮选择，不是独立测试集验证。

方法依据：[Dense X Retrieval 第 4.3 节](https://aclanthology.org/2024.emnlp-main.845.pdf)。
该节以来源内 sentence/proposition 与查询的最大相似度给 passage 评分，返回唯一原段落。
本轮借用这个现成聚合组件，而非复现论文全部设置；REMem gist 不自动等同于该论文的 atomic proposition。
该论文 passage recall 是答案出现率，本项目继续采用原六任务 loader、split、QA 指标，不能混合指标。
不新造权重公式、阈值、数据过滤、答案规则、模型或任务路由，不把 max-to-source 当作 novelty。

## 固定实现

- 复用已完成的 `optimization_contextual_gists_format_retry_seed42_20260913` 全部 14362 来源、123023 条 gist。
- 不再生成或修改 gist；保留原文、重复 gist 和来源身份。实际完整语料无空列表来源。
- 一条 gist 对应一个现有 Qwen3-Embedding-0.6B 向量，原编码 API、参数和 token 上限不变。
- 查询仍由 HippoRAG 的 passage 查询编码路径处理，来源分数为其 gist 相似度最大值。
- 之后沿用原 min-max 归一化、排序、top5、PPR 和 reader 设置。没有新检索轮次或 LLM 调用。
- 原 passage 向量不覆盖；逐 gist 向量与 unit-to-source 映射单独落盘。构建时不用问题或答案。
- 若来源没有任何 gist，明确报不支持并停止，不能用空向量、原文回退或删除来源来改变方法。
- 单 gist 的来源评分已与实际 HippoRAG 单向量方法直接测试相等；多 gist 使用标准 max 聚合。

模块：`retriever/gist_units.py`；来源索引与序列化仍在 `graph_construction/passage_index.py`。
构建入口复用 `index_contextual_gists.py --index-unit gist`，默认 passage 分支保持原行为。

## 完整对照

新根目录：`optimization_gist_units_seed42_20260913`。三个新候选：

| 目录内候选名 | 固定部分 | 唯一索引变化 |
|---|---|---|
| gist_hipporag | 原 HippoRAG 图、PPR、top5 原文 | 拼接单向量改为逐 gist 最大值 |
| gist_dense | HippoRAG 原 dense 接口、top5 原文 | 同上，不做图传播 |
| canonical_rrf_gist_index | 原强图、RRF60/window5、source windows 与 sentence facts | 同上 |

名称与拼接根目录一致以复用评测接口，必须连同根目录和 index_unit= gist 区分，不覆盖旧输出。
每候选仍为完整六任务、3386 题、三个 reader；共 30474 次新 QA。
原文 dense 对照继续由上一轮完整运行提供，不再重复其 QA 或择优取复跑；新轮仍执行该对照检索。
新轮全任务原 HippoRAG 与旧 packed 控制仍须逐题匹配，来源和索引全部冻结后才检索。

成本分开报告：复用生成成本计一次、新 unit embedding 成本、新检索成本、三 reader QA 成本。
三个 reader 均严格超过九组完整 baseline 的六任务最佳值才满足目标；不得逐任务拼接候选。

## 解释限制

作业链：构建 6347152；检索 6347323（gpu / norm-gpu，最多两个任务并发）；
六任务 QA 数组依次为 6347324 / 6347330 / 6347331 / 6347332 / 6347333 / 6347334；报告 6347335。
构建期间已核对落盘来源组的向量数量、维度、有限值、gist 内容和 unit-to-source 映射，均一致。
仍须全部索引完成及全任务控制通过才能运行 QA，排队不等于完成。

6347152 已正常完成（9:41）：全部 15 组、14362 来源、123023 个单位向量通过完整核对。
新 embedding 输入 2602135 token，编码调用时间之和 427.724 秒；该时间不等于作业 wall time。
无新增生成调用。检索数组 6347323 已启动，仍须原控制全部匹配后进入 QA。

这是索引粒度消融，不是新拓扑或全新数学结构。已有图方法是否有独立价值，需要同样的
逐 gist 表示下比较纯 dense 与 graph，再结合相同 readout 的原图/强图证据。
若逐 gist 也失败，就记录该失败，不把泛化或压缩收益当作预设结论。
