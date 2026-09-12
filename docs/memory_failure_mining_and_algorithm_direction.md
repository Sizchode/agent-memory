# 从真实失败决定 memory 算法，而不是根据分数换故事

当前决定（2026-09-12 UTC）：本轮作业已全部取消，memory 和结果保留。下面的候选与运行描述属于历史记录；现阶段先检查生成产物，不预设图或 compiler 框架，不再以多方法 ensemble 作为主线。最新分析顺序见 [Generator 与 memory 质量检查计划](generator_memory_quality_review.md)。

后续更新：[三小时研究与完整实验记录](memory_research_three_hour_report.md) 包含完整 AnchorMem/证据组合结果及新对照。本页保留前一轮的观察和当时的候选边界；“贡献只发生在写入阶段”不再是当前限制，后续已允许 training-free 的读取、组合和校验模块，但必须区分各阶段贡献。

更新：2026-09-11。以下区分实际观察、尚待解释的现象和候选算法。AnchorMem 全套 QA 尚未完成，只分析已经完整完成的任务和模型组合。不修改 benchmark、标准答案、baseline 参数或已生成 memory，不使用 LLM judge。

最新行动：按用户允许 heuristic 与组件组合的新要求，已实现并提交[关系分组原型与平铺事实对照](relation_memory_experiment.md)。全部六任务、先用 9B；复用已有 HippoRAG 抽取，无新增 Generator 调用。它是探索性 memory 组织实验，不是已验证的新算法。Slurm 数组 `6250429`，最多四张 gpu-he B200，与 AnchorMem 两个运行任务并行。

17:35 EDT：原型四个文档任务对照完成，具体分数与新增错例见上述实验页。来源内主体分组第二版数组 `6250508` 已提交，六任务最多两卡；原型 LoCoMo OOM 仅降低 embedding 执行 batch，以 `6250499_4` 单独重试，未修改内容和长度预算。分组前后仍有相同检索文本但变分的案例，继续避免把采样波动全部归因于模块。

## 1. 当前结果提供的线索

以下均为 9B Evaluation Backbone，四个 MemoryAgentBench 任务各完整 100 题，使用既定 substring exact match。不同任务不混成新排名指标。

| 任务 | AnchorMem memory + Dense 五条 | AnchorMem 官方检索 | 五条对照平均 QA 输入 tokens | 官方检索平均 QA 输入 tokens |
| --- | ---: | ---: | ---: | ---: |
| SH-Doc QA | 0.84 | 0.87 | 240.59 | 3,243.49 |
| MH-Doc QA | 0.47 | 0.54 | 257.06 | 2,846.60 |
| FactConsolidation-SH | 0.66 | 0.51 | 305.97 | 3,671.51 |
| FactConsolidation-MH | 0.03 | 0.06 | 314.22 | 3,667.70 |

两版 memory 相同，区别包含检索方法、内容类型、顺序和长度，并非只改变 token 数的单变量实验。官方版在部分任务更高，不能单独证明事件图有效；五条对照在冲突任务更高，也不能单独证明长上下文导致错误。分歧是查看案例的入口。

后续完整单元核对已覆盖四任务全部三个模型，共 24 个 QA 组合。FactConsolidation-SH 的模型差异尤其不能略去：

| Evaluation Backbone | 五条对照 | 官方版 |
| --- | ---: | ---: |
| 9B | 0.66 | 0.51 |
| 4B | 0.51 | 0.56 |
| 2B | 0.54 | 0.40 |

4B 的方向相反，故不能挑 9B、2B 宣称官方展开普遍有害。SH/MH 文档问答中，官方版在三个模型上的分数均高于五条对照；FactConsolidation-MH 则两版都低，且优势方向随模型改变。LoCoMo 与 2Wiki 尚在运行，不用它们的部分题目结果下结论。

LightMem 离线前后也不支持“多整理一定更好”：9B 的 FactConsolidation-SH 从 0.49 到 0.44，LoCoMo 从 0.43883 到 0.43728。先排除回答波动，再判断具体 memory 变更。

## 2. 已追到内容的失败案例

### 有删除记录：有效关系在整理后不再可取用

`FactConsolidation-SH/factconsolidation_sh_262k_no3` 问 AppleWorks 的开发者，官方答案 Boeing。该任务含与现实常识不同的输入陈述，不能拿现实知识替换官方答案。

离线前的 LightMem 检索同时包含 Apple Inc. 和 Boeing 两种开发者陈述；离线后的五条变成其他 Apple 软件内容。变更文件明确记录了两项删除：

- `383d8c44-ddff-4f4e-b246-d6b096d420ae`：AppleWorks was developed by Boeing.
- `63b8c2ef-b45b-4eca-9173-4bd3b9755139`：AppleWorks was developed by Apple Inc.

既有 benchmark loader 的原文分块中，零起始第 84 块包含第 2,951 条 Apple Inc. 陈述，第 490 块包含第 16,767 条 Boeing 陈述。Jasper Fforde 的 English/Kannada 也分别出现在原文第 6,746、7,636 条。因此这里不是分析者制造的冲突，也不能将这些 benchmark 陈述擅自“纠正”为现实知识。

读取离线后保存的 memory 文本也没有找到 `AppleWorks` 这一名称。这直接支持“条目被删除且该题可用检索证据消失”，但尚不能判断模型为何删除，也不能保证不存在其他措辞的等价条目。来源：`lightmem_offline_20260911T184409Z/lightmem/FactConsolidation-SH/memory_changes.jsonl` 第 201、375 行及前后 retrieval。

值得研究的是如何避免合并/删除时把不同取值或适用条件一并抹掉，不是宣称所有离线整合都破坏记忆。

### 信息在场但仍答错：知识缺失不是唯一解释

同一任务的 `no4` 问 Jasper Fforde 使用什么语言，官方答案 Kannada。AnchorMem 五条对照同时给出 English 和 Kannada，9B 回答 Kannada；官方版同样含有 Kannada 对应原文，却回答 English。`no7` 的 placekicker/first baseman、`no10` 的 Russia/United States 也出现类似分歧。

有效答案相关陈述没有简单地从原文或该检索上下文消失。不同版本、模型已有常识、内容干扰和适用条件都是待检查原因；当前对照没有隔离它们，不能自动标为时间图或长上下文失败。五条版答对也不是它已正确表示版本关系的证据，它同样保留了冲突陈述。

只有进一步证明 memory 表示遗漏了输入明确给出的有效性条件，才优先改 memory。若条件已经完整呈现而 evaluator 仍选错，不优先改模型推理能力。

### 相同上下文也有不同答案：阻止错误归因

LightMem 离线前后，Jasper Fforde 这道题的五条文本和顺序完全相同，分数却改变。LoCoMo `conv-26-8` 前后也同样保留“6 月 9 日记录、上周在学校演讲”，回答却不同，不能称为整理删掉日期。

全量计数：9B 的 LightMem LoCoMo 有 332 题 F1 改变，其中 157 题前后检索文本完全相同；FactConsolidation-SH 的 15 题分数变化中，7 题检索文本完全相同。这是排查错误归因的记录，不是新指标。

既定 LoCoMo temperature 为 0.4，MemoryAgentBench 为 0.7，seed 42 在整次运行开始设置。前面题目的生成过程变化，会影响后续采样状态；相同 run seed 不保证同题使用相同随机数位置。硬件或推理差异也可能存在。因此，单次逐题变化不能直接证明 memory 操作的因果效果；也不能据此断言所有变化都只来自采样。

不擅自修改 temperature、逐题 seed、预算或重跑 baseline。确定性评分函数不等于没有采样的答案生成。

## 3. 结果好和不好，分别怎样决定算法

不根据分数临时换故事。贡献始终针对持久化 memory 的形成和整合，而不是训练 retriever 或增强 QA。

| 最终观察 | 下一步检查 | 对候选的影响 |
| --- | --- | --- |
| 两版 AnchorMem 都强 | 已解决什么，还有什么跨任务重复的失败 | 将其作为强对照；只针对未解决的整合操作，不宣称建图本身新颖 |
| 官方版更强 | 五条事实/事件省掉了必要条件，还是仅仅返回更多内容 | 若确有表示遗漏，在写入时形成带必要条件的完整陈述，不靠查询时追加原文制造收益 |
| 五条版更强 | 展开是否混入不同版本或无关关系，输入是否明确规定有效性 | 若是表示问题，保留陈述的条件和版本边界；不靠事后删上下文或重排冒充 memory 改进 |
| 两版弱，事实在构建时遗漏或绑定错误 | 原输入到 memory 的哪个变换改变了含义 | 改形成与整合操作，保留完整关系及条件 |
| 两版弱，但 memory 正确，只是没取出 | 检索接口和持久化组织分别造成什么影响 | 只把改变 memory 组织作为候选，不转成训练 retriever |
| 证据和条件完整，只有 evaluator 推理失败 | 是否跨模型重复出现 | 不优先解决，缩小 memory 方法的主张 |
| 没有稳定的 memory 机制缺陷 | 是否只剩实现、预算或采样差异 | 放弃或重审候选，不堆系统术语硬造贡献 |

效率同样不能靠故事：构建/更新、检索、QA 时间及 tokens 分开报告。不同硬件的墙钟时间不能直接当同条件加速比。LightMem 部分离线任务主要耗时在 CPU 候选扫描，省生成 tokens 未必解决它。

## 4. PL/system 思想怎样真正进入机制

### 主线：表示变换保留什么含义

编译器的启发不是给 memory 换个“中间表示”的名称，而是明确每一步优化应保留什么。CompCert 对精确定义的程序语言证明语义保持；自然语言输入和 LLM 抽取没有同等基础，不能借此声称自然语言正确性已得到形式化保证。[CompCert 官方说明](https://compcert.org/man/manual001.html)

当前更值得试的操作是：固定 Generator 在建立 memory 时识别完整陈述，保存主体、关系、对象及明确限定；后续整合共享重复结构，而不是再次自由改写全部事实。共享某个人名不能让两次任职的公司和年份串换。所有问题读取同一份持久化结果，不在看到问题后补建答案路径。

表示可以参考已有 factorized relational representations：共享重复部分，紧凑表示原有关系组合。这是成熟的数学基础，不是我们的新公式；对已结构化数据的可恢复性，也不保证最初的 LLM 抽取正确。[Olteanu 与 Schleich，Factorized Databases](https://www.cs.ox.ac.uk/dan.olteanu/papers/os-sigrec16.pdf)

可能的贡献在于：怎样把关系边界保留落实为自然语言 memory 构建/整合操作，减少误合并和重复整理，在固定下游上取得净收益。结构化、对齐与保存关系的成本必须计入；若只是现成表示前套一个 prompt，新颖性仍不足。

### 辅线：依赖与版本，不强加在线系统

构建系统区分哪些内容需要重建、按什么顺序重建，可用于思考派生记忆何时失效，避免重新处理所有旧内容。[Mokhov 等，Build Systems à la Carte](https://simon.peytonjones.org/assets/pdfs/build-systems-original.pdf)

当前主要需求仍是建立 memory，不设计复杂在线服务。只在输入和官方任务规则提供修订依据时讨论版本关系，不伪造时间或“后写入就一定正确”的普遍规则。不引入数据库、文件哈希、实验追踪系统，也不为故事强加离线阶段。

### 新颖性边界

AnchorMem 已保留原文，以事实为检索锚点并构造关联事件；“原文不可变＋事件图”不是空白。[AnchorMem](https://arxiv.org/abs/2604.17377)

SodaMem 已有带类型的事实事件、时间和修订关系；只增加这些字段不够。其公开结果还包含自己的 reader/judge 设置，不能直接与我们的确定性评测分数比较。[SodaMem](https://arxiv.org/abs/2608.08055)

不把 graph、SSA、type system 或 cache coherence 等名词直接当贡献。当前主线是：**整合应共享重复表述，而不混淆不同陈述的关系和适用条件。** 这仍是待实现、待验证的方向。

## 5. 分析产物和后续约束

`experiments/compare_memory_runs.py` 按原 case ID 比较双方都完整完成的单元，输出 `comparison.json` 与 `score_changes.jsonl`。不对运行中的前若干题算平均，不自动赋予语义原因，保存全部分数变化案例。分析文件不是新的 benchmark 子集或训练数据。

产物在 scratch：

- `agent-memory-analysis/lightmem_offline_comparison_20260911/`
- `agent-memory-analysis/anchormem_controls_comparison_20260911/`

人工解释依次查看原文、memory、检索、答案，有实际修改时再看变更记录。区分直接观察与原因假设，不把字符串不存在当成严格的语义缺失。

本轮还修正了输入 token 记录错误：tokenizer 返回字典，旧 `len(...)` 得到字段数 2。现在读取 `input_ids` 长度；`repair_qa_token_counts.py` 用原 prompt、题目顺序和 tokenizer 重算已完成日志，不调用 Generator/evaluator、不改答案或得分。在运行的旧进程完成后再补计，未修正值不用于效率结论。

已检查的题目已经用于诊断，后续不能称为完全未见测试集。不做题目特判或根据这些题反复调阈值，不自己重划数据；新增开发/评测范围先遵守官方协议并确认。新方法尚未实现。
