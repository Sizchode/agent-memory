# 来源事实关联图对照

日期：2026-09-13。候选为 `source_fact_incidence`。
这是由相关工作核对引出的结构实验，不是已经证明更好的算法。

## 为什么做这一组

现有来源政策将保留断言投影成实体对计数，独立来源贡献不再作为传播节点存在。
REMem 提示中间上下文单元的重要性，但其本地实现对同一个 chunk 的 gist 和 fact
采用较粗的共同来源关联。这里先检验已有事实的显式来源关联，不同时引入新的 LLM 摘要。

使用的是标准 incidence / bipartite 表示。
[HyperGraphRAG 第 4.1 节公式 5](https://arxiv.org/html/2503.21322#S4.SS1)
给出了事实单元与成员实体之间的二部关联存储；该表示本身不是我们的创新。
本实验把已有 OpenIE 的每个来源事实出现作为独立单元，连接主体、客体和来源段落。
加入来源段落是本实验的来源关联适配，不声称逐项复现 HyperGraphRAG：
我们没有采用其 n-ary 抽取、向量检索、置信度权重、阈值或 generation pipeline。
HyperGraphRAG 与已排除的 HyperMem 不是同一工作，本轮不恢复 HyperMem。

## 实际构图

- 全部原始实体与段落节点保留，索引位置不变。
- 每个原始三元组出现新增一个事实节点，记录原始三元组、来源 id 和段内位置。
  相同三元组出现在不同来源，或同一来源中出现多次，均不合并。
- 每个事实节点只连接自己的来源、主体和客体，成员边权为 1。
  主客体为同一实体时只有一条成员边，两个角色都保存在 metadata 中。
- 不进行关系归一化、状态覆盖、语义剪枝、度数校准或同义扩展。
  原文、原 OpenIE 和向量缓存不覆盖；没有新增生成或 embedding 调用。

实现：[fact_incidence.py](graph_construction/fact_incidence.py)、
[replay_fact_incidence.py](replay_fact_incidence.py)。

## 控制与限制

直接机制参照是已有 `relation_only`：它也排除同义贡献，保留全部来源事实。
旧参照位置：`optimization_context_graph_seed42_20260913/relation_only`。
原 HippoRAG 2 及现有九个完整 baseline 仍保留为完整性能参照。
不能只对比有同义边的原图，然后把所有变化都归于 incidence 表示。

沿用原 query embedding、triple recognition、实体与段落 reset、PPR 参数和 top5 原文。
新增事实节点初始 reset 为零，靠已有 PPR 传播激活；不额外注入识别事实的新分数。
不使用 RRF、来源窗口、事实附录或新的问答 prompt。
因此该组不是此前最佳 RRF/window 候选上只切换一个组件，须按对应原文参照解释。

有三个重要限制不能略去：

1. 显式关联改变路径长度、度数及重复来源的传播作用；固定 damping 不代表相同的有效扩散距离。
2. 谓词、主客体角色和来源位置在元数据中保留，但原 PPR 是无向传播，不读取谓词语义。
   该组不是关系方向推理、事件消歧或时间冲突消解。仅改谓词文字而保持成员不变，传播不会变化。
3. 共享实体仍可连接不同事实；“每个事实有自己的来源”不保证不会跨来源误关联。
   没有声称建立了准确的事件身份，更没有从 gold answers 推断关联。

因此本组回答的是：在不更换抽取与在线推理的条件下，显式来源事实关联是否比投影表示更有用。
若没有独立收益，不将“表示更完整”写成“检索必然更好”。

## 评测执行

数据加载、切分、转换、QA prompt、解码及评分沿用既有协议。
六任务全部 3386 题，每个 reader 全量执行，共 10158 次 QA。
2Wiki 的“全量”仍指已约定的 HippoRAG 1000 题发布集；test 继续明确作为开发集。
报告指标不更换，不发明综合分数；五段原文长度可以不同，成本按实际 token 记录。

每个任务依次执行：

1. 在读取保存的问题和 seeds 前，完成该任务全部语料图落盘。
2. CPU 校验原 PPR 回放与旧结果一致，再运行 incidence 图。
3. 通过普通加载接口重新检索该任务全部问题，不读取保存的 query resets。
4. 只有第三步成功，才放行三个 reader 的 QA。

脚本：[replay_fact_incidence.sbatch](replay_fact_incidence.sbatch)；
作业记录：[fact_incidence_jobs.json](https://github.com/Sizchode/agent-memory/blob/f2121c11cc6d0c36bf931674152db0ca07669573/optimization/fact_incidence_jobs.json)。
SH/MH 首次提交错误地把目录 slug 当作 CLI task，参数校验失败，未产生图；
已改用既有带空格 task 名，并修正输出目录 slug。旧失败与依赖取消记录保留。

63 项测试通过，涵盖成员关联、重复事实、自指事实、原图不变、pickle 加载、
原节点索引保持、seed 零扩展及既有图权重接口回归。
首个 FCSH 完整普通检索校验为 100/100，未据此宣称 QA 提升。

结果汇总命令：

```bash
python -m optimization.report_results \
  --output-root /oscar/scratch/zliu328/agent-memory-outputs/optimization_fact_incidence_seed42_20260913 \
  --variants source_fact_incidence
```

构图增量为零 LLM 调用，不代表原 OpenIE、embedding 和 recognition 的历史成本为零。
QA 通过 `run_graph.py` 继续保存逐题 token 和时间。

## 2026-09-13 16:32 EDT 阶段结果

六任务构图与 3386 题 CPU 回放全部完成，保留了原有 318 题 dense fallback。
此时普通接口验证完成五任务 2386 题，2Wiki 的 1000 题验证仍在运行。
四个短任务的三个 reader QA 已完成，LoCoMo 和 2Wiki 尚无完整结果。
完整 QA 依赖链不因短任务结果或下述召回结果而取消。

| reader | SH | MH | FCSH | FCMH |
| --- | ---: | ---: | ---: | ---: |
| 9B | 84 | 59 | 50 | 5 |
| 4B | 87 | 56 | 46 | 4 |
| 2B | 79 | 43 | 44 | 4 |

相对 `relation_only`，9B 的 FCSH 从 41 到 50，但 SH 从 90 到 84；
这是混合结果，尚不能把显式来源关联视作普遍改进。
以上短任务均为原 substring exact match，不是新评分。

2Wiki 的全部 1000 题使用既有 `gold_passage_recall_at_k` 计算 Recall@5：
原 HippoRAG 为 74.975%，`relation_only` 为 87.225%，本组为 86.05%。
三组题目 id 集合和数量相同。对比原图有提高，但对排除同义边的直接参照没有提高，
不能只选择前一个对比来支持 incidence 的独立优势。

当前证据更适合收窄后续问题：除了显式关联，是否需要让上下文化文本参与索引匹配？
单靠本组不能确定答案，更不能推出应当增加某个新的经验权重。

16:33 EDT 补充：2Wiki 普通接口验证以 0:0 完成，1000/1000 一致。
至此六任务 3386/3386 的普通接口验证全部通过，所有任务的 QA 验证依赖均已满足。
完整 QA 结果仍待剩余作业完成后汇总，不将这一接口检查当作性能证据。

## 完整结果

2026-09-13 16:47 EDT 核对：剩余 QA 和汇总作业 6340424 均以 0:0 完成。
六任务、三个 reader 共 10158 条预测已由原报告程序检查完整题目集合与逐题均值。

| reader | SH | MH | FCSH | FCMH | LoCoMo | 2Wiki | 严格胜出 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9B | 84 | 59 | 50 | 5 | 50.83 | 60.44 | 1/6 |
| 4B | 87 | 56 | 46 | 4 | 48.80 | 58.75 | 2/6 |
| 2B | 79 | 43 | 44 | 4 | 44.37 | 45.73 | 1/6 |

严格胜出仍指超过现有九个完整 baseline 配置在相应任务、reader 上的最高值，平局不计。
每个 reader 的输入均为 3532866 token；输出分别为 36288、31515、26199 token。
这些是 QA 实际用量，不包含已有抽取、embedding 和 recognition 成本。
完整报告位于 scratch 实验目录下的 `results.md` 与 `comparison.json`。

本组没有取代旧最佳候选，也没有支持 incidence 相对 `relation_only` 的独立召回优势。
因此保留为结构对照，不将结果解释成“显式事实节点越多越好”。下一步转向
[上下文化索引表示对照](contextual_gists_experiment.md)，先区分表示匹配与结构传播的作用。
