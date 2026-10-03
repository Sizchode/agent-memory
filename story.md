# 论文故事与贡献边界

## Recommendation-centered analysis revision (2026-09-30, current)

The user reaffirmed the central claim: **recommendation is the key to solving agent memory tasks**. Organize the analysis around why a memory task needs recommendations and how AMOR supplies the required evidence. Edge-weight comparisons and consolidation/presentation diagnostics support that explanation; they must not replace it as the main story.

The revised section in `docs/memory_analysis/section.tex` has two positive findings. First, matched memories can identify different memories needed to answer: the controlled graph comparison preserves every selected gold source belonging to recognized facts while changing additional supporting sources. Official 2Wiki types distinguish two-source comparison from two-source compositional questions, tying the gain to evidence dependencies. Second, an update must influence recommended evidence even when the corrected assertion is already supplied: the FC-MH accepted-authorship intervention removes the death-location source while preserving the correction in the answer context. The RULER case illustrates how a remembered location identifies the requested town's dated population.

The all-six native-score recommendation table is the quantitative entry point, covering three benchmark suites/four data origins and four readers. LoCoMo remains a boundary with mixed graph-recommendation effects; its neighbor-expansion gains must not be relabeled as recommendation gains. FC-SH's retained-assertion intervention similarly explains presentation. Preserve these component distinctions while explaining the overall recommendation perspective.

Every further critique round must use a newly initialized GPT-6 reviewer without conversation history or previous reviews. Repeated follow-ups to the original critic do not count as independent rounds. Fresh round `acl_recommendation_review_02` supported the narrower empirical mechanisms while rejecting an unqualified universal claim. A separate fresh reviewer, `acl_manuscript_review_03`, has now **accepted** the current section as an ACL method-analysis section with two distinguishable findings, after independently checking the source, rendered layout, primary literature, quantitative tables, and intervention traces. It read no previous reviews. Its final PDF/source synchronization and QA-status requirements are fulfilled. The review is saved in `docs/memory_analysis/reviews/round_03_review.md`; the completed main PDF, supplemental results, source/data archive, and accepted review were submitted to zhenkeliu@163.com, with the mail server accepting queue ID `6913fdVk2240225`. Receipt and transport transcript are in `docs/memory_analysis/email_receipt.json` and `email_delivery.log`. Earlier story sections below are historical evidence and must not override this narrative direction.

## Memory failure findings, replacing the module-centered analysis (2026-09-30)

The user requires findings about memory use, not observations that recommendation adds sources or that weights affect ranking. The previous proposed analysis subsections below are superseded as the main narrative; their measurements remain available as supporting controls. Do not present the following selected examples as population estimates or a proof that AMOR guarantees consistency.

1. **A factual correction can change which other memories are relevant.** In FC-MH, the synthetic knowledge pool replaces the author of Dubliners with George Eliot. HippoRAG 2 and AMOR without recommendation both retrieve the correction but not Eliot's death location. Exact replay accepts `(Dubliners, author, George Eliot)`, initializes Dubliners and George Eliot, and selects Eliot's London source second. All four full-AMOR answers are correct; all four without recommendation fail. The memory requirement is to carry the consequences of an update into context selection, not merely store or retrieve the updated assertion. AMOR's retained facts link the currently selected association to its supporting sources. No online graph-update mechanism or first-ever relation propagation is claimed.

2. **Consolidation can be undermined when superseded information returns in the answer context.** In FC-SH, Witches of East End has an old USA assertion and a later UK assertion. Only the UK fact remains in AMOR's retained triples, but both original texts remain searchable and are selected. Even with the UK source ranked first, the unaugmented context yields zero correct answers out of four. Full AMOR appends the retained UK assertion and all four answers receive native credit. New local interventions now remove only that assertion while keeping five original texts and nine other facts fixed: all four readers switch to USA-containing wrong answers. Replacing it with the actual extracted older assertion also yields zero correct; restoring it restores all four scores. This motivates preserving which assertion should govern the answer across memory construction, selection and presentation. AMOR does not delete old source text or guarantee resolution of every conflict.

3. **Individually accurate passages need not describe a consistent answer to the question.** The MH-Doc question asks for the 2001 population of the town associated with Kirton End. A baseline retrieves Kirton End's Lincolnshire location together with a different Kirton's 2011 population of 1,146. AMOR accepts the location facts rather than the candidate 2011 population fact, then recommends the Boston, Lincolnshire source containing 35,124 for 2001. All four full-AMOR answers are correct; all seven baselines fail on this selected question. AMOR still includes the distracting Kirton source, so this is evidence for making the required entity/date association available, not an implemented consistency filter. Use as a supporting example if the paper has space.

These findings connect the recommendation perspective to concrete memory requirements: which facts currently apply determines what further text should be selected, and the answer context must preserve those qualifications. AMOR does not optimize a formal consistency objective or guarantee logical validity. The SH-Doc Bohemond/nephew case independently replicates attribution errors, but replay shows an empty recognized-fact set and dense fallback; do not use it as evidence for graph recommendation. Full provenance, native scores and rejected cases are in `results.md` and the current analysis artifacts.

Targeted validation is complete (`6872885`, `6872907`, `6873195`). On all 100 FC-MH questions, using all extracted recognition candidates with the same current graph yields 11/12/9/3 native EM versus AMOR's 14/16/10/6. In the author/death case, substituting the actual older accepted authorship at fixed matching confidence removes the Eliot death source and changes all four answers from correct to incorrect, although the corrected Eliot authorship remains in the answer context; restoration recovers all four. On all 100 FC-SH questions, selecting augmentation facts from all rather than retained extraction lowers EM from 75/82/71/43 to 60/68/56/26 at the same five-source and top-ten fact budgets. These controls support the two design implications across the tested readers, but are not independently held-out evidence or mechanism prevalence estimates. Full and control score differences include one Qwen4 disagreement with an unchanged FC-SH prompt; all other unchanged-context scores agree. Keep this qualification and harmful cases in the experimental record. The old Dubliners triple has reversed argument direction, so do not portray relation normalization or retention as flawless semantic update resolution.

## 搜索与推荐的分析框架（2026-09-30）

主线改为：**事实匹配提供当前信息需求的初始相关性，离线记忆构建定义推荐关联，最终检验推荐文本是否共同提供回答所需信息。** 不再用“dense 很差，实体加传播更好”作为 novelty。推荐是论文的组织视角；具体方法贡献仍是如何由保留事实及出处形成可复用的关联与权重，而不是首次提出 PPR、协同过滤或集合效用优化。

ItemRank（IJCAI 2007）的正式文献将关联结构与个性化初始信号分开。AMOR 采用相同的图推荐视角，但关联由事实及其原文出处构造，当前问题提供相关性信号，不依赖用户评分或新增推荐模型训练。不能把问题叫作真实用户或把事实出现次数叫作用户反馈。MemRec（ACL 2026）属于使用记忆辅助推荐任务，并不自动等同于将原文上下文作为被推荐对象；未据此宣称 recommendation + memory 从未被研究。

Analysis 建议两个 subsection：

1. **Completeness of Recommended Context**：多个 baseline 已能命中至少一条支持文本，差异在是否补齐其余证据；报告原指标和增损计数，再用 5/10/15 曲线检验扩大返回数量能否替代推荐。数据支持扩大数量仍未追平，而不支持更多文本降低覆盖。
2. **From Fact Matching to Context Recommendation**：固定查询识别与初始分数，比较关联结构和权重；再区分已识别事实的直接出处与关联推荐的其他出处，用完整推荐列表与原文例子解释。最新核对中，单位边权到 AMOR 的 88 个新增支持 occurrence 全部来自后者，同时丢失 5 个；已识别事实出处内的 gold 集合逐题相同。边权效果不是更换编码器、更好地识别事实或返回更多篇。

初始化及传播步数作为附录实现机制验证，不再承担范式的新颖性。不同上下文共同提供完整证据是评估结果，不暗示当前方法显式寻找缺失事实、优化集合目标、执行顺序推荐或保证逻辑一致性。定性展示由同一图单位/事实边权产生的完整五篇推荐列表，不用 Dense 失败案例代替受控建图案例；全部失败与混合结果保留在 `results.md`。

新增双向干预支持具体的推荐机制解释：相同事实识别下，Michael Curtiz 传记与实体的权重 26 改为 1，传记由前五退出；在单位权重图上仅恢复此边则重新进入前五。原始查询、初始相关性、其他边和 BM25 都不改变。正文可以因此展示“关联怎样改变推荐出的材料”，而不只展示“某方法比某方法好”。推荐框架的新视角与具体构图设计可以作为论文贡献，但已有 PageRank/ItemRank 原理仍应准确归属。

## 简化候选验证（2026-09-29）

用户已批准的简化候选全量验证现已完成，288/288 格逐题核验通过。候选仅保留事实加权实体/原文连接，而非投影新增的所有连接。新 one-shot 的全部 72 格中，完整候选相对论文原生 baseline 的逐格最佳值为 21 胜、3 负；相对 w/o recommendation 为 22 胜、2 负，相对 w/o context augmentation 为 22 胜、1 平、1 负。Recommendation 的两项回退均在 Qwen 的 LoCoMo；增强在 Llama 的 2Wiki 回退，在 Qwen9 MH-Doc 持平。不将组件表述为每个任务都必要。

2Wiki 的中心原文 Recall@5 为 87.400%，完整支持率 68.9%；无 recommendation 为 69.725% 和 38.5%。同一选文的无增强版本保持相同支持覆盖，说明选文和最终上下文呈现应分开讨论。四 reader 共用 one-shot 选文，不构成四次独立检索验证。

简化候选的 IRCoT 对原生 BM25 在 cap 1/3/5 分别严格胜出 24/24、21/24、20/24，总计 65/72；对同增强 BM25 为 20/24、17/24、19/24，总计 56/72，另有 5 个平局。旧完整 AMOR 对两类 baseline 的严格胜出分别为 61/72 和 59/72，因此不能只据前一类比较宣称简化版全面更强。新旧完整方法直接比较为 one-shot 10 胜/5 平/9 负、IRCoT 31 胜/13 平/28 负。

IRCoT 中 recommendation 的贡献为 22/24、15/24、15/24 严格胜出，增强为 18/24、21/24、21/24；多轮条件下仍有回退。可保留的故事是事实决定的连接权重、基于图的 context recommendation 和最终 context augmentation 各自承担不同作用。不能将结果改述为投影新增连接不可缺少，也不能将 21/24 或 65/72 写成全领域 SOTA、统计显著或独立泛化验证。下文保留的旧完整版本数值不移用于简化候选，最终完整结果、原生指标、token 及新旧重复运行差异见 `results.md` 首节。

本轮提供了采用更简单图结构的实证依据：它在当前已测试范围内保留了较广的 baseline 优势，而不要求保留投影新增连接。但现有研究仍是离线构图和复用，不能扩张成在线更新或完整冲突处理；也不因为采用推荐系统视角就把已有 PPR 算子表述为新算法。

## 消融展示安排（2026-09-27，用户批准）

- 2026-09-28 更新：用户确认投影消融继续计算并保存，但论文不展示 `w/o projection`。完整内部结果仍保留该条件，不改数值，不因省略该行声称超图在多数设置下更优。
- 正文展示：one-shot 与 IRCoT 的完整 AMOR、w/o recommendation、w/o context augmentation。w/o recommendation 明确定义为用 dense 排序代替 PPR，保留 BM25 融合和上下文增强，不把原生 BM25 改名为这个消融。
- 附录：相同拓扑二值边权、关系归一化、已有事实筛选四格、已有去掉 BM25 融合，以及参数敏感性。
- 参数敏感性固定为单因素对照：alpha 为 0.25、0.5、0.75；lambda 为 0.01、0.05、0.1。中心值复用主方法。不逐任务选最佳值，不改变主方法。
- 核心对照覆盖四个 LLM、六任务全量；IRCoT 新检索条件重新生成轨迹，到 5 轮后读取 1/3/5 的前缀。上下文增强只作用于最终回答。
- 实验入口为 `experiments/ablate_memory.py`；结果写入 scratch 的 `optimization_paper_ablation_seed42_20260927/`。所有预定实验完整保存；论文展示子集与完整实验记录分开，不以删行代替证据。

## 超图构图的设计依据（2026-09-28 核对）

研究主张应是以抽取事实及其来源共同决定连接和边权，而不是宣称普通图无法表示该系统。`optimization/graph_construction/statement_incidence.py` 的 `statement_incidence_graph` 为每个不同事实连接其主体、客体和选中的来源；`project_statement_graph` 投影后删除内部事实节点，实际 PPR 仍运行在加权普通图上。

投影采用 [Kumar et al., Applied Network Science 2020, Eq. 3](https://link.springer.com/article/10.1007/s41109-020-00300-3) 的度保持约简。该文 Proposition 1 与 random-walk 解释给出归一化依据：每个超边成员向其他成员分配的权重和等于该超边权重。代码中的单位权重事实采用 `1/(degree-1)`，去除对角线，然后加上原文连接残余项。已有 `StatementIncidenceTests` 覆盖投影度、残余连接和自环。因此，性质适用于投影项；完整 AMOR 还包含原文连接 C，不能把完整传播直接称作只有事实超边的随机游走。

该数学性质不是 AMOR 的新定理，不保证 QA 提升，也不证明投影保留了全部高阶信息或关系方向。AMOR 采用这种已有算子的理由，是从同一事实关联产生连接与权重，并复用原文节点进行上下文推荐。实验依据及混合结果见 `results.md` 的 2Wiki 连接与边权控制，不能外推成所有记忆任务上的优越性。

### 完整消融后的结论

IRCoT 已完成。Recommendation 的严格获胜数在 one-shot 与 1/3/5 轮分别为 21/24、20/24、15/24、15/24；上下文增强为 22/24、18/24、19/24、22/24。论文可以据此解释关联传播与最终上下文呈现的不同作用，不能把系统增益全部归给构图。

投影相对删除版本分别为 11/24、13/24、10/24、10/24 严格获胜，不能把“超图在多数设置下更好”作为总体结论。可支持的设计理由是：用事实及其来源作为统一构图单位，通过已有度保持投影得到连接和边权。2Wiki 的 one-shot 以及 cap 1 支持证据覆盖改善为该选择提供局部实证；更高轮数的图结构差距多有缩小，QA 结果仍有回退。该结果适合说明作用边界，而不是以数学术语掩盖混合效果。

建议核心表述：AMOR uses extracted facts and their sources to construct associations for context recommendation. Graph propagation complements direct matching, while context augmentation determines how selected information is presented to the answer model. 不将投影算子本身、PPR 或推荐系统视角写成新发明；新意应落实到实际构图组织和经控制的系统效果。

### 连接与权重分离后的更新（2026-09-29）

新增四格对照固定事实候选、识别、初始相关性、传播和上下文规则，仅分离投影对原有连接边权与新增连接的贡献。六任务、四个 LLM 的 one-shot 已完成并通过逐题核验，详见 `results.md` 和 `connection_controls/complete.json`。

2Wiki 中，二值原有连接的 Recall@5 为 84.775%；仅按事实加强原有连接达到 87.400%，完整 AMOR 为 87.425%。对应完整支持率为 63.5%、68.9% 和 69.1%。因此，当前证据主要支持事实决定的连接权重，而不是新增连接不可替代。完整 AMOR、仅加强原有连接和无投影版本都在 21/24 个设置超过论文的七项外部 baseline；这项系统层面的胜出不能证明完整投影的必要性。

超图仍是实际构图表示：同一事实的实体及来源共同贡献归一化权重。但它不是只能由超图实现的在线推理能力，投影后的普通加权图可以精确实现同样的传播。先前导演配偶案例中，仅加强原有连接与完整 AMOR 的五条原文及顺序完全相同，也不能用该案例证明新增连接必要。

可以表述为：AMOR organizes extracted facts together with their sources and uses their shared participation to determine graph weights for context recommendation. 不应写成：High-order connections are essential to AMOR's gains. 这里的区别是可验证的构图规则与不受结果支持的必要性主张，而不是换一个数学名称。

主算法暂不改变。简化图尚无对应 IRCoT 结果；是否替换主方法等待用户决定，不能复用完整 AMOR 的 IRCoT 数值作为简化图结果。

全部官方问题类型分析已完成：完整 AMOR 相对仅加强原有连接，在 bridge_comparison 的四 reader F1 均提高，但完整支持率相同；inference 的完整支持率略提高，四 reader F1 却均下降。该对照说明证据覆盖与答案使用需要分别讨论，不宜将某一类别的改善写成普遍的高阶推理优势。完整类别表、恢复与损失统计见 `results.md`。

更新：2026-09-22。本文是写作定位，不替代 [实际算法](algorithm.md) 或 [实验结果](results.md)。

## 1. 一句话故事

**记住一段信息，并不等于能在需要时把它作为证据取回来。我们研究如何将抽取事实与原文出处共同转化为检索图，使查询更容易找到回答所需的材料。**

建议题目方向：**Fact-Induced Graphs for Evidence Retrieval**。

“Context recommendation”用来解释任务视角：问题给出当前需求，原文是待选材料，系统利用直接相关性和事实连接选择上下文。它不是另一个尚未实现的模型，也不表示我们预测未来问题、学习用户偏好或训练了推荐网络。

## 2. Research Question

> 在不重新抽取语料、不训练 retriever 的条件下，如何把事实及其出处转化为图中的连接与权重，从而提高支持证据检索和下游问答效果？

这里有两个相关但不同的环节：

1. **找到证据：** 图中“实体有关联”是否能转化为“取回支持当前问题的原文”？
2. **使用证据：** 同一份原文交给 reader 时，紧凑事实和邻接对话分别带来什么收益与代价？

第一个是构图贡献的主线；第二个解释完整系统为何有效以及何时失效。不要用上下文组件的收益替代构图证据，也不要把研究问题扩张成通用在线记忆更新。

## 3. 为什么这个问题值得做

直接相似度能找到与问题表面相关的材料，但多跳问题还需要通过中间实体找到其他出处。图检索可以提供这种连接；问题在于，连接方式和权重会影响传播最终偏向哪些原文。

“某实体在某原文中出现”只表达成员关系，不表达该原文支持了哪些不同事实。我们的构造把每条事实与实体、出处的共同参与显式计入权重。例子与完整公式见 algorithm.md。

这是一个可以单独验证的设计问题：固定候选事实、PPR、BM25 和上下文政策，改变图，检查支持段落覆盖及答案是否变化。原图替换实验提供模块层面的证据；新增四格对照进一步分开原有连接的权重贡献与投影新增连接，发现前者解释了 2Wiki 的大部分覆盖增益。

## 4. 推荐系统与超图分别承担什么角色

| 视角 | 对本工作的实际作用 | 不应声称什么 |
|---|---|---|
| 图推荐 | 把查询看成当前需求，把原文看成候选项，用关系传播补充直接匹配 | 新的协同过滤算法、学习到的偏好或校准效用 |
| 事实与出处的超图表示 | 定义一个事实由哪些实体及原文共同支持；从关联结构生成图权重 | 首次应用超图、完整 n 元信息抽取或无损高阶推理 |
| 标准投影和 PPR | 给已有构图和排序算子明确解释 | 新的数学工具、最优证据选择保证 |
| 事实加原文上下文 | 明示抽取断言，同时保留抽取可能遗漏的细节 | 无成本增强、所有 reader 都能正确使用证据 |

推荐系统的相近先例是 [ItemRank（IJCAI 2007）](https://www.ijcai.org/Proceedings/07/Papers/444.pdf)：在物品关系图上用随机游走传播个性化信号。[LightGCN（SIGIR 2020）](https://hexiangnan.github.io/papers/sigir20-LightGCN.pdf) 则从用户-物品交互图学习表示。我们采用图排序视角，不使用交互历史或新增 retriever 训练。

投影采用 [Kumar et al.（Applied Network Science 2020）](https://link.springer.com/article/10.1007/s41109-020-00300-3) 的已有算子。当前事实通常关联主体、客体及一个被选中的出处，所以论文应具体解释三者怎样影响边权，不靠“hypergraph”这个名称制造理论贡献。

## 5. 方法章节如何线性展开

### 5.1 Memory Retrieval as Context Selection

先提出需求：选择能支持回答的原文，既要直接相关，也要覆盖事实连接指向的材料。说明冻结语料、原问题和输出上下文，不先罗列所有工程组件。

### 5.2 Graph Construction from Facts and Sources

定义实体、原文和抽取三元组；交代关系归一及来源支持选择。用一个简短例子说明，事实节点是构图时的关联单位，在线检索前会被消去。

主配置的支持规则必须如实写明：按加载顺序选择最后出处，不声称这是事件时间或可靠的冲突消解。schema 中没有被使用的字段不写成贡献。

### 5.3 From Fact Incidence to Passage Connections

给出代码中已有的标准投影，再加保留的实体到出处连接。解释每条事实怎样改变边权，说明 C 和 F 各自的作用，不将 C 隐藏在公式外。

归一化使一个事实对每个成员的投影出边贡献为 1；这只描述 F，不是整个图 A 的概率守恒定理或检索质量保证。该性质来自已有投影，不作为原创 theorem。

### 5.4 Query-Conditioned Ranking

交代事实匹配与识别、PPR 和 BM25/RRF。将它们视为采用的排序组件，明确我们的变化发生在构图和候选支持上。方法标题不必围绕实现底座，但相关算子及代码来源仍应得到归属。

### 5.5 Evidence Context and Iterative Retrieval

说明“原文 + 邻接记录 + 问题相关事实”的输入组成；单次 QA 与 IRCoT 共享最终上下文政策。明确当前多轮结果是固定轨迹后的最终 QA，不暗示每轮推理都运行了增强策略。

## 6. 贡献应该怎样写

**贡献一：具体的建图设计。** 从抽取事实、实体和出处的关联生成段落检索连接，在固定抽取结果上改变图结构及权重。不是提出新的 OpenIE、PPR、RRF 或投影算法。

**贡献二：构图对证据覆盖的受控验证。** 保留相同候选索引与下游政策，当前图相对原图使 2Wiki Recall@5 从 74.825% 到 87.425%，找齐支持段落的题数从 468 到 691；四 reader 的答案 F1 提升 7.07--9.47 个点。这是完整构图模块的作用，不是每条边或筛选规则均被证明必要。

**贡献三：跨任务和使用方式的效果与边界。** 四 reader 的 one-shot 对九项原生 baseline 配置分别在 5/6、6/6、5/6、5/6 任务领先；IRCoT 对同上下文组件的 BM25 为 59 胜、1 平、12 负。事实展示、邻接对话与支持筛选的消融解释不同收益来源，也保留拒答、历史事件和部分多轮任务上的退化。

这些结果是 test-as-dev、单 seed 的本地证据。独立验证尚不能由现有表格替代；论文不得把反复开发的数据写成从未用于方法选择的测试集。

## 7. 与相关工作的区别

- **HippoRAG / HippoRAG 2：** 同属抽取知识辅助图检索；我们重点检验事实与出处怎样生成连接及权重。继承的识别和传播组件不是原创。
- **HyperGraphRAG：** 已研究 n 元事实和实体/超边检索；我们的输入是既有三元组，重点是将实体及出处的事实关联转为段落传播图，不宣称更完整的 n 元抽取。
- **CatRAG：** 已研究查询相关的遍历；我们冻结图构建后供不同查询及检索流程使用，主贡献不在新增遍历控制器。
- **A-MEM、SeCom、MemInsight、LightMem 等：** 研究记忆组织、语义增强、粒度和整合。我们检验既有信息如何成为检索及最终 QA 证据，不概括为这些方法都丢弃原文或忽视上下文。

图方法的区别必须以实现和受控比较支撑；memory 方法的负例只能说明实际返回证据的差异，不能未经检查就断言它们的整个存储丢失了信息。

### 可用于论文的英文定位

```latex
We study memory retrieval as query-conditioned context selection: the goal is to identify source passages that support the current information need, including evidence not captured by direct query--passage matching.
Our method constructs a retrieval graph from the joint participation of entities and source passages in extracted facts.
A standard normalized hypergraph projection converts these associations into weighted connections for passage ranking.
The retrieved records are then supplemented with query-relevant facts and neighboring conversational context.
Controlled graph replacement and context ablations separate improvements in evidence discovery from changes in how evidence is presented to the reader.
```

### 推荐系统 Related Work 短段

```latex
Graph-based recommender systems exploit relational structure to rank items beyond direct matching.
ItemRank propagates personalized signals over item connections using random walks~\citep{gori2007itemrank}, while LightGCN learns user and item representations through neighborhood aggregation over interaction graphs~\citep{he2020lightgcn}.
We adopt the graph-ranking perspective for context selection: queries specify information needs, passages are candidate items, and fact--source associations provide the relational structure without user-interaction histories or additional retriever training.
```

已有 bib 文件中如未包含这两篇，可使用：

```bibtex
@inproceedings{gori2007itemrank,
  title = {ItemRank: A Random-Walk Based Scoring Algorithm for Recommender Engines},
  author = {Gori, Marco and Pucci, Augusto},
  booktitle = {Proceedings of the 20th International Joint Conference on Artificial Intelligence},
  year = {2007},
  pages = {2766--2771},
  url = {https://www.ijcai.org/Proceedings/07/Papers/444.pdf}
}
@inproceedings{he2020lightgcn,
  title = {{LightGCN}: Simplifying and Powering Graph Convolution Network for Recommendation},
  author = {He, Xiangnan and Deng, Kuan and Wang, Xiang and Li, Yan and Zhang, Yongdong and Wang, Meng},
  booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2020},
  pages = {639--648},
  doi = {10.1145/3397271.3401063}
}
```

## 8. 写作取舍

### 合作者提供的成本拆分与主表安排（2026-09-27）

用户确认稿中的 21.6M 信息抽取 tokens 与 5.91M 关系归一化 tokens 来自合作者，本轮写作保留该拆分，总计 27.51M，不再自动替换为本地日志的阶段拆分。
这是合作者提供的稿件数据口径，不将其标记为已由本地日志复核，也不修改历史成本 CSV 或原始记录。
One-shot 主表增加 `AMOR w/o context augmentation`，与 IRCoT 的增强开关术语保持一致；增加后按全部展示行重新计算最佳值，正文的 21/24 仍明确指相对外部 baselines。
主表在第一个模型分块为 BM25、HippoRAG 2、CatRAG、Mem0、LightMem、AnchorMem 加上对应 citation，后续分块不重复；Dense 所用 embedding 模型引用放在实现细节。
One-shot 的 BM25 同上下文增强对照尚未完成，本轮不能添加该行或填入假定结果。

### LaTeX 排版规则（2026-09-27）

论文正文与 figure/table caption 的 LaTeX 源码统一采用一句话一行。句内不为控制行宽手动换行，也不把多句话合并到同一行；段落之间留空行。后续提供可复制的论文文本时遵循同一规则。
Discussion、Conclusion、Limitations 使用独立 section，内容聚焦算法设计、实验支持的结论与方法适用边界；seed 和开发评测数据使用情况放 Experimental Setup，不用它们代替算法局限讨论。

### 术语偏好（2026-09-26）

用户已确认方法名称为 **AMOR**（Agent Memory Organization and Recommendation）。论文图表统一使用 AMOR，不再用 Ours 或上一篇论文的名称。IRCoT 主图保留 BM25 与 AMOR 各自 w/ 或 w/o context augmentation 四种条件。它们区分选取方法和最终上下文增强，不能单独隔离超图构建的作用。One-shot 主表增加无上下文增强版本；事实与相邻文本的完整展示拆分仍放附录，不等同于 BM25 与 AMOR 的四格对照。

用户明确反对使用 `indexing` 和 `retrieval`。本项目的算法名称、缩写展开、论文正文及方法介绍不使用这两个词，也不为了凑缩写重新引入。按实际含义使用 `memory construction`、`graph construction`、`memory organization` 或 `context recommendation`，不机械替换为另一个模糊术语。此前包含禁用词的名称展开不是确定的方法名称。引用文献的原始标题、代码标识符和文件路径保持原样。

主线是“事实及出处如何影响证据选择”，不是“算法能更新所有记忆”，也不是“借推荐系统和超图证明一切”。

可以自信强调已测得的图替换收益和跨 reader 结果；同时明确当前构图控制替换了一个模块组合。最新事实附录带来的额外提升要单独报告，不包装成图结构改动。

保留两个最有解释力的例子：2Wiki 导演比较展示构图补齐证据、事实展示又影响回答；Sigur Ros 展示抽取遗漏经附录影响答案。正负例共同说明该表示的作用，而不是为所有结果编一个原因。
