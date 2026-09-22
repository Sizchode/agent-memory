# 论文故事与贡献边界

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

这是一个可以单独验证的设计问题：固定候选事实、PPR、BM25 和上下文政策，改变图，检查支持段落覆盖及答案是否变化。现有原图替换实验提供了这一层证据，但没有把每一类边的独立效果全部分开。

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

主线是“事实及出处如何影响证据选择”，不是“算法能更新所有记忆”，也不是“借推荐系统和超图证明一切”。

可以自信强调已测得的图替换收益和跨 reader 结果；同时明确当前构图控制替换了一个模块组合。最新事实附录带来的额外提升要单独报告，不包装成图结构改动。

保留两个最有解释力的例子：2Wiki 导演比较展示构图补齐证据、事实展示又影响回答；Sigur Ros 展示抽取遗漏经附录影响答案。正负例共同说明该表示的作用，而不是为所有结果编一个原因。
