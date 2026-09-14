# 上下文化索引表示对照

日期：2026-09-13。来源抽取与真实索引已完整，检索及 QA 正在运行；尚无完整六任务 QA 结果。

## 文献依据与问题

[REMem 第 3.1 节](https://arxiv.org/html/2602.13530v1#S3.SS1) 将原始经历转成
带上下文的 gist，再为这些文本建立索引。其第 4.4 节另有单步检索版本 REMem-S。
我们借用其公开 passage 抽取模板检验表示，而不复现整套 REMem 或新增 agentic retrieval。

当前来源事实 incidence 对照没有改善相对直接参照的 2Wiki Recall@5。
这不能证明上下文化无效，因为该组没有改变原始文本 embedding，新增节点也没有语义 seed。
下一步的问题是：先改变用于匹配的文本表示，能否提高同一来源的可检索性？
这与检索结束后给 reader 附加事实是两个不同组件。

## 已实现与运行范围

实现为 [contextual_gists.py](graph_construction/contextual_gists.py)，运行脚本为
[build_contextual_gists.sbatch](build_contextual_gists.sbatch)。
直接加载本地 REMem 的 `episodic_gist_extraction_wikipedia.py`，不改模板文字。
六任务统一用这个 passage 模板，不沿用上游按数据集选择不同 prompt 的做法。
这项差异是明确的适配，不把所得分数记作 REMem baseline。

| 任务 | 原始来源段落数 |
| --- | ---: |
| SH-Doc_QA | 412 |
| MH-Doc_QA | 875 |
| FactConsolidation-SH | 537 |
| FactConsolidation-MH | 537 |
| LoCoMo | 5882 |
| 2WikiMultiHopQA | 6119 |
| 合计 | 14362 |

来源均为原冻结 OpenIE 缓存中的 `idx` 和 `passage`，共 15 个语料组。
不读取抽取三元组、问题、答案或 gold evidence，不重新切分、重组会话或修改 loader。
LoCoMo 保留现有单条消息中的日期、说话者及原文本，不增加邻居消息。
因此当前模板无法获得来源段落之外的指代上下文，不能预称为跨消息指代消解。

生成器沿用 Qwen3-30B-A3B-Instruct-2507，bf16、temperature 0、seed 42、关闭 thinking。
每次最大输出 8192 token；超出输入预算时报错，不截断或补造内容。
JSON schema 约束格式；不修复失败 JSON，不删除重复 gist，也不把空列表替换成原文。
所有调用保留原始响应、finish reason、实际 usage 与耗时；格式合法不等于语义忠实。
原始来源及既有图缓存不覆盖。

作业 6341669 使用 gpu 分区的 2 张 L40S，输出目录为：
`/oscar/scratch/zliu328/agent-memory-outputs/optimization_contextual_gists_seed42_20260913`。
16:48 EDT 快照已记录 378 次完整且格式合法的响应、12895 条 gist，无已记录调用错误；
这些调用的输入为 422935 token、输出为 198467 token，尚不是全量成本。
作业仍运行，完整语料组有各自 `complete.json`，所有组结束后才写根目录完成标记。

## 后续表示对照边界

先完成全部来源抽取，再建立索引；不在未完整落盘时挑题报告 QA。
复用 REMem 已实现的按原 chunk 拼接 gist 表示，保持每个来源一个向量和原来源 id。
同一份表示分别接入既有向量检索和 HippoRAG passage 匹配路径，与各自原文索引对照。
不同时更换图拓扑、RRF、来源读取范围或 QA prompt，不新增融合分数、权重或阈值。

替换 passage embedding 会改变 passage reset 和原有 dense fallback 的排名；
因此只能称检索程序与参数不变，不能称全部 query reset 不变。
最终仍读取原文，先隔离索引表示影响，不将 gist 同时作为新的 reader 附录。
索引实现为 [index_contextual_gists.py](index_contextual_gists.py)，包括 `gist_hipporag`、
`gist_dense` 和 `raw_dense_control`。后者直接用 HippoRAG 自带的 dense 方法检索原文，
不预设它与基线池中另一个 dense 实现逐题相同，单列全量 QA。
抽取全部完成后才统一构建六任务索引，随后才放行查询；每任务先验证原 HippoRAG
与保存结果全量一致，再用普通加载接口生成候选检索，不使用保存的 resets。
空 gist 列表按统一拼接操作得到空字符串并交给原 encoder，保留该来源及原图连接，
计数单列；不填回原文，不删除来源，不把空表示视为有事实内容。
超出原 embedding 长度时明确失败，不静默截断。三组完整 QA 尚未运行。
77 项测试通过，涵盖原来源绑定、向量顺序、维度、空列表、长度及普通检索接口。

这一步即使有效，也首先支持“上下文化索引表示有用”，不是新图拓扑的独立贡献。
之后才根据结果决定是否继续研究来源关联，不能把多个改变合并后的分数全算到建图。
gist 是事实改写，可能比原文更长；是否压缩必须按实际 token 验证，不由名称推出。

## 评测协议

沿用四个 MAB 任务的 substring exact match、LoCoMo 原类别评分和 2Wiki answer F1。
每个候选仍是六任务全部 3386 题、三个 reader 共 10158 次回答，test 用作开发集。
不新增划分、评分器、数据转换或自定义综合分数。
REMem 原论文的生成模型、embedding、检索单位与预算、LoCoMo F1 实现均与本协议有差异，
不直接拼接论文分数到本地结果表，也不将此组件适配命名为完整 REMem 复现。

## 环境恢复记录

6341669 在 FCSH 完整 537 条后继续 FCMH，后者有 535 条有效输出和 2 次连接错误；
作业以 1:0 结束，失败调用的 usage 未知，不补零。6341980 续跑未启动生成，
因为共享 vLLM 环境入口消失，退出码 127；与模型输出质量无关。
home/scratch 搜索未找到其他完整环境，但发现 CUDA 12.9 的 vLLM 与 PyTorch 缓存。
项目专用恢复目录为 `/oscar/scratch/zliu328/agent-memory-envs/vllm_cu129`。
[generator_runtime.txt](generator_runtime.txt) 固定主要运行依赖，
[original_generator_constraints.txt](original_generator_constraints.txt) 来自原目录残留的
dist-info 名称，不是对完整原环境执行的 pip freeze。复用缓存并补齐缺失依赖，未下载模型。
CUDA 模块仍为 `cuda/12.9.0-cinr`，抽取输入、模板、模型和解码参数不变；
恢复环境不保证跨硬件逐 token 相同，已有有效输出不重抽、不择优替换。

恢复安装后 190 个包通过 `uv pip check`，已确认 Torch 的 CUDA 为 12.9、证书文件存在。
安装命令保留如下，环境创建使用已有 Python 3.11，不覆盖已有环境：

```bash
uv venv --python /oscar/scratch/zliu328/agent-memory-envs/qwen_generator/bin/python \
  /oscar/scratch/zliu328/agent-memory-envs/vllm_cu129
uv pip install --python /oscar/scratch/zliu328/agent-memory-envs/vllm_cu129/bin/python \
  --default-index https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match \
  -r optimization/generator_runtime.txt -c optimization/original_generator_constraints.txt
```

续跑 6342535 已在 2 张 L40S 上启动；后续依赖链已经提交，见
[gist_index_jobs.json](https://github.com/Sizchode/agent-memory/blob/f2121c11cc6d0c36bf931674152db0ca07669573/optimization/gist_index_jobs.json)。索引作业 6342552 等待全部抽取完成，
检索数组 6342553 等待全部索引冻结，QA 数组 6342554 至 6342559 分别等待对应完整任务检索。
三组候选各 10158 次 QA，合计 30474 次；汇总作业 6342560 等待所有 QA 成功。
这只是执行范围，不是完成数量或效果声明。

17:22 EDT：新服务已通过 health 并产生真实的 200 生成响应，FCMH 的两个缺失来源已补齐。
FCSH 与 FCMH 现均完整 537/537，已进入 LoCoMo 抽取；旧失败记录未删除。

## 接入已有强组合

17:28 EDT 补充 `canonical_rrf_gist_index`，直接参照为完整的 `canonical_rrf_sentence_facts`。
该参照 9B/4B/2B 严格胜出为 5/6、6/6、3/6。只替换它的 passage embedding 为同一份
gist embedding，沿用其原图边权、RRF 常数和候选窗口、BM25 来源、去重窗口及逐行事实读出。
gist 不追加给 reader，不改变 QA prompt，也不重新抽取或筛选事实。

构建时复制参照的边权与词面来源顺序，引用已有的完整来源表示，并核对两者来自同一原图。
新加入的单测检查这些文件及 metadata 不变，只增加 gist 索引引用；总计 78 项测试通过。
完整检索阶段先通过普通接口分别复核原 HippoRAG 与旧强组合的全部问题，之后才运行新候选。
这样直接检查组合前的控制是否一致，不从某次重跑中挑更高分替代旧 QA 成绩。

这个组合使用当前已有的较强整体配置，不按任务或 reader 选配置；它增加完整六任务三 reader QA，
不增加 gist 抽取或 embedding 成本。固定的是读出规则，不保证不同来源排名产生相同 token 数。
即使有效，也只能先归为在已有组合中改进索引表示，不能称为提出了新的图拓扑或理论保证。

新增 QA 数组为 6342728 至 6342733，仍依赖对应的 6342553 检索数组任务。
旧三候选汇总 6342560 在 pending 状态取消，替换为等待全部四候选 QA 的 6342734。
当前范围为四候选共 40632 次 QA；现有三个候选的 QA 作业未取消或重启。
索引构建与检索尚未启动，新增组合没有成绩。

## 已完成来源的表示长度

17:50 EDT：FCSH、FCMH、LoCoMo 十组和 SH 共 7368 个来源已完整抽取。
对这些完整语料组的全部来源，使用现有 Qwen3-Embedding-0.6B tokenizer，
`truncation=False, add_special_tokens=True`，分别计算原文和实际换行拼接 gist 的输入长度。
这里只读取来源与模型输出，不读取问题、答案或按长度筛选来源。

| 任务 | 来源数 | 原文 token | gist token | gist / 原文 | 最长 gist token |
| --- | ---: | ---: | ---: | ---: | ---: |
| FactConsolidation-SH | 537 | 336557 | 238136 | 0.708 | 861 |
| FactConsolidation-MH | 537 | 336557 | 237809 | 0.707 | 872 |
| LoCoMo | 5882 | 313505 | 456208 | 1.455 | 297 |
| SH-Doc_QA | 412 | 212171 | 212745 | 1.003 | 932 |

上述完整来源没有空 gist。这些是索引文本的实际 tokenizer 长度，不是生成器调用成本，
也不是 reader context 用量。MH 和 2Wiki 尚未完成，不能称为全量六任务统计。
FCSH/FCMH 存在约 29% 的文本缩短，但 LoCoMo 增长约 46%，SH 基本不变。
因此本轮统一的操作是来源表示改写，而非统一压缩；是否改善检索仍待新索引及完整 QA。
不据此按任务切换原文或 gist，也不截断较长输出来制造压缩率。

## 第二次续跑

6342535 在 17:58 EDT 以 FAILED 1:0 结束，MH 有 872/875 个有效来源。
未完成来源为一次连接错误和两次 8192-token length 输出；后两者分别出现重复空白和括号文本。
原始输出与成本全部保留，不修复 JSON 或截去空白来接受非 stop 输出。
续跑 6343507 保持相同模板、schema、模型、seed、温度和预算，复用已有 8240 个有效来源，
只补缺失来源，随后继续全部 2Wiki 抽取。

原下游作业被 Slurm 自动取消，均未实际执行，不能仅修改原索引作业的依赖。
新链为索引 6343525、六任务检索数组 6343527、六个三 reader QA 数组 6343528 至 6343533、
报告 6343534。每个 QA 数组现在一起执行四候选，比较范围及次数不变。
完整新旧 ID、原因及依赖见 [gist_index_jobs.json](https://github.com/Sizchode/agent-memory/blob/f2121c11cc6d0c36bf931674152db0ca07669573/optimization/gist_index_jobs.json)。

6343507 在 18:01 EDT 再次 FAILED 1:0，用时 3:03。连接失败的来源已正常补齐，
两条 length 输出再次达到 8192 token，均仍不是合法完整 JSON；MH 现为 873/875 有效来源。
两次 length 的全部成本分别保留，不继续无条件重试。新下游链也由 Slurm 自动取消，均未启动。

独立诊断作业 6343703 输出到 `optimization_gist_format_diagnostic_seed42_20260913`。
代码 [diagnose_gist_generation.py](diagnose_gist_generation.py) 自动找出尝试过但仍无有效结果的来源，
不硬编码来源 ID，不读取 QA 数据，不包括尚未尝试的 2Wiki。调用现有 vLLM
`StructuredOutputsConfig(backend="xgrammar", disable_any_whitespace=True)` 的紧凑 JSON 模式，
模型、prompt、内容 schema、seed、温度和 token 预算沿用原配置；不修补输出。
诊断逐条请求，并发为 1，因此成功也不能单独证明格式开关是唯一原因。
诊断仅检查完整生成与格式合法性，不产生正式 gists/index，不自动提升到原抽取目录。
任何正式配置变化都必须另行记录其全语料执行范围，不能把两个诊断例当成全量评测。

## 紧凑 JSON 的完整语料版本

诊断 6343703 在 18:07 EDT COMPLETED 0:0。两条均 stop、原 schema 校验通过，
输出分别为 360 / 498 token，输入 1039 / 991，调用约 5.77 / 3.65 秒，没有输出修复。
诊断成功说明该设置可以生成完整表示，但因诊断并发与原运行不同，不作单因素因果断言。

正式新抽取为 `optimization_contextual_gists_compact_seed42_20260913`，生成作业 6344538。
所有六任务 14362 来源统一使用 xgrammar 的 `disable_any_whitespace=True`，
原 prompt、内容 schema、模型与解码预算不变，并发恢复正式协议的 16。
运行参数通过 `GIST_COMPACT_JSON=1` 同时传给服务端和抽取设置记录，旧设置原样保留。
全量重新抽取使新语料的配置一致，不把旧版本和两个诊断结果混合成正式索引。
旧失败调用、已成功来源及其成本保留；全量重抽的实际成本另行记录。

新索引/QA 目录为 `optimization_gist_index_compact_seed42_20260913`。
索引 6344548 通过 `GISTS_EXPERIMENT_ID` 明确指向新抽取根目录，不使用旧默认目录。
六任务检索数组 6344549，QA 数组为 6344550 / 6344551 / 6344553 / 6344554 / 6344555 / 6344556，
完整四候选汇总 6344557；执行顺序仍为全来源完成、全索引冻结、完整控制检索、QA。
这不是新图表示候选或新评分设置，仅是解决格式失败后重新执行同一表示组件实验。

## 全量格式失败与原生接口诊断

上述全量紧凑运行没有成功：FCSH 又出现多条 length，部分响应将模型对反事实材料的说明
写进字符串并持续重复。6344538 因已观察的格式失败主动停止，CANCELLED，用时 9:14。
记录最终为 399 个有效 stop、15 个 length、16 个停止造成的 abort、107 个清理期间的连接异常。
全部原始记录保留，已知 token 为输入 481209、输出 277437；异常 usage 未知，不能补零。
下游索引与 QA 均未启动。两个诊断例成功没有推广到全语料，不将本轮失败解释为图方法无效。

新诊断 6344799 使用 REMem `_extract_chunk(json_mode=True)` 的原生 json_object 请求，
默认服务端 JSON 格式，取消本地严格 schema 的生成约束；返回后仍按原 gists 内容结构检查，
不使用上游的 JSON 修复函数。模型、prompt、seed、T0、8192 预算不变，诊断并发为 1。
合并两目录所有尝试过但没有有效输出的来源得到 140 条，包含停止造成的 123 条请求，
不将它们全称为自然格式失败。输出到 `optimization_gist_json_object_diagnostic_seed42_20260913`，
不会自动进入正式抽取目录、图或 QA；未完整实现或验证新的正式生成协议。

## 当前有界格式重试版本

json_object 诊断最终 139/140 有效，作业 6344799 FAILED 1:0，用时 11:22。
唯一未完成仍是原 MH 的 `chunk-53a79c07fca298a488eb49c531ad04af`，length；
全部诊断输入 156909、输出 82182 token，不进入正式索引。

后续改为统一的有界格式失败处理，完整定义见 [gist_generation_protocol.md](gist_generation_protocol.md)。
正常原格式失败后才允许一次紧凑格式请求，不修补、不比较质量、不按任务和 ID 路由。
这是对原无格式重试协议的明确修订，不归为图或检索新算法。传输错误与 abort 单独处理。

生成作业 6345102，新抽取根目录为 `optimization_contextual_gists_retry_seed42_20260913`。
核对原设置和原文后，导入原单格式调用日志及其 8241 个有效来源，不重新生成已有效的来源；
不导入两次诊断或全局紧凑版本。成本以原调用来源去重，复制日志不计新请求。
原目录与全部失败历史保留。仍要求全部来源完成，再构建完整四候选索引及 QA。

6345102 未正确启用第二种格式：vLLM 当前后端忽略了请求级 whitespace 开关，已将本次接入标记 INVALID。
实际新增两次请求的输入为 2030、输出 8774 token；复制日志不属于新增请求。该组不进入索引。
当前修正版本为 `optimization_contextual_gists_format_retry_seed42_20260913`，作业 6345810。
使用编译得到的紧凑 EBNF 请求，并保留原内容检查；对 minLength/JSON 转义的兼容问题与真实测试
详见生成协议。没有从无效目录迁移候选输出，仍只导入经过输入与设置核对的原单格式日志。

6345810 于 19:20 EDT 正常结束（COMPLETED 0:0，29:27），完整覆盖 14362 来源和 15 个组，
得到 123023 条 gist，空列表来源为 0。输入 ID、有效调用、gists 文件与全部完成标记逐组相符。
新增 6122 次请求：6119 次原格式、3 次 grammar；其中 6121 stop、1 length。
新调用输入 3748150、输出 1018448 token，包含 2Wiki 原格式 length 及其成功重试的成本。
旧日志单独计入一次：8248 次调用，已知输入 5728793、输出 1927344，3 次异常 usage 未知。
这些数不含其他独立诊断与无效版本的研究成本；完整失败历史见 record.md。
四候选索引作业 6346491 已启动；全部后续检索与三 reader QA 仍按原依赖链等待，不提前报告分数。

索引 6346491 随后正常完成（5:04），完整覆盖 14362 来源，embedding 输入共 2493586 token。
全部向量数组来源键唯一且齐全、数值有限；packed 候选的权重、fusion 和 readout 与冻结控制一致。
四个 MAB 任务检索均完成，各自 100 题的原始和 packed 控制全部匹配旧结果。
LoCoMo / 2Wiki 检索与已就绪的 QA 正在运行，尚不能汇报六任务分数。
