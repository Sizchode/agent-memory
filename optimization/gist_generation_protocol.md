# Gist 生成接口协议

本版解决格式失败，不是新的记忆算法，也不把两次序列化请求称为 agentic retrieval。
全部请求只接收原始来源；不接收问题、答案、gold evidence、任务成绩或 reader 身份。

## 固定处理规则

1. 对每个来源，先用原 json_schema 请求。模型、prompt、内容 schema、T0、seed42、8192 输出预算不变。
2. 只有 length、无法解析的 stop 或内容结构不合法的 stop 才允许再次生成，使用同一 schema 的紧凑 JSON。
3. 每种格式至多一次完成尝试。跨续跑保留失败状态；两种都失败则报错，不继续采样。
4. 第一个合法完整输出直接采用，不比较文本质量或下游效果；重复 gist、合法空列表及原措辞照常保留。
5. HTTP 错误和 abort 不算内容格式失败，不触发格式切换；异常与未知 usage 明确保存。

这明确改变了此前“格式错误不自动重试”的执行协议。原单格式及全局紧凑格式实验仍记为未完成，
不改写历史，不将本版称为与原单次调用完全相同的运行设置。新规则不按任务或来源 ID 切换，
也没有字符串修补、JSON 补闭合、截断、重排或人工选择 gist。

实现：[gist_generation.py](graph_construction/gist_generation.py)。服务使用 xgrammar，
原格式允许字段间空白，第二次通过已有 grammar 请求接口传入紧凑 JSON grammar。
该 grammar 由 xgrammar 0.2.3 从原结构 schema 自动生成，文件为
[gists_compact.ebnf](graph_construction/gists_compact.ebnf)，并完整写入本次 settings。

这是一次已验证的接口兼容修正：当前 vLLM 虽接受请求级 disable_any_whitespace，
但 JSON 编译路径只读取服务端开关；旧请求级尝试 6345102 没有启用紧凑格式，明确记为无效接入。
此外，当前 xgrammar 将带 minLength 的字符串编译成了不接受 JSON 转义的规则，
真实 matcher 测试中引号、反斜杠、换行的合法转义均被拒绝。
紧凑 grammar 因此只在生成约束中省略 minLength；非空字符串要求仍由原 `validate_gists` 严格执行，
不接受空字符串或空白字符串。其他结构限制不变，没有手写新的解析或内容修补规则。
独立 compiler/matcher 测试验证生成文件与结构 schema 一致、标准转义可用、错误结构被拒绝。
为处理此前读取响应时的 ReadError，本版客户端发送 `Connection: close`，不复用 HTTP 连接；
SDK 自动重试仍为零。这是传输设置，不是内容或排名规则。

## 缓存与成本

当前目录 `optimization_contextual_gists_format_retry_seed42_20260913` 允许导入原单格式调用日志，
但必须逐字段核对原生成设置、逐字核对来源输入。原有效输出直接复用，未完成来源根据已记录的
格式失败决定下一次合法尝试；不导入全局紧凑实验或诊断结果，不覆盖原目录。
原目录有 8241 个有效来源，另外两个原 MH 来源已记录原格式失败，2Wiki 尚未抽取。

每组 complete 文件记录 `imported_calls_file`。复制的日志不是新增 LLM 调用，不能把两个目录
的相同调用重复相加。新产生的请求逐条保存 serialization、原始响应、finish reason、usage 与时间。
导入历史包含此前手动重试的失败调用，全部保留并作为历史成本披露，不伪称此前也遵循本版尝试上限。

全部 14362 来源完成后才冻结全六任务索引，再执行四候选完整检索与三个 reader QA。
来源完成、输出合法和索引完成都不是语义正确或 SOTA 的证据，QA 成绩仍独立报告。

## 已知证据

- 原格式完成了 8241 个来源，但两个 MH 来源反复 length。
- 紧凑格式诊断使这两条正常生成，但全量 FCSH 出现 15 条新的 length，故不能全局替换。
- REMem 默认 json_object 接口诊断为 139/140 有效，仍有一条 length；其中 123 条输入原本受停止影响，
  140 不是自然格式失败总数。该诊断输入 156909、输出 82182 token，全部保留。
- 当前处理规则由格式有效性触发；没有使用任何 QA 得分来选择单条输出。

完整实现测试 90 项通过，包括首次成功不重试、保留空列表、预算耗尽不接受前缀、
保持 prompt 和 schema、两格式尝试上限、跨续跑失败状态，以及传输中断不触发格式切换。
另有 3 项真实 xgrammar compiler/matcher 测试在 vllm_cu129 环境通过，不依赖伪造的 SDK 响应。
