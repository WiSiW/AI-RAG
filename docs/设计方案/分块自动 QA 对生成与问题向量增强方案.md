# 分块自动 QA 对生成与问题向量增强方案
## 摘要

复用现有 `hypothetical_questions` 生成路径和 `questions` 向量召回路由，将其升级为“每个 L1/L2/L3 chunk 生成 3 个多样化 QA 对”：详情查询、比较/关系、概括各 1 个。问题单独生成向量并关联原 chunk；答案严格来自 chunk，用于管理端展示和质检，不参与向量索引或最终回答上下文。现有 Milvus `question` 表示类型与 schema 可继续复用。

## QA 生成与清洗

- 新增独立 QA 生成阶段，使用知识库 `enrichment_profile_id`，每批最多 4 个 chunk；不与关键词/摘要 Prompt 混合，Prompt 版本升级为 `qa-generation-v1`。
- Prompt 输入包含 `context_prefix`、章节路径和原始 chunk；明确文本为不可信数据，禁止补充原文外事实。
- 模型返回严格 JSON，每个 chunk 固定 3 个对象：
  `question`、`answer`、`question_type`，类型分别为 `detail`、`comparison`、`summary`。
- 问题长度 8–100 字，必须是自然疑问句；答案 1–3 句、最多 300 字，只允许引用该 chunk 中可验证的信息。
- 清洗规则：统一空白和标点后去重；拒绝重复、泛化问题、非疑问句、空答案、拒答式答案及与 chunk 无词项/实体关联的问题；同类类型最多保留 1 个。
- 若清洗后类型或数量不足，针对缺失类型定向重试一次；仍不足时使用关键词模板补齐 3 个 QA，并将 chunk 标记为 `degraded`，不阻断文档入库。
- `hypothetical_questions` 保留为兼容字段，由 QA 对的 `question` 自动投影；新增 `qa_pairs` 作为权威结构。

## 存储、索引与检索

- `document_chunks` 增加 `qa_pairs JSON`，元素为 `{question, answer, question_type}`；现有 `hypothetical_questions` 字段继续保留。
- 每个 QA 问题分别生成一个 Milvus 表示行，继续使用 `representation_type="question"` 和 `question_index=1..3`，不新增 Milvus schema。
- L1、L2、L3 都生成问题向量：L1 同时保留 `l1_summary`，L2 保留 `original/summary`，L3 保留 `l3_atomic`，随后分别追加 3 条 `question` 表示。
- 答案不生成向量、不进入 BM25，也不替代原始 chunk 送入 Rerank 或生成模型；最终回答仍以原始 chunk/上下文窗口为证据。
- `questions` 路由的候选扩取系数从当前 2 提高到 3，避免同一 chunk 的 3 个问题占据候选后挤掉其他 chunk；该路由继续按 chunk_id 去重。
- 保留问答路由现有归一化向量与 RRF；`questions` 默认权重继续为 `1.0`，可按知识库配置调整。
- 问题向量命中结果记录命中的具体问题文本，例如 `matched_questions`，供检索质检和问题生成质量分析使用。

## API、页面与存量回填

- `ChunkRead` 增加 `qa_pairs`；`hypothetical_questions` 继续返回以兼容旧前端。
- 文档预览页展示每个 chunk 的 3 个问题、答案和问题类型，并对 `degraded` QA 生成标记警告。
- 检索质检页展示 `matched_questions`，便于判断是否由“查询-生成问题”对齐带来命中。
- `IngestionJob.result` 增加 QA 对总数、按类型数量、降级 chunk 数和 QA Prompt 版本。
- `enrichment_meta` 增加 `qa_prompt_version`、`qa_count`、`qa_degraded_reasons`。
- 新文档入库自动生成并索引 3 个 QA；存量文档通过现有“重建增强索引”流程重新生成，不改变 L1/L2/L3 chunk_id 或原始正文。
- 旧 `hypothetical_questions` 在回填完成前继续参与检索；回填后由新 QA 问题覆盖问题表示，保证兼容切换。

## 测试与验收

- 单元测试验证 JSON 解析、三类问题覆盖、答案长度、精确/近似去重、泛化问题过滤和模板补齐。
- 模拟模型只返回 1–2 个有效 QA，验证定向重试、补齐至 3 个及 `degraded` 状态。
- 验证 L1/L2/L3 每个 chunk 都生成 3 条 `question` 表示，`question_index`、chunk_id 和原始正文正确。
- 测试 `questions` 路由按 3 倍候选扩取、按 chunk 去重，并保留全部命中问题。
- 验证答案不进入 Milvus、BM25、Rerank 或生成 Prompt；最终上下文仍使用原始 chunk。
- 验证重建任务可重复执行、旧文档完成 QA 回填、历史评估快照包含 QA 信息。
- 前端验证 QA 展示、检索命中的问题文本和构建通过；后端运行 Ruff、全量 pytest。

## 假设与默认值

- 每个 chunk 目标固定为 3 个 QA 对，顺序和类型不要求固定，但最终必须覆盖 `detail/comparison/summary`。
- 对比问题只能比较 chunk 内实际出现的事物；若原文没有可比内容，可改为同一文本内的关系/条件问题，但仍使用 `comparison` 类型。
- QA 答案是辅助展示和质检数据，不作为最终回答的独立事实来源。
- 复用现有增强模型，不新增模型类型；模型不可用时允许规则模板生成并标记降级。
- 现有文档需要执行一次“重建增强索引”以生成正式 QA 对和三问题向量。
