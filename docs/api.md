# API 概览

基础路径：`/api/v1`

除登录和存活检查外，所有接口均需要：

```http
Authorization: Bearer <access_token>
```

## 认证

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/auth/login` | 管理员登录 |
| GET | `/auth/me` | 当前管理员 |

## 模型配置

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/model-profiles?kind=llm` | 模型列表 |
| POST | `/model-profiles` | 新建模型 |
| PATCH | `/model-profiles/{id}` | 修改模型 |
| DELETE | `/model-profiles/{id}` | 删除未被引用的模型 |
| POST | `/model-profiles/{id}/test` | 连通性测试 |

`kind` 支持 `embedding`、`rerank`、`llm`、`enrichment`。分块增强模型使用 OpenAI-compatible Chat Completions 接口。

LLM 与增强模型支持 `request_timeout_seconds`（默认 `1200` 秒）和 `request_max_retries`（默认 `2`，表示首次请求后的额外重试次数，最多共 3 次请求）。模型页面以分钟展示超时时间；Embedding 与 Rerank 继续使用全局请求超时。

Rerank 的高级配置支持 `rerank_path`（默认 `rerank`）、`rerank_batch_documents`（默认 `16`）和 `rerank_batch_tokens`（默认 `800`）。网关会按条数和估算 token 数自动拆批，超长单文档会切片并按最高相关分归并，避免本地 `llama-server --rerank` 因单次输入过大返回 500。

## 知识库

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/knowledge-bases` | 知识库列表 |
| POST | `/knowledge-bases` | 创建知识库 |
| GET | `/knowledge-bases/{id}` | 知识库详情 |
| PATCH | `/knowledge-bases/{id}` | 修改设置 |
| POST | `/knowledge-bases/{id}/rebuild-index` | 异步重建增强与 v2 多向量索引 |
| DELETE | `/knowledge-bases/{id}` | 删除知识库及数据 |

创建知识库时必须提供 `embedding_profile_id` 与 `enrichment_profile_id`。`settings.retrieval_weights` 支持 `bm25`、`l1`、`original`、`summary`、`questions`、`l3` 六个非负权重；默认分别为 `1.0/0.8/1.0/1.0/1.0/0.6`。

## 文档

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/knowledge-bases/{id}/documents` | multipart 上传 |
| GET | `/knowledge-bases/{id}/documents` | 文档列表 |
| GET | `/documents/{id}` | 文档详情 |
| GET | `/documents/{id}/content` | 提取文本 |
| GET | `/documents/{id}/chunks` | 分块列表 |
| POST | `/documents/{id}/reindex` | 重新入库 |\n| POST | `/knowledge-bases/{id}/documents/reparse-references` | 批量重建 ready 文档的正文/引用过滤 |
| DELETE | `/documents/{id}` | 删除文档与索引 |
| GET | `/ingestion-jobs/{id}` | 入库任务状态 |

分块响应额外包含 `granularity_level`（`L1/L2/L3`）、`granule_type`、`parent_chunk_id`、`context_prefix`、`summary`、`keywords`、`qa_pairs`、`hypothetical_questions`、`enrichment_status` 与 `enrichment_meta`。`qa_pairs` 元素包含 `question`、基于原文的 `answer` 和 `question_type`（`detail/comparison/summary`）。

文档入库会先分离正文与引用。文末参考文献、脚注、尾注及正文引用标记不会生成 chunk、Embedding 或 BM25 数据；引用条目保存在 `references_json`，只通过管理端引用参考弹窗展示，弹窗支持关键词搜索和分页翻页。文档详情和内容响应包含 `reference_count`、`references` 与 `reference_extraction`，其中 `reference_extraction.status=degraded` 表示歧义边界已保守保留为正文或 LLM 兜底不可用。

批量重建引用过滤接口只排队状态为 `ready` 的文档，返回 `queued`、`skipped` 和 `job_ids`；该操作会重新分块、增强并建立索引，页面必须二次确认。

上传字段：

- `file`：PDF、DOCX、MD、MARKDOWN、EPUB
- `chunk_size`：可选
- `chunk_overlap`：可选
- `tags`：可选，逗号分隔

## 问答与检索

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/chat` | 流式或非流式问答 |
| POST | `/search` | 检索调试 |
| GET | `/conversations` | 历史会话列表 |
| GET | `/conversations/{id}` | 会话详情 |
| DELETE | `/conversations/{id}` | 删除会话 |
| POST | `/feedback` | 回答反馈 |

## 分块语义完整性评估

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/chunk-evaluations` | 创建文档或知识库级评估任务 |
| GET | `/chunk-evaluations` | 查询评估历史 |
| GET | `/chunk-evaluations/{id}` | 查询运行、进度和汇总 |
| GET | `/chunk-evaluations/{id}/results` | 按粒度、等级、问题标签分页查询 chunk 结果 |
| PATCH | `/chunk-evaluations/{id}/results/{result_id}` | 提交人工评分、标签和备注 |
| GET | `/chunk-evaluations/compare?run_ids=id1,id2` | 比较两次运行 |
| DELETE | `/chunk-evaluations/{id}` | 删除历史运行 |

创建参数：

- `document_id` 与 `knowledge_base_id` 必须且只能提供一个。
- `granularities`：可选 `L1/L2/L3`，默认全部。
- `judge_profile_id`：可选，默认使用知识库增强模型；支持启用的 LLM 或增强模型。
- `llm_enabled`：默认 `true`；关闭时只执行规则评估。

## 分块大小与重叠窗口调优

### 检索评测集

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/retrieval-evaluation-datasets` | 按知识库查询评测集 |
| POST | `/retrieval-evaluation-datasets` | 创建单知识库评测集 |
| GET | `/retrieval-evaluation-datasets/{id}` | 查询评测集和问题 |
| PATCH | `/retrieval-evaluation-datasets/{id}` | 修改名称、说明或文档范围 |
| DELETE | `/retrieval-evaluation-datasets/{id}` | 删除评测集，历史运行快照保留 |
| POST | `/retrieval-evaluation-datasets/{id}/cases` | 新增问题、标准答案和黄金证据 |
| PATCH | `/retrieval-evaluation-datasets/{id}/cases/{case_id}` | 替换问题和证据 |
| DELETE | `/retrieval-evaluation-datasets/{id}/cases/{case_id}` | 删除问题 |
| POST | `/retrieval-evaluation-datasets/{id}/import` | 以 JSONL 原子导入问题 |
| GET | `/retrieval-evaluation-datasets/{id}/export` | 导出 UTF-8 JSONL |

创建评测集的 `document_ids` 必须属于同一知识库且文档状态为 `ready`。问题类型支持
`detail/comparison/summary/reasoning/table/code/no_answer`；除 `no_answer` 外必须提供标准答案和至少一条黄金证据。

黄金证据字段：

- `document_id`：证据所在文档。
- `quote`：从原文划选的最小充分片段，不超过 300 token。
- `section_path`：章节路径，默认“正文”。
- `page_start`、`page_end`：可选页码。

服务端会忽略空白差异定位 `quote` 并保存字符区间；无法在原文定位的证据会被拒绝。

JSONL 第一行可为数据集元信息，后续每行一条问题，所有行均带 `schema_version: 1`。任何一行校验失败时整批不导入。

### 调优运行

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/chunk-tuning-runs` | 创建快速筛选运行 |
| GET | `/chunk-tuning-runs` | 查询运行历史 |
| GET | `/chunk-tuning-runs/{id}` | 查询运行、进度、聚合和 trial |
| GET | `/chunk-tuning-runs/{id}/trials` | 查询 trial |
| GET | `/chunk-tuning-runs/{id}/results` | 分页查询逐题检索/答案结果 |
| POST | `/chunk-tuning-runs/{id}/verifications` | 对勾选 trial 创建深度验证 |
| POST | `/chunk-tuning-runs/{id}/cancel` | 请求取消运行 |
| POST | `/chunk-tuning-runs/{id}/apply` | 保存 trial 参数为知识库默认值 |
| DELETE | `/chunk-tuning-runs/{id}` | 删除运行并清理临时索引 |

创建快速筛选：

```json
{
  "knowledge_base_id": "kb-id",
  "dataset_id": "dataset-id",
  "chunk_sizes": [256, 512, 1024, 1536],
  "overlap_ratios": [0, 10, 20, 25]
}
```

重叠按 `round(chunk_size × ratio / 100)` 转为实际 token 数，矩阵最多 25 组。运行要求至少 20 条可回答问题；建议使用 50–100 条分层数据。系统额外保留当前知识库参数作为基线 trial。

快速筛选使用隔离的 L2 原文向量和 BM25，固定单跳、关闭 Rerank；深度验证使用完整 L1/L2/L3 多表示索引、固定 RRF/Rerank/LLM 及每候选最多 30 条分层答案评测。临时索引命名空间为 `eval-*`，不会写入生产 Milvus/Whoosh。

`apply` 只接受深度验证中状态为 `completed` 的 trial，仅写回：

```json
{
  "chunk_mode": "structure",
  "chunk_size": 512,
  "chunk_overlap": 64,
  "last_chunk_tuning": {"run_id": "...", "trial_id": "...", "applied_at": "..."}
}
```

不会自动重分块或重建现有文档。

### 显式重建现有文档

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/knowledge-bases/{kb_id}/chunk-settings/reindex` | 更新选中文档参数并批量排队重建 |

请求字段为 `document_ids`、`chunk_mode`、`chunk_size`、`chunk_overlap`。接口返回 `queued`、`skipped` 和 `job_ids`；处理中的文档会跳过。该操作会替换文档的生产分块和索引，页面必须二次确认。

## 知识库导出与导入

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/knowledge-bases/{id}/exports` | 创建知识库导出任务，`mode=documents|full` |
| GET | `/knowledge-base-transfers/{job_id}` | 查询导入/导出任务状态与进度 |
| GET | `/knowledge-base-transfers/{job_id}/download` | 下载已完成的导出包 |
| DELETE | `/knowledge-base-transfers/{job_id}` | 删除已结束任务和包文件 |
| POST | `/knowledge-base-imports/inspect` | 上传 ZIP 并校验 manifest、文件哈希和包结构 |
| POST | `/knowledge-base-imports` | 上传 ZIP 并创建新知识库 |

导入 multipart 字段：

- `package`：知识库 ZIP。
- `embedding_profile_id`、`enrichment_profile_id`：必填目标模型。
- `rerank_profile_id`、`llm_profile_id`：可选。
- `name`：可选的新知识库名称；冲突时自动加“（导入 N）”。
- `index_policy`：`strict` 或 `rebuild`；文档包始终使用 `rebuild`。

完整包严格恢复要求目标 Embedding 的 `provider`、`model_name`、`dimensions` 与包 manifest 完全一致。

多知识库请求使用 `knowledge_base_ids`：

```json
{
  "knowledge_base_ids": ["kb-1", "kb-2"],
  "query": "两个制度之间有什么差异？",
  "llm_profile_id": "llm-1",
  "stream": true
}
```

兼容规则：

- 旧的 `knowledge_base_id` 仍可用于单知识库请求。
- `knowledge_base_id` 与 `knowledge_base_ids` 不能同时传递。
- 多知识库请求不支持通过 `document_id` 限定单个文档。
- 会话会固定本次选择的知识库范围；更改范围需要新建会话。
- `retrieval_mode` 支持 `auto`、`single`、`multi`，默认 `auto`。

会话列表与详情响应在兼容保留 `summary` 字符串的同时，额外返回 `summary_status`、`summary_version`、`summary_message_count` 和 `summary_updated_at`。`summary_status` 取值为 `idle/running/succeeded/degraded/failed`。摘要由独立 Celery 任务异步更新，不阻塞 Chat 返回；达到未摘要消息数或 token 阈值后才触发，流式和非流式请求均不会同步等待摘要完成。

Chat `usage` 额外包含 `history_tokens_estimate`、`summary_tokens_estimate`、`history_truncated`、`query_tokens_estimate` 和 `prompt_tokens_estimate`。历史窗口与结构化会话摘要均受 token 预算限制；历史摘要只用于理解指代、用户目标和约束，事实性结论仍必须由本轮参考资料支撑并标注引用。

检索结果额外返回 `matched_representations`、`matched_questions`、`representation_scores`、`representation_ranks`、`granularity_level`、`granule_type`、`parent_chunk_id`、`context_prefix` 与 `context_text`，用于查看命中的召回路由、具体生成问题、粒度、上下文位置和 L3 父段落关系；`vector_score/vector_rank` 继续保持为最佳向量路由的兼容字段。Search 调试接口的 `context_text` 默认等于小块正文；Chat 生成时会动态填充父窗口和邻块。

Chat 和 Search 响应还返回 `retrieval_trace`，包含复杂度判定、每轮子问题、新增证据数、停止原因及不确定状态。Chat 的 `uncertainty_notice` 非空时，同一提示会追加在最终回答末尾。

发布接口的 `knowledge_base_ids` 是授权范围。调用时可传其子集；不传时默认查询全部已授权知识库。

### 流式事件

`POST /chat` 且 `stream=true` 返回 `text/event-stream`：

```text
event: stage
data: {"stage":"retrieving","conversation_id":"...","context_count":0}

event: retrieval_plan
data: {"is_multi_hop":true,"decision_source":"bert","rule_score":0.4,"bert_probability":0.87}

event: retrieval_step
data: {"round":1,"query":"...","new_chunk_count":8,"key_entities":["..."],"need_more":true}

event: retrieval_done
data: {"stop_reason":"planner_sufficient","uncertain":false,"total_rounds":2,"total_chunks":13}

event: stage
data: {"stage":"generating","conversation_id":"...","context_count":5}

event: start
data: {"conversation_id":"...","knowledge_base_ids":["kb-1","kb-2"],"rewritten_query":"...","context_count":5}

event: token
data: {"content":"回答片段"}

event: citations
data: {"citations":[{"index":1,"knowledge_base_id":"kb-1","knowledge_base_name":"制度库","source":"制度.pdf","section":"第一章 > 1.1"}]}

event: done
data: {"conversation_id":"...","message_id":"...","usage":{...}}
```

错误事件：

```text
event: error
data: {"message":"错误详情"}
```

首个 `stage` 事件在查询改写和检索开始前发送，用于确认会话已经创建并提示当前处于检索阶段；检索结束后、首个回答 token 前发送 `stage=generating`。客户端可通过关闭 SSE 连接停止当前问答。服务端会取消后续检索轮次和模型流，并将已得到的部分回答持久化，助手消息的 `usage` 中会包含 `stopped=true` 和 `stopped_stage=retrieving|generating`。已停止的助手回答仍会出现在会话历史中，但不会进入后续对话的模型上下文或会话摘要；用户问题会以“未完成提问”保留。

## 健康检查

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health/live` | 进程存活 |
| GET | `/health/ready` | PostgreSQL、Redis、MinIO、Milvus、BM25 检查 |
