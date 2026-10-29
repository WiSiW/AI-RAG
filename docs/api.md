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
| POST | `/documents/{id}/reindex` | 重新入库 |
| DELETE | `/documents/{id}` | 删除文档与索引 |
| GET | `/ingestion-jobs/{id}` | 入库任务状态 |

分块响应额外包含 `granularity_level`（`L1/L2/L3`）、`granule_type`、`parent_chunk_id`、`context_prefix`、`summary`、`keywords`、`qa_pairs`、`hypothetical_questions`、`enrichment_status` 与 `enrichment_meta`。`qa_pairs` 元素包含 `question`、基于原文的 `answer` 和 `question_type`（`detail/comparison/summary`）。

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

检索结果额外返回 `matched_representations`、`matched_questions`、`representation_scores`、`representation_ranks`、`granularity_level`、`granule_type`、`parent_chunk_id`、`context_prefix` 与 `context_text`，用于查看命中的召回路由、具体生成问题、粒度、上下文位置和 L3 父段落关系；`vector_score/vector_rank` 继续保持为最佳向量路由的兼容字段。Search 调试接口的 `context_text` 默认等于小块正文；Chat 生成时会动态填充父窗口和邻块。

发布接口的 `knowledge_base_ids` 是授权范围。调用时可传其子集；不传时默认查询全部已授权知识库。

### 流式事件

`POST /chat` 且 `stream=true` 返回 `text/event-stream`：

```text
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

## 健康检查

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health/live` | 进程存活 |
| GET | `/health/ready` | PostgreSQL、Redis、MinIO、Milvus、BM25 检查 |
