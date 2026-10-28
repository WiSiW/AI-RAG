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

## 知识库

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/knowledge-bases` | 知识库列表 |
| POST | `/knowledge-bases` | 创建知识库 |
| GET | `/knowledge-bases/{id}` | 知识库详情 |
| PATCH | `/knowledge-bases/{id}` | 修改设置 |
| DELETE | `/knowledge-bases/{id}` | 删除知识库及数据 |

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
