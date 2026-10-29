# 文档与 Chunk 元数据说明

本文档描述当前系统中 `Document`、`DocumentChunk` 的固定字段及其 `metadata_json` 扩展字段。`metadata_json` 是无固定 Schema 的 JSON，实际字段会随文档格式、上传来源和 chunk 粒度变化。

## 1. Document 固定字段

来源：`documents` 表。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | `VARCHAR(36)` | 文档 ID |
| `kb_id` | `VARCHAR(36)` | 所属知识库 ID |
| `filename` | `VARCHAR(500)` | 原始文件名 |
| `object_key` | `VARCHAR(1000)` | MinIO 对象键，全局唯一 |
| `content_type` | `VARCHAR(200)` | MIME 类型 |
| `size_bytes` | `INTEGER` | 文件大小，单位字节 |
| `status` | `VARCHAR(20)` | `pending`、`processing`、`ready`、`failed` |
| `extracted_text` | `TEXT` | 文档解析后的完整文本 |
| `error` | `TEXT` | 最近一次入库失败信息 |
| `tags` | `JSON` | 用户上传时填写的标签数组 |
| `metadata_json` | `JSON` | 扩展元数据 |
| `chunk_mode` | `VARCHAR(40)` | 当前文档使用的分块模式 |
| `chunk_size` | `INTEGER` | 分块大小 |
| `chunk_overlap` | `INTEGER` | 分块重叠 token 数 |
| `created_at` | `DATETIME` | 创建时间 |
| `updated_at` | `DATETIME` | 更新时间 |

## 2. Document.metadata_json 通用字段

通用字段由 `build_document_metadata()` 生成：

| 字段 | 说明 |
| --- | --- |
| `format` | 解析格式，如 `pdf`、`docx`、`markdown`、`epub` |
| `title` | 文档标题；优先使用解析器识别的标题，否则使用文件名 |
| `extension` | 文件扩展名，包含前导点，如 `.pdf` |
| `mime_type` | MIME 类型 |
| `file_size` | 临时文件大小，单位字节 |
| `modified_at` | 文件修改时间，UTC ISO 8601 |
| `sha256` | 文件内容 SHA-256 |
| `language` | 按提取文本检测出的语言：`zh`、`en`、`unknown` |
| `characters` | 解析元素文本总字符数 |
| `elements` | 解析元素数量 |
| `headings` | 标题数组，最多保留 200 个 |

### 上传来源字段

创建文档记录时额外写入：

| 字段 | 说明 |
| --- | --- |
| `upload_source` | `browser`、`archive_upload`、`server_archive` |
| `archive_name` | 压缩包文件名；仅压缩包导入存在 |
| `archive_path` | 文档在压缩包内的相对路径；仅压缩包导入存在 |

## 3. 格式专属字段

### 3.1 PDF

| 字段 | 说明 |
| --- | --- |
| `format` | 固定为 `pdf` |
| `pages` | PDF 页数 |
| `page_count` | PDF 页数，与 `pages` 相同 |
| `pdf_version` | PDF 版本 |
| `is_encrypted` | 是否加密 |
| `toc_items` | 目录项数量 |
| `author` | 作者 |
| `subject` | 主题 |
| `keywords` | PDF 关键词元数据 |
| `creator` | 创建工具 |
| `producer` | PDF 生成器 |
| `creation_date` | PDF 元数据中的创建时间 |
| `modified_date` | PDF 元数据中的修改时间 |
| `pdf_metadata` | PyMuPDF 读取到的原始字符串键值对 |

### 3.2 DOCX

| 字段 | 说明 |
| --- | --- |
| `format` | 固定为 `docx` |
| `author` | 作者 |
| `last_modified_by` | 最后修改人 |
| `subject` | 主题 |
| `keywords` | 文档关键词 |
| `category` | 分类 |
| `comments` | 备注 |
| `identifier` | 文档标识 |
| `language` | 文档属性中的语言 |
| `revision` | 修订号 |
| `created` | 文档创建时间 |
| `modified` | 文档修改时间 |
| `last_printed` | 最后打印时间 |
| `paragraph_count` | 非表格元素数量 |
| `table_count` | 表格数量 |
| `section_count` | DOCX section 数量 |

### 3.3 Markdown

| 字段 | 说明 |
| --- | --- |
| `format` | 固定为 `markdown` |
| `front_matter` | YAML front matter 解析结果 |
| `author` | `front_matter.author` |
| `description` | `front_matter.description` |
| `tags` | `front_matter.tags`；不同于 Document 固定字段 `tags` |
| `heading_count` | 标题数量 |
| `code_block_count` | 代码块数量 |
| `table_count` | 表格数量 |
| `list_count` | 列表数量 |
| `word_count` | 按中文单字和英文/数字词粗略统计的词数 |

### 3.4 EPUB

| 字段 | 说明 |
| --- | --- |
| `format` | 固定为 `epub` |
| `sections` | 解析出的文档 section 数量 |
| `authors` | 作者数组 |
| `language` | EPUB Dublin Core 语言 |
| `publisher` | 出版社 |
| `identifier` | 出版物标识 |
| `date` | 出版日期 |
| `description` | 描述 |
| `subjects` | 主题数组 |
| `chapter_count` | 章节数量 |

## 4. metadata_json 合并规则

入库时分为两个步骤：

1. 上传阶段先写入上传来源和压缩包来源字段。
2. 解析阶段调用 `build_document_metadata()`，生成通用字段和格式专属字段。
3. 最终以以下方式合并：

```python
document.metadata_json = {
    **(document.metadata_json or {}),
    **parsed_metadata,
}
```

因此解析阶段产生的同名字段会覆盖上传阶段字段。通用字段先写入，随后展开 `**parsed.metadata`，所以解析器返回的 `format`、`language` 等字段可覆盖通用默认值。

## 5. DocumentChunk 固定字段

来源：`document_chunks` 表。

| 字段 | 说明 |
| --- | --- |
| `id` | Chunk ID |
| `document_id` | 所属文档 ID |
| `kb_id` | 所属知识库 ID |
| `chunk_index` | 文档内全局序号 |
| `content` | 原始 chunk 正文，不被上下文前缀修改 |
| `section_path` | 章节路径 |
| `token_count` | token 数 |
| `metadata_json` | Chunk 扩展元数据 |
| `granularity_level` | `L1`、`L2`、`L3` |
| `granule_type` | `section_summary`、`paragraph`、`table_row`、`error_line`、`sentence` 等 |
| `parent_chunk_id` | L3 对应的父 L2 chunk；L1/L2 通常为空 |
| `context_prefix` | 索引时注入的文档、章节、上级摘要前缀 |
| `summary` | Chunk 摘要 |
| `keywords` | BM25 友好关键词数组 |
| `hypothetical_questions` | 由 3 个 QA 问题投影出的兼容字段 |
| `qa_pairs` | 3 个 QA 对数组：`question`、`answer`、`question_type` |
| `enrichment_status` | `pending`、`ready`、`degraded` |
| `enrichment_meta` | 增强模型、Prompt 版本、降级原因等 |
| `created_at` | 创建时间 |

`qa_pairs` 固定保存 3 个对象，`question_type` 覆盖 `detail`、`comparison`、`summary`：

```json
[
  {"question": "安装前需要关闭电源吗？", "answer": "需要，连接电源前必须关闭设备。", "question_type": "detail"},
  {"question": "关闭设备与检查指示灯有什么关系？", "answer": "断电是连接电源的前提，连接后再检查指示灯。", "question_type": "comparison"},
  {"question": "安装流程的核心要点是什么？", "answer": "先断电，再连接电源并检查指示灯。", "question_type": "summary"}
]
```

`hypothetical_questions` 是 `qa_pairs.question` 的兼容投影。只有问题生成向量；答案不进入 Milvus、BM25、Rerank 或最终回答上下文。

`enrichment_meta` 中 QA 字段包括 `qa_generated_by`、`qa_model_profile_id`、`qa_model_name`、`qa_prompt_version`、`qa_count`、`qa_types` 和 `qa_degraded_reasons`。

## 6. Chunk.metadata_json 常见字段

### 通用与分块字段

| 字段 | 说明 |
| --- | --- |
| `pages` | Chunk 覆盖的页码数组 |
| `chunk_mode` | 生成时的分块模式 |
| `split` | 是否由 token window 等策略拆分 |
| `parent_index` | `parent_child` 模式中的父块序号 |
| `parent_section` | 父块所属章节 |
| `parent_content` | `parent_child` 模式保存的父块正文 |
| `child_content` | 子块正文快照 |

### L1 章节摘要字段

| 字段 | 说明 |
| --- | --- |
| `granularity_level` | 固定为 `L1` |
| `parent_chunk_ids` | 组成该章节摘要的 L2 chunk ID 数组 |
| `pages` | 章节覆盖页码 |
| `summary_chars` | 摘要字符数 |
| `source_tokens` | 生成摘要所用章节源文本 token 数 |

### L3 原子块字段

| 字段 | 说明 |
| --- | --- |
| `granularity_level` | 固定为 `L3` |
| `parent_chunk_id` | 父 L2 chunk ID |
| `parent_chunk_index` | 父 L2 chunk 序号 |
| `pages` | 原段落页码 |
| `element_type` | 表格行、错误行、参数行、代码行或句子 |
| `selection_reason` | L3 抽取原因 |

## 7. 示例

### 7.1 PDF Document.metadata_json

```json
{
  "format": "pdf",
  "title": "设备安装手册",
  "extension": ".pdf",
  "mime_type": "application/pdf",
  "file_size": 2453100,
  "modified_at": "2026-10-29T08:00:00+00:00",
  "sha256": "9f86d081884c7d659a2feaa0c55ad015...",
  "language": "zh",
  "characters": 32580,
  "elements": 420,
  "headings": ["第一章 安装准备", "第二章 连接电源"],
  "upload_source": "browser",
  "pages": 38,
  "page_count": 38,
  "pdf_version": 1.7,
  "is_encrypted": false,
  "toc_items": 12,
  "author": "设备厂商",
  "subject": "设备安装",
  "keywords": "安装,电源,故障码",
  "creator": "Word",
  "producer": "PDF exporter",
  "creation_date": "2026-10-01T10:00:00Z",
  "modified_date": "2026-10-20T10:00:00Z",
  "pdf_metadata": {}
}
```

### 7.2 L3 Chunk.metadata_json

```json
{
  "granularity_level": "L3",
  "parent_chunk_id": "c7ba5f50-4f90-4fe8-9fd0-0123456789ab",
  "parent_chunk_index": 12,
  "pages": [18],
  "element_type": "error_line",
  "selection_reason": "error_or_exception"
}
```

### 7.3 压缩包导入附加字段

```json
{
  "upload_source": "archive_upload",
  "archive_name": "manuals.zip",
  "archive_path": "installation/manual-v3.pdf"
}
```

## 8. API 中的对应字段

- `DocumentRead.metadata_json`：返回文档扩展元数据。
- `DocumentDetail.metadata_json`：同上，并额外返回 `extracted_text`。
- `ChunkRead`：返回 Chunk 固定字段、增强字段和 Chunk `metadata_json`。
- `SearchResult.metadata`：返回命中时的路由、增强和上下文扩展信息。

## 9. 实现位置

- `backend/app/db/models.py`
- `backend/app/parsing/metadata.py`
- `backend/app/parsing/pdf_parser.py`
- `backend/app/parsing/docx_parser.py`
- `backend/app/parsing/markdown_parser.py`
- `backend/app/parsing/epub_parser.py`
- `backend/app/api/documents.py`
- `backend/app/services/ingestion_service.py`
- `backend/app/chunking/multigranularity.py`
