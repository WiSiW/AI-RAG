# PDF链路

```
前端上传 PDF
  → POST /api/v1/knowledge-bases/{kb_id}/documents
  → 校验知识库写锁、文件扩展名、大小和分块参数
  → 临时写入本地文件
  → 上传原始 PDF 到 MinIO
  → PostgreSQL 创建 Document(PENDING) + IngestionJob(PENDING)
  → 投递 Celery: zhiyuan.ingest_document
  → 返回 201，此时仅表示“已入队”

Celery Worker
  → Document(PROCESSING) / Job(RUNNING)
  → 从 MinIO 下载 PDF 到临时文件
  → PDFParser(PyMuPDF) 提取文本和页面信息
  → 生成文档级元数据
  → split_document 分块，默认 structure 模式
  → 保存 L2 原始 chunk
  → 关键词、摘要、QA 增强
  → 构建 L1/L3 多粒度块
  → 生成上下文前缀和多表示向量
  → 写入 Milvus / Whoosh
  → 保存 extracted_text
  → Document(READY) / Job(SUCCEEDED)
```

## 1. 上传阶段
入口是 [documents.py](D:/wsw/AI-RAG/backend/app/api/documents.py) 中的 POST /api/v1/knowledge-bases/{kb_id}/documents。
- 先检查知识库是否存在，并检查是否处于导出等写锁状态。
- 通过文件扩展名白名单识别格式，.pdf 映射到 PDF 解析器。
- 文件按 1MB 分段读取，默认最大 2048 MB；空文件直接拒绝。
- 分块参数处理顺序：
  - 请求参数优先；
  - 其次使用知识库 settings；
  - 默认 chunk_mode=structure、chunk_size=512、chunk_overlap=64。
- 原始文件上传到 MinIO，对象键格式类似：
  documents/<uuid>/<filename>。
- 数据库创建：
  - Document.status = PENDING
  - IngestionJob.status = PENDING
- 投递 Celery 任务 zhiyuan.ingest_document(document_id, job_id)。
- 队列不可用时，文档和任务直接标记为 FAILED。
因此，上传接口返回成功不代表 PDF 已完成解析，前端还需要轮询 GET /api/v1/ingestion-jobs/{job_id} 或文档状态。

Worker 入口在 [tasks.py](D:/wsw/AI-RAG/backend/app/workers/tasks.py)，失败时最多自动重试 2 次。

## 2. PDF 文本解析
PDF 解析器位于 [pdf_parser.py](D:/wsw/AI-RAG/backend/app/parsing/pdf_parser.py)。

具体行为：
1. 使用 pymupdf.open(file_path) 打开 PDF。
2. 逐页调用 page.get_text("dict")。
3. 只处理文本块 block.type == 0，图片块直接跳过。
4. 每个文本行内的多个 span：
   - 使用 "".join(...) 直接拼接；
   - 去除首尾空白；
   - 空行丢弃。
5. 取每个 span 的最大字号作为该行字号。
6. 计算全文所有有效行的字号中位数，作为正文字号基准。
7. 根据字号相对正文基准判断标题层级：
   - 大于等于 1.35x：一级标题；
   - 大于等于 1.2x：二级标题；
   - 大于等于 1.08x：三级标题；
   - 其他：普通段落。
8. 每个文本行生成一个 ParsedElement，并附加从 1 开始的页码：
   - type=title/paragraph
   - level=0/1/2/3
   - metadata={"page": page_number}
9. full_text 是所有元素文本按换行拼接后的结果。
同时提取 PDF 元数据：
- 页数、PDF 版本；
- 是否加密；
- 目录项数量；
- 标题、作者、主题、关键词；
- 创建者、生成器；
- 创建时间和修改时间；
- PDF 原始 metadata 字典。
之后由 [metadata.py](D:/wsw/AI-RAG/backend/app/parsing/metadata.py) 补充：
- SHA-256；
- 文件大小、MIME 类型；
- 标题；
- 中英文语言判断；
- 字符数；
- 元素数量和标题列表。
PDF 解析结果最终写入 Document.metadata_json 和 Document.extracted_text。

## 3. 分块阶段
解析完成后，[ingestion_service.py](D:/wsw/AI-RAG/backend/app/services/ingestion_service.py) 调用 [modes.py](D:/wsw/AI-RAG/backend/app/chunking/modes.py) 的 split_document()。

默认 structure 模式：
1. 根据 PDF 解析器识别出的标题层级维护章节栈。
2. 将内容按章节聚合，生成类似：
   第一章 > 安装 > 连接电源 的章节路径。
3. 章节内容再按 token 窗口聚合，尽量保持完整行和段落边界。
4. 超过 chunk_size 的长文本按 token 精确切分，并带 chunk_overlap。
5. 每个 chunk 保存：
   - 正文；
   - section_path；
   - token 数量；
   - 页码集合；
   - 分块模式等元数据。
还支持其他模式：
- fixed：固定 token 窗口；
- recursive：递归字符边界；
- paragraph：段落聚合；
- sentence：句子聚合；
- semantic：调用 Embedding 判断相邻句子语义边界；
- parent_child：子块召回、父块上下文补偿。
处理或重新入库文档时，会先删除该文档原有 DocumentChunk，再以新 UUID 创建 chunk。

## 4. 增强与多粒度构建
初始分块结果首先作为 L2 常规语义块保存，随后继续执行：
1. L2 关键词、摘要和兼容问题字段增强。
2. 按章节构建 L1 章节摘要块。
3. 从 L2 中抽取关键句、表格行、参数行、错误行等 L3 原子块。
4. 为 L1/L2/L3 生成 3 个 QA 对。
5. 为块生成上下文前缀：
   [文档: ... / 章节: ... / 上级摘要: ...]。
6. 为 L1/L2/L3 生成对应表示向量：
   - L2：original
   - L2：summary
   - L1：l1_summary
   - L3：l3_atomic
   - 每个 QA 问题：question
7. Milvus 写入表示向量；Whoosh 写入 L1/L2，L3 不进入 BM25。


## 5. 完成与失败处理
全部成功时：
- Document.status = READY
- Document.extracted_text = parsed.full_text
- IngestionJob.status = SUCCEEDED
- Job 结果记录 chunk 数、L1/L2/L3 数量、字符数、增强情况、QA 数量、表示向量数和增强模型。
任一步骤失败时：
- 删除已写入的 Milvus v1/v2 和 Whoosh v1/v2 数据；
- 删除该文档已生成的 chunk；
- Document 和 IngestionJob 标记为 FAILED；
- 删除临时文件；
- 失败原因写入 error。
当前 PDF 链路的主要边界
- 使用 PyMuPDF 原生文本层提取，没有 OCR，扫描版或纯图片 PDF 通常无法得到正文。
- 不支持输入密码解密加密 PDF，只记录 is_encrypted 元数据。
- 没有专门的表格结构识别，表格通常被拆成普通文本行，列关系和单元格边界可能丢失。
- 标题依赖字号启发式判断，没有优先使用 PDF 目录/书签恢复章节层级；目前只记录 TOC 项数量。
- 文本按行生成元素，复杂双栏 PDF 的阅读顺序依赖 PyMuPDF 默认文本块顺序。
- 上传侧主要按扩展名白名单校验，未做 PDF 魔数或深度 MIME 校验。