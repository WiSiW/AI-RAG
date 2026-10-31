# PDF OCR、复杂版面与表格语义解析方案

## 摘要

新增独立的 PDF 解析流水线：文本页直接提取，扫描页或乱码页按优先级执行 OCR；通过独立版面分析服务识别标题、正文、表格、图片和页眉页脚；表格按页特征路由到 pdfplumber/Camelot/Tabula，并保存完整结构化数据。PDF 自动采用版面感知分块，表格正文永不拆行；检索命中后按 token 预算附回完整表格或表格自然语言摘要。

新增独立 `pdf_parse` Celery 队列、解析引擎管理页面和表格数据接口。非 PDF 的解析与分块行为保持不变，存量 PDF 仅通过管理员手动重解析升级。

## PDF 解析与引擎体系

- 按页处理 PDF，不整篇固定走 OCR：
  - 原生文本充足且质量正常时使用 PyMuPDF，不调用 OCR。
  - 文本缺失、图片占比高或乱码时，渲染页面并调用 OCR 引擎链。
  - 混合页保留正常原生文本，仅对无文本的图片/表格区域 OCR，并按版面阅读顺序合并、去重。
- 固定 OCR 内部协议：
  - 输入：页面或区域图像、页码、DPI、语言、版面提示。
  - 输出：引擎 ID、文本块列表及 `text/bbox/confidence/block_type`、全文、平均置信度、耗时和警告。
  - `bbox` 统一使用页面归一化坐标 `[x0,y0,x1,y1]`。
- 默认 OCR 顺序为 RapidOCR → Tesseract，均提供 CPU 适配器；另提供通用 HTTP OCR 适配器，可配置远程服务。
- 每个引擎失败后继续下一个；所有引擎输出都先经过质量检查，再决定是否接受。默认判定条件：有效字符数不少于 20、置信度不低于 0.55、替换字符/私有字符/控制字符占比不超过 1%、重复 n-gram 占比不超过 35%。阈值可配置。
- 版面分析通过独立 HTTP 服务完成，使用 LayoutParser + Detectron2 兼容模型，返回归一化坐标、区域类型、置信度和阅读顺序。服务不可用时回退到 PyMuPDF/pdfplumber 的字体、坐标、文本密度和线条规则。
- 页眉、页脚和页码默认从 chunk 正文中剔除，但保留坐标和页面元数据。标题、正文、图片、图注、表格按阅读顺序组织。
- 表格候选区域同时来自版面模型和 pdfplumber 线条检测。表格引擎按管理员配置的顺序尝试：
  - pdfplumber 默认首选，处理带线表格和普通规则表格。
  - Camelot 在检测到明显网格时优先使用 lattice，无边框表格使用 stream。
  - Tabula 作为后备，仅在原生文本 PDF 上启用。
  - 扫描表格跳过 Camelot/Tabula，改用 OCR 文本坐标和检测到的行列线重建单元格，并标记较低置信度。
- 表格结果统一保存合并单元格的 `row/col/rowspan/colspan` 信息，并生成 headers、rows、Markdown 和自然语言描述。描述复用知识库 `enrichment_profile_id`，Prompt 版本为 `pdf-table-summary-v1`；模型失败时生成确定性模板描述。
- 每页保存引擎尝试链、质量分、选用引擎、OCR 状态和失败原因。部分页面成功时文档继续入库并标记 `degraded`；存在原生文本但 OCR 全失败时保留原生文本并降级；纯扫描 PDF 没有任何可用文本时文档失败。

## 存储、分块与检索

- 新增 `document_tables`：文档/KB 外键、页码、归一化 bbox、标题、合并单元格结构、headers、rows、Markdown、自然语言描述、提取引擎、质量分、质量详情和原始块 ID。表格 ID 写入对应 chunk 的 `metadata_json.table_ids`。
- 新增 `parsing_engine_configs`：类别 `ocr/table/layout`、适配器类型、显示名称、优先级、启停状态、参数 JSON。禁止在配置中保存明文密钥，HTTP 鉴权只允许引用环境变量名。
- PDF 的 `structure` 模式自动切换为版面感知分块；显式选择其他分块模式时尊重原模式。
- 版面感知分块优先把标题、正文、图注和小表格放在同一 chunk；表格作为原子块，不把表头和数据行切到两个 L2 chunk。
- 超大表格仍保持完整表格记录，不拆行。L2 表格 chunk 只索引标题、表头、行数、列结构和自然语言摘要；L3 可生成引用完整父表的 `table_row` 原子块，但父表始终完整可回取。
- Chunk 元数据增加页码、归一化 bbox、版面块 ID、块类型、OCR 引擎、OCR 置信度、`table_ids` 和降级原因。
- 检索回表后查询关联表格，在现有上下文窗口阶段追加 `[关联表格]` 和 Markdown。默认表格上下文预算 1000 token；完整表超预算时改为说明加自然语言摘要，并去重同一表格。
- Milvus schema、RRF、Rerank 和父窗口机制保持不变；表格 Markdown 作为普通 L2 内容进入 BM25/向量索引，表格描述进入 summary 表示。
- `SearchResult` 增加本次注入的表格摘要列表，供检索质检展示。
- 知识库导出包增加 `tables.jsonl`，manifest 版本升级并记录表格数量；导入时重映射表格 ID 和 chunk 中的 `table_ids`。旧包仍可导入，新包严格模式与 rebuild 模式都保留表格数据。

## API、页面与部署

- 新增引擎管理接口：`GET/POST /api/v1/parsing-engines`、`PATCH/DELETE /api/v1/parsing-engines/{id}`、`POST /api/v1/parsing-engines/reorder`、`POST /api/v1/parsing-engines/{id}/test` 和适配器能力查询接口。
- 新增 `GET /api/v1/documents/{id}/tables` 与单表详情接口；现有文档重处理接口对 PDF 改用新流水线，并增加知识库批量重解析 PDF 接口。
- `IngestionJob.result.pdf_pipeline` 记录总页数、原生页数、OCR 页数、失败页数、各引擎使用次数、版面服务状态、表格数、降级原因和耗时。
- 新增“解析引擎”管理页，按 OCR、表格、版面三类展示顺序、启停、参数、健康状态和测试结果；支持新增/删除 HTTP 引擎实例、调整已有引擎优先级。
- 文档预览页展示 OCR 状态、引擎、失败页、版面服务状态和表格；表格详情展示 Markdown、描述、页码、来源引擎和质量信息。
- 检索质检页为命中块显示表格标签和实际注入的表格上下文；知识库详情页增加“重新解析 PDF”批量操作。
- 增加独立 `pdf_parse` Celery 队列和 Worker，PDF 入库任务路由到该队列；现有 Worker 显式只消费默认队列，避免 OCR 阻塞评估、导出等任务。PDF 任务超时默认 2 小时。
- 独立版面服务使用单独的 Python 3.11 镜像和 `pdf-process` Compose profile，模型权重通过挂载目录提供，不提交仓库。后端依赖增加 RapidOCR、pytesseract、pdfplumber、Camelot 和 Tabula；镜像安装 Tesseract 中文语言包、Ghostscript 和 Java 运行时。

## 测试与验收

- 使用固定 fake OCR、表格和版面服务测试：原生页不调用 OCR、扫描页按优先级回退、乱码触发下一引擎、全部引擎失败、部分页面降级和纯扫描失败。
- 测试混合版面阅读顺序、页眉页脚剔除、标题+正文+表格打包、跨栏页面、合并单元格和扫描表格坐标重建。
- 验证表格永不拆行、超大表使用结构化摘要索引、检索在预算内注入完整 Markdown、超预算回退说明并在 L3 命中时返回完整父表。
- 验证引擎页面启停/排序/删除立即影响下一次任务、密钥不进入响应或日志、健康检查不可用时自动跳过。
- 验证表格随文档删除/重解析清理，知识库导出导入完整往返并正确重映射表格与 chunk 引用。
- 回归测试非 PDF 解析、现有 L1/L2/L3、QA、多表示索引、RRF、Rerank 和上下文窗口。
- 至少增加文字 PDF、扫描 PDF、混合 PDF、跨页表、无边框表、合并单元格表和超宽表样本；正式验收要求表格表头与数据完整率 100%、OCR 乱码页面正确回退、原生 PDF 不产生 OCR 耗时。
- 完成后运行 Ruff、后端全量 pytest、前端生产构建和 Docker Compose 配置检查。

## 假设与默认值

- 管理员页面可自由增删“已支持的适配器实例”；新增全新本地引擎协议仍需部署适配器，HTTP 引擎可直接通过页面添加。
- 默认 OCR 为 RapidOCR → Tesseract，默认表格引擎为 pdfplumber → Camelot → Tabula，版面服务优先于几何规则。
- 版面模型和重型依赖不进入主后端镜像；版面服务未部署时系统仍可运行，但复杂版面质量会降级。
- 存量 PDF 不自动重处理，由管理员通过单文档或知识库批量重解析触发。
- 表格描述是派生数据，可用于补充上下文，但不作为独立事实源替代完整表格或原始 chunk。
