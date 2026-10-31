# PDF Process 服务

`pdf-process` 是 AI-RAG 的独立 PDF 视觉处理服务，负责页面级 OCR 和版面分析。它不保存文档、不访问 PostgreSQL、Milvus 或 Whoosh，只接收页面图像并返回结构化结果。

## 功能

- OCR 引擎：
  - RapidOCR：本地 CPU OCR，默认优先；
  - Tesseract：使用 `chi_sim+eng` 语言包，作为后备引擎。
- 版面分析：
  - 通过 LayoutParser + Detectron2 识别标题、正文、表格、图片等区域；
  - 返回归一化 bbox、置信度和阅读顺序。
- 健康检查：
  - 分别报告 RapidOCR、Tesseract 和版面模型状态；
  - 至少一个 OCR 引擎可用时返回 HTTP 200，版面模型缺失时标记 degraded。
- 可选 Bearer Token 鉴权。
- OCR 请求使用 multipart，避免 Base64 带来的额外传输膨胀。

## 服务边界

```text
pdf-worker
  ├─ POST /ocr      页面 PNG + 目标 OCR 引擎
  └─ POST /analyze  页面图像 + 尺寸

pdf-process
  ├─ RapidOCR / Tesseract
  └─ LayoutParser / Detectron2
```

以下逻辑仍由后端和 `pdf-worker` 负责：

- OCR 引擎优先级和后端自动回退；
- OCR 文本质量判断；
- 表格提取与结构化；
- L1/L2/L3 分块；
- chunk、QA、Embedding 和检索索引持久化。

## 接口

### `GET /health`

```json
{
  "status": "ok",
  "service": "pdf-process",
  "components": {
    "ocr": {
      "rapidocr": {"available": true, "detail": "rapidocr_onnxruntime"},
      "tesseract": {"available": true, "detail": "tesseract"}
    },
    "layout": {"available": false, "detail": "版面模型配置或权重不存在"}
  }
}
```

可通过 `GET /health?engine=rapidocr` 检查指定 OCR 引擎；指定引擎不可用时返回 503。

### `POST /ocr`

Content-Type：`multipart/form-data`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `image` | File | PNG/JPEG 页面图像 |
| `engine` | string | `rapidocr` 或 `tesseract` |
| `page_number` | int | 页码 |
| `width` | float | 页面原始宽度 |
| `height` | float | 页面原始高度 |
| `language` | string | Tesseract 语言，默认 `chi_sim+eng` |
| `layout_hint` | string | 版面提示，当前用于协议兼容 |

示例：

```powershell
curl.exe -X POST "http://localhost:8020/ocr" `
  -H "Authorization: Bearer change-me" `
  -F "image=@page.png;type=image/png" `
  -F "engine=rapidocr" `
  -F "page_number=1" `
  -F "width=1654" `
  -F "height=2339"
```

响应：

```json
{
  "engine_id": "rapidocr",
  "engine_used": "rapidocr",
  "page_number": 1,
  "blocks": [
    {
      "text": "设备安装前必须关闭电源。",
      "bbox": [0.1, 0.2, 0.8, 0.24],
      "confidence": 0.94,
      "block_type": "paragraph"
    }
  ],
  "full_text": "设备安装前必须关闭电源。",
  "mean_confidence": 0.94,
  "duration_ms": 320.5,
  "warnings": []
}
```

错误码：

- `400`：空图像；
- `401`：Bearer Token 错误；
- `413`：图像超过大小限制；
- `415`：上传内容不是图片；
- `422`：OCR 引擎参数非法；
- `503`：指定 OCR 引擎不可用或识别失败。

### `POST /analyze`

保留 JSON + Base64 协议。

```json
{
  "image_base64": "<PNG base64>",
  "page_number": 1,
  "width": 1654,
  "height": 2339
}
```

返回 `regions`，每个区域包含：

- `id`；
- `type`；
- 归一化 `bbox`；
- `confidence`；
- `reading_order`；
- 原始类型元数据。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PDF_PROCESS_API_KEY` | 空 | 可选服务鉴权 Token |
| `PDF_PROCESS_MAX_IMAGE_MB` | `30` | 单页图像上限 |
| `PDF_PROCESS_LAYOUT_MODEL_DIR` | `/models/layout` | 版面模型目录 |
| `PDF_PROCESS_LAYOUT_CONFIG` | `$MODEL_DIR/config.yaml` | Detectron2 配置 |
| `PDF_PROCESS_LAYOUT_WEIGHTS` | `$MODEL_DIR/model_final.pth` | 模型权重 |
| `PDF_PROCESS_LAYOUT_DEVICE` | `cpu` | 推理设备 |
| `PDF_PROCESS_LAYOUT_SCORE_THRESHOLD` | `0.5` | 版面区域分数阈值 |

AI-RAG 后端使用：

- `ZHIYUAN_PDF_PROCESS_SERVICE_URL`：默认 `http://pdf-process:8020`；
- `ZHIYUAN_PDF_PROCESS_API_KEY`：与 `PDF_PROCESS_API_KEY` 对应；
- `ZHIYUAN_PDF_OCR_TIMEOUT_SECONDS`：OCR 请求超时；
- `ZHIYUAN_PDF_OCR_MAX_IMAGE_MB`：Worker 侧发送前限制；
- `ZHIYUAN_PDF_OCR_CIRCUIT_FAILURE_THRESHOLD`；
- `ZHIYUAN_PDF_OCR_CIRCUIT_COOLDOWN_SECONDS`。

## 模型目录

```text
data/models/layout/
├── config.yaml
└── model_final.pth
```

模型权重不提交仓库。未提供模型目录时：

- OCR 仍可正常使用；
- `/analyze` 返回 503；
- `pdf-worker` 自动退化到 PyMuPDF/pdfplumber 几何版面规则。

## Docker Compose

```powershell
docker compose up -d --build pdf-process
docker compose ps pdf-process
docker compose logs -f pdf-process
docker compose exec pdf-process curl -fsS http://localhost:8020/health
```

服务只在 Compose 内网暴露 `8020`，不映射宿主机端口。`pdf-worker` 通过服务健康检查后启动。

如果启用 Bearer Token：

```dotenv
ZHIYUAN_PDF_PROCESS_API_KEY=change-me
```

Compose 会将该值映射为服务内部的 `PDF_PROCESS_API_KEY`；后端 OCR 适配器会通过 `Authorization: Bearer ...` 发送。

## 本地开发

```powershell
cd pdf_process_service
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pip.exe install "git+https://github.com/facebookresearch/detectron2.git"
.\.venv\Scripts\uvicorn.exe app:app --host 0.0.0.0 --port 8020
```

如果只需 OCR，可跳过 Detectron2 安装；此时 `/analyze` 不可用，但 `/ocr` 仍可工作。

本地运行时可直接访问：

- Swagger：<http://localhost:8020/docs>
- 健康检查：<http://localhost:8020/health>

## 测试

服务测试位于 `backend/tests/test_pdf_process_service.py`，覆盖：

- multipart OCR 响应；
- 非法 OCR 引擎；
- 空图像和超大图像；
- 指定引擎不可用；
- OCR 可用但版面模型缺失时的健康状态；
- 可选 Bearer Token 鉴权。

后端 OCR 客户端测试覆盖 multipart、兼容 JSON、鉴权头和连接熔断。