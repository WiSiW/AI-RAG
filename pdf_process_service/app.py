import asyncio
import base64
import importlib.util
import io
import os
import shutil
import time
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

app = FastAPI(title="PDF Process Service", version="0.2.0")
MODEL_DIR = os.getenv("PDF_PROCESS_LAYOUT_MODEL_DIR", "/models/layout")
DEVICE = os.getenv("PDF_PROCESS_LAYOUT_DEVICE", "cpu")
SCORE_THRESHOLD = float(os.getenv("PDF_PROCESS_LAYOUT_SCORE_THRESHOLD", "0.5"))
MAX_IMAGE_MB = int(os.getenv("PDF_PROCESS_MAX_IMAGE_MB", "30"))
OCR_ENGINES = {"rapidocr", "tesseract"}
SERVICE_API_KEY = os.getenv("PDF_PROCESS_API_KEY", "")


class AnalyzeRequest(BaseModel):
    image_base64: str
    page_number: int = 1
    width: float = 1
    height: float = 1


class Region(BaseModel):
    id: str
    type: str
    bbox: list[float] = Field(default_factory=list)
    confidence: float = 0
    reading_order: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    regions: list[Region]
    model_version: str = "layoutparser-detectron2"


class OCRBlock(BaseModel):
    text: str
    bbox: list[float] = Field(default_factory=list)
    confidence: float | None = None
    block_type: str = "paragraph"


class OCRResponse(BaseModel):
    engine_id: str
    engine_used: str
    page_number: int
    blocks: list[OCRBlock]
    full_text: str
    mean_confidence: float | None = None
    duration_ms: float = 0
    warnings: list[str] = Field(default_factory=list)


def _require_auth(authorization: str | None = Header(default=None)) -> None:
    if not SERVICE_API_KEY:
        return
    if authorization != f"Bearer {SERVICE_API_KEY}":
        raise HTTPException(status_code=401, detail="PDF 处理服务鉴权失败")


def _confidence(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1:
        score /= 100
    return max(0.0, min(1.0, score))


def _mean_confidence(blocks: list[OCRBlock]) -> float | None:
    scores = [block.confidence for block in blocks if block.confidence is not None]
    return sum(scores) / len(scores) if scores else None


def _normalize_bbox(bbox: Any, width: float, height: float) -> list[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [0.0, 0.0, 1.0, 1.0]
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return [0.0, 0.0, 1.0, 1.0]
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1:
        return [max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)]
    width = max(width, 1.0)
    height = max(height, 1.0)
    return [
        max(0.0, min(1.0, x0 / width)),
        max(0.0, min(1.0, y0 / height)),
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
    ]


@lru_cache(maxsize=1)
def _load_layout_model():
    try:
        import layoutparser as lp
    except Exception as exc:
        raise RuntimeError(f"layoutparser 未安装：{exc}") from exc
    config_path = os.getenv("PDF_PROCESS_LAYOUT_CONFIG", os.path.join(MODEL_DIR, "config.yaml"))
    weights_path = os.getenv("PDF_PROCESS_LAYOUT_WEIGHTS", os.path.join(MODEL_DIR, "model_final.pth"))
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        raise RuntimeError("版面模型配置或权重不存在")
    return lp.Detectron2LayoutModel(
        config_path=config_path,
        model_path=weights_path,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", str(SCORE_THRESHOLD)],
        label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
        device=DEVICE,
    )


def _rapidocr_available() -> tuple[bool, str]:
    for module in ("rapidocr_onnxruntime", "rapidocr"):
        if importlib.util.find_spec(module):
            return True, module
    return False, "RapidOCR 未安装"


def _tesseract_available() -> tuple[bool, str]:
    if not shutil.which("tesseract"):
        return False, "tesseract 命令不存在"
    if not importlib.util.find_spec("pytesseract"):
        return False, "pytesseract 未安装"
    return True, "tesseract"


def _ocr_health(engine: str) -> dict[str, Any]:
    checks = {
        "rapidocr": _rapidocr_available,
        "tesseract": _tesseract_available,
    }
    available, detail = checks[engine]()
    return {"available": available, "detail": detail}


def _run_rapidocr(image_bytes: bytes, width: float, height: float) -> list[OCRBlock]:
    import numpy as np
    from PIL import Image as PILImage

    if importlib.util.find_spec("rapidocr_onnxruntime"):
        from rapidocr_onnxruntime import RapidOCR
    elif importlib.util.find_spec("rapidocr"):
        from rapidocr import RapidOCR
    else:
        raise RuntimeError("RapidOCR 未安装")
    image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    raw = RapidOCR()(np.asarray(image))
    result = raw[0] if isinstance(raw, tuple) else raw
    if hasattr(result, "txts") and hasattr(result, "boxes"):
        result = [
            {"box": box, "text": text, "score": score}
            for box, text, score in zip(
                getattr(result, "boxes", []),
                getattr(result, "txts", []),
                getattr(result, "scores", []),
                strict=False,
            )
        ]
    elif isinstance(result, dict) and "results" in result:
        result = result.get("results") or []
    blocks: list[OCRBlock] = []
    for item in result or []:
        if isinstance(item, dict):
            box = item.get("box") or item.get("bbox")
            text = str(item.get("text") or "").strip()
            score = item.get("score", item.get("confidence"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            box, text = item[0], item[1]
            score = item[2] if len(item) > 2 else None
        else:
            continue
        if not text:
            continue
        points = list(box or [])
        if points and isinstance(points[0], (list, tuple)):
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        else:
            bbox = list(points)[:4]
        blocks.append(
            OCRBlock(
                text=text,
                bbox=_normalize_bbox(bbox, width, height),
                confidence=_confidence(score),
            )
        )
    return blocks


def _run_tesseract(
    image_bytes: bytes,
    width: float,
    height: float,
    language: str,
) -> list[OCRBlock]:
    import pytesseract
    from PIL import Image as PILImage

    image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    data = pytesseract.image_to_data(
        image,
        lang=language or "chi_sim+eng",
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )
    grouped: dict[tuple[int, int, int], list[int]] = {}
    for index, text in enumerate(data.get("text", [])):
        if not str(text).strip():
            continue
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        grouped.setdefault(key, []).append(index)
    blocks: list[OCRBlock] = []
    for indexes in grouped.values():
        words = [str(data["text"][index]).strip() for index in indexes]
        xs = [int(data["left"][index]) for index in indexes]
        ys = [int(data["top"][index]) for index in indexes]
        rights = [int(data["left"][index]) + int(data["width"][index]) for index in indexes]
        bottoms = [int(data["top"][index]) + int(data["height"][index]) for index in indexes]
        scores = [_confidence(data["conf"][index]) for index in indexes]
        valid_scores = [score for score in scores if score is not None]
        blocks.append(
            OCRBlock(
                text=" ".join(words),
                bbox=_normalize_bbox([min(xs), min(ys), max(rights), max(bottoms)], width, height),
                confidence=sum(valid_scores) / len(valid_scores) if valid_scores else None,
            )
        )
    return blocks


@app.get("/health")
def health(engine: str | None = None):
    ocr_components = {name: _ocr_health(name) for name in ("rapidocr", "tesseract")}
    if engine:
        if engine not in OCR_ENGINES:
            raise HTTPException(status_code=422, detail=f"不支持的 OCR 引擎：{engine}")
        if not ocr_components[engine]["available"]:
            raise HTTPException(status_code=503, detail=ocr_components[engine]["detail"])
    try:
        _load_layout_model()
        layout = {"available": True, "detail": "layoutparser-detectron2"}
    except Exception as exc:
        layout = {"available": False, "detail": str(exc)}
    ocr_available = any(item["available"] for item in ocr_components.values())
    if not ocr_available and not layout["available"]:
        raise HTTPException(status_code=503, detail="OCR 与版面模型均不可用")
    return {
        "status": "ok" if ocr_available else "degraded",
        "service": "pdf-process",
        "components": {"ocr": ocr_components, "layout": layout},
    }


@app.post("/ocr", response_model=OCRResponse)
async def ocr(
    image: UploadFile = File(...),
    engine: str = Form(...),
    page_number: int = Form(1),
    width: float = Form(1),
    height: float = Form(1),
    language: str = Form("chi_sim+eng"),
    layout_hint: str = Form(""),
    _: None = Depends(_require_auth),
):
    del layout_hint
    engine = engine.strip().lower()
    if engine not in OCR_ENGINES:
        raise HTTPException(status_code=422, detail=f"不支持的 OCR 引擎：{engine}")
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="上传内容必须为图片")
    max_bytes = max(1, MAX_IMAGE_MB) * 1024 * 1024
    image_bytes = await image.read(max_bytes + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="OCR 页面图像为空")
    if len(image_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"OCR 页面图像不能超过 {MAX_IMAGE_MB} MB")
    available = _ocr_health(engine)
    if not available["available"]:
        raise HTTPException(status_code=503, detail=available["detail"])
    started = time.perf_counter()
    try:
        if engine == "rapidocr":
            blocks = await asyncio.to_thread(_run_rapidocr, image_bytes, width, height)
        else:
            blocks = await asyncio.to_thread(
                _run_tesseract, image_bytes, width, height, language
            )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"OCR 识别失败：{exc}") from exc
    text = "\n".join(block.text for block in blocks)
    return OCRResponse(
        engine_id=engine,
        engine_used=engine,
        page_number=page_number,
        blocks=blocks,
        full_text=text,
        mean_confidence=_mean_confidence(blocks),
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, _: None = Depends(_require_auth)):
    try:
        from PIL import Image
        model = _load_layout_model()
        image = Image.open(io.BytesIO(base64.b64decode(request.image_base64))).convert("RGB")
        layout = model.detect(image)
        regions: list[Region] = []
        for index, block in enumerate(
            sorted(layout, key=lambda item: (item.block.y_1, item.block.x_1))
        ):
            region_type = str(block.type)
            if region_type in {"text", "list"}:
                region_type = "paragraph"
            elif region_type in {"figure", "image"}:
                region_type = "figure"
            regions.append(
                Region(
                    id=f"p{request.page_number}-r{index}",
                    type=region_type,
                    bbox=[
                        block.block.x_1 / request.width,
                        block.block.y_1 / request.height,
                        block.block.x_2 / request.width,
                        block.block.y_2 / request.height,
                    ],
                    confidence=float(getattr(block, "score", 0) or 0),
                    reading_order=index,
                    metadata={"raw_type": str(block.type)},
                )
            )
        return AnalyzeResponse(regions=regions)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"版面分析失败：{exc}") from exc