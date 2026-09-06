from io import BytesIO

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import RedirectResponse
from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings
from app.schemas.prediction import PredictionResponse
from app.services.model_manager import ModelManager

router = APIRouter()


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/en", status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.get("/api", tags=["system"])
def api_info() -> dict[str, str]:
    return {
        "service": "AutoSnap API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", tags=["system"])
def health(request: Request) -> dict[str, str | bool]:
    manager: ModelManager = request.app.state.model_manager
    return {
        "status": "ok",
        "model_loaded": manager.is_loaded,
    }


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    status_code=status.HTTP_200_OK,
)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    settings = get_settings()
    manager: ModelManager = request.app.state.model_manager

    if not manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The trained model is not loaded. Set MODEL_PATH or add the model adapter.",
        )

    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WEBP images are supported.")

    image_bytes = await file.read(settings.max_upload_bytes + 1)
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image is larger than the configured upload limit.")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid image.") from None

    try:
        result = manager.predict(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Model inference failed.") from exc

    return PredictionResponse(filename=file.filename, result=result)
