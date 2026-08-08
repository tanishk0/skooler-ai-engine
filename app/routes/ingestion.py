from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.schemas.ingestion import ElementType, IngestionResponse, SourceChunk, TextIngestionRequest
from app.services.llm import LLMError

router = APIRouter(prefix="/ingestion", tags=["ingestion"])

ALLOWED_FILES = {
    "application/pdf": {".pdf"},
    "image/jpeg": {".jpg", ".jpeg"},
    "image/png": {".png"},
    "image/webp": {".webp"},
}


@router.post("/text", response_model=IngestionResponse)
def ingest_text(payload: TextIngestionRequest, request: Request) -> IngestionResponse:
    try:
        return request.app.state.ingestion_service.ingest_text(payload.content, payload.source_name or "pasted-notes.txt")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/file", response_model=IngestionResponse)
async def ingest_file(request: Request, file: UploadFile = File(...)) -> IngestionResponse:
    filename = file.filename or "upload"
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_type = file.content_type or ""
    permitted_extensions = ALLOWED_FILES.get(content_type)
    if not permitted_extensions or extension not in permitted_extensions:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF, JPEG, PNG, and WebP files are supported")

    max_bytes = request.app.state.settings.max_upload_size_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uploaded file is empty")
    if len(data) > max_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"File exceeds the {request.app.state.settings.max_upload_size_mb} MB limit")
    try:
        result = request.app.state.ingestion_service.ingest_file(data, content_type, filename)
        # OCR misses diagrams and handwriting. Preserve Gemini's visual analysis as a distinct chunk.
        if content_type.startswith("image/") and request.app.state.llm.is_configured:
            try:
                visual_context = request.app.state.llm.analyze_image(data, content_type)
                if visual_context:
                    result.chunks.append(SourceChunk(content=visual_context, type=ElementType.IMAGE, source=filename, metadata={"derived_from": "gemini_vision"}))
                    result.metadata.chunk_count = len(result.chunks)
            except LLMError:
                # Text extraction remains useful if a vision request is rate-limited or unavailable.
                pass
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
