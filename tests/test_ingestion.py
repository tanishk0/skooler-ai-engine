import pytest

from app.core.config import Settings
from app.schemas.ingestion import ElementType, ExtractedElement
from app.services.ingestion import IngestionService


def test_text_ingestion_returns_normalized_chunks():
    result = IngestionService(Settings()).ingest_text("Processes run.\n\nSchedulers choose work.")
    assert result.metadata.chunk_count == 1
    assert result.chunks[0].content == "Processes run.\n\nSchedulers choose work."


def test_pdf_extraction_uses_structured_elements(monkeypatch):
    service = IngestionService(Settings())
    monkeypatch.setattr(service.extractor, "extract_file", lambda *_: [ExtractedElement(text="Title", element_type=ElementType.HEADING, page=1), ExtractedElement(text="Body", page=1)])
    result = service.ingest_file(b"fake", "application/pdf", "slides.pdf")
    assert result.metadata.input_type.value == "pdf"
    assert result.chunks[0].heading == "Title"


def test_image_extraction_failure_falls_back_to_visual_context(monkeypatch):
    service = IngestionService(Settings())
    monkeypatch.setattr(service.extractor, "extract_file", lambda *_: (_ for _ in ()).throw(ValueError("bad image")))
    result = service.ingest_file(b"fake", "image/png", "notes.png")
    assert result.chunks[0].type == ElementType.IMAGE
