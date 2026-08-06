from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator

from app.schemas.base import APIModel


class InputType(str, Enum):
    TOPIC = "topic"
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"


class ElementType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"


class ExtractedElement(APIModel):
    text: str
    element_type: ElementType = ElementType.PARAGRAPH
    page: int | None = None
    heading: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceChunk(APIModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    type: ElementType = ElementType.PARAGRAPH
    heading: str | None = None
    page: int | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(APIModel):
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str | None = None
    content_type: str
    input_type: InputType
    page_count: int | None = None
    element_count: int = 0
    chunk_count: int = 0


class TextIngestionRequest(APIModel):
    topic: str | None = Field(default=None, max_length=300)
    content: str = Field(min_length=1, max_length=2_000_000)
    source_name: str | None = Field(default="pasted-notes.txt", max_length=255)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class IngestionResponse(APIModel):
    document_id: str
    filename: str | None = None
    content_type: str
    chunks: list[SourceChunk]
    metadata: DocumentMetadata
