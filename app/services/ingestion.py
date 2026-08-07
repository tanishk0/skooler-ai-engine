from app.core.config import Settings
from app.schemas.ingestion import DocumentMetadata, ElementType, ExtractedElement, IngestionResponse, InputType, SourceChunk
from app.services.chunking import StructureAwareChunker
from app.services.extraction import DocumentExtractor


class IngestionService:
    def __init__(self, settings: Settings):
        self.extractor = DocumentExtractor()
        self.chunker = StructureAwareChunker(settings)

    def ingest_text(self, content: str, source_name: str = "pasted-notes.txt") -> IngestionResponse:
        elements = self.extractor.extract_text(content)
        chunks = self.chunker.chunk(elements, source_name)
        if not chunks:
            raise ValueError("No usable content found in text")
        metadata = DocumentMetadata(filename=source_name, content_type="text/plain", input_type=InputType.TEXT, element_count=len(elements), chunk_count=len(chunks))
        return IngestionResponse(document_id=metadata.document_id, filename=source_name, content_type="text/plain", chunks=chunks, metadata=metadata)

    def ingest_file(self, data: bytes, content_type: str, filename: str) -> IngestionResponse:
        try:
            elements = self.extractor.extract_file(data, content_type, filename)
        except ValueError:
            if not content_type.startswith("image/"):
                raise
            # A visual-only image (for example a diagram) remains learnable through
            # Gemini vision even when OCR cannot yield ordinary text.
            elements = [ExtractedElement(text="Visual study material requiring image analysis.", element_type=ElementType.IMAGE)]
        chunks = self.chunker.chunk(elements, filename)
        if not chunks and content_type.startswith("image/"):
            chunks = [SourceChunk(content="Visual study material requiring image analysis.", type=ElementType.IMAGE, source=filename)]
        if not chunks:
            raise ValueError("No usable content found in file")
        input_type = InputType.PDF if content_type == "application/pdf" else InputType.IMAGE
        pages = [element.page for element in elements if element.page]
        metadata = DocumentMetadata(filename=filename, content_type=content_type, input_type=input_type, page_count=max(pages) if pages else None, element_count=len(elements), chunk_count=len(chunks))
        return IngestionResponse(document_id=metadata.document_id, filename=filename, content_type=content_type, chunks=chunks, metadata=metadata)
