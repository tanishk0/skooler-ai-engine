from app.core.config import Settings
from app.schemas.ingestion import ElementType, ExtractedElement, SourceChunk


class StructureAwareChunker:
    """Groups document elements under headings before applying a soft size limit."""

    def __init__(self, settings: Settings):
        self.chunk_size = settings.chunk_size
        self.overlap = settings.chunk_overlap

    def chunk(self, elements: list[ExtractedElement], source: str | None) -> list[SourceChunk]:
        chunks: list[SourceChunk] = []
        current_heading: str | None = None
        pending: list[ExtractedElement] = []

        def flush() -> None:
            nonlocal pending
            if not pending:
                return
            content = "\n\n".join(item.text.strip() for item in pending if item.text.strip())
            if content:
                first = pending[0]
                chunk_type = ElementType.TABLE if any(item.element_type == ElementType.TABLE for item in pending) else first.element_type
                chunks.append(SourceChunk(content=content, type=chunk_type, heading=current_heading, page=first.page, source=source))
            pending = []

        for element in elements:
            text = element.text.strip()
            if not text:
                continue
            if element.element_type in {ElementType.TITLE, ElementType.HEADING}:
                flush()
                current_heading = text
                continue
            # Tables carry their own row/column semantics and should not be blended
            # into surrounding prose merely because they fit the size budget.
            if element.element_type == ElementType.TABLE:
                flush()
                chunks.append(SourceChunk(content=text, type=ElementType.TABLE, heading=current_heading, page=element.page, source=source))
                continue
            if pending and (len("\n\n".join(item.text for item in pending)) + len(text) > self.chunk_size or (pending[0].page != element.page and len(pending) > 1)):
                flush()
            if len(text) > self.chunk_size:
                flush()
                chunks.extend(self._split_long_element(element, current_heading, source))
            else:
                pending.append(element.model_copy(update={"heading": current_heading}))
        flush()
        return chunks

    def _split_long_element(self, element: ExtractedElement, heading: str | None, source: str | None) -> list[SourceChunk]:
        words = element.text.split()
        result: list[SourceChunk] = []
        start = 0
        while start < len(words):
            end = start
            length = 0
            while end < len(words) and length + len(words[end]) + 1 <= self.chunk_size:
                length += len(words[end]) + 1
                end += 1
            result.append(SourceChunk(content=" ".join(words[start:end]), type=element.element_type, heading=heading, page=element.page, source=source))
            if end == len(words):
                break
            overlap_words = max(1, self.overlap // 6)
            start = max(start + 1, end - overlap_words)
        return result
