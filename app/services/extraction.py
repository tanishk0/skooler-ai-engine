from io import BytesIO
import re

from app.schemas.ingestion import ElementType, ExtractedElement


def normalize_text(value: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", value)).strip()


def classify_element(category: str, text: str) -> ElementType:
    category = category.lower()
    if "table" in category:
        return ElementType.TABLE
    if "title" in category:
        return ElementType.TITLE
    if "header" in category or "heading" in category:
        return ElementType.HEADING
    if "list" in category:
        return ElementType.LIST
    if "image" in category or "figure" in category:
        return ElementType.IMAGE
    return ElementType.PARAGRAPH


class DocumentExtractor:
    """Unstructured-first extraction with a small PDF fallback for portability."""

    def extract_text(self, content: str) -> list[ExtractedElement]:
        paragraphs = [normalize_text(part) for part in re.split(r"\n\s*\n", content) if normalize_text(part)]
        return [ExtractedElement(text=paragraph, element_type=ElementType.PARAGRAPH) for paragraph in paragraphs]

    def extract_file(self, data: bytes, content_type: str, filename: str) -> list[ExtractedElement]:
        try:
            return self._extract_with_unstructured(data, content_type, filename)
        except Exception as unstructured_error:
            if content_type == "application/pdf":
                elements = self._extract_pdf_fallback(data)
                if elements:
                    return elements
            raise ValueError("Could not extract readable content from this file") from unstructured_error

    def _extract_with_unstructured(self, data: bytes, content_type: str, filename: str) -> list[ExtractedElement]:
        if content_type == "application/pdf":
            from unstructured.partition.pdf import partition_pdf
            raw_elements = partition_pdf(file=BytesIO(data), strategy="fast", infer_table_structure=True)
        else:
            from unstructured.partition.image import partition_image
            raw_elements = partition_image(file=BytesIO(data), strategy="fast")
        elements: list[ExtractedElement] = []
        active_heading: str | None = None
        for raw in raw_elements:
            text = normalize_text(str(raw))
            if not text:
                continue
            element_type = classify_element(getattr(raw, "category", raw.__class__.__name__), text)
            metadata = getattr(raw, "metadata", None)
            page = getattr(metadata, "page_number", None)
            if element_type in {ElementType.TITLE, ElementType.HEADING}:
                active_heading = text
            elements.append(ExtractedElement(text=text, element_type=element_type, page=page, heading=active_heading))
        if not elements:
            raise ValueError("No text found")
        return elements

    def _extract_pdf_fallback(self, data: bytes) -> list[ExtractedElement]:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        result = []
        for index, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                result.append(ExtractedElement(text=text, element_type=ElementType.PARAGRAPH, page=index))
        return result
