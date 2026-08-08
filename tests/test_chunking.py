from app.core.config import Settings
from app.schemas.ingestion import ElementType, ExtractedElement
from app.services.chunking import StructureAwareChunker


def test_chunker_respects_headings_and_tables():
    chunks = StructureAwareChunker(Settings(chunk_size=100)).chunk(
        [
            ExtractedElement(text="Scheduling", element_type=ElementType.HEADING, page=1),
            ExtractedElement(text="A scheduler chooses a process.", page=1),
            ExtractedElement(text="Policy | Effect", element_type=ElementType.TABLE, page=2),
        ],
        "notes.pdf",
    )
    assert len(chunks) == 2
    assert chunks[0].heading == "Scheduling"
    assert chunks[1].type == ElementType.TABLE
    assert chunks[1].page == 2
