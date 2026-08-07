import re

from app.schemas.ingestion import SourceChunk


class ContextSelector:
    """Simple lexical retrieval that can be replaced with embeddings later."""

    def select(self, chunks: list[SourceChunk], query: str, limit: int = 4, char_budget: int = 6000) -> str:
        if not chunks:
            return ""
        terms = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        scored = []
        for position, chunk in enumerate(chunks):
            haystack = f"{chunk.heading or ''} {chunk.content}".lower()
            score = sum(term in haystack for term in terms)
            scored.append((score, -position, chunk))
        selected = [item[2] for item in sorted(scored, reverse=True)[:limit]]
        pieces: list[str] = []
        used = 0
        for chunk in selected:
            label = f"[{chunk.heading or 'Source'}] "
            piece = label + chunk.content
            if used + len(piece) > char_budget:
                piece = piece[: max(0, char_budget - used)]
            if piece:
                pieces.append(piece)
                used += len(piece)
            if used >= char_budget:
                break
        return "\n\n".join(pieces)
