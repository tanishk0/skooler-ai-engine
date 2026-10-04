from app.schemas.learning import Concept, Evaluation, LearningState
from app.services.context import ContextSelector
from app.services.llm import LLMClient


class EvaluatorService:
    def __init__(self, llm: LLMClient, context_selector: ContextSelector):
        self.llm = llm
        self.context_selector = context_selector

    def evaluate(self, state: LearningState, concept: Concept, question: str, answer: str) -> Evaluation:
        context = self.context_selector.select(state.source_chunks, f"{state.topic} {concept.name}")
        return self.llm.evaluate_explanation(
            topic=state.topic,
            concept=concept.name,
            description=concept.description,
            source_context=context,
            question=question,
            answer=answer,
            attempts=concept.attempts + 1,
        )
