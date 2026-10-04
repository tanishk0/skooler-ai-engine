from app.schemas.learning import Concept, InteractionOption, InteractionPrompt, InteractionType, LearningState
from app.services.context import ContextSelector
from app.services.llm import LLMClient, TeachingTurn


class TutorService:
    def __init__(self, llm: LLMClient, context_selector: ContextSelector):
        self.llm = llm
        self.context_selector = context_selector

    def teach(self, state: LearningState, concept: Concept, strategy: str, prior_answer: str | None = None) -> TeachingTurn:
        context = self.context_selector.select(state.source_chunks, f"{state.topic} {concept.name}")
        return self.llm.generate_teaching(
            topic=state.topic,
            concept=concept.name,
            description=concept.description,
            source_context=context,
            misconceptions=state.misconceptions,
            prior_answer=prior_answer,
            strategy=strategy,
        )

    def generate_interaction(self, state: LearningState, concept: Concept, teaching: TeachingTurn, strategy: str) -> InteractionPrompt:
        return self.llm.generate_interaction(
            concept=concept.name,
            teaching=teaching.message,
            strategy=strategy,
            attempts=concept.attempts,
            misconceptions=state.misconceptions,
        )

    def generate_understanding_check(self, concept: Concept, custom_question: str | None = None) -> InteractionPrompt:
        question = custom_question or f"Do you understand the core idea of {concept.name}?"
        return InteractionPrompt(
            type=InteractionType.UNDERSTANDING_CHECK,
            question=question,
            options=[
                InteractionOption(id="understood", label="I understand"),
                InteractionOption(id="not_understood", label="I don't understand"),
            ],
        )

