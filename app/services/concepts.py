from app.schemas.learning import LearningPlan, LearningModule
from app.services.context import ContextSelector
from app.services.llm import LLMClient


class ConceptService:
    def __init__(
        self,
        llm: LLMClient,
        context_selector: ContextSelector,
    ):
        self.llm = llm
        self.context_selector = context_selector

    def create_plan(self, topic: str, chunks: list) -> LearningPlan:
        context = (
            self.context_selector.select(chunks, topic, limit=6)
            if chunks
            else ""
        )

        plan = self.llm.generate_learning_plan(topic, context)

        # Backward compatibility:
        # If an older/flat plan has concepts but no modules,
        # wrap those concepts into a single module.
        if not plan.modules:
            module = LearningModule(
                id="module-1",
                name=topic,
                description=f"Foundational concepts for {topic}",
                concepts=[
                    concept.model_copy(
                        update={"module_id": "module-1"}
                    )
                    for concept in plan.concepts
                ],
            )

            plan.modules = [module]

        else:
            # Ensure every flattened concept knows its owning module.
            concepts_by_id = {
                concept.id: concept
                for concept in plan.concepts
            }

            for module in plan.modules:
                for module_concept in module.concepts:
                    if module_concept.id in concepts_by_id:
                        concepts_by_id[
                            module_concept.id
                        ].module_id = module.id

        return plan