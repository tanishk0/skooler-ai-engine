from app.schemas.ingestion import InputType, SourceChunk
from app.schemas.learning import (
    Concept,
    ConceptStatus,
    ContinueLearningResponse,
    Evaluation,
    InteractionPrompt,
    LearningAction,
    LearningEvent,
    LearningEventType,
    LearningStage,
    LearningState,
    RespondLearningResponse,
    StartLearningResponse,
)
from app.services.concepts import ConceptService
from app.services.evaluator import EvaluatorService
from app.services.ingestion import IngestionService
from app.services.llm import TeachingTurn
from app.services.tutor import TutorService


class InvalidTransitionError(ValueError):
    pass


class LearningOrchestrator:
    """
    Authoritative adaptive learning state machine.

    Responsibilities:
    - Control learning progression.
    - Keep the learner on one concept until sufficient evidence exists.
    - Prevent infinite DEEPEN loops.
    - Transition concept -> concept.
    - Transition module -> module.
    - Preserve the existing understanding-check / Continue flow.
    - Never pre-generate future learning steps.
    """

    def __init__(
        self,
        concepts: ConceptService,
        tutor: TutorService,
        evaluator: EvaluatorService,
        ingestion: IngestionService,
    ):
        self.concepts = concepts
        self.tutor = tutor
        self.evaluator = evaluator
        self.ingestion = ingestion

    # ============================================================
    # START
    # ============================================================

    def start(
        self,
        session_id: str,
        input_type: InputType,
        topic: str,
        content: str | None,
        source_chunks: list[SourceChunk] | None,
    ) -> StartLearningResponse:
        chunks = source_chunks or []

        if input_type == InputType.TEXT and not chunks:
            chunks = self.ingestion.ingest_text(content or "").chunks

        plan = self.concepts.create_plan(topic, chunks)

        # The schema keeps plan.concepts synchronized with modules.
        if not plan.concepts:
            raise InvalidTransitionError(
                "Learning plan contains no concepts"
            )

        concept = plan.concepts[0]
        concept.status = ConceptStatus.IN_PROGRESS

        module = self._module_for_concept(plan, concept.id)

        state = LearningState(
            session_id=session_id,
            topic=plan.topic,
            input_type=input_type,
            plan=plan,
            concepts=plan.concepts,
            current_module_index=0,
            current_module_id=module.id if module else None,
            current_concept_index=0,
            current_concept_id=concept.id,
            concept_depth=0,
            evidence_count=0,
            concept_progress={
                item.id: 0 for item in plan.concepts
            },
            module_progress={
                item.id: 0 for item in plan.modules
            },
            source_chunks=chunks,
        )

        teaching = self.tutor.teach(
            state,
            concept,
            "introduce the first small, prerequisite-sized idea",
        )

        check = self.tutor.generate_understanding_check(concept)

        state.current_interaction = check

        self._set_stage(
            state,
            LearningStage.AWAITING_UNDERSTANDING_CHECK,
        )

        events = self._append_events(
            state,
            [
                LearningEvent(
                    type=LearningEventType.TEACHING,
                    module_id=module.id if module else None,
                    concept_id=concept.id,
                    content=teaching.message,
                ),
                self._interaction_event(
                    concept,
                    check,
                    module_id=module.id if module else None,
                ),
            ],
        )

        return StartLearningResponse(
            session_id=session_id,
            stage=state.stage,
            concept=concept.name,
            message=teaching.message,
            question=check.question,
            learning_state=state,
            events=events,
        )

    # ============================================================
    # RESPOND
    # ============================================================

    def respond(
        self,
        session_id: str,
        state: LearningState,
        user_response: str,
    ) -> RespondLearningResponse:
        self._validate_session(session_id, state)

        if state.stage == LearningStage.AWAITING_UNDERSTANDING_CHECK:
            return self._respond_to_understanding_check(
                session_id,
                state,
                user_response,
            )

        if state.stage == LearningStage.AWAITING_INTERACTION_RESPONSE:
            return self._respond_to_interaction(
                session_id,
                state,
                user_response,
            )

        raise InvalidTransitionError(
            "This session is not waiting for a learner response"
        )

    # ============================================================
    # CONTINUE
    # ============================================================

    def continue_learning(
        self,
        session_id: str,
        state: LearningState,
    ) -> ContinueLearningResponse:
        self._validate_session(session_id, state)

        if state.stage == LearningStage.AWAITING_CONTINUE_AFTER_RETEACH:
            return self._continue_after_reteach(
                session_id,
                state,
            )

        if state.stage == LearningStage.AWAITING_CONTINUE_AFTER_UNDERSTOOD:
            return self._continue_after_understood(
                session_id,
                state,
            )

        if state.stage == LearningStage.AWAITING_CONTINUE_AFTER_EVALUATION:
            return self._continue_after_evaluation(
                session_id,
                state,
            )

        if state.stage == LearningStage.AWAITING_CONTINUE_AFTER_CONCEPT_TRANSITION:
            return self._continue_after_transition(
                session_id,
                state,
            )

        raise InvalidTransitionError(
            "Continue is not valid while the session awaits a learner response"
        )

    # ============================================================
    # UNDERSTANDING CHECK
    # ============================================================

    def _respond_to_understanding_check(
        self,
        session_id: str,
        state: LearningState,
        answer: str,
    ) -> RespondLearningResponse:
        concept = state.current_concept
        module = self._current_module(state)

        choice = (
            answer
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("'", "")
        )

        user_event = LearningEvent(
            type=LearningEventType.USER_ANSWER,
            module_id=module.id if module else None,
            concept_id=concept.id,
            content=answer,
        )

        if choice in {
            "understood",
            "i_understand",
            "yes",
        }:
            state.current_interaction = None

            self._set_stage(
                state,
                LearningStage.AWAITING_CONTINUE_AFTER_UNDERSTOOD,
            )

            transition_event = LearningEvent(
                type=LearningEventType.CONCEPT_TRANSITION,
                module_id=module.id if module else None,
                concept_id=concept.id,
                content=(
                    "Good. Let's test your understanding "
                    "when you're ready."
                ),
                metadata={
                    "transition_type": "continue_to_interaction"
                },
            )

            events = self._append_events(
                state,
                [
                    user_event,
                    transition_event,
                ],
            )

            return self._respond_response(
                session_id,
                state,
                LearningAction.DEEPEN,
                "Good. Let's test your understanding when you're ready.",
                None,
                None,
                events,
            )

        if choice in {
            "not_understood",
            "i_dont_understand",
            "no",
            "not_really",
        }:
            reteach = self.tutor.teach(
                state,
                concept,
                "reteach this same idea using a substantially different concrete analogy or mental model",
                answer,
            )

            state.current_interaction = None

            self._set_stage(
                state,
                LearningStage.AWAITING_CONTINUE_AFTER_RETEACH,
            )

            events = self._append_events(
                state,
                [
                    user_event,
                    LearningEvent(
                        type=LearningEventType.RETEACH,
                        module_id=module.id if module else None,
                        concept_id=concept.id,
                        content=reteach.message,
                    ),
                ],
            )

            return self._respond_response(
                session_id,
                state,
                LearningAction.RETEACH,
                reteach.message,
                None,
                None,
                events,
            )

        raise InvalidTransitionError(
            "The active understanding check only accepts understood or not_understood"
        )

    # ============================================================
    # INTERACTION RESPONSE
    # ============================================================

    def _respond_to_interaction(
        self,
        session_id: str,
        state: LearningState,
        answer: str,
    ) -> RespondLearningResponse:
        concept = state.current_concept
        module = self._current_module(state)

        question = (
            state.current_interaction.question
            if state.current_interaction
            else "Explain the concept in your own words."
        )

        evaluation = self.evaluator.evaluate(
            state,
            concept,
            question,
            answer,
        )

        concept.attempts += 1

        self._record_learning_evidence(
            state,
            concept,
            evaluation,
        )

        self._update_progress(
            state,
            concept,
            evaluation,
        )

        state.misconceptions = list(
            dict.fromkeys(
                state.misconceptions
                + evaluation.misconceptions
            )
        )[-10:]

        state.pending_action = self._enforced_action(
            evaluation,
            state,
            concept,
        )

        state.last_evaluation = evaluation
        state.current_interaction = None

        self._set_stage(
            state,
            LearningStage.AWAITING_CONTINUE_AFTER_EVALUATION,
        )

        events = self._append_events(
            state,
            [
                LearningEvent(
                    type=LearningEventType.USER_ANSWER,
                    module_id=module.id if module else None,
                    concept_id=concept.id,
                    content=answer,
                ),
                LearningEvent(
                    type=LearningEventType.EVALUATION,
                    module_id=module.id if module else None,
                    concept_id=concept.id,
                    content=evaluation.feedback,
                    metadata=evaluation.model_dump(
                        by_alias=True
                    ),
                ),
            ],
        )

        return self._respond_response(
            session_id,
            state,
            state.pending_action,
            evaluation.feedback,
            None,
            evaluation,
            events,
        )

    # ============================================================
    # CONTINUE AFTER UNDERSTOOD
    # ============================================================

    def _continue_after_understood(
        self,
        session_id: str,
        state: LearningState,
    ) -> ContinueLearningResponse:
        concept = state.current_concept
        module = self._current_module(state)

        interaction = self.tutor.generate_interaction(
            state,
            concept,
            TeachingTurn(
                message=(
                    "The learner reported that the initial "
                    "explanation was understood."
                )
            ),
            self._strategy_for_depth(
                concept,
                state.concept_depth,
            ),
        )

        state.current_interaction = interaction

        self._set_stage(
            state,
            LearningStage.AWAITING_INTERACTION_RESPONSE,
        )

        events = self._append_events(
            state,
            [
                self._interaction_event(
                    concept,
                    interaction,
                    module_id=module.id if module else None,
                )
            ],
        )

        return self._continue_response(
            session_id,
            state,
            LearningAction.DEEPEN,
            None,
            interaction.question,
            events,
        )

    # ============================================================
    # CONTINUE AFTER RETEACH
    # ============================================================

    def _continue_after_reteach(
        self,
        session_id: str,
        state: LearningState,
    ) -> ContinueLearningResponse:
        concept = state.current_concept
        module = self._current_module(state)

        check = self.tutor.generate_understanding_check(
            concept,
            "Does this explanation make more sense now?",
        )

        state.current_interaction = check

        self._set_stage(
            state,
            LearningStage.AWAITING_UNDERSTANDING_CHECK,
        )

        events = self._append_events(
            state,
            [
                self._interaction_event(
                    concept,
                    check,
                    module_id=module.id if module else None,
                )
            ],
        )

        return self._continue_response(
            session_id,
            state,
            None,
            None,
            check.question,
            events,
        )

    # ============================================================
    # CONTINUE AFTER EVALUATION
    # ============================================================

    def _continue_after_evaluation(
        self,
        session_id: str,
        state: LearningState,
    ) -> ContinueLearningResponse:
        action = state.pending_action

        if action is None:
            raise InvalidTransitionError(
                "No evaluated next action is available"
            )

        concept = state.current_concept
        module = self._current_module(state)

        state.pending_action = None

        # --------------------------------------------------------
        # RETEACH / CLARIFY / TRY AGAIN
        # --------------------------------------------------------

        if action in {
            LearningAction.RETEACH,
            LearningAction.CLARIFY,
            LearningAction.TRY_AGAIN,
            LearningAction.ASK_AGAIN,
        }:
            if action == LearningAction.RETEACH:
                strategy = (
                    "reteach the identified misconception using "
                    "a genuinely different explanation, analogy, "
                    "or concrete example"
                )
            elif action == LearningAction.CLARIFY:
                strategy = (
                    "clarify exactly one missing piece of reasoning "
                    "without repeating the entire previous explanation"
                )
            else:
                strategy = (
                    "give the learner another focused opportunity "
                    "to reason about the same idea using a different angle"
                )

            artifact = self.tutor.teach(
                state,
                concept,
                strategy,
            )

            concept.status = ConceptStatus.PARTIAL

            self._set_stage(
                state,
                LearningStage.AWAITING_CONTINUE_AFTER_RETEACH,
            )

            event_type = (
                LearningEventType.RETEACH
                if action == LearningAction.RETEACH
                else LearningEventType.TEACHING
            )

            events = self._append_events(
                state,
                [
                    LearningEvent(
                        type=event_type,
                        module_id=module.id if module else None,
                        concept_id=concept.id,
                        content=artifact.message,
                    )
                ],
            )

            return self._continue_response(
                session_id,
                state,
                action,
                artifact.message,
                None,
                events,
            )

        # --------------------------------------------------------
        # DEEPEN
        # --------------------------------------------------------

        if action in {
            LearningAction.DEEPEN,
            LearningAction.TEST_DEEPER,
        }:
            # IMPORTANT:
            # The evaluator may say "correct", but that does not mean
            # we should ask another question forever.
            #
            # If enough evidence has already been collected,
            # transition to mastery instead.
            if self._can_master_concept(state, concept):
                return self._master_current_concept(
                    session_id,
                    state,
                )

            interaction = self.tutor.generate_interaction(
                state,
                concept,
                TeachingTurn(
                    message=(
                        "The learner has demonstrated some understanding "
                        "but more evidence is needed."
                    )
                ),
                self._strategy_for_depth(
                    concept,
                    state.concept_depth,
                ),
            )

            state.current_interaction = interaction

            self._set_stage(
                state,
                LearningStage.AWAITING_INTERACTION_RESPONSE,
            )

            events = self._append_events(
                state,
                [
                    self._interaction_event(
                        concept,
                        interaction,
                        module_id=module.id if module else None,
                    )
                ],
            )

            return self._continue_response(
                session_id,
                state,
                LearningAction.DEEPEN,
                None,
                interaction.question,
                events,
            )

        # --------------------------------------------------------
        # TEACH NEXT
        # --------------------------------------------------------

        if action == LearningAction.TEACH_NEXT:
            return self._master_current_concept(
                session_id,
                state,
            )

        # --------------------------------------------------------
        # COMPLETE
        # --------------------------------------------------------

        if action == LearningAction.COMPLETE:
            return self._complete_session(
                session_id,
                state,
            )

        raise InvalidTransitionError(
            "Unsupported evaluated next action"
        )

    # ============================================================
    # CONCEPT TRANSITION
    # ============================================================

    def _continue_after_transition(
        self,
        session_id: str,
        state: LearningState,
    ) -> ContinueLearningResponse:
        concept = state.current_concept
        module = self._current_module(state)

        teaching = self.tutor.teach(
            state,
            concept,
            "introduce one small, prerequisite-sized idea in the new concept",
        )

        check = self.tutor.generate_understanding_check(
            concept
        )

        state.current_interaction = check
        state.concept_depth = 0
        state.evidence_count = 0
        concept.attempts = 0
        concept.status = ConceptStatus.IN_PROGRESS

        self._set_stage(
            state,
            LearningStage.AWAITING_UNDERSTANDING_CHECK,
        )

        events = self._append_events(
            state,
            [
                LearningEvent(
                    type=LearningEventType.TEACHING,
                    module_id=module.id if module else None,
                    concept_id=concept.id,
                    content=teaching.message,
                ),
                self._interaction_event(
                    concept,
                    check,
                    module_id=module.id if module else None,
                ),
            ],
        )

        return self._continue_response(
            session_id,
            state,
            LearningAction.TEACH_NEXT,
            teaching.message,
            check.question,
            events,
        )

    # ============================================================
    # MASTER CURRENT CONCEPT
    # ============================================================

    def _master_current_concept(
        self,
        session_id: str,
        state: LearningState,
    ) -> ContinueLearningResponse:
        concept = state.current_concept
        module = self._current_module(state)

        concept.status = ConceptStatus.MASTERED
        state.concept_progress[concept.id] = 100

        mastered_event = LearningEvent(
            type=LearningEventType.MASTERED,
            module_id=module.id if module else None,
            concept_id=concept.id,
            content=(
                f"You demonstrated mastery of {concept.name}."
            ),
            metadata={
                "completed_concept_name": concept.name,
                "concept_depth": concept.depth,
                "target_depth": concept.target_depth,
            },
        )

        # Update module progress before deciding where to go.
        if module is not None:
            self._update_module_progress(
                state,
                module.id,
            )

        # --------------------------------------------------------
        # More concepts in current module
        # --------------------------------------------------------

        next_concept_index = self._next_concept_index_in_module(
            state,
            state.current_concept_index,
        )

        if next_concept_index is not None:
            state.current_concept_index = next_concept_index

            next_concept = state.concepts[
                next_concept_index
            ]

            next_concept.status = ConceptStatus.IN_PROGRESS
            state.current_concept_id = next_concept.id
            state.concept_depth = 0
            state.evidence_count = 0

            transition_event = LearningEvent(
                type=LearningEventType.CONCEPT_TRANSITION,
                module_id=module.id if module else None,
                concept_id=next_concept.id,
                content=(
                    f"You mastered {concept.name}. "
                    f"Next, we will explore {next_concept.name}."
                ),
                metadata={
                    "transition_type": "next_concept",
                    "previous_concept_id": concept.id,
                    "next_concept_id": next_concept.id,
                },
            )

            self._set_stage(
                state,
                LearningStage.AWAITING_CONTINUE_AFTER_CONCEPT_TRANSITION,
            )

            events = self._append_events(
                state,
                [
                    mastered_event,
                    transition_event,
                ],
            )

            return self._continue_response(
                session_id,
                state,
                LearningAction.TEACH_NEXT,
                transition_event.content,
                None,
                events,
                response_concept=concept.name,
            )

        # --------------------------------------------------------
        # Current module complete
        # --------------------------------------------------------

        if module is not None:
            module.status = ConceptStatus.MASTERED
            module.progress = 100
            state.module_progress[module.id] = 100

        next_module_index = self._next_module_index(state)

        # --------------------------------------------------------
        # More modules
        # --------------------------------------------------------

        if next_module_index is not None:
            next_module = state.plan.modules[
                next_module_index
            ]

            state.current_module_index = next_module_index
            state.current_module_id = next_module.id

            # Find first concept belonging to the next module.
            next_concept_index = self._first_concept_index_in_module(
                state,
                next_module.id,
            )

            if next_concept_index is None:
                raise InvalidTransitionError(
                    f"Module {next_module.name} contains no concepts"
                )

            state.current_concept_index = next_concept_index

            next_concept = state.concepts[
                next_concept_index
            ]

            state.current_concept_id = next_concept.id
            state.concept_depth = 0
            state.evidence_count = 0

            next_concept.status = ConceptStatus.NOT_STARTED

            transition_event = LearningEvent(
                type=LearningEventType.CONCEPT_TRANSITION,
                module_id=next_module.id,
                concept_id=next_concept.id,
                content=(
                    f"You completed {module.name}. "
                    f"Next, we'll explore {next_module.name}."
                ),
                metadata={
                    "transition_type": "next_module",
                    "completed_module_id": module.id,
                    "completed_module_name": module.name,
                    "next_module_id": next_module.id,
                    "next_module_name": next_module.name,
                },
            )

            self._set_stage(
                state,
                LearningStage.AWAITING_CONTINUE_AFTER_CONCEPT_TRANSITION,
            )

            events = self._append_events(
                state,
                [
                    mastered_event,
                    transition_event,
                ],
            )

            return self._continue_response(
                session_id,
                state,
                LearningAction.TEACH_NEXT,
                transition_event.content,
                None,
                events,
                response_concept=concept.name,
            )

        # --------------------------------------------------------
        # Entire curriculum complete
        # --------------------------------------------------------

        return self._complete_session(
            session_id,
            state,
            existing_events=[mastered_event],
        )

    # ============================================================
    # COMPLETE SESSION
    # ============================================================

    def _complete_session(
        self,
        session_id: str,
        state: LearningState,
        existing_events: list[LearningEvent] | None = None,
    ) -> ContinueLearningResponse:
        concept = state.current_concept

        concept.status = ConceptStatus.MASTERED
        state.concept_progress[concept.id] = 100
        state.mastery = 100

        self._update_all_module_progress(state)

        self._set_stage(
            state,
            LearningStage.COMPLETE,
        )

        events = existing_events or []

        if not any(
            event.type == LearningEventType.MASTERED
            and event.concept_id == concept.id
            for event in events
        ):
            events.append(
                LearningEvent(
                    type=LearningEventType.MASTERED,
                    module_id=state.current_module_id,
                    concept_id=concept.id,
                    content=(
                        f"You have demonstrated mastery of {concept.name}."
                    ),
                    metadata={
                        "completed_concept_name": concept.name
                    },
                )
            )

        events = self._append_events(
            state,
            events,
        )

        return self._continue_response(
            session_id,
            state,
            LearningAction.COMPLETE,
            "You have completed this learning curriculum.",
            None,
            events,
            is_complete=True,
            response_concept=concept.name,
        )

    # ============================================================
    # MASTERY / DEPTH
    # ============================================================

    @staticmethod
    def _record_learning_evidence(
        state: LearningState,
        concept: Concept,
        evaluation: Evaluation,
    ) -> None:
        """
        Increase conceptual depth only when the learner provides
        meaningful evidence.

        Incorrect answers do not artificially advance depth.
        Partial answers only advance evidence_count.
        Correct/mastered answers advance conceptual depth.
        """

        state.evidence_count += 1

        if evaluation.understanding in {
            "correct",
            "mastered",
        }:
            concept.depth = min(
                concept.depth + 1,
                concept.target_depth,
            )
            state.concept_depth = concept.depth

        elif evaluation.understanding == "partial":
            state.concept_depth = concept.depth

    @staticmethod
    def _can_master_concept(
        state: LearningState,
        concept: Concept,
    ) -> bool:
        """
        A concept can be mastered only when:

        1. The learner has reached the intended conceptual depth.
        2. The latest evaluation demonstrates strong enough understanding.
        3. There is no material misconception.
        """

        evaluation = state.last_evaluation

        if evaluation is None:
            return False

        depth_sufficient = (
            concept.depth >= concept.target_depth
        )

        evaluation_sufficient = (
            evaluation.understanding
            in {"correct", "mastered"}
            and evaluation.confidence >= 0.75
            and evaluation.reasoning_quality
            in {"adequate", "strong"}
            and not evaluation.misconceptions
        )

        return depth_sufficient and evaluation_sufficient

    # ============================================================
    # ACTION ENFORCEMENT
    # ============================================================

    @staticmethod
    def _enforced_action(
        evaluation: Evaluation,
        state: LearningState,
        concept: Concept,
    ) -> LearningAction:

        # Never advance on an actually incorrect answer.
        if evaluation.understanding == "incorrect":
            return LearningAction.RETEACH

        # Partial understanding must be addressed first.
        if evaluation.understanding == "partial":
            if evaluation.misconceptions:
                return LearningAction.RETEACH
            return LearningAction.CLARIFY

        # A correct/mastered answer with explicit mastery evidence
        # should advance immediately.
        if (
            evaluation.understanding in {"correct", "mastered"}
            and evaluation.mastery_reached
        ):
            if evaluation.next_action == LearningAction.COMPLETE:
                return LearningAction.COMPLETE

            return LearningAction.TEACH_NEXT

        # Correct but not yet mastered -> gather another piece of evidence.
        if evaluation.understanding in {"correct", "mastered"}:
            return LearningAction.DEEPEN

        return LearningAction.DEEPEN

    # ============================================================
    # STRATEGY
    # ============================================================

    @staticmethod
    def _strategy_for_depth(
        concept: Concept,
        depth: int,
    ) -> str:
        """
        Give the interaction generator progressively different
        pedagogical goals instead of asking the same question
        repeatedly.
        """

        if depth <= 0:
            return (
                "test the learner's basic understanding of "
                "the concept with a simple explanation or distinction"
            )

        if depth == 1:
            return (
                "test the learner's ability to apply the concept "
                "to a concrete example or predict an outcome"
            )

        if depth == 2:
            return (
                "test the learner's ability to explain the concept "
                "independently using the Feynman technique"
            )

        if depth == 3:
            return (
                "test a deeper application, edge case, or "
                "common misconception"
            )

        return (
            "test the learner's strongest remaining mastery criterion "
            "without repeating a previous interaction"
        )

    # ============================================================
    # MODULE / CONCEPT HELPERS
    # ============================================================

    @staticmethod
    def _module_for_concept(
        plan,
        concept_id: str,
    ):
        for module in plan.modules:
            for concept in module.concepts:
                if concept.id == concept_id:
                    return module

        return None

    @staticmethod
    def _current_module(
        state: LearningState,
    ):
        if state.plan is None:
            return None

        if (
            state.current_module_index < 0
            or state.current_module_index
            >= len(state.plan.modules)
        ):
            return None

        return state.plan.modules[
            state.current_module_index
        ]

    @staticmethod
    def _first_concept_index_in_module(
        state: LearningState,
        module_id: str,
    ) -> int | None:
        if state.plan is None:
            return None

        module = next(
            (
                module
                for module in state.plan.modules
                if module.id == module_id
            ),
            None,
        )

        if module is None:
            return None

        concept_ids = {
            concept.id
            for concept in module.concepts
        }

        for index, concept in enumerate(state.concepts):
            if concept.id in concept_ids:
                return index

        return None

    @staticmethod
    def _next_concept_index_in_module(
        state: LearningState,
        current_index: int,
    ) -> int | None:
        module = LearningOrchestrator._current_module(
            state
        )

        if module is None:
            return None

        module_concept_ids = {
            concept.id
            for concept in module.concepts
        }

        for index in range(
            current_index + 1,
            len(state.concepts),
        ):
            if state.concepts[index].id in module_concept_ids:
                return index

        return None

    @staticmethod
    def _next_module_index(
        state: LearningState,
    ) -> int | None:
        if state.plan is None:
            return None

        next_index = (
            state.current_module_index + 1
        )

        if next_index >= len(state.plan.modules):
            return None

        return next_index

    # ============================================================
    # PROGRESS
    # ============================================================

    @staticmethod
    def _update_progress(
        state: LearningState,
        concept: Concept,
        evaluation: Evaluation,
    ) -> None:
        points = {
            "incorrect": 15,
            "partial": 45,
            "correct": 80,
            "mastered": 100,
        }[evaluation.understanding]

        # Depth also contributes to progress.
        depth_ratio = (
            concept.depth
            / max(concept.target_depth, 1)
        )

        depth_points = round(
            min(depth_ratio, 1) * 100
        )

        combined = round(
            (points * 0.6)
            + (depth_points * 0.4)
        )

        state.concept_progress[concept.id] = max(
            state.concept_progress.get(
                concept.id,
                0,
            ),
            combined,
        )

        state.mastery = round(
            sum(
                state.concept_progress.values()
            )
            / max(len(state.concepts), 1),
            1,
        )

        module = LearningOrchestrator._current_module(
            state
        )

        if module is not None:
            LearningOrchestrator._update_module_progress(
                state,
                module.id,
            )

    @staticmethod
    def _update_module_progress(
        state: LearningState,
        module_id: str,
    ) -> None:
        if state.plan is None:
            return

        module = next(
            (
                module
                for module in state.plan.modules
                if module.id == module_id
            ),
            None,
        )

        if module is None:
            return

        concept_ids = [
            concept.id
            for concept in module.concepts
        ]

        if not concept_ids:
            return

        progress = sum(
            state.concept_progress.get(
                concept_id,
                0,
            )
            for concept_id in concept_ids
        ) / len(concept_ids)

        module.progress = round(
            min(progress, 100),
            1,
        )

        state.module_progress[module_id] = module.progress

    @staticmethod
    def _update_all_module_progress(
        state: LearningState,
    ) -> None:
        if state.plan is None:
            return

        for module in state.plan.modules:
            LearningOrchestrator._update_module_progress(
                state,
                module.id,
            )

    # ============================================================
    # EVENTS
    # ============================================================

    @staticmethod
    def _interaction_event(
        concept: Concept,
        interaction: InteractionPrompt,
        module_id: str | None = None,
    ) -> LearningEvent:
        event_type = LearningEventType(
            interaction.type.value
        )

        return LearningEvent(
            type=event_type,
            module_id=module_id,
            concept_id=concept.id,
            question=interaction.question,
            options=interaction.options,
        )

    @staticmethod
    def _append_events(
        state: LearningState,
        events: list[LearningEvent],
    ) -> list[LearningEvent]:
        for offset, event in enumerate(
            events,
            start=1,
        ):
            event.sequence = (
                state.event_sequence + offset
            )

        state.event_sequence += len(events)

        return events

    # ============================================================
    # STATE / RESPONSE HELPERS
    # ============================================================

    @staticmethod
    def _validate_session(
        session_id: str,
        state: LearningState,
    ) -> None:
        if state.session_id != session_id:
            raise InvalidTransitionError(
                "session_id does not match learning_state"
            )

    @staticmethod
    def _set_stage(
        state: LearningState,
        stage: LearningStage,
    ) -> None:
        state.stage = stage
        state.current_stage = stage

        if state.plan is not None:
            state.plan.concepts = state.concepts

    @staticmethod
    def _respond_response(
        session_id: str,
        state: LearningState,
        action: LearningAction,
        message: str,
        question: str | None,
        evaluation: Evaluation | None,
        events: list[LearningEvent],
    ) -> RespondLearningResponse:
        # Understanding checks do not require evaluation;
        # retain a typed neutral result for legacy response fields.
        result = evaluation or Evaluation(
            understanding="partial",
            confidence=0,
            next_action=action,
            feedback="",
        )

        return RespondLearningResponse(
            session_id=session_id,
            stage=state.stage,
            action=action,
            concept=state.current_concept.name,
            message=message,
            question=question,
            evaluation=result,
            learning_state=state,
            events=events,
        )

    @staticmethod
    def _continue_response(
        session_id: str,
        state: LearningState,
        action: LearningAction | None,
        message: str | None,
        question: str | None,
        events: list[LearningEvent],
        is_complete: bool = False,
        response_concept: str | None = None,
    ) -> ContinueLearningResponse:
        concept = (
            response_concept
            if response_concept is not None
            else (
                None
                if is_complete
                else state.current_concept.name
            )
        )

        return ContinueLearningResponse(
            session_id=session_id,
            stage=state.stage,
            action=action,
            concept=concept,
            message=message,
            question=question,
            learning_state=state,
            events=events,
            is_complete=is_complete,
        )