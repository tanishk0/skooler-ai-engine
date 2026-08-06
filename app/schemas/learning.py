from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from app.schemas.base import APIModel
from app.schemas.ingestion import InputType, SourceChunk


# ============================================================
# ENUMS
# ============================================================

class ConceptStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PARTIAL = "partial"
    MASTERED = "mastered"


class LearningStage(str, Enum):
    AWAITING_UNDERSTANDING_CHECK = "awaiting_understanding_check"
    AWAITING_CONTINUE_AFTER_UNDERSTOOD = "awaiting_continue_after_understood"
    AWAITING_CONTINUE_AFTER_RETEACH = "awaiting_continue_after_reteach"
    AWAITING_INTERACTION_RESPONSE = "awaiting_interaction_response"
    AWAITING_CONTINUE_AFTER_EVALUATION = "awaiting_continue_after_evaluation"
    AWAITING_CONTINUE_AFTER_CONCEPT_TRANSITION = "awaiting_continue_after_concept_transition"
    COMPLETE = "complete"


class LearningAction(str, Enum):
    RETEACH = "reteach"
    CLARIFY = "clarify"
    TRY_AGAIN = "try_again"
    DEEPEN = "deepen"
    ASK_AGAIN = "ask_again"          # Legacy compatibility
    TEST_DEEPER = "test_deeper"      # Legacy compatibility
    TEACH_NEXT = "teach_next"
    COMPLETE = "complete"


class InteractionType(str, Enum):
    UNDERSTANDING_CHECK = "understanding_check"
    CHOICE = "choice"
    FEYNMAN = "feynman"
    PREDICTION = "prediction"
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"


class LearningEventType(str, Enum):
    TEACHING = "teaching"
    UNDERSTANDING_CHECK = "understanding_check"
    CHOICE = "choice"
    FEYNMAN = "feynman"
    PREDICTION = "prediction"
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    USER_ANSWER = "user_answer"
    EVALUATION = "evaluation"
    RETEACH = "reteach"
    MASTERED = "mastered"
    CONCEPT_TRANSITION = "concept_transition"


# ============================================================
# CURRICULUM
# ============================================================

class Concept(APIModel):
    id: str
    name: str
    description: str = ""
    status: ConceptStatus = ConceptStatus.NOT_STARTED
    attempts: int = 0

    module_id: str | None = None
    depth: int = 0
    target_depth: int = 2
    importance: str = "core"
    difficulty: int = 3
    prerequisites: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)
    teaching_depth: str = "standard"
    suitable_interactions: list[str] = Field(default_factory=list)
    mastery_evidence: list[str] = Field(default_factory=list)
    requires_application: bool = False


class LearningModule(APIModel):
    id: str = Field(default_factory=lambda: str(uuid4()))

    name: str
    description: str = ""

    # Major concepts belonging to this module.
    concepts: list[Concept] = Field(default_factory=list)

    # Optional prerequisite module IDs.
    prerequisites: list[str] = Field(default_factory=list)

    # Helps the UI and completion logic later.
    status: ConceptStatus = ConceptStatus.NOT_STARTED

    # Module-level progress, 0–100.
    progress: float = 0


class LearningPlan(APIModel):
    topic: str
    prerequisites: list[str] = Field(default_factory=list)

    # Default keeps old tests/legacy plans valid.
    modules: list[LearningModule] = Field(default_factory=list)

    concepts: list[Concept] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)
    mastery_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def synchronize_concepts(self) -> "LearningPlan":
        """
        Keep the legacy flat concept list synchronized with
        the new hierarchical module structure.

        New curriculum generation should populate `modules`.
        Existing code can continue reading `plan.concepts`.
        """

        if self.modules:
            flattened: list[Concept] = []

            for module in self.modules:
                flattened.extend(module.concepts)

            # If modules were supplied, they are the source
            # of truth.
            self.concepts = flattened

        if not self.concepts:
            raise ValueError("LearningPlan must contain at least one concept")

        return self


# ============================================================
# INTERACTIONS
# ============================================================

class InteractionOption(APIModel):
    id: str
    label: str


class InteractionPrompt(APIModel):
    type: InteractionType
    question: str
    options: list[InteractionOption] = Field(
        default_factory=list
    )


# ============================================================
# TIMELINE EVENTS
# ============================================================

class LearningEvent(APIModel):
    sequence: int = Field(default=1, ge=1)

    type: LearningEventType

    module_id: str | None = None
    concept_id: str | None = None

    content: str = ""

    question: str | None = None

    options: list[InteractionOption] = Field(
        default_factory=list
    )

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


# ============================================================
# EVALUATION
# ============================================================

class Evaluation(APIModel):
    understanding: Literal[
        "incorrect",
        "partial",
        "correct",
        "mastered",
    ]

    confidence: float = 0.8

    mastery_reached: bool = False

    misconceptions: list[str] = Field(
        default_factory=list
    )

    missing_concepts: list[str] = Field(
        default_factory=list
    )

    reasoning_quality: Literal[
        "weak",
        "adequate",
        "strong",
    ] = "adequate"

    next_action: LearningAction

    feedback: str = ""


# ============================================================
# LEARNING STATE
# ============================================================

class LearningState(APIModel):
    session_id: str

    topic: str

    input_type: InputType

    # --------------------------------------------------------
    # CURRICULUM
    # --------------------------------------------------------

    # Legacy flattened representation.
    #
    # Existing orchestrator code currently uses this heavily.
    concepts: list[Concept] = Field(
        min_length=1
    )

    # Full hierarchical curriculum.
    plan: LearningPlan | None = None

    # --------------------------------------------------------
    # CURRENT POSITION
    # --------------------------------------------------------

    # Current module within the curriculum.
    current_module_index: int = Field(
        default=0,
        ge=0
    )

    current_module_id: str | None = None

    # Current concept within the flattened compatibility list.
    current_concept_index: int = Field(
        default=0,
        ge=0
    )

    current_concept_id: str | None = None

    # --------------------------------------------------------
    # CONCEPT DEPTH
    # --------------------------------------------------------

    # How many meaningful pedagogical steps have been completed
    # for the current concept.
    concept_depth: int = Field(
        default=0,
        ge=0
    )

    # Evidence collected during the current concept.
    evidence_count: int = Field(
        default=0,
        ge=0
    )

    # --------------------------------------------------------
    # CURRENT LEARNING STAGE
    # --------------------------------------------------------

    current_stage: LearningStage = (
        LearningStage.AWAITING_UNDERSTANDING_CHECK
    )

    stage: LearningStage = (
        LearningStage.AWAITING_UNDERSTANDING_CHECK
    )

    current_interaction: InteractionPrompt | None = None

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    concept_progress: dict[str, float] = Field(
        default_factory=dict
    )

    module_progress: dict[str, float] = Field(
        default_factory=dict
    )

    mastery: float = Field(
        default=0,
        ge=0,
        le=100
    )

    # --------------------------------------------------------
    # EVALUATION / ADAPTATION
    # --------------------------------------------------------

    pending_action: LearningAction | None = None

    last_evaluation: Evaluation | None = None

    misconceptions: list[str] = Field(
        default_factory=list
    )

    recent_interactions: list[dict[str, str]] = Field(
        default_factory=list,
        max_length=8
    )

    # --------------------------------------------------------
    # EVENT TRACKING
    # --------------------------------------------------------

    event_sequence: int = Field(
        default=0,
        ge=0
    )

    # --------------------------------------------------------
    # SOURCE MATERIAL
    # --------------------------------------------------------

    source_chunks: list[SourceChunk] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def synchronize_curriculum(self) -> "LearningState":
        """
        Keep the flattened concept representation compatible
        with the hierarchical plan.

        The hierarchical plan becomes the long-term source of
        truth, while the flattened list keeps the current
        orchestrator functional during migration.
        """

        if self.plan is not None:
            if self.plan.modules:
                flattened: list[Concept] = []

                for module in self.plan.modules:
                    flattened.extend(module.concepts)

                if flattened:
                    self.concepts = flattened

        if not self.concepts:
            raise ValueError(
                "LearningState must contain at least one concept"
            )

        if (
            self.stage != LearningStage.COMPLETE
            and self.current_concept_index >= len(self.concepts)
        ):
            raise ValueError(
                "current_concept_index must identify a concept"
            )

        # Synchronize current concept ID.
        if (
            self.current_concept_index < len(self.concepts)
            and self.current_concept_id is None
        ):
            self.current_concept_id = (
                self.concepts[self.current_concept_index].id
            )

        # Synchronize module ID.
        if (
            self.plan is not None
            and self.plan.modules
            and self.current_module_index < len(self.plan.modules)
            and self.current_module_id is None
        ):
            self.current_module_id = (
                self.plan.modules[
                    self.current_module_index
                ].id
            )

        return self

    @property
    def current_concept(self) -> Concept:
        return self.concepts[self.current_concept_index]

    @property
    def current_module(self) -> LearningModule | None:
        if self.plan is None:
            return None

        if (
            self.current_module_index < 0
            or self.current_module_index >= len(self.plan.modules)
        ):
            return None

        return self.plan.modules[self.current_module_index]


# ============================================================
# START LEARNING
# ============================================================

class StartLearningRequest(APIModel):
    session_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    input_type: InputType = InputType.TOPIC

    topic: str = Field(
        min_length=1,
        max_length=300
    )

    content: str | None = Field(
        default=None,
        max_length=2_000_000
    )

    source_chunks: list[SourceChunk] | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "StartLearningRequest":
        if (
            self.input_type == InputType.TEXT
            and not self.content
            and not self.source_chunks
        ):
            raise ValueError(
                "text learning requires content or source_chunks"
            )

        if (
            self.input_type in {
                InputType.PDF,
                InputType.IMAGE
            }
            and not self.source_chunks
        ):
            raise ValueError(
                "file learning requires processed source_chunks "
                "from /ingestion/file"
            )

        return self


# ============================================================
# TOPIC VALIDATION
# ============================================================

class TopicValidationRequest(APIModel):
    topic: str = Field(
        min_length=1,
        max_length=300
    )


class TopicValidationResult(APIModel):
    is_valid: bool

    normalized_topic: str | None = None

    message: str


# ============================================================
# API RESPONSES
# ============================================================

class StartLearningResponse(APIModel):
    session_id: str

    stage: LearningStage

    concept: str

    message: str

    question: str

    learning_state: LearningState

    events: list[LearningEvent] = Field(
        default_factory=list
    )

    is_complete: bool = False


class RespondLearningRequest(APIModel):
    session_id: str | None = None

    user_response: str = Field(
        min_length=1,
        max_length=20_000
    )

    learning_state: LearningState

    @field_validator("user_response")
    @classmethod
    def response_must_not_be_blank(
        cls,
        value: str
    ) -> str:
        if not value.strip():
            raise ValueError(
                "user_response must not be blank"
            )

        return value

    @model_validator(mode="after")
    def use_state_session_id_when_omitted(
        self
    ) -> "RespondLearningRequest":
        if self.session_id is None:
            self.session_id = (
                self.learning_state.session_id
            )

        return self


class ContinueLearningRequest(APIModel):
    session_id: str

    learning_state: LearningState


class RespondLearningResponse(APIModel):
    session_id: str

    stage: LearningStage

    action: LearningAction

    concept: str | None = None

    message: str = ""

    question: str | None = None

    evaluation: Evaluation | None = None

    learning_state: LearningState

    events: list[LearningEvent] = Field(
        default_factory=list
    )

    is_complete: bool = False


class ContinueLearningResponse(APIModel):
    session_id: str

    stage: LearningStage

    action: LearningAction | None = None

    concept: str | None = None

    message: str | None = None

    question: str | None = None

    learning_state: LearningState

    events: list[LearningEvent] = Field(
        default_factory=list
    )

    is_complete: bool = False