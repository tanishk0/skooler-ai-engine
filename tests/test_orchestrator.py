import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
from app.schemas.ingestion import InputType
from app.schemas.learning import Concept, Evaluation, InteractionPrompt, InteractionType, LearningAction, LearningEventType, LearningPlan, LearningStage
from app.services.concepts import ConceptService
from app.services.context import ContextSelector
from app.services.evaluator import EvaluatorService
from app.services.ingestion import IngestionService
from app.services.llm import TeachingTurn
from app.services.orchestrator import InvalidTransitionError, LearningOrchestrator
from app.services.tutor import TutorService


class FakeLLM:
    def __init__(self, evaluations=None):
        self.evaluations = list(evaluations or [])

    def generate_learning_plan(self, topic, source_context):
        return LearningPlan(
            topic=topic,
            prerequisites=["Functions"],
            concepts=[Concept(id="recursion-basics", name="What recursion is"), Concept(id="base-case", name="Base cases")],
            common_misconceptions=["Recursion never stops"],
            mastery_questions=["Why does recursion need a base case?"],
        )

    def generate_teaching(self, **kwargs):
        return TeachingTurn(message=f"Teach only {kwargs['concept']}.")

    def generate_interaction(self, **kwargs):
        return InteractionPrompt(type=InteractionType.FEYNMAN, question=f"Explain {kwargs['concept']} in your own words.")

    def evaluate_explanation(self, **kwargs):
        return self.evaluations.pop(0) if self.evaluations else make_evaluation("correct")


def make_evaluation(understanding, misconceptions=None, mastery_reached=False, confidence=0.8, reasoning_quality="adequate", next_action=LearningAction.DEEPEN):
    return Evaluation(
        understanding=understanding,
        confidence=confidence,
        mastery_reached=mastery_reached,
        misconceptions=misconceptions or [],
        missing_concepts=[],
        reasoning_quality=reasoning_quality,
        next_action=next_action,
        feedback="Targeted evaluation feedback.",
    )


def make_engine(evaluations=None):
    llm = FakeLLM(evaluations)
    selector = ContextSelector()
    return LearningOrchestrator(ConceptService(llm, selector), TutorService(llm, selector), EvaluatorService(llm, selector), IngestionService(Settings()))


def start_engine(engine):
    return engine.start("session-1", InputType.TOPIC, "Recursion", None, None)


def assert_sequences(events, first):
    assert [event.sequence for event in events] == list(range(first, first + len(events)))


def test_start_stops_after_teaching_and_understanding_check():
    start = start_engine(make_engine())
    assert [event.type for event in start.events] == [LearningEventType.TEACHING, LearningEventType.UNDERSTANDING_CHECK]
    assert start.learning_state.stage == LearningStage.AWAITING_UNDERSTANDING_CHECK
    assert len(start.events[1].options) == 2
    assert_sequences(start.events, 1)


def test_understood_stops_at_transition_message():
    engine = make_engine()
    start = start_engine(engine)
    result = engine.respond("session-1", start.learning_state, "understood")
    assert [event.type for event in result.events] == [LearningEventType.USER_ANSWER, LearningEventType.CONCEPT_TRANSITION]
    assert result.learning_state.stage == LearningStage.AWAITING_CONTINUE_AFTER_UNDERSTOOD
    assert not any(event.type == LearningEventType.FEYNMAN for event in result.events)


def test_not_understood_stops_after_one_reteach():
    engine = make_engine()
    start = start_engine(engine)
    result = engine.respond("session-1", start.learning_state, "not_understood")
    assert [event.type for event in result.events] == [LearningEventType.USER_ANSWER, LearningEventType.RETEACH]
    assert result.learning_state.stage == LearningStage.AWAITING_CONTINUE_AFTER_RETEACH


def test_continue_after_reteach_returns_only_check():
    engine = make_engine()
    after_reteach = engine.respond("session-1", start_engine(engine).learning_state, "not_understood")
    result = engine.continue_learning("session-1", after_reteach.learning_state)
    assert [event.type for event in result.events] == [LearningEventType.UNDERSTANDING_CHECK]
    assert result.learning_state.stage == LearningStage.AWAITING_UNDERSTANDING_CHECK


def test_continue_after_understood_returns_one_interaction():
    engine = make_engine()
    after_understood = engine.respond("session-1", start_engine(engine).learning_state, "understood")
    result = engine.continue_learning("session-1", after_understood.learning_state)
    assert [event.type for event in result.events] == [LearningEventType.FEYNMAN]
    assert result.learning_state.stage == LearningStage.AWAITING_INTERACTION_RESPONSE


def test_interaction_response_stops_after_evaluation():
    engine = make_engine([make_evaluation("partial", ["Base case is missing"])])
    after_understood = engine.respond("session-1", start_engine(engine).learning_state, "understood")
    interaction = engine.continue_learning("session-1", after_understood.learning_state)
    result = engine.respond("session-1", interaction.learning_state, "It keeps calling itself.")
    assert [event.type for event in result.events] == [LearningEventType.USER_ANSWER, LearningEventType.EVALUATION]
    assert result.learning_state.stage == LearningStage.AWAITING_CONTINUE_AFTER_EVALUATION
    assert not any(event.type == LearningEventType.RETEACH for event in result.events)


def test_continue_after_evaluation_returns_one_next_step():
    engine = make_engine([make_evaluation("partial", ["Base case is missing"])])
    interaction = engine.continue_learning("session-1", engine.respond("session-1", start_engine(engine).learning_state, "understood").learning_state)
    evaluation = engine.respond("session-1", interaction.learning_state, "It repeats.")
    result = engine.continue_learning("session-1", evaluation.learning_state)
    assert [event.type for event in result.events] == [LearningEventType.RETEACH]
    assert result.learning_state.stage == LearningStage.AWAITING_CONTINUE_AFTER_RETEACH


def test_strong_feynman_answer_marks_mastered_and_selects_next_concept():
    strong = make_evaluation("correct", mastery_reached=True, confidence=0.94, reasoning_quality="strong", next_action=LearningAction.TEACH_NEXT)
    engine = make_engine([strong])
    start = start_engine(engine)
    challenge = engine.continue_learning("session-1", engine.respond("session-1", start.learning_state, "understood").learning_state)
    evaluation = engine.respond("session-1", challenge.learning_state, "Recursion solves a smaller version of the same problem and a base case stops it.")
    assert evaluation.learning_state.pending_action == LearningAction.TEACH_NEXT
    transition = engine.continue_learning("session-1", evaluation.learning_state)
    assert [event.type for event in transition.events] == [LearningEventType.MASTERED, LearningEventType.CONCEPT_TRANSITION]
    assert transition.learning_state.concepts[0].status.value == "mastered"
    assert transition.learning_state.current_concept_id == "base-case"
    assert transition.learning_state.current_concept_id != start.learning_state.concepts[0].id
    assert transition.concept == "What recursion is"
    assert transition.events[0].metadata["completed_concept_name"] == "What recursion is"
    next_teaching = engine.continue_learning("session-1", transition.learning_state)
    assert [event.type for event in next_teaching.events] == [LearningEventType.TEACHING, LearningEventType.UNDERSTANDING_CHECK]
    assert all(event.concept_id == "base-case" for event in next_teaching.events)


def test_strong_answer_after_reteach_advances_without_old_concept_loop():
    strong = make_evaluation("mastered", mastery_reached=True, confidence=0.95, reasoning_quality="strong", next_action=LearningAction.TEACH_NEXT)
    engine = make_engine([strong])
    initial = start_engine(engine)
    reteach = engine.respond("session-1", initial.learning_state, "not_understood")
    check = engine.continue_learning("session-1", reteach.learning_state)
    transition = engine.respond("session-1", check.learning_state, "understood")
    challenge = engine.continue_learning("session-1", transition.learning_state)
    evaluation = engine.respond("session-1", challenge.learning_state, "A base case stops recursive calls after reducing the problem.")
    advanced = engine.continue_learning("session-1", evaluation.learning_state)
    assert advanced.learning_state.current_concept_id == "base-case"
    assert advanced.learning_state.concepts[0].attempts == 1


def test_last_concept_mastery_completes_without_another_challenge():
    strong_first = make_evaluation("mastered", mastery_reached=True, confidence=0.95, reasoning_quality="strong", next_action=LearningAction.TEACH_NEXT)
    strong_last = make_evaluation("mastered", mastery_reached=True, confidence=0.95, reasoning_quality="strong", next_action=LearningAction.COMPLETE)
    engine = make_engine([strong_first, strong_last])
    start = start_engine(engine)
    first_challenge = engine.continue_learning("session-1", engine.respond("session-1", start.learning_state, "understood").learning_state)
    first_eval = engine.respond("session-1", first_challenge.learning_state, "Correct explanation")
    transition = engine.continue_learning("session-1", first_eval.learning_state)
    next_start = engine.continue_learning("session-1", transition.learning_state)
    last_challenge = engine.continue_learning("session-1", engine.respond("session-1", next_start.learning_state, "understood").learning_state)
    last_eval = engine.respond("session-1", last_challenge.learning_state, "Correct explanation of base cases")
    completed = engine.continue_learning("session-1", last_eval.learning_state)
    assert completed.is_complete
    assert completed.learning_state.stage == LearningStage.COMPLETE
    assert [event.type for event in completed.events] == [LearningEventType.MASTERED]


def test_invalid_continue_and_stale_response_are_rejected():
    engine = make_engine()
    start = start_engine(engine)
    with pytest.raises(InvalidTransitionError):
        engine.continue_learning("session-1", start.learning_state)
    progressed = engine.respond("session-1", start.learning_state, "understood")
    with pytest.raises(InvalidTransitionError):
        engine.respond("session-1", progressed.learning_state, "understood")


def test_continue_endpoint_returns_409_in_an_invalid_state():
    engine = make_engine()
    start = start_engine(engine)
    with TestClient(app) as client:
        app.state.orchestrator = engine
        response = client.post("/learning/continue", json={"sessionId": "session-1", "learningState": start.learning_state.model_dump(by_alias=True)})
    assert response.status_code == 409
