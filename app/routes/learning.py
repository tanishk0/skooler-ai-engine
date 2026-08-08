from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.learning import ContinueLearningRequest, ContinueLearningResponse, RespondLearningRequest, RespondLearningResponse, StartLearningRequest, StartLearningResponse, TopicValidationRequest, TopicValidationResult
from app.services.llm import LLMError
from app.services.orchestrator import InvalidTransitionError

router = APIRouter(prefix="/learning", tags=["learning"])


@router.post("/validate-topic", response_model=TopicValidationResult)
def validate_topic(payload: TopicValidationRequest, request: Request) -> TopicValidationResult:
    try:
        return request.app.state.llm.validate_topic(payload.topic)
    except LLMError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.post("/start", response_model=StartLearningResponse)
def start_learning(payload: StartLearningRequest, request: Request) -> StartLearningResponse:
    try:
        validation = request.app.state.llm.validate_topic(payload.topic)
        if not validation.is_valid:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=validation.message)
        payload.topic = validation.normalized_topic or payload.topic
        return request.app.state.orchestrator.start(payload.session_id, payload.input_type, payload.topic, payload.content, payload.source_chunks)
    except LLMError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/respond", response_model=RespondLearningResponse)
def respond_to_learner(payload: RespondLearningRequest, request: Request) -> RespondLearningResponse:
    try:
        return request.app.state.orchestrator.respond(payload.session_id or payload.learning_state.session_id, payload.learning_state, payload.user_response)
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.post("/continue", response_model=ContinueLearningResponse)
def continue_learning(payload: ContinueLearningRequest, request: Request) -> ContinueLearningResponse:
    try:
        return request.app.state.orchestrator.continue_learning(payload.session_id, payload.learning_state)
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
