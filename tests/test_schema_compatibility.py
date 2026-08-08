from app.schemas.learning import StartLearningRequest, TopicValidationResult


def test_start_request_accepts_nextjs_camel_case_payload():
    payload = StartLearningRequest.model_validate(
        {"sessionId": "session-1", "inputType": "topic", "topic": "Recursion"}
    )
    assert payload.session_id == "session-1"
    assert payload.input_type.value == "topic"


def test_models_serialize_in_camel_case_for_nextjs():
    payload = StartLearningRequest(session_id="session-1", input_type="topic", topic="Recursion")
    assert payload.model_dump(by_alias=True)["sessionId"] == "session-1"
    assert payload.model_dump(by_alias=True)["inputType"] == "topic"


def test_topic_validation_serializes_for_the_nextjs_preflight():
    result = TopicValidationResult(is_valid=False, message="Please enter a real concept.")
    assert result.model_dump(by_alias=True) == {
        "isValid": False,
        "normalizedTopic": None,
        "message": "Please enter a real concept.",
    }
