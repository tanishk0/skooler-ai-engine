from fastapi.testclient import TestClient

from app.main import app


class RejectingTopicLLM:
    def validate_topic(self, topic):
        from app.schemas.learning import TopicValidationResult

        return TopicValidationResult(
            is_valid=False,
            message="Please enter a real concept, such as recursion, photosynthesis, or Python lists.",
        )


class MustNotStart:
    def start(self, *args, **kwargs):
        raise AssertionError("Invalid topics must not enter the learning pipeline")


def test_invalid_topic_is_rejected_before_learning_starts():
    with TestClient(app) as client:
        app.state.llm = RejectingTopicLLM()
        app.state.orchestrator = MustNotStart()
        response = client.post(
            "/learning/start",
            json={"sessionId": "invalid-topic", "inputType": "topic", "topic": "gu gu ga ga"},
        )
    assert response.status_code == 422
    assert "real concept" in response.json()["detail"]


def test_topic_validation_endpoint_returns_a_clear_rejection():
    with TestClient(app) as client:
        app.state.llm = RejectingTopicLLM()
        response = client.post("/learning/validate-topic", json={"topic": "gu gu ga ga"})
    assert response.status_code == 200
    assert response.json()["isValid"] is False
