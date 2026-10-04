from unittest.mock import MagicMock, patch
import pytest

from app.core.config import Settings
from app.schemas.learning import Concept, Evaluation, InteractionPrompt, InteractionType, LearningAction, LearningPlan, TopicValidationResult
from app.services.llm import BaseLLMClient, GeminiLLMClient, LLMError, OpenAILLMClient, TeachingTurn, create_llm_client


def test_settings_active_provider_resolution():
    # Explicit gemini
    s1 = Settings(llm_provider="gemini", gemini_api_key="gem-key", openai_api_key="oa-key")
    assert s1.active_provider == "gemini"

    # Explicit openai
    s2 = Settings(llm_provider="openai", gemini_api_key="gem-key", openai_api_key="oa-key")
    assert s2.active_provider == "openai"

    # Auto: only OpenAI key provided
    s3 = Settings(llm_provider="auto", gemini_api_key=None, openai_api_key="sk-test")
    assert s3.active_provider == "openai"

    # Auto: only Gemini key provided
    s4 = Settings(llm_provider="auto", gemini_api_key="AIzaSyTest", openai_api_key=None)
    assert s4.active_provider == "gemini"

    # Auto: both provided -> backwards-compatible default to gemini
    s5 = Settings(llm_provider="auto", gemini_api_key="AIzaSyTest", openai_api_key="sk-test")
    assert s5.active_provider == "gemini"


def test_create_llm_client_factory():
    s_gemini = Settings(llm_provider="gemini", gemini_api_key="gemini-key")
    client_gemini = create_llm_client(s_gemini)
    assert isinstance(client_gemini, GeminiLLMClient)
    assert client_gemini.provider_name == "gemini"
    assert client_gemini.is_configured is True

    s_openai = Settings(llm_provider="openai", openai_api_key="sk-openai-key")
    client_openai = create_llm_client(s_openai)
    assert isinstance(client_openai, OpenAILLMClient)
    assert client_openai.provider_name == "openai"
    assert client_openai.is_configured is True

    s_invalid = Settings(llm_provider="anthropic")
    with pytest.raises(LLMError, match="Unsupported LLM provider 'anthropic'"):
        create_llm_client(s_invalid)


def test_unconfigured_clients_raise_error():
    s_gemini = Settings(gemini_api_key=None)
    client_gemini = GeminiLLMClient(s_gemini)
    assert client_gemini.is_configured is False
    with pytest.raises(LLMError, match="Gemini is not configured"):
        client_gemini._get_client()

    s_openai = Settings(openai_api_key=None)
    client_openai = OpenAILLMClient(s_openai)
    assert client_openai.is_configured is False
    with pytest.raises(LLMError, match="OpenAI is not configured"):
        client_openai._get_client()


def test_openai_structured_plan_generation():
    settings = Settings(openai_api_key="sk-test", openai_model="gpt-4o-mini")
    client = OpenAILLMClient(settings)

    mock_plan = LearningPlan(
        topic="Recursion",
        prerequisites=[],
        concepts=[Concept(id="c1", name="Base Case", description="Stopping condition")],
    )

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_plan
    mock_choice.message.refusal = None
    mock_completion.choices = [mock_choice]

    mock_openai_instance = MagicMock()
    mock_openai_instance.beta.chat.completions.parse.return_value = mock_completion
    client._client = mock_openai_instance

    result = client.generate_learning_plan("Recursion", "")
    assert result.topic == "Recursion"
    assert len(result.concepts) == 1
    assert result.concepts[0].name == "Base Case"


def test_openai_validate_topic():
    settings = Settings(openai_api_key="sk-test")
    client = OpenAILLMClient(settings)

    mock_val = TopicValidationResult(is_valid=True, normalized_topic="Binary Search", message="Valid")
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_val
    mock_choice.message.refusal = None
    mock_completion.choices = [mock_choice]

    mock_openai_instance = MagicMock()
    mock_openai_instance.beta.chat.completions.parse.return_value = mock_completion
    client._client = mock_openai_instance

    result = client.validate_topic("binary search")
    assert result.is_valid is True
    assert result.normalized_topic == "Binary Search"


def test_openai_teaching_and_interaction():
    settings = Settings(openai_api_key="sk-test")
    client = OpenAILLMClient(settings)

    # Teaching
    mock_teaching = TeachingTurn(message="This is a test lesson.")
    mock_comp1 = MagicMock()
    mock_comp1.choices = [MagicMock(message=MagicMock(parsed=mock_teaching, refusal=None))]

    # Interaction
    mock_interaction = InteractionPrompt(type=InteractionType.FEYNMAN, question="Explain in your words.")
    mock_comp2 = MagicMock()
    mock_comp2.choices = [MagicMock(message=MagicMock(parsed=mock_interaction, refusal=None))]

    mock_openai_instance = MagicMock()
    mock_openai_instance.beta.chat.completions.parse.side_effect = [mock_comp1, mock_comp2]
    client._client = mock_openai_instance

    t_res = client.generate_teaching(
        topic="Math", concept="Addition", description="Adding", source_context="",
        misconceptions=[], prior_answer=None, strategy="standard"
    )
    assert t_res.message == "This is a test lesson."

    i_res = client.generate_interaction(
        concept="Addition", teaching="Lesson", strategy="standard", attempts=1, misconceptions=[]
    )
    assert i_res.type == InteractionType.FEYNMAN
    assert i_res.question == "Explain in your words."


def test_openai_refusal_raises_error():
    settings = Settings(openai_api_key="sk-test")
    client = OpenAILLMClient(settings)

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.parsed = None
    mock_choice.message.refusal = "Safety policy violation"
    mock_completion.choices = [mock_choice]

    mock_openai_instance = MagicMock()
    mock_openai_instance.beta.chat.completions.parse.return_value = mock_completion
    client._client = mock_openai_instance

    with pytest.raises(LLMError, match="OpenAI refused request: Safety policy violation"):
        client.validate_topic("something unsafe")


def test_openai_analyze_image():
    settings = Settings(openai_api_key="sk-test")
    client = OpenAILLMClient(settings)

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Diagram of a plant cell"))]

    mock_openai_instance = MagicMock()
    mock_openai_instance.chat.completions.create.return_value = mock_completion
    client._client = mock_openai_instance

    desc = client.analyze_image(b"fake-image-bytes", "image/png")
    assert desc == "Diagram of a plant cell"
