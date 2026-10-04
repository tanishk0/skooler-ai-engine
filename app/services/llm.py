import json
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.schemas.learning import Evaluation, InteractionPrompt, LearningPlan, TopicValidationResult
from app.services.prompts import TUTOR_SYSTEM, evaluation_prompt, image_analysis_prompt, interaction_prompt, plan_prompt, teaching_prompt, topic_validation_prompt

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMError(RuntimeError):
    """Safe error surfaced when a provider request cannot produce valid output."""


class TeachingTurn(BaseModel):
    message: str


class BaseLLMClient(ABC):
    """Abstract interface that all LLM provider clients implement."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider, e.g. 'gemini' or 'openai'."""

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True if the necessary credentials are configured."""

    @abstractmethod
    def generate_learning_plan(self, topic: str, source_context: str) -> LearningPlan:
        """Generate an internal adaptive learning curriculum."""

    @abstractmethod
    def validate_topic(self, topic: str) -> TopicValidationResult:
        """Validate whether the user-supplied topic is a learnable topic."""

    @abstractmethod
    def generate_teaching(self, **kwargs: object) -> TeachingTurn:
        """Generate a concise teaching step."""

    @abstractmethod
    def evaluate_explanation(self, **kwargs: object) -> Evaluation:
        """Evaluate learner explanation."""

    @abstractmethod
    def generate_interaction(self, **kwargs: object) -> InteractionPrompt:
        """Generate the next interaction question."""

    @abstractmethod
    def analyze_image(self, image_bytes: bytes, mime_type: str) -> str:
        """Analyze visual material and extract educational concepts."""


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini GenAI SDK provider implementation."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.model = settings.llm_model or settings.gemini_model
        self._client = None

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.gemini_api_key)

    def _get_client(self):
        if not self.is_configured:
            raise LLMError("Gemini is not configured. Set GEMINI_API_KEY before starting a learning session.")
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.settings.gemini_api_key)
            except Exception as exc:  # pragma: no cover - depends on installed SDK
                raise LLMError("Could not initialize the Gemini client") from exc
        return self._client

    def _structured(self, prompt: str, response_model: type[ModelT]) -> ModelT:
        try:
            from google.genai import types
            response = self._get_client().models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=TUTOR_SYSTEM,
                    response_mime_type="application/json",
                    response_schema=response_model,
                    temperature=0.25,
                ),
            )
            # SDK versions may expose parsed output, so accept either stable representation.
            if getattr(response, "parsed", None) is not None:
                parsed = response.parsed
                return parsed if isinstance(parsed, response_model) else response_model.model_validate(parsed)
            return response_model.model_validate_json(response.text)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            raise LLMError("Gemini returned an invalid structured response") from exc
        except LLMError:
            raise
        except Exception as exc:  # avoid leaking SDK/provider detail to API callers
            raise LLMError("Gemini request failed; please try again shortly") from exc

    def generate_learning_plan(self, topic: str, source_context: str) -> LearningPlan:
        return self._structured(plan_prompt(topic, source_context), LearningPlan)

    def validate_topic(self, topic: str) -> TopicValidationResult:
        return self._structured(topic_validation_prompt(topic), TopicValidationResult)

    def generate_teaching(self, **kwargs: object) -> TeachingTurn:
        return self._structured(teaching_prompt(**kwargs), TeachingTurn)  # type: ignore[arg-type]

    def evaluate_explanation(self, **kwargs: object) -> Evaluation:
        return self._structured(evaluation_prompt(**kwargs), Evaluation)  # type: ignore[arg-type]

    def generate_interaction(self, **kwargs: object) -> InteractionPrompt:
        return self._structured(interaction_prompt(**kwargs), InteractionPrompt)  # type: ignore[arg-type]

    def analyze_image(self, image_bytes: bytes, mime_type: str) -> str:
        """Use Gemini vision only when ordinary document extraction is insufficient."""
        try:
            from google.genai import types
            response = self._get_client().models.generate_content(
                model=self.model,
                contents=[image_analysis_prompt(), types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                config=types.GenerateContentConfig(system_instruction=TUTOR_SYSTEM, temperature=0.1),
            )
            return (response.text or "").strip()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError("Gemini could not analyze this image") from exc


class OpenAILLMClient(BaseLLMClient):
    """OpenAI SDK provider implementation (compatible with OpenAI, Groq, DeepSeek, Ollama, etc.)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.model = settings.llm_model or settings.openai_model
        self._client = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.openai_api_key)

    def _get_client(self):
        if not self.is_configured:
            raise LLMError("OpenAI is not configured. Set OPENAI_API_KEY before starting a learning session.")
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.settings.openai_api_key,
                    base_url=self.settings.openai_base_url,
                    timeout=self.settings.llm_timeout_seconds,
                )
            except Exception as exc:  # pragma: no cover
                raise LLMError("Could not initialize the OpenAI client") from exc
        return self._client

    def _structured(self, prompt: str, response_model: type[ModelT]) -> ModelT:
        try:
            completion = self._get_client().beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": TUTOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_model,
                temperature=0.25,
            )
            message = completion.choices[0].message
            if message.parsed is not None:
                return message.parsed
            if message.refusal:
                raise LLMError(f"OpenAI refused request: {message.refusal}")
            if message.content:
                return response_model.model_validate_json(message.content)
            raise LLMError("OpenAI returned an empty structured response")
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            raise LLMError("OpenAI returned an invalid structured response") from exc
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError("OpenAI request failed; please try again shortly") from exc

    def generate_learning_plan(self, topic: str, source_context: str) -> LearningPlan:
        return self._structured(plan_prompt(topic, source_context), LearningPlan)

    def validate_topic(self, topic: str) -> TopicValidationResult:
        return self._structured(topic_validation_prompt(topic), TopicValidationResult)

    def generate_teaching(self, **kwargs: object) -> TeachingTurn:
        return self._structured(teaching_prompt(**kwargs), TeachingTurn)  # type: ignore[arg-type]

    def evaluate_explanation(self, **kwargs: object) -> Evaluation:
        return self._structured(evaluation_prompt(**kwargs), Evaluation)  # type: ignore[arg-type]

    def generate_interaction(self, **kwargs: object) -> InteractionPrompt:
        return self._structured(interaction_prompt(**kwargs), InteractionPrompt)  # type: ignore[arg-type]

    def analyze_image(self, image_bytes: bytes, mime_type: str) -> str:
        """Use OpenAI vision to inspect images and diagrams."""
        try:
            import base64
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:{mime_type};base64,{encoded}"
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": TUTOR_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": image_analysis_prompt()},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.1,
            )
            return (response.choices[0].message.content or "").strip()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError("OpenAI could not analyze this image") from exc


def create_llm_client(settings: Settings) -> BaseLLMClient:
    """Factory function that returns the configured LLM provider client."""
    provider = settings.active_provider
    if provider == "openai":
        return OpenAILLMClient(settings)
    if provider == "gemini":
        return GeminiLLMClient(settings)
    raise LLMError(f"Unsupported LLM provider '{provider}'. Supported providers are: 'gemini', 'openai'.")


# Common alias for type hints
LLMClient = BaseLLMClient
