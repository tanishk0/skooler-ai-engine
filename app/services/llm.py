import json
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


class GeminiLLMClient:
    """The only module that knows about Google's GenAI SDK."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

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
                model=self.settings.gemini_model,
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
                model=self.settings.gemini_model,
                contents=[image_analysis_prompt(), types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                config=types.GenerateContentConfig(system_instruction=TUTOR_SYSTEM, temperature=0.1),
            )
            return (response.text or "").strip()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError("Gemini could not analyze this image") from exc
