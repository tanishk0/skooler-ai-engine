# Skooler AI Engine

Stateless FastAPI service for Skooler AI's adaptive, Feynman-style learning loop. Next.js owns authentication and MongoDB persistence; it sends the current `learning_state` to this service and persists the returned state.

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

Set `GEMINI_API_KEY` in `.env` (never commit it). `GEMINI_MODEL` defaults to `gemini-2.5-flash` and can be changed without touching application code. Gemini's structured JSON responses are validated as Pydantic models before the state machine uses them.

## API

`GET /health` returns `{ "status": "ok" }`.

`POST /ingestion/text` accepts notes and returns structure-aware chunks. `POST /ingestion/file` accepts PDF, JPEG, PNG, or WebP as multipart field `file`; PDFs and images are parsed with Unstructured (with a PDF text fallback). Image visual analysis is added through Gemini when a key is configured.

Start a topic-only session:

```json
POST /learning/start
{
  "session_id": "session-123",
  "input_type": "topic",
  "topic": "Recursion"
}
```

For text, include `input_type: "text"` and `content`; the service ingests it first. For a PDF/image, upload to `/ingestion/file`, then pass its returned `chunks` as `source_chunks` with `input_type` `pdf` or `image`.

Send each learner answer to `/learning/respond` with `session_id`, `user_response`, and the exact `learning_state` returned by the preceding call. The service intentionally stops after the resulting current artifact (a reteach, a transition message, or an evaluation); it never pre-generates the next interaction in that response.

When the returned state is a continuation state, call `POST /learning/continue` with `session_id` and the latest `learning_state`. The endpoint rejects invalid continuations with `409` and returns exactly one next pedagogical artifact. The core runtime stages are:

```text
awaiting_understanding_check
awaiting_continue_after_understood
awaiting_continue_after_reteach
awaiting_interaction_response
awaiting_continue_after_evaluation
```

Learning plans are internal state for persistence and orchestration. Learner-facing API events expose only the current teaching, check, evaluation, or transition—not the full curriculum.

## Architecture

One tutoring engine is used for all input types:

```text
input adapter -> Unstructured extraction -> structure-aware chunks -> lexical context selection
             -> Gemini plan/teaching/evaluation -> enforced Feynman state machine
```

There is deliberately no database, vector database, WebSocket, or background worker. `ContextSelector` is a small lexical retrieval interface that can later be replaced by embeddings without changing tutoring code.

## Tests

```bash
pytest
```

Tests use a fake LLM and make no network calls.
