from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routes import ingestion, learning
from app.services.concepts import ConceptService
from app.services.context import ContextSelector
from app.services.evaluator import EvaluatorService
from app.services.ingestion import IngestionService
from app.services.llm import create_llm_client
from app.services.orchestrator import LearningOrchestrator
from app.services.tutor import TutorService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    context_selector = ContextSelector()
    llm = create_llm_client(settings)
    ingestion_service = IngestionService(settings)
    app.state.settings = settings
    app.state.llm = llm
    app.state.ingestion_service = ingestion_service
    app.state.orchestrator = LearningOrchestrator(
        ConceptService(llm, context_selector),
        TutorService(llm, context_selector),
        EvaluatorService(llm, context_selector),
        ingestion_service,
    )
    yield


app = FastAPI(title="Skooler AI Engine", version="0.1.0", lifespan=lifespan)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(ingestion.router)
app.include_router(learning.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
