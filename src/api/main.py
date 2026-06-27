from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from src.config import get_settings
from src.graph.neo4j_client import Neo4jClient
from src.llm.llm_client import check_ollama_available
from src.api.routes import router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Запуск KG Fact Checker (LOCAL)")

    neo4j = Neo4jClient.get_instance()
    neo4j.create_indexes()
    neo4j_ok = neo4j.health_check()
    ollama_ok = check_ollama_available()

    logger.info(f"Neo4j: {'✅' if neo4j_ok else '❌'}")
    logger.info(f"Ollama: {'✅' if ollama_ok else '❌'}")
    logger.info(f"Модель: {settings.ollama_model}")
    logger.info(
        f"Эмбеддинги: "
        f"{settings.embedding_provider}/{settings.embedding_model}"
    )

    yield

    neo4j.close()
    logger.info("Завершение работы")


app = FastAPI(
    title="KG Fact Checker (Local)",
    description="Верификация фактов: Ollama + Neo4j",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "neo4j": Neo4jClient.get_instance().health_check(),
        "ollama": check_ollama_available(),
        "model": settings.ollama_model,
        "embeddings": (
            f"{settings.embedding_provider}/{settings.embedding_model}"
        ),
    }