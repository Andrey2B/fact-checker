import httpx
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from loguru import logger
from src.config import get_settings

settings = get_settings()


def check_ollama_available() -> bool:
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    if not check_ollama_available():
        raise ConnectionError(
            "Ollama не запущен! Запусти: ollama serve"
        )
    logger.info(
        f"LLM: {settings.ollama_model} @ {settings.ollama_base_url}"
    )
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_ctx=4096,
        timeout=120,
    )