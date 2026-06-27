from functools import lru_cache
from loguru import logger
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import get_settings

settings = get_settings()


@lru_cache()
def get_embeddings() -> Embeddings:
    logger.info(
        f"Embeddings: HuggingFace / {settings.embedding_model} "
        f"[{settings.embedding_device}]"
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )