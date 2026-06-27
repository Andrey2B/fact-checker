from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from src.verification.pipeline import VerificationPipeline
from src.verification.models import VerificationReport
from src.graph.kg_builder import KGBuilder

router = APIRouter()

pipeline = VerificationPipeline()
builder = KGBuilder()


class VerifyRequest(BaseModel):
    text: str


class IngestRequest(BaseModel):
    text: str
    source: str = "manual"


class TripleRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0


@router.post("/verify", response_model=VerificationReport)
async def verify(req: VerifyRequest):
    if not req.text.strip():
        raise HTTPException(400, "Текст пуст")
    try:
        return pipeline.verify(req.text)
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise HTTPException(500, str(e))


@router.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        stats = builder.ingest_text(req.text, req.source)
        return {"status": "ok", "stats": stats}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/graph/triple")
async def add_triple(req: TripleRequest):
    try:
        result = builder.add_triple(
            req.subject, req.predicate, req.object, req.confidence
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/graph/entity/{name}")
async def get_entity(name: str):
    from src.graph.kg_retriever import KGRetriever
    retriever = KGRetriever()
    facts = retriever.find_by_entity(name)
    return {"entity": name, "facts": facts}