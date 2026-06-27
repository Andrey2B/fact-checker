"""
Теперь все данные в Neo4j — никаких внешних запросов!
"""
from loguru import logger
from src.verification.models import Evidence


class WikidataRetriever:
    """
    Заглушка — все данные теперь в Neo4j.
    Этот класс больше не нужен для внешних запросов.
    """

    def get_evidence_for_claim(
        self,
        subject: str,
        predicate: str,
        obj: str,
        claim_text: str = ""
    ) -> list[Evidence]:
        # Всё в Neo4j — смотри KGRetriever
        return []