from loguru import logger
from src.config import get_settings
from src.graph.neo4j_client import Neo4jClient
from src.verification.models import Evidence

settings = get_settings()


class KGRetriever:
    def __init__(self):
        self.client = Neo4jClient.get_instance()

    def find_by_entity(self, entity_name: str) -> list[dict]:
        cypher = """
        MATCH (s:Entity)-[r:RELATION]->(o:Entity)
        WHERE toLower(s.name) CONTAINS toLower($name)
        RETURN s.name AS subject, r.type AS predicate, o.name AS object
        LIMIT 20
        """
        return self.client.run_query(cypher, {"name": entity_name})

    def get_evidence_for_claim(
        self, subject: str, predicate: str, obj: str
    ) -> list[Evidence]:
        evidence = []
        seen = set()

        if subject:
            rows = self.find_by_entity(subject)
            for row in rows:
                content = (
                    f"{row['subject']} "
                    f"--[{row['predicate']}]--> "
                    f"{row['object']}"
                )
                if content not in seen:
                    seen.add(content)
                    evidence.append(Evidence(
                        source="neo4j",
                        content=content,
                        confidence=0.9,
                    ))

        return evidence