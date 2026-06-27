from loguru import logger
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

from src.config import get_settings
from src.llm.llm_client import get_llm
from src.graph.neo4j_client import Neo4jClient

settings = get_settings()


class KGBuilder:
    def __init__(self):
        self.client = Neo4jClient.get_instance()
        self.graph = Neo4jGraph(
            url=settings.neo4j_uri,
            username=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        self.llm = get_llm(temperature=0.0)
        self.transformer = LLMGraphTransformer(llm=self.llm)
        logger.info("KGBuilder инициализирован (локальная LLM)")

    def ingest_text(self, text: str, source: str = "unknown") -> dict:
        logger.info(f"Обработка текста из: {source}")
        documents = [
            Document(page_content=text, metadata={"source": source})
        ]
        graph_documents = self.transformer.convert_to_graph_documents(documents)
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True,
        )
        stats = {
            "nodes": sum(len(d.nodes) for d in graph_documents),
            "relationships": sum(len(d.relationships) for d in graph_documents),
            "source": source,
        }
        logger.info(f"Граф пополнен: {stats}")
        return stats

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
    ):
        cypher = """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        MERGE (s)-[r:RELATION {type: $predicate, confidence: $confidence}]->(o)
        RETURN s.name AS subject, r.type AS predicate, o.name AS object
        """
        return self.client.run_write(
            cypher,
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "confidence": confidence,
            },
        )