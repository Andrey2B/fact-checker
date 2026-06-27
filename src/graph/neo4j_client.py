from neo4j import GraphDatabase, Driver
from loguru import logger
from src.config import get_settings

settings = get_settings()


class Neo4jClient:
    _instance: "Neo4jClient | None" = None

    def __init__(self):
        self.driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info(f"Neo4j подключён: {settings.neo4j_uri}")

    @classmethod
    def get_instance(cls) -> "Neo4jClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def close(self):
        if self.driver:
            self.driver.close()

    def run_query(self, cypher: str, params: dict | None = None) -> list[dict]:
        params = params or {}
        with self.driver.session(database=settings.neo4j_database) as session:
            result = session.run(cypher, params)
            return [record.data() for record in result]

    def run_write(self, cypher: str, params: dict | None = None) -> list[dict]:
        params = params or {}
        with self.driver.session(database=settings.neo4j_database) as session:
            result = session.execute_write(
                lambda tx: list(tx.run(cypher, params))
            )
            return [record.data() for record in result]

    def health_check(self) -> bool:
        try:
            self.run_query("RETURN 1 AS ok")
            return True
        except Exception as e:
            logger.error(f"Neo4j недоступен: {e}")
            return False

    def create_indexes(self):
        queries = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name]",
        ]
        for q in queries:
            try:
                self.run_write(q)
            except Exception:
                pass