import sys
sys.path.append(".")

from loguru import logger
from src.graph.kg_builder import KGBuilder
from src.graph.neo4j_client import Neo4jClient

SEED_TRIPLES = [
    ("Эйфелева башня", "РАСПОЛОЖЕНА_В", "Париж"),
    ("Эйфелева башня", "ПОСТРОЕНА_В", "1889"),
    ("Эйфелева башня", "ВЫСОТА", "330 метров"),
    ("Эйфелева башня", "СПРОЕКТИРОВАНА", "Гюстав Эйфель"),
    ("Париж", "СТОЛИЦА", "Франция"),
    ("Москва", "СТОЛИЦА", "Россия"),
    ("Москва", "НАСЕЛЕНИЕ", "12 миллионов"),
    ("Москва", "ОСНОВАНА_В", "1147"),
    ("Альберт Эйнштейн", "РОДИЛСЯ_В", "1879"),
    ("Альберт Эйнштейн", "РАЗРАБОТАЛ", "Теория относительности"),
    ("Альберт Эйнштейн", "ГРАЖДАНСТВО", "Германия"),
    ("Вода", "ФОРМУЛА", "H2O"),
    ("Земля", "ВРАЩАЕТСЯ_ВОКРУГ", "Солнце"),
    ("Луна", "СПУТНИК", "Земля"),
    ("Python", "СОЗДАН", "Гвидо ван Россум"),
    ("Python", "ГОД_СОЗДАНИЯ", "1991"),
    ("Bitcoin", "СОЗДАН", "Сатоши Накамото"),
    ("Bitcoin", "ГОД_СОЗДАНИЯ", "2009"),
]


def main():
    logger.info("Заполнение графа знаний...")
    client = Neo4jClient.get_instance()
    client.create_indexes()
    builder = KGBuilder()

    for subj, pred, obj in SEED_TRIPLES:
        builder.add_triple(subj, pred, obj, confidence=1.0)
        logger.debug(f"  + ({subj})-[{pred}]->({obj})")

    logger.info(f" Добавлено {len(SEED_TRIPLES)} фактов")

    results = client.run_query(
        "MATCH (n:Entity) RETURN count(n) AS nodes"
    )
    logger.info(f"Узлов в графе: {results[0]['nodes']}")


if __name__ == "__main__":
    main()