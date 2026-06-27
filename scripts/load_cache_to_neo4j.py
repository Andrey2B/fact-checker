import sys
sys.path.append(".")

import json
import os
from neo4j import GraphDatabase

CACHE_FILE = "data/wikidata_cache.json"

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)


def main():
    print("=" * 60)
    print("ЗАГРУЗКА WIKIDATA КЭША В NEO4J")
    print("=" * 60)

    if not os.path.exists(CACHE_FILE):
        print(f"❌ Файл не найден: {CACHE_FILE}")
        return

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    print(f"Загружен кэш: {len(cache)} сущностей")
    total_facts = sum(len(v) for v in cache.values())
    print(f"Фактов всего: {total_facts}")
    print("Загружаем в Neo4j...")

    loaded = 0
    with driver.session() as session:
        # Создаём индексы
        session.run(
            "CREATE INDEX entity_name IF NOT EXISTS "
            "FOR (e:Entity) ON (e.name)"
        )
        session.run(
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name]"
        )

        for entity_name, facts in cache.items():
            if not facts:
                continue

            for fact in facts:
                subject = fact.get("subject", "")
                predicate = fact.get("predicate", "")
                obj = fact.get("object", "")

                if not subject or not predicate or not obj:
                    continue

                session.run("""
                    MERGE (s:Entity {name: $subject})
                    MERGE (o:Entity {name: $object})
                    MERGE (s)-[r:RELATION {
                        type: $predicate,
                        confidence: 1.0,
                        source: 'wikidata'
                    }]->(o)
                """, {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                })
                loaded += 1

            print(f"   {entity_name}: {len(facts)} фактов")

    print(f"\n Загружено {loaded} фактов в Neo4j!")

    with driver.session() as session:
        nodes = session.run(
            "MATCH (n:Entity) RETURN count(n) AS c"
        ).single()["c"]
        rels = session.run(
            "MATCH ()-[r:RELATION]->() RETURN count(r) AS c"
        ).single()["c"]
        print(f"Узлов в Neo4j:  {nodes}")
        print(f"Связей в Neo4j: {rels}")

    driver.close()
    print("\n Готово! Теперь всё работает локально!")


if __name__ == "__main__":
    main()