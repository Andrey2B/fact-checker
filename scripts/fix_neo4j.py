import sys
sys.path.append(".")

from loguru import logger
from src.graph.neo4j_client import Neo4jClient


def main():
    client = Neo4jClient.get_instance()



    # 1. Смотрим все факты об Эйфелевой башне
    logger.info("=== Факты об Эйфелевой башне ===")
    rows = client.run_query("""
        MATCH (s:Entity)-[r:RELATION]->(o:Entity)
        WHERE toLower(s.name) CONTAINS 'эйфел'
        RETURN s.name AS subject, r.type AS predicate, o.name AS object
        ORDER BY r.type
    """)
    for row in rows:
        logger.info(f"  ({row['subject']}) -[{row['predicate']}]-> ({row['object']})")

    # 2. Смотрим все факты об Альберте Эйнштейне
    logger.info("=== Факты об Эйнштейне ===")
    rows = client.run_query("""
        MATCH (s:Entity)-[r:RELATION]->(o:Entity)
        WHERE toLower(s.name) CONTAINS 'эйнш'
           OR toLower(s.name) CONTAINS 'einst'
        RETURN s.name AS subject, r.type AS predicate, o.name AS object
        ORDER BY r.type
    """)
    for row in rows:
        logger.info(f"  ({row['subject']}) -[{row['predicate']}]-> ({row['object']})")

    # 3. Считаем общую статистику графа
    logger.info("=== Статистика графа ===")
    rows = client.run_query("""
        MATCH (n:Entity) RETURN count(n) AS nodes
    """)
    logger.info(f"  Узлов: {rows[0]['nodes']}")

    rows = client.run_query("""
        MATCH ()-[r:RELATION]->() RETURN count(r) AS rels
    """)
    logger.info(f"  Рёбер: {rows[0]['rels']}")

    # 4. Удаляем конфликтующие факты с 1887
    logger.info("=== Удаляем конфликтующие факты (1887) ===")
    result = client.run_query("""
        MATCH (s:Entity)-[r:RELATION]->(o:Entity)
        WHERE toLower(s.name) CONTAINS 'эйфел'
          AND o.name CONTAINS '1887'
        RETURN s.name AS subject, r.type AS predicate, o.name AS object
    """)
    if result:
        for row in result:
            logger.warning(
                f"  Найден конфликт: "
                f"({row['subject']}) -[{row['predicate']}]-> ({row['object']})"
            )
        client.run_query("""
            MATCH (s:Entity)-[r:RELATION]->(o:Entity)
            WHERE toLower(s.name) CONTAINS 'эйфел'
              AND o.name CONTAINS '1887'
            DELETE r
        """)
        logger.success(f"  Удалено {len(result)} конфликтующих рёбер")
    else:
        logger.info("  Конфликтов с 1887 не найдено")

    # 5. Удаляем слабые предикаты Wikidata для Эйфелевой башни
    weak_predicates = [
        'inception', 'start_time', 'end_time',
        'dissolved', 'latest_date', 'earliest_date',
    ]
    logger.info("=== Удаляем слабые предикаты Wikidata ===")
    for pred in weak_predicates:
        result = client.run_query("""
            MATCH (s:Entity)-[r:RELATION]->(o:Entity)
            WHERE toLower(s.name) CONTAINS 'эйфел'
              AND toLower(r.type) = $pred
            RETURN s.name AS subject, r.type AS predicate, o.name AS object
        """, {"pred": pred})
        if result:
            for row in result:
                logger.warning(
                    f"  Слабый предикат: "
                    f"({row['subject']}) -[{row['predicate']}]-> ({row['object']})"
                )
            client.run_query("""
                MATCH (s:Entity)-[r:RELATION]->(o:Entity)
                WHERE toLower(s.name) CONTAINS 'эйфел'
                  AND toLower(r.type) = $pred
                DELETE r
            """, {"pred": pred})
            logger.success(f"  Удалено {len(result)} рёбер с предикатом '{pred}'")

    # 6. Проверяем результат — факты об Эйфелевой башне после очистки
    logger.info("=== Эйфелева башня ПОСЛЕ очистки ===")
    rows = client.run_query("""
        MATCH (s:Entity)-[r:RELATION]->(o:Entity)
        WHERE toLower(s.name) CONTAINS 'эйфел'
        RETURN s.name AS subject, r.type AS predicate, o.name AS object
        ORDER BY r.type
    """)
    for row in rows:
        logger.info(
            f"  ({row['subject']}) -[{row['predicate']}]-> ({row['object']})"
        )

        # 7. Удаляем мусорные высоты Эйфелевой башни (300, 324)
        logger.info("=== Чистим дублирующиеся высоты ===")
        result = client.run_query("""
            MATCH (s:Entity)-[r:RELATION]->(o:Entity)
            WHERE toLower(s.name) CONTAINS 'эйфел'
              AND toLower(r.type) CONTAINS 'высот'
              AND o.name IN ['300', '324']
            RETURN s.name AS subject, r.type AS predicate, o.name AS object
        """)
        if result:
            for row in result:
                logger.warning(
                    f"  Мусор: ({row['subject']}) "
                    f"-[{row['predicate']}]-> ({row['object']})"
                )
            client.run_query("""
                MATCH (s:Entity)-[r:RELATION]->(o:Entity)
                WHERE toLower(s.name) CONTAINS 'эйфел'
                  AND toLower(r.type) CONTAINS 'высот'
                  AND o.name IN ['300', '324']
                DELETE r
            """)
            logger.success(f"  Удалено {len(result)} мусорных рёбер высоты")
        else:
            logger.info("  Мусорных высот не найдено")

        # 8. Добавляем место рождения Эйнштейна
        logger.info("=== Добавляем МЕСТО_РОЖДЕНИЯ Эйнштейна ===")

        # Проверяем — вдруг уже есть
        existing = client.run_query("""
            MATCH (s:Entity)-[r:RELATION]->(o:Entity)
            WHERE toLower(s.name) CONTAINS 'эйнш'
              AND toLower(r.type) CONTAINS 'рожд'
              AND toLower(o.name) CONTAINS 'герман'
            RETURN s.name AS subject, r.type AS predicate, o.name AS object
        """)
        if existing:
            logger.info("  МЕСТО_РОЖДЕНИЯ уже есть — пропускаем")
        else:
            client.run_query("""
                MERGE (s:Entity {name: 'Альберт Эйнштейн'})
                MERGE (o:Entity {name: 'Германия'})
                MERGE (s)-[:RELATION {type: 'МЕСТО_РОЖДЕНИЯ'}]->(o)
            """)
            logger.success("  Добавлен факт: Эйнштейн -[МЕСТО_РОЖДЕНИЯ]-> Германия")

        # 9. Финальная проверка обоих объектов
        logger.info("=== ИТОГ: Эйфелева башня ===")
        rows = client.run_query("""
            MATCH (s:Entity)-[r:RELATION]->(o:Entity)
            WHERE toLower(s.name) CONTAINS 'эйфел'
            RETURN s.name AS subject, r.type AS predicate, o.name AS object
            ORDER BY r.type
        """)
        for row in rows:
            logger.info(
                f"  ({row['subject']}) -[{row['predicate']}]-> ({row['object']})"
            )

        logger.info("=== ИТОГ: Альберт Эйнштейн ===")
        rows = client.run_query("""
            MATCH (s:Entity)-[r:RELATION]->(o:Entity)
            WHERE toLower(s.name) CONTAINS 'эйнш'
            RETURN s.name AS subject, r.type AS predicate, o.name AS object
            ORDER BY r.type
        """)
        for row in rows:
            logger.info(
                f"  ({row['subject']}) -[{row['predicate']}]-> ({row['object']})"
            )

    client.run_query("""
        MATCH (s:Entity)-[r:RELATION]->(o:Entity)
        WHERE toLower(r.type) CONTAINS 'населен'
           OR toLower(r.type) = 'population'
        WITH s, r.type AS pred, collect(o.name) AS vals
        WHERE size(vals) > 1
        RETURN s.name, pred, vals
    """)
if __name__ == "__main__":
    main()