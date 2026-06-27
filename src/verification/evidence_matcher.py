from loguru import logger
from src.graph.neo4j_client import Neo4jClient
from src.verification.models import AtomicClaim, Evidence

# Предикаты которые часто дают ложные конфликты
# (начало строительства ≠ год открытия и т.д.)
WEAK_PREDICATES = {
    "inception", "start_time", "end_time",
    "dissolved", "latest_date", "earliest_date",
    "modified", "retrieved",
}

# Суффиксы русских падежей — от длинных к коротким,
# чтобы не обрезать лишнее
RU_SUFFIXES = [
    "ями", "ами", "ого", "ему", "ому",
    "ях", "ах", "ой", "ем", "ом", "ию",
    "ии", "ье", "ья", "ей",
    "ю", "я", "е", "и",
]


class EvidenceMatcher:
    def __init__(self):
        self.client = Neo4jClient.get_instance()

    def _normalize(self, text: str) -> str:
        """
        Нормализация: убираем падежные окончания.
        Проверяем суффиксы от длинных к коротким.
        Минимальная длина основы — 3 символа.
        """
        text = text.strip()
        lower = text.lower()
        for suffix in RU_SUFFIXES:
            if lower.endswith(suffix):
                stem = text[:-len(suffix)]
                if len(stem) >= 3:           # не обрезаем до пустоты
                    return stem
        return text

    def _search_by_name(self, name: str, limit: int = 5) -> list[dict]:
        """
        Ищет факты по точному имени субъекта.
        Без мусорных candidates — только оригинал и первое слово.
        """
        if not name or len(name) < 2:
            return []

        # Только осмысленные варианты
        candidates = [name]
        words = name.split()
        if len(words) > 1:
            candidates.append(words[0])   # "Альберт" из "Альберт Эйнштейн"

        results = []
        seen = set()

        for candidate in candidates:
            if len(candidate) < 2:
                continue

            rows = self.client.run_query(f"""
                MATCH (s:Entity)-[r:RELATION]->(o:Entity)
                WHERE toLower(s.name) CONTAINS toLower($name)
                RETURN
                    s.name AS subject,
                    r.type AS predicate,
                    o.name AS object
                LIMIT {limit}
            """, {"name": candidate})

            for row in rows:
                content = (
                    f"{row['subject']} "
                    f"--[{row['predicate']}]--> "
                    f"{row['object']}"
                )
                if content not in seen:
                    seen.add(content)
                    results.append(row)

        return results

    def _search_exact_object(
        self, subject: str, obj: str
    ) -> list[dict]:
        """
        Ищет факты где субъект и объект совпадают.
        """
        if not subject or not obj or len(obj) < 2:
            return []

        rows = self.client.run_query("""
            MATCH (s:Entity)-[r:RELATION]->(o:Entity)
            WHERE toLower(s.name) CONTAINS toLower($subject)
              AND toLower(o.name) CONTAINS toLower($object)
            RETURN
                s.name AS subject,
                r.type AS predicate,
                o.name AS object
            LIMIT 3
        """, {"subject": subject, "object": obj})

        return rows

    def _fulltext_search(self, query: str) -> list[dict]:
        try:
            return self.client.run_query("""
                CALL db.index.fulltext.queryNodes(
                    'entity_fulltext', $query
                )
                YIELD node, score
                MATCH (node)-[r:RELATION]->(o:Entity)
                RETURN
                    node.name AS subject,
                    r.type AS predicate,
                    o.name AS object
                ORDER BY score DESC
                LIMIT 3
            """, {"query": query})
        except Exception:
            return []

    def match(self, claim: AtomicClaim) -> list[Evidence]:
        evidence = []
        seen_content = set()

        def add(rows: list[dict], source: str, confidence: float):
            for row in rows:
                subject = row.get("subject") or ""
                predicate = row.get("predicate") or ""
                obj = row.get("object") or ""

                if not subject or not predicate or not obj:
                    continue

                # Понижаем вес слабых предикатов Wikidata
                if predicate.lower() in WEAK_PREDICATES:
                    confidence = min(confidence, 0.4)
                    logger.debug(
                        f"  Слабый предикат '{predicate}' "
                        f"→ confidence снижен до 0.4"
                    )

                content = f"{subject} --[{predicate}]--> {obj}"
                if content not in seen_content:
                    seen_content.add(content)
                    evidence.append(Evidence(
                        source=source,
                        content=content,
                        confidence=confidence,
                    ))

        subj = claim.subject or ""
        obj = claim.object or ""
        obj_norm = self._normalize(obj)

        # 1. Точный поиск субъект + объект (оригинал)
        if subj and obj:
            logger.debug(f"Точный поиск: '{subj}' + '{obj}'")
            rows = self._search_exact_object(subj, obj)
            add(rows, "neo4j_exact", 1.0)
            logger.debug(f"Точный поиск: {len(rows)} фактов")

        # 2. Точный поиск с нормализованным объектом
        #    "Париже" → "Париж", "Германии" → "Герман" (≥3 символа)
        if subj and obj_norm and obj_norm != obj:
            logger.debug(
                f"Нормализованный поиск: '{subj}' + '{obj_norm}'"
            )
            rows = self._search_exact_object(subj, obj_norm)
            add(rows, "neo4j_exact_norm", 1.0)
            logger.debug(
                f"Нормализованный поиск: {len(rows)} фактов"
            )

        # 3. Поиск по субъекту — контекст об объекте
        if subj:
            logger.debug(f"Ищем субъект: '{subj}'")
            rows = self._search_by_name(subj, limit=5)
            add(rows, "neo4j", 0.9)
            logger.debug(f"По субъекту: {len(rows)} фактов")

        # 4. Полнотекстовый поиск — только если ничего не нашли
        if not evidence and claim.text:
            logger.debug("Полнотекстовый поиск...")
            words = [w for w in claim.text.split() if len(w) > 4]
            for word in words[:2]:
                rows = self._fulltext_search(word)
                add(rows, "neo4j_fulltext", 0.6)
                if evidence:
                    break

        # Сортируем по confidence — лучшие доказательства первыми
        evidence.sort(key=lambda e: e.confidence, reverse=True)
        evidence = evidence[:7]

        logger.debug(
            f"Итого: {len(evidence)} доказательств "
            f"для '{claim.text[:50]}'"
        )
        return evidence