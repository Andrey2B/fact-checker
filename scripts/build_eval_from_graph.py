import sys
import json
import random
from pathlib import Path

sys.path.append(".")
from loguru import logger
from src.graph.neo4j_client import Neo4jClient

OUTPUT_PY   = Path("tests/test_dataset_wikidata.py")
OUTPUT_JSON = Path("data/graph_eval.json")

# ── Предикаты которые ПРОПУСКАЕМ ──────────────────────────────
# население: много значений разных лет → всегда CONFLICTING
SKIP_PREDICATES = {
    "население", "population",
    "координаты", "coordinates",
    "изображение", "image",
    "сайт", "website", "url",
}

# ── Шаблоны текста ────────────────────────────────────────────
TEMPLATES = {
    "родился_в":              "{subject} родился в {object} году",
    "место_рождения":         "{subject} родился в {object}",
    "дата рождения":          "{subject} родился в {object} году",
    "место рождения":         "{subject} родился в {object}",
    "гражданство":            "{subject} имеет гражданство {object}",
    "столица":                "{subject} является столицей {object}",
    "capital_of":             "{subject} является столицей {object}",
    "расположена_в":          "{subject} расположена в {object}",
    "расположен в":           "{subject} расположен в {object}",
    "построена_в":            "{subject} построена в {object} году",
    "высота":                 "Высота {subject} составляет {object} метров",
    "спроектирована":         "{subject} спроектирована {object}",
    "формула":                "Химическая формула {subject} — {object}",
    "вращается_вокруг":       "{subject} вращается вокруг {object}",
    "спутник":                "{subject} является спутником {object}",
    "создан":                 "{subject} создан {object}",
    "год_создания":           "{subject} создан в {object} году",
    "разработал":             "{subject} разработал {object}",
    "основана_в":             "{subject} основана в {object} году",
    "дата основания":         "{subject} основана в {object} году",
    "дата смерти":            "{subject} умер в {object} году",
    "страна":                 "{subject} находится в стране {object}",
    "страна происхождения":   "{subject} происходит из {object}",
    "архитектор":             "{subject} спроектирован архитектором {object}",
    "создатель":              "{subject} создан {object}",
    "род занятий":            "{subject} по роду занятий является {object}",
    "является экземпляром":   "{subject} является {object}",
    "дата публикации":        "{subject} опубликована в {object} году",
}

# ── NOT_ENOUGH_INFO пул ───────────────────────────────────────
# Только неочевидные факты — LLM не должна знать их без графа
NEI_ENTITIES = [
    ("Антарктида",        "является материком"),
    ("Марс",              "является четвёртой планетой от Солнца"),
    ("Квантовая механика","была разработана в 20 веке"),
    ("Нил",               "является самой длинной рекой в мире"),
    ("Монна Лиза",        "написана Леонардо да Винчи"),
    ("Амазонка",          "является самой полноводной рекой"),
    ("Эверест",           "является высочайшей вершиной мира"),
    ("Шекспир",           "написал Гамлета"),
    ("Великая китайская стена", "построена в Китае"),
    ("Карл Линней",       "разработал систему классификации"),
    ("Мария Кюри",        "открыла полоний"),
    ("Архимед",           "открыл закон вытеснения"),
    ("Галилей",           "изобрёл телескоп"),
    ("Коперник",          "разработал гелиоцентрическую модель"),
    ("Моцарт",            "написал Реквием"),
    ("Микеланджело",      "создал скульптуру Давид"),
    ("Гомер",             "написал Илиаду"),
    ("Пифагор",           "доказал теорему о прямоугольном треугольнике"),
    ("Ньютон",            "открыл закон всемирного тяготения"),
    ("Планк",             "ввёл понятие кванта энергии"),
]

# ── Пары которые случайно оказываются правдой ─────────────────
# Исключаем их из REFUTED
KNOWN_TRUE_PAIRS = {
    ("Статуя Свободы",    "создатель",    "Фредерик Огюст Бартольди"),
    ("Стамбул",           "страна",       "Османская империя"),
    ("Биг-Бен",           "страна",       "Великобритания"),
    ("Великобритания",    "страна",       "Великобритания"),
    ("Германия",          "страна",       "Германия"),
    ("Франция",           "страна",       "Франция"),
    ("Китай",             "страна",       "Китай"),
    ("Россия",            "страна",       "Россия"),
    ("Япония",            "страна",       "Япония"),
}


def get_all_facts(client) -> list[dict]:
    rows = client.run_query("""
        MATCH (s:Entity)-[r:RELATION]->(o:Entity)
        RETURN s.name AS subject, r.type AS predicate, o.name AS object
        ORDER BY s.name, r.type
    """)
    logger.info(f"Всего фактов в графе: {len(rows)}")
    return rows


def fact_to_text(subject: str, predicate: str, obj: str) -> str:
    key = predicate.lower()
    template = TEMPLATES.get(key)
    if template:
        return template.format(subject=subject, predicate=predicate, object=obj)
    # Читаемый дефолт вместо тире
    return f"{subject}: {predicate.lower()} — {obj}"


def is_good_fact(subject: str, predicate: str, obj: str) -> bool:
    if len(obj) < 1 or len(subject) < 2:
        return False
    if obj.startswith("http") or subject.startswith("http"):
        return False
    # Пропускаем проблемные предикаты
    if predicate.lower() in SKIP_PREDICATES:
        return False
    # Пропускаем самоссылки ("Германия находится в стране Германия")
    if subject.lower() == obj.lower():
        return False
    return True


def build_supported(facts: list[dict], n: int) -> list[dict]:
    good = [
        f for f in facts
        if is_good_fact(f["subject"], f["predicate"], f["object"])
    ]
    random.shuffle(good)

    cases = []
    seen = set()

    for fact in good:
        if len(cases) >= n:
            break
        key = (fact["subject"], fact["predicate"])
        if key in seen:
            continue
        seen.add(key)

        text = fact_to_text(fact["subject"], fact["predicate"], fact["object"])
        cases.append({
            "text":         text,
            "ground_truth": "SUPPORTED",
            "category":     fact["predicate"].lower(),
            "source":       "graph",
            "fact":         fact,
        })

    logger.info(f"SUPPORTED: {len(cases)}")
    return cases


def build_refuted(
    supported: list[dict],
    facts: list[dict],
    n: int,
) -> list[dict]:
    # Все значения по предикату для подмены
    values_by_pred: dict[str, list[str]] = {}
    for f in facts:
        pred = f["predicate"].lower()
        if pred not in values_by_pred:
            values_by_pred[pred] = []
        if f["object"] not in values_by_pred[pred]:
            values_by_pred[pred].append(f["object"])

    # Дополнительные ложные значения
    extra_values = {
        "дата рождения":  ["1800", "1750", "1950", "2000", "1600", "1400"],
        "родился_в":      ["1800", "1750", "1950", "2000", "1600"],
        "дата смерти":    ["1700", "1600", "2011", "1950", "2000"],
        "дата основания": ["1800", "1400", "1999", "1776", "1868"],
        "построена_в":    ["1800", "1750", "1950", "2000"],
        "год_создания":   ["1800", "1750", "1950", "2000"],
        "высота":         ["50", "100", "200", "500", "1000"],
        "столица":        ["Берлин", "Лондон", "Токио", "Пекин", "Рим",
                           "Каир", "Оттава", "Вена", "Кишинёв"],
        "гражданство":    ["Бразилия", "Китай", "Япония", "Франция"],
        "страна":         ["Индия", "Бразилия", "Турция", "Австралия"],
        "формула":        ["CO2", "NaCl", "CH4", "O2", "N2"],
        "создан":         ["Билл Гейтс", "Стив Джобс", "Марк Цукерберг"],
        "создатель":      ["Ричард Моррис Хант", "Карл Готтгард Лангганс"],
        "архитектор":     ["Генрих Штрак", "Карл Готтгард Лангганс"],
        "страна происхождения": ["США", "Франция", "Германия", "Китай"],
    }

    cases = []
    random.shuffle(supported)

    for item in supported:
        if len(cases) >= n:
            break

        fact = item["fact"]
        pred = fact["predicate"].lower()
        correct_obj = fact["object"]

        candidates = [
            v for v in values_by_pred.get(pred, [])
            if v != correct_obj
        ]
        candidates += [
            v for v in extra_values.get(pred, [])
            if v != correct_obj
        ]

        if not candidates:
            continue

        # Пробуем несколько кандидатов — исключаем случайно-верные
        random.shuffle(candidates)
        chosen = None
        for candidate in candidates:
            triple = (fact["subject"], fact["predicate"], candidate)
            if triple not in KNOWN_TRUE_PAIRS:
                chosen = candidate
                break

        if not chosen:
            continue

        text = fact_to_text(fact["subject"], fact["predicate"], chosen)
        cases.append({
            "text":         text,
            "ground_truth": "REFUTED",
            "category":     pred,
            "source":       "graph_refuted",
            "fact":         {**fact, "object": chosen},
        })

    logger.info(f"REFUTED: {len(cases)}")
    return cases


def build_nei(n: int) -> list[dict]:
    pool = NEI_ENTITIES.copy()
    while len(pool) < n:
        pool += NEI_ENTITIES
    random.shuffle(pool)

    cases = []
    for subject, predicate_text in pool[:n]:
        text = f"{subject} {predicate_text}"
        cases.append({
            "text":         text,
            "ground_truth": "NOT_ENOUGH_INFO",
            "category":     "not_in_graph",
            "source":       "manual_nei",
        })

    logger.info(f"NOT_ENOUGH_INFO: {len(cases)}")
    return cases


def print_stats(dataset: list[dict]):
    from collections import Counter
    verdicts = Counter(d["ground_truth"] for d in dataset)
    cats     = Counter(d["category"] for d in dataset)

    logger.info("=" * 50)
    logger.info("СТАТИСТИКА ДАТАСЕТА")
    logger.info("=" * 50)
    for verdict, count in sorted(verdicts.items()):
        logger.info(f"  {verdict:<25} {count}")
    logger.info("-" * 50)
    for cat, count in cats.most_common(10):
        logger.info(f"  {cat:<30} {count}")


def save_as_python(dataset: list[dict], path: Path):
    lines = [
        '"""',
        f"Тестовая выборка из графа знаний.",
        f"Всего: {len(dataset)} случаев",
        '"""',
        "",
        "TEST_CASES = [",
    ]
    for i, item in enumerate(dataset, 1):
        lines.append("    {")
        lines.append(f'        "id":           {i},')
        lines.append(f'        "text":         {json.dumps(item["text"], ensure_ascii=False)},')
        lines.append(f'        "ground_truth": "{item["ground_truth"]}",')
        lines.append(f'        "category":     {json.dumps(item["category"], ensure_ascii=False)},')
        lines.append(f'        "source":       "{item["source"]}",')
        lines.append("    },")
    lines.append("]")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.success(f"Python датасет: {path} ({len(dataset)} случаев)")


def main():
    client = Neo4jClient.get_instance()
    facts  = get_all_facts(client)

    if not facts:
        logger.error("Граф пустой!")
        return

    n_supported = 70
    n_refuted   = 70
    n_nei       = 20

    supported = build_supported(facts, n_supported)
    refuted   = build_refuted(supported, facts, n_refuted)
    nei       = build_nei(n_nei)

    dataset = supported + refuted + nei
    random.shuffle(dataset)

    print_stats(dataset)
    logger.info(f"Итого: {len(dataset)} случаев")

    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logger.success(f"JSON: {OUTPUT_JSON}")

    save_as_python(dataset, OUTPUT_PY)


if __name__ == "__main__":
    main()