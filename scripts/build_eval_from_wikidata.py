"""
Формирует тестовую выборку из дампа Wikidata (latest-all.json.gz).
Читает файл потоково — не загружает 133 ГБ в память.

Запуск:
    python scripts/build_eval_from_wikidata.py
    python scripts/build_eval_from_wikidata.py --limit 500 --lang ru
"""
import gzip
import json
import random
import argparse
from pathlib import Path
from loguru import logger

# ──────────────────────────────────────────────────────────────
# Настройки
# ──────────────────────────────────────────────────────────────
DUMP_PATH   = Path("data/latest-all.json.gz")
OUTPUT_PATH = Path("tests/test_dataset_wikidata.py")
JSON_OUTPUT = Path("data/wikidata_eval.json")

# Свойства Wikidata нужно извлечь
# PID - (русский предикат, тип значения)
PROPERTIES = {
    "P569":  ("дата рождения",      "date"),    # date of birth
    "P570":  ("дата смерти",        "date"),    # date of death
    "P19":   ("место рождения",     "entity"),  # place of birth
    "P20":   ("место смерти",       "entity"),  # place of death
    "P27":   ("гражданство",        "entity"),  # country of citizenship
    "P36":   ("столица",            "entity"),  # capital
    "P31":   ("является",           "entity"),  # instance of
    "P17":   ("страна",             "entity"),  # country
    "P131":  ("расположен в",       "entity"),  # located in
    "P625":  ("координаты",         "coord"),   # coordinates (пропускаем)
    "P18":   ("изображение",        "skip"),    # image (пропускаем)
    "P856":  ("сайт",               "skip"),    # website (пропускаем)
}

# Только эти типы значений используем
SUPPORTED_TYPES = {"date", "entity"}


def get_label(entity: dict, lang: str = "ru") -> str | None:
    """Получить название сущности на нужном языке."""
    labels = entity.get("labels", {})
    if lang in labels:
        return labels[lang]["value"]
    # Fallback на английский
    if "en" in labels:
        return labels["en"]["value"]
    return None


def get_value(snak: dict, lang: str = "ru") -> str | None:
    """Извлечь значение из snak Wikidata."""
    try:
        snak_type = snak.get("snaktype")
        if snak_type != "value":
            return None

        datavalue = snak.get("datavalue", {})
        dtype = datavalue.get("type")
        value = datavalue.get("value")

        if dtype == "wikibase-entityid":
            # Ссылка на другую сущность — нужно резолвить
            # Возвращаем QID, позже заменим на label
            return f"Q:{value.get('id', '')}"

        elif dtype == "time":
            # Дата: "+1879-03-14T00:00:00Z" → "1879"
            time_str = value.get("time", "")
            if time_str.startswith("+") or time_str.startswith("-"):
                year = time_str[1:5]
                return year if year.isdigit() else None

        elif dtype == "string":
            return str(value)

        elif dtype == "monolingualtext":
            return value.get("text")

    except Exception:
        return None

    return None


def extract_facts(entity: dict, lang: str = "ru") -> list[dict]:
    """Извлечь факты из сущности Wikidata."""
    facts = []
    label = get_label(entity, lang)
    if not label:
        return []

    claims = entity.get("claims", {})

    for pid, (pred_name, value_type) in PROPERTIES.items():
        if value_type not in SUPPORTED_TYPES:
            continue
        if pid not in claims:
            continue

        for claim in claims[pid][:1]:  # берём только первое значение
            mainsnak = claim.get("mainsnak", {})
            raw_value = get_value(mainsnak, lang)

            if not raw_value:
                continue

            # Если значение — QID, пропускаем (нет label без второго прохода)
            if isinstance(raw_value, str) and raw_value.startswith("Q:"):
                # Пробуем взять label из sitelinks или пропустим
                continue

            facts.append({
                "subject":    label,
                "predicate":  pred_name,
                "object":     raw_value,
                "pid":        pid,
                "qid":        entity.get("id"),
            })

    return facts


def fact_to_text(fact: dict) -> str:
    """Преобразуем факт в естественный текст."""
    templates = {
        "дата рождения":  "{subject} родился в {object} году",
        "дата смерти":    "{subject} умер в {object} году",
        "место рождения": "{subject} родился в {object}",
        "место смерти":   "{subject} умер в {object}",
        "гражданство":    "{subject} имеет гражданство {object}",
        "столица":        "Столица {subject} — {object}",
        "является":       "{subject} является {object}",
        "страна":         "{subject} находится в стране {object}",
        "расположен в":   "{subject} расположен в {object}",
    }
    template = templates.get(
        fact["predicate"],
        "{subject} — {predicate} — {object}"
    )
    return template.format(**fact)


def make_refuted(fact: dict, all_values: dict) -> dict | None:
    """
    Создаём ложное утверждение — подменяем объект на неправильный.
    all_values: {predicate: [список реальных значений из выборки]}
    """
    pred = fact["predicate"]
    correct_obj = fact["object"]

    candidates = [
        v for v in all_values.get(pred, [])
        if v != correct_obj
    ]
    if not candidates:
        return None

    wrong_obj = random.choice(candidates)
    refuted_fact = {**fact, "object": wrong_obj}
    return refuted_fact


def stream_wikidata(path: Path, limit: int, lang: str):
    """
    Потоковое чтение дампа Wikidata.
    Возвращает факты по мере чтения.
    """
    facts = []
    items_read = 0
    items_with_facts = 0

    logger.info(f"Читаем дамп: {path}")
    logger.info(f"Лимит сущностей: {limit:,}")

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")

            # Первая и последняя строки — скобки массива
            if line in ("[", "]", ""):
                continue

            try:
                entity = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Только items (не property)
            if entity.get("type") != "item":
                continue

            items_read += 1
            extracted = extract_facts(entity, lang)

            if extracted:
                facts.extend(extracted)
                items_with_facts += 1

            # Прогресс каждые 100k
            if items_read % 100_000 == 0:
                logger.info(
                    f"  Прочитано: {items_read:,} сущностей | "
                    f"Фактов: {len(facts):,}"
                )

            if items_read >= limit:
                logger.info(f"Достигнут лимит {limit:,}")
                break

    logger.info(
        f"Итого прочитано: {items_read:,} сущностей, "
        f"из них с фактами: {items_with_facts:,}, "
        f"фактов: {len(facts):,}"
    )
    return facts


def build_dataset(
    facts: list[dict],
    n_supported: int = 100,
    n_refuted: int = 100,
    n_nei: int = 50,
) -> list[dict]:
    """
    Строим сбалансированный датасет:
    - SUPPORTED: реальные факты из Wikidata
    - REFUTED: реальные факты с подменённым объектом
    - NOT_ENOUGH_INFO: факты о сущностях которых нет в нашем графе
    """
    random.shuffle(facts)

    # Собираем все значения по предикату для генерации REFUTED
    all_values: dict[str, list] = {}
    for fact in facts:
        pred = fact["predicate"]
        obj = fact["object"]
        if pred not in all_values:
            all_values[pred] = []
        if obj not in all_values[pred]:
            all_values[pred].append(obj)

    dataset = []
    case_id = 1

    # SUPPORTED
    used_subjects = set()
    supported_count = 0
    for fact in facts:
        if supported_count >= n_supported:
            break
        # Берём не более 2 фактов на одну сущность
        key = (fact["subject"], fact["predicate"])
        if key in used_subjects:
            continue
        used_subjects.add(key)

        text = fact_to_text(fact)
        dataset.append({
            "id":           case_id,
            "text":         text,
            "ground_truth": "SUPPORTED",
            "category":     fact["predicate"],
            "source":       "wikidata",
            "qid":          fact.get("qid"),
            "fact":         fact,
        })
        case_id += 1
        supported_count += 1

    logger.info(f"SUPPORTED: {supported_count}")

    # REFUTED — берём SUPPORTED факты и подменяем объект
    refuted_count = 0
    for item in dataset[:]:
        if refuted_count >= n_refuted:
            break
        if item["ground_truth"] != "SUPPORTED":
            continue

        wrong_fact = make_refuted(item["fact"], all_values)
        if not wrong_fact:
            continue

        text = fact_to_text(wrong_fact)
        dataset.append({
            "id":           case_id,
            "text":         text,
            "ground_truth": "REFUTED",
            "category":     item["category"],
            "source":       "wikidata_refuted",
            "qid":          item["qid"],
            "fact":         wrong_fact,
        })
        case_id += 1
        refuted_count += 1

    logger.info(f"REFUTED: {refuted_count}")

    # NOT_ENOUGH_INFO — факты о редких сущностях
    # (берём последние факты — они реже встречаются в нашем графе)
    nei_candidates = facts[n_supported * 3:]
    random.shuffle(nei_candidates)
    nei_count = 0
    nei_subjects = set()

    for fact in nei_candidates:
        if nei_count >= n_nei:
            break
        if fact["subject"] in nei_subjects:
            continue
        nei_subjects.add(fact["subject"])

        text = fact_to_text(fact)
        dataset.append({
            "id":           case_id,
            "text":         text,
            "ground_truth": "NOT_ENOUGH_INFO",
            "category":     fact["predicate"],
            "source":       "wikidata_nei",
            "qid":          fact.get("qid"),
            "fact":         fact,
        })
        case_id += 1
        nei_count += 1

    logger.info(f"NOT_ENOUGH_INFO: {nei_count}")

    random.shuffle(dataset)
    return dataset


def save_as_python(dataset: list[dict], path: Path):
    """Сохраняем как Python файл совместимый с evaluate.py."""
    lines = [
        '"""',
        'Тестовая выборка сформированная из дампа Wikidata.',
        f'Всего случаев: {len(dataset)}',
        '"""',
        "",
        "TEST_CASES = [",
    ]
    for item in dataset:
        lines.append("    {")
        lines.append(f'        "id":           {item["id"]},')
        lines.append(f'        "text":         {json.dumps(item["text"], ensure_ascii=False)},')
        lines.append(f'        "ground_truth": "{item["ground_truth"]}",')
        lines.append(f'        "category":     {json.dumps(item["category"], ensure_ascii=False)},')
        lines.append(f'        "source":       "{item["source"]}",')
        lines.append(f'        "qid":          {json.dumps(item.get("qid"), ensure_ascii=False)},')
        lines.append("    },")
    lines.append("]")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.success(f"Python датасет: {path} ({len(dataset)} случаев)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump",    default=str(DUMP_PATH),
        help="Путь к latest-all.json.gz"
    )
    parser.add_argument(
        "--limit",   type=int, default=500_000,
        help="Сколько сущностей прочитать из дампа"
    )
    parser.add_argument(
        "--lang",    default="ru",
        help="Язык меток (ru/en)"
    )
    parser.add_argument(
        "--supported", type=int, default=100,
        help="Количество SUPPORTED примеров"
    )
    parser.add_argument(
        "--refuted",   type=int, default=100,
        help="Количество REFUTED примеров"
    )
    parser.add_argument(
        "--nei",       type=int, default=50,
        help="Количество NOT_ENOUGH_INFO примеров"
    )
    args = parser.parse_args()

    # Читаем дамп
    facts = stream_wikidata(
        Path(args.dump),
        limit=args.limit,
        lang=args.lang,
    )

    if not facts:
        logger.error("Факты не извлечены — проверьте путь к дампу")
        return

    # Строим датасет
    dataset = build_dataset(
        facts,
        n_supported=args.supported,
        n_refuted=args.refuted,
        n_nei=args.nei,
    )

    logger.info(f"Итого в датасете: {len(dataset)} случаев")

    # Сохраняем
    JSON_OUTPUT.parent.mkdir(exist_ok=True)
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logger.success(f"JSON: {JSON_OUTPUT}")

    save_as_python(dataset, OUTPUT_PATH)


if __name__ == "__main__":
    main()