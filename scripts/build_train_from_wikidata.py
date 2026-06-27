"""
Генерирует обучающие данные для MiniVerifier из дампа Wikidata.
Читает потоково — не загружает 133 ГБ в память.

Формат обучающего примера:
  claim:    "Альберт Эйнштейн дата рождения 1879"
  evidence: "Альберт Эйнштейн дата рождения 1879 | Эйнштейн гражданство США"
  label:    SUPPORTED / REFUTED / NOT_ENOUGH_INFO
"""
import gzip
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from loguru import logger

# ── Пути ──────────────────────────────────────────────────────
OUTPUT_PATH = Path("data/mini_lm_train.json")
DUMP_PATH   = Path("C:/Users/Andrey/Downloads/latest-all.json.gz")

# ── Свойства Wikidata которые берём ───────────────────────────
PROPERTIES = {
    "P569":  "дата рождения",
    "P570":  "дата смерти",
    "P19":   "место рождения",
    "P20":   "место смерти",
    "P27":   "гражданство",
    "P36":   "столица",
    "P17":   "страна",
    "P131":  "расположен в",
    "P571":  "дата основания",
    "P582":  "дата окончания",
    "P276":  "местонахождение",
    "P112":  "основатель",
    "P123":  "издатель",
    "P50":   "автор",
    "P57":   "режиссёр",
    "P86":   "композитор",
    "P31":   "является",
    "P106":  "род занятий",
    "P21":   "пол",
    "P495":  "страна производства",
}

# ── Языки (приоритет) ─────────────────────────────────────────
LANG_PRIORITY = ["ru", "en"]


def get_label(entity: dict) -> str | None:
    """Получить метку сущности на русском или английском."""
    labels = entity.get("labels", {})
    for lang in LANG_PRIORITY:
        if lang in labels:
            return labels[lang]["value"]
    return None


def get_snak_value(snak: dict) -> str | None:
    """Извлечь значение из snak."""
    try:
        if snak.get("snaktype") != "value":
            return None

        dv    = snak.get("datavalue", {})
        dtype = dv.get("type")
        val   = dv.get("value")

        if dtype == "wikibase-entityid":
            # QID — позже резолвим через labels
            return f"_QID_{val.get('id', '')}"

        elif dtype == "time":
            time_str = val.get("time", "")
            # "+1879-03-14T..." → "1879"
            sign = time_str[0] if time_str else "+"
            year = time_str[1:5] if len(time_str) >= 5 else ""
            if year.isdigit():
                return ("" if sign == "+" else "-") + year
            return None

        elif dtype == "string":
            return str(val)[:100]

        elif dtype == "monolingualtext":
            return val.get("text", "")[:100]

        elif dtype == "quantity":
            amount = val.get("amount", "0")
            # "+42" → "42"
            return amount.lstrip("+")[:20]

    except Exception:
        return None

    return None


def extract_entity_facts(
    entity: dict,
    qid_to_label: dict,
) -> list[dict]:
    """
    Извлечь факты из одной сущности Wikidata.
    Возвращает список {subject, predicate, object}.
    """
    label = get_label(entity)
    if not label or len(label) < 2:
        return []

    facts   = []
    claims  = entity.get("claims", {})

    for pid, pred_name in PROPERTIES.items():
        if pid not in claims:
            continue

        for claim in claims[pid][:2]:  # макс 2 значения
            mainsnak = claim.get("mainsnak", {})
            raw_val  = get_snak_value(mainsnak)

            if not raw_val:
                continue

            # Резолвим QID → label
            if raw_val.startswith("_QID_"):
                qid = raw_val[5:]
                obj = qid_to_label.get(qid)
                if not obj:
                    continue  # нет label — пропускаем
            else:
                obj = raw_val

            if len(obj) < 1 or len(obj) > 80:
                continue

            facts.append({
                "subject":   label,
                "predicate": pred_name,
                "object":    obj,
                "qid":       entity.get("id"),
            })

    return facts


def make_training_sample(
    fact:           dict,
    all_facts:      list[dict],
    subject_facts:  list[dict],
    values_by_pred: dict,
    mode:           str,  # "supported" | "refuted" | "nei"
) -> dict | None:
    """Создать один обучающий пример."""

    subj = fact["subject"]
    pred = fact["predicate"]
    obj  = fact["object"]

    # Доказательства — другие факты о том же субъекте
    evidence_facts = [
        f for f in subject_facts
        if not (f["predicate"] == pred and f["object"] == obj)
    ][:4]

    if not evidence_facts and mode != "nei":
        return None

    evidence_text = " | ".join(
        f"{f['subject']} {f['predicate']} {f['object']}"
        for f in evidence_facts
    ) if evidence_facts else "нет данных"

    if mode == "supported":
        claim = f"{subj} {pred} {obj}"
        label = "SUPPORTED"

        # КЛЮЧЕВОЕ: evidence ДОЛЖЕН содержать сам факт
        # + другие факты о субъекте для контекста
        self_fact = f"{subj} {pred} {obj}"
        other_facts = [
            f for f in subject_facts
            if not (f["predicate"] == pred and f["object"] == obj)
        ][:3]
        evidence_parts = [self_fact] + [
            f"{f['subject']} {f['predicate']} {f['object']}"
            for f in other_facts
        ]
        # Перемешиваем чтобы модель не запоминала позицию
        random.shuffle(evidence_parts)
        evidence_text = " | ".join(evidence_parts)


    elif mode == "refuted":
        candidates = [
            v for v in values_by_pred.get(pred, [])
            if v != obj and not v.startswith("_QID_")
        ]
        if not candidates:
            return None
        wrong_obj = random.choice(candidates)
        claim = f"{subj} {pred} {wrong_obj}"
        label = "REFUTED"
        # Evidence содержит ПРАВИЛЬНЫЙ факт — модель учится
        # что claim противоречит evidence
        correct_fact = f"{subj} {pred} {obj}"
        other_facts = [
            f for f in subject_facts
            if not (f["predicate"] == pred and f["object"] == obj)
        ][:3]
        evidence_parts = [correct_fact] + [
            f"{f['subject']} {f['predicate']} {f['object']}"
            for f in other_facts
        ]
        random.shuffle(evidence_parts)
        evidence_text = " | ".join(evidence_parts)

    elif mode == "nei":
        # Доказательства о ДРУГОЙ сущности
        other_facts = [
            f for f in all_facts
            if f["subject"] != subj
        ]
        if not other_facts:
            return None
        sample_size = min(3, len(other_facts))
        other = random.sample(other_facts, sample_size)
        evidence_text = " | ".join(
            f"{f['subject']} {f['predicate']} {f['object']}"
            for f in other
        )
        claim = f"{subj} {pred} {obj}"
        label = "NOT_ENOUGH_INFO"

    else:
        return None

    return {
        "claim":    claim,
        "evidence": evidence_text,
        "label":    label,
    }


def stream_and_build(
    dump_path:    Path,
    output_path:  Path,
    limit:        int   = 1_000_000,
    n_samples:    int   = 100_000,
    batch_size:   int   = 10_000,
    save_every:   int   = 50_000,
):
    """
    Потоково читаем дамп, извлекаем факты, генерируем обучающие примеры.
    Сохраняем инкрементально каждые save_every примеров.
    """
    logger.info(f"Читаем дамп: {dump_path}")
    logger.info(f"Лимит сущностей: {limit:,} | Целевых примеров: {n_samples:,}")

    # Первый проход — строим словарь QID → label
    # (берём первые 500k сущностей для словаря)
    logger.info("Проход 1/2: строим словарь QID → label...")
    qid_to_label: dict[str, str] = {}
    items_read = 0

    with gzip.open(dump_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if line in ("[", "]", ""):
                continue
            try:
                entity = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entity.get("type") != "item":
                continue

            qid   = entity.get("id", "")
            label = get_label(entity)
            if label:
                qid_to_label[qid] = label

            items_read += 1
            if items_read % 200_000 == 0:
                logger.info(
                    f"  QID словарь: {items_read:,} сущностей, "
                    f"{len(qid_to_label):,} меток"
                )
            if items_read >= 500_000:
                break

    logger.info(f"QID словарь готов: {len(qid_to_label):,} меток")

    # Второй проход — генерируем обучающие примеры
    logger.info("Проход 2/2: генерируем обучающие примеры...")

    all_samples:    list[dict] = []
    buffer_facts:   list[dict] = []   # накапливаем факты батчами
    values_by_pred: dict       = defaultdict(list)

    counts  = {"SUPPORTED": 0, "REFUTED": 0, "NOT_ENOUGH_INFO": 0}
    target  = n_samples // 3  # равномерный баланс
    items_read = 0

    # Загружаем уже сохранённые примеры если есть
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            all_samples = json.load(f)
        for s in all_samples:
            counts[s["label"]] = counts.get(s["label"], 0) + 1
        logger.info(f"Загружено {len(all_samples)} существующих примеров")

    with gzip.open(dump_path, "rt", encoding="utf-8") as f:
        for line in f:
            # Достигли нужного количества?
            if all(c >= target for c in counts.values()):
                logger.info("Целевое количество примеров достигнуто")
                break

            line = line.strip().rstrip(",")
            if line in ("[", "]", ""):
                continue
            try:
                entity = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entity.get("type") != "item":
                continue

            items_read += 1

            # Извлекаем факты сущности
            facts = extract_entity_facts(entity, qid_to_label)
            if not facts:
                continue

            # Накапливаем значения для генерации REFUTED
            for f in facts:
                pred = f["predicate"]
                obj  = f["object"]
                if (
                    obj not in values_by_pred[pred]
                    and len(values_by_pred[pred]) < 200
                ):
                    values_by_pred[pred].append(obj)

            buffer_facts.extend(facts)

            # Генерируем примеры из буфера
            if len(buffer_facts) >= batch_size:
                new_samples = process_batch(
                    buffer_facts, values_by_pred, counts, target
                )
                all_samples.extend(new_samples)
                for s in new_samples:
                    counts[s["label"]] += 1
                buffer_facts = []

                logger.info(
                    f"  Сущностей: {items_read:,} | "
                    f"Примеров: {len(all_samples):,} | "
                    f"SUP={counts['SUPPORTED']} "
                    f"REF={counts['REFUTED']} "
                    f"NEI={counts['NOT_ENOUGH_INFO']}"
                )

            # Периодически сохраняем
            if len(all_samples) % save_every < batch_size:
                _save(all_samples, output_path)

            if items_read >= limit:
                logger.info(f"Достигнут лимит {limit:,} сущностей")
                break

    # Обрабатываем остаток буфера
    if buffer_facts:
        new_samples = process_batch(
            buffer_facts, values_by_pred, counts, target
        )
        all_samples.extend(new_samples)

    # Финальное сохранение
    random.shuffle(all_samples)
    _save(all_samples, output_path)

    logger.success(
        f"Готово: {len(all_samples):,} примеров → {output_path}"
    )
    logger.info(f"  SUPPORTED:       {counts['SUPPORTED']:,}")
    logger.info(f"  REFUTED:         {counts['REFUTED']:,}")
    logger.info(f"  NOT_ENOUGH_INFO: {counts['NOT_ENOUGH_INFO']:,}")

    return all_samples


def process_batch(
    facts:         list[dict],
    values_by_pred: dict,
    counts:        dict,
    target:        int,
) -> list[dict]:
    """Генерируем примеры из батча фактов."""
    samples = []

    # Группируем факты по субъекту
    by_subject: dict[str, list] = defaultdict(list)
    for f in facts:
        by_subject[f["subject"]].append(f)

    for subj, subj_facts in by_subject.items():
        if len(subj_facts) < 1:
            continue

        for fact in subj_facts:
            # SUPPORTED
            if counts["SUPPORTED"] < target:
                s = make_training_sample(
                    fact, facts, subj_facts,
                    values_by_pred, "supported"
                )
                if s:
                    samples.append(s)
                    counts["SUPPORTED"] += 1

            # REFUTED
            if counts["REFUTED"] < target:
                s = make_training_sample(
                    fact, facts, subj_facts,
                    values_by_pred, "refuted"
                )
                if s:
                    samples.append(s)
                    counts["REFUTED"] += 1

            # NOT_ENOUGH_INFO
            if counts["NOT_ENOUGH_INFO"] < target:
                s = make_training_sample(
                    fact, facts, subj_facts,
                    values_by_pred, "nei"
                )
                if s:
                    samples.append(s)
                    counts["NOT_ENOUGH_INFO"] += 1

    return samples


def _save(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.debug(f"Сохранено {len(samples):,} примеров → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump",     default=str(DUMP_PATH),
        help="Путь к latest-all.json.gz"
    )
    parser.add_argument(
        "--output",   default=str(OUTPUT_PATH),
        help="Путь к выходному файлу"
    )
    parser.add_argument(
        "--limit",    type=int, default=2_000_000,
        help="Сколько сущностей прочитать"
    )
    parser.add_argument(
        "--samples",  type=int, default=150_000,
        help="Целевое количество обучающих примеров"
    )
    args = parser.parse_args()

    stream_and_build(
        dump_path   = Path(args.dump),
        output_path = Path(args.output),
        limit       = args.limit,
        n_samples   = args.samples,
    )


if __name__ == "__main__":
    main()