import urllib.request
import urllib.parse
import json
import time
import os

OUTPUT_FILE = "data/wikidata_cache.json"

HEADERS = {
    "User-Agent": "KGFactChecker/1.0",
    "Accept": "application/json"
}

MISSING_ENTITIES = {
    "Q406":    "Стамбул",
    "Q991":    "Фёдор Достоевский",
    "Q9036":   "Никола Тесла",
    "Q3884":   "Amazon",
    "Q41225":  "Биг-Бен",
    "Q37200":  "Пирамида Хеопса",
    "Q9202":   "Статуя Свободы",
    "Q19837":  "Стив Джобс",
    "Q317521": "Илон Маск",
    "Q8409":   "Александр Македонский",
    "Q380":    "Meta",
    "Q319":    "Юпитер",
    "Q193":    "Нептун",
    "Q48":     "Сатурн",
    "Q405":    "Луна",
    "Q525":    "Солнце",
    "Q111":    "Марс",
    "Q308":    "Меркурий",
    "Q313":    "Венера",
    "Q11660":  "искусственный интеллект",
    "Q28865":  "Python",
    "Q7583":   "Bitcoin",
    "Q362":    "Вторая мировая война",
}

PROPERTIES = {
    "P17":   "страна",
    "P19":   "место рождения",
    "P20":   "место смерти",
    "P27":   "гражданство",
    "P31":   "является экземпляром",
    "P36":   "столица",
    "P37":   "официальный язык",
    "P50":   "автор",
    "P84":   "архитектор",
    "P131":  "расположен в",
    "P170":  "создатель",
    "P178":  "разработчик",
    "P276":  "местонахождение",
    "P571":  "дата основания",
    "P569":  "дата рождения",
    "P570":  "дата смерти",
    "P1082": "население",
    "P2048": "высота",
    "P6":    "глава правительства",
    "P35":   "глава государства",
    "P112":  "основатель",
    "P108":  "работодатель",
    "P69":   "место учёбы",
    "P166":  "награда",
    "P101":  "область деятельности",
    "P106":  "род занятий",
    "P159":  "штаб-квартира",
    "P452":  "отрасль",
    "P856":  "сайт",
    "P749":  "материнская компания",
}

# Кэш меток QID
label_cache: dict[str, str] = {}


def get_label_by_qid(qid: str) -> str:
    if qid in label_cache:
        return label_cache[qid]
    url = (
        f"https://www.wikidata.org/w/api.php"
        f"?action=wbgetentities&ids={qid}"
        f"&format=json&props=labels&languages=ru|en"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            entity = data.get("entities", {}).get(qid, {})
            labels = entity.get("labels", {})
            if "ru" in labels:
                label = labels["ru"]["value"]
            elif "en" in labels:
                label = labels["en"]["value"]
            else:
                label = qid
            label_cache[qid] = label
            time.sleep(0.5)
            return label
    except Exception:
        return qid


def get_entity(qid: str) -> dict:
    url = (
        f"https://www.wikidata.org/w/api.php"
        f"?action=wbgetentities&ids={qid}"
        f"&format=json&languages=ru|en"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode("utf-8"))
                return data.get("entities", {}).get(qid, {})
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"  Rate limit! Ждём 60 сек...")
                time.sleep(3)
            else:
                return {}
        except Exception:
            time.sleep(3)
    return {}


def extract_facts(entity: dict, name: str) -> list[dict]:
    facts = []
    seen = set()
    claims = entity.get("claims", {})

    for prop_id, statements in claims.items():
        prop_label = PROPERTIES.get(prop_id)
        if not prop_label:
            continue

        for statement in statements:
            try:
                mainsnak = statement.get("mainsnak", {})
                if mainsnak.get("snaktype") != "value":
                    continue

                datavalue = mainsnak.get("datavalue", {})
                dtype = datavalue.get("type", "")
                value = datavalue.get("value", "")

                obj = None

                if dtype == "wikibase-entityid":
                    target_qid = value.get("id", "")
                    obj = get_label_by_qid(target_qid)

                elif dtype == "string":
                    if not str(value).startswith("http"):
                        obj = str(value)

                elif dtype == "monolingualtext":
                    obj = value.get("text", "")

                elif dtype == "quantity":
                    amount = value.get("amount", "0")
                    try:
                        num = float(amount)
                        obj = str(int(num)) if num == int(num) else f"{num:.2f}"
                    except Exception:
                        obj = amount

                elif dtype == "time":
                    time_str = value.get("time", "")
                    try:
                        obj = time_str.split("-")[0].lstrip("+")
                    except Exception:
                        obj = time_str

                if (obj
                        and len(obj) > 0
                        and not obj.startswith("http")
                        and not (obj.startswith("Q") and obj[1:].isdigit())):
                    key = f"{prop_label}:{obj}"
                    if key not in seen:
                        seen.add(key)
                        facts.append({
                            "subject": name,
                            "predicate": prop_label,
                            "object": obj,
                        })

            except Exception:
                continue

    return facts


def main():
    print("=" * 60)
    print("СКАЧИВАНИЕ НЕДОСТАЮЩИХ СУЩНОСТЕЙ ЧЕРЕЗ API")
    print(f"Сущностей: {len(MISSING_ENTITIES)}")
    print("=" * 60)

    # Загружаем кэш
    cache = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Загружен кэш: {len(cache)} сущностей")

    total = len(MISSING_ENTITIES)
    done = 0

    for qid, name in MISSING_ENTITIES.items():
        done += 1
        print(f"\n[{done}/{total}] {name} ({qid})...")

        try:
            entity = get_entity(qid)
            if not entity:
                print(f"  ⚠️ Не получено")
                continue

            facts = extract_facts(entity, name)
            old_count = len(cache.get(name, []))
            cache[name] = facts

            print(f"  ✅ {len(facts)} фактов (было: {old_count})")

            # Сохраняем после каждой
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

            time.sleep(2)

        except KeyboardInterrupt:
            print("\nПрервано! Прогресс сохранён.")
            break

    total_facts = sum(len(v) for v in cache.values())
    print("\n" + "=" * 60)
    print(f" ГОТОВО!")
    print(f"Сущностей: {len(cache)}")
    print(f"Фактов:    {total_facts}")


if __name__ == "__main__":
    main()