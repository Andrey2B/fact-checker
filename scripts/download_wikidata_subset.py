"""
Скачивает данные о конкретных сущностях через REST API Wikidata.
БЕЗ rate limit, БЕЗ 133GB дампа!
Сохраняет в data/wikidata_cache.json

Запуск: python scripts/download_wikidata_subset.py
"""
import urllib.request
import urllib.error
import json
import time
import os

os.makedirs("data", exist_ok=True)
CACHE_FILE = "data/wikidata_cache.json"

HEADERS = {
    "User-Agent": "KGFactChecker/1.0 (educational project)",
    "Accept": "application/json"
}

# Нужные свойства
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
    "P495":  "страна происхождения",
    "P571":  "дата основания",
    "P569":  "дата рождения",
    "P570":  "дата смерти",
    "P577":  "дата публикации",
    "P1082": "население",
    "P2048": "высота",
    "P2067": "масса",
    "P6":    "глава правительства",
    "P35":   "глава государства",
    "P488":  "председатель",
    "P112":  "основатель",
    "P123":  "издатель",
    "P57":   "режиссёр",
    "P161":  "актёр",
    "P108":  "работодатель",
    "P69":   "место учёбы",
    "P166":  "награда",
    "P101":  "область деятельности",
    "P106":  "род занятий",
    "P127":  "владелец",
    "P159":  "штаб-квартира",
    "P749":  "материнская компания",
}

# QID - название
TARGET_ENTITIES = {
    # Архитектура
    "Q243":    "Эйфелева башня",
    "Q9202":   "Статуя Свободы",
    "Q10285":  "Колизей",
    "Q82425":  "Биг-Бен",
    "Q5776":   "Московский Кремль",
    "Q160236": "Тадж-Махал",
    "Q130003": "Пирамида Хеопса",

    # Учёные
    "Q937":    "Альберт Эйнштейн",
    "Q9695":   "Исаак Ньютон",
    "Q9312":   "Никола Тесла",
    "Q7186":   "Мария Кюри",
    "Q1035":   "Чарлз Дарвин",
    "Q9294":   "Стивен Хокинг",
    "Q5582":   "Галилео Галилей",
    "Q9391":   "Зигмунд Фрейд",

    # Изобретатели / предприниматели
    "Q9439":   "Томас Эдисон",
    "Q7604":   "Стив Джобс",
    "Q5284":   "Билл Гейтс",
    "Q20921":  "Илон Маск",
    "Q4970706":"Марк Цукерберг",

    # Города
    "Q90":     "Париж",
    "Q649":    "Москва",
    "Q84":     "Лондон",
    "Q60":     "Нью-Йорк",
    "Q1490":   "Токио",
    "Q1794":   "Санкт-Петербург",
    "Q270":    "Рим",
    "Q585":    "Берлин",
    "Q1524":   "Пекин",
    "Q1350":   "Стамбул",

    # Страны
    "Q159":    "Россия",
    "Q142":    "Франция",
    "Q183":    "Германия",
    "Q30":     "США",
    "Q148":    "Китай",
    "Q145":    "Великобритания",
    "Q38":     "Италия",
    "Q155":    "Бразилия",
    "Q17":     "Япония",
    "Q668":    "Индия",

    # Планеты и космос
    "Q2":      "Земля",
    "Q111":    "Марс",
    "Q319":    "Юпитер",
    "Q193":    "Нептун",
    "Q48":     "Сатурн",
    "Q405":    "Луна",
    "Q525":    "Солнце",
    "Q308":    "Меркурий",
    "Q313":    "Венера",

    # Технологии
    "Q28865":  "Python",
    "Q7583":   "Bitcoin",
    "Q75":     "Интернет",
    "Q9141":   "Java",
    "Q15777":  "C++",
    "Q251":    "JavaScript",
    "Q11660":  "искусственный интеллект",

    # Компании
    "Q95":     "Google",
    "Q380":    "Facebook",
    "Q312":    "Apple",
    "Q37156":  "Microsoft",
    "Q11036":  "Amazon",
    "Q7749":   "Tesla",

    # История
    "Q362":    "Вторая мировая война",
    "Q361":    "Первая мировая война",
    "Q517":    "Наполеон Бонапарт",
    "Q1048":   "Юлий Цезарь",
    "Q32522":  "Александр Македонский",
    "Q9439":   "Виктория Английская",

    # Деятели культуры
    "Q762":    "Леонардо да Винчи",
    "Q5679":   "Уильям Шекспир",
    "Q7242":   "Вольфганг Амадей Моцарт",
    "Q1339":   "Иоганн Себастьян Бах",
    "Q9685":   "Лев Толстой",
    "Q5712":   "Фёдор Достоевский",
    "Q7317":   "Александр Пушкин",

    # Фильмы и книги
    "Q47703":  "Властелин колец",
    "Q208460": "Гарри Поттер",
    "Q24871":  "Звёздные войны",
}


def get_entity(qid: str) -> dict:
    """
    Получает данные об одной сущности через REST API.
    https://www.wikidata.org/wiki/Wikidata:REST_API
    """
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={qid}&format=json&languages=ru|en"
    req = urllib.request.Request(url, headers=HEADERS)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
                entities = data.get("entities", {})
                return entities.get(qid, {})

        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"  Rate limit! Ждём 60 сек...")
                time.sleep(3)
            else:
                print(f"  HTTP {e.code}")
                return {}

        except Exception as e:
            print(f"  Ошибка: {e}")
            time.sleep(5)

    return {}


def get_label(entity: dict) -> str:
    """Русское или английское название"""
    labels = entity.get("labels", {})
    if "ru" in labels:
        return labels["ru"]["value"]
    if "en" in labels:
        return labels["en"]["value"]
    return ""


def get_entity_label_by_qid(qid: str, label_cache: dict) -> str:
    """Получает название сущности по QID (с кэшированием)"""
    if qid in label_cache:
        return label_cache[qid]

    # Запрашиваем только метки (быстро!)
    url = (
        f"https://www.wikidata.org/w/api.php"
        f"?action=wbgetentities&ids={qid}"
        f"&format=json&props=labels&languages=ru|en"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            entity = data.get("entities", {}).get(qid, {})
            label = get_label(entity)
            label_cache[qid] = label
            time.sleep(0.2)
            return label
    except Exception:
        return qid


def extract_facts(entity: dict, entity_name: str, label_cache: dict) -> list[dict]:
    """Извлекает нужные факты из сущности"""
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
                    obj = get_entity_label_by_qid(target_qid, label_cache)

                elif dtype == "string":
                    obj = str(value)
                    if obj.startswith("http"):
                        continue

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

                if obj and len(obj) > 0 and not obj.startswith("Q"):
                    key = f"{prop_label}:{obj}"
                    if key not in seen:
                        seen.add(key)
                        facts.append({
                            "subject": entity_name,
                            "predicate": prop_label,
                            "object": obj,
                        })

            except Exception:
                continue

    return facts


def main():
    print("=" * 60)
    print("СКАЧИВАНИЕ ДАННЫХ ИЗ WIKIDATA REST API")
    print(f"Сущностей для скачивания: {len(TARGET_ENTITIES)}")
    print("=" * 60)

    # Загружаем кэш
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Загружен кэш: {len(cache)} сущностей")

    # Кэш меток QID чтобы не делать лишние запросы
    label_cache = {}
    # Заполняем из TARGET_ENTITIES
    for qid, name in TARGET_ENTITIES.items():
        label_cache[qid] = name

    total = len(TARGET_ENTITIES)
    done = 0

    for qid, name in TARGET_ENTITIES.items():
        done += 1

        if name in cache:
            print(f"[{done}/{total}] '{name}' — уже в кэше ✅")
            continue

        print(f"[{done}/{total}] Скачиваю: '{name}' ({qid})...")

        try:
            entity = get_entity(qid)
            if not entity:
                print(f"  ⚠️ Не найдено")
                cache[name] = []
            else:
                facts = extract_facts(entity, name, label_cache)
                cache[name] = facts
                print(f"  ✅ {len(facts)} фактов")

            # Сохраняем после каждой сущности
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

            time.sleep(1)

        except KeyboardInterrupt:
            print("\nПрервано! Прогресс сохранён.")
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            break

    # Итог
    total_facts = sum(len(v) for v in cache.values())
    print("=" * 60)
    print(f"✅ ГОТОВО!")
    print(f"Сущностей: {len(cache)}")
    print(f"Фактов:    {total_facts}")
    print(f"Файл:      {CACHE_FILE} (~{total_facts * 50 // 1024}KB)")


if __name__ == "__main__":
    main()