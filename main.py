import urllib.request
import json

BASE_URL = "http://localhost:8000/api/v1"

def verify(text: str) -> dict:
    data = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/verify",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))

def print_result(title: str, result: dict):
    print("=" * 60)
    print(f"ТЕСТ: {title}")
    print(f"Текст:       {result['original_text']}")
    print(f"Вердикт:     {result['overall_verdict']}")
    print(f"Уверенность: {result['overall_confidence']}")
    print(f"Итог:        {result['summary']}")
    print()
    for i, claim in enumerate(result['claims']):
        print(f"  Утверждение {i+1}: {claim['claim']['text']}")
        print(f"  Субъект:  {claim['claim']['subject']}")
        print(f"  Объект:   {claim['claim']['object']}")
        print(f"  Вердикт:  {claim['verdict']}")
        print(f"  Объясн.:  {claim['explanation'][:100]}")
        print(f"  Доказательств: {len(claim['evidence'])}")
        print()

# Тест 1 — ПРАВДА
print("Отправляем тест 1...")
result = verify("Эйфелева башня построена в 1889 году в Париже")
print_result("ПРАВДА (ожидаем SUPPORTED)", result)

# Тест 2 — ЛОЖЬ
print("Отправляем тест 2...")
result = verify("Эйфелева башня построена в 1950 году в Лондоне")
print_result("ЛОЖЬ (ожидаем REFUTED)", result)

# Тест 3 — НЕТ ДАННЫХ
print("Отправляем тест 3...")
result = verify("Марс населён синими существами")
print_result("НЕТ ДАННЫХ (ожидаем NOT_ENOUGH_INFO)", result)

# Тест 4 — ПРАВДА
print("Отправляем тест 4...")
result = verify("Альберт Эйнштейн родился в 1879 году в Германии")
print_result("ПРАВДА (ожидаем SUPPORTED)", result)

# Тест 5 — НОВЫЙ: данные из Wikidata
print("Отправляем тест 5...")
result = verify("Москва является столицей России")
print_result("WIKIDATA (ожидаем SUPPORTED)", result)

print("Все тесты завершены!")