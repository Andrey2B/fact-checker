import sys
import json
import time
import urllib.request
from collections import defaultdict

sys.path.append(".")
from loguru import logger
from tests.test_dataset_wikidata import TEST_CASES

BASE_URL = "http://localhost:8000/api/v1"
LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


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


def run_evaluation():
    logger.info(f"Запуск оценки: {len(TEST_CASES)} тестов")
    logger.info("=" * 60)

    results = []
    errors = []

    for i, case in enumerate(TEST_CASES, 1):
        logger.info(
            f"[{i:02d}/{len(TEST_CASES)}] "
            f"id={case['id']} | {case['text'][:50]}..."
        )
        try:
            response = verify(case["text"])
            predicted = response["overall_verdict"]
            confidence = response["overall_confidence"]
            correct = predicted == case["ground_truth"]

            results.append({
                "id":           case["id"],
                "text":         case["text"],
                "category":     case["category"],
                "ground_truth": case["ground_truth"],
                "predicted":    predicted,
                "confidence":   confidence,
                "correct":      correct,
            })

            status = "✅" if correct else "❌"
            logger.info(
                f"  {status} GT={case['ground_truth']:<20} "
                f"PRED={predicted:<20} conf={confidence:.2f}"
            )

        except Exception as e:
            logger.error(f"  Ошибка: {e}")
            errors.append({"id": case["id"], "error": str(e)})

        # Пауза между запросами чтобы не перегружать LLM
        time.sleep(0.5)

    # ──────────────────────────────────────────
    # Подсчёт метрик
    # ──────────────────────────────────────────
    print_metrics(results, errors)
    save_results(results, errors)


def print_metrics(results: list, errors: list):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 60)
    print(f"Всего тестов:    {len(TEST_CASES)}")
    print(f"Выполнено:       {total}")
    print(f"Ошибок API:      {len(errors)}")
    print(f"Правильных:      {correct}")
    print(f"Неправильных:    {total - correct}")
    print(f"\nОБЩАЯ ТОЧНОСТЬ: {accuracy:.1%}  ({correct}/{total})")

    # ── Precision / Recall / F1 по каждому классу ──
    print("\n" + "-" * 60)
    print(f"{'Класс':<22} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'Support':>10}")
    print("-" * 60)

    f1_scores = []
    for label in LABELS:
        tp = sum(
            1 for r in results
            if r["predicted"] == label and r["ground_truth"] == label
        )
        fp = sum(
            1 for r in results
            if r["predicted"] == label and r["ground_truth"] != label
        )
        fn = sum(
            1 for r in results
            if r["predicted"] != label and r["ground_truth"] == label
        )
        support = sum(
            1 for r in results if r["ground_truth"] == label
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        f1_scores.append(f1)

        print(
            f"{label:<22} {precision:>10.1%} {recall:>10.1%} "
            f"{f1:>10.1%} {support:>10}"
        )

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print("-" * 60)
    print(f"{'Macro F1':<22} {'':>10} {'':>10} {macro_f1:>10.1%}")

    # ── Матрица ошибок ──
    print("\n" + "-" * 60)
    print("МАТРИЦА ОШИБОК (строки=GT, столбцы=Predicted)")
    print("-" * 60)
    short = {"SUPPORTED": "SUP", "REFUTED": "REF", "NOT_ENOUGH_INFO": "NEI"}
    header = f"{'GT \\ Pred':<22}" + "".join(f"{short[l]:>8}" for l in LABELS)
    print(header)
    for gt in LABELS:
        row = f"{gt:<22}"
        for pred in LABELS:
            count = sum(
                1 for r in results
                if r["ground_truth"] == gt and r["predicted"] == pred
            )
            row += f"{count:>8}"
        print(row)

    # ── Точность по категориям ──
    print("\n" + "-" * 60)
    print("ТОЧНОСТЬ ПО КАТЕГОРИЯМ")
    print("-" * 60)
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r["correct"])

    for cat, correctness in sorted(by_category.items()):
        cat_acc = sum(correctness) / len(correctness)
        bar = "█" * int(cat_acc * 20) + "░" * (20 - int(cat_acc * 20))
        print(f"  {cat:<12} {bar} {cat_acc:.1%}  ({sum(correctness)}/{len(correctness)})")

    # ── Ошибочные предсказания ──
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n" + "-" * 60)
        print(f"НЕПРАВИЛЬНЫЕ ПРЕДСКАЗАНИЯ ({len(wrong)} шт.)")
        print("-" * 60)
        for r in wrong:
            print(
                f"  id={r['id']:02d} | GT={r['ground_truth']:<20} "
                f"PRED={r['predicted']:<20} conf={r['confidence']:.2f}"
            )
            print(f"         {r['text'][:70]}")

    # ── Средняя уверенность ──
    print("\n" + "-" * 60)
    print("СРЕДНЯЯ УВЕРЕННОСТЬ")
    print("-" * 60)
    correct_conf = [r["confidence"] for r in results if r["correct"]]
    wrong_conf   = [r["confidence"] for r in results if not r["correct"]]
    if correct_conf:
        print(f"  Верные предсказания:   {sum(correct_conf)/len(correct_conf):.3f}")
    if wrong_conf:
        print(f"  Неверные предсказания: {sum(wrong_conf)/len(wrong_conf):.3f}")

    print("=" * 60)


def save_results(results: list, errors: list):
    output = {
        "total":   len(TEST_CASES),
        "done":    len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "accuracy": (
            sum(1 for r in results if r["correct"]) / len(results)
            if results else 0
        ),
        "results": results,
        "errors":  errors,
    }
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.success("Результаты сохранены в evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()