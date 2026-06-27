import sys
import json
import time
from collections import defaultdict

sys.path.append(".")
from loguru import logger
from src.verification.cascade_pipeline import CascadePipeline
from tests.test_dataset_wikidata import TEST_CASES

LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


def run_evaluation():
    logger.info("Загружаем CascadePipeline...")
    pipeline = CascadePipeline()

    logger.info(f"Запуск оценки: {len(TEST_CASES)} тестов")
    logger.info("=" * 60)

    results = []
    errors  = []

    for i, case in enumerate(TEST_CASES, 1):
        text = case["text"]
        gt   = case["ground_truth"]

        logger.info(f"[{i:03d}/{len(TEST_CASES)}] {text[:55]}...")

        try:
            t0       = time.time()
            report   = pipeline.verify(text)
            elapsed  = time.time() - t0

            predicted  = report.overall_verdict.value
            confidence = report.overall_confidence
            correct    = predicted == gt

            results.append({
                "id":           case["id"],
                "text":         text,
                "category":     case["category"],
                "ground_truth": gt,
                "predicted":    predicted,
                "confidence":   confidence,
                "elapsed":      elapsed,
                "correct":      correct,
            })

            status = "OK" if correct else "NOT OK"
            logger.info(
                f"  {status} GT={gt:<20} "
                f"PRED={predicted:<20} "
                f"conf={confidence:.2f} "
                f"({elapsed:.1f}с)"
            )

        except Exception as e:
            logger.error(f"  Ошибка: {e}")
            errors.append({"id": case["id"], "error": str(e)})

        time.sleep(0.1)

    print_metrics(results, errors)
    save_results(results, errors)


def print_metrics(results: list, errors: list):
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    acc     = correct / total if total else 0

    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ МЕТРИКИ — КАСКАДНЫЙ ПАЙПЛАЙН")
    print("=" * 60)
    print(f"Всего тестов:    {len(TEST_CASES)}")
    print(f"Выполнено:       {total}")
    print(f"Ошибок:          {len(errors)}")
    print(f"Правильных:      {correct}")
    print(f"\nОБЩАЯ ТОЧНОСТЬ: {acc:.1%}  ({correct}/{total})")

    # Precision / Recall / F1
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
        support   = sum(1 for r in results if r["ground_truth"] == label)
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

    # Матрица ошибок
    print("\n" + "-" * 60)
    print("МАТРИЦА ОШИБОК (строки=GT, столбцы=Predicted)")
    short = {"SUPPORTED": "SUP", "REFUTED": "REF", "NOT_ENOUGH_INFO": "NEI"}
    print(f"{'GT \\ Pred':<22}" + "".join(f"{short[l]:>8}" for l in LABELS))
    for gt in LABELS:
        row = f"{gt:<22}"
        for pred in LABELS:
            count = sum(
                1 for r in results
                if r["ground_truth"] == gt and r["predicted"] == pred
            )
            row += f"{count:>8}"
        print(row)

    # По категориям
    print("\n" + "-" * 60)
    print("ТОЧНОСТЬ ПО КАТЕГОРИЯМ")
    print("-" * 60)
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["correct"])
    for cat, correctness in sorted(by_cat.items()):
        cat_acc = sum(correctness) / len(correctness)
        bar = "█" * int(cat_acc * 20) + "░" * (20 - int(cat_acc * 20))
        print(f"  {cat:<20} {bar} {cat_acc:.1%} ({sum(correctness)}/{len(correctness)})")

    # Неправильные предсказания
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n" + "-" * 60)
        print(f"НЕПРАВИЛЬНЫЕ ПРЕДСКАЗАНИЯ ({len(wrong)} шт.)")
        print("-" * 60)
        for r in wrong:
            print(
                f"  id={r['id']:02d} | "
                f"GT={r['ground_truth']:<20} "
                f"PRED={r['predicted']:<20} "
                f"conf={r['confidence']:.2f}"
            )
            print(f"         {r['text'][:70]}")

    # Среднее время
    print("\n" + "-" * 60)
    print("ВРЕМЯ ОТКЛИКА")
    print("-" * 60)
    times = [r["elapsed"] for r in results]
    print(f"  Среднее:    {sum(times)/len(times):.2f}с")
    print(f"  Минимальное:{min(times):.2f}с")
    print(f"  Максимальное:{max(times):.2f}с")

    # Сколько прошло через Mini vs Ollama
    fast = [r for r in results if r["elapsed"] < 1.0]
    slow = [r for r in results if r["elapsed"] >= 1.0]
    print(f"\n  Через Mini (быстро <1с): {len(fast)} запросов")
    print(f"  Через Ollama (>1с):      {len(slow)} запросов")
    if fast:
        print(f"  Точность Mini-filtered:  "
              f"{sum(1 for r in fast if r['correct'])/len(fast):.1%}")
    if slow:
        print(f"  Точность Ollama:         "
              f"{sum(1 for r in slow if r['correct'])/len(slow):.1%}")

    print("=" * 60)


def save_results(results: list, errors: list):
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    output  = {
        "total":    total,
        "correct":  correct,
        "accuracy": correct / total if total else 0,
        "results":  results,
        "errors":   errors,
    }
    with open("cascade_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.success("Результаты: cascade_evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()