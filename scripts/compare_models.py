"""
Сравниваем Ollama LLM vs MiniVerifier на одном датасете.
"""
import sys
import json
import time
import torch
import urllib.request
from pathlib import Path
from collections import defaultdict

sys.path.append(".")
from loguru import logger
from src.llm.mini_lm.model      import MiniVerifier
from src.llm.mini_lm.tokenizer  import SimpleTokenizer
from src.graph.neo4j_client     import Neo4jClient
from src.verification.models    import AtomicClaim
from tests.test_dataset_wikidata import TEST_CASES

MODEL_PATH = Path("data/mini_lm/model.pt")
TOK_PATH   = Path("data/mini_lm/tokenizer.json")
BASE_URL   = "http://localhost:8000/api/v1"
LABELS     = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


# ── Ollama через API ───────────────────────────────────────────
def verify_ollama(text: str) -> dict:
    data = json.dumps({"text": text}).encode("utf-8")
    req  = urllib.request.Request(
        f"{BASE_URL}/verify", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


# ── MiniVerifier ──────────────────────────────────────────────
class MiniVerifierPredictor:
    def __init__(self):
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

        checkpoint  = torch.load(MODEL_PATH, map_location=self.device)
        self.model  = MiniVerifier(checkpoint["config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.tokenizer = SimpleTokenizer.load(TOK_PATH)
        self.client    = Neo4jClient.get_instance()

        logger.info(
            f"MiniVerifier загружен | "
            f"val_acc={checkpoint['val_acc']:.1%} | "
            f"params={self.model.count_params():,}"
        )

    def get_evidence(self, text: str) -> str:
        """Получаем доказательства из графа."""
        words = [w for w in text.split() if len(w) > 3][:3]
        if not words:
            return "no evidence"

        all_rows = []
        for word in words:
            rows = self.client.run_query("""
                MATCH (s:Entity)-[r:RELATION]->(o:Entity)
                WHERE toLower(s.name) CONTAINS toLower($word)
                RETURN s.name AS subject,
                       r.type AS predicate,
                       o.name AS object
                LIMIT 3
            """, {"word": word})
            all_rows.extend(rows)

        if not all_rows:
            return "no evidence"

        return " | ".join(
            f"{r['subject']} {r['predicate']} {r['object']}"
            for r in all_rows[:5]
        )

    def predict(self, text: str) -> dict:
        t0       = time.time()
        evidence = self.get_evidence(text)

        ids, mask = self.tokenizer.encode(text, evidence, max_length=256)
        input_ids = torch.tensor([ids],  dtype=torch.long).to(self.device)
        attn_mask = torch.tensor([mask], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits      = self.model(input_ids, attn_mask)
            probs       = torch.softmax(logits, dim=-1)[0]
            pred_idx    = probs.argmax().item()

        verdict    = LABELS[pred_idx]
        confidence = probs[pred_idx].item()
        elapsed    = time.time() - t0

        return {
            "overall_verdict":    verdict,
            "overall_confidence": round(confidence, 3),
            "elapsed":            elapsed,
        }


# ── Метрики ───────────────────────────────────────────────────
def compute_metrics(results: list[dict], model_name: str) -> dict:
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    acc     = correct / total if total else 0

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
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_scores.append(f1)

    macro_f1   = sum(f1_scores) / len(f1_scores)
    avg_conf   = sum(r["confidence"] for r in results) / total
    avg_time   = sum(r.get("elapsed", 0) for r in results) / total

    return {
        "model":     model_name,
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "avg_conf":  avg_conf,
        "avg_time":  avg_time,
        "total":     total,
        "correct":   correct,
    }


def print_comparison(metrics_a: dict, metrics_b: dict):
    print("\n" + "=" * 65)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 65)
    print(f"{'Метрика':<25} {metrics_a['model']:>18} {metrics_b['model']:>18}")
    print("-" * 65)

    rows = [
        ("Точность (Accuracy)", "accuracy", ".1%"),
        ("Macro F1",            "macro_f1", ".1%"),
        ("Средняя уверенность", "avg_conf", ".3f"),
        ("Время на запрос (с)", "avg_time", ".2f"),
        ("Параметры",           "params",   ""),
    ]

    for name, key, fmt in rows:
        a = metrics_a.get(key, "—")
        b = metrics_b.get(key, "—")
        if fmt and isinstance(a, float):
            a_str = format(a, fmt)
            b_str = format(b, fmt)
        else:
            a_str = str(a)
            b_str = str(b)

        # Победитель
        if isinstance(a, float) and isinstance(b, float):
            marker_a = " ✅" if a > b else ""
            marker_b = " ✅" if b > a else ""
        else:
            marker_a = marker_b = ""

        print(f"{name:<25} {a_str+marker_a:>18} {b_str+marker_b:>18}")

    print("=" * 65)


def run_comparison():
    logger.info("Загружаем MiniVerifier...")
    mini = MiniVerifierPredictor()

    results_ollama = []
    results_mini   = []

    total = len(TEST_CASES)

    for i, case in enumerate(TEST_CASES, 1):
        text = case["text"]
        gt   = case["ground_truth"]
        logger.info(f"[{i:03d}/{total}] {text[:55]}...")

        # ── Ollama ─────────────────────────────────────────────
        t0 = time.time()
        try:
            resp_o     = verify_ollama(text)
            pred_o     = resp_o["overall_verdict"]
            conf_o     = resp_o["overall_confidence"]
            elapsed_o  = time.time() - t0
        except Exception as e:
            logger.warning(f"  Ollama error: {e}")
            pred_o, conf_o, elapsed_o = "NOT_ENOUGH_INFO", 0.0, 0.0

        results_ollama.append({
            "ground_truth": gt,
            "predicted":    pred_o,
            "confidence":   conf_o,
            "elapsed":      elapsed_o,
            "correct":      pred_o == gt,
        })

        # ── MiniVerifier ───────────────────────────────────────
        resp_m    = mini.predict(text)
        pred_m    = resp_m["overall_verdict"]
        conf_m    = resp_m["overall_confidence"]
        elapsed_m = resp_m["elapsed"]

        results_mini.append({
            "ground_truth": gt,
            "predicted":    pred_m,
            "confidence":   conf_m,
            "elapsed":      elapsed_m,
            "correct":      pred_m == gt,
        })

        status_o = "✅" if pred_o == gt else "❌"
        status_m = "✅" if pred_m == gt else "❌"

        logger.info(
            f"  Ollama: {status_o} {pred_o:<22} ({elapsed_o:.1f}s) | "
            f"Mini: {status_m} {pred_m:<22} ({elapsed_m:.3f}s)"
        )

        time.sleep(0.3)

    # ── Метрики ────────────────────────────────────────────────
    m_ollama = compute_metrics(results_ollama, "Ollama (llama3)")
    m_mini   = compute_metrics(results_mini,   "MiniVerifier")

    m_ollama["params"] = "~8B"
    m_mini["params"]   = f"~{mini.model.count_params()//1_000_000}M"

    print_comparison(m_ollama, m_mini)

    # Сохраняем
    output = {
        "ollama":  {"metrics": m_ollama, "results": results_ollama},
        "mini_lm": {"metrics": m_mini,   "results": results_mini},
    }
    with open("comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.success("Результаты: comparison_results.json")


if __name__ == "__main__":
    run_comparison()