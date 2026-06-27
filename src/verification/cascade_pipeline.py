import sys
import torch
sys.path.append(".")

from loguru import logger
from pathlib import Path

from src.verification.pipeline import VerificationPipeline
from src.verification.models import VerificationReport, Verdict
from src.llm.mini_lm.model import MiniVerifier
from src.llm.mini_lm.tokenizer import SimpleTokenizer
from src.graph.neo4j_client import Neo4jClient

MODEL_PATH = Path("data/mini_lm/model.pt")
TOK_PATH   = Path("data/mini_lm/tokenizer.json")
LABELS     = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


class MiniVerifierPredictor:
    def __init__(self):
        device_name   = "cuda" if torch.cuda.is_available() else "cpu"
        self.device   = torch.device(device_name)
        checkpoint    = torch.load(MODEL_PATH, map_location=self.device)
        self.model    = MiniVerifier(checkpoint["config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.tokenizer = SimpleTokenizer.load(TOK_PATH)
        self.client    = Neo4jClient.get_instance()
        logger.info(f"MiniVerifier загружен | val_acc={checkpoint['val_acc']:.1%}")

    def get_evidence(self, text: str) -> str:
        words = [w for w in text.split() if len(w) > 3][:3]
        if not words:
            return "no evidence"
        all_rows = []
        for word in words:
            rows = self.client.run_query("""
                MATCH (s:Entity)-[r:RELATION]->(o:Entity)
                WHERE toLower(s.name) CONTAINS toLower($word)
                RETURN s.name AS subject, r.type AS predicate, o.name AS object
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
        evidence  = self.get_evidence(text)
        ids, mask = self.tokenizer.encode(text, evidence, max_length=256)
        input_ids = torch.tensor([ids],  dtype=torch.long).to(self.device)
        attn_mask = torch.tensor([mask], dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits   = self.model(input_ids, attn_mask)
            probs    = torch.softmax(logits, dim=-1)[0]
            pred_idx = probs.argmax().item()
        return {
            "overall_verdict":    LABELS[pred_idx],
            "overall_confidence": round(probs[pred_idx].item(), 3),
        }


class CascadePipeline:
    """
    Каскадная верификация:
    Уровень 1: MiniVerifier (~0.01с) — быстрый фильтр NEI
    Уровень 2: Ollama Llama3 (~8с)   — полная верификация
    """

    def __init__(self):
        self.mini     = MiniVerifierPredictor()
        self.pipeline = VerificationPipeline()
        logger.info("CascadePipeline готов")

    def verify(self, text: str) -> VerificationReport:
        logger.info(f"Каскад: {text[:60]}...")

        # ── Уровень 1: проверяем есть ли субъект в графе ──────
        subject_in_graph = self._subject_in_graph(text)
        logger.info(f"  Субъект в графе: {subject_in_graph}")

        if subject_in_graph:
            # Субъект найден → всегда идём в Ollama
            logger.info("  → Субъект в графе, передаём в Ollama")
            report = self.pipeline.verify(text)
            logger.info(
                f"  L2 Ollama: {report.overall_verdict} "
                f"(conf={report.overall_confidence:.2f})"
            )
            return report

        # ── Субъекта нет в графе — спрашиваем MiniVerifier ────
        mini_result = self.mini.predict(text)
        mini_verdict = mini_result["overall_verdict"]
        mini_conf = mini_result["overall_confidence"]
        logger.info(f"  L1 Mini: {mini_verdict} (conf={mini_conf:.2f})")

        # NEI с высокой уверенностью — пропускаем Ollama
        if mini_verdict == "NOT_ENOUGH_INFO" and mini_conf > 0.90:
            logger.info("  → Субъекта нет в графе + Mini уверен: NEI")
            return self._make_report(
                text=text,
                verdict=Verdict.NOT_ENOUGH_INFO,
                confidence=mini_conf,
                method="mini_filter",
            )

        # Во всех остальных случаях → Ollama
        logger.info("  → Передаём в Ollama")
        report = self.pipeline.verify(text)
        logger.info(
            f"  L2 Ollama: {report.overall_verdict} "
            f"(conf={report.overall_confidence:.2f})"
        )
        return report

    def _subject_in_graph(self, text: str) -> bool:
        """Проверяем есть ли хоть одно слово из текста в графе."""
        words = [w for w in text.split() if len(w) > 3][:5]
        for word in words:
            rows = self.mini.client.run_query("""
                MATCH (s:Entity)
                WHERE toLower(s.name) CONTAINS toLower($word)
                RETURN s.name LIMIT 1
            """, {"word": word})
            if rows:
                logger.debug(f"  Найден субъект: '{rows[0]['s.name']}' по слову '{word}'")
                return True
        return False

    def _make_report(
        self,
        text:       str,
        verdict:    Verdict,
        confidence: float,
        method:     str,
    ) -> VerificationReport:
        from src.verification.models import ClaimVerificationResult, AtomicClaim

        claim = AtomicClaim(id=0, text=text)
        result = ClaimVerificationResult(
            claim       = claim,
            verdict     = verdict,
            confidence  = confidence,
            evidence    = [],
            explanation = f"Классифицировано MiniVerifier ({method})",
        )
        return VerificationReport(
            original_text      = text,
            overall_verdict    = verdict,
            overall_confidence = confidence,
            claims             = [result],
            summary            = (
                f"Быстрый фильтр (MiniVerifier): "
                f"{verdict.value} (conf={confidence:.2f})"
            ),
        )