from loguru import logger
from src.llm.claim_decomposer import ClaimDecomposer
from src.llm.verdict_generator import VerdictGenerator
from src.verification.evidence_matcher import EvidenceMatcher
from src.verification.models import (
    VerificationReport, ClaimVerificationResult, Verdict
)


class VerificationPipeline:
    def __init__(self):
        self.decomposer = ClaimDecomposer()
        self.matcher = EvidenceMatcher()
        self.verdict_gen = VerdictGenerator()
        logger.info("✅ VerificationPipeline готов (локальная LLM)")

    def verify(self, text: str) -> VerificationReport:
        logger.info("=" * 50)
        logger.info(f"Верификация: {text[:100]}...")

        # Шаг 1: Декомпозиция
        claims = self.decomposer.decompose(text)
        logger.info(f"[1/3] Декомпозиция → {len(claims)} утверждений")

        # Шаги 2-3: Поиск и вердикт
        results: list[ClaimVerificationResult] = []
        for claim in claims:
            evidence = self.matcher.match(claim)
            logger.info(
                f"[2/3] '{claim.text[:40]}...' "
                f"→ {len(evidence)} доказательств"
            )
            result = self.verdict_gen.generate(claim, evidence)
            results.append(result)
            logger.info(
                f"[3/3] Вердикт: {result.verdict} "
                f"(conf={result.confidence:.2f})"
            )

        overall = self._overall_verdict(results)
        confidence = self._overall_confidence(results)
        summary = self._summary(results)

        report = VerificationReport(
            original_text=text,
            overall_verdict=overall,
            overall_confidence=confidence,
            claims=results,
            summary=summary,
        )
        logger.info(f"Итог: {overall} (conf={confidence:.2f})")
        return report

    def _overall_verdict(
        self, results: list[ClaimVerificationResult]
    ) -> Verdict:
        if not results:
            return Verdict.NOT_ENOUGH_INFO

        verdicts = [r.verdict for r in results]

        # Считаем количество каждого вердикта
        n_supported = verdicts.count(Verdict.SUPPORTED)
        n_refuted = verdicts.count(Verdict.REFUTED)
        n_not_enough = verdicts.count(Verdict.NOT_ENOUGH_INFO)
        n_conflicting = verdicts.count(Verdict.CONFLICTING)
        total = len(verdicts)

        # Если есть REFUTED — итог REFUTED
        if n_refuted > 0:
            return Verdict.REFUTED

        # Все SUPPORTED
        if n_supported == total:
            return Verdict.SUPPORTED

        # Большинство SUPPORTED (включая CONFLICTING)
        # CONFLICTING = LLM не уверена, не обязательно ложь
        if n_supported > 0:
            if n_conflicting > 0 and n_supported >= n_conflicting:
                return Verdict.SUPPORTED
            if n_not_enough > 0:
                return Verdict.SUPPORTED

        # Все NOT_ENOUGH_INFO
        if n_not_enough == total:
            return Verdict.NOT_ENOUGH_INFO

        # Только CONFLICTING
        if n_conflicting == total:
            return Verdict.CONFLICTING

        return Verdict.NOT_ENOUGH_INFO

    def _overall_confidence(
            self, results: list[ClaimVerificationResult]
    ) -> float:
        if not results:
            return 0.0

        # NOT_ENOUGH_INFO с нулевой уверенностью не должен
        # занижать итог — берём только информативные results
        informative = [
            r for r in results
            if r.verdict != Verdict.NOT_ENOUGH_INFO
        ]

        # Если есть информативные — считаем по ним
        if informative:
            return round(
                sum(r.confidence for r in informative) / len(informative),
                3
            )

        # Все NOT_ENOUGH_INFO — возвращаем 0
        return 0.0

    def _summary(self, results: list[ClaimVerificationResult]) -> str:
        s = sum(1 for r in results if r.verdict == Verdict.SUPPORTED)
        r = sum(1 for r in results if r.verdict == Verdict.REFUTED)
        n = sum(
            1 for r in results if r.verdict == Verdict.NOT_ENOUGH_INFO
        )
        c = sum(
            1 for r in results if r.verdict == Verdict.CONFLICTING
        )
        return (
            f"Проверено {len(results)} утверждений: "
            f"✅ {s} подтверждено, "
            f"❌ {r} опровергнуто, "
            f"❓ {n} без данных, "
            f"⚠️ {c} противоречивых."
        )