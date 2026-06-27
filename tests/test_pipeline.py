import pytest
from unittest.mock import patch, MagicMock
from src.verification.models import *


@pytest.fixture
def pipeline():
    with patch("src.verification.pipeline.ClaimDecomposer") as D, \
         patch("src.verification.pipeline.EvidenceMatcher") as M, \
         patch("src.verification.pipeline.VerdictGenerator") as V:

        from src.verification.pipeline import VerificationPipeline
        p = VerificationPipeline()
        p.decomposer = D()
        p.matcher = M()
        p.verdict_gen = V()
        yield p


def test_supported(pipeline):
    claim = AtomicClaim(id=0, text="X", subject="A", predicate="B", object="C")
    evidence = [Evidence(source="kg", content="A-B->C", confidence=0.9)]
    result = ClaimVerificationResult(
        claim=claim, verdict=Verdict.SUPPORTED,
        confidence=0.9, evidence=evidence, explanation="ok"
    )

    pipeline.decomposer.decompose.return_value = [claim]
    pipeline.matcher.match.return_value = evidence
    pipeline.verdict_gen.generate.return_value = result

    report = pipeline.verify("test")
    assert report.overall_verdict == Verdict.SUPPORTED
    assert report.supported_count == 1


def test_refuted_overrides(pipeline):
    claims = [
        AtomicClaim(id=0, text="A"),
        AtomicClaim(id=1, text="B"),
    ]
    results = [
        ClaimVerificationResult(
            claim=claims[0], verdict=Verdict.SUPPORTED,
            confidence=0.9, explanation="ok"
        ),
        ClaimVerificationResult(
            claim=claims[1], verdict=Verdict.REFUTED,
            confidence=0.8, explanation="wrong"
        ),
    ]

    pipeline.decomposer.decompose.return_value = claims
    pipeline.matcher.match.return_value = []
    pipeline.verdict_gen.generate.side_effect = results

    report = pipeline.verify("test")
    assert report.overall_verdict == Verdict.REFUTED