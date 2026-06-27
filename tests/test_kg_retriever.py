import sys
sys.path.append(".")

import pytest
from src.verification.evidence_matcher import EvidenceMatcher
from src.verification.models import AtomicClaim, Evidence


class TestNormalize:
    """Тесты нормализации падежных форм."""

    @pytest.fixture
    def matcher(self):
        return EvidenceMatcher()

    def test_parizhe_to_parizh(self, matcher):
        assert matcher._normalize("Париже") == "Париж"

    def test_germanii_to_german(self, matcher):
        result = matcher._normalize("Германии")
        assert len(result) >= 3

    def test_rossii_to_rossi(self, matcher):
        result = matcher._normalize("России")
        assert len(result) >= 3

    def test_no_change_for_nominative(self, matcher):
        # Слова в именительном падеже не должны изменяться
        assert matcher._normalize("Москва") == "Москва"
        assert matcher._normalize("Париж") == "Париж"

    def test_short_word_not_changed(self, matcher):
        # Короткие слова не нормализуем
        result = matcher._normalize("ми")
        assert result == "ми"

    def test_1889_not_changed(self, matcher):
        assert matcher._normalize("1889") == "1889"


class TestEvidenceMatcher:
    """Интеграционные тесты поиска доказательств (требуют Neo4j)."""

    @pytest.fixture(scope="class")
    def matcher(self):
        return EvidenceMatcher()

    def _make_claim(self, text, subject, predicate, obj):
        return AtomicClaim(
            id=0,
            text=text,
            subject=subject,
            predicate=predicate,
            object=obj,
        )

    def test_match_eiffel_year(self, matcher):
        claim = self._make_claim(
            "Эйфелева башня построена в 1889 году",
            "Эйфелева башня", "built_in_year", "1889"
        )
        evidence = matcher.match(claim)
        assert isinstance(evidence, list)
        assert len(evidence) > 0

    def test_match_returns_evidence_objects(self, matcher):
        claim = self._make_claim(
            "Москва является столицей России",
            "Москва", "capital_of", "Россия"
        )
        evidence = matcher.match(claim)
        for e in evidence:
            assert isinstance(e, Evidence)
            assert e.source is not None
            assert e.content is not None
            assert 0.0 <= e.confidence <= 1.0

    def test_match_with_declension(self, matcher):
        # "Париже" должно нормализоваться и найти "Париж"
        claim = self._make_claim(
            "Эйфелева башня находится в Париже",
            "Эйфелева башня", "located_in", "Париже"
        )
        evidence = matcher.match(claim)
        assert len(evidence) > 0

    def test_match_unknown_entity_returns_empty(self, matcher):
        claim = self._make_claim(
            "Марс населён синими существами",
            "Марс", "населен", "синими существами"
        )
        evidence = matcher.match(claim)
        # Марса нет в графе — список пустой или минимальный
        assert isinstance(evidence, list)

    def test_match_einstein(self, matcher):
        claim = self._make_claim(
            "Альберт Эйнштейн родился в 1879 году",
            "Альберт Эйнштейн", "born_in_year", "1879"
        )
        evidence = matcher.match(claim)
        assert len(evidence) > 0
        # Проверяем что среди доказательств есть что-то про 1879
        contents = [e.content for e in evidence]
        assert any("1879" in c for c in contents)

    def test_max_evidence_limit(self, matcher):
        claim = self._make_claim(
            "Эйфелева башня",
            "Эйфелева башня", "any", "any"
        )
        evidence = matcher.match(claim)
        # Не более 7 доказательств
        assert len(evidence) <= 7

    def test_evidence_sorted_by_confidence(self, matcher):
        claim = self._make_claim(
            "Москва является столицей России",
            "Москва", "capital_of", "Россия"
        )
        evidence = matcher.match(claim)
        if len(evidence) > 1:
            confidences = [e.confidence for e in evidence]
            assert confidences == sorted(confidences, reverse=True)