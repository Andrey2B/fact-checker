import sys
sys.path.append(".")

import pytest
from src.llm.claim_decomposer import ClaimDecomposer, _clean_object, _is_valid_subject


class TestCleanObject:
    """Тесты очистки объекта от шумовых слов."""

    def test_removes_году(self):
        assert _clean_object("1889 году") == "1889"

    def test_removes_metrov(self):
        assert _clean_object("330 метров") == "330"

    def test_keeps_clean_value(self):
        assert _clean_object("Париж") == "Париж"

    def test_keeps_name(self):
        assert _clean_object("Гюстав Эйфель") == "Гюстав Эйфель"

    def test_empty_string(self):
        assert _clean_object("") == ""

    def test_only_noise(self):
        # Если только шумовое слово — возвращаем оригинал
        result = _clean_object("году")
        assert result == "году"


class TestIsValidSubject:
    """Тесты валидации субъекта."""

    def test_valid_name(self):
        assert _is_valid_subject("Эйфелева башня") is True

    def test_valid_person(self):
        assert _is_valid_subject("Альберт Эйнштейн") is True

    def test_invalid_year(self):
        assert _is_valid_subject("1889") is False

    def test_invalid_short_year(self):
        assert _is_valid_subject("89") is False

    def test_invalid_empty(self):
        assert _is_valid_subject("") is False

    def test_invalid_one_char(self):
        assert _is_valid_subject("А") is False

    def test_valid_city(self):
        assert _is_valid_subject("Москва") is True

    def test_invalid_four_digits(self):
        assert _is_valid_subject("2026") is False


class TestClaimDecomposer:
    """Интеграционные тесты декомпозиции (требуют Ollama)."""

    @pytest.fixture(scope="class")
    def decomposer(self):
        return ClaimDecomposer()

    def test_decompose_returns_list(self, decomposer):
        result = decomposer.decompose("Москва является столицей России")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_decompose_eiffel_tower(self, decomposer):
        result = decomposer.decompose(
            "Эйфелева башня построена в 1889 году в Париже"
        )
        assert len(result) >= 1

        subjects = [c.subject for c in result]
        assert "Эйфелева башня" in subjects

    def test_decompose_einstein(self, decomposer):
        result = decomposer.decompose(
            "Альберт Эйнштейн родился в 1879 году в Германии"
        )
        subjects = [c.subject for c in result]

        # LLM может написать с опечаткой "Эйнщтейн" вместо "Эйнштейн"
        # проверяем только что субъект содержит "Эйн"
        # и точно не является годом или страной
        for s in subjects:
            assert "Эйн" in s, f"Субъект должен содержать Эйн: {s}"
            assert s != "1879", "Год не должен быть субъектом"
            assert s != "Германии", "Страна не должна быть субъектом"
            assert s != "Германия", "Страна не должна быть субъектом"

    def test_no_year_as_subject(self, decomposer):
        result = decomposer.decompose(
            "Python создан в 1991 году Гвидо ван Россумом"
        )
        for claim in result:
            assert not claim.subject.isdigit(), (
                f"Год не должен быть субъектом: {claim.subject}"
            )

    def test_claim_has_required_fields(self, decomposer):
        result = decomposer.decompose("Москва является столицей России")
        for claim in result:
            assert claim.subject is not None
            assert claim.predicate is not None
            assert claim.object is not None
            assert claim.text is not None

    def test_fallback_on_empty(self, decomposer):
        result = decomposer.decompose("")
        assert isinstance(result, list)