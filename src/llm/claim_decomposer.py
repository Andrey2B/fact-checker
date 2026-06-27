import json
import re
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate

from src.llm.llm_client import get_llm
from src.verification.models import AtomicClaim

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a fact extraction assistant. "
     "Extract atomic facts from the text as subject-predicate-object triples.\n"
     "Return ONLY a JSON array, no other text.\n\n"
     "STRICT RULES for subject/object:\n"
     "- subject: ALWAYS the main named entity (person, building, city, country)\n"
     "- object: ALWAYS the value, date, place, or related entity\n"
     "- NEVER put a date, year, place or attribute as subject\n"
     "- For 'X was born in YEAR in PLACE': "
     "subject=X, one triple (X, born_in_year, YEAR), "
     "another triple (X, born_in_place, PLACE)\n"
     "- For 'X was built in YEAR in PLACE': "
     "subject=X for BOTH triples\n"
     "- For 'X is capital of Y': subject=X, object=Y\n\n"
     "Each item must have:\n"
     "- id: number starting from 0\n"
     "- text: copy relevant part of original text\n"
     "- subject: the main entity (NEVER a year or place modifier)\n"
     "- predicate: the relationship in snake_case\n"
     "- object: the value or target entity\n\n"
     "Example: 'Albert Einstein was born in 1879 in Germany'\n"
     "Output:\n"
     "[{{\"id\": 0, \"text\": \"...\", \"subject\": \"Albert Einstein\","
     "\"predicate\": \"born_in_year\", \"object\": \"1879\"}},"
     "{{\"id\": 1, \"text\": \"...\", \"subject\": \"Albert Einstein\","
     "\"predicate\": \"born_in_place\", \"object\": \"Germany\"}}]\n\n"
     "Example: 'Paris is capital of France'\n"
     "[{{\"id\": 0, \"text\": \"Paris is capital of France\","
     "\"subject\": \"Paris\", \"predicate\": \"capital_of\","
     "\"object\": \"France\"}}]"),
    ("human", "Extract atomic facts from:\n{text}")
])

NOISE_WORDS = [
    "году", "год", "лет", "года", "г.",
    "метров", "метра", "метр",
    "километров", "км",
    "человек", "людей",
]

def _is_valid_subject(subject: str) -> bool:
    """Отсеиваем субъекты-даты, годы, числа"""
    if not subject:
        return False
    if re.fullmatch(r'\d{1,4}', subject.strip()):  # ← re, не _re
        return False
    if len(subject.strip()) < 2:
        return False
    return True

def _clean_object(obj: str) -> str:
    if not obj:
        return obj
    words = obj.split()
    cleaned = [w for w in words if w.lower() not in NOISE_WORDS]
    return " ".join(cleaned).strip() if cleaned else obj


def _extract_json(text: str) -> list[dict]:
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    return []


class ClaimDecomposer:
    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.chain = DECOMPOSE_PROMPT | self.llm

    def decompose(self, text: str) -> list[AtomicClaim]:
        logger.info(f"Декомпозиция: {text[:80]}...")
        try:
            response = self.chain.invoke({"text": text})
            raw_text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            logger.debug(f"Ответ LLM: {raw_text[:300]}")

            items = _extract_json(raw_text)
            if not items:
                logger.warning(f"Невалидный JSON: {raw_text[:200]}")
                return self._fallback(text)

            claims = []
            for item in items:
                subj = item.get("subject", "")
                if not _is_valid_subject(subj):
                    logger.warning(
                        f"  Пропуск невалидного субъекта: '{subj}'"
                    )
                    continue
                clean_obj = _clean_object(item.get("object", ""))
                claim = AtomicClaim(
                    id=item.get("id", len(claims)),
                    text=item.get("text", text),
                    subject=subj,
                    predicate=item.get("predicate"),
                    object=clean_obj if clean_obj else item.get("object"),
                )
                claims.append(claim)
                logger.debug(
                    f"  Утверждение [{claim.id}]: "
                    f"({claim.subject})-[{claim.predicate}]->({claim.object})"
                )

            logger.info(f"Извлечено {len(claims)} утверждений")
            return claims

        except Exception as e:
            logger.error(f"Ошибка декомпозиции: {e}")
            return self._fallback(text)

    def _fallback(self, text: str) -> list[AtomicClaim]:
        return [AtomicClaim(id=0, text=text)]