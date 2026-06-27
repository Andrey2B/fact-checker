import json
import re
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate

from src.llm.llm_client import get_llm
from src.verification.models import (
    AtomicClaim, Evidence, ClaimVerificationResult, Verdict
)

VERDICT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a fact verification judge.\n"
     "Check if the CLAIM is supported by the EVIDENCE.\n\n"
     "Rules:\n"
     "- SUPPORTED: evidence directly confirms the claim\n"
     "- REFUTED: evidence directly contradicts the claim "
     "(different value for same attribute)\n"
     "- NOT_ENOUGH_INFO: evidence is about the same entity "
     "but does NOT mention this specific attribute, "
     "OR no evidence at all\n"
     "- CONFLICTING: some evidence confirms, "
     "other evidence contradicts the SAME attribute\n\n"
     "IMPORTANT:\n"
     "- Evidence about OTHER attributes of the entity "
     "is NOT_ENOUGH_INFO for THIS claim\n"
     "- Example: claim 'Eiffel Tower built in 1889'\n"
     "  evidence 'Eiffel Tower --[ВЫСОТА]--> 330 метров' "
     "→ NOT_ENOUGH_INFO (different attribute)\n"
     "  evidence 'Eiffel Tower --[ПОСТРОЕНА_В]--> 1889' "
     "→ SUPPORTED\n"
     "  evidence 'Eiffel Tower --[ПОСТРОЕНА_В]--> 1900' "
     "→ REFUTED\n\n"
     "Return ONLY JSON:\n"
     "{{\"verdict\": \"SUPPORTED|REFUTED|NOT_ENOUGH_INFO|CONFLICTING\","
     "\"confidence\": 0.0-1.0, \"explanation\": \"brief reason\"}}"),
    ("human",
     "CLAIM: {claim}\n\n"
     "EVIDENCE:\n{evidence}\n\n"
     "JSON verdict:")
])


def _parse_verdict(text: str) -> dict:
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


class VerdictGenerator:
    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.chain = VERDICT_PROMPT | self.llm

    def generate(
        self, claim: AtomicClaim, evidence: list[Evidence]
    ) -> ClaimVerificationResult:

        if evidence:
            evidence_text = "\n".join(
                f"[{i+1}] {e.content}"
                for i, e in enumerate(evidence)
            )
        else:
            evidence_text = "No evidence found."

        # Используем полный текст утверждения
        claim_text = claim.text
        if claim.subject and claim.object:
            claim_text = (
                f"{claim.subject} {claim.predicate} {claim.object}"
            )

        logger.debug(f"Генерация вердикта: {claim_text[:60]}")

        try:
            response = self.chain.invoke({
                "claim": claim_text,
                "evidence": evidence_text,
            })
            raw = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            logger.debug(f"Ответ LLM (вердикт): {raw[:300]}")

            parsed = _parse_verdict(raw)
            verdict_str = (
                parsed.get("verdict", "NOT_ENOUGH_INFO").upper().strip()
            )
            verdict_map = {
                "SUPPORTED": Verdict.SUPPORTED,
                "REFUTED": Verdict.REFUTED,
                "NOT_ENOUGH_INFO": Verdict.NOT_ENOUGH_INFO,
                "NOT ENOUGH INFO": Verdict.NOT_ENOUGH_INFO,
                "NOTENOUGHINFO": Verdict.NOT_ENOUGH_INFO,
                "CONFLICTING": Verdict.CONFLICTING,
            }
            verdict = verdict_map.get(verdict_str, Verdict.NOT_ENOUGH_INFO)
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            explanation = parsed.get("explanation", "Нет объяснения")

            return ClaimVerificationResult(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                explanation=explanation,
            )

        except Exception as e:
            logger.error(f"Ошибка вердикта: {e}")
            return ClaimVerificationResult(
                claim=claim,
                verdict=Verdict.NOT_ENOUGH_INFO,
                confidence=0.0,
                evidence=evidence,
                explanation=f"Ошибка LLM: {str(e)}",
            )