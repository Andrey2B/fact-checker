from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"
    CONFLICTING = "CONFLICTING"


class AtomicClaim(BaseModel):
    id: int
    text: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None


class Evidence(BaseModel):
    source: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    cypher_query: Optional[str] = None


class ClaimVerificationResult(BaseModel):
    claim: AtomicClaim
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = []
    explanation: str


class VerificationReport(BaseModel):
    original_text: str
    overall_verdict: Verdict
    overall_confidence: float
    claims: list[ClaimVerificationResult]
    summary: str
    supported_count: int = 0
    refuted_count: int = 0
    not_enough_info_count: int = 0

    def model_post_init(self, __context):
        self.supported_count = sum(
            1 for c in self.claims if c.verdict == Verdict.SUPPORTED
        )
        self.refuted_count = sum(
            1 for c in self.claims if c.verdict == Verdict.REFUTED
        )
        self.not_enough_info_count = sum(
            1 for c in self.claims if c.verdict == Verdict.NOT_ENOUGH_INFO
        )