"""Validated input, judgment, failure, and round contracts."""
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Dimension = Literal["relevance", "grammar", "structure", "depth"]
DIMENSIONS: tuple[Dimension, ...] = ("relevance", "grammar", "structure", "depth")
WEIGHTS = {"relevance": 0.3, "grammar": 0.2, "structure": 0.2, "depth": 0.3}
ExitReason = Literal["threshold_met", "max_rounds", "evaluation_failed", "generation_failed"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EvaluationInput(Contract):
    topic: str = Field(min_length=1)
    essay: str | None = None

    @field_validator("topic", "essay")
    @classmethod
    def nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Text must not be blank")
        return value


class Judgment(Contract):
    dimension: Dimension
    score: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    suggestions: list[str] = Field(min_length=1)

    @field_validator("score", mode="before")
    @classmethod
    def numeric_score(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("score must be a number, not text or boolean")
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value

    @field_validator("rationale")
    @classmethod
    def nonblank_rationale(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rationale must not be blank")
        return value

    @field_validator("evidence", "suggestions")
    @classmethod
    def nonblank_items(cls, values: list[str]) -> list[str]:
        if any(not v.strip() for v in values):
            raise ValueError("List items must not be blank")
        return values


class AttemptRecord(Contract):
    attempt: int = Field(ge=1)
    status: Literal["ok", "parse_error", "provider_error"]
    raw_response: str | None = None
    error: str | None = None
    http_status: int | None = None
    timeout_phase: Literal["connect", "read", "write", "pool"] | None = None
    connection_cause: Literal["dns_resolution_failed", "tls_certificate_error", "tls_handshake_error",
                              "connection_refused", "connection_reset", "network_unreachable", "connection_failed"] | None = None
    provider_code: str | None = None
    provider_reason: str | None = None
    response_format: Literal["json", "html", "other"] | None = None
    retryable: bool | None = None
    hint: str | None = None


class DimensionResult(Contract):
    round_id: int = Field(ge=1)
    essay_hash: str
    dimension: Dimension
    status: Literal["ok", "parse_error", "provider_error"]
    judgment: Judgment | None = None
    attempts: list[AttemptRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def coherent(self) -> "DimensionResult":
        if (self.status == "ok") != (self.judgment is not None):
            raise ValueError("Only successful results may contain a judgment")
        if self.judgment and self.judgment.dimension != self.dimension:
            raise ValueError("Judgment dimension mismatch")
        if self.attempts[-1].status != self.status:
            raise ValueError("Last attempt and result status must match")
        return self


class RoundRecord(Contract):
    round_id: int = Field(ge=1)
    essay: str
    results: list[DimensionResult]
    final_score: float | None = Field(default=None, ge=0, le=1)
    feedback: str
    exit_reason: ExitReason | None = None
