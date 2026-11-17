"""Validated EssayLoop configuration. Imports never construct model clients."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    model: str = Field(default="qwen-plus", min_length=1)
    evaluator_model: str | None = Field(default=None, min_length=1)
    temperature: float = Field(default=0.3, ge=0, le=1)
    evaluator_temperature: float = Field(default=0.0, ge=0, le=1)
    threshold: float = Field(default=0.85, ge=0, le=1)
    min_dimension_score: float | None = Field(default=None, ge=0, le=1)
    max_rounds: int = Field(default=3, ge=1, le=20)
    max_attempts: int = Field(default=2, ge=1, le=5)
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    retry_delay_seconds: float = Field(default=1, ge=0, le=30)
    max_output_tokens: int = Field(default=2048, ge=128, le=32768)
    trust_env: bool = True
    execution: Literal["parallel", "serial"] = "parallel"
    feedback_mode: Literal["all", "relevance", "none"] = "all"
    stop_policy: Literal["threshold", "fixed"] = "threshold"


def load_environment() -> None:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
