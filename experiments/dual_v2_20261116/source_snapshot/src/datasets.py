"""Validated fixed-essay JSONL datasets. No fabricated human labels."""
import hashlib
import json
from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
from src.schemas import Contract, DIMENSIONS, EvaluationInput


class Sample(Contract):
    sample_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    topic: str
    essay: str
    split: Literal["dev", "test"]
    source: str = Field(min_length=1)
    human_scores: dict[str, float] | None = None

    @field_validator("human_scores")
    @classmethod
    def validate_scores(cls, value):
        if value is not None:
            if set(value) != set(DIMENSIONS) or any(isinstance(x, bool) or not 0 <= x <= 1 for x in value.values()):
                raise ValueError("Human labels must contain all four dimensions with scores in [0,1]")
        return value

    @field_validator("topic", "essay", "source")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Sample text cannot be blank")
        return value


def load_dataset(path: Path):
    path = Path(path)
    data = path.read_bytes()
    samples = [Sample.model_validate(json.loads(line)).model_dump() for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    ids = [s["sample_id"] for s in samples]
    if not samples or len(ids) != len(set(ids)):
        raise ValueError("Dataset must be nonempty and have unique sample_id values")
    return samples, hashlib.sha256(data).hexdigest()
