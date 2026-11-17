"""Strict parsing: invalid responses never become normal scores."""
import hashlib
import json
import re
from src.schemas import Dimension, Judgment


def essay_digest(essay: str) -> str:
    return hashlib.sha256(essay.encode("utf-8")).hexdigest()


def parse_judgment(content: str, dimension: Dimension, essay: str) -> Judgment:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1)

    def unique_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON key")
            obj[key] = value
        return obj

    result = Judgment.model_validate(json.loads(text, object_pairs_hook=unique_keys))
    if result.dimension != dimension:
        raise ValueError("Wrong dimension")
    if any(quote not in essay for quote in result.evidence):
        raise ValueError("Evidence must quote an exact substring of the essay")
    return result
