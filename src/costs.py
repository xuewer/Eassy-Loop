"""Optional offline cost calculation using an explicit dated price snapshot."""
from datetime import date
import math


def validate_prices(data):
    if set(data) != {"currency", "as_of", "source", "models"}:
        raise ValueError("Price table requires currency, as_of, source, models")
    date.fromisoformat(data["as_of"])
    if not isinstance(data["currency"], str) or not data["currency"].strip() or not isinstance(data["source"], str) or not data["source"].strip():
        raise ValueError("Currency and price source are required")
    if not isinstance(data["models"], dict) or not data["models"]:
        raise ValueError("At least one explicit model price is required")
    for rates in data["models"].values():
        if set(rates) != {"input_per_million", "output_per_million"}:
            raise ValueError("Each model requires input/output prices per million tokens")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in rates.values()):
            raise ValueError("Prices must be finite and nonnegative")
    return data


def calculate_cost(calls, prices):
    prices = validate_prices(prices)
    total, missing = 0.0, 0
    for call in calls:
        rates = prices["models"].get(call["model"])
        if rates is None or call.get("input_tokens") is None or call.get("output_tokens") is None:
            missing += 1
            continue
        total += (call["input_tokens"] * rates["input_per_million"] + call["output_tokens"] * rates["output_per_million"]) / 1_000_000
    return {"currency": prices["currency"], "price_as_of": prices["as_of"], "price_source": prices["source"],
            "observed_cost": total, "complete_cost": total if calls and missing == 0 else None,
            "calls_unpriced": missing,
            "note": "Simple token estimate; excludes tiering, cached-token discounts, taxes and unreported usage."}
