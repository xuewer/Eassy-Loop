"""Offline agreement statistics. Undefined correlations remain null."""
import math
import random
import statistics
from collections import defaultdict
from src.schemas import DIMENSIONS, WEIGHTS


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            result[order[k]] = (i + j - 1) / 2 + 1
        i = j
    return result


def correlation(x, y):
    if len(x) < 2:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else None


def agreement(pairs):
    if not pairs:
        return {"n": 0, "mae": None, "spearman": None}
    human, model = zip(*pairs)
    return {"n": len(pairs), "mae": statistics.mean(abs(h - m) for h, m in pairs),
            "spearman": correlation(ranks(human), ranks(model))}


def bootstrap_mean_interval(values, seed=0, repetitions=2000):
    """Resample independent SAMPLE-level differences, not repeated calls."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return [means[int(0.025 * repetitions)], means[min(repetitions - 1, int(0.975 * repetitions))]]


def reliability_report(jobs, samples, planned_repeats=None):
    references = {s["sample_id"]: s.get("human_scores") for s in samples}
    predictions = defaultdict(list)
    for entry in jobs:
        result = entry["result"]
        for r in result.get("history", [])[-1:]:
            for d in r["results"]:
                if d["status"] == "ok":
                    predictions[(entry["sample_id"], d["dimension"])].append(d["judgment"]["score"])
    output = {}
    for dim in DIMENSIONS:
        pairs, deviations = [], []
        repeats = {}
        for (sample_id, dimension), scores in predictions.items():
            if dimension != dim:
                continue
            repeats[sample_id] = len(scores)
            if len(scores) > 1:
                deviations.append(statistics.stdev(scores))
            ref = references.get(sample_id)
            if ref and dim in ref:
                pairs.append((ref[dim], statistics.mean(scores)))
        output[dim] = {**agreement(pairs), "mean_within_sample_sd": statistics.mean(deviations) if deviations else None,
                       "samples_with_repeats": len(deviations),
                       "samples_with_human_reference": sum(bool(s.get("human_scores")) for s in samples),
                       "successful_dimension_decisions": sum(repeats.values()),
                       "planned_dimension_decisions": len(samples) * planned_repeats if planned_repeats else None,
                       "completed_job_count": len(jobs),
                       "failed_or_missing_dimension_decisions": len(samples) * planned_repeats - sum(repeats.values()) if planned_repeats else None, "successful_repeats_per_sample": repeats}
    per_sample = []
    for sample in samples:
        entries = [e for e in jobs if e["sample_id"] == sample["sample_id"]]
        totals = [e["result"]["final_score"] for e in entries if e["result"].get("final_score") is not None]
        dims = {}
        for dim in DIMENSIONS:
            values = predictions.get((sample["sample_id"], dim), [])
            dims[dim] = {"n": len(values), "scores": values,
                         "mean": statistics.mean(values) if values else None,
                         "sample_sd": statistics.stdev(values) if len(values)>1 else None,
                         "min": min(values) if values else None, "max": max(values) if values else None}
        per_sample.append({"sample_id": sample["sample_id"], "planned_repeats": planned_repeats,
            "completed_repeats": len(entries), "successful_repeats": len(totals),
            "failed_repeats": len(entries)-len(totals),
            "missing_repeats": planned_repeats-len(entries) if planned_repeats else None,
            "failure_rate_completed": (len(entries)-len(totals))/len(entries) if entries else None,
            "dimensions": dims, "total_score": {"scores": totals,
                "mean": statistics.mean(totals) if totals else None,
                "sample_sd": statistics.stdev(totals) if len(totals)>1 else None,
                "min": min(totals) if totals else None,
                "max": max(totals) if totals else None,
                "range": max(totals)-min(totals) if totals else None}})
    failures = sum(e["result"].get("final_score") is None for e in jobs)
    return {"dimensions": output, "per_sample": per_sample,
            "completed_jobs": len(jobs), "failed_jobs": failures,
            "failure_rate_completed": failures/len(jobs) if jobs else None,
            "reference_policy": "One human score per sample/dimension; model repetitions averaged before agreement.",
            "warning": "No reference labels means no accuracy estimate. Stability does not imply correctness."}
