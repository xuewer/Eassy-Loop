"""Deterministic fake models for wiring checks, NEVER evaluation evidence."""
import json
from collections import Counter
import threading


class DemoGenerator:
    def invoke(self, prompt):
        data = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        suffix = " Training takes time and access differs across sectors." if data.get("essay") else ""
        return "AI changes work. Routine tasks can be automated, while workers may gain useful tools." + suffix


class DemoEvaluator:
    def __init__(self):
        self.calls, self.lock = Counter(), threading.Lock()

    def invoke(self, prompt):
        d = prompt.splitlines()[0].split(": ")[1]
        data = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        quote = data["essay"].split(". ", 1)[0].strip()
        if not quote:
            raise ValueError("Demo essay must contain a quoteable substring")
        with self.lock:
            round_id = self.calls[d]
            self.calls[d] += 1
        return json.dumps({"dimension": d, "score": 0.5 if round_id == 0 else 0.9,
                          "rationale": "Scripted plumbing check; this is not a model judgment.",
                          "evidence": [quote], "suggestions": [f"Add specific {d} analysis."]})
