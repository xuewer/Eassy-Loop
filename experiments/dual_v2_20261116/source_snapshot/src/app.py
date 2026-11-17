"""EssayLoop: evaluate, revise, save complete results. No calls at import."""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph
from src.artifacts import new_run_id, provenance, save_result, summary
from src.client import QwenClient, TextModel
from src.config import PROJECT_ROOT, Settings
from src.nodes import EvaluationState, Nodes, RUBRIC_VERSION
from src.schemas import DIMENSIONS, EvaluationInput
from src.telemetry import ObservedModel, Recorder, token_summary


def route_decision(state: EvaluationState) -> Literal["stop", "continue"]:
    return "stop" if state.get("exit_reason") else "continue"


def build_app(generator: TextModel, evaluator: TextModel, settings: Settings | None = None,
              mode: Literal["evaluate", "optimize"] = "optimize"):
    if mode not in ("evaluate", "optimize"):
        raise ValueError("mode must be evaluate or optimize")
    settings = settings or Settings()
    nodes = Nodes(generator, evaluator, settings)
    graph = StateGraph(EvaluationState)
    graph.add_node("generator", nodes.generate)
    graph.add_node("dispatch", lambda state: {})
    for dimension in DIMENSIONS:
        def evaluate(state, d=dimension):
            return nodes.evaluate(state, d)
        graph.add_node(dimension, evaluate)
    graph.add_node("aggregate", nodes.aggregate)
    if mode == "optimize":
        graph.add_edge(START, "generator")
        graph.add_conditional_edges("generator", route_decision, {"stop": END, "continue": "dispatch"})
    else:
        graph.add_edge(START, "dispatch")
    if settings.execution == "serial":
        graph.add_edge("dispatch", DIMENSIONS[0])
        for previous, following in zip(DIMENSIONS, DIMENSIONS[1:]):
            graph.add_edge(previous, following)
        graph.add_edge(DIMENSIONS[-1], "aggregate")
    else:
        for dimension in DIMENSIONS:
            graph.add_edge("dispatch", dimension)
        graph.add_edge(list(DIMENSIONS), "aggregate")
    if mode == "optimize":
        graph.add_conditional_edges("aggregate", route_decision, {"stop": END, "continue": "generator"})
    else:
        graph.add_edge("aggregate", END)
    return graph.compile()


def run(topic: str, *, essay: str | None = None, mode: Literal["evaluate", "optimize"] = "optimize",
        settings: Settings | None = None, generator: TextModel | None = None,
        evaluator: TextModel | None = None, recorder: Recorder | None = None) -> dict:
    settings = settings or Settings()
    request = EvaluationInput(topic=topic, essay=essay)
    if mode not in ("evaluate", "optimize"):
        raise ValueError("Unknown mode")
    if mode == "evaluate" and request.essay is None:
        raise ValueError("evaluate mode requires an essay")
    clients, recorder = [], recorder if recorder is not None else Recorder()
    started = time.perf_counter()
    try:
        if generator is None and mode == "optimize":
            generator = QwenClient(settings)
            clients.append(generator)
        if evaluator is None:
            evaluator = QwenClient(settings.model_copy(update={"model": settings.evaluator_model or settings.model,
                                                                "temperature": settings.evaluator_temperature}))
            clients.append(evaluator)
        generator = generator if generator is not None else evaluator
        graph = build_app(ObservedModel(generator, "generator", settings.model, recorder),
                          ObservedModel(evaluator, "evaluator", settings.evaluator_model or settings.model, recorder), settings, mode)
        initial = {"topic": request.topic, "current_round": 1 if mode == "evaluate" else 0,
                   "history": [], "generation_history": [], "feedback": "", "exit_reason": None,
                   "rewrites_completed": 0}
        if request.essay is not None:
            initial["essay"] = request.essay
        result = graph.invoke(initial, config={"max_concurrency": 4,
                                               "recursion_limit": 8 * settings.max_rounds + 12})
        calls = recorder.snapshot()
        elapsed = time.perf_counter() - started
        telemetry = {**token_summary(calls), "elapsed_seconds": elapsed}
        telemetry["parse_failures"] = sum(a["status"] == "parse_error"
            for h in result.get("history", []) for d in h["results"] for a in d["attempts"])
        return {**result, "mode": mode, "settings": settings.model_dump(), "run_id": new_run_id(),
                "evidence_type": "injected_models_unvalidated" if generator not in clients or evaluator not in clients else "live_api_run_unvalidated_quality",
                "rubric_version": RUBRIC_VERSION, "calls": calls, "telemetry": telemetry, "provenance": provenance()}
    finally:
        for client in clients:
            client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EssayLoop: multi-dimensional text evaluation and feedback revision")
    parser.add_argument("--demo", action="store_true", help="Offline scripted wiring check, no model or API key")
    parser.add_argument("--topic")
    parser.add_argument("--mode", choices=["evaluate", "optimize"], default="optimize")
    parser.add_argument("--essay-file", type=Path)
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--evaluator-model")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--evaluator-temperature", type=float, default=0)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--min-dimension-score", type=float)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--execution", choices=["parallel", "serial"], default="parallel")
    parser.add_argument("--feedback-mode", choices=["all", "relevance", "none"], default="all")
    parser.add_argument("--stop-policy", choices=["threshold", "fixed"], default="threshold")
    parser.add_argument("--ignore-env-proxy", action="store_true", help="Disable HTTPX environment proxies/custom CA paths; TLS stays enabled")
    parser.add_argument("--output-dir", type=Path, help="Fresh directory; existing paths are never overwritten")
    parser.add_argument("--json", action="store_true", help="Also print full JSON instead of summary")
    args = parser.parse_args(argv)
    if not args.topic and not args.demo:
        parser.error("--topic is required unless --demo is selected")
    output = args.output_dir or PROJECT_ROOT / "runs" / new_run_id()
    try:
        if output.exists():
            raise ValueError("Output directory already exists; choose a new path")
        settings = Settings(model=args.model, evaluator_model=args.evaluator_model, temperature=args.temperature,
                            evaluator_temperature=args.evaluator_temperature, threshold=args.threshold,
                            min_dimension_score=args.min_dimension_score, max_rounds=args.max_rounds,
                            max_attempts=args.max_attempts, timeout_seconds=args.timeout, max_output_tokens=args.max_output_tokens,
                            execution=args.execution, feedback_mode=args.feedback_mode, stop_policy=args.stop_policy,
                            trust_env=not args.ignore_env_proxy)
        essay = args.essay_file.read_text(encoding="utf-8-sig") if args.essay_file else None
        injected = {}
        if args.demo:
            from src.demo import DemoEvaluator, DemoGenerator
            injected = {"generator": DemoGenerator(), "evaluator": DemoEvaluator()}
        result = run(args.topic or "AI and future employment", essay=essay, mode=args.mode, settings=settings, **injected)
        result["evidence_type"] = "scripted_demo_not_experiment" if args.demo else "live_api_run_unvalidated_quality"
        path = save_result(result, output)
    except (ValueError, OSError):
        print("Configuration/input/output error; check arguments, output path and DASHSCOPE_API_KEY.", file=sys.stderr)
        return 2
    print(json.dumps(result if args.json else summary(result), ensure_ascii=False, indent=2))
    print(f"Complete result saved: {path}", file=sys.stderr)
    return 1 if result.get("exit_reason") in ("generation_failed", "evaluation_failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
