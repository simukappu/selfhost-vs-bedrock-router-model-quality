"""Shared scoring harness for the front-stage extraction benchmark.

The "front stage" is one LLM call that reads a shopping query (plus a little page
context) and emits a single JSON object: an intent label and the list of tools to
call. Every request in the agent crosses it, so it has to be small, cheap and
fast. This harness scores how well a given model does that job, on a fixed
dataset, with one grading rule, so self-hosted and managed models are judged on
identical terms.

This file is deliberately self-contained (standard library only): loading the
dataset, building the user message, normalizing the reply, and grading. The
per-backend scripts (bench_bedrock.py / bench_luna.py / bench_vllm.py) import it
so the grade is the same across backends.

Scope: quality and latency only. Cost per token is a separate axis and lives in
the companion repository, selfhost-vs-bedrock-token-economics.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "scenarios.json"
SYSTEM_PROMPT_PATH = ROOT / "dataset" / "system_prompt.txt"
SCHEMA_PATH = ROOT / "dataset" / "schema.json"

SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text()
SCHEMA = json.loads(SCHEMA_PATH.read_text())
INTENTS = SCHEMA["properties"]["intent"]["enum"]
TOOL_NAMES = SCHEMA["properties"]["tools"]["items"]["properties"]["name"]["enum"]


def load_scenarios() -> list[dict]:
    """Flatten dataset/scenarios.json into rows the grader can use.

    Each row: scenario_id, source, input, context, expected_intent,
    expected_tools (names that MUST be called), optional_tools (names allowed but
    not required).
    """
    rows = []
    for s in json.loads(DATASET.read_text()):
        exp = s.get("expected") or {}
        rows.append({
            "scenario_id": s["scenario_id"],
            "source": s.get("source", ""),
            "input": s["input"],
            "context": s.get("context", ""),
            "expected_intent": exp.get("intent"),
            "expected_tools": [t["name"] for t in (exp.get("tools") or [])],
            "optional_tools": [t["name"] for t in (exp.get("optional_tools") or [])],
        })
    return rows


def build_user_msg(row: dict) -> str:
    """The user turn: page context (if any) followed by the query."""
    ctx = row.get("context") or ""
    inp = row["input"]
    return f"{ctx}\n\nクエリ: {inp}" if ctx else f"クエリ: {inp}"


def strip_fence(text: str) -> str:
    """Remove a Markdown code fence around the JSON payload.

    The grader parses the reply as JSON. Some models (Claude Haiku 4.5, Nova 2
    Lite) wrap otherwise-correct JSON in a ```json fence, which would score every
    one of their answers as a parse failure. Stripping the fence removes an output
    convention difference, not a capability difference, and is applied uniformly to
    every backend so the comparison stays symmetric. A model served with guided
    JSON decoding (the vLLM path) never emits a fence, so this is a no-op there.
    """
    t = text.strip()
    if not t.startswith("```"):
        return t
    body = t.split("\n", 1)[1] if "\n" in t else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: -len("```")]
    return body.strip()


def grade(text: str, row: dict) -> dict:
    """Grade one reply against one scenario.

    - json_ok        the reply parsed as JSON
    - schema_ok       intent is one of the 7, every tool name is one of the 13,
                      and tools is a list
    - intent_correct  intent equals the expected label
    - tools_match     set match: every required tool was called, and any extra
                      tool is on the optional list (order and args are not scored
                      here; args accuracy is reported separately by backends that
                      can measure it)
    """
    expected_intent = row["expected_intent"]
    expected_tools = set(row["expected_tools"])
    optional_tools = set(row["optional_tools"])
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"json_ok": False, "schema_ok": False, "intent_correct": False,
                "tools_match": False, "intent": None, "tool_names": []}

    intent = obj.get("intent")
    tools_raw = obj.get("tools")
    tool_names = {t["name"] for t in tools_raw
                  if isinstance(t, dict) and t.get("name") in TOOL_NAMES} \
        if isinstance(tools_raw, list) else set()

    intent_in_enum = intent in INTENTS
    tools_in_enum = all(n in TOOL_NAMES for n in tool_names)
    schema_ok = intent_in_enum and tools_in_enum and isinstance(tools_raw, list)

    missing = expected_tools - tool_names
    extra_disallowed = (tool_names - expected_tools) - optional_tools
    tools_match = (not missing) and (not extra_disallowed)

    return {
        "json_ok": True,
        "schema_ok": schema_ok,
        "intent": intent,
        "intent_correct": intent == expected_intent,
        "tool_names": sorted(tool_names),
        "tools_match": tools_match,
        "missing_tools": sorted(missing),
        "extra_disallowed": sorted(extra_disallowed),
    }


def summarize(records: list[dict], model_label: str, extra: dict | None = None) -> dict:
    """Aggregate per-scenario records into one result object."""
    import statistics
    ok = [r for r in records if "latency_ms" in r]
    from collections import defaultdict
    per_intent = defaultdict(lambda: {"n": 0, "intent_correct": 0, "tools_match": 0})
    for r in records:
        b = per_intent[r["source"]]
        b["n"] += 1
        b["intent_correct"] += bool(r.get("intent_correct"))
        b["tools_match"] += bool(r.get("tools_match"))
    lat = sorted(r["latency_ms"] for r in ok)
    out = {
        "model": model_label,
        "n": len(records),
        "errors": len(records) - len(ok),
        "json_ok": sum(bool(r.get("json_ok")) for r in records),
        "schema_ok": sum(bool(r.get("schema_ok")) for r in records),
        "intent_correct": sum(bool(r.get("intent_correct")) for r in records),
        "tools_match": sum(bool(r.get("tools_match")) for r in records),
    }
    if ok and "in_tok" in ok[0]:
        out["in_tok_median"] = statistics.median(r["in_tok"] for r in ok)
        out["out_tok_median"] = statistics.median(r["out_tok"] for r in ok)
    if lat:
        out["latency_p50"] = statistics.median(lat)
        out["latency_p90"] = lat[int(len(lat) * 0.9)]
    out["per_intent"] = {k: dict(v) for k, v in per_intent.items()}
    if extra:
        out.update(extra)
    out["records"] = records
    return out


def write_result(path: Path, meta: dict, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**meta, "n_scenarios": len(load_scenarios()), "results": results}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
