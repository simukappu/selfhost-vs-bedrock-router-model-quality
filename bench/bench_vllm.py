"""Run the front-stage extraction task on a self-hosted Qwen3.5-4B (vLLM).

Talks to a vLLM server exposing the OpenAI-compatible API, with guided JSON
decoding against dataset/schema.json, so the reply is always schema-valid. This
is the self-hosted side of the comparison. It records the same quality fields as
the managed backends, plus arg_accuracy when the reply carries tool arguments.

Start the server (see the companion repo's RUNBOOK for the GPU setup):

    vllm serve Qwen/Qwen3.5-4B \\
      --language-model-only --reasoning-parser qwen3 --enable-prefix-caching

Then:

    pip install requests
    python bench/bench_vllm.py --endpoint http://localhost:8000

The committed results/qwen_selfhost.json was produced by the original verification
run (see its `provenance` field), not by this script; this script reproduces that
measurement against any vLLM endpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

import harness as H

MODEL = "Qwen/Qwen3.5-4B"


def arg_accuracy(obj: dict, row: dict) -> tuple[int, int]:
    """Fraction of expected tool-arg keys present, across required tools.

    Coarse on purpose: it checks that the model produced argument keys for the
    tools it was supposed to call, not that the values are right. Matches the
    original run's arg_accuracy definition.
    """
    got = {t["name"]: (t.get("args") or {}) for t in obj.get("tools", [])
           if isinstance(t, dict) and "name" in t}
    matched = total = 0
    for name in row["expected_tools"]:
        total += 1
        if name in got and got[name]:
            matched += 1
    return matched, total


def one(endpoint: str, row: dict) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": H.SYSTEM_PROMPT},
                     {"role": "user", "content": H.build_user_msg(row)}],
        "max_tokens": 600, "temperature": 0.0,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "extraction", "schema": H.SCHEMA,
                                            "strict": True}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{endpoint}/v1/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return {"scenario_id": row["scenario_id"], "source": row["source"],
                "error": type(e).__name__, "json_ok": False, "schema_ok": False,
                "intent_correct": False, "tools_match": False}
    lat = (time.perf_counter() - t0) * 1000
    text = data["choices"][0]["message"]["content"].strip()
    g = H.grade(H.strip_fence(text), row)
    usage = data.get("usage") or {}
    try:
        m, t = arg_accuracy(json.loads(text), row)
    except (json.JSONDecodeError, TypeError):
        m, t = 0, 0
    g |= {"scenario_id": row["scenario_id"], "source": row["source"],
          "latency_ms": round(lat, 1),
          "in_tok": usage.get("prompt_tokens"), "out_tok": usage.get("completion_tokens"),
          "arg_matched": m, "arg_total": t}
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000")
    args = ap.parse_args()

    rows = H.load_scenarios()
    print(f"scenarios: {len(rows)}   model: {MODEL}   endpoint: {args.endpoint}")
    records = [one(args.endpoint, r) for r in rows]  # serial: one GPU, no client-side concurrency
    r = H.summarize(records, MODEL)
    am = sum(x.get("arg_matched", 0) for x in records if "latency_ms" in x)
    at = sum(x.get("arg_total", 0) for x in records if "latency_ms" in x)
    if at:
        r["arg_accuracy"] = round(am / at, 4)
    print(f"  intent {r['intent_correct']}/{r['n']}  tools {r['tools_match']}/{r['n']}  "
          f"arg_acc {r.get('arg_accuracy','-')}  p50 {r.get('latency_p50',0):.0f}ms")

    H.write_result(
        Path(H.ROOT) / "results" / "qwen_selfhost_rerun.json",
        {"backend": "vllm (OpenAI-compatible, guided JSON)", "endpoint": args.endpoint,
         "system_prompt_chars": len(H.SYSTEM_PROMPT)},
        {"qwen3.5-4b": r},
    )
    print("wrote results/qwen_selfhost_rerun.json")


if __name__ == "__main__":
    main()
