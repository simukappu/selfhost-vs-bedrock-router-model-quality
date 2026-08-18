"""Run the front-stage extraction task on managed Bedrock models via Converse.

Models: Nova Micro, Nova 2 Lite, Claude Haiku 4.5 (inference profile IDs). Each is
prompt-only (no guided decoding), so a formatting slip shows up as a parse
failure; strip_fence() removes the one convention difference (a ```json wrapper)
that is not a capability difference.

Records intent accuracy, tools set_match, per-request input/output tokens, and
latency. Writes results/managed.json.

    pip install boto3
    python bench/bench_bedrock.py [--region us-east-1] [--models nova-micro,haiku-4.5]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3

import harness as H

# label -> Bedrock inference profile id
MODELS = {
    "nova-micro": "us.amazon.nova-micro-v1:0",
    "nova-2-lite": "us.amazon.nova-2-lite-v1:0",
    "haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


def one(client, model_id: str, row: dict) -> dict:
    t0 = time.perf_counter()
    try:
        resp = client.converse(
            modelId=model_id,
            system=[{"text": H.SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": H.build_user_msg(row)}]}],
            inferenceConfig={"maxTokens": 600, "temperature": 0.0},
        )
    except Exception as e:  # noqa: BLE001
        return {"scenario_id": row["scenario_id"], "source": row["source"],
                "error": type(e).__name__, "json_ok": False, "schema_ok": False,
                "intent_correct": False, "tools_match": False}
    lat = (time.perf_counter() - t0) * 1000
    text = next((b["text"] for b in resp["output"]["message"]["content"] if "text" in b), "")
    g = H.grade(H.strip_fence(text), row)
    u = resp["usage"]
    g |= {"scenario_id": row["scenario_id"], "source": row["source"],
          "latency_ms": round(lat, 1), "in_tok": u["inputTokens"], "out_tok": u["outputTokens"]}
    return g


def run_model(client, model_id: str, rows: list[dict], workers: int) -> dict:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda r: one(client, model_id, r), rows))
    return H.summarize(records, model_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = H.load_scenarios()
    client = boto3.client("bedrock-runtime", region_name=args.region)
    print(f"scenarios: {len(rows)}   region: {args.region}   "
          f"prompt: {len(H.SYSTEM_PROMPT):,} chars\n")

    results = {}
    for label in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"  running {label} ...")
        r = run_model(client, MODELS[label], rows, args.workers)
        results[label] = r
        print(f"    intent {r['intent_correct']}/{r['n']}  tools {r['tools_match']}/{r['n']}  "
              f"in {r.get('in_tok_median',0):.0f} out {r.get('out_tok_median',0):.0f}  "
              f"p50 {r.get('latency_p50',0):.0f}ms  errors {r['errors']}\n")

    H.write_result(
        Path(H.ROOT) / "results" / "managed.json",
        {"backend": "bedrock-runtime (Converse)", "region": args.region,
         "system_prompt_chars": len(H.SYSTEM_PROMPT)},
        results,
    )
    print("wrote results/managed.json")


if __name__ == "__main__":
    main()
