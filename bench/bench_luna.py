"""Run the front-stage extraction task on GPT-5.6 Luna.

Luna is not on the bedrock-runtime endpoint, so Converse cannot reach it and
ListFoundationModels does not list it. It is served on bedrock-mantle through the
OpenAI Responses API:

  endpoint : https://bedrock-mantle.{region}.api.aws/openai/v1
  model id : openai.gpt-5.6-luna
  auth     : a Bedrock API key as a bearer token, not SigV4. A long-term key can
             be made in the Bedrock console; this script instead derives a
             short-term token from the caller's existing SigV4 credentials, so no
             key has to be created or stored.
  regions  : us-east-1, us-east-2, us-west-2 (In-Region only)

Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-luna.html

Same dataset, same system prompt, same grade() as bench_bedrock.py, so the row
lines up in the same table. Writes results/luna.json.

    pip install openai aws-bedrock-token-generator
    python bench/bench_luna.py [--region us-east-1]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aws_bedrock_token_generator import provide_token
from openai import OpenAI

import harness as H

MODEL = "openai.gpt-5.6-luna"


def one(client: OpenAI, row: dict) -> dict:
    t0 = time.perf_counter()
    try:
        r = client.responses.create(
            model=MODEL,
            instructions=H.SYSTEM_PROMPT,
            input=H.build_user_msg(row),
            max_output_tokens=600,
        )
    except Exception as e:  # noqa: BLE001
        return {"scenario_id": row["scenario_id"], "source": row["source"],
                "error": type(e).__name__, "json_ok": False, "schema_ok": False,
                "intent_correct": False, "tools_match": False}
    lat = (time.perf_counter() - t0) * 1000
    g = H.grade(H.strip_fence(r.output_text or ""), row)
    u = r.usage
    g |= {"scenario_id": row["scenario_id"], "source": row["source"],
          "latency_ms": round(lat, 1), "in_tok": u.input_tokens, "out_tok": u.output_tokens}
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = H.load_scenarios()
    client = OpenAI(
        api_key=provide_token(region=args.region),
        base_url=f"https://bedrock-mantle.{args.region}.api.aws/openai/v1",
    )
    print(f"scenarios: {len(rows)}   model: {MODEL}   region: {args.region}\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda r: one(client, r), rows))
    r = H.summarize(records, MODEL)
    print(f"  intent {r['intent_correct']}/{r['n']}  tools {r['tools_match']}/{r['n']}  "
          f"in {r.get('in_tok_median',0):.0f} out {r.get('out_tok_median',0):.0f}  "
          f"p50 {r.get('latency_p50',0):.0f}ms  errors {r['errors']}")

    H.write_result(
        Path(H.ROOT) / "results" / "luna.json",
        {"backend": "bedrock-mantle (Responses API)", "region": args.region,
         "endpoint": f"https://bedrock-mantle.{args.region}.api.aws/openai/v1",
         "system_prompt_chars": len(H.SYSTEM_PROMPT)},
        {"gpt-5.6-luna": r},
    )
    print("wrote results/luna.json")


if __name__ == "__main__":
    main()
