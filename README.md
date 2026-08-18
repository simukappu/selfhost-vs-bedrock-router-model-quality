# Self-hosted vs Bedrock — Front-Stage Quality

Measured quality and latency of a self-hosted Qwen3.5-4B against managed Bedrock models (Nova Micro, Nova 2 Lite, Claude Haiku 4.5, GPT-5.6 Luna) on an agent's **one-shot front-stage extraction** task, over one shared 200-scenario dataset and one grading rule.

Companion to [selfhost-vs-bedrock-token-economics](https://github.com/oktomoya/selfhost-vs-bedrock-token-economics), which measures the **cost** of moving tokens. This repo measures whether a small model is **accurate and fast enough** for the role. The two axes are separate on purpose: a model can be the cheapest and still not do the job, so cost is not mixed in here.

## The task

A shopping agent puts one small LLM call in front of every request. It reads the user's query plus a little page context and emits a single JSON object: an intent label (7 classes) and the list of tools to call (13 names), with their arguments. Because every request crosses it, this "front stage" has to be small, cheap, and fast. The question here is which model does that job well.

```
input:  {"page_type": "PDP", "viewing_product_id": "..."}  +  "この靴のサイズ感は？"
output: {"intent": "UC2_PRODUCT_DETAIL",
         "tools": [{"name": "get_product_details", "args": {"product_id": "..."}}]}
```

The full contract (7 intents, 13 tools, the JSON schema) is in `dataset/`.

## Method

- **Dataset**: 200 scenarios in `dataset/scenarios.json`, each with an expected intent and an expected tool set (plus an optional-tool allowance). Queries are short shopping utterances; contexts are page metadata (`page_type`, `viewing_product_id`). No personal data.
- **Prompt**: one system prompt for every model, `dataset/system_prompt.txt`.
- **Grading** (`bench/harness.py`, identical across backends):
  - `intent_correct` — intent equals the expected label
  - `tools_match` — set match: every required tool is called, and any extra tool is on the optional list (order and argument values are not scored here)
  - `arg_accuracy` — a coarse key-presence measure, reported only for the self-hosted run that exposed arguments
- **Backends**:
  - Managed models over `bedrock-runtime` Converse, prompt-only (`bench/bench_bedrock.py`)
  - GPT-5.6 Luna over `bedrock-mantle` Responses API, prompt-only (`bench/bench_luna.py`)
  - Qwen3.5-4B on vLLM with guided JSON decoding (`bench/bench_vllm.py`)

## Results

200 scenarios, one shared system prompt (1,784 chars). Managed and Luna runs are from this repo's scripts (us-east-1); the Qwen row is carried over from the original verification run (see `results/qwen_selfhost.json`).

| Model | Serving | intent | tools (set match) | in tok | out tok | latency p50 |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | self-hosted, vLLM (guided JSON) | 181/200 = 90.5% | 171/200 = 85.5% | 706 | 62 | 2,233 ms |
| Claude Haiku 4.5 | Bedrock, prompt-only | 185/200 = 92.5% | 178/200 = 89.0% | 995 | 73 | 1,385 ms |
| GPT-5.6 Luna | Bedrock, prompt-only | 186/200 = 93.0% | 179/200 = 89.5% | 745 | 78 | 1,009 ms |
| Nova 2 Lite | Bedrock, prompt-only | 172/200 = 86.0% | 148/200 = 74.0% | 732 | 56 | 680 ms |
| Nova Micro | Bedrock, prompt-only | 169/200 = 84.5% | 143/200 = 71.5% | 853 | 50 | 559 ms |

What the numbers say, and only that:

- **The self-hosted small model clears the cheapest managed models on quality.** Qwen3.5-4B lands intent +4.5 to +6.0 points and tools set-match +11.5 to +14.0 points above Nova Micro and Nova 2 Lite. It sits just under Haiku 4.5 and Luna.
- **Haiku 4.5 and Luna are effectively tied at the top**, within a point or two of each other. Do not read Luna as "most accurate" from one run (see variance below).
- **Latency and quality point in opposite directions.** Nova Micro is fastest and least accurate; the more accurate models are slower. Whether the front stage can afford a given model's latency depends on the system's end-to-end budget, not on quality alone.

## Read this before quoting the numbers

- **Two backends have an inherent edge that the numbers do not separate out.** Qwen runs with guided JSON decoding, so its output is schema-valid by construction; the managed models are prompt-only and can format-slip. That helps Qwen on `tools_match`. Argument-value correctness is not graded (`arg_accuracy` is only key-presence, and only for Qwen). On the one run where arguments were exposed, Qwen's `arg_accuracy` was 0.49, a reminder that getting the tool name right is not the same as getting the call right.
- **This is one run per model.** Intent moves a couple of points run to run at temperature 0. Latency moves much more: Luna measured p50 1,009 ms here but 3,504-6,370 ms on earlier runs, so treat its latency as "variable and often the slowest of this set," not as a fixed number.
- **Latency is not a serving benchmark.** The Bedrock figures are managed endpoints at one point in time; the Qwen figure is one self-hosted GPU at another time. They are not comparable as throughput.
- **Quality here is task accuracy on this dataset, not general model quality.** A 4B model matching Haiku on one-shot extraction says nothing about either model on open-ended generation.
- **Cost is deliberately absent.** Whether self-hosting Qwen is cheaper than paying per token is answered in the companion repo, not here.

## Reproducing

```bash
pip install boto3 openai aws-bedrock-token-generator requests

python bench/bench_bedrock.py --region us-east-1      # -> results/managed.json
python bench/bench_luna.py    --region us-east-1      # -> results/luna.json

# self-hosted side needs a vLLM server (see the companion repo's RUNBOOK for GPU setup):
#   vllm serve Qwen/Qwen3.5-4B --language-model-only --reasoning-parser qwen3 --enable-prefix-caching
python bench/bench_vllm.py --endpoint http://localhost:8000   # -> results/qwen_selfhost_rerun.json
```

Bedrock access uses your AWS credentials. Luna needs the `bedrock-mantle` endpoint; `bench/bench_luna.py` mints a short-term bearer token from those credentials, so no separate API key is required.

## Repository layout

```
dataset/
  scenarios.json       200 scenarios (id, input, context, expected intent + tools)
  system_prompt.txt    the one system prompt every model is given
  schema.json          extraction JSON schema (7 intents, 13 tool names)
bench/
  harness.py           load / build message / normalize / grade (stdlib only)
  bench_bedrock.py     Nova Micro / Nova 2 Lite / Haiku 4.5, over Converse
  bench_luna.py        GPT-5.6 Luna, over bedrock-mantle Responses API
  bench_vllm.py        Qwen3.5-4B, over a vLLM OpenAI-compatible endpoint
results/
  managed.json         Nova Micro / Nova 2 Lite / Haiku 4.5
  luna.json            GPT-5.6 Luna
  qwen_selfhost.json   Qwen3.5-4B (from the original verification run)
```

## Attribution

The task, the dataset, the prompt, and the tool schema come from Tomoya Okuno's ([oktomoya](https://github.com/oktomoya)) Architecture Dojo 2026 verification of an AI shopping assistant at AWS Summit Japan (the [session deck](https://pages.awscloud.com/rs/112-TZM-766/images/R01-03_0626_ARC446_v2.pdf) is in Japanese). This repository is the front-stage quality and latency slice of that work, packaged to be reproducible. The measurement scripts and this write-up are by Shota Yamazaki ([simukappu](https://github.com/simukappu)). Any errors in the repackaging are mine.

## Dependencies

| Component | License | How it is used |
|---|---|---|
| [boto3](https://github.com/boto/boto3) | Apache 2.0 | Bedrock Converse calls (managed models) |
| [openai](https://github.com/openai/openai-python) | Apache 2.0 | Responses API calls to Luna on bedrock-mantle |
| [aws-bedrock-token-generator](https://pypi.org/project/aws-bedrock-token-generator/) | Apache 2.0 | short-term Bedrock bearer token from SigV4 credentials |
| requests | Apache 2.0 | vLLM OpenAI-compatible calls (self-hosted) |
| [vLLM](https://github.com/vllm-project/vllm) | Apache 2.0 | serves Qwen3.5-4B (self-hosted side; not vendored here) |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | see the model card | the self-hosted model |

This repository's own code is covered by [LICENSE](LICENSE).
