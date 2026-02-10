"""
Speculative Decoding Throughput Benchmark  (Qwen3 thinking mode)
=================================================================
Compares maximum throughput of:
  1. Speculative decoding: Qwen3-14B (target) + Qwen3-1.7B (draft)
  2. Autoregressive baseline: Qwen3-14B alone

All prompts use the Qwen3 chat template with **thinking mode enabled**
(enable_thinking=True) so the model produces <think>…</think> reasoning
before its answer.

Two input-context regimes:
  - Short context:  MMLU (cais/mmlu) — multiple-choice knowledge questions,
                    naturally short prompts.
  - Long context:   LongBench v2 (THUDM/LongBench-v2) — full document
                    contexts (no truncation), filtered to fit max_model_len.

Throughput is measured across many concurrent requests to saturate the engine.
"""

import gc
import time
import argparse
import json
import os
import random
from datetime import datetime

import torch

# Enable per-step stat logging (must be set before importing vllm)
# This controls how often vLLM's built-in loggers print stats.
# Setting to a tiny value makes it log on virtually every scheduler step.
os.environ["VLLM_LOG_STATS_INTERVAL"] = "5"

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import prometheus_client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_MODEL = "Qwen/Qwen3-14B"
#DRAFT_MODEL = "Qwen/Qwen3-1.7B"
DRAFT_MODEL = "AngelSlim/Qwen3-14B_eagle3"

NUM_SPEC_TOKENS = 5          # tokens the draft model proposes per step
MAX_NEW_TOKENS = 4096        # tokens to generate per request (thinking tokens count)
NUM_REQUESTS = 64            # total requests per throughput run
TEMPERATURE = 0.1            # sampling temperature (0 = greedy, >0 = stochastic)
GPU_MEMORY_UTIL = 0.90       # fraction of GPU memory vLLM may use
TENSOR_PARALLEL_SIZE = 1     # adjust for multi-GPU setups
MAX_MODEL_LEN = 40_960       # max sequence length (input + output)
MIN_LONG_TOKENS = 16_000     # minimum prompt tokens for long-context bucket
MAX_LONG_TOKENS = 32_000     # maximum prompt tokens for long-context bucket

ENABLE_THINKING = False

# ---------------------------------------------------------------------------
# Dataset identifiers
# ---------------------------------------------------------------------------
MMLU_DATASET = "cais/mmlu"              # short-context knowledge QA
LONGBENCH_DATASET = "THUDM/LongBench-v2"  # long-context document QA


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_mmlu() -> list[dict]:
    """Load the MMLU dataset (all subjects, test split)."""
    ds = load_dataset(MMLU_DATASET, "all", split="test")
    return list(ds)


def load_longbench_v2() -> list[dict]:
    """Load the LongBench v2 dataset."""
    ds = load_dataset(LONGBENCH_DATASET, split="train")
    return list(ds)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_mmlu_prompt(row: dict) -> str:
    """Format an MMLU row as a multiple-choice QA prompt."""
    choices = row["choices"]
    labels = ["A", "B", "C", "D"]
    choice_lines = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    subject = row["subject"].replace("_", " ")
    return (
        f"The following is a multiple choice question about {subject}.\n\n"
        f"Question: {row['question']}\n"
        f"{choice_lines}\n"
        f"Answer:"
    )


def _format_longbench_prompt(row: dict) -> str:
    """Format a LongBench v2 row as a QA prompt with the full context."""
    return (
        f"{row['context']}\n\n"
        f"Question: {row['question']}\n"
        f"A. {row['choice_A']}\n"
        f"B. {row['choice_B']}\n"
        f"C. {row['choice_C']}\n"
        f"D. {row['choice_D']}\n"
        f"Answer:"
    )


def _apply_chat_template(tokenizer, prompt_text: str) -> str:
    """Wrap prompt text in the Qwen3 chat template with thinking mode enabled."""
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_short_prompts(
    dataset: list[dict],
    tokenizer,
    num_requests: int,
    seed: int = 42,
) -> list[str]:
    """
    Build `num_requests` short prompts from MMLU, formatted with the Qwen3
    chat template and thinking mode enabled.
    """
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    prompts = []
    for i in range(num_requests):
        row = dataset[indices[i % len(indices)]]
        text = _format_mmlu_prompt(row)
        prompts.append(_apply_chat_template(tokenizer, text))
    return prompts


def build_long_prompts(
    dataset: list[dict],
    tokenizer,
    num_requests: int,
    min_tokens: int,
    max_tokens: int,
    seed: int = 123,
) -> list[str]:
    """
    Build `num_requests` long prompts from LongBench v2 with full context
    (no truncation).  Only rows whose formatted+chat-templated prompt length
    falls in [min_tokens, max_tokens] are kept.  Prompts use Qwen3 thinking
    mode.
    """
    rng = random.Random(seed)

    # Pre-filter by word count to avoid tokenizing multi-million-word contexts.
    # Rough heuristic: 1 word ≈ 1.3 tokens.
    min_words_estimate = int(min_tokens / 1.5)   # generous lower bound
    max_words_estimate = int(max_tokens / 1.0)   # conservative upper bound
    candidates = [
        row for row in dataset
        if min_words_estimate <= len(row["context"].split()) <= max_words_estimate
    ]
    print(f"  Pre-filter: {len(candidates)}/{len(dataset)} rows with "
          f"~{min_words_estimate}–{max_words_estimate} words")

    # Full tokenization pass on surviving candidates
    valid_prompts: list[str] = []
    for row in candidates:
        text = _format_longbench_prompt(row)
        prompt = _apply_chat_template(tokenizer, text)
        n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if min_tokens <= n_tokens <= max_tokens:
            valid_prompts.append(prompt)

    if not valid_prompts:
        raise RuntimeError(
            f"No LongBench v2 entries have {min_tokens}–{max_tokens} tokens. "
            "Adjust --min-long-tokens / --max-long-tokens."
        )

    print(f"  After tokenization: {len(valid_prompts)} prompts in "
          f"{min_tokens}–{max_tokens} token range")

    rng.shuffle(valid_prompts)
    # Cycle if we need more prompts than available
    return [valid_prompts[i % len(valid_prompts)] for i in range(num_requests)]


def _read_spec_decode_counters() -> dict[str, float]:
    """Snapshot the current values of vLLM's spec-decode Prometheus counters."""
    counters: dict[str, float] = {
        "num_drafts": 0.0,
        "num_draft_tokens": 0.0,
        "num_accepted_tokens": 0.0,
    }
    per_pos: dict[int, float] = {}

    for metric in prometheus_client.REGISTRY.collect():
        if metric.name == "vllm:spec_decode_num_drafts":
            for sample in metric.samples:
                if sample.name.endswith("_total"):
                    counters["num_drafts"] += sample.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            for sample in metric.samples:
                if sample.name.endswith("_total"):
                    counters["num_draft_tokens"] += sample.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            for sample in metric.samples:
                if sample.name.endswith("_total"):
                    counters["num_accepted_tokens"] += sample.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            for sample in metric.samples:
                if sample.name.endswith("_total"):
                    pos = int(sample.labels.get("position", 0))
                    per_pos[pos] = per_pos.get(pos, 0.0) + sample.value

    if per_pos:
        counters["accepted_per_pos"] = [per_pos[k] for k in sorted(per_pos)]
    return counters


def _compute_spec_decode_stats(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, object] | None:
    """Compute spec-decode acceptance metrics from counter deltas."""
    d_drafts = after["num_drafts"] - before["num_drafts"]
    d_draft_tokens = after["num_draft_tokens"] - before["num_draft_tokens"]
    d_accepted = after["num_accepted_tokens"] - before["num_accepted_tokens"]

    if d_draft_tokens == 0 or d_drafts == 0:
        return None

    acceptance_rate = d_accepted / d_draft_tokens
    # Conventionally, mean acceptance length includes the bonus token
    mean_acceptance_len = 1 + (d_accepted / d_drafts)

    stats: dict[str, object] = {
        "num_drafts": int(d_drafts),
        "num_draft_tokens": int(d_draft_tokens),
        "num_accepted_tokens": int(d_accepted),
        "acceptance_rate": round(acceptance_rate * 100, 2),
        "mean_acceptance_length": round(mean_acceptance_len, 2),
    }

    # Per-position acceptance rates
    if "accepted_per_pos" in before and "accepted_per_pos" in after:
        before_pp = before["accepted_per_pos"]
        after_pp = after["accepted_per_pos"]
        if len(before_pp) == len(after_pp) and d_drafts > 0:
            per_pos_rates = [
                round((after_pp[i] - before_pp[i]) / d_drafts, 4)
                for i in range(len(after_pp))
            ]
            stats["per_position_acceptance_rate"] = per_pos_rates

    return stats


def measure_throughput(llm: LLM, prompts: list[str], sampling_params: SamplingParams):
    """
    Run generation on all prompts and return throughput metrics.
    Returns dict with total tokens, wall-clock time, and tokens/sec.
    """
    # Warm-up run (single short prompt) to ensure model is loaded & compiled
    llm.generate(prompts[:1], sampling_params)

    counters_before = _read_spec_decode_counters()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    counters_after = _read_spec_decode_counters()

    total_input_tokens = 0
    total_output_tokens = 0
    for output in outputs:
        total_input_tokens += len(output.prompt_token_ids)
        total_output_tokens += len(output.outputs[0].token_ids)

    result = {
        "num_requests": len(prompts),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "output_tokens_per_second": round(total_output_tokens / elapsed, 2),
        "total_tokens_per_second": round(
            (total_input_tokens + total_output_tokens) / elapsed, 2
        ),
    }

    spec_stats = _compute_spec_decode_stats(counters_before, counters_after)
    if spec_stats is not None:
        result["spec_decode_stats"] = spec_stats

    return result


def run_benchmark(args):
    """Run the full benchmark suite."""

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    # sampling_params = SamplingParams(
    #     temperature=args.temperature,
    #     max_tokens=args.max_new_tokens,
    # )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_tokens=args.max_new_tokens
    )
    # ----- Load datasets -----
    print("Loading MMLU dataset (short prompts) …")
    mmlu_data = load_mmlu()
    print(f"  Loaded {len(mmlu_data)} rows from {MMLU_DATASET}")

    print("Loading LongBench v2 dataset (long prompts) …")
    longbench_data = load_longbench_v2()
    print(f"  Loaded {len(longbench_data)} rows from {LONGBENCH_DATASET}")

    # ----- Build prompts (Qwen3 chat template, thinking mode) -----
    print("\nBuilding short prompts from MMLU …")
    short_prompts = build_short_prompts(
        mmlu_data, tokenizer, args.num_requests, seed=42,
    )

    print(f"Building long prompts from LongBench v2 "
          f"({args.min_long_tokens}–{args.max_long_tokens} tokens) …")
    long_prompts = build_long_prompts(
        longbench_data, tokenizer, args.num_requests,
        min_tokens=args.min_long_tokens,
        max_tokens=args.max_long_tokens,
        seed=123,
    )

    actual_short_len = len(tokenizer.encode(short_prompts[0], add_special_tokens=False))
    actual_long_len = len(tokenizer.encode(long_prompts[0], add_special_tokens=False))
    print(f"\n{'='*70}")
    print(f"Thinking mode   : ENABLED (enable_thinking=True)")
    print(f"Short prompts   : MMLU ({MMLU_DATASET}), ~{actual_short_len} tokens")
    print(f"Long prompts    : LongBench v2 ({LONGBENCH_DATASET}), "
          f"{args.min_long_tokens}–{args.max_long_tokens} tok range, "
          f"~{actual_long_len} tokens (first prompt)")
    print(f"Max model len   : {args.max_model_len}")
    print(f"Requests per run: {args.num_requests}")
    print(f"Max new tokens  : {args.max_new_tokens}")
    print(f"Temperature     : {args.temperature}")
    print(f"{'='*70}\n")

    results = {}

    # ===================================================================
    # 1. Autoregressive baseline  (Qwen3-14B only)
    # ===================================================================
    print("=" * 70)
    print("  AUTOREGRESSIVE BASELINE — Qwen3-14B  (thinking mode)")
    print("=" * 70)

    llm_baseline = LLM(
        model=TARGET_MODEL,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=args.eager,
        disable_log_stats=False,
    )

    print("\n>>> Short context  (MMLU) ...")
    res = measure_throughput(llm_baseline, short_prompts, sampling_params)
    results["baseline_short"] = res
    print(json.dumps(res, indent=2))

    print(f"\n>>> Long context  (LongBench v2, {args.min_long_tokens}–{args.max_long_tokens} tok) ...")
    res = measure_throughput(llm_baseline, long_prompts, sampling_params)
    results["baseline_long"] = res
    print(json.dumps(res, indent=2))

    # Free GPU memory before loading speculative model
    del llm_baseline
    gc.collect()
    torch.cuda.empty_cache()

    # ===================================================================
    # 2. Speculative decoding  (Qwen3-14B + Qwen3-1.7B draft)
    # ===================================================================
    print("\n" + "=" * 70)
    print(f"  SPECULATIVE DECODING — Qwen3-14B + {DRAFT_MODEL.split('/')[-1]} draft  (thinking mode)")
    print("=" * 70)

    if 'eagle' in DRAFT_MODEL.lower():
        speculative_config = {
            "method": "eagle3",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": args.num_spec_tokens,
        }
    else:
        speculative_config = {
            "method": "draft_model",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": args.num_spec_tokens,
        },

    llm_spec = LLM(
        model=TARGET_MODEL,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=args.eager,
        disable_log_stats=False,
        speculative_config=speculative_config,
    )

    print("\n>>> Short context  (MMLU) ...")
    res = measure_throughput(llm_spec, short_prompts, sampling_params)
    results["specdec_short"] = res
    print(json.dumps(res, indent=2))

    print(f"\n>>> Long context  (LongBench v2, {args.min_long_tokens}–{args.max_long_tokens} tok) ...")
    res = measure_throughput(llm_spec, long_prompts, sampling_params)
    results["specdec_long"] = res
    print(json.dumps(res, indent=2))

    del llm_spec

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    for label, (base_key, spec_key) in [
        ("Short context  (MMLU)", ("baseline_short", "specdec_short")),
        (f"Long context   (LongBench v2, {args.min_long_tokens}–{args.max_long_tokens} tok)",
         ("baseline_long", "specdec_long")),
    ]:
        b = results[base_key]
        s = results[spec_key]
        speedup = s["output_tokens_per_second"] / b["output_tokens_per_second"]
        print(f"\n  {label}:")
        print(f"    Autoregressive : {b['output_tokens_per_second']:>10.2f} tok/s  ({b['elapsed_seconds']:.1f}s)")
        print(f"    Spec decoding  : {s['output_tokens_per_second']:>10.2f} tok/s  ({s['elapsed_seconds']:.1f}s)")
        print(f"    Speedup        : {speedup:>10.2f}x")

        sd = s.get("spec_decode_stats")
        if sd:
            print(f"    --- Spec Decode Acceptance ---")
            print(f"    Avg acceptance rate   : {sd['acceptance_rate']:>8.2f}%")
            print(f"    Mean acceptance length: {sd['mean_acceptance_length']:>8.2f}  (includes bonus token)")
            print(f"    Total drafts          : {sd['num_drafts']:>8d}")
            print(f"    Drafted tokens        : {sd['num_draft_tokens']:>8d}")
            print(f"    Accepted tokens       : {sd['num_accepted_tokens']:>8d}")
            if "per_position_acceptance_rate" in sd:
                rates_str = ", ".join(f"{r:.3f}" for r in sd["per_position_acceptance_rate"])
                print(f"    Per-position rates    : [{rates_str}]")

    # ===================================================================
    # Save results to disk
    # ===================================================================
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/specdec_throughput_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "target_model": TARGET_MODEL,
                    "draft_model": DRAFT_MODEL,
                    "thinking_mode": True,
                    "short_prompt_dataset": MMLU_DATASET,
                    "long_prompt_dataset": LONGBENCH_DATASET,
                    "num_spec_tokens": args.num_spec_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "max_model_len": args.max_model_len,
                    "min_long_tokens": args.min_long_tokens,
                    "max_long_tokens": args.max_long_tokens,
                    "num_requests": args.num_requests,
                    "temperature": args.temperature,
                    "tensor_parallel_size": args.tp,
                    "short_prompt_tokens": actual_short_len,
                    "long_prompt_tokens": actual_long_len,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark speculative decoding throughput with Qwen3 thinking mode"
    )
    parser.add_argument(
        "--num-requests", type=int, default=NUM_REQUESTS,
        help=f"Number of concurrent requests per throughput run (default: {NUM_REQUESTS})"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
        help=f"Max tokens to generate per request; includes thinking tokens (default: {MAX_NEW_TOKENS})"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=MAX_MODEL_LEN,
        help=f"Max sequence length for the vLLM engine (default: {MAX_MODEL_LEN})"
    )
    parser.add_argument(
        "--min-long-tokens", type=int, default=MIN_LONG_TOKENS,
        help=f"Min prompt tokens for long-context bucket (default: {MIN_LONG_TOKENS})"
    )
    parser.add_argument(
        "--max-long-tokens", type=int, default=MAX_LONG_TOKENS,
        help=f"Max prompt tokens for long-context bucket (default: {MAX_LONG_TOKENS})"
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help=f"Sampling temperature; 0=greedy, >0=stochastic (default: {TEMPERATURE})"
    )
    parser.add_argument(
        "--num-spec-tokens", type=int, default=NUM_SPEC_TOKENS,
        help=f"Draft tokens per speculative step (default: {NUM_SPEC_TOKENS})"
    )
    parser.add_argument(
        "--tp", type=int, default=TENSOR_PARALLEL_SIZE,
        help=f"Tensor parallel size (default: {TENSOR_PARALLEL_SIZE})"
    )
    parser.add_argument(
        "--gpu-mem", type=float, default=GPU_MEMORY_UTIL,
        help=f"GPU memory utilization fraction (default: {GPU_MEMORY_UTIL})"
    )
    parser.add_argument(
        "--eager", action="store_true", default=False,
        help="Disable CUDA graph capture (use eager mode)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
