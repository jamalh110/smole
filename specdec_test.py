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
  - Short context:  MT-Bench (HuggingFaceH4/mt_bench_prompts) — open-ended
                    multi-turn questions (first turn only), naturally short.
  - Long context:   LongBench v2 (THUDM/LongBench-v2) — full document
                    contexts (no truncation), filtered to fit max_model_len.

Throughput is measured across many concurrent requests to saturate the engine.
"""

import gc
import time
import argparse
import csv
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#TARGET_MODEL = "Qwen/Qwen3-8B"
TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
#DRAFT_MODEL = "Qwen/Qwen3-1.7B"
#DRAFT_MODEL = "AngelSlim/Qwen3-14B_eagle3"
DRAFT_MODEL = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
#DRAFT_MODEL = "Tengyunw/qwen3_8b_eagle3"
#DRAFT_MODEL = "Qwen/Qwen3-0.6B"

NUM_SPEC_TOKENS = 3         # tokens the draft model proposes per step
MAX_NEW_TOKENS = 4096        # tokens to generate per request (thinking tokens count)
NUM_REQUESTS = 64            # total requests per throughput run
REPEAT_MULTIPLIER = 1        # each batch is run this many times for stable measurement
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
MT_BENCH_DATASET = "HuggingFaceH4/mt_bench_prompts"  # short-context open-ended QA
LONGBENCH_DATASET = "THUDM/LongBench-v2"  # long-context document QA


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_mt_bench() -> list[dict]:
    """Load the MT-Bench dataset (first-turn prompts)."""
    ds = load_dataset(MT_BENCH_DATASET, split="train")
    return list(ds)


def load_longbench_v2() -> list[dict]:
    """Load the LongBench v2 dataset."""
    ds = load_dataset(LONGBENCH_DATASET, split="train")
    return list(ds)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_mt_bench_prompt(row: dict) -> str:
    """Format an MT-Bench row using only the first-turn question."""
    return row["prompt"][0]


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
    """Wrap prompt text in the model's chat template (if available)."""
    messages = [{"role": "user", "content": prompt_text}]

    if not getattr(tokenizer, "chat_template", None):
        return prompt_text

    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if "qwen" in TARGET_MODEL.lower() and ENABLE_THINKING:
        kwargs["enable_thinking"] = True

    return tokenizer.apply_chat_template(messages, **kwargs)


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
    Build `num_requests` short prompts from MT-Bench (first turn),
    formatted with the model's chat template.
    """
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    prompts = []
    for i in range(num_requests):
        row = dataset[indices[i % len(indices)]]
        text = _format_mt_bench_prompt(row)
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


def measure_throughput(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    batch_size: int | None = None,
):
    """
    Run generation on all prompts and return throughput metrics.

    If `batch_size` is given, prompts are submitted in chunks of that size
    so vLLM never processes more than `batch_size` concurrent requests.
    Wall-clock time and token counts are aggregated across all chunks.
    """
    # Warm-up run (single short prompt) to ensure model is loaded & compiled
    llm.generate(prompts[:1], sampling_params)

    if batch_size is None:
        batch_size = len(prompts)

    total_input_tokens = 0
    total_output_tokens = 0
    elapsed = 0.0

    counters_before = _read_spec_decode_counters()

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        start = time.perf_counter()
        outputs = llm.generate(chunk, sampling_params)
        elapsed += time.perf_counter() - start
        for output in outputs:
            total_input_tokens += len(output.prompt_token_ids)
            total_output_tokens += len(output.outputs[0].token_ids)

    counters_after = _read_spec_decode_counters()

    result = {
        "num_requests": len(prompts),
        "batch_size": batch_size,
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


SWEEP_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def run_batch_sweep(
    llm: LLM,
    all_prompts: list[str],
    sampling_params: SamplingParams,
    batch_sizes: list[int],
    label: str,
    repeats: int = REPEAT_MULTIPLIER,
) -> list[dict]:
    """
    For each batch size, call llm.generate() `repeats` times with exactly
    `bs` prompts per call so vLLM never sees more than `bs` concurrent
    requests.  Aggregates tokens and wall-clock time across the repeats.
    """
    # Single warmup
    llm.generate(all_prompts[:1], sampling_params)

    results = []
    for bs in batch_sizes:
        batch_prompts = all_prompts[:bs]
        total_output_tokens = 0
        total_input_tokens = 0
        total_elapsed = 0.0

        print(f"  [{label}] batch_size={bs}, repeats={repeats} "
              f"({bs * repeats} total requests) ...")

        for r in range(repeats):
            start = time.perf_counter()
            outputs = llm.generate(batch_prompts, sampling_params)
            elapsed = time.perf_counter() - start
            total_elapsed += elapsed
            for out in outputs:
                total_input_tokens += len(out.prompt_token_ids)
                total_output_tokens += len(out.outputs[0].token_ids)

        tok_per_sec = round(total_output_tokens / total_elapsed, 2)
        print(f"    -> {tok_per_sec:.2f} tok/s  "
              f"({total_output_tokens} tokens in {total_elapsed:.1f}s)")

        results.append({
            "batch_size": bs,
            "output_tokens_per_second": tok_per_sec,
            "total_output_tokens": total_output_tokens,
            "elapsed_seconds": round(total_elapsed, 3),
        })
    return results


def save_sweep_csv(
    sweep_data: dict[str, list[dict]],
    path: str,
) -> None:
    """Save sweep results to a CSV file.

    sweep_data maps a descriptive key (e.g. 'baseline_short') to the list of
    per-batch-size dicts returned by run_batch_sweep.
    """
    fieldnames = ["mode", "context", "batch_size", "output_tokens_per_second"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, rows in sweep_data.items():
            if "baseline" in key:
                mode = "autoregressive"
            else:
                mode = "specdec"
            context = "short" if "short" in key else "long"
            for row in rows:
                writer.writerow({
                    "mode": mode,
                    "context": context,
                    "batch_size": row["batch_size"],
                    "output_tokens_per_second": row["output_tokens_per_second"],
                })
    print(f"Sweep CSV saved to {path}")


def plot_sweep(
    sweep_data: dict[str, list[dict]],
    path: str,
) -> None:
    """Plot output tokens/sec vs batch size for each (mode, context) combo."""
    short_keys = [k for k in sweep_data if "short" in k]
    long_keys = [k for k in sweep_data if "long" in k]
    has_long = len(long_keys) > 0
    ncols = 2 if has_long else 1

    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), squeeze=False)

    def _plot_panel(ax, keys, title):
        for key in sorted(keys):
            rows = sweep_data[key]
            xs = [r["batch_size"] for r in rows]
            ys = [r["output_tokens_per_second"] for r in rows]
            label = "Autoregressive" if "baseline" in key else "Spec Decoding"
            marker = "o" if "baseline" in key else "s"
            ax.plot(xs, ys, marker=marker, label=label, linewidth=2)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Output Tokens / sec")
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(SWEEP_BATCH_SIZES)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend()
        ax.grid(True, alpha=0.3)

    _plot_panel(axes[0, 0], short_keys, "Short Context (MT-Bench)")
    if has_long:
        _plot_panel(axes[0, 1], long_keys, "Long Context (LongBench v2)")

    fig.suptitle("Throughput vs Batch Size: Autoregressive vs Speculative Decoding",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Sweep plot saved to {path}")


def run_benchmark(args):
    """Run the full benchmark suite."""

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    # sampling_params = SamplingParams(
    #     temperature=args.temperature,
    #     top_p=0.95,
    #     top_k=20,
    #     min_p=0.0,
    #     max_tokens=args.max_new_tokens
    # )
    # ----- Load datasets -----
    print("Loading MT-Bench dataset (short prompts) …")
    mt_bench_data = load_mt_bench()
    print(f"  Loaded {len(mt_bench_data)} rows from {MT_BENCH_DATASET}")

    long_prompts = None
    if not args.skip_long_context:
        print("Loading LongBench v2 dataset (long prompts) …")
        longbench_data = load_longbench_v2()
        print(f"  Loaded {len(longbench_data)} rows from {LONGBENCH_DATASET}")

    # ----- Build prompts (Qwen3 chat template, thinking mode) -----
    total_requests = REPEAT_MULTIPLIER * args.num_requests
    sweep_max = max(SWEEP_BATCH_SIZES) if args.sweep else 0
    num_prompts = max(total_requests, sweep_max)

    print("\nBuilding short prompts from MT-Bench …")
    short_prompts = build_short_prompts(
        mt_bench_data, tokenizer, num_prompts, seed=42,
    )

    if not args.skip_long_context:
        print(f"Building long prompts from LongBench v2 "
              f"({args.min_long_tokens}–{args.max_long_tokens} tokens) …")
        long_prompts = build_long_prompts(
            longbench_data, tokenizer, num_prompts,
            min_tokens=args.min_long_tokens,
            max_tokens=args.max_long_tokens,
            seed=123,
        )

    actual_short_len = len(tokenizer.encode(short_prompts[0], add_special_tokens=False))
    print(f"\n{'='*70}")
    print(f"Thinking mode   : ENABLED (enable_thinking=True)")
    print(f"Short prompts   : MT-Bench ({MT_BENCH_DATASET}), ~{actual_short_len} tokens")
    if long_prompts is not None:
        actual_long_len = len(tokenizer.encode(long_prompts[0], add_special_tokens=False))
        print(f"Long prompts    : LongBench v2 ({LONGBENCH_DATASET}), "
              f"{args.min_long_tokens}–{args.max_long_tokens} tok range, "
              f"~{actual_long_len} tokens (first prompt)")
    else:
        actual_long_len = 0
        print(f"Long prompts    : SKIPPED (--skip-long-context)")
    print(f"Max model len   : {args.max_model_len}")
    print(f"Requests per run: {args.num_requests}")
    print(f"Max new tokens  : {args.max_new_tokens}")
    print(f"Temperature     : {args.temperature}")
    print(f"{'='*70}\n")

    results = {}

    # ===================================================================
    # Build speculative config (shared by sweep and regular modes)
    # ===================================================================
    if 'eagle' in DRAFT_MODEL.lower():
        method = "eagle3" if "eagle3" in DRAFT_MODEL.lower() else "eagle"
        speculative_config = {
            "method": method,
            "model": DRAFT_MODEL,
            "num_speculative_tokens": args.num_spec_tokens,
        }
    else:
        speculative_config = {
            "method": "draft_model",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": args.num_spec_tokens,
        },

    if args.sweep:
        # ==============================================================
        # Batch-size sweep mode
        # ==============================================================
        batch_sizes = SWEEP_BATCH_SIZES
        sweep_data: dict[str, list[dict]] = {}

        # --- Autoregressive sweep ---
        print("=" * 70)
        print("  SWEEP — AUTOREGRESSIVE BASELINE")
        print("=" * 70)

        llm_baseline = LLM(
            model=TARGET_MODEL,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_mem,
            max_model_len=args.max_model_len,
            enforce_eager=args.eager,
            disable_log_stats=False,
            enable_prefix_caching=False,
        )

        print("\n>>> Short context sweep (MT-Bench) ...")
        sweep_data["baseline_short"] = run_batch_sweep(
            llm_baseline, short_prompts, sampling_params,
            batch_sizes, "AR-short",
        )

        if long_prompts is not None:
            print(f"\n>>> Long context sweep (LongBench v2) ...")
            sweep_data["baseline_long"] = run_batch_sweep(
                llm_baseline, long_prompts, sampling_params,
                batch_sizes, "AR-long",
            )

        del llm_baseline
        gc.collect()
        torch.cuda.empty_cache()

        # --- Speculative decoding sweep ---
        print("\n" + "=" * 70)
        print(f"  SWEEP — SPECULATIVE DECODING ({DRAFT_MODEL.split('/')[-1]})")
        print("=" * 70)

        llm_spec = LLM(
            model=TARGET_MODEL,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_mem,
            max_model_len=args.max_model_len,
            enforce_eager=args.eager,
            disable_log_stats=False,
            enable_prefix_caching=False,
            speculative_config=speculative_config,
        )

        print("\n>>> Short context sweep (MT-Bench) ...")
        sweep_data["specdec_short"] = run_batch_sweep(
            llm_spec, short_prompts, sampling_params,
            batch_sizes, "SD-short",
        )

        if long_prompts is not None:
            print(f"\n>>> Long context sweep (LongBench v2) ...")
            sweep_data["specdec_long"] = run_batch_sweep(
                llm_spec, long_prompts, sampling_params,
                batch_sizes, "SD-long",
            )

        del llm_spec
        gc.collect()
        torch.cuda.empty_cache()

        # --- Save CSV and plot ---
        os.makedirs("results", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"results/sweep_{ts}.csv"
        plot_path = f"results/sweep_{ts}.png"
        save_sweep_csv(sweep_data, csv_path)
        plot_sweep(sweep_data, plot_path)

        # Also dump the raw sweep data as JSON for later analysis
        json_path = f"results/sweep_{ts}.json"
        with open(json_path, "w") as f:
            json.dump({"config": {
                "target_model": TARGET_MODEL,
                "draft_model": DRAFT_MODEL,
                "batch_sizes": batch_sizes,
                "max_new_tokens": args.max_new_tokens,
                "max_model_len": args.max_model_len,
                "temperature": args.temperature,
                "num_spec_tokens": args.num_spec_tokens,
            }, "sweep": sweep_data}, f, indent=2)
        print(f"\nSweep JSON saved to {json_path}")
        return

    # ===================================================================
    # Regular (non-sweep) benchmark
    # ===================================================================

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
        enable_prefix_caching=False,
    )

    print("\n>>> Short context  (MT-Bench) ...")
    res = measure_throughput(llm_baseline, short_prompts, sampling_params, batch_size=args.num_requests)
    results["baseline_short"] = res
    print(json.dumps(res, indent=2))

    if long_prompts is not None:
        print(f"\n>>> Long context  (LongBench v2, {args.min_long_tokens}–{args.max_long_tokens} tok) ...")
        res = measure_throughput(llm_baseline, long_prompts, sampling_params, batch_size=args.num_requests)
        results["baseline_long"] = res
        print(json.dumps(res, indent=2))

    del llm_baseline
    gc.collect()
    torch.cuda.empty_cache()

    # ===================================================================
    # 2. Speculative decoding  (Qwen3-14B + Qwen3-1.7B draft)
    # ===================================================================
    print("\n" + "=" * 70)
    print(f"  SPECULATIVE DECODING — Qwen3-14B + {DRAFT_MODEL.split('/')[-1]} draft  (thinking mode)")
    print("=" * 70)

    llm_spec = LLM(
        model=TARGET_MODEL,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=args.eager,
        disable_log_stats=False,
        enable_prefix_caching=False,
        speculative_config=speculative_config,
    )

    print("\n>>> Short context  (MT-Bench) ...")
    res = measure_throughput(llm_spec, short_prompts, sampling_params, batch_size=args.num_requests)
    results["specdec_short"] = res
    print(json.dumps(res, indent=2))

    if long_prompts is not None:
        print(f"\n>>> Long context  (LongBench v2, {args.min_long_tokens}–{args.max_long_tokens} tok) ...")
        res = measure_throughput(llm_spec, long_prompts, sampling_params, batch_size=args.num_requests)
        results["specdec_long"] = res
        print(json.dumps(res, indent=2))

    del llm_spec

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    summary_pairs = [
        ("Short context  (MT-Bench)", ("baseline_short", "specdec_short")),
    ]
    if long_prompts is not None:
        summary_pairs.append(
            (f"Long context   (LongBench v2, {args.min_long_tokens}–{args.max_long_tokens} tok)",
             ("baseline_long", "specdec_long")),
        )

    for label, (base_key, spec_key) in summary_pairs:
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
                    "short_prompt_dataset": MT_BENCH_DATASET,
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
    parser.add_argument(
        "--skip-long-context", action="store_true", default=False,
        help="Skip the long-context (LongBench v2) experiments entirely"
    )
    parser.add_argument(
        "--sweep", action="store_true", default=False,
        help="Run a batch-size sweep (1,2,4,8,16,32,64,128) and save CSV + plot"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
