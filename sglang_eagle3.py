"""
Speculative Decoding Throughput Benchmark (SGLang + EAGLE3)
============================================================
Compares maximum throughput of:
  1. Speculative decoding: target model + EAGLE3 draft head
  2. Autoregressive baseline: target model alone

Uses SGLang's offline Engine API with proper EAGLE3 speculative decoding.

Two input-context regimes:
  - Short context:  MT-Bench (HuggingFaceH4/mt_bench_prompts) — open-ended
                    multi-turn questions (first turn only), naturally short.
  - Long context:   LongBench v2 (THUDM/LongBench-v2) — full document
                    contexts (no truncation), filtered to fit max_model_len.

Throughput is measured by submitting all requests at once; the engine's
--max-running-requests setting controls the effective batch size (concurrency).

Usage examples:
  # Regular benchmark (short + long context)
  python bench_sglang_eagle3.py

  # Sweep across batch sizes, short context only
  python bench_sglang_eagle3.py --sweep --skip-long-context

  # Use Triton backend for tree attention on Ampere (A100)
  python bench_sglang_eagle3.py --attention-backend triton

  # Chain-only decoding (topk=1), works on all backends
  python bench_sglang_eagle3.py --eagle-topk 1
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
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
#DRAFT_MODEL = "lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge"
TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "Tengyunw/qwen3_8b_eagle3"

NUM_SPEC_TOKENS = 60          # --speculative-num-draft-tokens (max parallel verification)
NUM_SPEC_STEPS = 6           # --speculative-num-steps (depth of autoregressive drafting)
EAGLE_TOPK = 10               # --speculative-eagle-topk (branching factor per step)
MAX_NEW_TOKENS = 512        # tokens to generate per request
NUM_REQUESTS = 64            # total requests per throughput run
REPEAT_MULTIPLIER = 4        # each batch is run this many times for stable measurement
TEMPERATURE = 0.0            # sampling temperature (0 = greedy, >0 = stochastic)
GPU_MEMORY_UTIL = 0.6       # fraction of GPU memory SGLang may use (leave headroom for draft model)
TENSOR_PARALLEL_SIZE = 1     # adjust for multi-GPU setups
MAX_MODEL_LEN = 40_960       # max sequence length (input + output)
MIN_LONG_TOKENS = 16_000     # minimum prompt tokens for long-context bucket
MAX_LONG_TOKENS = 32_000     # maximum prompt tokens for long-context bucket

ENABLE_THINKING = False

# ---------------------------------------------------------------------------
# Dataset identifiers
# ---------------------------------------------------------------------------
MT_BENCH_DATASET = "HuggingFaceH4/mt_bench_prompts"
LONGBENCH_DATASET = "THUDM/LongBench-v2"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_mt_bench() -> list[dict]:
    ds = load_dataset(MT_BENCH_DATASET, split="train")
    return list(ds)


def load_longbench_v2() -> list[dict]:
    ds = load_dataset(LONGBENCH_DATASET, split="train")
    return list(ds)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_mt_bench_prompt(row: dict) -> str:
    return row["prompt"][0]


def _format_longbench_prompt(row: dict) -> str:
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

def build_short_prompts(dataset, tokenizer, num_requests, seed=42):
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    prompts = []
    for i in range(num_requests):
        row = dataset[indices[i % len(indices)]]
        text = _format_mt_bench_prompt(row)
        prompts.append(_apply_chat_template(tokenizer, text))
    return prompts


def build_long_prompts(dataset, tokenizer, num_requests, min_tokens, max_tokens, seed=123):
    rng = random.Random(seed)
    min_words_estimate = int(min_tokens / 1.5)
    max_words_estimate = int(max_tokens / 1.0)
    candidates = [
        row for row in dataset
        if min_words_estimate <= len(row["context"].split()) <= max_words_estimate
    ]
    print(f"  Pre-filter: {len(candidates)}/{len(dataset)} rows with "
          f"~{min_words_estimate}–{max_words_estimate} words")

    valid_prompts = []
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
    return [valid_prompts[i % len(valid_prompts)] for i in range(num_requests)]


# ---------------------------------------------------------------------------
# SGLang engine helpers
# ---------------------------------------------------------------------------

def create_engine(args, speculative: bool = False, max_running_requests: int | None = None):
    """
    Create an SGLang Engine instance.

    SGLang's offline `Engine` API is analogous to vLLM's `LLM` class.
    Speculative decoding is configured via server-arg-style kwargs.
    max_running_requests controls the scheduler's maximum concurrent batch size.
    """
    import sglang as sgl

    engine_kwargs = dict(
        model_path=TARGET_MODEL,
        tp_size=args.tp,
        mem_fraction_static=args.gpu_mem,
        context_length=args.max_model_len,
        disable_cuda_graph=args.eager,
        disable_radix_cache=True,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        log_level="info",
    )

    if max_running_requests is not None:
        engine_kwargs["max_running_requests"] = max_running_requests

    if args.attention_backend:
        engine_kwargs["attention_backend"] = args.attention_backend

    if speculative:
        engine_kwargs["speculative_algorithm"] = "EAGLE3"
        engine_kwargs["speculative_draft_model_path"] = DRAFT_MODEL
        engine_kwargs["speculative_num_steps"] = args.num_spec_steps
        engine_kwargs["speculative_eagle_topk"] = args.eagle_topk
        engine_kwargs["speculative_num_draft_tokens"] = args.num_spec_tokens

    return sgl.Engine(**engine_kwargs)


def engine_generate(engine, prompts: list[str], args) -> list[dict]:
    """
    Run generation on a list of prompts using SGLang's Engine.generate().

    Returns a list of dicts with 'prompt_tokens' and 'output_tokens' counts,
    plus the raw 'text' output.
    """
    # SGLang's Engine.generate() accepts a list of prompts and a dict of
    # sampling params.  It returns a list of result dicts.
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.temperature > 0:
        sampling_params["top_p"] = 0.95

    outputs = engine.generate(prompts, sampling_params)

    results = []
    for out in outputs:
        meta = out.get("meta_info", {})
        entry = {
            "prompt_tokens": meta.get("prompt_tokens", 0),
            "output_tokens": meta.get("completion_tokens", 0),
            "text": out.get("text", ""),
        }
        if "spec_accept_rate" in meta:
            entry["spec_accept_rate"] = meta["spec_accept_rate"]
            entry["spec_accept_length"] = meta["spec_accept_length"]
            entry["spec_verify_ct"] = meta["spec_verify_ct"]
            entry["spec_accept_token_num"] = meta["spec_accept_token_num"]
            entry["spec_draft_token_num"] = meta["spec_draft_token_num"]
        if "spec_accept_histogram" in meta:
            entry["spec_accept_histogram"] = meta["spec_accept_histogram"]
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Throughput measurement
# ---------------------------------------------------------------------------

def measure_throughput(engine, prompts: list[str], args):
    """
    Submit all prompts at once and return throughput metrics.
    Concurrency is controlled by the engine's max_running_requests setting.
    """
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.temperature > 0:
        sampling_params["top_p"] = 0.95

    engine.generate(prompts[:1], sampling_params)

    total_input_tokens = 0
    total_output_tokens = 0
    total_accepted = 0
    total_drafted = 0
    total_verify_ct = 0

    start = time.perf_counter()
    outputs = engine.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    for out in outputs:
        meta = out.get("meta_info", {})
        total_input_tokens += meta.get("prompt_tokens", 0)
        total_output_tokens += meta.get("completion_tokens", 0)
        total_accepted += meta.get("spec_accept_token_num", 0)
        total_drafted += meta.get("spec_draft_token_num", 0)
        total_verify_ct += meta.get("spec_verify_ct", 0)

    result = {
        "num_requests": len(prompts),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "output_tokens_per_second": round(total_output_tokens / elapsed, 2) if elapsed > 0 else 0,
        "total_tokens_per_second": round(
            (total_input_tokens + total_output_tokens) / elapsed, 2
        ) if elapsed > 0 else 0,
    }
    if total_drafted > 0:
        result["spec_accept_rate"] = round(total_accepted / total_drafted, 4)
        result["spec_avg_accept_length"] = round(total_output_tokens / total_verify_ct, 2) if total_verify_ct > 0 else 0
        result["spec_total_verify_steps"] = total_verify_ct
    return result


SWEEP_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


def run_batch_sweep(args, all_prompts, batch_sizes, speculative, label, repeats=REPEAT_MULTIPLIER):
    """
    For each batch size, create an engine with max_running_requests=bs,
    submit all bs*repeats requests at once, and aggregate metrics.
    """
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.temperature > 0:
        sampling_params["top_p"] = 0.95

    results = []
    for bs in batch_sizes:
        total_requests = bs * repeats
        prompts = all_prompts[:total_requests]

        print(f"  [{label}] max_running_requests={bs}, "
              f"total_requests={total_requests} ...")

        engine = create_engine(args, speculative=speculative, max_running_requests=bs)
        engine.generate(prompts[:1], sampling_params)

        total_output_tokens = 0
        total_input_tokens = 0
        total_accepted = 0
        total_drafted = 0
        total_verify_ct = 0

        start = time.perf_counter()
        outputs = engine.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start

        for out in outputs:
            meta = out.get("meta_info", {})
            total_input_tokens += meta.get("prompt_tokens", 0)
            total_output_tokens += meta.get("completion_tokens", 0)
            total_accepted += meta.get("spec_accept_token_num", 0)
            total_drafted += meta.get("spec_draft_token_num", 0)
            total_verify_ct += meta.get("spec_verify_ct", 0)

        engine.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()

        tok_per_sec = round(total_output_tokens / elapsed, 2) if elapsed > 0 else 0
        accept_rate = round(total_accepted / total_drafted, 4) if total_drafted > 0 else None
        avg_accept_len = round(total_output_tokens / total_verify_ct, 2) if total_verify_ct > 0 else None

        rate_str = f", accept_rate={accept_rate:.2%}, avg_accept_len={avg_accept_len:.1f}" if accept_rate is not None else ""
        print(f"    -> {tok_per_sec:.2f} tok/s  "
              f"({total_output_tokens} tokens in {elapsed:.1f}s{rate_str})")

        entry = {
            "batch_size": bs,
            "output_tokens_per_second": tok_per_sec,
            "total_output_tokens": total_output_tokens,
            "elapsed_seconds": round(elapsed, 3),
        }
        if accept_rate is not None:
            entry["spec_accept_rate"] = accept_rate
            entry["spec_avg_accept_length"] = avg_accept_len
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# CSV / Plot helpers
# ---------------------------------------------------------------------------

def save_sweep_csv(sweep_data, path):
    fieldnames = ["mode", "context", "batch_size", "output_tokens_per_second"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, rows in sweep_data.items():
            mode = "autoregressive" if "baseline" in key else "specdec"
            context = "short" if "short" in key else "long"
            for row in rows:
                writer.writerow({
                    "mode": mode,
                    "context": context,
                    "batch_size": row["batch_size"],
                    "output_tokens_per_second": row["output_tokens_per_second"],
                })
    print(f"Sweep CSV saved to {path}")


def plot_sweep(sweep_data, path):
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
            label = "Autoregressive" if "baseline" in key else f"EAGLE3 Spec Decoding"
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

    fig.suptitle("Throughput vs Batch Size: Autoregressive vs EAGLE3 Speculative Decoding (SGLang)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Sweep plot saved to {path}")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args):
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)

    # ----- Load datasets -----
    print("Loading MT-Bench dataset (short prompts) …")
    mt_bench_data = load_mt_bench()
    print(f"  Loaded {len(mt_bench_data)} rows from {MT_BENCH_DATASET}")

    long_prompts = None
    if not args.skip_long_context:
        print("Loading LongBench v2 dataset (long prompts) …")
        longbench_data = load_longbench_v2()
        print(f"  Loaded {len(longbench_data)} rows from {LONGBENCH_DATASET}")

    # ----- Build prompts -----
    total_requests = REPEAT_MULTIPLIER * args.num_requests
    sweep_max = max(SWEEP_BATCH_SIZES) * REPEAT_MULTIPLIER if args.sweep else 0
    num_prompts = max(total_requests, sweep_max)

    print("\nBuilding short prompts from MT-Bench …")
    short_prompts = build_short_prompts(mt_bench_data, tokenizer, num_prompts, seed=42)

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
    print(f"Engine          : SGLang (EAGLE3 speculative decoding)")
    print(f"Target model    : {TARGET_MODEL}")
    print(f"Draft model     : {DRAFT_MODEL}")
    print(f"EAGLE3 config   : steps={args.num_spec_steps}, topk={args.eagle_topk}, "
          f"draft_tokens={args.num_spec_tokens}")
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
    print(f"Max running reqs: {args.num_requests}")
    print(f"Repeat multipli.: {REPEAT_MULTIPLIER}x ({args.num_requests * REPEAT_MULTIPLIER} total requests)")
    print(f"Max new tokens  : {args.max_new_tokens}")
    print(f"Temperature     : {args.temperature}")
    if args.attention_backend:
        print(f"Attention backend: {args.attention_backend}")
    print(f"{'='*70}\n")

    results = {}

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

        print("\n>>> Short context sweep (MT-Bench) ...")
        sweep_data["baseline_short"] = run_batch_sweep(
            args, short_prompts, batch_sizes, speculative=False, label="AR-short",
        )

        if long_prompts is not None:
            print(f"\n>>> Long context sweep (LongBench v2) ...")
            sweep_data["baseline_long"] = run_batch_sweep(
                args, long_prompts, batch_sizes, speculative=False, label="AR-long",
            )

        # --- EAGLE3 speculative decoding sweep ---
        print("\n" + "=" * 70)
        print(f"  SWEEP — EAGLE3 SPECULATIVE DECODING ({DRAFT_MODEL.split('/')[-1]})")
        print("=" * 70)

        print("\n>>> Short context sweep (MT-Bench) ...")
        sweep_data["specdec_short"] = run_batch_sweep(
            args, short_prompts, batch_sizes, speculative=True, label="SD-short",
        )

        if long_prompts is not None:
            print(f"\n>>> Long context sweep (LongBench v2) ...")
            sweep_data["specdec_long"] = run_batch_sweep(
                args, long_prompts, batch_sizes, speculative=True, label="SD-long",
            )

        # --- Save CSV and plot ---
        os.makedirs("results", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"results/sweep_sglang_{ts}.csv"
        plot_path = f"results/sweep_sglang_{ts}.png"
        save_sweep_csv(sweep_data, csv_path)
        plot_sweep(sweep_data, plot_path)

        json_path = f"results/sweep_sglang_{ts}.json"
        with open(json_path, "w") as f:
            json.dump({"config": {
                "engine": "sglang",
                "target_model": TARGET_MODEL,
                "draft_model": DRAFT_MODEL,
                "eagle3_steps": args.num_spec_steps,
                "eagle3_topk": args.eagle_topk,
                "eagle3_draft_tokens": args.num_spec_tokens,
                "batch_sizes": batch_sizes,
                "max_new_tokens": args.max_new_tokens,
                "max_model_len": args.max_model_len,
                "temperature": args.temperature,
                "attention_backend": args.attention_backend,
            }, "sweep": sweep_data}, f, indent=2)
        print(f"\nSweep JSON saved to {json_path}")
        return

    # ===================================================================
    # Regular (non-sweep) benchmark
    # ===================================================================

    # --- Autoregressive baseline ---
    print("=" * 70)
    print(f"  AUTOREGRESSIVE BASELINE — {TARGET_MODEL}")
    print("=" * 70)

    engine_baseline = create_engine(args, speculative=False, max_running_requests=args.num_requests)

    print("\n>>> Short context  (MT-Bench) ...")
    res = measure_throughput(engine_baseline, short_prompts, args)
    results["baseline_short"] = res
    print(json.dumps(res, indent=2))

    if long_prompts is not None:
        print(f"\n>>> Long context  (LongBench v2, {args.min_long_tokens}–{args.max_long_tokens} tok) ...")
        res = measure_throughput(engine_baseline, long_prompts, args)
        results["baseline_long"] = res
        print(json.dumps(res, indent=2))

    engine_baseline.shutdown()
    del engine_baseline
    gc.collect()
    torch.cuda.empty_cache()

    # --- EAGLE3 speculative decoding ---
    print("\n" + "=" * 70)
    print(f"  EAGLE3 SPECULATIVE DECODING — {TARGET_MODEL} + {DRAFT_MODEL.split('/')[-1]}")
    print("=" * 70)

    engine_spec = create_engine(args, speculative=True, max_running_requests=args.num_requests)

    print("\n>>> Short context  (MT-Bench) ...")
    res = measure_throughput(engine_spec, short_prompts, args)
    results["specdec_short"] = res
    print(json.dumps(res, indent=2))

    if long_prompts is not None:
        print(f"\n>>> Long context  (LongBench v2, {args.min_long_tokens}–{args.max_long_tokens} tok) ...")
        res = measure_throughput(engine_spec, long_prompts, args)
        results["specdec_long"] = res
        print(json.dumps(res, indent=2))

    engine_spec.shutdown()
    del engine_spec

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
        b_tps = b["output_tokens_per_second"]
        s_tps = s["output_tokens_per_second"]
        speedup = s_tps / b_tps if b_tps > 0 else float("inf")
        print(f"\n  {label}:")
        print(f"    Autoregressive : {b_tps:>10.2f} tok/s  ({b['elapsed_seconds']:.1f}s)")
        print(f"    EAGLE3 spec dec: {s_tps:>10.2f} tok/s  ({s['elapsed_seconds']:.1f}s)")
        print(f"    Speedup        : {speedup:>10.2f}x")
        if "spec_accept_rate" in s:
            print(f"    Accept rate    : {s['spec_accept_rate']:>10.2%}")
            print(f"    Avg accept len : {s['spec_avg_accept_length']:>10.2f} tokens/step")

    # ===================================================================
    # Save results
    # ===================================================================
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/specdec_sglang_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "engine": "sglang",
                    "target_model": TARGET_MODEL,
                    "draft_model": DRAFT_MODEL,
                    "eagle3_steps": args.num_spec_steps,
                    "eagle3_topk": args.eagle_topk,
                    "eagle3_draft_tokens": args.num_spec_tokens,
                    "short_prompt_dataset": MT_BENCH_DATASET,
                    "long_prompt_dataset": LONGBENCH_DATASET,
                    "max_new_tokens": args.max_new_tokens,
                    "max_model_len": args.max_model_len,
                    "min_long_tokens": args.min_long_tokens,
                    "max_long_tokens": args.max_long_tokens,
                    "num_requests": args.num_requests,
                    "temperature": args.temperature,
                    "tensor_parallel_size": args.tp,
                    "attention_backend": args.attention_backend,
                    "short_prompt_tokens": actual_short_len,
                    "long_prompt_tokens": actual_long_len,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark EAGLE3 speculative decoding throughput with SGLang"
    )
    parser.add_argument("--num-requests", type=int, default=NUM_REQUESTS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--min-long-tokens", type=int, default=MIN_LONG_TOKENS)
    parser.add_argument("--max-long-tokens", type=int, default=MAX_LONG_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)

    # EAGLE3-specific parameters
    parser.add_argument(
        "--num-spec-tokens", type=int, default=NUM_SPEC_TOKENS,
        help="Max parallel verification capacity (--speculative-num-draft-tokens). Default: 8"
    )
    parser.add_argument(
        "--num-spec-steps", type=int, default=NUM_SPEC_STEPS,
        help="Depth of autoregressive drafting (--speculative-num-steps). Default: 3"
    )
    parser.add_argument(
        "--eagle-topk", type=int, default=EAGLE_TOPK,
        help="Branching factor per step (--speculative-eagle-topk). "
             "Set to 1 for chain-only decoding (works on all backends). "
             "Set >1 for tree decoding (requires FlashInfer or Triton). Default: 4"
    )

    parser.add_argument("--tp", type=int, default=TENSOR_PARALLEL_SIZE)
    parser.add_argument("--gpu-mem", type=float, default=GPU_MEMORY_UTIL)
    parser.add_argument("--eager", action="store_true", default=False,
                        help="Disable CUDA graph capture")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=64,
                        help="Max batch size for CUDA graph capture (default: 64)")
    parser.add_argument("--skip-long-context", action="store_true", default=False)
    parser.add_argument("--sweep", action="store_true", default=False,
                        help="Run batch-size sweep and save CSV + plot")
    parser.add_argument(
        "--attention-backend", type=str, default=None,
        choices=["flashinfer", "triton", "fa3"],
        help="Attention backend override. Default: auto (flashinfer on Ampere, fa3 on Hopper). "
             "Use 'triton' for tree attention on Ampere with MLA models."
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())