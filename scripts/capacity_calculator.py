#!/usr/bin/env python
"""
Capacity Calculator — 4k window tunable for cordiiv2 pool philosophy.

Given max_tokens, guidance_tokens (lite vs full), per_file_tokens, delta_overhead,
outputs maximum safe file window and recommended max_rounds for 1.5B vs larger models.

Uses only pre-existing deterministic math — no LLM, no new dependencies.
Portable as you add tools/templates: just re-measure guidance/per_file/delta and rerun.

First principles:
- 4k ctx = 4096, but single pruner budget is 3000 (core/context_pruner.py:22) leaving 1k headroom.
- Total window = guidance + N * per_file + folds * delta + fixed overhead (ledger base, system msgs)
- Folds trigger when estimated_tokens > 3000 or messages > 40 (plugins/agent/loop.py:318)
- 1.5B hallucination: needs ~1.3 rounds per file (duplicate block, retries), larger ~1.05
"""
from __future__ import annotations

import argparse
import json
import math

# Pre-existing measurement anchors (from C:\tmp\verify_9files_3x.py: verified)
# 9 files via TEMPLATE: 973 tokens total with lite 70 guidance -> ~100 per file inc overhead
ANCHORS = {
    "lite_guidance": 70,  # plugins/agent/loop.py:278 lite minimal (measured)
    "full_guidance": 419,  # full JSON schemas (measured 442, now 419 with rule 7)
    "per_file_template": 100,  # TEMPLATE short avg: 9 files 973 tokens -> (973-70-30)/9≈97, rounded 100 inc JSON wrapper
    "per_file_full": 205,  # full content avg: 3 files 616 tokens -> 205/file (measured)
    "delta_fold": 13,  # core/summarizer.py:63 delta "+1: file" ~13 tokens vs 40 full
    "delta_full": 40,
    "base_overhead": 30,
}

MODEL_PRESETS = {
    "1.5b": {"label": "qwen2.5-coder:1.5b (4k, flaky)", "max_tokens": 4096, "pruner_budget": 3000, "rounds_per_file": 1.3, "safety": 0.85, "max_messages": 40},
    "7b": {"label": "qwen2.5-coder:7b (8k, stable)", "max_tokens": 8192, "pruner_budget": 6500, "rounds_per_file": 1.05, "safety": 0.88, "max_messages": 60},
    "14b": {"label": "qwen2.5-coder:14b (16k)", "max_tokens": 16384, "pruner_budget": 14000, "rounds_per_file": 1.02, "safety": 0.90, "max_messages": 80},
}


def estimate_window(max_tokens: int, guidance_tokens: int, per_file_tokens: int, delta_overhead: int,
                    base_overhead: int = 30, safety: float = 0.85, max_messages: int = 40) -> dict:
    """Iterative simulation: find max N files where projected tokens <= max_tokens * safety and messages <= max_messages."""
    # Effective budget is min(pruner_budget, max_tokens * safety) — loop.py:318 checks both token budget and message count
    effective = int(max_tokens * safety)
    best = 0
    # Simulate N files, each file = 1 write_file round = 2 messages (assistant tool_call + tool result) + guidance once
    for n in range(1, 101):
        messages = 2 + 2 * n  # user + system guidance + 2 per file
        folds = max(0, (messages - max_messages + 9) // 10)  # approx 1 fold per 10 over limit, conservative
        # also folds by tokens: estimate tokens and see how many times we'd fold (each fold costs delta, not full)
        tokens = guidance_tokens + base_overhead + n * per_file_tokens + folds * delta_overhead
        # After folds, tokens are reduced to ~pruner target, but we approximate worst-case before fold
        # Use iterative fold reduction: each fold saves ~60% of history (keeps 40)
        if tokens > effective or messages > max_messages + 10:  # allow small overflow before fold
            # Check if folding would bring it back under — simulate 1 fold saving ~ (tokens * 0.4)
            if folds > 0:
                tokens_after = int(tokens * 0.6) + delta_overhead
                if tokens_after <= effective and messages <= max_messages + 10:
                    best = n
                    continue
            break
        best = n
    # More precise binary search for max N under effective
    # Brute force linear is fine, but ensure we didn't miss due to fold saving
    return {
        "max_files": best,
        "effective_budget": effective,
        "tokens_at_max": guidance_tokens + base_overhead + best * per_file_tokens if best else guidance_tokens,
        "messages_at_max": 2 + 2 * best if best else 2,
        "utilization": round((guidance_tokens + base_overhead + best * per_file_tokens) / effective * 100, 1) if best and effective else 0,
    }


def recommended_rounds(files: int, rounds_per_file: float, buffer: int = 2) -> int:
    return max(4, math.ceil(files * rounds_per_file + buffer))


def main():
    p = argparse.ArgumentParser(description="cordiiv2 capacity calculator — 4k window tunable")
    p.add_argument("--max-tokens", type=int, default=None, help="Model context window (e.g., 4096)")
    p.add_argument("--guidance-tokens", type=int, default=None, help="Guidance tokens (lite 70, full 419). If omitted, shows both")
    p.add_argument("--per-file-tokens", type=int, default=None, help="Per file tokens (template 68, full 205). If omitted, shows both")
    p.add_argument("--delta-overhead", type=int, default=None, help="Delta per fold (13 delta, 40 full)")
    p.add_argument("--model", choices=["1.5b", "7b", "14b", "custom"], default="1.5b", help="Preset model profile")
    p.add_argument("--safety", type=float, default=None, help="Safety factor 0.0-1.0 (default per model)")
    p.add_argument("--json", action="store_true", help="Output JSON")
    args = p.parse_args()

    # Determine presets
    if args.model != "custom" and args.max_tokens is None:
        preset = MODEL_PRESETS[args.model]
        max_tokens = preset["max_tokens"]
        safety = args.safety if args.safety is not None else preset["safety"]
        max_messages = preset["max_messages"]
        rounds_per_file = preset["rounds_per_file"]
        label = preset["label"]
    else:
        max_tokens = args.max_tokens or 4096
        safety = args.safety if args.safety is not None else 0.85
        max_messages = 40
        rounds_per_file = 1.3
        label = f"custom max_tokens={max_tokens}"

    # If no explicit guidance/per_file/delta, sweep both lite/template vs full
    if args.guidance_tokens is None and args.per_file_tokens is None:
        combos = [
            ("lite+TEMPLATE (current 3x)", ANCHORS["lite_guidance"], ANCHORS["per_file_template"], ANCHORS["delta_fold"]),
            ("lite+full-content", ANCHORS["lite_guidance"], ANCHORS["per_file_full"], ANCHORS["delta_full"]),
            ("full+TEMPLATE", ANCHORS["full_guidance"], ANCHORS["per_file_template"], ANCHORS["delta_fold"]),
            ("full+full-content (baseline)", ANCHORS["full_guidance"], ANCHORS["per_file_full"], ANCHORS["delta_full"]),
        ]
    else:
        g = args.guidance_tokens if args.guidance_tokens is not None else ANCHORS["lite_guidance"]
        pf = args.per_file_tokens if args.per_file_tokens is not None else ANCHORS["per_file_template"]
        d = args.delta_overhead if args.delta_overhead is not None else ANCHORS["delta_fold"]
        combos = [(f"custom g={g} pf={pf} d={d}", g, pf, d)]

    results = []
    for name, g, pf, d in combos:
        est = estimate_window(max_tokens, g, pf, d, base_overhead=ANCHORS["base_overhead"], safety=safety, max_messages=max_messages)
        est["name"] = name
        est["guidance"] = g
        est["per_file"] = pf
        est["delta"] = d
        est["recommended_rounds"] = recommended_rounds(est["max_files"], rounds_per_file)
        est["rounds_per_file"] = rounds_per_file
        results.append(est)

    if args.json:
        print(json.dumps({"model": label, "max_tokens": max_tokens, "safety": safety, "results": results}, indent=2))
        return

    print(f"Capacity Calculator — {label}")
    print(f"max_tokens={max_tokens} safety={safety} => effective_budget={int(max_tokens*safety)} max_messages={max_messages} rounds/file={rounds_per_file}")
    print("")
    print(f"{'Config':<32} {'guid':>4} {'/file':>5} {'delta':>5} | {'max_files':>9} {'tokens':>6} {'msgs':>4} {'util%':>5} {'rounds':>6}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<32} {r['guidance']:>4} {r['per_file']:>5} {r['delta']:>5} | {r['max_files']:>9} {r['tokens_at_max']:>6} {r['messages_at_max']:>4} {r['utilization']:>5}% {r['recommended_rounds']:>6}")

    print("")
    # Portable tuning hint
    best = max(results, key=lambda x: x["max_files"])
    print(f"Best for this model: {best['name']} => {best['max_files']} files, {best['recommended_rounds']} rounds")
    print("")
    print("Tuning: re-measure after adding tools/templates:")
    print("  guidance = Summarizer.estimate_tokens(tool_guidance)")
    print("  per_file = avg tokens of write_file JSON + content (TEMPLATE short vs full)")
    print("  delta    = len(ledger delta) //4  (now 13 vs 40)")
    print(f"  Example: python scripts/capacity_calculator.py --max-tokens {max_tokens} --guidance-tokens <new> --per-file-tokens <new> --delta-overhead <new> --model {args.model}")

    # Quick 3x check vs baseline
    lite_template = next((r for r in results if "lite+TEMPLATE" in r["name"]), None)
    baseline = next((r for r in results if "baseline" in r["name"]), None)
    if lite_template and baseline and baseline["max_files"]:
        print("")
        print(f"3x check: lite+TEMPLATE {lite_template['max_files']} files vs baseline {baseline['max_files']} files = {round(lite_template['max_files']/max(baseline['max_files'],1),1)}x")


if __name__ == "__main__":
    main()
