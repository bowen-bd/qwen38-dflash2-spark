#!/usr/bin/env python3
"""Comprehensive Multi-Condition Standard Benchmark Suite for Qwen3.8-27B Serving.

Evaluates 6 standard operational dimensions:
1. Workload Diversity & Output Predictability (Prose, Tech, Code Gen, Code Edit, JSON, Math)
2. Concurrency & Throughput Scaling (1, 2, 4, 8, 16 parallel streams)
3. Context / Prompt Length Scaling (Prefill tok/s & TTFT from 128 to 64k tokens)
4. Prefix Caching / Radix Cache (Cold vs Warm TTFT speedup)
5. Output Generation Length Scaling (64 to 1024 tokens decode stability)
6. Reasoning / Thinking Mode (Thinking On vs Off)
"""

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.request

try:
    import numpy as np
except ImportError:
    np = None

URL = os.getenv("BENCH_URL", "http://127.0.0.1:8888/v1/chat/completions")
SERVER_INFO_URL = os.getenv("SERVER_INFO_URL", "http://127.0.0.1:8888/get_server_info")
MODEL = os.getenv("BENCH_MODEL", "qwen3.8-27b-sglang")

CODE_SNIPPET = '''\
def compute_rdf(positions, box_length, n_bins=100, r_max=None):
    """Radial distribution function for a cubic periodic cell."""
    n = len(positions)
    if r_max is None:
        r_max = box_length / 2.0
    edges = np.linspace(0.0, r_max, n_bins + 1)
    hist = np.zeros(n_bins)
    for i in range(n):
        for j in range(i + 1, n):
            d = positions[i] - positions[j]
            d -= box_length * np.round(d / box_length)
            r = np.linalg.norm(d)
            if r < r_max:
                hist[int(r / r_max * n_bins)] += 2
    shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = n / box_length ** 3
    return edges[:-1], hist / (n * rho * shell)
'''

WORKLOADS = [
    ("Free-form Prose",
     "Write three detailed paragraphs about the history of thermal conductivity measurement in experimental physics.",
     400),
    ("Technical Explanation",
     "Explain how Gated DeltaNet linear attention differs from standard softmax attention, and why it changes KV-cache scaling. Be thorough and rigorous.",
     400),
    ("Python Code Generation",
     "Write a complete, well-commented Python function that computes the structure factor S(q) from a radial distribution function, including argument validation, type hints, and a docstring.",
     400),
    ("Code Editing / Refactor",
     "Here is a function:\n\n```python\n" + CODE_SNIPPET + "```\n\nRewrite it in full, vectorised with numpy broadcasting instead of the double loop. Output the complete updated function.",
     400),
    ("Structured JSON Output",
     "Emit a JSON array of 12 objects, each with keys: id (int), formula (string), band_gap_eV (float), stable (bool), notes (string). Use realistic inorganic semiconductors. Output only JSON.",
     400),
    ("Step-by-Step Math / Reasoning",
     "A high-throughput lab runs 17 synthesis samples per batch, 23 batches per week, for 6 weeks, but discards 4.5% of samples due to purity checks. Show your arithmetic step by step, then give the exact final count. Then do the same for 19 samples per batch over 8 weeks with a 3.2% discard rate.",
     400),
]

FILLER_BASE = (
    "Materials science investigates the relationship between the structure of materials at atomic scales "
    "and their macroscopic properties. Density functional theory provides a quantum mechanical modeling method "
    "used in physics, chemistry and materials science to investigate the electronic structure of many-body systems. "
    "Machine learning interatomic potentials combine the accuracy of ab initio calculations with the computational "
    "efficiency of empirical force fields, accelerating molecular dynamics simulations by orders of magnitude. "
)


def get_server_stats():
    for endpoint in ["/get_server_info", "/server_info"]:
        try:
            base = URL.split("/v1")[0]
            req = urllib.request.Request(f"{base}{endpoint}", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.load(r)
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
                return {
                    "avg_spec_accept_length": data.get("avg_spec_accept_length"),
                    "memory_usage": data.get("memory_usage", {}),
                }
        except Exception:
            pass
    return {}


def make_context_prompt(target_tokens, task_instruction="Summarize the key themes discussed above in two concise bullet points."):
    target_chars = target_tokens * 4
    reps = max(1, target_chars // len(FILLER_BASE))
    full_text = (FILLER_BASE * reps)[:target_chars]
    return f"Document text:\n{full_text}\n\nTask: {task_instruction}"


def stream_request(prompt, max_tokens, enable_thinking=False, temperature=0.0, url=URL, model=MODEL, timeout=1200):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0 if temperature == 0.0 else 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    ttft = None
    first_content_t = None
    chunks_times = []
    reasoning_tok_count = 0
    completion_tok_count = 0
    prompt_tokens = 0
    text_chunks = []
    reasoning_chunks = []

    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw_line in r:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except Exception:
                continue

            now = time.time()
            if chunk.get("usage"):
                u = chunk["usage"]
                prompt_tokens = u.get("prompt_tokens", 0)
                completion_tok_count = u.get("completion_tokens", 0)
                reasoning_tok_count = u.get("reasoning_tokens", 0)

            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                r_content = delta.get("reasoning_content")
                c_content = delta.get("content")

                if r_content:
                    if ttft is None:
                        ttft = now - t0
                    chunks_times.append(now)
                    reasoning_chunks.append(r_content)

                if c_content:
                    if ttft is None:
                        ttft = now - t0
                    if first_content_t is None:
                        first_content_t = now - t0
                    chunks_times.append(now)
                    text_chunks.append(c_content)

    t_end = time.time()
    total_time = t_end - t0
    if ttft is None:
        ttft = total_time

    decode_time = max(total_time - ttft, 1e-6)
    if completion_tok_count == 0 and chunks_times:
        completion_tok_count = len(chunks_times)

    decode_tok_s = completion_tok_count / decode_time if decode_time > 0 else 0.0
    itl_ms = (decode_time / max(completion_tok_count - 1, 1)) * 1000.0 if completion_tok_count > 1 else 0.0

    return {
        "ttft_s": ttft,
        "first_content_s": first_content_t or ttft,
        "total_s": total_time,
        "decode_s": decode_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tok_count,
        "reasoning_tokens": reasoning_tok_count,
        "decode_tok_s": decode_tok_s,
        "itl_ms": itl_ms,
        "output_sample": ("".join(text_chunks))[:150],
    }


def run_warmup():
    print(" Warming up engine...", flush=True)
    try:
        stream_request("Ping! Return 1 word.", 16, enable_thinking=False)
        stream_request("Warmup query 2.", 32, enable_thinking=True)
    except Exception as e:
        print(f"Warmup warning: {e}")
    print(" Warmup complete.\n", flush=True)


def test_workload_diversity(reps=3):
    print("================================================================================")
    print(" DIMENSION 1: Workload Diversity & Output Predictability (Batch=1, Thinking Off)")
    print("================================================================================")
    results = []
    print(f"{'Workload Domain':<30} | {'Decode tok/s':>12} | {'TTFT (ms)':>10} | {'ITL (ms)':>9} | {'Tokens':>7}")
    print("-" * 80)

    for name, prompt, maxtok in WORKLOADS:
        rep_results = []
        for _ in range(reps):
            r = stream_request(prompt, maxtok, enable_thinking=False, temperature=0.0)
            rep_results.append(r)

        med_decode = statistics.median([r["decode_tok_s"] for r in rep_results])
        med_ttft = statistics.median([r["ttft_s"] * 1000 for r in rep_results])
        med_itl = statistics.median([r["itl_ms"] for r in rep_results])
        med_toks = int(statistics.median([r["completion_tokens"] for r in rep_results]))

        results.append({
            "workload": name,
            "decode_tok_s": med_decode,
            "ttft_ms": med_ttft,
            "itl_ms": med_itl,
            "tokens": med_toks,
            "runs": rep_results,
        })
        print(f"{name:<30} | {med_decode:12.2f} | {med_ttft:10.1f} | {med_itl:9.2f} | {med_toks:7d}")

    stats = get_server_stats()
    print(f"\n-> Server Avg Speculative Accept Length: {stats.get('avg_spec_accept_length', 'N/A')}")
    return results


def test_concurrency_scaling(concurrencies=[1, 2, 4, 8, 16], tokens=300):
    print("\n================================================================================")
    print(f" DIMENSION 2: Concurrency / Batch Scaling ({tokens} tokens per client, Thinking Off)")
    print("================================================================================")
    print(f"{'Concurrency':<12} | {'Aggregate tok/s':>16} | {'Per-Stream tok/s':>17} | {'Avg TTFT (ms)':>14} | {'Wall (s)':>9}")
    print("-" * 80)

    results = []
    base_prompt = "Write a comprehensive summary describing the thermal and electronic transport properties of transition metal dichalcogenides."

    for conc in concurrencies:
        prompts = [f"{base_prompt} (Client request ID: {i})" for i in range(conc)]
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as executor:
            futures = [executor.submit(stream_request, p, tokens, False, 0.0) for p in prompts]
            stream_res = [f.result() for f in futures]
        wall_time = time.time() - t0

        total_toks = sum(r["completion_tokens"] for r in stream_res)
        agg_tok_s = total_toks / wall_time if wall_time > 0 else 0.0
        per_stream_tok_s = agg_tok_s / conc if conc > 0 else 0.0
        avg_ttft_ms = statistics.mean([r["ttft_s"] * 1000 for r in stream_res])
        p90_ttft_ms = float(np.percentile([r["ttft_s"] * 1000 for r in stream_res], 90)) if (np and stream_res) else 0.0

        res_item = {
            "concurrency": conc,
            "aggregate_tok_s": agg_tok_s,
            "per_stream_tok_s": per_stream_tok_s,
            "avg_ttft_ms": avg_ttft_ms,
            "p90_ttft_ms": p90_ttft_ms,
            "wall_s": wall_time,
            "total_tokens": total_toks,
            "streams": stream_res,
        }
        results.append(res_item)
        print(f"{conc:<12} | {agg_tok_s:16.2f} | {per_stream_tok_s:17.2f} | {avg_ttft_ms:14.1f} | {wall_time:9.2f}")

    return results


def test_prompt_context_scaling(token_sizes=[128, 512, 1024, 4096, 16384, 32768, 65536]):
    print("\n================================================================================")
    print(" DIMENSION 3: Context / Prompt Length Scaling (Prefill tok/s & TTFT)")
    print("================================================================================")
    print(f"{'Target Prompt Tok':<18} | {'Actual Prompt Tok':>18} | {'TTFT (s)':>10} | {'Prefill tok/s':>16} | {'Gen Speed':>11}")
    print("-" * 80)

    results = []
    for t_size in token_sizes:
        salt = f"\n[Session timestamp: {time.time_ns()}]"
        prompt = make_context_prompt(t_size) + salt
        try:
            r = stream_request(prompt, max_tokens=16, enable_thinking=False)
            actual_ptok = r["prompt_tokens"] if r["prompt_tokens"] > 0 else t_size
            prefill_tok_s = actual_ptok / max(r["ttft_s"], 1e-6)

            res_item = {
                "target_prompt_tokens": t_size,
                "actual_prompt_tokens": actual_ptok,
                "ttft_s": r["ttft_s"],
                "ttft_ms": r["ttft_s"] * 1000.0,
                "prefill_tok_s": prefill_tok_s,
                "decode_tok_s": r["decode_tok_s"],
            }
            results.append(res_item)
            print(f"{t_size:<18} | {actual_ptok:18d} | {r['ttft_s']:10.3f} | {prefill_tok_s:16.1f} | {r['decode_tok_s']:10.1f} tok/s")
        except Exception as e:
            print(f"{t_size:<18} | ERROR: {e}")

    return results


def test_prefix_caching(context_size=8192):
    print("\n================================================================================")
    print(" DIMENSION 4: Prefix Caching / Radix Cache (Cold vs Warm Context Hit)")
    print("================================================================================")

    prefix_body = (FILLER_BASE * (context_size // 25))[:context_size * 4]
    cold_tag = f"\n[Unique Cache ID: {time.time_ns()}]\n"
    shared_context = f"{cold_tag}{prefix_body}\n"

    print(" 1. Sending cold prefix request (0% Radix cache hit)...", flush=True)
    cold_prompt = f"{shared_context}\nQuestion 1: What is the main subject discussed in this text?"
    r_cold = stream_request(cold_prompt, max_tokens=24, enable_thinking=False)

    print(" 2. Sending warm prefix request (100% Radix cache hit on shared prefix)...", flush=True)
    warm_prompt = f"{shared_context}\nQuestion 2: Briefly explain how DFT is mentioned in this text."
    r_warm = stream_request(warm_prompt, max_tokens=24, enable_thinking=False)

    speedup = r_cold["ttft_s"] / max(r_warm["ttft_s"], 1e-6)
    print("-" * 80)
    print(f" Cold TTFT: {r_cold['ttft_s']*1000:8.2f} ms ({r_cold['prompt_tokens']} prompt tokens)")
    print(f" Warm TTFT: {r_warm['ttft_s']*1000:8.2f} ms ({r_warm['prompt_tokens']} prompt tokens)")
    print(f" Cache Hit Speedup on TTFT: {speedup:6.2f}x faster")

    return {
        "context_tokens": r_cold["prompt_tokens"],
        "cold_ttft_ms": r_cold["ttft_s"] * 1000,
        "warm_ttft_ms": r_warm["ttft_s"] * 1000,
        "speedup_factor": speedup,
    }


def test_generation_length_scaling(lengths=[64, 128, 256, 512, 1024]):
    print("\n================================================================================")
    print(" DIMENSION 5: Output Generation Length Scaling (Decode Stability)")
    print("================================================================================")
    print(f"{'Target Output Tok':<18} | {'Actual Output Tok':>18} | {'Decode tok/s':>14} | {'ITL (ms)':>10} | {'Total Time (s)':>15}")
    print("-" * 80)

    prompt = "Write an exhaustive textbook chapter detailing the physical chemistry of solid-state battery electrolytes, ion conduction mechanisms, and interfacial degradation pathways."
    results = []

    for l in lengths:
        r = stream_request(prompt, max_tokens=l, enable_thinking=False)
        res_item = {
            "target_tokens": l,
            "actual_tokens": r["completion_tokens"],
            "decode_tok_s": r["decode_tok_s"],
            "itl_ms": r["itl_ms"],
            "total_s": r["total_s"],
        }
        results.append(res_item)
        print(f"{l:<18} | {r['completion_tokens']:18d} | {r['decode_tok_s']:14.2f} | {r['itl_ms']:10.2f} | {r['total_s']:15.2f}")

    return results


def test_thinking_mode_comparison():
    print("\n================================================================================")
    print(" DIMENSION 6: Reasoning / Thinking Mode (Thinking On vs Off)")
    print("================================================================================")
    print(f"{'Mode':<18} | {'Total Tok':>10} | {'Reasoning Tok':>14} | {'TTFT (ms)':>11} | {'Decode tok/s':>14} | {'Total (s)':>10}")
    print("-" * 80)

    math_prompt = "A battery pack has 96 cells in series, each with 3.2V nominal voltage and 50Ah capacity. Calculate total nominal energy in kWh, total capacity in Ah, and then calculate energy if arranged in 48S2P configuration."

    r_off = stream_request(math_prompt, max_tokens=512, enable_thinking=False)
    print(f"{'Thinking OFF':<18} | {r_off['completion_tokens']:10d} | {r_off['reasoning_tokens']:14d} | {r_off['ttft_s']*1000:11.1f} | {r_off['decode_tok_s']:14.2f} | {r_off['total_s']:10.2f}")

    r_on = stream_request(math_prompt, max_tokens=512, enable_thinking=True)
    print(f"{'Thinking ON':<18} | {r_on['completion_tokens']:10d} | {r_on['reasoning_tokens']:14d} | {r_on['ttft_s']*1000:11.1f} | {r_on['decode_tok_s']:14.2f} | {r_on['total_s']:10.2f}")

    return {
        "thinking_off": r_off,
        "thinking_on": r_on,
    }


def main():
    # `global` must precede any use of the name in this scope; reading URL/MODEL as
    # argparse defaults above the declaration is a SyntaxError, not a runtime one, so
    # the whole module failed to import.
    global URL, MODEL

    parser = argparse.ArgumentParser(description="Multi-Condition Standard Benchmark Suite")
    parser.add_argument("--url", default=URL, help="OpenAI chat completions URL")
    parser.add_argument("--model", default=MODEL, help="Served model name")
    parser.add_argument("--output", default="standard-benchmark-results.json", help="Output JSON path")
    args = parser.parse_args()

    URL = args.url
    MODEL = args.model

    print(f"\n================================================================================")
    print(f" LLM STANDARD PERFORMANCE BENCHMARK SUITE")
    print(f" Model: {MODEL}")
    print(f" Endpoint: {URL}")
    print(f" Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"================================================================================\n")

    run_warmup()

    workloads_data = test_workload_diversity(reps=3)
    concurrency_data = test_concurrency_scaling(concurrencies=[1, 2, 4, 8, 16], tokens=300)
    context_data = test_prompt_context_scaling(token_sizes=[128, 512, 1024, 4096, 16384, 32768, 65536])
    prefix_data = test_prefix_caching(context_size=8192)
    gen_len_data = test_generation_length_scaling(lengths=[64, 128, 256, 512, 1024])
    thinking_data = test_thinking_mode_comparison()
    server_stats = get_server_stats()

    summary_payload = {
        "model": MODEL,
        "endpoint": URL,
        "timestamp": time.time(),
        "server_stats": server_stats,
        "workload_diversity": workloads_data,
        "concurrency_scaling": concurrency_data,
        "prompt_context_scaling": context_data,
        "prefix_caching": prefix_data,
        "generation_length_scaling": gen_len_data,
        "thinking_comparison": thinking_data,
    }

    with open(args.output, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"\n Full benchmark results saved to: {args.output}")


if __name__ == "__main__":
    main()
