#!/usr/bin/env python3
"""Single-stream decode speed by workload type.

Speculative decoding only pays off when the next tokens are predictable, so a
single tok/s number is meaningless without saying what was being generated.
This measures the same axis the DSpark write-up reports, so the two are
comparable: batch-1 decode, thinking disabled.
"""
import json
import statistics
import sys
import time
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1") + "/chat/completions"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen38-27b"
REPS = 3

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
    ("free-form prose",
     "Write three detailed paragraphs about the history of thermal conductivity measurement.",
     400),
    ("technical explanation",
     "Explain how Gated DeltaNet linear attention differs from standard softmax attention, "
     "and why it changes KV-cache scaling. Be thorough.",
     400),
    ("math / structured reasoning",
     "A lab runs 17 samples per batch, 23 batches per week, for 6 weeks, but discards 4% of samples. "
     "Show your arithmetic step by step, then give the final count. "
     "Then do the same for 19 samples per batch over 8 weeks.",
     400),
    ("code generation",
     "Write a complete, well-commented Python function that computes the structure factor S(q) "
     "from a radial distribution function, including argument validation and a docstring.",
     400),
    ("code editing (repeat + modify)",
     "Here is a function:\n\n```python\n" + CODE_SNIPPET + "```\n\n"
     "Rewrite it in full, vectorised with numpy broadcasting instead of the double loop. "
     "Output the complete updated function.",
     400),
    ("JSON structured output",
     "Emit a JSON array of 12 objects, each with keys: id (int), formula (string), "
     "band_gap_eV (float), stable (bool), notes (string). Use realistic inorganic compounds. "
     "Output only JSON.",
     400),
]


def stream(prompt, maxtok):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": maxtok, "temperature": 0.0, "top_p": 1.0,
               "chat_template_kwargs": {"enable_thinking": False},
               "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            d = json.loads(body)
            if d.get("usage"):
                ntok = d["usage"]["completion_tokens"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content") and ttft is None:
                ttft = time.time() - t0
    total = time.time() - t0
    return ttft or total, total, ntok


print(f"model={MODEL}  batch-1 decode, thinking off, temperature 0\n")
print(f"{'workload':<32} {'tok/s':>8} {'tokens':>8}")
print("-" * 52)
rows = []
for name, prompt, maxtok in WORKLOADS:
    rates, toks = [], 0
    for _ in range(REPS):
        ttft, total, n = stream(prompt, maxtok)
        if n:
            rates.append(n / max(total - ttft, 1e-6))
            toks = n
    med = statistics.median(rates) if rates else 0.0
    rows.append((name, med))
    print(f"{name:<32} {med:8.2f} {toks:8d}")
print("-" * 52)
best = max(rows, key=lambda r: r[1])
worst = min(rows, key=lambda r: r[1])
print(f"spread: {worst[1]:.1f} tok/s ({worst[0]}) -> {best[1]:.1f} tok/s ({best[0]})")
