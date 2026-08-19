#!/usr/bin/env python3
"""Throughput + accuracy harness for a local vLLM OpenAI endpoint.

Reports single-stream decode, concurrent aggregate decode, and prefill rate,
using the same shapes the DGX Spark community threads publish so the numbers
are directly comparable.

  python3 bench-qwen38.py throughput [--url ...] [--model ...]
  python3 bench-qwen38.py accuracy   [--n 100]
  python3 bench-qwen38.py vision     --image path_or_url
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

DEF_URL = "http://127.0.0.1:8000/v1"


def post(url, payload, timeout=1800):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stream(url, payload, timeout=1800):
    """Return (ttft_s, total_s, n_completion_tokens)."""
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    t0 = time.time()
    ttft = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
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
            if ch and (ch[0].get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.time() - t0
    return ttft or (time.time() - t0), time.time() - t0, ntok


def chat_payload(model, prompt, maxtok, effort="none"):
    """effort='none' disables thinking via the chat template; vLLM's qwen3
    reasoning parser only accepts low/medium/xhigh for reasoning_effort."""
    p = {"model": model, "messages": [{"role": "user", "content": prompt}],
         "max_tokens": maxtok, "temperature": 0.7, "top_p": 0.8,
         "presence_penalty": 1.5}
    if effort in (None, "", "none", "off"):
        p["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        p["reasoning_effort"] = effort
    return p


FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank. "
          "Scientists measured the reaction rate carefully. ")


def make_prompt(approx_tokens):
    reps = max(1, approx_tokens // 20)
    return (FILLER * reps) + "\n\nSummarize the text above in one sentence."


def throughput(args):
    url = args.url + "/chat/completions"
    print(f"model={args.model}  endpoint={args.url}\n")

    print("== warmup ==")
    stream(url, chat_payload(args.model, "Say OK.", 8))

    print("\n== single-stream decode (short prompt, 256 new tokens) ==")
    rates = []
    for i in range(args.reps):
        ttft, tot, n = stream(url, chat_payload(
            args.model, "Write a detailed paragraph about heat transfer in metals.", 256))
        dec = n / max(tot - ttft, 1e-6)
        rates.append(dec)
        print(f"  run {i+1}: ttft={ttft*1000:7.0f} ms  {n:4d} tok  decode={dec:6.2f} tok/s")
    print(f"  -> median single-stream decode: {statistics.median(rates):.2f} tok/s")

    print("\n== prefill (TTFT vs prompt length) ==")
    for plen in args.prefill:
        pr = make_prompt(plen)
        ttft, tot, n = stream(url, chat_payload(args.model, pr, 16))
        approx = len(pr) // 4
        print(f"  ~{approx:6d} tok prompt: ttft={ttft:6.2f} s  -> {approx/max(ttft,1e-6):8.0f} tok/s prefill")

    print("\n== concurrent aggregate decode (256 new tokens each) ==")
    for conc in args.concurrency:
        res = [None] * conc
        def go(i):
            res[i] = stream(url, chat_payload(
                args.model, f"Explain topic #{i}: the physics of thermal conduction, in detail.", 256))
        th = [threading.Thread(target=go, args=(i,)) for i in range(conc)]
        t0 = time.time()
        [t.start() for t in th]
        [t.join() for t in th]
        wall = time.time() - t0
        toks = sum(r[2] for r in res if r)
        print(f"  conc={conc:3d}: {toks:5d} tok in {wall:6.2f} s -> {toks/wall:7.2f} tok/s aggregate "
              f"({toks/wall/conc:5.2f} tok/s per stream)")


GSM_PATH = os.environ.get(
    "GSM8K_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_test.json")
)
NUM_RE = __import__("re").compile(r"-?\d+(?:\.\d+)?")


def extract_answer(txt):
    """Last number in the reply. Returned as a float so that 57, 57.0 and
    57.00 all score as the same answer -- the model formats freely and we are
    measuring arithmetic, not formatting."""
    cleaned = txt.replace(",", "").replace("$", "").replace("**", "")
    nums = NUM_RE.findall(cleaned)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def same_number(got, gold):
    if got is None:
        return False
    try:
        return abs(got - float(gold)) < 1e-6
    except ValueError:
        return False


def accuracy(args):
    url = args.url + "/chat/completions"
    items = json.load(open(GSM_PATH))
    n = args.n or 100
    items = items[:n]
    lock = threading.Lock()
    stats = {"ok": 0, "done": 0, "err": 0, "ctok": 0}
    results = [None] * len(items)

    def one(i, it):
        p = chat_payload(args.model, it["q"] + "\n\nGive the final numeric answer on the last line.",
                         args.max_tokens, effort=args.effort)
        p["temperature"] = 0.0
        p["top_p"] = 1.0
        p.pop("presence_penalty", None)
        try:
            r = post(url, p)
            txt = r["choices"][0]["message"]["content"] or ""
            ctok = r["usage"]["completion_tokens"]
        except Exception as e:
            with lock:
                stats["err"] += 1
                stats["done"] += 1
            results[i] = (it["a"], f"ERR {type(e).__name__}", False)
            return
        got = extract_answer(txt)
        hit = same_number(got, it["a"])
        with lock:
            stats["ok"] += hit
            stats["done"] += 1
            stats["ctok"] += ctok
            if stats["done"] % 10 == 0:
                print(f"    {stats['done']}/{len(items)}  running acc={100*stats['ok']/stats['done']:.1f}%",
                      flush=True)
        results[i] = (it["a"], got, hit)

    t0 = time.time()
    q = list(enumerate(items))
    qlock = threading.Lock()

    def worker():
        while True:
            with qlock:
                if not q:
                    return
                i, it = q.pop(0)
            one(i, it)

    th = [threading.Thread(target=worker) for _ in range(args.acc_conc)]
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.time() - t0
    n = len(items)
    print(f"\n  GSM8K {stats['ok']}/{n} = {100*stats['ok']/n:.1f}%   effort={args.effort}"
          f"   errors={stats['err']}")
    print(f"  {stats['ctok']} completion tokens, {wall:.0f}s wall, "
          f"{stats['ctok']/wall:.1f} tok/s aggregate, {stats['ctok']/n:.0f} tok/question")
    miss = [(g, o) for g, o, h in results if not h][:10]
    if miss:
        print(f"  first misses (gold -> got): {miss}")


def vision(args):
    url = args.url + "/chat/completions"
    src = args.image
    if not src.startswith("http"):
        import base64, mimetypes
        mt = mimetypes.guess_type(src)[0] or "image/png"
        src = f"data:{mt};base64," + base64.b64encode(open(src, "rb").read()).decode()
    p = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0.2,
         "messages": [{"role": "user", "content": [
             {"type": "image_url", "image_url": {"url": src}},
             {"type": "text", "text": args.question}]}]}
    if args.effort in (None, "", "none", "off"):
        p["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        p["reasoning_effort"] = args.effort
    t0 = time.time()
    r = post(url, p)
    print(r["choices"][0]["message"]["content"])
    print(f"\n[{time.time()-t0:.1f}s, {r['usage']['completion_tokens']} completion tokens]")


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["throughput", "accuracy", "vision"])
ap.add_argument("--url", default=DEF_URL)
ap.add_argument("--model", default="qwen38-27b")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--acc-conc", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--effort", default="none")
ap.add_argument("--image", default="")
ap.add_argument("--question", default="Describe this image in detail. If it contains a chart, read out the numbers.")
ap.add_argument("--concurrency", type=int, nargs="*", default=[1, 2, 4, 8])
ap.add_argument("--prefill", type=int, nargs="*", default=[1000, 4000, 16000])
a = ap.parse_args()
{"throughput": throughput, "accuracy": accuracy, "vision": vision}[a.mode](a)
