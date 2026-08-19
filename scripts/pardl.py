#!/usr/bin/env python3
"""Resumable parallel-range downloader for Hugging Face files.

This host's network: IPv6 is dead, the Xet CAS endpoint is unreachable, and a
single HTTPS stream tops out near 0.5 MB/s. Parallel ranged GETs against the
classic /resolve/ endpoint are the only fast path, so that is what this does.

Usage: pardl.py <repo_id> <outdir> <nworkers> [comma,separated,filename,filters]
"""
import json
import os
import socket
import sys
import threading
import time
import urllib.request

_orig = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, fl=0: _orig(h, p, socket.AF_INET, t, pr, fl)

REPO, OUTDIR, NWORK = sys.argv[1], sys.argv[2], int(sys.argv[3])
ONLY = sys.argv[4].split(",") if len(sys.argv) > 4 else None
REV = os.environ.get("REV", "main")
CHUNK = 32 << 20
UA = {"User-Agent": "curl/8.5.0"}


def api(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def size_of(url):
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[1])


def fetch(url, path, total, state):
    nchunks = (total + CHUNK - 1) // CHUNK
    if nchunks == 0:                      # zero-byte file: just create it
        open(path, "wb").close()
        json.dump({"done": []}, open(path + ".state", "w"))
        return
    done = set(state.get("done", []))
    if not os.path.exists(path) or os.path.getsize(path) != total:
        with open(path, "ab") as f:
            f.truncate(total)
    lock = threading.Lock()
    todo = [i for i in range(nchunks) if i not in done]
    started = len(done)
    counter = {"n": started, "t0": time.time()}

    def worker():
        fh = os.open(path, os.O_WRONLY)
        try:
            while True:
                with lock:
                    if not todo:
                        return
                    i = todo.pop(0)
                lo, hi = i * CHUNK, min((i + 1) * CHUNK, total) - 1
                for attempt in range(8):
                    try:
                        req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={lo}-{hi}"})
                        with urllib.request.urlopen(req, timeout=180) as r:
                            buf = r.read()
                        if len(buf) != hi - lo + 1:
                            raise IOError("short read %d" % len(buf))
                        os.pwrite(fh, buf, lo)
                        with lock:
                            done.add(i)
                            counter["n"] += 1
                            if counter["n"] % 8 == 0:
                                state["done"] = sorted(done)
                                json.dump(state, open(path + ".state", "w"))
                        break
                    except Exception:
                        time.sleep(2 * (attempt + 1))
                else:
                    with lock:
                        todo.append(i)
        finally:
            os.close(fh)

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(NWORK)]
    for t in ts:
        t.start()
    while any(t.is_alive() for t in ts):
        time.sleep(20)
        el = max(time.time() - counter["t0"], 1)
        gained = counter["n"] - started
        rate = gained * CHUNK / 1048576 / el
        eta = (nchunks - counter["n"]) * CHUNK / 1048576 / max(rate, 0.01) / 60
        print("  %s: %5.1f%%  %6.0f/%.0f MB  %5.2f MB/s  ETA %5.1f min"
              % (os.path.basename(path), 100.0 * counter["n"] / nchunks,
                 counter["n"] * CHUNK / 1048576, total / 1048576, rate, eta), flush=True)
    for t in ts:
        t.join()
    state["done"] = sorted(done)
    json.dump(state, open(path + ".state", "w"))


os.makedirs(OUTDIR, exist_ok=True)
meta = api("https://huggingface.co/api/models/%s/revision/%s" % (REPO, REV))
for fn in [f["rfilename"] for f in meta["siblings"]]:
    if ONLY and not any(fn == o or fn.endswith(o) for o in ONLY):
        continue
    url = "https://huggingface.co/%s/resolve/%s/%s" % (REPO, REV, fn)
    out = os.path.join(OUTDIR, fn)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    total = size_of(url)
    st = json.load(open(out + ".state")) if os.path.exists(out + ".state") else {}
    nchunks = (total + CHUNK - 1) // CHUNK
    if os.path.exists(out) and os.path.getsize(out) == total and len(st.get("done", [])) >= nchunks:
        print("[skip] %s" % fn, flush=True)
        continue
    print("[get ] %s  %.0f MB" % (fn, total / 1048576), flush=True)
    fetch(url, out, total, st)
print("ALL DONE", flush=True)
