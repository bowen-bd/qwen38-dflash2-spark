# Qwen3.8-27B NVFP4 + DFlash2 on a DGX Spark (GB10)

Measured 2026-08-19 on a single NVIDIA DGX Spark (GB10, 121.7 GiB unified memory,
CUDA 13.0), serving `RadixArk/Qwen3.8-27B-NVFP4` under SGLang.

DFlash2 landed upstream in SGLang on **2026-08-19** ([PR #35371]). Its published
numbers are a **BF16 target on an H200** — Hopper, which has no FP4 tensor cores — so
the NVFP4 path had never been exercised. This repo is what happened when we ran it on
one: **three blocking bugs**, a controlled three-arm benchmark that separates the
drafter's contribution from the cost of the stack change it requires, and the accuracy
measurement nobody else published.

[PR #35371]: https://github.com/sgl-project/sglang/pull/35371

## TL;DR

- **DFlash2 does not run on an NVFP4 target at all.** Its candidate selector requires a
  dense `lm_head`; an NVFP4 head is packed `U8` and it raises on the first decode.
- **A second bug** kills it whenever `max-running-requests > cuda-graph-max-bs`.
- **A third**, found while productionising: on the DFlash2 path the GDN state pool needs
  **5 slots per request, not 4**, or concurrency is silently clamped (8 → 6).
- Isolated on one stack, DFlash2 beats DSpark **+41% to +54% on every workload** and
  **+61% at 8 streams**, from acceptance **3.29 → 5.03** tokens per verification pass.
- **No accuracy loss.** With the stack held constant both drafters score **144/150** on
  GSM8K — identical, exactly as losslessness predicts.
- **The shipping config beats every arm we built**: **+43% prose,
  +95% technical, +88% math,
  +49% code gen** over the DSpark baseline,
  **194.5 tok/s at 8 concurrent streams**, **145/150** on GSM8K, in
  **6.8 GB less memory** and a **4× faster start** — because it never dequantizes the head.
  See [Performance](#performance) and [deploy/](deploy/).

> **Serving is delegated to [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark][mia].** They solved
> the `lm_head` problem better than we did — running the quantized head in place via
> `quant_method.apply` instead of dequantizing 2.5 GB — and their GB10 tuning (FP8 KV,
> GDN pool sizing, mamba memory ratio, CPU pinning) beats ours. This repo is the
> measurement and integration layer: the controlled experiment, the accuracy data, the
> NVFP4 decode validation, and the Claude Code / Codex / Harbor wiring.

[mia]: https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark

## Quickstart

```bash
git clone https://github.com/bowen-bd/qwen38-dflash2-spark.git && cd qwen38-dflash2-spark
./setup.sh                    # clones the serving stack, builds the image, fetches weights
deploy/serve-production.sh    # :8888, model qwen3.8-27b-sglang
clients/smoke-test.sh         # 9 checks across all three wire protocols
```

Everything lands under `$QWEN_HOME` (default `~/llm`). No path in this repo is specific
to the machine it was developed on, and the GSM8K test split is vendored, so the accuracy
benchmark needs no external fetch.

## Do you need this repo?

**If you only want to serve the model: no.** Clone [MiaAI-Lab][mia] and stop. Their stack
is better tuned for GB10 than anything here — FP8 KV, GDN pool sizing, mamba memory
ratio, CPU pinning, and an in-place quantized `lm_head` that avoids dequantizing 2.5 GB.
This repo *delegates* serving to them precisely because they do it better.

**This repo adds four things their repo does not have:**

1. **Accuracy.** They publish no correctness numbers at all. This is the only measurement
   of whether DFlash2 is lossless on NVFP4: GSM8K **144/150 for both drafters** on an
   identical stack, and 143 → 144 → 145 as `lm_head` fidelity rises.
2. **A controlled comparison.** Their benchmark is two numbers with no control. The
   three-arm design here separates the *drafter* from the *stack change it requires* —
   which is why the isolated +41–54% and the end-to-end +8–29% are both true and are
   different quantities.
3. **Three bugs**, two of which bite their own stack: the GDN pool needs **5 slots per
   request, not 4** (worth +54% at 8 streams), and their config **500s every Claude Code
   request** because no `--chat-template` is passed.
4. **Agent integration.** Claude Code, Codex, and Harbor wiring, with a smoke test —
   none of which exists upstream.

**Also still needed for vLLM.** Their `lm_head` fix is SGLang-only. vLLM's DFlash2 PR is
still open with the quantized-head gap unresolved, so `scripts/patch_lmhead_nvfp4.py`
remains the only route there.

If findings 3 are upstreamed and accepted, that part of this repo's value merges into
theirs — which would be a good outcome, and the measurement and integration layers would
remain.

## Performance

Full report with all six dimensions: **[BENCHMARK.md](BENCHMARK.md)**, generated from
[`results/standard-benchmark-results.json`](results/standard-benchmark-results.json) by
`scripts/render-benchmark-report.py`. Headline, single stream, thinking off:

| Workload | Decode tok/s |
| :--- | :---: |
| Free-form prose | 25.23 |
| Technical explanation | 39.27 |
| Python code generation | 44.43 |
| Code editing / refactor | 52.37 |
| Structured JSON | 54.70 |
| Step-by-step math | **60.40** |

- **194.5 tok/s aggregate at 8 concurrent streams**
  (283 ms TTFT). 16 streams is not a win — aggregate goes flat
  (197.7) while TTFT degrades to 6204 ms,
  because `--max-running-requests 8` admits eight and the rest queue.
- **11.86× faster TTFT** on a warm radix prefix — which is why subagents sharing a system
  prompt cost far less than N× one agent.
- **No decode degradation to 1k output tokens**, and **thinking is nearly free**
  (53.68 tok/s off vs 52.43 on) — it costs tokens, not throughput.
- **Acceptance 3.98 tokens per verification pass** (2.23–7.08, ceiling 8). Decode is
  memory-bandwidth-bound, so accepted-tokens-per-pass is very nearly the whole story —
  it is what produces the 2.39× spread in the table above.

## The controlled experiment

The numbers above characterise the shipping stack. They cannot tell you how much of it is
*DFlash2* rather than the rest of the stack, because DFlash2 cannot run on the baseline
configuration at all. That needed three arms, all at a 200k KV cap on the earlier overlay
build:

| | drafter | engine | `lm_head` |
|---|---|---|---|
| **A** | DSpark | fork image | NVFP4 |
| **B** | DSpark | upstream overlay | BF16 |
| **C** | **DFlash2** | upstream overlay | BF16 |

**B is the control.** C vs A mixes the drafter with the stack change it forces; B changes
only the stack, so **C vs B isolates the drafter**:

- **The drafter is worth +41% to +54% on every workload** and +61% at 8 streams, from
  acceptance 3.29 → 5.03.
- **The stack change cost −3% to −26%** on its own. The proof it is not a drafter effect:
  acceptance is identical across builds (3.32 vs 3.29), so DSpark speculates the same on
  both and the loss is pure per-pass cost — the dense BF16 head streams 2.54 GB per pass
  instead of NVFP4's 0.64 GB. The shipping stack avoids this entirely by keeping the head
  quantized, which is why it beats every arm here.
- **Accuracy is unchanged by the drafter.** GSM8K, 150 questions, identical settings:

  | arm | GSM8K | note |
  |---|---|---|
  | A — DSpark, NVFP4 head | 143/150 | |
  | B — DSpark, BF16 head | 144/150 | |
  | C — DFlash2, BF16 head | 144/150 | **identical to B** — the drafter changes nothing |
  | D — DFlash2, native BF16 head | 145/150 | highest head fidelity |

  With the stack held constant the drafter moves accuracy by *zero* questions, which is
  what losslessness predicts. At n=150 the standard error is ±1.8 pt, so all four are
  statistically indistinguishable and the honest claim is "no detectable loss".
- **Memory: +3.0 GB**, ~1.0 GB of it the larger drafter (confirmed twice: drafter
  `mem usage` 3.03 → 4.04 GB, and nvidia-smi B→C +998 MiB).

Per-arm logs are in [`results/`](results/); provenance for what each server actually
loaded is in
[`results/serve-log-provenance.txt`](results/serve-log-provenance.txt).

## The `lm_head` problem

DFlash2's selector reads the target's `lm_head` directly to get its top-16 candidates per
position, and the guard is a pure dtype test (`srt/speculative/dflash_utils.py:678`):

```python
def is_dense_head_weight(weight: Any) -> bool:
    return weight is not None and weight.dtype in _DENSE_HEAD_DTYPES
```

An NVFP4 head is packed `U8`, so it fails and `compute_candidates` raises. Upstream *does*
screen the head in `_maybe_build_draft_sampler` (line 487) and returns
`_eager("quantized lm_head")`, so it is meant to degrade gracefully — but the eager fallback
still routes through `compute_candidates`. Upstream also has a quantized-head path,
`_greedy_sample_from_quantized_head` (line 1069), but only for DFlash **v1**'s greedy
sampler; the v2 selector never got one. vLLM's [PR #52883] does not close this either — it
only widens a type check so heads that are *already* unquantized stop being falsely
rejected. **There is currently no working quantized-head path in either engine.**

[PR #52883]: https://github.com/vllm-project/vllm/pull/52883

**The fix that actually ships is none of these.** [MiaAI-Lab][mia]'s
`patch/dflash2_nvfp4_head.patch` calls `quant_method.apply(lm_head, hidden, None)` and
runs the *quantized* head in place, so the selector never needs a dense matrix: no
dequantization, no extra 2.5 GB, no fidelity question. Their build script notes that a
dequant-once approach allocated 2.5–5 GB at draft-graph capture and hard-rebooted the
box. Prefer it on SGLang.

The dense-head routes below remain relevant for **vLLM**, where the quantized-head gap is
still open, and they are how the NVFP4 decode was validated:

| approach | fidelity | memory | notes |
|---|---|---|---|
| `patch_lmhead_nvfp4.py` | cosine **0.9955** | +2.54 GB | dequantizes NVFP4 exactly — dense, *not* more accurate |
| `swap_native_head.py` | cosine **1.000000** | +2.54 GB | drops in the official FP8 checkpoint's genuinely unquantized BF16 head |
| full BF16 target | 1.0 | +24 GB | what upstream benchmarked; defeats the point of NVFP4 |

`swap_native_head.py` is strictly better than dequantizing at identical cost, and may also
raise acceptance, since the selector picks candidates *through* that head.

### Decoding NVFP4 correctly

`lm_head` is stored as `weight` `U8 [248320, 2560]` (two E2M1 nibbles per byte),
`weight_scale` `F8_E4M3 [248320, 320]` (one FP8 scale per 16-element block), and a global
F32 `weight_scale_2`:

```
W[i,j] = e2m1(nibble) * f8e4m3(weight_scale[i, j//16]) * weight_scale_2
```

Nibble order was **verified, not assumed** — element `2k` is the *low* nibble of byte `k`.
Validated against the official FP8 checkpoint's dense BF16 head, which is the same weights
unquantized:

| decode | cosine vs reference |
|---|---|
| **low-nibble-first** | **0.995463** |
| high-nibble-first | 0.001564 |

0.9955 is exactly NVFP4's own quantization error. Getting this backwards would silently
corrupt the output head — and because the *target* uses that head for its real logits, not
just the drafter, it would quietly degrade quality rather than crash.

## Bugs worth filing

1. **DFlash2's selector cannot use a quantized `lm_head`**, and the eager fallback raises
   instead of degrading. Blocks every NVFP4/FP8 deployment. See
   [`results/dflash2-failures.md`](results/dflash2-failures.md).
2. **`max-running-requests > cuda-graph-max-bs` crashes DFlash2.**
   `_SelectorDraftSampler` allocates `greedy_mask`/`temperatures`/`candidate_out` at
   `cuda_graph_max_bs` — addresses baked into the captured graph — then
   `stage_sampling_params()` indexes them with the live batch size:
   `RuntimeError: The size of tensor a (4) must match the size of tensor b (8)`.
   Upstream's own `run.sh` ships `4` with `8`; harmless for DSpark/MTP, fatal here.
3. **The GDN state pool needs 5 slots per request on the DFlash2 path, not 4.** The
   `concurrency × 4` formula is right for `extra_buffer_lazy` + EAGLE/MTP, but
   `start-dflash.sh` switches to `extra_buffer` with block-8 drafting. A request for 8
   concurrent silently becomes 6: *"max_running_requests is capped to 6 by the mamba
   state cache (max_mamba_cache_size=32, 5 state slots per request)"*. Worth **+54% at 8
   streams** (107.4 → 165.7 tok/s) — a one-line change to `MAMBA_SLOTS_PER_REQ`.
4. **No `--chat-template` means every Claude Code request 500s.** The stock template
   rejects the `reasoning_effort: "max"` that Claude Code sends:
   `ValueError: Unexpected reasoning effort max`. `scripts/sglang/chat-template-sglang.jinja`
   maps `max`/`high` → `xhigh` and `minimal` → `low`, and renders mid-conversation system
   turns as `<system-reminder>` blocks in place.

## Operating notes

**The KV pool is shared, not partitioned.** `context_len` caps one request;
`--max-total-tokens` sizes the pool everyone draws from. Four concurrent agents do *not*
get 65k each — one can use 200k while the others stay small. When the pool cannot fit
everything, the scheduler queues (`fcfs`) and retracts longest-first
(`retraction_policy='length'`) rather than failing; you lose throughput, not correctness.
Radix prefix caching is on, so agents sharing a system prompt store it **once** — the
common case for subagents. Useful lever: **the pool may exceed `context_len`**. Setting
`MAX_TOTAL=524288` gives four agents ~131k each concurrently while still capping any
single request at 262k, for ~10.5 GB more KV at the measured ~40 KB/token.

**Surviving a reboot takes three things**, not just `systemctl enable`: `loginctl
enable-linger` (user units do not start until someone logs in), the unit installed, and
any stale predecessor disabled. There is also a lifecycle trap — `start.sh` detaches once
the server answers, so under `Type=simple` systemd reads that exit as the service
stopping and fires `ExecStop`, killing the container it just started.
`serve-production.sh` therefore blocks on `docker wait`, and the unit uses
`Restart=always` because `docker wait` exits 0 carrying the container status.

## Reproducing

`./setup.sh` does all of this; the steps are spelled out here for transparency.

No published SGLang tag ships DFlash2 — it merged upstream on 2026-08-19, after every
tag — so it is overlaid onto the pinned `qwen38-27b` base image. The overlay is five
hand-adapted files plus a patch that lets the selector drive a **quantized** `lm_head`
in place, all from [MiaAI-Lab][mia]'s `patch/`. Two things make the naive approach fail:

- The base image is a **fork**: `561c8f3` is not an upstream commit and it lacks
  `mamba_extra_buffer_enabled`, so upstream's `spec_utils` will not import against it.
- Pin upstream at the **merge commit** `c14312a66`, not `main`. Later drift pulls in
  symbols the fork does not have, which is why a minimal overlay taken from HEAD
  cascades into a whole-tree swap.

Benchmarks. The suite plus renderer is the primary path — it regenerates every table in
BENCHMARK.md from one JSON, so a rerun cannot leave stale numbers behind:

```bash
scripts/bench-standard-suite.py \
    --url http://127.0.0.1:8888/v1/chat/completions \
    --model qwen3.8-27b-sglang \
    --output results/standard-benchmark-results.json
scripts/render-benchmark-report.py                    # -> BENCHMARK.md
```

`scripts/bench-dflash2.sh` is the older per-arm driver, kept because it is what produced
the controlled experiment (and it is the only one that runs GSM8K):

```bash
scripts/bench-dflash2.sh myrun        # workloads, concurrency, prefill, vision, GSM8K
OUT=/tmp/runs URL=http://127.0.0.1:8888/v1 MODEL=qwen3.8-27b-sglang scripts/bench-dflash2.sh myrun
```

### Two things that will bite you on GB10

**Cap the KV cache.** Left alone, SGLang sized a **744,276-token** pool (36.9 GiB) and
left only 10.4 GiB of host memory free once torch.compile and graph capture were
resident. On unified memory an OOM freezes the whole host, not just the process.
`serve-production.sh` sets `--max-total-tokens 262144` (the checkpoint's full context)
and refuses to launch below 8 GiB of headroom.

**`--mem-fraction-static` is a share of the whole pool,** and SGLang derives "already
used" from `total − free`, which on unified memory counts your desktop. Free-after is
always `(1−frac) × total`, so the guard is exact. Upstream reports a hard reboot at
`0.95`; `0.90` is the working value.

## Files

```
scripts/run-sglang.sh          launcher; SPEC/MEM_FRAC/MAX_TOTAL/CG_MAX_BS/UPSTREAM_TREE knobs
scripts/bench-standard-suite.py multi-condition benchmark: 6-axis suite across workloads/concurrency/context/caching
scripts/bench-dflash2.sh       one arm end-to-end: workloads, concurrency, vision, GSM8K
scripts/bench-standard-suite.py  the 6-dimension suite that produces BENCHMARK.md
scripts/render-benchmark-report.py  renders BENCHMARK.md from the suite's JSON
scripts/bench-workloads.py     batch-1 decode by workload type (the acceptance-sensitive axis)
scripts/bench-qwen38.py        throughput / accuracy / vision modes
scripts/patch_lmhead_nvfp4.py  NVFP4 lm_head -> dense BF16
scripts/swap_native_head.py    swap in the FP8 checkpoint's native BF16 head instead
scripts/pardl.py               parallel ranged-GET downloader (HF Xet is unreachable here)
scripts/model-router.py        routes qwen* -> local, everything else -> Anthropic
scripts/sglang/                chat template (fixes the Claude Code 500)
scripts/test-chart.png         vision benchmark input
scripts/gsm8k_test.json        GSM8K test split, vendored (1,319 questions)
setup.sh                       clone serving stack + build image + fetch weights
deploy/                        production launcher, systemd units, deployment notes
clients/                       Claude Code / Codex / Harbor configs + smoke test
results/                       per-arm logs, curated failures, serve-log provenance
```

## Caveats

- **Accuracy is bounded at roughly ±1.8 pt**, not tighter. Every GSM8K figure here is
  n=150, so 143 vs 144 vs 145 are statistically indistinguishable and the honest claim is
  "no detectable loss". The full 1,319-question split is vendored at
  `scripts/gsm8k_test.json` and would tighten this to ±0.6 pt; nobody has run it yet.
- **Accuracy was never measured on the shipping stack's exact head.** 145/150 came from the
  native-BF16-head arm. Production runs the quantized head in place, which should be
  equivalent-or-better, but that is an inference, not a measurement.
- **Acceptance lengths** are means over each run's decode batches, and the arms cover
  slightly different phase mixes, so treat them as approximate. A and B are directly
  comparable (169 vs 168 batches on the same suite).
- **Arm A's footprint** was sampled after vision but before its GSM8K run; B and C after.
- **The two result sets are not interchangeable.** A/B/C/P used a 200k KV cap and the
  upstream-tree overlay; the standard suite uses the shipping 262k configuration. Compare
  within a set, not across.
- **vLLM is not benchmarked here.** Its DFlash2 PR is still open, and its reviewers flagged
  the same quantized-head gap that blocks NVFP4.
- **The Harbor solver path is verified at the protocol and network layers only** — no real
  trial has executed against the local model end to end.
- **Reboot survival is verified by construction, not by rebooting.** Linger, units, and the
  `docker wait` lifecycle were each checked; the machine has not actually been restarted.
