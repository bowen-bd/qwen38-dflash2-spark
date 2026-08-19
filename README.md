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
- **Production config is faster than any arm we built**: **+43% prose, +94% technical,
  +87% math, +48% code gen** over the DSpark baseline, **198.6 tok/s aggregate at 8 streams**,
  **145/150** on GSM8K, in **6.8 GB less memory** and a **4× faster start**. Full
  characterisation in the [standard suite](#multi-condition-standard-benchmark-suite);
  deployment in [deploy/](deploy/).

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

## The three arms

Everything held constant except where noted: same NVFP4 body, same 200,000-token KV
cap, same prompts, idle machine, thinking disabled, temperature 0.

| | drafter | engine | `lm_head` | `cuda-graph-max-bs` |
|---|---|---|---|---|
| **A** | DSpark | fork image (`lmsysorg/sglang:qwen38-27b`, built 2026-08-14) | NVFP4 | 4 |
| **B** | DSpark | upstream `main` tree overlay | BF16 (dequantized) | 8 |
| **C** | **DFlash2** | upstream `main` tree overlay | BF16 (dequantized) | 8 |

**A** is the production baseline. **C** is the DFlash2 candidate. **B** is the control that
makes the comparison honest: DFlash2 cannot run on A's stack, so C-vs-A mixes the drafter
with the stack change. B changes *only* the stack, so **C vs B isolates the drafter**.

## Results

Batch-1 decode, tok/s:

| Workload | A | B | **C** | C vs B<br>*(drafter)* | C vs A<br>*(end-to-end)* |
|---|---|---|---|---|---|
| Free-form prose | 17.61 | 14.87 | **22.64** | **+52%** | +29% |
| Technical explanation | 20.09 | 19.51 | **30.13** | **+54%** | +50% |
| Math / structured | 32.12 | 25.50 | **39.12** | **+53%** | +22% |
| Code generation | 29.77 | 25.73 | **38.12** | **+48%** | +28% |
| Code editing (rewrite) | 40.02 | 29.70 | **43.17** | **+45%** | +8% |
| JSON structured | 40.31 | 31.09 | **43.70** | **+41%** | +8% |

| | A | B | **C** | C vs B | C vs A |
|---|---|---|---|---|---|
| Single-stream decode | 16.11 | 15.27 | **23.54** | **+54%** | +46% |
| 1 concurrent (agg) | 17.83 | 16.53 | **30.36** | +84% | +70% |
| 2 concurrent | 38.01 | 30.32 | **56.00** | +85% | +47% |
| 4 concurrent | 65.30 | 59.02 | **88.03** | +49% | +35% |
| **8 concurrent** | 104.74 | 96.11 | **154.30** | **+61%** | +47% |
| Vision, per chart | 3.0 s | 3.8 s | **2.6 s** | +32% | +13% |
| TTFT | 208–220 ms | 228–233 ms | 232–234 ms | ~0 | slightly worse |
| Prefill @1.4K/5.7K/22.6K | 2545/3227/2786 | 2498/3208/3058 | 2436/3148/3037 | ~0 | ~0 |
| **Mean acceptance length** | **3.32** | **3.29** | **5.03** | **+53%** | +51% |
| **GSM8K (150 q)** | 143/150 | **144/150** | **144/150** | **0** | +1 |
| Footprint, steady state | 50,675 MiB | 52,735 MiB | 53,733 MiB | +998 MiB | +3,058 MiB |

### Production configuration — what you should actually run

Arm **P** is the [MiaAI-Lab][mia] stack (their image, their in-place quantized head, FP8
mamba state) plus our three corrections. Same prompts, same 200k KV cap, idle machine.

| | **A** DSpark baseline | **C** our DFlash2 | **P** production (latest) | P vs A |
|---|---|---|---|---|
| Free-form prose | 17.61 | 22.64 | **25.09** | **+42%** |
| Technical explanation | 20.09 | 30.13 | **39.00** | **+94%** |
| Math / structured | 32.12 | 39.12 | **59.93** | **+87%** |
| Code generation | 29.77 | 38.12 | **44.08** | **+48%** |
| Code editing | 40.02 | 43.17 | **52.02** | **+30%** |
| JSON structured | 40.31 | 43.70 | **54.22** | **+34%** |
| Single-stream decode | 16.11 | 23.54 | **39.83** | **+147%** |
| 8 concurrent (agg) | 104.74 | 154.30 | **198.63** | **+90%** |
| Prefill @22.6K / 32K | 2786 | 3037 | **3262** | +17% |
| Vision, per chart | 3.0 s | 2.6 s | **2.8 s** | +7% |
| **GSM8K (150 q)** | 143/150 | 144/150 | **145/150** | **+2** |
| Footprint | 50,675 MiB | 53,733 MiB | **45,582 MiB** | **−5.1 GB** |
| Time to ready | 550 s | 510 s | **130 s** | **4× faster** |

It is faster than every arm we built, on less memory, with better accuracy, and starts
four times quicker. Three reasons, none of them the drafter: the head is never
dequantized (−2.5 GB and −1.9 GB of streamed weights per pass), the mamba state is FP8
rather than BF16 (0.95+0.95 GB vs 1.91+1.91), and torch.compile is off with no
measurable cost. The 8-stream figure is **after** fixing the GDN slot clamp — before it,
the same config managed only 107.4 tok/s because concurrency was capped at 6.

### Multi-Condition Standard Benchmark Suite

Measured with `scripts/bench-standard-suite.py`. Full raw logs and data are in
[results/standard-benchmark-results.md](results/standard-benchmark-results.md) and
[results/standard-benchmark-results.json](results/standard-benchmark-results.json).

**This supersedes the A/B/C/P tables above for "how fast is it".** Those answer a
different question — a *controlled* comparison isolating the drafter from the stack
change it requires, run at a 200k KV cap on an earlier config. This suite exercises the
shipping configuration (262k KV pool, 8 concurrent slots) across more conditions, which
is why every number here is higher.

#### 1. Workload Diversity & Predictability (Single-Stream, Batch = 1)
*400 completion tokens per workload, median over 3 runs*

| Workload Domain | Decode (tok/s) | TTFT (ms) | Inter-Token Latency (ms) |
| :--- | :---: | :---: | :---: |
| **Free-form Creative Prose** | **25.09** | 188.3 | 39.95 |
| **Technical Explanation** | **39.00** | 219.9 | 25.71 |
| **Python Code Generation** | **44.08** | 217.4 | 22.74 |
| **Code Editing / Refactor** | **52.02** | 201.4 | 19.27 |
| **Structured JSON Output** | **54.22** | 196.5 | 18.49 |
| **Step-by-Step Math / Reasoning** | **59.93** | 191.7 | 16.73 |

#### 2. Concurrency & Throughput Scaling (300 tokens / stream)

| Concurrency ($C$) | Aggregate Throughput | Per-Stream Speed | Avg TTFT | Total Wall Time |
| :---: | :---: | :---: | :---: | :---: |
| **1** | **39.83 tok/s** | 39.83 tok/s | 146.9 ms | 7.53 s |
| **2** | **63.58 tok/s** | 31.79 tok/s | 842.6 ms | 9.44 s |
| **4** | **118.59 tok/s** | 29.65 tok/s | 227.7 ms | 10.12 s |
| **8** | **198.63 tok/s** | 24.83 tok/s | 266.3 ms | 12.08 s |
| **16** | **194.28 tok/s** | 12.14 tok/s | 6,352.4 ms | 24.71 s |

**Throughput saturates at 8, and 16 is worse than useless.** Aggregate goes flat
(198.6 → 194.3) while average TTFT degrades **24×** (266 ms → 6,352 ms), because
`--max-running-requests 8` admits eight and the rest queue. To go beyond eight you must
raise *both* that and the GDN pool (`concurrency × 5` slots) — raising one alone silently
clamps you back.

#### 3. Context / Prompt Scaling (Prefill Throughput & TTFT)

| Prompt Context | TTFT | Prefill Speed | Generation Speed |
| :---: | :---: | :---: | :---: |
| **128 tokens** | **0.220 s** | 599.9 tok/s | 19.8 tok/s |
| **1,024 tokens** | **0.324 s** | 1,845.1 tok/s | 19.7 tok/s |
| **4,096 tokens** | **1.069 s** | 2,216.6 tok/s | 19.6 tok/s |
| **16,384 tokens** | **4.640 s** | 2,037.3 tok/s | 17.2 tok/s |
| **32,768 tokens** | **5.802 s** | 3,261.6 tok/s | 25.6 tok/s |
| **65,536 tokens** | **18.424 s** | 2,055.3 tok/s | 25.1 tok/s |

#### 4. Radix Prefix Caching & Reasoning Modes

| Condition / Mode | Context / Tokens | TTFT | Decode Speed | Note |
| :--- | :---: | :---: | :---: | :--- |
| **Cold Request** (0% Cache) | 4,787 prompt | 2,286.3 ms | 19.6 tok/s | Baseline full prefill |
| **Warm Request** (100% Cache) | 4,789 prompt | **201.8 ms** | 19.8 tok/s | **11.33× faster TTFT** |
| **Thinking OFF** | 512 completion | 204.6 ms | **53.44 tok/s** | Direct output |
| **Thinking ON** | 512 completion | 213.5 ms | **52.22 tok/s** | Sustained speed during CoT |

### Reading the numbers

**Prefill is flat and acceptance explains everything.** Speculative decoding only touches
decode, and prefill is unchanged across all three arms — the sanity check that nothing else
moved. The entire speedup is acceptance length: 3.29 → 5.03 tokens per verification pass.

**The stack tax is real: −3% to −26% (B vs A).** Two causes are bundled: the dense BF16
head streams 2.54 GB per forward pass instead of NVFP4's 0.64 GB, and decode here is
bandwidth-bound; plus the upstream tree lacks the fork's GB10 tuning. The decisive evidence
that this is *not* a drafter effect is that **acceptance is identical across builds**
(3.32 vs 3.29) — DSpark speculates the same on both, so the loss is pure per-pass cost.

**So C-vs-A understates DFlash2.** The +8% on code editing is the drafter's +45% minus the
stack's −26%. Most of that tax should disappear once the fork rebases onto a post-2026-08-19
upstream, leaving the +41–54%.

**Accuracy is unchanged.** B and C both score 144/150 — with the stack held constant the
drafter changes accuracy by exactly zero questions, as losslessness predicts. The 143→144
between A and B is the head dtype, not the drafter. At n=150 the standard error is ±1.8 pt,
so this bounds the loss at roughly ±2 pt, not tighter.

**Memory: +3.0 GB total, ~1.0 GB of it the drafter.** The drafter figure is confirmed twice
(drafter `mem usage` 3.03 → 4.04 GB; nvidia-smi B→C +998 MiB). The remaining ~2.06 GB mixes
the BF16 head with the upstream build and cannot be cleanly split — the per-model `mem usage`
counter is ±0.6 GB noisy (arms B and C load the *same* target and report 23.00 vs 22.43 GB).

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

Benchmarks:

```bash
scripts/bench-dflash2.sh myrun        # workloads, concurrency, prefill, vision, GSM8K
OUT=/tmp/runs scripts/bench-dflash2.sh myrun
URL=http://127.0.0.1:8888/v1 MODEL=qwen3.8-27b-sglang scripts/bench-dflash2.sh myrun
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
