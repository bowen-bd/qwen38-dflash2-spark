# Qwen3.8-27B NVFP4 + DFlash2 on a DGX Spark (GB10)

Measured 2026-08-19 on a single NVIDIA DGX Spark (GB10, 121.7 GiB unified memory,
CUDA 13.0), serving `RadixArk/Qwen3.8-27B-NVFP4` under SGLang.

DFlash2 landed upstream in SGLang on **2026-08-19** ([PR #35371]). Its published
numbers are a **BF16 target on an H200** — Hopper, which has no FP4 tensor cores — so
the NVFP4 path had never been exercised. This repo is what happened when we ran it on
one: **two blocking bugs**, a fix for the more serious one, and a three-arm benchmark
that separates the drafter's contribution from the cost of the stack change needed to
get it.

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
- **Production config is faster than any arm we built**: **+40% prose, +79% technical,
  +67% math** over the DSpark baseline, **165.7 tok/s at 8 streams**, **145/150** on
  GSM8K, in **6.8 GB less memory** and a **4× faster start**. See [deploy/](deploy/).

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

| | **A** DSpark baseline | **C** our DFlash2 | **P** production | P vs A |
|---|---|---|---|---|
| Free-form prose | 17.61 | 22.64 | **24.59** | **+40%** |
| Technical explanation | 20.09 | 30.13 | **35.95** | **+79%** |
| Math / structured | 32.12 | 39.12 | **53.59** | **+67%** |
| Code generation | 29.77 | 38.12 | **38.91** | **+31%** |
| Code editing | 40.02 | 43.17 | **49.12** | **+23%** |
| JSON structured | 40.31 | 43.70 | **49.83** | **+24%** |
| Single-stream decode | 16.11 | 23.54 | **27.26** | **+69%** |
| 8 concurrent (agg) | 104.74 | 154.30 | **165.65** | **+58%** |
| Prefill @22.6K | 2786 | 3037 | **3210** | +15% |
| Vision, per chart | 3.0 s | 2.6 s | **2.8 s** | +7% |
| **GSM8K (150 q)** | 143/150 | 144/150 | **145/150** | **+2** |
| Footprint | 50,675 MiB | 53,733 MiB | **43,750 MiB** | **−6.8 GB** |
| Time to ready | 550 s | 510 s | **130 s** | **4× faster** |

It is faster than every arm we built, on less memory, with better accuracy, and starts
four times quicker. Three reasons, none of them the drafter: the head is never
dequantized (−2.5 GB and −1.9 GB of streamed weights per pass), the mamba state is FP8
rather than BF16 (0.95+0.95 GB vs 1.91+1.91), and torch.compile is off with no
measurable cost. The 8-stream figure is **after** fixing the GDN slot clamp — before it,
the same config managed only 107.4 tok/s because concurrency was capped at 6.

### Multi-Condition Standard Benchmark Suite

To thoroughly characterize the live serving performance under varying operational conditions, we run a 6-axis standard benchmark suite (`scripts/bench-standard-suite.py`). Full tabulated logs are in [results/standard-benchmark-results.md](results/standard-benchmark-results.md) and [results/standard-benchmark-results.json](results/standard-benchmark-results.json):

1. **Workload Predictability (Batch=1)**: Speculative decode throughput varies from **25.09 tok/s** on high-entropy prose to **59.93 tok/s** on step-by-step math (**2.39× spread**; code generation at **44.08 tok/s**, code refactoring at **52.02 tok/s**, JSON at **54.22 tok/s**).
2. **Concurrency & Saturation**: Scales from **39.83 tok/s** at 1 stream to a peak of **198.63 aggregate tok/s** at 8 concurrent streams (24.8 tok/s per user, 266 ms TTFT). At 16 streams, throughput holds at **194.28 agg tok/s** with queueing.
3. **Context / Prompt Length Scaling**: Prefill throughput reaches **2,000 – 3,260 prompt tok/s** for medium-to-long prompts (TTFT scales smoothly from 0.22 s at 128 tokens to 18.4 s at 38k prompt tokens).
4. **Radix Prefix Caching**: Reusing a ~4.8k token document context reduces TTFT from **2,286 ms** (cold) down to **201 ms** (warm), providing an **11.33× TTFT speedup**.
5. **Decode Horizon Stability**: Generation rate remains flat across sequence lengths (**35.05 tok/s** at 64 tokens, **31.66 tok/s** at 1024 tokens; ITL ~31 ms).
6. **Reasoning / Thinking Mode**: DFlash2 maintains high throughput during active chain-of-thought generation (**52.22 tok/s** with `enable_thinking=True` vs **53.44 tok/s** with `enable_thinking=False`).

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

Three ways to give it a dense head:

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

- **n=150 on GSM8K** bounds accuracy at roughly ±1.8 pt. It does not rule out a sub-2-point
  regression; the full 1,319-question set would tighten that to ±0.6 pt.
- **Acceptance lengths** are means over each run's decode batches, and the arms cover
  slightly different phase mixes, so treat them as approximate. A and B are directly
  comparable (169 vs 168 batches on the same suite).
- **Arm A's footprint** was sampled after vision but before its GSM8K run; B and C were
  sampled after GSM8K.
- **vLLM is not benchmarked here.** Its DFlash2 PR is still open, and its reviewers flagged
  the same quantized-head gap.
- The upstream-tree overlay is a **measurement vehicle, not a deployment**. For production,
  wait for the fork to rebase onto post-2026-08-19 upstream, which should keep the drafter
  gain and drop most of the stack tax.
