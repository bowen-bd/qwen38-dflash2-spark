# The two DFlash2 failures on an NVFP4 target

Both hit on Qwen3.8-27B NVFP4 + DGX Spark GB10, SGLang upstream main (post PR #35371).

## 1. A quantized lm_head blocks DFlash2 entirely

Model loads fine (`type=DFlash2DraftModel`), then the first decode raises:

```
```

The guard is a pure dtype test (`srt/speculative/dflash_utils.py:678`):

```python
def is_dense_head_weight(weight: Any) -> bool:
    """Whether an lm_head weight can be read as a plain matrix. A quantized head
    stores packed values, which a dense matmul would read as if they were
    activations."""
    return weight is not None and weight.dtype in _DENSE_HEAD_DTYPES

```

Upstream *does* screen the head in `_maybe_build_draft_sampler` (line 487) and returns
`_eager("quantized lm_head")`, so it is meant to degrade gracefully -- but the eager
fallback still routes through `compute_candidates`, which raises. Upstream also has a
quantized-head path, `_greedy_sample_from_quantized_head` (line 1069), but only for
DFlash **v1**'s greedy sampler; the v2 selector never got one.

Worked around by `scripts/patch_lmhead_nvfp4.py` (dequantize) or
`scripts/swap_native_head.py` (drop in the official FP8 checkpoint's native BF16 head).

## 2. max-running-requests > cuda-graph-max-bs crashes DFlash2

Survived 1/2/4 concurrent streams, died on the first batch of 8:

```
    self._draft_sampler.stage_sampling_params(
  File "/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_worker_v2.py", line 230, in stage_sampling_params
    self.greedy_mask[:bs].copy_(
RuntimeError: The size of tensor a (4) must match the size of tensor b (8) at non-singleton dimension 0
```

`_SelectorDraftSampler` allocates `greedy_mask`/`temperatures`/`candidate_out` at
`cuda_graph_max_bs` -- addresses baked into the captured graph -- then
`stage_sampling_params()` indexes them with the live batch size. Upstream's own
`run.sh` ships `--cuda-graph-max-bs 4` with `--max-running-requests 8`, which is
harmless for DSpark/MTP and fatal here.

Worked around with `CG_MAX_BS=8`.
