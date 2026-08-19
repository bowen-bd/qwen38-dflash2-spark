#!/usr/bin/env bash
# Reproduce hasso5703/dgx-spark-qwen38's SGLang + NVFP4 + DSpark config on GB10.
#
# Flags are verbatim from that repo's run.sh. The one deviation: the checkpoints
# are passed as local paths instead of HF repo ids, because this host's Xet
# endpoint is unreachable so they were fetched with pardl.py rather than
# snapshot_download. Same pinned revisions either way.
#
#   model  RadixArk/Qwen3.8-27B-NVFP4  @ 52d1adc5f38aa5ebf099c29ed7025ba34cfbb854
#   draft  RadixArk/Qwen3.8-27B-DSpark @ 923ed3a8572615643f0137e424e4ce4edd7f1cda
#   image  lmsysorg/sglang@sha256:febfb971...  (= :qwen38-27b, 2026-08-15)
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1}"
MODEL_DIR="${MODEL_DIR:-/home/bdeng/llm/radixark-nvfp4}"
DRAFT_DIR="${DRAFT_DIR:-/home/bdeng/llm/radixark-dspark}"
CONFIG_DIR="${CONFIG_DIR:-/home/bdeng/llm/sglang}"
PORT="${PORT:-30000}"
MEM_FRAC="${MEM_FRAC:-0.50}"
SPEC="${SPEC:-DSPARK}"   # DSPARK, or EAGLE for the checkpoint's own MTP head
MAX_TOTAL="${MAX_TOTAL:-}"   # optional cap on KV tokens; see the note below
# DFlash2 needs this >= --max-running-requests. Its selector sampler allocates
# greedy_mask/temperatures/candidate_out at cuda_graph_max_bs (the addresses are
# baked into the captured graph) but stage_sampling_params() indexes them with the
# real running batch, so bs > cuda_graph_max_bs dies with
#   dflash_worker_v2.py:230  size of tensor a (4) must match tensor b (8)
# Upstream run.sh ships 4 vs 8, which is fine for DSpark/MTP and fatal for DFlash2.
CG_MAX_BS="${CG_MAX_BS:-4}"
# Point this at an upstream sglang python tree to run DFlash2, which landed
# upstream 2026-08-19 (PR #35371) -- four days after this image was built. The
# image's own tree only registers DFlashDraftModel/DFlashLagunaForCausalLM, so the
# DFlash2 checkpoint (architectures: ["DFlash2DraftModel"]) cannot load without it.
# Overlaying the whole tree keeps engine build constant across arms; overlaying
# only the three PR files does not work -- upstream spec_utils needs
# mamba_extra_buffer_enabled, which this fork does not have.
UPSTREAM_TREE="${UPSTREAM_TREE:-}"

mkdir -p "$CONFIG_DIR/sglang-cache"

# GB10 unified memory: --mem-fraction-static is a share of the WHOLE pool, and
# SGLang derives "already used" from (total - free) -- which on unified memory
# includes the desktop and every other container, not just SGLang. So the budget
# has to clear everyone else's footprint before any of it reaches the KV cache:
#
#   KV = mem_fraction_static * total  -  used_by_others  -  weights
#
# That is why the upstream repo's 0.50 works on an idle Spark and fails here.
WEIGHTS_GIB=${WEIGHTS_GIB:-34.2}          # measured: target 30.7 + DSpark draft 3.2
TOTAL_GIB=$(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo)
AVAIL_GIB=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
USED_GIB=$(awk -v t="$TOTAL_GIB" -v a="$AVAIL_GIB" 'BEGIN{printf "%.1f", t-a}')
BUDGET=$(awk -v t="$TOTAL_GIB" -v f="$MEM_FRAC" 'BEGIN{printf "%.1f", t*f}')
KV_GIB=$(awk -v b="$BUDGET" -v u="$USED_GIB" -v w="$WEIGHTS_GIB" 'BEGIN{printf "%.1f", b-u-w}')
FREE_AFTER=$(awk -v t="$TOTAL_GIB" -v u="$USED_GIB" -v w="$WEIGHTS_GIB" -v k="$KV_GIB" \
             'BEGIN{printf "%.1f", t-u-w-k}')
echo "unified pool : ${TOTAL_GIB} GiB total, ${AVAIL_GIB} GiB available, ${USED_GIB} GiB held by others"
echo "sglang budget: ${BUDGET} GiB (mem-fraction-static=${MEM_FRAC})"
echo "  -> weights ${WEIGHTS_GIB} GiB, KV ${KV_GIB} GiB, ${FREE_AFTER} GiB left free"
# Left uncapped, sglang spends the whole remaining budget on KV: measured here it
# took 744,276 tokens = 36.9 GiB (22.7 FP8 attention + 14.2 BF16 mamba state), which
# left only 10.4 GiB free once torch.compile and graph capture were resident. Nothing
# we serve needs a 744K-token pool, and pool size does not affect decode rate, so cap
# it. 36.92 GiB / 744,276 tokens = 49.6 KiB per token, measured on this checkpoint.
if [ -n "$MAX_TOTAL" ]; then
  KV_GIB=$(awk -v c="$MAX_TOTAL" 'BEGIN{printf "%.1f", c*36.92/744276}')
  FREE_AFTER=$(awk -v t="$TOTAL_GIB" -v u="$USED_GIB" -v w="$WEIGHTS_GIB" -v k="$KV_GIB" \
               'BEGIN{printf "%.1f", t-u-w-k}')
  echo "  -> --max-total-tokens ${MAX_TOTAL} caps KV to ${KV_GIB} GiB, ${FREE_AFTER} GiB left free"
fi

if awk -v k="$KV_GIB" 'BEGIN{exit !(k < 4)}'; then
  echo "REFUSING: that leaves ${KV_GIB} GiB for the KV cache. Raise MEM_FRAC or free memory." >&2
  exit 1
fi
if awk -v f="$FREE_AFTER" 'BEGIN{exit !(f < 8)}'; then
  echo "REFUSING: only ${FREE_AFTER} GiB would stay free; an OOM here freezes the host." >&2
  exit 1
fi

# Upstream hard-fails here ("needs the machine to itself"). We only warn: the
# budget above is computed against real free memory, so a small co-tenant is
# accounted for rather than assumed away. Set STRICT_GPU=1 for upstream behaviour.
BUSY="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
if [ -n "$BUSY" ]; then
  echo "NOTE: other GPU processes are running -" >&2
  echo "$BUSY" | sed 's/^/  /' >&2
  [ "${STRICT_GPU:-0}" = "1" ] && { echo "STRICT_GPU=1 set, refusing." >&2; exit 1; }
fi

SPEC_ARGS=(--speculative-algorithm DSPARK
           --speculative-draft-model-path /draft
           --speculative-dspark-block-size 7
           --speculative-draft-model-quantization unquant)
if [ "$SPEC" = "DFLASH2" ]; then
  # incoai/Qwen3.8-27B-DFlash2: block_size 8 in the checkpoint config, so 7 draft
  # tokens per verification step. Same DFLASH algorithm name as v1 -- the drafter
  # architecture in config.json selects v2.
  SPEC_ARGS=(--speculative-algorithm DFLASH
             --speculative-draft-model-path /draft
             --speculative-num-draft-tokens 8
             --speculative-draft-model-quantization unquant)
fi
if [ "$SPEC" = "EAGLE" ]; then
  # The one-flag swap the repo documents for prose-heavy work: the checkpoint's
  # own MTP head instead of the separate DSpark drafter.
  SPEC_ARGS=(--speculative-algorithm EAGLE --speculative-num-steps 3
             --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
fi

MAXTOK_ARG=()
[ -n "$MAX_TOTAL" ] && MAXTOK_ARG=(--max-total-tokens "$MAX_TOTAL")

UPSTREAM_ARG=()
[ -n "$UPSTREAM_TREE" ] && UPSTREAM_ARG=(-v "$UPSTREAM_TREE":/sgl-workspace/sglang/python/sglang:ro)

TPL_ARG=()
[ -f "$CONFIG_DIR/chat-template-sglang.jinja" ] && TPL_ARG=(--chat-template /out/chat-template-sglang.jinja)

docker rm -f qwen38-sglang-run >/dev/null 2>&1 || true
exec docker run --rm --name qwen38-sglang-run --gpus all \
  --memory 100g --memory-swap 100g --shm-size 16g --network host --ipc=host \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -v "$CONFIG_DIR/sglang-cache":/cache \
  -v "$MODEL_DIR":/model:ro \
  -v "$DRAFT_DIR":/draft:ro \
  -v "$CONFIG_DIR":/out \
  "${UPSTREAM_ARG[@]}" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --trust-remote-code --model-path /model --tp-size 1 \
    --served-model-name qwen3.8-27b \
    --mem-fraction-static "$MEM_FRAC" \
    --attention-backend flashinfer --chunked-prefill-size 8192 \
    --disable-prefill-cuda-graph --cuda-graph-max-bs "$CG_MAX_BS" \
    "${SPEC_ARGS[@]}" "${MAXTOK_ARG[@]}" \
    --mamba-radix-cache-strategy extra_buffer_lazy --mamba-ssm-dtype bfloat16 \
    --max-mamba-cache-size 96 --max-running-requests 8 \
    --enable-torch-compile --torch-compile-max-bs 4 \
    --num-continuous-decode-steps 2 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    "${TPL_ARG[@]}" \
    --host 0.0.0.0 --port "$PORT"
