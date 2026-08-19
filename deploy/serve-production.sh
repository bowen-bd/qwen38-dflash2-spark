#!/usr/bin/env bash
# Production launcher for Qwen3.8-27B NVFP4 + DFlash2 on a DGX Spark (GB10).
#
# The serving stack is MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark -- it is better
# tuned for this box than anything we wrote (FP8 KV, GDN pool sizing, mamba
# memory ratio, CPU pinning, and an in-place quantized lm_head that avoids
# dequantizing 2.5 GB). This wrapper adds only the three things we measured that
# their defaults do not cover:
#
#   1. --max-mamba-cache-size = concurrency * 5.
#      Their start.sh sizes the GDN pool as concurrency * 4, which is right for
#      the extra_buffer_lazy + EAGLE/MTP path. start-dflash.sh switches to
#      extra_buffer with block-8 drafting, which needs 5 slots per request, so
#      MAX_CONCURRENT_REQUESTS=8 silently became 6:
#        "max_running_requests is capped to 6 by the mamba state cache
#         (max_mamba_cache_size=32, 5 state slots per request)"
#      Measured cost of the clamp: 107.4 -> 165.7 tok/s at 8 streams (+54%).
#
#   2. --max-total-tokens. Uncapped, SGLang sized a 744k-token pool (36.9 GiB)
#      and left 10.4 GiB free; on unified memory an OOM freezes the host rather
#      than killing the process. 200k is ample (longest prompt here is 22.6k).
#
#   3. A pre-flight memory guard. This box runs a desktop session holding
#      24-31 GiB, which their 0.90 default does not account for. Upstream
#      reports a hard reboot at 0.95.
#
# Local checkpoints are hardlinked into their HF cache (see deploy/README.md),
# so nothing is re-downloaded.
set -euo pipefail

QWEN_HOME="${QWEN_HOME:-$HOME/llm}"
MIAAI_DIR="${MIAAI_DIR:-$QWEN_HOME/miaai}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TOTAL="${MAX_TOTAL:-200000}"
SLOTS_PER_REQ="${SLOTS_PER_REQ:-5}"
MEM_FRAC="${MEM_FRAC:-0.90}"
MODEL_PATH="${MODEL_PATH:-/root/.cache/huggingface/radixark-nvfp4}"
MIN_FREE_GIB="${MIN_FREE_GIB:-8}"
# Claude Code sends reasoning_effort "max", which the checkpoint's stock template
# rejects outright ("Unexpected reasoning effort max. Supported types are xhigh
# (default), medium, and low.") -> HTTP 500 on every request. MiaAI's start.sh does
# not pass --chat-template, so the stock one is used. Ours maps max/high -> xhigh and
# minimal -> low, and renders mid-conversation system turns as <system-reminder>
# blocks in place, which is what Claude Code sends. Must live under HF_HOME: that is
# the only host directory mounted into the container.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/root/.cache/huggingface/chat-template-sglang.jinja}"

MAMBA_SLOTS=$(( CONCURRENCY * SLOTS_PER_REQ ))

TOTAL_GIB=$(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo)
AVAIL_GIB=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
FREE_AFTER=$(awk -v t="$TOTAL_GIB" -v f="$MEM_FRAC" 'BEGIN{printf "%.1f", t*(1-f)}')
echo "unified pool : ${TOTAL_GIB} GiB total, ${AVAIL_GIB} GiB available"
echo "sglang budget: mem-fraction-static=${MEM_FRAC} -> ${FREE_AFTER} GiB stays free"
echo "GDN pool     : ${CONCURRENCY} concurrent x ${SLOTS_PER_REQ} slots = ${MAMBA_SLOTS}"

# free-after is (1-frac)*total regardless of co-tenants, so this is the real floor.
if awk -v f="$FREE_AFTER" -v m="$MIN_FREE_GIB" 'BEGIN{exit !(f < m)}'; then
  echo "REFUSING: only ${FREE_AFTER} GiB would stay free; an OOM here freezes the host." >&2
  exit 1
fi
if awk -v a="$AVAIL_GIB" 'BEGIN{exit !(a < 45)}'; then
  echo "WARNING: only ${AVAIL_GIB} GiB available right now; weights+KV need ~45 GiB." >&2
  echo "         Close other GPU work or lower CONCURRENCY." >&2
fi

cd "$MIAAI_DIR"
export MAX_CONCURRENT_REQUESTS="$CONCURRENCY"
export DF_TARGET="${DF_TARGET:-nvfp4}"
export DF_EXTRA="--model-path ${MODEL_PATH} --max-total-tokens ${MAX_TOTAL} --max-mamba-cache-size ${MAMBA_SLOTS} --chat-template ${CHAT_TEMPLATE} ${DF_EXTRA:-}"

echo "delegating to ${MIAAI_DIR}/start-dflash.sh"
./start-dflash.sh

# start.sh launches the container with `docker run -d` and returns once the server
# answers ("shell is now free"). Under systemd Type=simple that exit looks like the
# service dying, so systemd fires ExecStop and kills the container it just started.
# Blocking on the container keeps the unit alive for as long as the server runs, and
# lets Restart= react when the container actually dies.
if [ -n "${WAIT_FOR_CONTAINER:-}" ]; then
  exec docker wait "${CONTAINER_NAME:-qwen3.8-27b-sglang}"
fi
