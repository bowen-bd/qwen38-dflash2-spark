#!/usr/bin/env bash
# One-command bootstrap for Qwen3.8-27B NVFP4 + DFlash2 on a DGX Spark (GB10).
#
# Serving is delegated to MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark. This script clones
# it, builds the DFlash2 image, fetches the checkpoints, and installs the chat template
# that Claude Code needs. Everything lands under $QWEN_HOME (default ~/llm).
#
#   ./setup.sh                 # clone + build + fetch
#   ./setup.sh --no-download   # skip checkpoints (already have them)
#
# Then:  deploy/serve-production.sh   and   clients/smoke-test.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_HOME="${QWEN_HOME:-$HOME/llm}"
MIAAI_DIR="$QWEN_HOME/miaai"
HF="$MIAAI_DIR/.cache/huggingface"
IMAGE="${IMAGE:-lmsysorg/sglang:qwen38-27b-dflash2}"
TARGET_REPO="RadixArk/Qwen3.8-27B-NVFP4"
DRAFT_REPO="z-lab/Qwen3.8-27B-DFlash2"
DRAFT_REV="50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
DOWNLOAD=1
[ "${1:-}" = "--no-download" ] && DOWNLOAD=0

say () { printf '\n== %s\n' "$1"; }

command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
command -v git    >/dev/null || { echo "git is required" >&2; exit 1; }
mkdir -p "$QWEN_HOME"

say "1/5  serving stack"
if [ -d "$MIAAI_DIR/.git" ]; then
  echo "   already at $MIAAI_DIR"
else
  git clone --quiet https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark.git "$MIAAI_DIR"
  echo "   cloned -> $MIAAI_DIR"
fi
chmod +x "$MIAAI_DIR"/*.sh "$MIAAI_DIR"/patch/*.sh 2>/dev/null || true

say "2/5  DFlash2 image"
# No published SGLang tag ships DFlash2 (merged upstream 2026-08-19, after every tag),
# so it is overlaid onto the pinned qwen38-27b base. Their build script also applies the
# NVFP4 head patch that lets the selector use a quantized lm_head in place.
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "   $IMAGE present"
else
  "$MIAAI_DIR/patch/build-dflash2-image.sh"
fi

say "3/5  checkpoints"
mkdir -p "$HF"
if [ "$DOWNLOAD" = "1" ]; then
  # Target: plain dir under HF_HOME, the only host path mounted into the container.
  if [ -d "$HF/radixark-nvfp4" ] && [ -n "$(ls -A "$HF/radixark-nvfp4" 2>/dev/null)" ]; then
    echo "   target already present"
  elif [ -d "$QWEN_HOME/radixark-nvfp4" ]; then
    mkdir -p "$HF/radixark-nvfp4"
    for f in "$QWEN_HOME"/radixark-nvfp4/*; do ln -f "$f" "$HF/radixark-nvfp4/$(basename "$f")" 2>/dev/null || cp "$f" "$HF/radixark-nvfp4/"; done
    echo "   target hardlinked from $QWEN_HOME/radixark-nvfp4 (no extra disk)"
  else
    echo "   downloading $TARGET_REPO (~21 GB) ..."
    docker run --rm --network host -e HF_HOME=/root/.cache/huggingface \
      -e HF_TOKEN="${HF_TOKEN:-}" -v "$HF":/root/.cache/huggingface "$IMAGE" \
      python3 -c "from huggingface_hub import snapshot_download as d; d('$TARGET_REPO', local_dir='/root/.cache/huggingface/radixark-nvfp4')"
  fi
  # Drafter: exact HF cache layout so start-dflash.sh and sglang both resolve it offline.
  D="$HF/hub/models--${DRAFT_REPO//\//--}"
  if [ -n "$(find -L "$D/snapshots/$DRAFT_REV" -maxdepth 2 -type f -print -quit 2>/dev/null)" ]; then
    echo "   drafter already cached"
  else
    echo "   downloading $DRAFT_REPO @ ${DRAFT_REV:0:7} (~3.6 GB) ..."
    docker run --rm --network host -e HF_HOME=/root/.cache/huggingface \
      -e HF_TOKEN="${HF_TOKEN:-}" -v "$HF":/root/.cache/huggingface "$IMAGE" \
      python3 -c "from huggingface_hub import snapshot_download as d; d('$DRAFT_REPO', revision='$DRAFT_REV')"
  fi
else
  echo "   skipped (--no-download)"
fi

say "4/5  chat template"
# Without this, the stock template rejects the reasoning_effort "max" that Claude Code
# sends and every request 500s. Must live under HF_HOME to be visible in the container.
cp "$HERE/scripts/sglang/chat-template-sglang.jinja" "$HF/chat-template-sglang.jinja"
echo "   installed -> $HF/chat-template-sglang.jinja"

say "5/5  done"
cat <<EOF

   Start:   $HERE/deploy/serve-production.sh
   Verify:  $HERE/clients/smoke-test.sh
   Bench:   $HERE/scripts/bench-dflash2.sh myrun

   Always-on (needs linger, or nothing starts until you log in):
     loginctl enable-linger \$USER
     cp $HERE/deploy/*.service ~/.config/systemd/user/
     systemctl --user daemon-reload
     systemctl --user enable --now qwen38-sglang qwen-router
EOF
