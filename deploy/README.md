# Production deployment

Serving is delegated to [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark][mia], which is
better tuned for GB10 than anything here. This directory adds the parts that repo
does not cover: three measured corrections, a memory guard, a systemd unit, and the
client wiring for Claude Code / Codex / Harbor.

[mia]: https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark

## One-time setup

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark.git ~/llm/miaai
cd ~/llm/miaai && ./patch/build-dflash2-image.sh      # -> lmsysorg/sglang:qwen38-27b-dflash2
```

If you already have the checkpoints locally, hardlink them into their HF cache so
nothing re-downloads (same filesystem, so this costs no disk):

```bash
HF=~/llm/miaai/.cache/huggingface
REV=50307d4c4cde6860d4eee73e2547cd786fe8e8a4          # z-lab DFlash2 drafter pin

# Target: a plain dir under HF_HOME, which is the only thing mounted into the container.
mkdir -p "$HF/radixark-nvfp4"
for f in ~/llm/radixark-nvfp4/*; do ln -f "$f" "$HF/radixark-nvfp4/$(basename $f)"; done

# Drafter: exact HF cache layout so their ensure_cached() and sglang both resolve offline.
D="$HF/hub/models--z-lab--Qwen3.8-27B-DFlash2"
mkdir -p "$D/snapshots/$REV" "$D/refs" && echo "$REV" > "$D/refs/main"
for f in ~/llm/dflash2-drafter/*; do ln -f "$f" "$D/snapshots/$REV/$(basename $f)"; done
```

`incoai/Qwen3.8-27B-DFlash2` and `z-lab/…@50307d4` are the same weights — verified
identical `model.safetensors` size (3,848,817,896 B) and byte-identical `config.json`.

## Run

```bash
deploy/serve-production.sh          # :8888, model qwen3.8-27b-sglang
clients/smoke-test.sh               # verifies all three wire protocols
~/llm/miaai/stop.sh                 # stop
```

Always-on:

```bash
cp deploy/qwen38-sglang.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now qwen38-sglang
```

## What this wrapper changes, and why

**`--max-mamba-cache-size` = concurrency × 5, not × 4.** Their `start.sh` sizes the
GDN state pool at `concurrency × 4`, correct for the `extra_buffer_lazy` + EAGLE/MTP
path. `start-dflash.sh` switches to `extra_buffer` with block-8 drafting, which needs
5 slots per request, so a request for 8 concurrent silently becomes 6:

```
max_running_requests is capped to 6 by the mamba state cache
(max_mamba_cache_size=32, 5 state slots per request)
```

Measured cost: **107.4 → 165.7 tok/s at 8 streams (+54%)**. Worth upstreaming as a
one-line change to `MAMBA_SLOTS_PER_REQ` on the DFlash path.

**`--max-total-tokens 200000`.** Uncapped, SGLang sized a 744,276-token pool
(36.9 GiB) and left 10.4 GiB of host memory free once graph capture was resident.
On unified memory an OOM freezes the whole box. 200k is ample here.

**A pre-flight memory guard.** `free-after` is always `(1−frac) × total` regardless
of co-tenants, so the guard is exact. This box runs a desktop holding 24–31 GiB;
upstream reports a hard reboot at `0.95`.

## Clients

All three protocols are served natively on `:8888` — **no shim**. `anthropic-shim.py`
existed only to work around vLLM's stricter validation; SGLang reports real
`input_tokens`, accepts `system` inside `messages`, and returns `/v1/responses`
reasoning items that already carry ids. `clients/smoke-test.sh` asserts all of it.

| client | config | note |
|---|---|---|
| Claude Code, Qwen only | `clients/claude-qwen.json` | straight at `:8888` |
| Claude Code, **both models** | `clients/claude-router.json` | via `scripts/model-router.py`; switch with `/model` |
| Codex CLI | `clients/codex-config-snippet.toml` | adds a `qwen-local` profile; default model unchanged |
| Harbor solver | `clients/harbor-qwen-local.json` | `OPENAI_BASE_URL=http://172.17.0.1:8888/v1` (docker0 gateway) |

### Selecting the model

**Claude Code.** One `ANTHROPIC_BASE_URL` covers a whole session, so keeping both the
local model and the real Anthropic models means routing by model name:

```bash
ROUTER_PASSTHROUGH=1 python3 scripts/model-router.py 8787 http://127.0.0.1:8888 https://api.anthropic.com &
claude --settings clients/claude-router.json      # /model switches between them
```

Anything containing `qwen` goes local, everything else is forwarded to Anthropic with
its headers untouched — so your existing claude.ai login keeps working. Do **not** set
`ANTHROPIC_AUTH_TOKEN` in that profile; a dummy token breaks the Anthropic half.
`ROUTER_PASSTHROUGH=1` disables the router's vLLM-era request fixups, one of which
(`hoist_system`) would otherwise move Claude Code's mid-conversation system turns to the
top and lose the positioning our chat template preserves.

**Codex.** `clients/codex-config-snippet.toml` appends a provider and profile to
`~/.codex/config.toml`, leaving your default `gpt-5.6-sol` alone:

```bash
codex --profile qwen-local exec "..."
```

**Harbor / terminal-bench-science.** Copy `clients/harbor-qwen-local.json` into
`scripts/atomisticskills/configs/` and select it as the job config:

```bash
OPENAI_API_KEY=local scripts/atomisticskills/run_harbor_job.sh \
  scripts/atomisticskills/configs/qwen38-local.json
```

Two traps worth knowing:

- **Harbor agents run in containers**, so `127.0.0.1` there is the container, not the
  host. Use the docker0 gateway `172.17.0.1`. `smoke-test.sh` checks this path.
- **`run_harbor_job.sh` sets `CODEX_FORCE_AUTH_JSON=1` when `OPENAI_API_KEY` is
  unset**, which uploads your real `~/.codex/auth.json` and authenticates against
  api.openai.com instead of the local server. Export `OPENAI_API_KEY=local`.

## The Claude Code 500

MiaAI's `start.sh` does not pass `--chat-template`, so the checkpoint's stock template is
used -- and it rejects the `reasoning_effort: max` that Claude Code sends:

```
ValueError: Unexpected reasoning effort max. Supported types are xhigh (default), medium, and low.
[Anthropic error response api_error] Internal server error
```

Every Claude Code request 500s. `serve-production.sh` fixes it by passing
`scripts/sglang/chat-template-sglang.jinja`, which maps `max`/`high` -> `xhigh` and
`minimal` -> `low`, and renders mid-conversation system turns as `<system-reminder>`
blocks in place. Copy it under `HF_HOME` -- that is the only host directory mounted
into the container:

```bash
cp scripts/sglang/chat-template-sglang.jinja ~/llm/miaai/.cache/huggingface/
```

Verified end-to-end after the fix: `claude --settings clients/claude-qwen.json -p "..."`
returns a normal completion.

## Thinking is on by default

The checkpoint defaults to `reasoning_effort: xhigh` and will burn tens of thousands
of tokens on trivial prompts. `reasoning_effort: "none"` is **rejected** — only
`low`/`medium`/`xhigh` are accepted. To actually disable it, send
`chat_template_kwargs: {"enable_thinking": false}`.
