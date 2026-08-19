#!/usr/bin/env bash
# Verify the local Qwen server actually speaks every protocol our clients need.
#
# Each client uses a different wire format, and they fail in different ways:
#   /v1/chat/completions  generic agents
#   /v1/messages          Claude Code   (needs real input_tokens for auto-compaction)
#   /v1/responses         Codex         (needs ids on replayed reasoning items)
# Tool calling is checked separately because an agent that cannot call tools is
# useless even when plain chat works.
#
#   clients/smoke-test.sh [host:port] [model]
set -uo pipefail

ENDPOINT="${1:-127.0.0.1:8888}"
MODEL="${2:-qwen3.8-27b-sglang}"
BASE="http://${ENDPOINT}"
pass=0; fail=0

ok ()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; pass=$((pass+1)); }
bad ()  { printf '  \033[31mFAIL\033[0m  %s -- %s\n' "$1" "$2"; fail=$((fail+1)); }
note () { printf '        %s\n' "$1"; }

jqp () { python3 -c "import json,sys;d=json.load(sys.stdin);print(eval(sys.argv[1]))" "$1" 2>/dev/null; }

echo "Target: ${BASE}  model=${MODEL}"
echo

# ---------- reachability ----------
code=$(curl -s --max-time 10 -o /dev/null -w '%{http_code}' "${BASE}/v1/models")
if [ "$code" = "200" ]; then ok "/v1/models reachable"; else bad "/v1/models" "http=$code"; fi

served=$(curl -s --max-time 10 "${BASE}/v1/models" | jqp "d['data'][0]['id']")
if [ "$served" = "$MODEL" ]; then ok "served model name matches (${served})"
else bad "served model name" "server says '${served}', configs say '${MODEL}'"; fi

# ---------- OpenAI chat (generic agents) ----------
r=$(curl -s --max-time 120 -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\":\"${MODEL}\",\"max_tokens\":16,\"temperature\":0,
  \"chat_template_kwargs\":{\"enable_thinking\":false},
  \"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ready\"}]}")
if echo "$r" | grep -qi ready; then ok "/v1/chat/completions"
else bad "/v1/chat/completions" "$(echo "$r" | head -c 160)"; fi

# ---------- OpenAI tool calling ----------
r=$(curl -s --max-time 120 -X POST "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\":\"${MODEL}\",\"max_tokens\":128,\"temperature\":0,
  \"chat_template_kwargs\":{\"enable_thinking\":false},
  \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_lattice\",
    \"description\":\"Lattice constant of a cubic element in angstrom\",
    \"parameters\":{\"type\":\"object\",\"properties\":{\"element\":{\"type\":\"string\"}},\"required\":[\"element\"]}}}],
  \"tool_choice\":\"auto\",
  \"messages\":[{\"role\":\"user\",\"content\":\"Use the tool to get the lattice constant of Cu.\"}]}")
if echo "$r" | grep -q '"tool_calls"'; then ok "/v1/chat/completions tool calling"
else bad "tool calling (openai)" "no tool_calls in response"; note "$(echo "$r" | head -c 160)"; fi

# ---------- Anthropic messages (Claude Code) ----------
r=$(curl -s --max-time 120 -X POST "${BASE}/v1/messages" -H 'Content-Type: application/json' -d "{
  \"model\":\"${MODEL}\",\"max_tokens\":24,\"system\":\"Be terse.\",
  \"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ready\"}]}")
itok=$(echo "$r" | jqp "d['usage']['input_tokens']")
if [ -n "$itok" ] && [ "$itok" != "0" ]; then ok "/v1/messages reports input_tokens=${itok}"
else bad "/v1/messages input_tokens" "got '${itok}' -- Claude Code auto-compaction needs this"; fi

# Claude Code puts system turns inside messages; vLLM rejected that, SGLang must not.
code=$(curl -s --max-time 120 -o /dev/null -w '%{http_code}' -X POST "${BASE}/v1/messages" \
  -H 'Content-Type: application/json' -d "{
  \"model\":\"${MODEL}\",\"max_tokens\":16,
  \"messages\":[{\"role\":\"system\",\"content\":\"Be terse.\"},{\"role\":\"user\",\"content\":\"hi\"}]}")
if [ "$code" = "200" ]; then ok "/v1/messages accepts system inside messages (no shim needed)"
else bad "/v1/messages system-in-messages" "http=$code -- a shim would be required"; fi

# ---------- Responses (Codex) ----------
r=$(curl -s --max-time 120 -X POST "${BASE}/v1/responses" -H 'Content-Type: application/json' -d "{
  \"model\":\"${MODEL}\",\"input\":\"Reply with the single word: ready\",\"max_output_tokens\":64}")
if echo "$r" | grep -q '"object":\s*"response"'; then ok "/v1/responses"
else bad "/v1/responses" "$(echo "$r" | head -c 160)"; fi

# Codex replays reasoning items; they must carry ids or the turn after the first
# tool call dies with a wall of validation errors.
if echo "$r" | grep -q '"type":\s*"reasoning"'; then
  rid=$(echo "$r" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(next((o.get('id') for o in d.get('output',[]) if o.get('type')=='reasoning'), ''))" 2>/dev/null)
  if [ -n "$rid" ]; then ok "/v1/responses reasoning items carry ids (${rid:0:12}...)"
  else bad "/v1/responses reasoning ids" "missing -- Codex replay will fail"; fi
fi

# ---------- container reachability (Harbor agents) ----------
gw=$(ip -4 route show default 2>/dev/null | awk '/docker0/{print $3}')
gw="${gw:-172.17.0.1}"
if command -v docker >/dev/null 2>&1; then
  port="${ENDPOINT##*:}"
  if docker run --rm --network bridge curlimages/curl:latest \
       -s --max-time 10 -o /dev/null -w '%{http_code}' "http://${gw}:${port}/v1/models" 2>/dev/null | grep -q 200; then
    ok "reachable from a container at ${gw}:${port} (Harbor solver path)"
  else
    bad "container reachability" "http://${gw}:${port} unreachable from a bridge-network container"
    note "Harbor agents run in containers; 127.0.0.1 there is the container, not the host."
  fi
fi

echo
echo "  ${pass} passed, ${fail} failed"
[ "$fail" -eq 0 ]
