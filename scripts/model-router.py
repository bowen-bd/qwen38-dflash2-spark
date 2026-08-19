#!/usr/bin/env python3
"""Model-routing gateway: local Qwen3.8 when asked for, Anthropic otherwise.

Claude Code has one ANTHROPIC_BASE_URL for the whole session, so the only way to
keep the built-in Anthropic models working *and* add a local model to the picker
is to route by model name. Requests naming a local model go to vLLM; everything
else is forwarded to api.anthropic.com with its headers untouched, so an
existing claude.ai OAuth session keeps working.

Two vLLM-specific fixes are applied on the local path only:
  * /v1/messages  -- hoist `system` turns out of `messages` (Claude Code)
  * /v1/responses -- give replayed `reasoning` items an `id` (Codex)

  python3 model-router.py [port] [local_url] [upstream_url]
"""
import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
LOCAL = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000").rstrip("/")
UPSTREAM = (sys.argv[3] if len(sys.argv) > 3 else "https://api.anthropic.com").rstrip("/")
# Any model id containing one of these routes to the local server.
PASSTHROUGH = os.environ.get("ROUTER_PASSTHROUGH", "").strip() not in ("", "0", "false")
LOCAL_MARKERS = tuple(m.strip().lower() for m in
                      os.environ.get("LOCAL_MODEL_MARKERS", "qwen").split(",") if m.strip())
HOP = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(b.get("text", "") for b in content
                           if isinstance(b, dict) and b.get("type") == "text")
    return ""


def hoist_system(body):
    """vLLM requires `system` top-level; Claude Code sends it inside messages."""
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return 0
    sys_parts, kept = [], []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            t = text_of(m.get("content"))
            if t:
                sys_parts.append(t)
        else:
            kept.append(m)
    if not sys_parts:
        return 0
    existing = body.get("system")
    blocks = []
    if isinstance(existing, str) and existing:
        blocks.append({"type": "text", "text": existing})
    elif isinstance(existing, list):
        blocks.extend(existing)
    blocks.extend({"type": "text", "text": t} for t in sys_parts)
    body["system"] = blocks
    body["messages"] = kept or [{"role": "user", "content": "."}]
    return len(sys_parts)


def id_reasoning(body):
    """vLLM's Responses schema requires an id on every reasoning item."""
    items = body.get("input")
    if not isinstance(items, list):
        return 0
    n = 0
    for i, it in enumerate(items):
        if isinstance(it, dict) and it.get("type") == "reasoning" and not it.get("id"):
            it["id"] = f"rs_shim_{i}"
            n += 1
    return n


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    def _proxy(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""

        target, model, note = UPSTREAM, "", ""
        if raw:
            try:
                body = json.loads(raw)
                model = str(body.get("model") or "")
                if any(m in model.lower() for m in LOCAL_MARKERS):
                    target = LOCAL
                    # The two fixups below exist for vLLM. SGLang needs neither, and
                    # hoist_system is actively worse there: our chat template renders
                    # mid-conversation system turns as <system-reminder> blocks in
                    # place, while hoisting moves them to the top and loses position.
                    # Set ROUTER_PASSTHROUGH=1 when the local backend is SGLang.
                    fixed = (None if PASSTHROUGH
                             else hoist_system(body) if self.path.startswith("/v1/messages")
                             else id_reasoning(body) if self.path.startswith("/v1/responses")
                             else 0)
                    if fixed:
                        note = f" (+{fixed} fixed)"
                    raw = json.dumps(body).encode()
            except (ValueError, TypeError):
                pass  # not JSON we understand -> upstream, verbatim

        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if raw:
            headers["Content-Length"] = str(len(raw))
        req = urllib.request.Request(target + self.path, data=raw or None,
                                     headers=headers, method=method)
        try:
            up = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            up = e
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            payload = json.dumps({"type": "error", "error": {
                "type": "api_error", "message": f"router: {e}"}}).encode()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if model:
            where = "local" if target == LOCAL else "anthropic"
            print(f"  {model} -> {where}{note} [{up.status}]", flush=True)

        self.send_response(up.status)
        streaming = (up.headers.get("Content-Type") or "").startswith("text/event-stream")
        for k, v in up.headers.items():
            if k.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(k, v)
        if not streaming:
            payload = up.read()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                chunk = up.read1(8192) if hasattr(up, "read1") else up.read(8192)
                if not chunk:
                    break
                self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            up.close()

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")

    def do_DELETE(self):
        self._proxy("DELETE")


if __name__ == "__main__":
    print(f"model router :{PORT}  '{'/'.join(LOCAL_MARKERS)}' -> {LOCAL}, all else -> {UPSTREAM}",
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
