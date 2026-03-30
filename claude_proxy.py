#!/usr/bin/env python3
"""
Minimal local HTTP handler for the research dashboard "Ask AI Assistant" panel.

No third-party packages (stdlib only). Binds to 127.0.0.1 — do not expose publicly.

Run from the project root:
    python claude_proxy.py

Environment
-----------
ANTHROPIC_API_KEY
    If unset or empty, POST /api/claude returns HTTP 200 with a JSON placeholder
    ``{"reply": "..."}`` and does not call Anthropic. Set the key to enable live answers.

CLAUDE_PROXY_PORT
    Listen port (default 8765).

ANTHROPIC_MODEL
    Default: claude-3-5-haiku-20241022

HTML / report generation
------------------------
Point the dashboard at this server, e.g. when generating HTML:
    set QQ_AUTO_LOCAL_CLAUDE=1
or explicitly:
    set QQ_CLAUDE_DASHBOARD_ENDPOINT=http://127.0.0.1:8765/api/claude

POST /api/claude
------------------
Content-Type: application/json

Body (dashboard sends):
    {"prompt": "...", "context": {...} | null, "reportContext": {...} | null, "source": "research_dashboard"}
    ``context`` and ``reportContext`` are aliases; must be a JSON object when present.

Response:
    {"reply": "...", "response": "..."}  (same text twice for older vs Flask clients)
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from claude_client import DASHBOARD_COPILOT_SYSTEM, build_dashboard_copilot_user_message
except ImportError:  # pragma: no cover
    DASHBOARD_COPILOT_SYSTEM = (
        "You are a research copilot. Use only the dashboard JSON in the user message. "
        "Answer ONLY the question under === USER QUESTION ===; do not generalize. "
        "Under ~120 words unless the user explicitly asks for more. "
        "No generic headers (Top Line Insight) or bracketed section labels. "
        "List results first, brief justification second. Plain text, no markdown. "
        "If the JSON cannot answer, say so in one sentence."
    )

    def build_dashboard_copilot_user_message(user_question: str, context: dict[str, Any] | None) -> str:
        q = str(user_question or "").strip()
        if context is None:
            ctx = "(No dashboard context JSON was provided.)"
        else:
            try:
                ctx = json.dumps(context, ensure_ascii=False, indent=2, default=str)
            except TypeError:
                ctx = str(context)[:24000]
            if len(ctx) > 24000:
                ctx = ctx[:24000] + "\n…(truncated)"
        return (
            "=== DASHBOARD CONTEXT (JSON) ===\n"
            f"{ctx}\n\n=== USER QUESTION ===\n{q}"
        )

PORT = int(os.environ.get("CLAUDE_PROXY_PORT", "8765"))
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022").strip() or "claude-3-5-haiku-20241022"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

# Shown when the proxy runs but no API key is configured (matches dashboard copy).
NO_KEY_REPLY = (
    "The AI assistant is not connected in this local build. The UI is ready; add ANTHROPIC_API_KEY "
    "and a backend request handler to enable live answers."
)


def _cors_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")


def _send_json(handler: BaseHTTPRequestHandler, status: int, obj: dict[str, Any]) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    _cors_headers(handler)
    handler.end_headers()
    handler.wfile.write(body)


def _call_anthropic(user_text: str, *, system: str | None = None) -> str:
    payload: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": user_text}],
    }
    if system and str(system).strip():
        payload["system"] = str(system).strip()
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        ANTHROPIC_URL,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:1200]
        return f"Anthropic API HTTP {e.code}: {err}"
    except URLError as e:
        return f"Could not reach Anthropic API: {e.reason!r}"
    except Exception as e:
        return f"Unexpected error calling Anthropic: {e!r}"

    blocks = raw.get("content") or []
    parts: list[str] = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(str(b.get("text", "")))
    if parts:
        return "\n".join(parts).strip()
    return json.dumps(raw, ensure_ascii=False)[:2000]


class ClaudeProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_OPTIONS(self) -> None:
        if self.path.split("?")[0] != "/api/claude":
            self.send_error(404)
            return
        self.send_response(204)
        _cors_headers(self)
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/health"):
            _send_json(
                self,
                200,
                {"ok": True, "anthropic_configured": bool(API_KEY), "port": PORT},
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path != "/api/claude":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            _send_json(self, 400, {"reply": "Invalid JSON body."})
            return

        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            _send_json(self, 400, {"reply": "Missing prompt."})
            return

        ctx = body.get("context")
        if ctx is None:
            ctx = body.get("reportContext")
        ctx_dict = ctx if isinstance(ctx, dict) else None

        if not API_KEY:
            _send_json(self, 200, {"reply": NO_KEY_REPLY, "response": NO_KEY_REPLY})
            return

        user_blob = build_dashboard_copilot_user_message(prompt, ctx_dict)
        reply = _call_anthropic(user_blob, system=DASHBOARD_COPILOT_SYSTEM)
        _send_json(self, 200, {"reply": reply, "response": reply})


def main() -> None:
    server = HTTPServer(("127.0.0.1", PORT), ClaudeProxyHandler)
    print(f"claude_proxy: http://127.0.0.1:{PORT}")
    print("  POST /api/claude  — JSON: prompt, optional context / reportContext (dashboard snapshot)")
    print(f"  ANTHROPIC_API_KEY set: {bool(API_KEY)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
