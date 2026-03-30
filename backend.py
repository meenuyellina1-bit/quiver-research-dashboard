"""
Minimal Flask proxy: POST /ask-ai → Anthropic Messages API (requests, no SDK).

JSON body:
  { "prompt": "...", "context": { ... dashboard snapshot ... } }
  ``reportContext`` is accepted as an alias for ``context`` (legacy dashboard).

Run: set ANTHROPIC_API_KEY, then: python backend.py

Open the dashboard at http://127.0.0.1:PORT/dashboard (default PORT=5000). The browser does not
start this app for you—run ``python backend.py`` in a terminal and keep that window open.

Override port: ``set QQ_BACKEND_PORT=5001`` (Windows) then use the same value when you run
``python main.py`` so the dashboard's Ask AI URL matches.
"""

from __future__ import annotations

import html as html_module
import os
import traceback
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

from claude_client import CLAUDE_MISSING_KEY_REPLY, call_claude_with_dashboard_context

app = Flask(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent
_DASHBOARD_HTML = _BACKEND_DIR / "output" / "research_dashboard.html"


def _listen_port() -> int:
    raw = (os.environ.get("QQ_BACKEND_PORT") or "5000").strip()
    try:
        p = int(raw)
        if 1 <= p <= 65535:
            return p
    except ValueError:
        pass
    return 5000


@app.after_request
def _cors(resp):
    """Allow local dashboards (e.g. file:// or another port) to call this API."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    # GET for /health; POST + OPTIONS for /ask-ai. Preflight must allow every header the browser sends.
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Authorization"
    # Chrome: pages opened as file:// or from the public web may send a PNA preflight to localhost.
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp


@app.route("/health", methods=["GET"])
def health():
    return {"ok": True, "service": "backend"}, 200


@app.route("/dashboard", methods=["GET"])
def serve_dashboard():
    """Same-origin HTML so Ask AI fetch() to /ask-ai is not treated as cross-site."""
    if not _DASHBOARD_HTML.is_file():
        path_esc = html_module.escape(str(_DASHBOARD_HTML))
        body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Dashboard missing</title></head>
<body style="font-family:system-ui,sans-serif;max-width:42rem;margin:2rem auto;line-height:1.5;">
<h1>Dashboard file not found</h1>
<p>Generate it from the project folder, then refresh this page:</p>
<pre style="background:#f4f4f5;padding:1rem;border-radius:8px;">python main.py</pre>
<p>Expected file: <code>{path_esc}</code></p>
<p><a href="/health">Check API health</a></p>
</body></html>"""
        return Response(body, status=404, mimetype="text/html; charset=utf-8")
    return send_file(_DASHBOARD_HTML, mimetype="text/html; charset=utf-8")


@app.route("/ask-ai", methods=["OPTIONS"])
def ask_ai_options():
    return "", 204


@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    print("Received /ask-ai request")
    if not request.is_json:
        return jsonify({"error": 'Expected Content-Type: application/json'}), 400

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": 'Missing or empty "prompt" field'}), 400

    ctx = body.get("context")
    if ctx is None:
        ctx = body.get("reportContext")
    ctx_dict = ctx if isinstance(ctx, dict) else None

    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return jsonify(
            {
                "response": CLAUDE_MISSING_KEY_REPLY,
                "error": "ANTHROPIC_API_KEY is not set in the environment",
            }
        ), 503

    text = call_claude_with_dashboard_context(prompt, ctx_dict)

    if text == CLAUDE_MISSING_KEY_REPLY:
        return jsonify({"response": text, "error": "ANTHROPIC_API_KEY is not set"}), 503
    if text.startswith("[Assistant error:") or text.startswith("[Assistant skipped:"):
        return jsonify({"response": text}), 200

    return jsonify({"response": text})


if __name__ == "__main__":
    _port = _listen_port()
    print("Backend starting...")
    _key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    print(f"ANTHROPIC_API_KEY present: {'yes' if _key else 'no'}")
    print(f"Listening on http://127.0.0.1:{_port} (QQ_BACKEND_PORT overrides default 5000)")
    print("Keep this terminal window OPEN — closing it stops the server (browser will show connection refused).")
    print("After you see 'Running on http://...' below, open in the browser:")
    print(f"  http://127.0.0.1:{_port}/dashboard")
    print(f"  http://127.0.0.1:{_port}/health   (expect JSON with ok: true)")
    try:
        app.run(host="127.0.0.1", port=_port, debug=False)
    except Exception:
        traceback.print_exc()
