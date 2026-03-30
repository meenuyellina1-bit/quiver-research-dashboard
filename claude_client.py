"""
Reusable helper for Anthropic Claude via the Messages API (HTTP + requests only).

Usage:
    from claude_client import call_claude, call_claude_with_dashboard_context, enhance_with_ai

    answer = call_claude("Summarize this run in two sentences.")
    answer = call_claude_with_dashboard_context("What stands out?", {"meta": {...}})
    text, used_ai = enhance_with_ai("What changed this run", raw_text, {"k": "v"})

Successful ``call_claude`` responses are deduped via ``output/claude_run_cache.json`` (SHA-256 of model,
max_tokens, system, user text). Set ``QQ_DISABLE_CLAUDE_CACHE=1`` to bypass.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import requests

# --- Anthropic HTTP contract (Messages API) ---------------------------------
# https://docs.anthropic.com/en/api/messages — model / max_tokens / messages only (no legacy prompt API).
_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_DEFAULT_MESSAGES_MODEL = "claude-sonnet-4-5"
# Tried after primary (``ANTHROPIC_MODEL`` or default) when the API returns 404 / model-not-available style errors.
_MODEL_FALLBACK_ORDER: tuple[str, ...] = (
    "claude-sonnet-4",
    "claude-haiku-4-5",
    "claude-3-5-haiku-latest",
)
_ANTHROPIC_VERSION = "2023-06-01"
# Long-running prompts (e.g. reports) can pass a higher max_tokens at call site.
_DEFAULT_TIMEOUT_SEC = 120

# Per-process JSON cache (``output/claude_run_cache.json``): SHA-256 key over model, max_tokens, system, user text.
# Successful assistant text only; skips entries starting with ``[Assistant``. Disable with ``QQ_DISABLE_CLAUDE_CACHE=1``.
_CLAUDE_RUN_CACHE_PATH = Path("output") / "claude_run_cache.json"
_claude_run_cache_loaded: bool = False
_claude_run_cache: dict[str, str] = {}


def _claude_run_cache_disabled() -> bool:
    return os.environ.get("QQ_DISABLE_CLAUDE_CACHE", "").strip().lower() in ("1", "true", "yes")


def _load_claude_run_cache() -> dict[str, str]:
    """Lazy-load merge file → in-memory dict (same process / same run dedupes here too)."""
    global _claude_run_cache_loaded, _claude_run_cache
    if _claude_run_cache_loaded:
        return _claude_run_cache
    _claude_run_cache_loaded = True
    p = _CLAUDE_RUN_CACHE_PATH
    if p.is_file():
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _claude_run_cache = {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
        except (OSError, ValueError, TypeError):
            _claude_run_cache = {}
    return _claude_run_cache


def _claude_response_cache_key(system_text: str, user_text: str, max_tokens: int) -> str:
    primary_model = (os.environ.get("ANTHROPIC_MODEL") or "").strip() or _DEFAULT_MESSAGES_MODEL
    payload = f"{primary_model}\n{max_tokens}\n{system_text}\n{user_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Returned when ANTHROPIC_API_KEY is unset — compare to this if you need HTTP 503 vs 200.
CLAUDE_MISSING_KEY_REPLY = (
    "The AI assistant is not configured: set ANTHROPIC_API_KEY in your environment to enable responses."
)


def _store_claude_run_cache(key: str, text: str) -> None:
    if text.startswith("[Assistant") or text == CLAUDE_MISSING_KEY_REPLY:
        return
    cache = _load_claude_run_cache()
    cache[key] = text
    _CLAUDE_RUN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_CLAUDE_RUN_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except OSError:
        pass

# Ask Claude / research-copilot: grounded answers for a single dashboard run (Ask AI panel).
DASHBOARD_COPILOT_SYSTEM = """You are a research desk copilot. The user message contains one JSON dashboard snapshot and exactly one question under "=== USER QUESTION ===".

ABSOLUTE RULES (non-negotiable):
- Answer ONLY that question. Do not generalize, change the topic, or substitute a generic "dashboard summary."
- Use ONLY facts from the JSON in that message. Copy tickers, counts, scores, and dates exactly—never invent, round, or rephrase numbers.
- If the JSON cannot answer the question, say so in one short sentence. No apology spiral.
- Be specific, data-driven, and concise. Prefer lists and short lines over dense paragraphs.
- Default length: stay under ~120 words unless the user explicitly asks for more (e.g. "long", "detailed", "full rewrite", "paragraph", "step by step").
- Plain text only: no Markdown (no **, #, backticks, bullets as markdown syntax). No asterisks for emphasis.
- Do NOT use generic section headers or labels such as "Top Line Insight", "Executive summary", or bracketed labels like [Direct Answer] or [Anything]—unless the user explicitly asks for labeled sections.
- Prefer: list results first, then brief justification per item when needed.

QUESTION-AWARE FORMAT (match intent; still grounded in JSON only):
- "which names" / "which tickers" / "which anomaly" / ranking: lead with a ranked list of tickers (1. 2. 3. or lines). Put the answer first; then one line per ticker with the key counts/scores if the question needs justification.
- "why" / "explain": give the answer or conclusion in one line if applicable, then 2–4 short lines of reasoning tied to counts from the JSON.
- "compare" / "versus" / "vs": compare items side-by-side; cite the same fields for each (e.g. desk score, filings, overlap).
- "what changed" / "delta" / "this run": focus only on what differs or is notable for this run—do not restate the whole dashboard.

EXAMPLE (pattern for "which names are most supported by multiple datasets?" only—use this shape when the question actually asks that):
1) First line: 2–3 tickers in order (most supported first), comma-separated or numbered.
2) Next lines: one short reason per ticker (overlap, contract/lobbying flags, filing counts—only fields present in JSON).
3) Optional: one sentence summary only if non-redundant.

No preamble ("Here is an analysis..."). No closing essay. No fluff."""

# One-shot section rewrites for dashboard HTML (not Ask-AI copilot format).
DASHBOARD_TEXT_ENHANCEMENT_SYSTEM = """You are a senior equity research editor revising desk output for clarity and punch.

Rules:
- Output plain text only: no Markdown (no **, #, backticks, bullets as markdown syntax unless the user asked for line breaks).
- Preserve every digit, count, score, and date exactly as given in the source; do not invent or alter facts.
- Stay grounded in the structured JSON context when tightening language.

Return only the improved passage requested in the user message—no preamble or meta-commentary."""

_MAX_CONTEXT_JSON_CHARS = 24000


def _strip_dashboard_markdown_artifacts(text: str) -> str:
    """Remove common Markdown so Ask AI output renders cleanly as plain text in HTML (textContent)."""
    if not text.strip():
        return text
    t = text
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"__([^_]+)__", r"\1", t)
    t = t.replace("**", "")
    t = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\1", t)
    t = re.sub(r"(?m)^#{1,6}\s*", "", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = t.replace("```", "")
    return t.strip()


_CLAUDE_ALL_MODELS_FAILED_MSG = (
    "Claude model not available for this account/workspace. "
    "Check Anthropic model access and current supported model names."
)


def _models_to_try() -> list[str]:
    primary = (os.environ.get("ANTHROPIC_MODEL") or "").strip() or _DEFAULT_MESSAGES_MODEL
    out: list[str] = [primary]
    for m in _MODEL_FALLBACK_ORDER:
        if m not in out:
            out.append(m)
    return out


def _should_try_next_model_for_failure(response: requests.Response) -> bool:
    """True when the next model id in the chain may help (404 / model not available)."""
    if response.status_code == 404:
        return True
    if response.status_code == 200:
        return False
    detail = _short_error_detail(response).lower()
    if "model" in detail:
        return True
    try:
        body = response.json()
    except ValueError:
        return False
    err = body.get("error")
    if isinstance(err, dict):
        msg = str(err.get("message", "")).lower()
        typ = str(err.get("type", "")).lower()
        if "model" in msg or "model" in typ:
            return True
    return False


def _print_messages_api_response(response: requests.Response) -> None:
    """Log full API body to stdout (visible in the terminal running ``backend.py``)."""
    print("[Anthropic Messages API] full response body:", flush=True)
    try:
        body = response.json()
        print(json.dumps(body, ensure_ascii=False, indent=2), flush=True)
    except ValueError:
        print(response.text or "(empty body)", flush=True)


def build_dashboard_copilot_user_message(
    user_question: str,
    context: dict[str, Any] | None,
) -> str:
    """User turn for Ask Claude: embed JSON snapshot + question (no SDK)."""
    q = (user_question or "").strip()
    if context is None:
        ctx_text = "(No dashboard context JSON was provided; you cannot cite run-specific data.)"
    else:
        try:
            ctx_text = json.dumps(context, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            ctx_text = str(context)[:_MAX_CONTEXT_JSON_CHARS]
        if len(ctx_text) > _MAX_CONTEXT_JSON_CHARS:
            ctx_text = ctx_text[:_MAX_CONTEXT_JSON_CHARS] + "\n…(truncated)"
    return (
        "=== DASHBOARD CONTEXT (JSON — this run only; sole source of facts) ===\n"
        f"{ctx_text}\n\n"
        "=== USER QUESTION ===\n"
        f"{q}\n\n"
        "=== INSTRUCTIONS ===\n"
        "Answer ONLY the user's question above. Do not generalize. Use the JSON only; suggested chip text "
        "counts as the full question. Under ~120 words unless they explicitly ask for more. "
        "No generic section headers (no Top Line Insight, no bracketed labels). "
        "List concrete results first, then brief justification."
    )


def call_claude_with_dashboard_context(
    user_question: str,
    context: dict[str, Any] | None,
    *,
    max_tokens: int = 1024,
    system_prompt: str | None = None,
) -> str:
    """
    Ask Claude with the dashboard JSON snapshot + tight copilot system prompt (``backend`` ``/ask-ai``).

    Default ``max_tokens`` is capped to encourage short, question-specific answers; pass a higher value
    when callers need long-form output. ``system_prompt`` overrides the default copilot prompt.
    """
    payload = build_dashboard_copilot_user_message(user_question, context)
    sp = DASHBOARD_COPILOT_SYSTEM if system_prompt is None else system_prompt
    out = call_claude(
        payload,
        system_prompt=sp,
        max_tokens=max_tokens,
    )
    if out == CLAUDE_MISSING_KEY_REPLY:
        return out
    if out.startswith("[Assistant error:") or out.startswith("[Assistant skipped:"):
        return out
    return _strip_dashboard_markdown_artifacts(out)


def enhance_with_ai(
    section_name: str,
    raw_text: str,
    structured_data: dict[str, Any] | None,
) -> tuple[str, bool]:
    """
    Rewrite dashboard copy in one API call. Returns ``(text, used_ai)``; on failure ``used_ai`` is False
    and ``text`` is the original ``raw_text``.
    """
    raw = (raw_text or "").strip()
    if not raw:
        return (raw_text or "", False)

    ctx = structured_data if isinstance(structured_data, dict) else {}
    try:
        sd_json = json.dumps(ctx, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        sd_json = str(structured_data)[:_MAX_CONTEXT_JSON_CHARS]
    if len(sd_json) > _MAX_CONTEXT_JSON_CHARS:
        sd_json = sd_json[:_MAX_CONTEXT_JSON_CHARS] + "\n…(truncated)"

    user_prompt = (
        "Rewrite the following research output to be more specific, insightful, and differentiated.\n\n"
        "Rules:\n"
        "- Preserve ALL numbers exactly\n"
        "- Do NOT change factual content\n"
        "- Improve clarity and conciseness\n"
        "- Eliminate generic phrasing\n"
        "- Add deeper reasoning (why the signal matters)\n"
        "- Make it sound like a professional equity research note\n"
        "- Avoid repetition\n"
        "- No markdown formatting (no **, no symbols)\n\n"
        f"Section: {section_name}\n\n"
        f"Original text:\n{raw}\n\n"
        f"Structured data:\n{sd_json}\n\n"
        "Return only the improved version."
    )

    out = call_claude_with_dashboard_context(
        user_prompt,
        ctx,
        max_tokens=4096,
        system_prompt=DASHBOARD_TEXT_ENHANCEMENT_SYSTEM,
    )
    if (
        out == CLAUDE_MISSING_KEY_REPLY
        or out.startswith("[Assistant error:")
        or out.startswith("[Assistant skipped:")
    ):
        return (raw, False)

    improved = out.strip()
    return (improved if improved else raw, True)


def call_claude(
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 500,
) -> str:
    """
    Send ``prompt`` to Claude and return the assistant's text.

    - Uses ``ANTHROPIC_API_KEY`` from the environment (never crashes if missing).
    - On any failure, returns a short, human-readable fallback string (never raises).
    - Reuses prior successful responses when the cache key matches (see module docstring).
    """
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        return CLAUDE_MISSING_KEY_REPLY

    user_text = (prompt or "").strip()
    if not user_text:
        return "[Assistant skipped: empty prompt.]"

    system_text = (system_prompt or "").strip()

    # Clamp to API-safe bounds (Haiku supports large max_tokens; keep a sane ceiling).
    mt = max(1, min(int(max_tokens), 8192))

    if not _claude_run_cache_disabled():
        ck = _claude_response_cache_key(system_text, user_text, mt)
        hit = _load_claude_run_cache().get(ck)
        if hit is not None and not hit.startswith("[Assistant"):
            print(f"[Anthropic] cache hit {ck[:12]}…", flush=True)
            return hit

    headers = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    data_base: dict[str, Any] = {
        "max_tokens": mt,
        "messages": [{"role": "user", "content": user_text}],
    }
    if system_text:
        data_base["system"] = system_text

    models_to_try = _models_to_try()

    for i, model_id in enumerate(models_to_try):
        data = dict(data_base)
        data["model"] = model_id
        print(f"[Anthropic] using model: {model_id}", flush=True)
        try:
            response = requests.post(
                _MESSAGES_URL,
                headers=headers,
                json=data,
                timeout=_DEFAULT_TIMEOUT_SEC,
            )
        except requests.Timeout:
            return "[Assistant error: request timed out. Try a shorter prompt or retry later.]"
        except requests.RequestException as exc:
            return f"[Assistant error: network failure ({type(exc).__name__}).]"

        _print_messages_api_response(response)

        if response.status_code == 200:
            try:
                resp_data = response.json()
            except ValueError:
                return "[Assistant error: response was not valid JSON.]"

            parts: list[str] = []
            for block in resp_data.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))

            combined = "".join(parts).strip()
            if not combined:
                return "[Assistant error: model returned no text content.]"
            if not _claude_run_cache_disabled():
                _store_claude_run_cache(_claude_response_cache_key(system_text, user_text, mt), combined)
            return combined

        retryable = _should_try_next_model_for_failure(response)
        if i + 1 < len(models_to_try) and retryable:
            continue

        if i + 1 == len(models_to_try) and retryable:
            return _CLAUDE_ALL_MODELS_FAILED_MSG

        note = _short_error_detail(response)
        tried = ", ".join(models_to_try[: i + 1])
        return (
            f"[Assistant error: HTTP {response.status_code}. {note}] "
            f"(models tried in order: {tried})"
        )

    return _CLAUDE_ALL_MODELS_FAILED_MSG


def _short_error_detail(response: requests.Response) -> str:
    """Best-effort one-line reason from a non-200 Anthropic response."""
    try:
        body = response.json()
    except ValueError:
        return (response.text or "")[:180].replace("\n", " ").strip() or "No body."

    err = body.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("type")
        if msg:
            return str(msg)[:180]
    if isinstance(err, str):
        return err[:180]
    return str(body)[:180]
