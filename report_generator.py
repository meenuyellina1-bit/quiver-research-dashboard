"""
Write the weekly research brief to Markdown from signal outputs (no fabricated numbers).
"""

from __future__ import annotations

import base64
import html
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_PATH = Path("output/weekly_research_brief.md")
HTML_DASHBOARD_PATH = Path("output/research_dashboard.html")
CHARTS_DIR = Path("output/charts")
CONGRESS_CHART_FILE = "congress_top_tickers.png"
INSIDER_CHART_FILE = "insider_top_tickers.png"

# Shown when the dashboard has no POST URL, fetch fails, or the response has no reply field.
# Matches claude_proxy.NO_KEY_REPLY when the local proxy runs without ANTHROPIC_API_KEY.
CLAUDE_DASHBOARD_PLACEHOLDER = (
    "The AI assistant is not connected in this local build. The UI is ready; add ANTHROPIC_API_KEY "
    "and a backend request handler to enable live answers."
)

# Appended when fetch() fails (e.g. claude_proxy.py not running while URL is set).
CLAUDE_DASHBOARD_FETCH_FAILED_SUFFIX = (
    " Could not reach the backend—start the local hook with: python claude_proxy.py"
)


def _qq_backend_port() -> int:
    """Port for local Flask ``backend.py``; set ``QQ_BACKEND_PORT`` to match when generating HTML and when running the server."""
    raw = (os.environ.get("QQ_BACKEND_PORT") or "5000").strip()
    try:
        p = int(raw)
        if 1 <= p <= 65535:
            return p
    except ValueError:
        pass
    return 5000


def _resolve_claude_dashboard_endpoint() -> str:
    """
    POST target for Ask AI Assistant. Explicit QQ_CLAUDE_DASHBOARD_ENDPOINT wins.
    If QQ_AUTO_LOCAL_CLAUDE is 1/true/yes, use the stdlib proxy (claude_proxy.py).
    Otherwise default to Flask ``backend.py`` at /ask-ai.
    """
    explicit = os.environ.get("QQ_CLAUDE_DASHBOARD_ENDPOINT", "").strip()
    if explicit:
        return explicit
    auto = os.environ.get("QQ_AUTO_LOCAL_CLAUDE", "").strip().lower()
    if auto in ("1", "true", "yes"):
        port = os.environ.get("CLAUDE_PROXY_PORT", "8765").strip() or "8765"
        return f"http://127.0.0.1:{port}/api/claude"
    return f"http://127.0.0.1:{_qq_backend_port()}/ask-ai"

# (button label, prompt text for textarea; None → focus-only “custom prompt” chip)
_ASK_CLAUDE_SUGGESTED_CHIPS: list[tuple[str, str | None]] = [
    ("Which anomaly is the strongest and why?", "Which anomaly is the strongest and why?"),
    ("Rewrite the top 3 signals in a sharper editorial tone.", "Rewrite the top 3 signals in a sharper editorial tone."),
    ("What follow-up research should David do next?", "What follow-up research should David do next?"),
    ("Which names are most supported by multiple datasets?", "Which names are most supported by multiple datasets?"),
    (
        "Suggest a better visualization for the strongest overlap signals.",
        "Suggest a better visualization for the strongest overlap signals.",
    ),
    ("Write my own prompt", None),
]


def _desk_score_str(val: Any) -> str:
    """Format Top Ranked desk score (0.0–10.0) for display."""
    try:
        return f"{float(val):.1f}"
    except (TypeError, ValueError):
        return "0.0"


def _top_ranked_scoring_breakdown_md(row: dict[str, Any]) -> list[str]:
    """Markdown lines for the additive score components (empty if missing)."""
    comp = row.get("score_components")
    if not isinstance(comp, dict) or not comp:
        return []
    order = [
        ("congress_frequency", "Congress frequency"),
        ("representative_breadth", "Rep breadth"),
        ("high_dollar_trade", "High-dollar trade"),
        ("insider_overlap", "Insider overlap"),
        ("insider_bias", "Insider bias"),
        ("recency", "Recency"),
    ]
    lines = ["", "Scoring breakdown:"]
    for key, label in order:
        if key not in comp:
            continue
        try:
            v = float(comp[key])
        except (TypeError, ValueError):
            continue
        lines.append(f"- {label}: {_md_escape(f'{v:.1f}')}")
    pen_order = [
        ("penalty_concentration", "Concentration penalty"),
        ("penalty_mixed_signal", "Mixed insider penalty"),
    ]
    for key, label in pen_order:
        if key not in comp:
            continue
        try:
            pv = float(comp[key])
        except (TypeError, ValueError):
            continue
        if pv <= 0:
            continue
        lines.append(f"- {label}: −{_md_escape(f'{pv:.1f}')}")
    return lines


def _html_scoring_breakdown_block(row: dict[str, Any]) -> str:
    comp = row.get("score_components")
    if not isinstance(comp, dict) or not comp:
        return ""
    order = [
        ("congress_frequency", "Congress frequency"),
        ("representative_breadth", "Rep breadth"),
        ("high_dollar_trade", "High-dollar trade"),
        ("insider_overlap", "Insider overlap"),
        ("insider_bias", "Insider bias"),
        ("recency", "Recency"),
    ]
    lis: list[str] = []
    for key, label in order:
        if key not in comp:
            continue
        try:
            v = float(comp[key])
        except (TypeError, ValueError):
            continue
        lis.append(
            f"<li><strong>{html.escape(label)}:</strong> {html.escape(f'{v:.1f}')}</li>"
        )
    pen_order = [
        ("penalty_concentration", "Concentration penalty"),
        ("penalty_mixed_signal", "Mixed insider penalty"),
    ]
    for key, label in pen_order:
        if key not in comp:
            continue
        try:
            pv = float(comp[key])
        except (TypeError, ValueError):
            continue
        if pv <= 0:
            continue
        lis.append(
            f"<li><strong>{html.escape(label)}:</strong> −{html.escape(f'{pv:.1f}')}</li>"
        )
    if not lis:
        return ""
    return (
        '<div class="evidence-panel scoring-breakdown-panel"><h4>Scoring breakdown</h4>'
        f'<ul class="dash-list">{"".join(lis)}</ul></div>'
    )


class SignalsBundle(TypedDict, total=False):
    """Structured inputs produced in main / signal_logic (all optional keys have defaults)."""

    large_trades: pd.DataFrame
    top_tickers: pd.DataFrame
    overlap: list[str]
    overlap_signals: list[Any]
    ranked_signals: list[dict[str, Any]]
    congress_row_count: int
    insider_row_count: int
    contracts_row_count: int
    lobbying_row_count: int
    off_exchange_row_count: int
    patents_row_count: int
    patents_disabled: bool
    contracts_symbols_reinforced: int
    lobbying_symbols_reinforced: int
    had_core_tickers_for_support: bool
    contracts_api_queried: bool
    lobbying_api_queried: bool
    off_exchange_api_queried: bool
    congress_df: pd.DataFrame
    insider_df: pd.DataFrame
    insider_only_fallback_mode: bool
    insider_fallback_watchlist: list[dict[str, Any]]
    qualified_ranked_signals: list[dict[str, Any]]
    hero_ranked_signals: list[dict[str, Any]]


def _md_escape(text: Any) -> str:
    """Avoid breaking Markdown when values contain special characters."""
    s = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    return s.replace("\n", " ").strip()


def _format_inline_md(text: Any) -> str:
    """Escape HTML; wrap segments between ** in <strong> (simple markdown bold)."""
    s = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    parts = s.split("**")
    chunks: list[str] = []
    for i, p in enumerate(parts):
        esc = html.escape(p, quote=False)
        if i % 2 == 1:
            chunks.append(f"<strong>{esc}</strong>")
        else:
            chunks.append(esc)
    return "".join(chunks)


def _md_bullets_to_html(md: str) -> str:
    """Turn leading '- ' lines into a <ul>."""
    items: list[str] = []
    for line in md.strip().split("\n"):
        line = line.strip()
        if line.startswith("- "):
            items.append(f"<li>{_format_inline_md(line[2:])}</li>")
    if not items:
        return ""
    return f'<ul class="dash-list">{"".join(items)}</ul>'


def _methodology_body_html(md: str) -> str:
    """Convert _methodology_section() output into compact stacked info cards."""
    md = md.strip()
    md = re.sub(r"^## Methodology\s*\n+", "", md)
    blocks = re.split(r"\n(?=### )", md)
    out: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        inner_parts: list[str] = []
        if block.startswith("### "):
            rest = block[4:].lstrip()
            nl = rest.find("\n")
            if nl == -1:
                inner_parts.append(f"<h3>{_format_inline_md(rest)}</h3>")
            else:
                title, body = rest[:nl].strip(), rest[nl + 1 :].strip()
                inner_parts.append(f"<h3>{_format_inline_md(title)}</h3>")
                for para in re.split(r"\n\s*\n", body):
                    p = para.strip()
                    if p:
                        inner_parts.append(f"<p>{_format_inline_md(p)}</p>")
            out.append(
                f'<article class="info-box info-box--compact">{"".join(inner_parts)}</article>'
            )
        else:
            for para in re.split(r"\n\s*\n", block):
                p = para.strip()
                if p:
                    inner_parts.append(f"<p>{_format_inline_md(p)}</p>")
            if inner_parts:
                out.append(
                    f'<div class="info-box info-box--compact info-box--lead">{"".join(inner_parts)}</div>'
                )
    if not out:
        return "<p class=\"muted\">No methodology text.</p>"
    return f'<div class="section-stack methodology-stack">{"".join(out)}</div>'


def _data_availability_list_html(md: str) -> str:
    """Strip section title; render '- ' lines as a compact info box list."""
    lines_out: list[str] = []
    for line in md.split("\n"):
        line = line.strip()
        if line.startswith("## ") or not line:
            continue
        if line.startswith("- "):
            lines_out.append(f"<li>{_format_inline_md(line[2:])}</li>")
    if not lines_out:
        return "<p class=\"muted\">No availability data.</p>"
    ul = f'<ul class="dash-list dash-list--tight">{"".join(lines_out)}</ul>'
    return f'<div class="info-box info-box--compact info-box--availability">{ul}</div>'


def _verification_body_html(md: str) -> str:
    """Parse Potential Next-Step markdown into intro info box + one panel per ### subsection."""
    md = md.strip()
    md = re.sub(r"^##[^\n]+\n+", "", md)
    chunks = re.split(r"\n(?=### )", md)
    out: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("### "):
            rest = chunk[4:].lstrip()
            nl = rest.find("\n")
            title = rest[:nl].strip() if nl != -1 else rest.strip()
            body = rest[nl + 1 :].strip() if nl != -1 else ""
            inner = f"<h3>{_format_inline_md(title)}</h3>"
            if body:
                inner += _verification_block_html(body)
            out.append(f'<article class="panel verify-card">{inner}</article>')
        elif i == 0:
            intro = _verification_block_html(chunk)
            out.append(f'<div class="info-box info-box--compact">{intro}</div>')
    if not out:
        return "<p class=\"muted\">No verification items.</p>"
    return f'<div class="section-stack verify-stack">{"".join(out)}</div>'


def _verification_block_html(body: str) -> str:
    lines = [ln.rstrip() for ln in body.split("\n")]
    segments: list[str] = []
    buf: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("- "):
            buf.append(s)
        else:
            if buf:
                segments.append(("ul", "\n".join(buf)))
                buf = []
            if s:
                segments.append(("p", s))
    if buf:
        segments.append(("ul", "\n".join(buf)))
    parts: list[str] = []
    for kind, content in segments:
        if kind == "ul":
            parts.append(_md_bullets_to_html(content))
        else:
            t = content.strip()
            if t.startswith("_") and t.endswith("_") and len(t) >= 2:
                inner = t[1:-1]
                parts.append(f"<p class=\"muted\">{_format_inline_md(inner)}</p>")
            else:
                parts.append(f"<p>{_format_inline_md(t)}</p>")
    return "\n".join(parts)


def _directional_buckets_html(
    ranked_signals: list[dict[str, Any]],
    overlap_signals: list[dict[str, Any]],
    *,
    max_per_bucket: int = 12,
) -> str:
    bullish: list[str] = []
    bearish: list[str] = []
    mixed: list[str] = []
    seen: set[str] = set()

    def _consume(row: dict[str, Any]) -> None:
        t = str(row.get("ticker", "")).strip().upper()
        if not t or t in seen:
            return
        seen.add(t)
        d = str(row.get("direction", "Mixed")).strip()
        if d == "Bullish" and len(bullish) < max_per_bucket:
            bullish.append(t)
        elif d == "Bearish" and len(bearish) < max_per_bucket:
            bearish.append(t)
        elif len(mixed) < max_per_bucket:
            mixed.append(t)

    for row in ranked_signals:
        if isinstance(row, dict):
            _consume(row)
    for row in overlap_signals:
        if isinstance(row, dict):
            _consume(row)

    def col(title: str, names: list[str], dir_slug: str) -> str:
        if not names:
            inner = "<p class=\"muted\">None in this bucket for the merged set.</p>"
        else:
            inner = ", ".join(f"<strong>{html.escape(n)}</strong>" for n in names)
            inner = f'<p class="dir-col-tickers">{inner}</p>'
        tag = _dir_tag_html(dir_slug)
        return (
            f'<div class="dir-col">'
            f'<div class="dir-col-head"><h3>{html.escape(title)}</h3>{tag}</div>'
            f"{inner}</div>"
        )

    return (
        '<div class="directional-grid">'
        f"{col('Bullish names', bullish, 'Bullish')}"
        f"{col('Bearish names', bearish, 'Bearish')}"
        f"{col('Mixed / watchlist names', mixed, 'Mixed')}"
        "</div>"
    )


def _executive_summary(signals: SignalsBundle) -> str:
    """2–3 sentences from real counts only."""
    n_congress = int(signals.get("congress_row_count", 0))
    n_insider = int(signals.get("insider_row_count", 0))
    large = signals.get("large_trades", pd.DataFrame())
    top = signals.get("top_tickers", pd.DataFrame())
    overlap = signals.get("overlap", [])

    n_large = len(large) if isinstance(large, pd.DataFrame) else 0
    n_overlap = len(overlap) if overlap else 0

    if bool(signals.get("insider_only_fallback_mode")):
        lines = [
            (
                f"This run loaded **{n_insider}** insider rows while **congressional trading was unavailable**, "
                "so the brief uses insider-only fallback highlights—no composite desk scores or cross-feed overlap ranking."
            ),
            "Treat this as a partial snapshot until Capitol-side data is available again; verify against primary insider filings.",
            "Figures below combine the insider feed with supporting API coverage notes only.",
        ]
        return "\n".join(f"- {s}" for s in lines)

    lines = [
        (
            f"This brief summarizes the current Quiver pull: **{n_congress}** congress rows and **{n_insider}** insider rows. "
            f"The run surfaces **{n_large}** large-range congressional row(s) (after the cap of ten), "
            f"ranks the busiest tickers in congress, and lists **{n_overlap}** symbol(s) that appear in both feeds."
        ),
        "Use the sections below as a starting point for deeper reading of filings and fundamentals—not as trading instructions.",
        "All figures below are computed only from the data returned in this run; nothing is simulated or padded.",
    ]
    if n_congress == 0 and n_insider == 0:
        lines = [
            "No API data was available for this run, so no signals could be computed.",
            "Re-run after confirming your API key and connectivity.",
        ]
    return "\n".join(f"- {s}" for s in lines)


def _data_availability_section(signals: SignalsBundle | dict[str, Any]) -> str:
    """Per-dataset load status for the brief: core feeds, conditional supporting pulls, and patents."""
    n_congress = int(signals.get("congress_row_count", 0))
    n_insider = int(signals.get("insider_row_count", 0))
    c_rows = int(signals.get("contracts_row_count", 0))
    l_rows = int(signals.get("lobbying_row_count", 0))
    ox_rows = int(signals.get("off_exchange_row_count", 0))
    pat_rows = int(signals.get("patents_row_count", 0))
    pat_off = bool(signals.get("patents_disabled"))
    c_n = int(signals.get("contracts_symbols_reinforced", 0))
    l_n = int(signals.get("lobbying_symbols_reinforced", 0))

    # Legacy bundles (before conditional pulls): absence of *_api_queried means APIs were hit.
    def _queried(key: str) -> bool | None:
        v = signals.get(key)
        if v is None:
            return None
        return bool(v)

    c_q = _queried("contracts_api_queried")
    l_q = _queried("lobbying_api_queried")
    ox_q = _queried("off_exchange_api_queried")

    def supporting_line(display: str, queried: bool | None, api_rows: int, reinforced: int) -> str:
        if queried is False:
            return (
                f"- **{display}:** Not queried in this run (supporting datasets run only after at least one "
                "top-ranked desk anomaly exists)."
            )
        if api_rows == 0:
            return f"- **{display}:** Queried; no relevant rows."
        if reinforced == 0:
            return (
                f"- **{display}:** Queried; no relevant rows (loaded **{api_rows}** row(s) from API; "
                "none matched the ranked/overlap ticker filter after local filtering)."
            )
        return (
            f"- **{display}:** Queried; loaded **{api_rows}** row(s); reinforced **{reinforced}** ranked/overlap symbol(s)."
        )

    def core_line(display: str, n: int) -> str:
        if n > 0:
            return f"- **{display}:** Loaded successfully (**{n}** row(s))."
        return f"- **{display}:** Unavailable or empty (**0** rows)."

    def off_exchange_line(queried: bool | None, n: int) -> str:
        if queried is False:
            return "- **Off-exchange:** Not queried in this run."
        if n == 0:
            return "- **Off-exchange:** Queried; no relevant rows."
        return (
            f"- **Off-exchange:** Queried; loaded **{n}** row(s) (not merged into ranked or overlap signals in this build)."
        )

    def patents_line() -> str:
        if pat_off:
            return "- **Patents:** Not fetched (endpoint not confirmed)."
        if pat_rows > 0:
            return f"- **Patents:** Loaded **{pat_rows}** row(s); not used in this brief."
        return "- **Patents:** Queried; no relevant rows."

    parts = [
        "## Data Availability Notes",
        "",
        core_line("Congressional trading", n_congress),
        core_line("Insider trading", n_insider),
        supporting_line("Government contracts (supporting)", c_q, c_rows, c_n),
        supporting_line("Lobbying (supporting)", l_q, l_rows, l_n),
        off_exchange_line(ox_q, ox_rows),
        patents_line(),
        "",
    ]
    return "\n".join(parts)


def _large_trade_explanation(row: pd.Series) -> str:
    """One–two sentences tied to the row's Range and context (no invented dollar amounts)."""
    r = _md_escape(row.get("Range", ""))
    return (
        f"The disclosure's **Range** field includes a monitored high-dollar band marker (e.g. $500k+ or $1M+ segments in the text: «{r}»). "
        "Compare with the member's other filings and primary House/Senate sources when prioritizing review."
    )


def _signal1_section(large: pd.DataFrame) -> str:
    parts: list[str] = ["## Signal 1: Large Congressional Trades", ""]
    if large.empty:
        parts.append("_No rows matched the large-trade rules for this run._")
        parts.append("")
        return "\n".join(parts)

    for idx, (_, row) in enumerate(large.iterrows(), start=1):
        ticker = _md_escape(row.get("Ticker", ""))
        parts.append(f"### {idx}. {ticker}")
        parts.append("")
        parts.append(f"- **Representative:** {_md_escape(row.get('Representative', ''))}")
        parts.append(f"- **Ticker:** {ticker}")
        parts.append(f"- **Transaction:** {_md_escape(row.get('Transaction', ''))}")
        parts.append(f"- **Range:** {_md_escape(row.get('Range', ''))}")
        if "CongressConviction" in large.columns:
            parts.append(
                f"- **Congressional conviction (trade-level):** {_md_escape(row.get('CongressConviction', ''))}/100"
            )
        if "ConvictionNote" in large.columns:
            parts.append(f"- **Why conviction on this line:** {_md_escape(row.get('ConvictionNote', ''))}")
        parts.append(f"- **Explanation:** {_large_trade_explanation(row)}")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _signal2_section(top: pd.DataFrame) -> str:
    parts: list[str] = [
        "## Signal 2: Most Active Tickers",
        "",
        "Ranked primarily by **congressional conviction** (rule-based Capitol score: purchases vs sales, large **Range** share, "
        "high-profile members, repetition, multi-member breadth, recency). **Count** is the raw row tally; higher conviction "
        "means the pattern looks more intentional than frequency alone.",
        "",
    ]
    if top.empty:
        parts.append("_No ticker frequency data (empty congress data or missing Ticker column)._")
        parts.append("")
        return "\n".join(parts)

    parts.append("| Rank | Ticker | Count | Conviction (0–100) |")
    parts.append("| ---: | --- | ---: | ---: |")
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        t = _md_escape(row["Ticker"])
        c = int(row["Count"])
        cv = int(row["CongressConviction"]) if "CongressConviction" in top.columns else 0
        parts.append(f"| {i} | {t} | {c} | {cv} |")
    parts.append("")
    if "ConvictionNote" in top.columns:
        parts.append("**Why conviction was elevated (one line per name):**")
        parts.append("")
        for _, row in top.iterrows():
            parts.append(f"- **{_md_escape(row['Ticker'])}:** {_md_escape(row.get('ConvictionNote', ''))}")
        parts.append("")
    parts.append(
        "**Explanation:** Conviction summarizes several interpretable disclosure cues at once; a high **Count** with lower conviction "
        "can mean repetitive but thin filings, while high conviction with a modest count can mean a concentrated, newsworthy cluster."
    )
    parts.append("")
    return "\n".join(parts)


def _strongest_overlap_section(overlap: list[str], overlap_signals: list[Any]) -> str:
    """
    Ranked overlap insight only — no full symbol dump.
    Ranking uses congress lines, unique reps, insider rows, buy-like count, and large-dollar Range flag.
    """
    n_total = len(overlap)
    parts: list[str] = [
        "## Strongest Cross-Dataset Overlap Signals",
        "",
        f"Total symbols in the raw intersection: **{n_total}**. Below are the **top ranked** names for this pull (not the full list).",
        "",
    ]
    if not overlap:
        parts.append("_No cross-dataset overlap this run (or one feed was empty)._")
        parts.append("")
        return "\n".join(parts)

    if not overlap_signals:
        parts.append("_Overlap exists but detailed ranking could not be built (check Ticker columns)._")
        parts.append("")
        return "\n".join(parts)

    for i, row in enumerate(overlap_signals, start=1):
        if not isinstance(row, dict):
            continue
        t = _md_escape(row.get("ticker", ""))
        parts.append(f"### {i}.")
        parts.append("")
        parts.append(f"Ticker: {t}")
        parts.append(
            f"Directional tilt (rule-based): **{_md_escape(row.get('direction', 'Mixed'))}**"
        )
        if row.get("direction_note"):
            parts.append(f"Why this label: {_md_escape(row.get('direction_note', ''))}")
        parts.append(f"Congress filing count: {_md_escape(row.get('congress_filing_count', 0))}")
        parts.append(f"Unique representatives: {_md_escape(row.get('unique_representatives', 0))}")
        parts.append(f"Insider rows: {_md_escape(row.get('insider_rows', 0))}")
        parts.append(f"Buy-like count: {_md_escape(row.get('buy_like_count', 0))}")
        parts.append(f"Sell-like count: {_md_escape(row.get('sell_like_count', 0))}")
        lg = bool(row.get("large_dollar_congress", False))
        parts.append(f"Large-dollar congressional trade (Range filter): {'yes' if lg else 'no'}")
        if row.get("contract_activity_count") is not None:
            parts.append(
                f"Federal contract rows (supporting signal, filtered to this symbol): {_md_escape(row.get('contract_activity_count', 0))}"
            )
        if row.get("lobbying_activity_count") is not None:
            parts.append(
                f"Lobbying activity count (supporting signal, filtered to this symbol): {_md_escape(row.get('lobbying_activity_count', 0))}"
            )
        if row.get("lobbying_amount") is not None:
            parts.append(f"Lobbying amount (summed from API field): {_md_escape(row.get('lobbying_amount'))}")
        if row.get("congress_conviction_0_100") is not None:
            parts.append(
                f"Congressional conviction (ticker-level, 0–100): {_md_escape(row.get('congress_conviction_0_100', ''))}"
            )
        if row.get("congress_conviction_note"):
            parts.append(f"Why conviction is elevated: {_md_escape(row.get('congress_conviction_note', ''))}")
        if row.get("congress_recency_factor") is not None:
            parts.append(
                f"Congress recency factor (0–1): {_md_escape(row.get('congress_recency_factor', ''))}"
            )
        if row.get("insider_recency_factor") is not None:
            parts.append(
                f"Insider recency factor (0–1): {_md_escape(row.get('insider_recency_factor', ''))}"
            )
        parts.append("Why it may matter:")
        parts.append(f"- {_md_escape(row.get('why_may_matter', ''))}")
        parts.append("")

    parts.append(
        "_Rank order = weighted sum of congress filing count, unique representative count, insider row count, "
        "buy-like (A-code) count, a fixed boost when a large-dollar congressional Range appears, ticker-level "
        "congressional conviction (0–22 point scale from the 0–100 rule score), plus small recency bumps from "
        "newest congress and insider dates._"
    )
    parts.append("")
    return "\n".join(parts)


def _high_profile_section(congress_df: pd.DataFrame) -> str:
    """## High-Profile Congressional Trades — name + Range + conflict-of-interest angle (editorial)."""
    from signal_logic import format_high_profile_markdown, get_high_profile_congress_trades

    hp = get_high_profile_congress_trades(congress_df)
    body = format_high_profile_markdown(hp)
    return "\n".join(
        [
            "## High-Profile Congressional Trades",
            "",
            "Curated for Quiver-style **name + dollar band + conflict** framing: priority members (Pelosi, McCormick, "
            "leadership bench), then **purchases over sales**, **larger reported Range** bands, then **trade-level congressional "
            "conviction** and recency as tie-breakers. **Separate from Top Ranked Signals** (not scored the same way).",
            "",
            body,
        ]
    )


def _top_ranked_section(ranked: list[dict[str, Any]]) -> str:
    """## Top Ranked Signals — formatted like research desk leads (see exact field labels below)."""
    parts: list[str] = [
        "## Top Ranked Signals",
        "",
        "Desk-style leads built only from this run’s counts and flags (additive **0.0–10.0** desk score, one decimal; see Methodology).",
        "",
    ]
    if not ranked:
        parts.append("_No ranked signals for this run (insufficient congress data)._")
        parts.append("")
        return "\n".join(parts)

    for i, row in enumerate(ranked, start=1):
        t = _md_escape(row.get("ticker", ""))
        sc = _desk_score_str(row.get("score", 0))
        ev = row.get("evidence") or {}
        large = bool(ev.get("large_congress_trade"))
        large_txt = "yes" if large else "no"

        parts.append(f"### {i}.")
        parts.append("")
        parts.append(f"Ticker: {t}")
        parts.append(f"Score: {sc}/10")
        parts.extend(_top_ranked_scoring_breakdown_md(row))
        ddir = row.get("direction") or ev.get("direction", "Mixed")
        parts.append(f"Directional tilt (rule-based): **{_md_escape(ddir)}**")
        if row.get("direction_note") or ev.get("direction_note"):
            parts.append(
                f"Why this label: {_md_escape(row.get('direction_note') or ev.get('direction_note', ''))}"
            )
        parts.append("")
        parts.append("Why it triggered:")
        for line in row.get("why_triggered") or []:
            parts.append(f"- {_md_escape(line)}")
        parts.append("")
        parts.append("Evidence:")
        parts.append(f"- congressional filing count: {_md_escape(ev.get('congress_filings', 0))}")
        parts.append(f"- unique representative count: {_md_escape(ev.get('unique_representatives', 0))}")
        parts.append(f"- insider row count: {_md_escape(ev.get('insider_rows', 0))}")
        parts.append(f"- buy-like count: {_md_escape(ev.get('insider_buys', 0))}")
        parts.append(f"- sell-like count: {_md_escape(ev.get('insider_sells', 0))}")
        parts.append(
            f"- congress recency factor (0–1, from newest ReportDate/TransactionDate): {_md_escape(ev.get('congress_recency_factor', ''))}"
        )
        parts.append(
            f"- insider recency factor (0–1, from newest Date/fileDate): {_md_escape(ev.get('insider_recency_factor', ''))}"
        )
        parts.append(f"- whether there was a large-dollar congressional trade: {large_txt}")
        if ev.get("contract_activity_count") is not None:
            parts.append(f"- contract activity count: {_md_escape(ev.get('contract_activity_count', 0))}")
        if ev.get("lobbying_activity_count") is not None:
            parts.append(f"- lobbying activity count: {_md_escape(ev.get('lobbying_activity_count', 0))}")
        if ev.get("lobbying_amount") is not None:
            parts.append(f"- lobbying amount (summed from API field): {_md_escape(ev.get('lobbying_amount'))}")
        if ev.get("congress_conviction_0_100") is not None:
            parts.append(
                f"- congressional conviction (ticker-level, 0–100): {_md_escape(ev.get('congress_conviction_0_100', ''))}"
            )
        if ev.get("congress_conviction_note"):
            parts.append(f"- why conviction is elevated: {_md_escape(ev.get('congress_conviction_note', ''))}")
        parts.append("")
        parts.append("Why it may matter:")
        parts.append(f"- {_md_escape(row.get('why_may_matter', ''))}")
        parts.append("")
        parts.append("Suggested newsletter angle:")
        parts.append(f"- {_md_escape(row.get('newsletter_angle', ''))}")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _directional_signal_summary_section(
    ranked_signals: list[dict[str, Any]],
    overlap_signals: list[dict[str, Any]],
    *,
    max_per_bucket: int = 12,
) -> str:
    """
    ## Directional Signal Summary — group strongest tickers by Bullish / Bearish / Mixed.

    Order: top-ranked signals first, then strongest overlap (dedupe by ticker).
    """
    bullish: list[str] = []
    bearish: list[str] = []
    mixed: list[str] = []
    seen: set[str] = set()

    def _consume(row: dict[str, Any]) -> None:
        t = str(row.get("ticker", "")).strip().upper()
        if not t:
            return
        if t in seen:
            return
        seen.add(t)
        d = str(row.get("direction", "Mixed")).strip()
        if d == "Bullish":
            if len(bullish) < max_per_bucket:
                bullish.append(t)
        elif d == "Bearish":
            if len(bearish) < max_per_bucket:
                bearish.append(t)
        else:
            if len(mixed) < max_per_bucket:
                mixed.append(t)

    for row in ranked_signals:
        if isinstance(row, dict):
            _consume(row)
    for row in overlap_signals:
        if isinstance(row, dict):
            _consume(row)

    def _fmt(names: list[str]) -> str:
        if not names:
            return "_No tickers in this bucket for the merged ranked + overlap set._"
        return ", ".join(f"**{_md_escape(x)}**" for x in names)

    return "\n".join(
        [
            "## Directional Signal Summary",
            "",
            "Quick scan of **Top Ranked** and **Strongest Cross-Dataset Overlap** names by a simple rule: "
            "**Bullish** = congress rows are purchase-heavy *and* insiders are acquisition-heavy; "
            "**Bearish** = congress sale-heavy *and* insiders disposition-heavy; **Mixed** = anything else "
            "(conflicting tilts, ties, or uncoded rows). This is disclosure tilt only—not a price call.",
            "",
            "### Bullish names",
            "",
            _fmt(bullish),
            "",
            "### Bearish names",
            "",
            _fmt(bearish),
            "",
            "### Mixed / watchlist names",
            "",
            _fmt(mixed),
            "",
        ]
    )


def _methodology_section() -> str:
    """How the MVP spots and orders anomalies—short, repeatable rules (no ML)."""
    return "\n".join(
        [
            "## Methodology",
            "",
            "This brief is generated from one API pull. Each block below is rule-based so the workflow stays intentional and reproducible. "
            "**Top Ranked Signals** use a transparent **additive desk score from 0.0 to 10.0** (one decimal after capping the sum). "
            "Six components (congress filing frequency, representative breadth, parsed high-dollar **Range** bands, insider overlap volume, "
            "insider buy/sell imbalance with slightly more weight on buy skew, and calendar recency of the latest congress or insider date) are summed; "
            "then **penalties** may subtract up to **0.5** total: **concentration** (−0.3 when there are ≥4 congress rows but only 1–2 distinct representatives) "
            "and **mixed insider** (−0.2 when buy-like and sell-like counts are both ≥3 and within a close band). "
            "The desk score is **floored at 0**, then capped at **10.0** (one decimal). **Ticker-level congressional conviction (0–100)** tie-breaks equal desk scores. "
            f"**Charts** are raw top-10 frequency bars only; files live under `{CHARTS_DIR.as_posix()}/`.",
            "",
            "### Large trade signal",
            "",
            "Congressional rows whose **Range** field contains fixed large-dollar markers (e.g. `500,001` through `25,000,000`) qualify. "
            "Up to ten rows are shown, sorted by a **trade-level conviction** score (purchase vs sale text, high-profile member, row recency)—not raw table order.",
            "",
            "### Congressional consensus signal",
            "",
            "The **most active tickers** table ranks symbols with a **ticker-level congressional conviction** score, not frequency alone: "
            "it blends purchase share, large-**Range** share, high-profile filings, repetition, how many distinct **Representatives** touched the name, and recency. "
            "Symbols with **two or more** members filing the same ticker act as a separate consensus-style breadth check inside overlap and ranked logic.",
            "",
            "### Cross-dataset overlap signal",
            "",
            "Overlap is the intersection of tickers in the congressional and insider feeds for this run. "
            "The brief surfaces only the **strongest** overlaps, ranked by filing counts, member breadth, insider volume, buy-like codes, large congressional ranges, conviction, and small recency bumps.",
            "",
            "### Insider bias signal",
            "",
            "Insider rows are classified **buy-like** vs **sell-like** using **AcquiredDisposedCode** (A/D) with **TransactionCode** and text fallbacks. "
            "The desk score’s **insider bias** component rewards buy/sell skew (buy-heavy names score slightly higher than sell-heavy at the same imbalance); "
            "**Bullish/Bearish/Mixed** labels still require congress and insider tilts to line up or else default to mixed.",
            "",
            "### Conviction score",
            "",
            "**Ticker-level** conviction (0–100) is a weighted mix of Capitol-only cues: purchases vs sales, large-range share, high-attention members, disclosure volume, multi-representative breadth, and filing recency. "
            "**Trade-level** conviction applies the same idea to single large-trade rows. "
            "Conviction **tie-breaks** tickers with the **same** additive desk score; everything is hand-tuned weights, not machine learning.",
            "",
            "### Recency weighting",
            "",
            "The desk score’s **recency** component uses the single newest **ReportDate/TransactionDate** (congress) or **Date/fileDate** (insider) for that ticker and maps to fixed day buckets (3 / 7 / 14 / 30+ days). "
            "Separately, **conviction** scoring still uses its own 0–1 recency curve for Capitol-only context.",
            "",
            "### Supporting dataset confirmation signals",
            "",
            "**Government contracts**, **lobbying**, and **off-exchange** are fetched **only when** this run has at least one **top-ranked desk anomaly**; each supporting endpoint is called **at most once**, then rows are filtered locally to tickers already in **Top Ranked** or **Strongest Overlap**. "
            "Contracts and lobbying attach only as supporting counts (and optional lobbying amounts)—never as standalone narrative sections.",
            "",
        ]
    )


def _potential_next_step_verification_section(signals: SignalsBundle | dict[str, Any]) -> str:
    """
    Analyst-style follow-ups tied to what this run actually surfaced—no invented filings or prices.
    """
    ranked = list(signals.get("ranked_signals") or [])
    overlap_signals_list = list(signals.get("overlap_signals") or [])
    overlap_tickers = list(signals.get("overlap") or [])
    large = signals.get("large_trades", pd.DataFrame())
    if not isinstance(large, pd.DataFrame):
        large = pd.DataFrame()
    insider_df = signals.get("insider_df", pd.DataFrame())
    if not isinstance(insider_df, pd.DataFrame):
        insider_df = pd.DataFrame()

    tickers_ordered: list[str] = []
    seen: set[str] = set()
    for row in ranked[:6]:
        if not isinstance(row, dict):
            continue
        t = str(row.get("ticker", "")).strip().upper()
        if t and t not in seen:
            seen.add(t)
            tickers_ordered.append(t)
    for row in overlap_signals_list[:6]:
        if not isinstance(row, dict):
            continue
        t = str(row.get("ticker", "")).strip().upper()
        if t and t not in seen:
            seen.add(t)
            tickers_ordered.append(t)

    reinforced: list[str] = []
    seen_r: set[str] = set()
    for row in ranked:
        if not isinstance(row, dict):
            continue
        t = str(row.get("ticker", "")).strip().upper()
        if not t:
            continue
        ev = row.get("evidence")
        if isinstance(ev, dict) and (
            "contract_activity_count" in ev or "lobbying_activity_count" in ev
        ):
            if t not in seen_r:
                seen_r.add(t)
                reinforced.append(t)
    for row in overlap_signals_list:
        if not isinstance(row, dict):
            continue
        t = str(row.get("ticker", "")).strip().upper()
        if not t:
            continue
        if row.get("contract_activity_count") is not None or row.get("lobbying_activity_count") is not None:
            if t not in seen_r:
                seen_r.add(t)
                reinforced.append(t)

    rel_cols = [
        c
        for c in insider_df.columns
        if "relationship" in str(c).lower()
        or str(c).lower() in ("officertitle", "officer title", "title")
    ]

    parts: list[str] = [
        "## Potential Next-Step Verification",
        "",
        "Practical checks a human analyst would run after this automated pass. Suggestions reference only what appears in the sections above.",
        "",
        "### Top-ranked and strongest overlap names",
        "",
    ]

    has_bull_bear = any(
        isinstance(r, dict) and str(r.get("direction", "")).strip() in ("Bullish", "Bearish")
        for r in ranked + overlap_signals_list
    )

    if tickers_ordered:
        tick_txt = ", ".join(f"**{_md_escape(t)}**" for t in tickers_ordered)
        blk = [
            f"- **SEC / issuer filings:** For {tick_txt}, review recent **Form 4**, **10-Q/10-K**, and **8-K** on EDGAR (or your terminal) for the same symbol—this brief does not embed filing text.",
            "- **Prior weeks:** Compare these tickers against your last Quiver export or notes (counts, conviction, directional labels) to see what is **new** versus a lingering multi-week pattern.",
            "- **News and earnings:** Scan headlines and the company's confirmed **earnings** calendar so disclosure bursts are not already explained by public catalysts.",
        ]
        if has_bull_bear:
            blk.append(
                "- **Directional tilt:** At least one lead is tagged **Bullish** or **Bearish** from parsed disclosure text and insider codes only—stress-test that read against fundamentals and news before treating it as a thesis."
            )
        blk.append("")
        parts.extend(blk)
    else:
        parts.extend(
            [
                "_No top-ranked or overlap rows this run._ When those sections populate, repeat the SEC / prior-week / news checks for whichever symbols surface.",
                "",
            ]
        )

    parts.extend(
        [
            "### Large congressional trades (Signal 1)",
            "",
        ]
    )
    if not large.empty:
        parts.append(
            "- **Primary sources:** For each listed member, open the official House/Senate disclosure site (and committee filings if needed) and verify **Transaction** type and **Range** wording against this table—bands are self-reported, not executed prices."
        )
    else:
        parts.append(
            "_No large-range rows this run._ When Signal 1 lists names, verify each line on the member's primary disclosure before using dollar-band language in copy."
        )
    parts.extend(["", "### Cross-dataset overlap", ""])
    if overlap_tickers:
        parts.append(
            "- **Timeline sanity:** For tickers in **both** feeds, compare congress **dates** and insider **dates** from the raw tables to see whether activity clusters in the same window or is only a symbol-level coincidence."
        )
    else:
        parts.append(
            "_No overlap set this run._ When overlap returns, reconcile timestamps across feeds before implying one narrative across Capitol and Form 4 data."
        )
    parts.extend(["", "### Insider activity depth", ""])
    if not insider_df.empty and rel_cols:
        parts.append(
            f"- **Role split:** The insider pull includes **{_md_escape(rel_cols[0])}** (or similar). Tabulate whether buy/sell skew is concentrated among **officers** vs **directors** before treating the signal as executive-driven."
        )
    else:
        parts.append(
            "- **Role split:** On EDGAR Form 4s (or the raw insider columns if **Relationship** / **Title** appear next run), check whether volume sits with **officers** versus **directors**; the MVP only aggregates by ticker unless those fields exist."
        )
    parts.extend(["", "### Supporting datasets (contracts / lobbying)", ""])
    if reinforced:
        rtxt = ", ".join(f"**{_md_escape(x)}**" for x in reinforced[:8])
        parts.append(
            f"- **Same-name reinforcement:** For {rtxt}, skim the underlying contract and lobbying rows in Quiver (or exports) to confirm they reference the **same issuer narrative** you are drawing from congress/insider—not a coincidental ticker match."
        )
    else:
        parts.append(
            "- **Same-name reinforcement:** If this brief later attaches contract or lobbying counts to a lead, open those source rows and confirm they support the same story as the core signal rather than an unrelated line item."
        )
    parts.append("")
    return "\n".join(parts)


def _save_top_ticker_bar_chart(
    df: pd.DataFrame,
    filename: str,
    *,
    chart_title: str,
) -> bool:
    """
    Vertical bar chart of top 10 tickers by occurrence. Returns True if a PNG was written.
    """
    if df.empty or "Ticker" not in df.columns:
        return False
    counts = df["Ticker"].dropna().astype(str).str.strip()
    counts = counts[counts != ""]
    if counts.empty:
        return False
    top = counts.value_counts().head(10)
    if top.empty:
        return False

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / filename

    fig, ax = plt.subplots(figsize=(8, 4.2))
    top.plot(kind="bar", ax=ax, color="#2c5282", edgecolor="white")
    ax.set_title(chart_title)
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Number of rows")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def _charts_section(*, congress_chart_ok: bool, insider_chart_ok: bool) -> str:
    """Markdown for embedded charts with captions (paths relative to project root)."""
    rel = CHARTS_DIR.as_posix()
    lines: list[str] = [
        "## Charts",
        "",
        "### Congressional tickers",
        "",
    ]
    if congress_chart_ok:
        lines.append(f"![Top 10 congressional tickers by row count]({rel}/{CONGRESS_CHART_FILE})")
        lines.append("")
        lines.append(
            "**What it shows:** Counts how many times each stock **Ticker** appears in the congressional disclosure table "
            "for this API pull, and displays the ten most frequent symbols as bars."
        )
        lines.append("")
        lines.append(
            "**Why it matters:** Names with the most rows are where members reported activity most often in this snapshot—"
            "useful for spotting crowded symbols or repeated filings before you drill into individual trades."
        )
    else:
        lines.append("_Chart not generated: no congressional rows or no **Ticker** column._")
    lines.extend(["", "### Insider tickers", ""])
    if insider_chart_ok:
        lines.append(f"![Top 10 insider tickers by row count]({rel}/{INSIDER_CHART_FILE})")
        lines.append("")
        lines.append(
            "**What it shows:** Same frequency view for the insider feed—each bar is how many rows reference that **Ticker** "
            "among the insider filings returned this run."
        )
        lines.append("")
        lines.append(
            "**Why it matters:** Corporate insiders file Form 4 data unevenly across names; the busiest tickers here "
            "are where filing volume clustered for this download, which can guide which stories to read first."
        )
    else:
        lines.append("_Chart not generated: no insider rows or no **Ticker** column._")
    lines.append("")
    return "\n".join(lines)


def _html_panel(title: str, inner: str, *, extra_class: str = "") -> str:
    cls = f"panel {extra_class}".strip()
    return f'<div class="{cls}"><h3>{_format_inline_md(title)}</h3>{inner}</div>'


def _dir_tag_html(direction: Any) -> str:
    """Compact Bullish / Bearish / Mixed label for scan-friendly dashboards."""
    d = str(direction or "Mixed").strip() or "Mixed"
    if d == "Bullish":
        cls = "dir-tag dir-bullish"
    elif d == "Bearish":
        cls = "dir-tag dir-bearish"
    else:
        cls = "dir-tag dir-mixed"
    return f'<span class="{cls}">{html.escape(d)}</span>'


def _hero_card_title_line(row: dict[str, Any]) -> str:
    """Single executive-style headline; prefer first trigger, else fallback."""
    triggers = row.get("why_triggered") or []
    if triggers:
        s = str(triggers[0]).strip()
        if len(s) > 118:
            s = s[:115].rsplit(" ", 1)[0] + "…"
        return s
    t = str(row.get("ticker", "")).strip()
    sc = _desk_score_str(row.get("score", 0))
    return f"Highest desk-ranked name this run ({sc}/10)." if t else "Ranked lead"


def _hero_evidence_bullets(row: dict[str, Any]) -> list[str]:
    """Up to three concise, factual bullets for the hero card."""
    ev = row.get("evidence") or {}
    if not isinstance(ev, dict):
        ev = {}
    cf = int(ev.get("congress_filings") or 0)
    ur = int(ev.get("unique_representatives") or 0)
    it = int(ev.get("insider_rows") or 0)
    ib = int(ev.get("insider_buys") or 0)
    is_ = int(ev.get("insider_sells") or 0)
    cc = ev.get("congress_conviction_0_100")
    d = str(row.get("direction") or ev.get("direction") or "Mixed").strip()

    ordered: list[str] = []
    if ev.get("large_congress_trade"):
        ordered.append("Large-dollar congressional **Range** band hits this symbol in the current pull.")
    if cc is not None:
        ordered.append(f"Congressional conviction (rule score): **{cc}/100**.")
    ordered.append(f"**{cf}** Capitol row(s) across **{ur}** distinct member(s).")
    if it > 0:
        ordered.append(f"Insider feed: **{it}** row(s), **{ib}** buy-like vs **{is_}** sell-like (coded fields).")
    elif len(ordered) < 3:
        ordered.append("No insider rows for this ticker in this download.")
    if d in ("Bullish", "Bearish"):
        ordered.append(f"**{d}** tilt: congress + insider parsed directions agree under MVP rules.")
    if ev.get("contract_activity_count") is not None:
        ordered.append(f"Supporting: **{ev.get('contract_activity_count')}** contract row(s) on the same ticker.")
    if ev.get("lobbying_activity_count") is not None:
        ordered.append(f"Supporting: **{ev.get('lobbying_activity_count')}** lobbying row(s) on the same ticker.")
    return ordered[:3]


def _hero_why_matters_line(row: dict[str, Any]) -> str:
    w = str(row.get("why_may_matter", "")).strip()
    if not w:
        return (
            "Use this as a disclosure bookmark only—confirm filings and context before implying timing, motive, or edge."
        )
    if len(w) > 240:
        w = w[:237].rsplit(" ", 1)[0] + "…"
    return w


def _ticker_scroll_token(ticker: str) -> str:
    """Stable token for ``data-target`` and ``id="ticker-{token}"`` (HTML id–safe, uppercase)."""
    s = str(ticker).strip().upper()
    if not s:
        return "UNKNOWN"
    out: list[str] = []
    for ch in s:
        if ch.isalnum() or ch in ".-_":
            out.append(ch)
        else:
            out.append("-")
    token = "".join(out).strip("-")
    return token or "UNKNOWN"


# Recognized large-cap / mega-cap tech tickers for light thematic clustering (rule-based, not sector data).
_KEY_INSIGHT_TECH_TICKERS: frozenset[str] = frozenset(
    {
        "AAPL",
        "MSFT",
        "GOOGL",
        "GOOG",
        "META",
        "AMZN",
        "NVDA",
        "AMD",
        "INTC",
        "CRM",
        "ORCL",
        "AVGO",
        "CSCO",
        "ADBE",
        "QCOM",
        "TXN",
        "IBM",
        "NOW",
        "INTU",
        "AMAT",
        "MU",
        "LRCX",
        "KLAC",
        "SNPS",
        "CDNS",
    }
)


def _key_insight_for_dashboard(signals: SignalsBundle | dict[str, Any]) -> str:
    """
    One or two short sentences synthesizing the run from ranked signals, overlap, direction, large trades,
    and representative breadth—no ML, only MVP outputs.

    When ``qualified_ranked_signals`` is present, its order follows **Story-Worthiness** (then distinctiveness, desk score)—
    not desk score alone—so the lead name may be a better *story* than the raw table’s #1.
    """
    if bool(signals.get("insider_only_fallback_mode")):
        wl = [w for w in (signals.get("insider_fallback_watchlist") or []) if isinstance(w, dict)]
        n_ins = int(signals.get("insider_row_count", 0))
        names = [str(w.get("ticker", "")).strip().upper() for w in wl if str(w.get("ticker", "")).strip()][:3]
        if names:
            joined = ", ".join(names)
            return (
                f"Congressional data was unavailable in this pull, so desk-ranked scores are omitted. "
                f"The insider feed still logged {n_ins} row(s), with the busiest symbols including {joined}."
            )
        return (
            f"Congressional data was unavailable in this pull, so desk-ranked scores are omitted; "
            f"this dashboard highlights insider-only context from {n_ins} row(s) in the download."
        )

    ranked_raw = signals.get("ranked_signals") or []
    qsig = signals.get("qualified_ranked_signals")
    if isinstance(qsig, list):
        ranked = [r for r in qsig if isinstance(r, dict)]
    else:
        ranked = [r for r in ranked_raw if isinstance(r, dict)]
    overlap_raw = list(signals.get("overlap") or [])
    overlap_set = {str(x).strip().upper() for x in overlap_raw if str(x).strip()}
    overlap_sigs = [x for x in (signals.get("overlap_signals") or []) if isinstance(x, dict)]
    large = signals.get("large_trades")
    if not isinstance(large, pd.DataFrame):
        large = pd.DataFrame()

    overlap_n = len(overlap_set)

    top5: list[tuple[str, dict[str, Any]]] = []
    for r in ranked:
        t = str(r.get("ticker", "")).strip().upper()
        if t:
            top5.append((t, r))
        if len(top5) >= 5:
            break

    if not top5:
        if (
            isinstance(qsig, list)
            and len(qsig) == 0
            and any(isinstance(r, dict) and str(r.get("ticker", "")).strip() for r in ranked_raw)
        ):
            return (
                "Desk scores ran, but no ticker cleared the tightened anomaly bar for top highlights "
                "(Capitol breadth, filings vs the ranked median, overlap depth, recency, and add-ons). "
                "Use the full ranked table below for the complete list."
            )
        if overlap_n > 0:
            ft = ""
            if overlap_sigs:
                ft = str(overlap_sigs[0].get("ticker", "")).strip().upper()
            if ft:
                return (
                    f"No desk-ranked leaderboard this pull, but {overlap_n} symbols sit in the congressional–insider "
                    f"intersection—{ft} tops the overlap ranking on combined counts."
                )
        return (
            "This pull did not produce ranked desk leads or overlap; use the raw sections below until filings support scoring."
        )

    leader_tk, leader_row = top5[0]
    top3_tk = [t for t, _ in top5[:3]]

    tech_hits = sum(1 for t in top3_tk if t in _KEY_INSIGHT_TECH_TICKERS)
    leaders_in_overlap = [t for t in top3_tk if t in overlap_set]

    bearish_leaders = [
        t for t, r in top5 if str(r.get("direction", "")).strip() == "Bearish"
    ]
    bullish_leaders = [
        t for t, r in top5 if str(r.get("direction", "")).strip() == "Bullish"
    ]

    max_reps = 0
    for _, r in top5[:3]:
        ev = r.get("evidence") if isinstance(r.get("evidence"), dict) else {}
        max_reps = max(max_reps, int(ev.get("unique_representatives") or 0))

    any_large_top3 = any(
        bool((r.get("evidence") or {}).get("large_congress_trade")) if isinstance(r.get("evidence"), dict) else False
        for _, r in top5[:3]
    )

    ev_lead = leader_row.get("evidence") if isinstance(leader_row.get("evidence"), dict) else {}
    conv = leader_row.get("congress_conviction_0_100")
    if conv is None:
        conv = ev_lead.get("congress_conviction_0_100")
    try:
        conv_i = int(conv) if conv is not None else None
    except (TypeError, ValueError):
        conv_i = None

    # --- Template selection (priority: overlap + tech, then directional tech, then overlap + large, breadth, default)

    if tech_hits >= 2 and len(leaders_in_overlap) >= 2 and overlap_n >= 3:
        tail = f"{leader_tk} stands out as the clearest high-conviction ranked lead"
        if conv_i is not None and conv_i >= 72:
            tail += f" (Capitol conviction {conv_i}/100)"
        tail += "."
        return (
            "This run is dominated by large-cap tech names showing overlap between congressional disclosures and "
            f"insider activity, and {tail}"
        )

    if tech_hits >= 2 and len(bearish_leaders) >= 2:
        bpair = " and ".join(bearish_leaders[:2])
        return (
            "The strongest desk-ranked names cluster around repeated congressional attention in mega-cap tech, while "
            f"directional labels tilt bearish for {bpair}."
        )

    if overlap_n >= 3 and len(leaders_in_overlap) >= 2:
        pair = " and ".join(leaders_in_overlap[:2])
        return (
            f"Cross-feed overlap is the spine of this pull ({overlap_n} symbols in both datasets), and top-ranked "
            f"{pair} sit in that intersection; {leader_tk} carries the highest desk score."
        )

    if any_large_top3 and len(leaders_in_overlap) >= 1 and overlap_n >= 2:
        return (
            "High-dollar congressional ranges coincide with insider activity on overlap names this run; "
            f"{leader_tk} leads the composite desk read among those cross-feed stories."
        )

    if max_reps >= 5:
        return (
            f"Unusually broad Capitol participation—up to {max_reps} distinct representatives on a leading symbol—"
            f"shapes the desk story, with {leader_tk} at the top of the ranking."
        )

    if overlap_n >= 5 and len(leaders_in_overlap) >= 1:
        return (
            f"With {overlap_n} overlapping tickers, the run rewards names that appear in both disclosure feeds; "
            f"{leader_tk} is the strongest ranked entry."
        )

    # Default: name top names + directional hint
    pair = " and ".join(top3_tk[:2])
    extra = ""
    if len(bearish_leaders) >= 2:
        extra = (
            f" Insider and rule-based labels skew bearish on {' and '.join(bearish_leaders[:2])} among ranked leaders."
        )
    elif len(bullish_leaders) >= 2:
        extra = (
            f" Bullish tagging is concentrated on {' and '.join(bullish_leaders[:2])} in the ranked set."
        )
    elif len(bearish_leaders) == 1 and not extra:
        extra = f" The bearish read is most pronounced on {bearish_leaders[0]}."
    elif len(bullish_leaders) == 1 and not extra:
        extra = f" The bullish read is most pronounced on {bullish_leaders[0]}."

    return f"This pull’s strongest desk scores center on {pair}, led by {leader_tk}.{extra}".strip()


def _html_key_insight_callout(signals: SignalsBundle | dict[str, Any]) -> str:
    """Top-of-page synthesis; prefers Claude dashboard pass when present."""
    ovr = signals.get("_dash_claude_key_insight")
    if isinstance(ovr, str) and ovr.strip():
        text = ovr.strip()
    else:
        text = _key_insight_for_dashboard(signals)
    body = html.escape(text)
    return (
        '<aside class="key-insight-callout" id="key-insight" aria-labelledby="key-insight-label">'
        '<span class="key-insight-callout__label" id="key-insight-label">Key Insight</span>'
        f'<p class="key-insight-callout__text">{body}</p>'
        "</aside>"
    )


def _what_changed_this_run_bullets(signals: SignalsBundle | dict[str, Any]) -> list[str]:
    """
    Up to three plain-English, data-grounded lines for the dashboard “What changed” strip.
    Uses only fields already produced by the MVP (congress/insider frames, overlap list, large trades).
    """
    from signal_logic import get_congressional_clusters, get_insider_activity_by_ticker

    cdf = signals.get("congress_df")
    idf = signals.get("insider_df")
    large = signals.get("large_trades")
    overlap_signals = list(signals.get("overlap_signals") or [])

    if not isinstance(cdf, pd.DataFrame):
        cdf = pd.DataFrame()
    if not isinstance(idf, pd.DataFrame):
        idf = pd.DataFrame()
    if not isinstance(large, pd.DataFrame):
        large = pd.DataFrame()

    if bool(signals.get("insider_only_fallback_mode")):
        wl = [w for w in (signals.get("insider_fallback_watchlist") or []) if isinstance(w, dict)]
        out_fc: list[str] = []
        for w in wl[:3]:
            sl = str(w.get("summary_line", "")).strip()
            if sl:
                out_fc.append(sl)
        n_ins = int(signals.get("insider_row_count", 0))
        pad = (
            f"The insider feed returned {n_ins} row(s) in this download; desk scoring and overlap require Capitol data."
        )
        while len(out_fc) < 3:
            if pad not in out_fc:
                out_fc.append(pad)
            else:
                break
        return out_fc[:3]

    bullets: list[str] = []
    used_tickers: set[str] = set()

    def _primary_ticker_from_text(tk: str) -> str:
        return str(tk).strip().upper()

    def _add_bullet(text: str, ticker_key: str) -> None:
        if len(bullets) >= 3:
            return
        tku = _primary_ticker_from_text(ticker_key)
        if tku and tku in used_tickers:
            return
        bullets.append(text)
        if tku:
            used_tickers.add(tku)

    # 1) Busiest congressional symbol by raw row count in this pull
    if not cdf.empty and "Ticker" in cdf.columns:
        s = cdf["Ticker"].astype(str).str.strip()
        s = s[(s != "") & s.str.lower().ne("nan")]
        if not s.empty:
            vc = s.value_counts()
            tk = str(vc.index[0])
            n = int(vc.iloc[0])
            ur = 0
            if "Representative" in cdf.columns:
                ur = int(cdf.loc[cdf["Ticker"].astype(str).str.strip() == tk, "Representative"].nunique())
            if n >= 2:
                if ur >= 1:
                    line = (
                        f"{tk} is the busiest congressional symbol in this run with {n} filings "
                        f"across {ur} representatives."
                    )
                else:
                    line = f"{tk} is the busiest congressional symbol in this run with {n} filings."
                _add_bullet(line, tk)

    # 2) Strongest insider buy/sell skew (requires meaningful volume on the dominant side)
    act = get_insider_activity_by_ticker(idf)
    best: tuple[int, str, int, int, str] | None = None  # gap, ticker, buys, sells, side
    if not act.empty:
        for _, row in act.iterrows():
            tkr = str(row.get("Ticker", "")).strip()
            if not tkr:
                continue
            b = int(row.get("Buys", 0) or 0)
            s2 = int(row.get("Sells", 0) or 0)
            if s2 >= 5 and s2 > b:
                gap = s2 - b
                if gap >= max(4, s2 // 5):
                    if best is None or gap > best[0]:
                        best = (gap, tkr, b, s2, "sell")
            if b >= 5 and b > s2:
                gap = b - s2
                if gap >= max(4, b // 5):
                    if best is None or gap > best[0]:
                        best = (gap, tkr, b, s2, "buy")
    if best is not None and len(bullets) < 3:
        _gap, tk, b, s2, side = best
        if side == "sell":
            line = (
                f"{tk} shows a strong insider sell skew, with {s2} sell-like rows versus {b} buy-like rows."
            )
        else:
            line = (
                f"{tk} shows a strong insider buy skew, with {b} buy-like rows versus {s2} sell-like rows."
            )
        _add_bullet(line, tk)

    # 3) Cross-dataset overlap (try ranked overlap list; skip tickers already used)
    for o in overlap_signals:
        if len(bullets) >= 3:
            break
        if not isinstance(o, dict):
            continue
        tk = str(o.get("ticker", "")).strip().upper()
        if not tk:
            continue
        if tk in used_tickers:
            continue
        cc = int(o.get("congress_filing_count") or 0)
        ir = int(o.get("insider_rows") or 0)
        if cc < 1 or ir < 1:
            continue
        ur_o = int(o.get("unique_representatives") or 0)
        large_f = bool(o.get("large_dollar_congress"))
        if large_f:
            line = (
                f"{tk} is one of the strongest overlap signals this run, combining a large congressional "
                "trade with insider activity."
            )
        elif ur_o >= 3:
            line = (
                f"{tk} is a standout overlap name: {cc} congressional lines from {ur_o} representatives "
                f"and {ir} insider rows in the same pull."
            )
        else:
            line = (
                f"{tk} ranks high on the overlap list with {cc} congressional lines and {ir} insider rows."
            )
        _add_bullet(line, tk)

    # 4) High-dollar congressional disclosure (first conviction-ranked large-trade row)
    if len(bullets) < 3 and not large.empty and "Ticker" in large.columns:
        for _, lrow in large.head(5).iterrows():
            tkr = str(lrow.get("Ticker", "")).strip().upper()
            if not tkr or tkr in used_tickers:
                continue
            _add_bullet(
                f"{tkr} appears among high-dollar congressional disclosures this run (wide reported Range bands).",
                tkr,
            )
            break

    # 5) Unusually broad representative breadth (cluster table, already MVP)
    if len(bullets) < 3:
        cl = get_congressional_clusters(cdf)
        if not cl.empty and "Ticker" in cl.columns:
            for _, crow in cl.head(5).iterrows():
                tk = str(crow.get("Ticker", "")).strip().upper()
                ur = int(crow.get("UniqueRepresentatives", 0) or 0)
                if not tk or ur < 5 or tk in used_tickers:
                    continue
                cnt = int((cdf["Ticker"].astype(str).str.strip() == tk).sum()) if not cdf.empty else 0
                if cnt >= 2:
                    line = (
                        f"{tk} shows unusually broad Capitol breadth: {ur} distinct representatives "
                        f"with filings in this run ({cnt} congressional lines)."
                    )
                else:
                    line = (
                        f"{tk} shows unusually broad Capitol breadth: {ur} distinct representatives "
                        "touched the symbol this run."
                    )
                _add_bullet(line, tk)
                break

    return bullets


def _what_changed_display_bullets(signals: SignalsBundle | dict[str, Any]) -> list[str]:
    """Same sources as the What Changed HTML strip (dash override or deterministic)."""
    ovr = signals.get("_dash_claude_what_changed")
    if (
        isinstance(ovr, list)
        and len(ovr) == 3
        and all(isinstance(x, str) and str(x).strip() for x in ovr)
    ):
        return [str(x).strip() for x in ovr]
    return _what_changed_this_run_bullets(signals)


def _newsletter_display_bullets(signals: SignalsBundle | dict[str, Any]) -> list[str]:
    """Same sources as the newsletter HTML strip (dash override or deterministic)."""
    ovr = signals.get("_dash_claude_newsletter")
    if (
        isinstance(ovr, list)
        and 3 <= len(ovr) <= 5
        and all(isinstance(x, str) and str(x).strip() for x in ovr)
    ):
        return [str(x).strip() for x in ovr]
    return _executive_copy_bullets(signals)


def _lines_from_ai_plain_text(text: str, *, max_items: int, min_items: int | None = None) -> list[str]:
    """Split model plain text into list items (strips leading bullets / numbering)."""
    lines: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^[-*•]\s*", "", s)
        s = re.sub(r"^\d+\.\s*", "", s)
        if s:
            lines.append(s)
        if len(lines) >= max_items:
            break
    if min_items is not None and len(lines) < min_items:
        return []
    return lines


def _html_what_changed_this_run_section(signals: SignalsBundle | dict[str, Any]) -> str:
    """Compact strip between hero anomalies and newsletter insights; omitted when no bullets qualify."""
    ai_b = signals.get("_ai_what_changed_bullets")
    if (
        isinstance(ai_b, list)
        and len(ai_b) == 3
        and all(isinstance(x, str) and str(x).strip() for x in ai_b)
    ):
        bullets = [str(x).strip() for x in ai_b]
    else:
        bullets = _what_changed_display_bullets(signals)
    if not bullets:
        return ""
    h2_id = "what-changed-this-run-h"
    lis = "".join(f"<li>{html.escape(t)}</li>" for t in bullets)
    return (
        '<section class="what-changed-run" id="what-changed-this-run" '
        f'aria-labelledby="{h2_id}">'
        f'<h2 class="what-changed-run__title" id="{h2_id}">What Changed This Run</h2>'
        '<p class="what-changed-run__dek">Snapshot of what stands out in this pull—numbers come straight '
        "from the datasets below.</p>"
        f'<ul class="what-changed-run__list">{lis}</ul>'
        "</section>"
    )


def _html_quick_actions_bar(*, include_what_changed: bool = True) -> str:
    """Compact in-page nav; targets section ids generated elsewhere in the dashboard."""
    pairs: list[tuple[str, str]] = [
        ("View top anomalies", "#top-anomalies"),
    ]
    if include_what_changed:
        pairs.append(("What changed this run", "#what-changed-this-run"))
    pairs.extend(
        [
            ("Jump to ranked signals", "#top-ranked"),
            ("Jump to overlap signals", "#overlap"),
            ("Jump to high-profile trades", "#high-profile"),
            ("Ask AI Assistant", "#ask-claude-panel"),
            ("Newsletter-ready insights", "#newsletter-ready-insights"),
        ]
    )
    links = "".join(
        f'<a class="quick-action" href="{html.escape(href, quote=True)}">{html.escape(label)}</a>'
        for label, href in pairs
    )
    return (
        '<nav class="quick-actions" aria-label="Quick actions">'
        '<span class="quick-actions-label">Quick actions</span>'
        f'<div class="quick-actions-row">{links}</div>'
        "</nav>"
    )


def _html_congress_unavailable_banner(signals: SignalsBundle | dict[str, Any]) -> str:
    if not bool(signals.get("insider_only_fallback_mode")):
        return ""
    msg = (
        "Congressional data was unavailable for this run, so rankings and top anomalies are limited."
    )
    return (
        '<aside class="congress-unavailable-banner" role="alert" aria-live="polite">'
        f'<p class="congress-unavailable-banner__text">{html.escape(msg)}</p>'
        "</aside>"
    )


def ensure_ranked_distinctiveness(signals: SignalsBundle | dict[str, Any]) -> None:
    """
    Attach ``distinctiveness_bonus`` / ``distinctiveness_components`` to each ranked row when missing,
    using only structured evidence + ``congress_df`` (see ``signal_logic.compute_ranked_row_distinctiveness``).
    """
    from signal_logic import refresh_distinctiveness_on_ranked

    if bool(signals.get("insider_only_fallback_mode")):
        return
    ranked_raw = signals.get("ranked_signals")
    if not isinstance(ranked_raw, list) or not ranked_raw:
        return
    ranked = [r for r in ranked_raw if isinstance(r, dict)]
    if not ranked:
        return
    cdf = signals.get("congress_df", pd.DataFrame())
    if not isinstance(cdf, pd.DataFrame):
        cdf = pd.DataFrame()
    refresh_distinctiveness_on_ranked(ranked, cdf)


def ensure_ranked_story_worthiness(signals: SignalsBundle | dict[str, Any]) -> None:
    """
    Attach ``story_worthiness_score`` (0–5) and ``story_worthiness_components`` to each ranked row (recomputed each time),
    using structured evidence + ``congress_df`` + overlap (see ``signal_logic.attach_story_worthiness_to_ranked``).

    Desk composite ``score`` is unchanged; this layer only affects qualified/hero ordering and newsletter-facing context.
    """
    from signal_logic import attach_story_worthiness_to_ranked

    if bool(signals.get("insider_only_fallback_mode")):
        return
    ranked_raw = signals.get("ranked_signals")
    if not isinstance(ranked_raw, list) or not ranked_raw:
        return
    ranked = [r for r in ranked_raw if isinstance(r, dict)]
    if not ranked:
        return
    cdf = signals.get("congress_df", pd.DataFrame())
    if not isinstance(cdf, pd.DataFrame):
        cdf = pd.DataFrame()
    overlap = signals.get("overlap") or []
    attach_story_worthiness_to_ranked(ranked, cdf, overlap)


def ensure_dashboard_anomaly_views(signals: SignalsBundle | dict[str, Any]) -> None:
    """
    Populate ``qualified_ranked_signals`` and ``hero_ranked_signals`` for tightened top-of-dashboard use.
    Full ``ranked_signals`` is unchanged for tables and markdown (desk order preserved there).

    Calls ``ensure_ranked_distinctiveness`` then ``ensure_ranked_story_worthiness`` before computing views.
    """
    from signal_logic import compute_dashboard_anomaly_views

    if bool(signals.get("insider_only_fallback_mode")):
        signals["qualified_ranked_signals"] = []
        signals["hero_ranked_signals"] = []
        return

    ensure_ranked_distinctiveness(signals)
    ensure_ranked_story_worthiness(signals)
    ranked = [r for r in (signals.get("ranked_signals") or []) if isinstance(r, dict)]
    cdf = signals.get("congress_df", pd.DataFrame())
    if not isinstance(cdf, pd.DataFrame):
        cdf = pd.DataFrame()
    overlap = signals.get("overlap") or []
    qual, hero = compute_dashboard_anomaly_views(ranked, cdf, overlap)
    signals["qualified_ranked_signals"] = qual
    signals["hero_ranked_signals"] = hero


def _top_ranked_hero_leaders(signals: SignalsBundle | dict[str, Any]) -> list[dict[str, Any]]:
    hero = signals.get("hero_ranked_signals")
    if isinstance(hero, list):
        return [x for x in hero if isinstance(x, dict) and str(x.get("ticker", "")).strip()]
    ranked_signals = list(signals.get("ranked_signals") or [])
    leaders: list[dict[str, Any]] = []
    for item in ranked_signals:
        if isinstance(item, dict) and str(item.get("ticker", "")).strip():
            leaders.append(item)
        if len(leaders) >= 3:
            break
    return leaders


def _collect_hero_explanations_raw(leaders: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for row in leaders:
        tk_raw = str(row.get("ticker", "")).strip()
        title_line = _hero_card_title_line(row)
        bullets = list(_hero_evidence_bullets(row))
        triggers = row.get("why_triggered") or []
        if len(bullets) < 2 and len(triggers) > 1:
            t2 = str(triggers[1]).strip()
            if len(t2) > 130:
                t2 = t2[:127].rsplit(" ", 1)[0] + "…"
            if t2 and t2 != title_line:
                bullets.append(t2)
        bullets = bullets[:3]
        bl_txt = "\n".join(f"- {b}" for b in bullets) if bullets else "(no evidence bullets)"
        chunks.append(f"{tk_raw}\n{title_line}\n{bl_txt}")
    return "\n\n".join(chunks)


def _collect_hero_why_it_leads_raw(leaders: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in leaders:
        tk_raw = str(row.get("ticker", "")).strip()
        why = _hero_why_matters_line(row)
        lines.append(f"{tk_raw}: {why}")
    return "\n".join(lines)


def _collect_single_hero_card_raw(row: dict[str, Any]) -> str:
    """One hero anomaly card: title, evidence bullets, optional second trigger, why-it-matters (for a single Claude call)."""
    tk_raw = str(row.get("ticker", "")).strip()
    title_line = _hero_card_title_line(row)
    bullets = list(_hero_evidence_bullets(row))
    triggers = row.get("why_triggered") or []
    if len(bullets) < 2 and len(triggers) > 1:
        t2 = str(triggers[1]).strip()
        if len(t2) > 130:
            t2 = t2[:127].rsplit(" ", 1)[0] + "…"
        if t2 and t2 != title_line:
            bullets.append(t2)
    bullets = bullets[:3]
    bl_txt = "\n".join(f"- {b}" for b in bullets) if bullets else "(no evidence bullets)"
    why = _hero_why_matters_line(row)
    return (
        f"Ticker: {tk_raw}\n"
        f"Headline / title line: {title_line}\n"
        f"Evidence:\n{bl_txt}\n"
        f"Why it matters (desk framing): {why}"
    )


def _collect_insider_watchlist_single_raw(w: dict[str, Any]) -> str:
    tk = str(w.get("ticker", "")).strip()
    title_line = str(w.get("title_line", "")).strip()
    parts = [
        str(w.get("skew_summary", "")).strip(),
        str(w.get("recency_note", "")).strip(),
        str(w.get("selection_note", "")).strip(),
    ]
    bullets = [b for b in parts if b]
    bl_txt = "\n".join(f"- {b}" for b in bullets[:4]) if bullets else "(no detail lines)"
    return f"Ticker: {tk}\nTitle: {title_line}\nDetails:\n{bl_txt}"


def _slim_insider_watchlist_for_context(w: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": str(w.get("ticker", "")).strip().upper(),
        "title_line": str(w.get("title_line", "")).strip()[:500],
        "skew_summary": str(w.get("skew_summary", "")).strip()[:400],
        "recency_note": str(w.get("recency_note", "")).strip()[:400],
        "selection_note": str(w.get("selection_note", "")).strip()[:400],
        "newsletter_line": str(w.get("newsletter_line", "")).strip()[:400],
    }


def _collect_insider_watchlist_raw(wl: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for w in wl[:3]:
        tk = str(w.get("ticker", "")).strip()
        title_line = str(w.get("title_line", "")).strip()
        bl_parts = [
            str(w.get("skew_summary", "")).strip(),
            str(w.get("recency_note", "")).strip(),
            str(w.get("selection_note", "")).strip(),
        ]
        bullets = [b for b in bl_parts if b]
        bl_txt = "\n".join(f"- {b}" for b in bullets[:4]) if bullets else ""
        parts.append(f"{tk}\n{title_line}\n{bl_txt}".strip())
    return "\n\n".join(parts)


def _html_dashboard_ai_strip(
    body_text: str,
    *,
    aria_label: str,
    heading: str | None = None,
) -> str:
    esc = html.escape(body_text.strip())
    h = ""
    if heading:
        h = f'<h3 class="dashboard-ai-strip__title">{html.escape(heading)}</h3>'
    elab = html.escape(aria_label)
    return (
        f'<div class="dashboard-ai-strip" role="region" aria-label="{elab}">'
        f"{h}"
        f'<div class="dashboard-ai-strip__body">{esc}</div>'
        "</div>"
    )


def _html_insider_only_watchlist_hero(signals: SignalsBundle | dict[str, Any]) -> str:
    h2_id = "top-anomalies-this-run-h"
    wl = [w for w in (signals.get("insider_fallback_watchlist") or []) if isinstance(w, dict)]
    dek = (
        '<p class="hero-anomalies-dek insider-only-watchlist-dek">'
        "Generated from insider data only because congressional data was unavailable."
        "</p>"
    )
    if not wl:
        return f"""<section class="hero-anomalies-wrap insider-only-watchlist-wrap" id="top-anomalies" aria-labelledby="{h2_id}">
  <h2 class="hero-anomalies-title" id="{h2_id}">Insider-Only Watchlist</h2>
  {dek}
  <p class="hero-anomalies-empty muted">No insider symbols met the criteria for this fallback list.</p>
</section>"""

    card_ai_list = signals.get("_ai_insider_hero_card_texts")
    if not isinstance(card_ai_list, list):
        card_ai_list = []
    minimal_insider = bool(signals.get("_ai_insider_hero_minimal_cards"))

    cards: list[str] = []
    for idx, w in enumerate(wl[:3]):
        tk_raw = str(w.get("ticker", "")).strip()
        tk = html.escape(tk_raw)
        badge = (
            '<span class="insider-fallback-badge" title="No desk score without congressional data">Insider fallback</span>'
        )
        card_ai_html = ""
        if idx < len(card_ai_list) and str(card_ai_list[idx]).strip():
            card_ai_html = _html_dashboard_ai_strip(
                str(card_ai_list[idx]).strip(),
                aria_label=f"Refined insider watchlist context for {tk_raw}",
                heading=None,
            )
        if minimal_insider:
            cards.append(
                f'''<article class="anomaly-hero-card anomaly-hero-card--insider-fallback anomaly-hero-card--minimal">
  <div class="anomaly-hero-card-head">
    <span class="anomaly-hero-ticker">{tk}</span>
    <div class="anomaly-hero-meta">{badge}</div>
  </div>
  {card_ai_html}
</article>'''
            )
            continue
        title_line = _format_inline_md(str(w.get("title_line", "")))
        bl_parts = [
            str(w.get("skew_summary", "")).strip(),
            str(w.get("recency_note", "")).strip(),
            str(w.get("selection_note", "")).strip(),
        ]
        bullets = [b for b in bl_parts if b]
        bl_html = "".join(f"<li>{_format_inline_md(b)}</li>" for b in bullets[:4])
        cards.append(
            f'''<article class="anomaly-hero-card anomaly-hero-card--insider-fallback">
  <div class="anomaly-hero-card-head">
    <span class="anomaly-hero-ticker">{tk}</span>
    <div class="anomaly-hero-meta">{badge}</div>
  </div>
  {card_ai_html}
  <p class="anomaly-hero-card-title">{title_line}</p>
  <div class="evidence-panel evidence-panel--hero">
    <ul class="anomaly-hero-evidence">{bl_html}</ul>
  </div>
</article>'''
        )

    grid = f'<div class="anomaly-hero-grid">{"".join(cards)}</div>'
    return f"""<section class="hero-anomalies-wrap insider-only-watchlist-wrap" id="top-anomalies" aria-labelledby="{h2_id}">
  <h2 class="hero-anomalies-title" id="{h2_id}">Insider-Only Watchlist</h2>
  {dek}
  {grid}
</section>"""


def _html_top_anomalies_hero_section(signals: SignalsBundle | dict[str, Any]) -> str:
    """
    Prominent top-of-page cards from the strongest **Top Ranked** signals (max 3),
    or insider-only fallback cards when congressional data is missing.
    """
    if bool(signals.get("insider_only_fallback_mode")):
        return _html_insider_only_watchlist_hero(signals)

    h2_id = "top-anomalies-this-run-h"
    intro = (
        '<p class="hero-anomalies-dek">The strongest desk-ranked names from this API run—read details in the sections below.</p>'
    )

    leaders = _top_ranked_hero_leaders(signals)

    if not leaders:
        raw_ranked = [r for r in (signals.get("ranked_signals") or []) if isinstance(r, dict) and str(r.get("ticker", "")).strip()]
        if raw_ranked:
            empty_msg = (
                "No tickers met the tightened hero bar this pull (≥3 anomaly dimensions desk-wide, and ≥4 dimensions "
                "or an exceptional hook such as a high-profile purchase, large Capitol line with insider overlap, "
                "or very broad member breadth with heavy insider rows). Fewer cards are shown rather than padding with "
                "weaker names—see the full ranked table below."
            )
        else:
            empty_msg = (
                "No ranked signals this pull—congressional data was empty or unavailable, so composite desk scores "
                "were not computed. Use the raw sections below or re-run when Capitol data is available."
            )
        return f"""<!--
  When hero cards are shown, order follows Story-Worthiness (0–5) for newsletter fit; desk score on each card is unchanged.
  Strong desk anomalies can be weak stories and vice versa—see internal story_worthiness_score on ranked rows.
-->
<section class="hero-anomalies-wrap" id="top-anomalies" aria-labelledby="{h2_id}">
  <h2 class="hero-anomalies-title" id="{h2_id}">Top Anomalies This Run</h2>
  {intro}
  <p class="hero-anomalies-empty muted">{html.escape(empty_msg)}</p>
</section>"""

    card_ai_list = signals.get("_ai_hero_card_texts")
    if not isinstance(card_ai_list, list):
        card_ai_list = []
    minimal_cards = bool(signals.get("_ai_hero_minimal_cards"))

    cards: list[str] = []
    for idx, row in enumerate(leaders):
        tk_raw = str(row.get("ticker", "")).strip()
        tk_tok = _ticker_scroll_token(tk_raw)
        tk_attr = html.escape(tk_tok)
        tk = html.escape(tk_raw)
        sc = _desk_score_str(row.get("score", 0))
        ev_h = row.get("evidence") or {}
        if not isinstance(ev_h, dict):
            ev_h = {}
        d_raw = str(row.get("direction") or ev_h.get("direction", "Mixed")).strip()
        dir_tag = _dir_tag_html(d_raw)
        card_ai_html = ""
        if idx < len(card_ai_list) and str(card_ai_list[idx]).strip():
            card_ai_html = _html_dashboard_ai_strip(
                str(card_ai_list[idx]).strip(),
                aria_label=f"Refined narrative for {tk_raw}",
                heading=None,
            )
        has_card_ai = idx < len(card_ai_list) and bool(str(card_ai_list[idx]).strip())
        if minimal_cards:
            why_fb = ""
            if not has_card_ai:
                why_fb = (
                    "<p class=\"anomaly-hero-why\"><strong>Why it matters:</strong> "
                    f"{_format_inline_md(_hero_why_matters_line(row))}</p>"
                )
            cards.append(
                f'''<article class="anomaly-hero-card top-card anomaly-hero-card--minimal" data-target="{tk_attr}">
  <div class="anomaly-hero-card-head">
    <span class="anomaly-hero-ticker">{tk}</span>
    <div class="anomaly-hero-meta">
      <span class="score-badge" title="Composite desk score (0.0–10.0). Card order uses separate Story-Worthiness (0–5) for newsletter fit.">{sc}/10</span>
      {dir_tag}
    </div>
  </div>
  {card_ai_html}
  {why_fb}
</article>'''
            )
            continue
        title_line = _hero_card_title_line(row)
        bullets = list(_hero_evidence_bullets(row))
        triggers = row.get("why_triggered") or []
        if len(bullets) < 2 and len(triggers) > 1:
            t2 = str(triggers[1]).strip()
            if len(t2) > 130:
                t2 = t2[:127].rsplit(" ", 1)[0] + "…"
            if t2 and t2 != title_line:
                bullets.append(t2)
        bullets = bullets[:3]
        bl_html = "".join(
            f"<li>{_format_inline_md(b)}</li>" for b in bullets
        )
        why = _format_inline_md(_hero_why_matters_line(row))
        cards.append(
            f'''<article class="anomaly-hero-card top-card" data-target="{tk_attr}">
  <div class="anomaly-hero-card-head">
    <span class="anomaly-hero-ticker">{tk}</span>
    <div class="anomaly-hero-meta">
      <span class="score-badge" title="Composite desk score (0.0–10.0). Card order uses separate Story-Worthiness (0–5) for newsletter fit.">{sc}/10</span>
      {dir_tag}
    </div>
  </div>
  {card_ai_html}
  <p class="anomaly-hero-card-title">{_format_inline_md(title_line)}</p>
  <div class="evidence-panel evidence-panel--hero">
    <ul class="anomaly-hero-evidence">{bl_html}</ul>
  </div>
  <p class="anomaly-hero-why"><strong>Why it matters:</strong> {why}</p>
</article>'''
        )

    grid = f'<div class="anomaly-hero-grid">{"".join(cards)}</div>'
    return f"""<!--
  Story-Worthiness (0–5) orders these cards for newsletter fit; the /10 badge is the unchanged desk composite.
  Some tickers are strong desk signals but weak stories; others read well despite a lower desk score.
-->
<section class="hero-anomalies-wrap" id="top-anomalies" aria-labelledby="{h2_id}">
  <h2 class="hero-anomalies-title" id="{h2_id}">Top Anomalies This Run</h2>
  {intro}
  {grid}
</section>"""


def _executive_copy_bullets(signals: SignalsBundle | dict[str, Any]) -> list[str]:
    """
    3–5 copy-ready lines for newsletter drafting—same numbers as ranked evidence (no invented filings).
    Uses qualified desk-ranked rows when present (tightened anomaly bar), in **Story-Worthiness** order
    (then distinctiveness, desk score)—not raw desk-table order.
    """
    ranked_raw = signals.get("ranked_signals") or []
    qsig = signals.get("qualified_ranked_signals")
    if isinstance(qsig, list):
        ranked = [r for r in qsig if isinstance(r, dict) and str(r.get("ticker", "")).strip()]
    else:
        ranked = [r for r in ranked_raw if isinstance(r, dict) and str(r.get("ticker", "")).strip()]
    large = signals.get("large_trades", pd.DataFrame())
    if not isinstance(large, pd.DataFrame):
        large = pd.DataFrame()
    overlap = list(signals.get("overlap") or [])
    ov_set = {str(x).strip().upper() for x in overlap if str(x).strip()}
    n_congress = int(signals.get("congress_row_count", 0))
    n_insider = int(signals.get("insider_row_count", 0))
    n_large = len(large)

    bullets: list[str] = []

    if bool(signals.get("insider_only_fallback_mode")):
        wl = [w for w in (signals.get("insider_fallback_watchlist") or []) if isinstance(w, dict)]
        tickers_sample = [
            str(w.get("ticker", "")).strip().upper() for w in wl if str(w.get("ticker", "")).strip()
        ][:3]
        t_join = ", ".join(tickers_sample) if tickers_sample else "several issuers"
        lines_io = [
            (
                "Congressional data was unavailable in this run, but insider activity was still dense "
                f"in names like {t_join}."
            ),
            "This output should be treated as an incomplete snapshot until Capitol data is restored.",
        ]
        for w in wl:
            nl = str(w.get("newsletter_line", "")).strip()
            if nl and nl not in lines_io and len(lines_io) < 5:
                lines_io.append(nl)
        if len(lines_io) < 3 and n_insider > 0:
            lines_io.append(
                f"The insider download contained {n_insider} row(s); use the Insider-Only Watchlist for the busiest symbols."
            )
        return lines_io[:5]

    for row in ranked[:5]:
        t = str(row.get("ticker", "")).strip().upper()
        sc = _desk_score_str(row.get("score", 0))
        ev = row.get("evidence") or {}
        if not isinstance(ev, dict):
            ev = {}
        cf = int(ev.get("congress_filings") or 0)
        ur = int(ev.get("unique_representatives") or 0)
        ins = int(ev.get("insider_rows") or 0)
        large_hit = bool(ev.get("large_congress_trade"))
        d = str(row.get("direction") or ev.get("direction", "Mixed")).strip()

        line = (
            f"{t} — desk score {sc}/10 this pull: {cf} congressional row(s) across {ur} member(s), "
            f"{ins} insider row(s) in this download."
        )

        extra: str | None = None
        if t in ov_set:
            extra = "Also in both feeds this run—easy cross-check for a one-line Quiver hook."
        if extra is None and large_hit:
            extra = "Capitol side includes at least one row that matched the large-dollar Range filter here."
        if extra is None and d == "Bullish":
            extra = "Desk tags it Bullish off parsed buy/sell rules (sanity-check before you lean on tone)."
        if extra is None and d == "Bearish":
            extra = "Desk tags it Bearish off parsed buy/sell rules (sanity-check before you lean on tone)."
        if extra is None:
            try:
                cac = ev.get("contract_activity_count")
                if cac is not None and int(cac) > 0:
                    extra = f"Contracts add-on: {int(cac)} row(s) on this ticker in the same run."
            except (TypeError, ValueError):
                pass
        if extra is None:
            try:
                lac = ev.get("lobbying_activity_count")
                if lac is not None and int(lac) > 0:
                    extra = f"Lobbying add-on: {int(lac)} row(s) on this ticker in the same run."
            except (TypeError, ValueError):
                pass

        if extra:
            line = f"{line} {extra}"
        if len(line) > 300:
            line = line[:297].rsplit(" ", 1)[0] + "…"
        bullets.append(line)

    while len(bullets) < 3:
        added = False
        if n_congress > 0 or n_insider > 0:
            s = (
                f"Pull size check: {n_congress} congressional row(s) and {n_insider} insider row(s) "
                "in the download before desk scoring."
            )
            if s not in bullets:
                bullets.append(s)
                added = True
        if not added and len(overlap) > 0:
            s = (
                f"{len(overlap)} symbol(s) land in both feeds this run—open Overlap below for ranked framing."
            )
            if s not in bullets:
                bullets.append(s)
                added = True
        if not added and n_large > 0:
            s = (
                f"{n_large} Capitol row(s) cleared the large-dollar Range filter—detail tables may cap rows shown."
            )
            if s not in bullets:
                bullets.append(s)
                added = True
        if not added:
            break

    if not bullets:
        bullets.append(
            "No ranked tickers to lift this week—the candidate set needs stronger congress-side coverage."
        )
        if n_congress > 0 or n_insider > 0:
            bullets.append(
                f"Still worth noting the pull size: {n_congress} congressional and {n_insider} insider rows."
            )

    seen: set[str] = set()
    out: list[str] = []
    for b in bullets:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out[:5]


def _html_executive_copy_block(signals: SignalsBundle | dict[str, Any]) -> str:
    """Newsletter-oriented strip below hero anomalies; prefers section AI enhancement, then dash bullets."""
    ai_n = signals.get("_ai_newsletter_bullets")
    if (
        isinstance(ai_n, list)
        and 3 <= len(ai_n) <= 5
        and all(isinstance(x, str) and str(x).strip() for x in ai_n)
    ):
        lines = [str(x).strip() for x in ai_n]
    else:
        lines = _newsletter_display_bullets(signals)
    lis = "".join(f"<li>{html.escape(t)}</li>" for t in lines)
    return (
        '<section class="executive-copy-block" id="newsletter-ready-insights" '
        'aria-labelledby="newsletter-ready-insights-h">'
        '<h2 class="executive-copy-title" id="newsletter-ready-insights-h">Newsletter-Ready Insights</h2>'
        '<p class="executive-copy-dek">Short, copy-ready lines built from this run’s strongest signals. '
        "Use these as a drafting base, then verify primary filings before publication.</p>"
        f'<ul class="executive-copy-list">{lis}</ul>'
        "</section>"
    )


def _directional_bucket_tickers(
    ranked_signals: list[dict[str, Any]],
    overlap_signals: list[dict[str, Any]],
    *,
    max_per_bucket: int = 12,
) -> tuple[list[str], list[str], list[str]]:
    """Same bucketing as the directional summary section; ticker lists only."""
    bullish: list[str] = []
    bearish: list[str] = []
    mixed: list[str] = []
    seen: set[str] = set()

    def _consume(row: dict[str, Any]) -> None:
        t = str(row.get("ticker", "")).strip().upper()
        if not t:
            return
        if t in seen:
            return
        seen.add(t)
        d = str(row.get("direction", "Mixed")).strip()
        if d == "Bullish" and len(bullish) < max_per_bucket:
            bullish.append(t)
        elif d == "Bearish" and len(bearish) < max_per_bucket:
            bearish.append(t)
        elif len(mixed) < max_per_bucket:
            mixed.append(t)

    for row in ranked_signals:
        if isinstance(row, dict):
            _consume(row)
    for row in overlap_signals:
        if isinstance(row, dict):
            _consume(row)
    return bullish, bearish, mixed


def _data_availability_plain_lines(signals: SignalsBundle | dict[str, Any]) -> list[str]:
    """Strip markdown emphasis from availability bullets for JSON embedding."""
    md = _data_availability_section(signals)
    lines: list[str] = []
    for line in md.split("\n"):
        s = line.strip()
        if s.startswith("- "):
            t = s[2:].replace("**", "")
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                lines.append(t)
    return lines


def _slim_ranked_for_context(row: dict[str, Any]) -> dict[str, Any]:
    ev_in = row.get("evidence")
    ev = ev_in if isinstance(ev_in, dict) else {}
    wt = row.get("why_triggered") or []
    wt_out = [str(x).strip() for x in wt[:2] if str(x).strip()]
    se: dict[str, Any] = {}
    for k in ("congress_filings", "unique_representatives", "insider_rows", "insider_buys", "insider_sells"):
        if k in ev and ev.get(k) is not None:
            se[k] = ev.get(k)
    se["large_congress_trade"] = bool(ev.get("large_congress_trade"))
    if ev.get("congress_conviction_0_100") is not None:
        se["congress_conviction_0_100"] = ev.get("congress_conviction_0_100")
    if ev.get("contract_activity_count") is not None:
        se["contract_activity_count"] = ev.get("contract_activity_count")
    if ev.get("lobbying_activity_count") is not None:
        se["lobbying_activity_count"] = ev.get("lobbying_activity_count")
    out: dict[str, Any] = {
        "ticker": str(row.get("ticker", "")).strip().upper(),
        "score": float(row.get("score", 0) or 0),
        "direction": str(row.get("direction") or ev.get("direction", "Mixed")).strip(),
        "why_triggered": wt_out,
        "evidence": se,
    }
    scmp = row.get("score_components")
    if isinstance(scmp, dict) and scmp:
        out["score_components"] = {
            k: round(float(v), 1) for k, v in scmp.items() if isinstance(v, (int, float))
        }
    db = row.get("distinctiveness_bonus")
    if isinstance(db, (int, float)):
        out["distinctiveness_bonus"] = round(float(db), 4)
    dc = row.get("distinctiveness_components")
    if isinstance(dc, dict) and dc:
        out["distinctiveness_components"] = {
            str(k): round(float(v), 4)
            for k, v in dc.items()
            if isinstance(v, (int, float))
        }
    sw = row.get("story_worthiness_score")
    if isinstance(sw, (int, float)):
        out["story_worthiness_score"] = round(float(sw), 1)
    swc = row.get("story_worthiness_components")
    if isinstance(swc, dict) and swc:
        out["story_worthiness_components"] = {
            str(k): round(float(v), 4)
            for k, v in swc.items()
            if isinstance(v, (int, float))
        }
    return out


def _slim_overlap_for_context(row: dict[str, Any]) -> dict[str, Any]:
    w = str(row.get("why_may_matter", "")).strip()
    if len(w) > 200:
        w = w[:197].rsplit(" ", 1)[0] + "…"
    return {
        "ticker": str(row.get("ticker", "")).strip().upper(),
        "direction": str(row.get("direction", "Mixed")).strip(),
        "congress_filing_count": row.get("congress_filing_count"),
        "unique_representatives": row.get("unique_representatives"),
        "insider_rows": row.get("insider_rows"),
        "large_dollar_congress": bool(row.get("large_dollar_congress")),
        "why_may_matter": w,
    }


def _high_profile_trades_for_context(congress_df: pd.DataFrame) -> list[dict[str, Any]]:
    from signal_logic import get_high_profile_congress_trades

    hp = get_high_profile_congress_trades(congress_df)
    if hp.empty:
        return []
    out: list[dict[str, Any]] = []
    for _, row in hp.head(12).iterrows():
        expl = str(row.get("Explanation", "")).strip()
        if len(expl) > 220:
            expl = expl[:217].rsplit(" ", 1)[0] + "…"
        out.append(
            {
                "representative": str(row.get("Representative", ""))[:120],
                "ticker": str(row.get("Ticker", ""))[:24],
                "transaction": str(row.get("Transaction", ""))[:100],
                "range": str(row.get("Range", ""))[:100],
                "why_matters": expl,
            }
        )
    return out


def _build_claude_report_context(
    signals: SignalsBundle | dict[str, Any],
    *,
    generated_utc: str,
    congress_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compact, JSON-serializable snapshot of this run for Claude POST bodies.
    Kept small: caps on list lengths, short strings, no raw DataFrames.
    """
    ranked = [r for r in (signals.get("ranked_signals") or []) if isinstance(r, dict)]
    overlap_sigs = [r for r in (signals.get("overlap_signals") or []) if isinstance(r, dict)]

    hero_rows = signals.get("hero_ranked_signals")
    if isinstance(hero_rows, list) and hero_rows:
        leaders = [r for r in hero_rows if isinstance(r, dict) and str(r.get("ticker", "")).strip()][:3]
    else:
        leaders = [r for r in ranked if str(r.get("ticker", "")).strip()][:3]
    top_anomalies: list[dict[str, Any]] = []
    for row in leaders:
        slim = _slim_ranked_for_context(row)
        slim["headline"] = _hero_card_title_line(row)
        top_anomalies.append(slim)

    top_ranked = [
        _slim_ranked_for_context(r) for r in ranked[:10] if str(r.get("ticker", "")).strip()
    ]
    overlap_ctx = [
        _slim_overlap_for_context(r) for r in overlap_sigs[:10] if str(r.get("ticker", "")).strip()
    ]

    bull, bear, mix = _directional_bucket_tickers(ranked, overlap_sigs)
    dir_md = _directional_signal_summary_section(ranked, overlap_sigs)
    dm = re.search(r"##[^\n]+\n\n(.+?)\n\n###", dir_md, re.DOTALL)
    intro = dm.group(1).strip() if dm else ""
    intro = re.sub(r"\*\*([^*]+)\*\*", r"\1", intro)
    intro = re.sub(r"\s+", " ", intro).strip()
    if len(intro) > 480:
        intro = intro[:477].rsplit(" ", 1)[0] + "…"

    wl_raw = signals.get("insider_fallback_watchlist") or []
    wl_slim: list[dict[str, Any]] = []
    for w in wl_raw:
        if not isinstance(w, dict):
            continue
        tk = str(w.get("ticker", "")).strip().upper()
        if not tk:
            continue
        wl_slim.append(
            {
                "ticker": tk,
                "total_trades": w.get("total_trades"),
                "skew_summary": str(w.get("skew_summary", ""))[:240],
                "recency_note": str(w.get("recency_note", ""))[:240],
            }
        )
        if len(wl_slim) >= 5:
            break

    return {
        "meta": {"source": "research_dashboard", "generated_utc": generated_utc},
        "insider_only_fallback_mode": bool(signals.get("insider_only_fallback_mode")),
        "insider_fallback_watchlist": wl_slim,
        "top_anomalies": top_anomalies,
        "top_ranked_signals": top_ranked,
        "strongest_overlap_signals": overlap_ctx,
        "directional_summary": {
            "intro": intro,
            "bullish": bull,
            "bearish": bear,
            "mixed": mix,
        },
        "high_profile_trades": _high_profile_trades_for_context(congress_df),
        "data_availability_notes": _data_availability_plain_lines(signals),
    }


def _embed_claude_report_context_b64(signals: SignalsBundle | dict[str, Any], *, generated_utc: str) -> str:
    cdf = signals.get("congress_df", pd.DataFrame())
    if not isinstance(cdf, pd.DataFrame):
        cdf = pd.DataFrame()
    ctx = _build_claude_report_context(signals, generated_utc=generated_utc, congress_df=cdf)
    raw = json.dumps(ctx, ensure_ascii=False, separators=(",", ":"), default=str)
    return base64.standard_b64encode(raw.encode("utf-8")).decode("ascii")


def _html_ask_claude_suggested_chips_html() -> str:
    parts: list[str] = []
    for label, prompt_text in _ASK_CLAUDE_SUGGESTED_CHIPS:
        lab_esc = html.escape(label)
        if prompt_text is None:
            parts.append(
                f'<button type="button" class="ask-claude-chip ask-claude-chip--custom" '
                f'data-own="true">{lab_esc}</button>'
            )
        else:
            p_esc = html.escape(prompt_text, quote=True)
            parts.append(
                f'<button type="button" class="ask-claude-chip" data-prompt="{p_esc}">{lab_esc}</button>'
            )
    inner = "\n    ".join(parts)
    return (
        f'<h3 class="ask-claude-subhead" id="ask-claude-suggested-h">Suggested prompts</h3>'
        f'<div class="ask-claude-chips" role="group" aria-labelledby="ask-claude-suggested-h">\n    {inner}\n  </div>'
    )


def _html_ask_claude_panel(*, api_endpoint: str = "", context_b64: str = "") -> str:
    """
    Claude prompt UI with POST endpoint from ``_resolve_claude_dashboard_endpoint()`` at build time.

    No endpoint → submit shows CLAUDE_DASHBOARD_PLACEHOLDER only (no network).
    With endpoint → initial hint text instead of the offline placeholder; errors avoid that placeholder.
    ``context_b64`` is base64(JSON) of this run (decoded client-side as ``context`` on submit).
    """
    ep = api_endpoint.strip()
    ep_attr = html.escape(ep, quote=True)
    ph_attr = html.escape(CLAUDE_DASHBOARD_PLACEHOLDER, quote=True)
    if ep:
        initial_display = (
            "Submit a question above. The assistant answers that question directly first, then may add context from this run."
        )
    else:
        initial_display = CLAUDE_DASHBOARD_PLACEHOLDER
    ph_body = html.escape(initial_display)
    fetch_suffix_attr = html.escape(CLAUDE_DASHBOARD_FETCH_FAILED_SUFFIX.strip(), quote=True)
    chips_html = _html_ask_claude_suggested_chips_html()
    ctx_node = (
        f'<script type="text/plain" id="dashboard-report-context-b64" hidden>{context_b64}</script>'
        if context_b64
        else '<script type="text/plain" id="dashboard-report-context-b64" hidden></script>'
    )
    return (
        '<section class="ask-claude-panel panel" id="ask-claude-panel" '
        f'data-endpoint="{ep_attr}" data-placeholder="{ph_attr}" '
        f'data-fetch-failed-suffix="{fetch_suffix_attr}" '
        'aria-labelledby="ask-claude-h">'
        '<h2 class="ask-claude-title" id="ask-claude-h">Ask AI Assistant</h2>'
        '<p class="ask-claude-lead">Research copilot for <strong>this run only</strong>—answers use the embedded dashboard snapshot, not the open web. '
        'Replies start with a direct answer to your question (including suggested prompts below), then may add extra context.</p>'
        '<p class="ask-claude-hint">Each submit sends your question plus the same JSON context (top anomalies, rankings, overlap, direction, high-profile trades, data notes). '
        "The assistant must answer your question first, stay within that context, decline when facts are missing, and may add interpretation or follow-ups after the direct answer. "
        "Point <code>QQ_CLAUDE_DASHBOARD_ENDPOINT</code> at <code>http://127.0.0.1:5000/ask-ai</code> (Flask <code>backend.py</code>) or use "
        "<code>QQ_AUTO_LOCAL_CLAUDE=1</code> / <code>python claude_proxy.py</code> with <code>ANTHROPIC_API_KEY</code>.</p>"
        f"{chips_html}"
        '<label class="ask-claude-label" for="ask-claude-input">Your prompt</label>'
        '<textarea id="ask-claude-input" class="ask-claude-input" rows="5" '
        'placeholder="Example: Propose three neutral headline options using only the overlap section."></textarea>'
        '<button type="button" class="ask-claude-submit">Submit</button>'
        f"{ctx_node}"
        '<div class="ask-claude-response-wrap">'
        '<span class="ask-claude-response-label">Response</span>'
        f'<div id="ask-claude-output" class="ask-claude-response ask-claude-response--placeholder" '
        f'role="status" aria-live="polite">{ph_body}</div>'
        "</div>"
        "</section>"
    )


def _html_ask_claude_script(*, backend_port: int | None = None) -> str:
    """Wired to #ask-claude-panel; safe no-op if the DOM node is missing."""
    port = _qq_backend_port() if backend_port is None else int(backend_port)
    return r"""<script>
(function () {
  var root = document.getElementById("ask-claude-panel");
  if (!root) return;
  var endpoint = (root.getAttribute("data-endpoint") || "").trim();
  var placeholder = root.getAttribute("data-placeholder") || "";
  var ta = document.getElementById("ask-claude-input");
  var btn = root.querySelector(".ask-claude-submit");
  var out = document.getElementById("ask-claude-output");
  if (!btn || !out || !ta) return;
  var reportContext = null;
  try {
    var ctxEl = document.getElementById("dashboard-report-context-b64");
    var rawB64 = ctxEl && ctxEl.textContent ? ctxEl.textContent.trim() : "";
    if (rawB64) {
      reportContext = JSON.parse(atob(rawB64));
    }
  } catch (e) {
    reportContext = null;
  }
  root.addEventListener("click", function (ev) {
    var chip = ev.target && ev.target.closest(".ask-claude-chip");
    if (!chip || !root.contains(chip)) return;
    if (chip.getAttribute("data-own") === "true") {
      ta.focus();
      return;
    }
    var p = chip.getAttribute("data-prompt");
    if (p != null) {
      ta.value = p;
      ta.focus();
    }
  });
  function setOut(text, kind) {
    out.textContent = text;
    out.className = "ask-claude-response" + (kind ? " ask-claude-response--" + kind : "");
  }
  btn.addEventListener("click", function () {
    var q = (ta && ta.value || "").trim();
    if (!q) {
      setOut("Enter a question in the text area, then submit.", "error");
      return;
    }
    if (!endpoint) {
      setOut(placeholder, "placeholder");
      return;
    }
    btn.disabled = true;
    setOut("Thinking...", "loading");
    fetch(endpoint, {
      method: "POST",
      mode: "cors",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: q,
        context: reportContext,
      }),
    })
      .then(function (r) {
        return r.text().then(function (t) {
          var j = null;
          try {
            j = JSON.parse(t);
          } catch (e) {}
          return { ok: r.ok, j: j, raw: t };
        });
      })
      .then(function (x) {
        var reply = x.j && x.j.response;
        if (reply != null && reply !== "") {
          setOut(String(reply), x.ok ? "" : "error");
          return;
        }
        if (x.j && x.j.error) {
          setOut(String(x.j.error), "error");
          return;
        }
        if (!x.ok) {
          var detail =
            x.raw && x.raw.length < 500 ? x.raw : "Request failed.";
          setOut(detail, "error");
          return;
        }
        setOut("No response field in the server reply.", "error");
      })
      .catch(function (err) {
        console.error("[Ask AI Assistant] fetch failed — endpoint:", endpoint, err);
        var br = "\n\n";
        var why = err && err.message ? "(" + err.message + ")" : "";
        var tips =
          "Troubleshooting: (1) Start the API in a terminal: python backend.py — leave that window open; nothing appears in the browser until Flask is running. " +
          "(2) Open http://127.0.0.1:__QQ_BACKEND_PORT__/health — if you see connection refused, the server is not running or the port is wrong (try set QQ_BACKEND_PORT=5001 and use the same when you run main.py). " +
          "(3) Prefer http://127.0.0.1:__QQ_BACKEND_PORT__/dashboard instead of opening the HTML file directly. " +
          "(4) pip install flask if backend.py exits immediately.";
        setOut(
          "Could not reach the local AI backend. Make sure backend.py is running." +
            (why ? br + why : "") +
            br +
            tips,
          "error"
        );
      })
      .finally(function () {
        btn.disabled = false;
      });
  });
})();
</script>""".replace("__QQ_BACKEND_PORT__", str(port))


def _html_top_card_scroll_script() -> str:
    """Smooth-scroll from hero top anomaly cards to matching ranked detail sections."""
    return """
<script>
document.querySelectorAll(".top-card").forEach(card => {
  card.addEventListener("click", () => {
    const ticker = card.getAttribute("data-target");
    const target = document.getElementById("ticker-" + ticker);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
});
</script>"""


def _html_top_anomalies_section(
    large: pd.DataFrame,
    top: pd.DataFrame,
    overlap: list[str],
    ranked_signals: list[dict[str, Any]],
) -> str:
    """Summary cards: large trades, conviction leaders, overlap breadth, top-ranked tickers."""
    cards: list[str] = []

    n_large = len(large) if isinstance(large, pd.DataFrame) else 0
    if n_large and isinstance(large, pd.DataFrame) and "Ticker" in large.columns:
        prev = large["Ticker"].astype(str).head(5).tolist()
        prev_html = "<ul>" + "".join(f"<li>{html.escape(t)}</li>" for t in prev) + "</ul>"
    else:
        prev_html = "<p class=\"muted\">No large-range rows matched this run.</p>"
    cards.append(
        _html_panel(
            "Large-range trades (Signal 1)",
            f"<p><strong>{n_large}</strong> row(s) after filters (max 10 shown in detail).</p>{prev_html}",
        )
    )

    if isinstance(top, pd.DataFrame) and not top.empty and "Ticker" in top.columns:
        rows_html = "<ul>"
        for _, row in top.head(5).iterrows():
            t = html.escape(str(row["Ticker"]))
            c = int(row["Count"]) if "Count" in row else 0
            cv = int(row["CongressConviction"]) if "CongressConviction" in top.columns else 0
            rows_html += f"<li><strong>{t}</strong> — count {c}, conviction {cv}</li>"
        rows_html += "</ul>"
    else:
        rows_html = "<p class=\"muted\">No conviction-ranked ticker table this run.</p>"
    cards.append(
        _html_panel(
            "Most active tickers (conviction-ranked)",
            rows_html,
        )
    )

    no = len(overlap) if overlap else 0
    o_prev = overlap[:5] if overlap else []
    o_list = "<ul>" + "".join(f"<li>{html.escape(t)}</li>" for t in o_prev) + "</ul>" if o_prev else ""
    cards.append(
        _html_panel(
            "Cross-dataset overlap",
            f"<p><strong>{no}</strong> symbol(s) in congress ∩ insider (alphabetical sample below).</p>"
            + (o_list or "<p class=\"muted\">No overlap this run.</p>"),
        )
    )

    if ranked_signals:
        rl = "<ul>"
        for item in ranked_signals[:6]:
            if not isinstance(item, dict):
                continue
            tk = html.escape(str(item.get("ticker", "")))
            sc = _desk_score_str(item.get("score", 0))
            rl += f"<li><strong>{tk}</strong> — score {sc}/10</li>"
        rl += "</ul>"
    else:
        rl = "<p class=\"muted\">No ranked signals this run.</p>"
    cards.append(_html_panel("Top ranked (preview)", rl))

    return (
        '<p class="lead">Snapshot of the strongest anomaly-style outputs before the detailed sections below.</p>'
        f'<div class="anomaly-grid">{"".join(cards)}</div>'
    )


def _html_high_profile_blocks(congress_df: pd.DataFrame) -> str:
    from signal_logic import get_high_profile_congress_trades

    hp = get_high_profile_congress_trades(congress_df)
    if hp.empty:
        return "<p class=\"muted\">No trades matched the high-profile name list in this pull.</p>"
    parts: list[str] = []
    for i, (_, row) in enumerate(hp.iterrows(), start=1):
        rep = html.escape(str(row.get("Representative", "")))
        tkr = html.escape(str(row.get("Ticker", "")))
        txn = html.escape(str(row.get("Transaction", "")))
        rng = html.escape(str(row.get("Range", "")))
        expl = _format_inline_md(row.get("Explanation", ""))
        extras: list[str] = []
        if "ReportDate" in hp.columns:
            extras.append(
                f"<p><strong>Report date:</strong> {html.escape(str(row.get('ReportDate', '')))}</p>"
            )
        if "CongressConviction" in hp.columns:
            extras.append(
                f"<p><strong>Conviction (trade-level):</strong> "
                f"{html.escape(str(row.get('CongressConviction', '')))}/100</p>"
            )
        if "ConvictionNote" in hp.columns:
            extras.append(
                f"<p><strong>Conviction note:</strong> {_format_inline_md(row.get('ConvictionNote', ''))}</p>"
            )
        extras_html = ""
        if extras:
            extras_html = (
                f'<div class="hp-card-extras text-secondary">{"".join(extras)}</div>'
            )
        parts.append(
            f'<article class="panel hp-card hp-card--compact">'
            f'<div class="hp-card-index">Trade {i}</div>'
            f'<div class="hp-card-main">'
            f'<div class="hp-kv"><span class="hp-k">Representative</span>'
            f'<span class="hp-v">{rep}</span></div>'
            f'<div class="hp-kv"><span class="hp-k">Ticker</span>'
            f'<span class="hp-v hp-v-ticker">{tkr}</span></div>'
            f'<div class="hp-kv"><span class="hp-k">Transaction</span>'
            f'<span class="hp-v">{txn}</span></div>'
            f'<div class="hp-kv"><span class="hp-k">Range</span>'
            f'<span class="hp-v">{rng}</span></div>'
            f'<div class="hp-kv hp-kv--block"><span class="hp-k">Why it matters</span>'
            f'<div class="hp-v hp-v-prose">{expl}</div></div>'
            f"</div>{extras_html}</article>"
        )
    return f'<div class="section-stack hp-stack">{"".join(parts)}</div>'


def _html_top_ranked_blocks(ranked: list[dict[str, Any]]) -> str:
    if not ranked:
        return "<p class=\"muted\">No ranked signals for this run.</p>"
    parts: list[str] = []
    for i, row in enumerate(ranked, start=1):
        if not isinstance(row, dict):
            continue
        t = html.escape(str(row.get("ticker", "")))
        sc = _desk_score_str(row.get("score", 0))
        bd_html = _html_scoring_breakdown_block(row)
        ev = row.get("evidence") or {}
        d_raw = str(row.get("direction") or ev.get("direction", "Mixed")).strip()
        dir_tag = _dir_tag_html(d_raw)
        dn = row.get("direction_note") or ev.get("direction_note", "")
        dn_html = f"<p class=\"text-secondary\"><strong>Why this label:</strong> {_format_inline_md(dn)}</p>" if dn else ""
        wt_list = row.get("why_triggered") or []
        if wt_list:
            triggers_block = "<ul class=\"dash-list\">" + "".join(
                f"<li>{_format_inline_md(x)}</li>" for x in wt_list
            ) + "</ul>"
        else:
            triggers_block = '<p class="muted">No trigger lines for this signal.</p>'
        ev_lines: list[str] = []
        for k, v in [
            ("Congress filings", ev.get("congress_filings", 0)),
            ("Unique representatives", ev.get("unique_representatives", 0)),
            ("Insider rows", ev.get("insider_rows", 0)),
            ("Buy-like / sell-like", f"{ev.get('insider_buys', 0)} / {ev.get('insider_sells', 0)}"),
            ("Large-dollar congress trade", "yes" if ev.get("large_congress_trade") else "no"),
            ("Congress recency (0–1)", ev.get("congress_recency_factor", "")),
            ("Insider recency (0–1)", ev.get("insider_recency_factor", "")),
        ]:
            ev_lines.append(f"<li><strong>{html.escape(k)}:</strong> {html.escape(str(v))}</li>")
        if ev.get("contract_activity_count") is not None:
            ev_lines.append(
                f"<li><strong>Contract activity:</strong> {html.escape(str(ev.get('contract_activity_count')))}</li>"
            )
        if ev.get("lobbying_activity_count") is not None:
            ev_lines.append(
                f"<li><strong>Lobbying activity:</strong> {html.escape(str(ev.get('lobbying_activity_count')))}</li>"
            )
        if ev.get("congress_conviction_0_100") is not None:
            ev_lines.append(
                f"<li><strong>Congressional conviction:</strong> {html.escape(str(ev.get('congress_conviction_0_100')))}/100</li>"
            )
        if ev.get("congress_conviction_note"):
            ev_lines.append(
                f"<li><strong>Conviction note:</strong> {_format_inline_md(ev.get('congress_conviction_note'))}</li>"
            )
        wmt = _format_inline_md(row.get("why_may_matter", ""))
        ang = _format_inline_md(row.get("newsletter_angle", ""))
        tk_raw = str(row.get("ticker", "")).strip()
        tk_tok = _ticker_scroll_token(tk_raw)
        parts.append(
            f'<section class="ranked-detail-anchor" id="ticker-{html.escape(tk_tok)}">'
            f'<article class="panel ranked-card">'
            f'<div class="card-top"><h3>{i}. {t}</h3>'
            f'<div class="card-meta"><span class="score-badge score-badge--sm">{sc}/10</span>{dir_tag}</div></div>'
            f"{dn_html}"
            f"{bd_html}"
            f'<div class="evidence-panel"><h4>Why it triggered</h4>{triggers_block}</div>'
            f'<div class="evidence-panel"><h4>Evidence</h4><ul class="dash-list">{"".join(ev_lines)}</ul></div>'
            f'<div class="evidence-panel"><h4>Why it may matter</h4>'
            f'<p class="card-body-text">{wmt}</p></div>'
            f'<div class="evidence-panel"><h4>Suggested newsletter angle</h4>'
            f'<p class="card-body-text text-secondary">{ang}</p></div></article>'
            f"</section>"
        )
    return f'<div class="section-stack ranked-stack">{"".join(parts)}</div>'


def _html_overlap_blocks(overlap: list[str], overlap_signals: list[Any]) -> str:
    if not overlap:
        return (
            '<div class="info-box info-box--compact">'
            '<p class="muted">No cross-dataset overlap this run (or one feed was empty).</p></div>'
        )
    if not overlap_signals:
        return (
            '<div class="info-box info-box--compact">'
            '<p class="muted">Overlap exists but detailed ranking could not be built.</p></div>'
        )
    parts: list[str] = []
    for i, row in enumerate(overlap_signals, start=1):
        if not isinstance(row, dict):
            continue
        t = html.escape(str(row.get("ticker", "")))
        d_raw = str(row.get("direction", "Mixed")).strip()
        dir_tag = _dir_tag_html(d_raw)
        lines: list[str] = []
        if row.get("direction_note"):
            lines.append(
                f"<li><strong>Why this label:</strong> {_format_inline_md(row.get('direction_note'))}</li>"
            )
        for label, key in [
            ("Congress filings", "congress_filing_count"),
            ("Unique representatives", "unique_representatives"),
            ("Insider rows", "insider_rows"),
            ("Buy-like / sell-like", None),
            ("Large-dollar congress", "large_dollar_congress"),
        ]:
            if key is None:
                lines.append(
                    f"<li><strong>{label}:</strong> {row.get('buy_like_count', 0)} / {row.get('sell_like_count', 0)}</li>"
                )
            elif key == "large_dollar_congress":
                lg = "yes" if row.get(key) else "no"
                lines.append(f"<li><strong>{label}:</strong> {html.escape(lg)}</li>")
            else:
                lines.append(
                    f"<li><strong>{label}:</strong> {html.escape(str(row.get(key, 0)))}</li>"
                )
        if row.get("contract_activity_count") is not None:
            lines.append(
                f"<li><strong>Contract rows (supporting):</strong> {html.escape(str(row.get('contract_activity_count')))}</li>"
            )
        if row.get("lobbying_activity_count") is not None:
            lines.append(
                f"<li><strong>Lobbying rows (supporting):</strong> {html.escape(str(row.get('lobbying_activity_count')))}</li>"
            )
        if row.get("congress_conviction_0_100") is not None:
            lines.append(
                f"<li><strong>Congressional conviction:</strong> {html.escape(str(row.get('congress_conviction_0_100')))}/100</li>"
            )
        if row.get("congress_recency_factor") is not None:
            lines.append(
                f"<li><strong>Congress recency (0–1):</strong> {html.escape(str(row.get('congress_recency_factor')))}</li>"
            )
        if row.get("insider_recency_factor") is not None:
            lines.append(
                f"<li><strong>Insider recency (0–1):</strong> {html.escape(str(row.get('insider_recency_factor')))}</li>"
            )
        wmt = _format_inline_md(row.get("why_may_matter", ""))
        parts.append(
            f'<article class="panel overlap-card">'
            f'<div class="card-top"><h3>{i}. {t}</h3><div class="card-meta">{dir_tag}</div></div>'
            f'<div class="evidence-panel"><h4>Counts &amp; context</h4>'
            f'<ul class="dash-list">{"".join(lines)}</ul></div>'
            f'<div class="evidence-panel"><h4>Why it may matter</h4>'
            f'<p class="card-body-text">{wmt}</p></div></article>'
        )
    foot = (
        "<p class=\"footnote\">Rank order uses congress and insider counts, buy-like weighting, "
        "large-range boost, conviction, and recency—see Methodology.</p>"
    )
    lead = (
        f"<p class=\"overlap-lead\"><strong>{len(overlap)}</strong> symbols in the raw intersection; "
        "below are the ranked highlights.</p>"
    )
    return (
        f'<div class="section-stack overlap-stack">'
        f'<div class="info-box info-box--compact">{lead}</div>'
        f'{"".join(parts)}'
        f'<div class="info-box info-box--compact info-box--subtle">{foot}</div>'
        f"</div>"
    )


def _html_dashboard_styles() -> str:
    return """
:root {
  --bg: #f2f4f7;
  --surface: #ffffff;
  --text: #1c1f24;
  --muted: #5e6670;
  --accent: #1a4f7c;
  --accent-light: #e8eef5;
  --border: #dce1e8;
  --shadow: 0 1px 2px rgba(0,0,0,.04), 0 4px 12px rgba(0,0,0,.05);
  --radius: 12px;
}
* { box-sizing: border-box; }
html {
  scroll-behavior: smooth;
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
}
body {
  margin: 0;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 16px;
  line-height: 1.55;
  color: var(--text);
  background: var(--bg);
  -webkit-font-smoothing: antialiased;
}
header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  box-shadow: 0 1px 0 rgba(0,0,0,.03);
  padding: 1.75rem clamp(1.25rem, 4vw, 2.5rem);
}
header h1 {
  margin: 0 0 0.4rem 0;
  font-size: clamp(1.65rem, 2.5vw, 1.95rem);
  font-weight: 700;
  color: var(--accent);
  letter-spacing: -0.02em;
}
header .sub {
  margin: 0;
  color: var(--muted);
  font-size: 0.98rem;
}
main {
  max-width: 1040px;
  margin: 0 auto;
  padding: clamp(1.5rem, 3vw, 2.5rem) clamp(1.1rem, 3vw, 1.75rem) 3.5rem;
}
.congress-unavailable-banner {
  margin: 0 0 1.25rem 0;
  padding: 0.9rem 1.1rem;
  background: #fff8e6;
  border: 1px solid #e8c96a;
  border-radius: var(--radius);
}
.congress-unavailable-banner__text {
  margin: 0;
  font-size: 0.98rem;
  font-weight: 600;
  color: #5c4a1a;
  line-height: 1.45;
}
.insider-only-watchlist-dek {
  font-weight: 500;
  color: var(--muted);
}
.insider-fallback-badge {
  display: inline-block;
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 0.2rem 0.5rem;
  border-radius: 6px;
  background: #eef2f7;
  color: var(--accent);
  border: 1px solid var(--border);
}
.anomaly-hero-card--insider-fallback {
  border: 1px dashed var(--border);
}
.key-insight-callout {
  margin: 0 0 1.5rem 0;
  padding: 1.15rem 1.35rem 1.2rem;
  background: linear-gradient(145deg, #eef4fa 0%, #e6eef8 55%, #dfe8f3 100%);
  border: 1px solid #c5d4e3;
  border-radius: var(--radius);
  box-shadow: 0 2px 10px rgba(26, 79, 124, 0.09);
}
.key-insight-callout__label {
  display: block;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--accent);
  margin-bottom: 0.55rem;
}
.key-insight-callout__text {
  margin: 0;
  font-size: 1.08rem;
  line-height: 1.52;
  font-weight: 500;
  color: var(--text);
  letter-spacing: -0.015em;
}
@media (max-width: 640px) {
  .key-insight-callout { padding: 1rem 1.1rem 1.05rem; }
  .key-insight-callout__text { font-size: 1.02rem; }
}
.quick-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.55rem 0.85rem;
  margin: 0 0 1.35rem 0;
  padding: 0.65rem 0.85rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}
.quick-actions-label {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--muted);
  flex: 0 0 auto;
  margin-right: 0.15rem;
}
.quick-actions-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 0.45rem;
  align-items: center;
  flex: 1 1 auto;
  min-width: 0;
}
a.quick-action {
  display: inline-block;
  margin: 0;
  padding: 0.32rem 0.72rem;
  font: inherit;
  font-size: 0.78rem;
  font-weight: 600;
  line-height: 1.35;
  color: var(--text);
  text-decoration: none;
  background: #f0f3f7;
  border: 1px solid #d0d8e4;
  border-radius: 999px;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
  white-space: nowrap;
}
a.quick-action:hover {
  background: var(--accent-light);
  border-color: #b8c8d8;
  color: var(--accent);
}
a.quick-action:focus {
  outline: none;
}
a.quick-action:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}
#key-insight,
#top-anomalies,
#what-changed-this-run,
#newsletter-ready-insights,
.executive-copy-block,
#ask-claude-panel,
.report-section {
  scroll-margin-top: 0.85rem;
}
.report-section {
  margin: 0;
  padding-top: 2.5rem;
  margin-top: 2.5rem;
  border-top: 1px solid var(--border);
}
main > .report-section:first-of-type {
  border-top: none;
  padding-top: 0;
  margin-top: 2rem;
}
.section-title {
  font-size: clamp(1.2rem, 2vw, 1.4rem);
  font-weight: 700;
  color: var(--accent);
  margin: 0 0 1.15rem 0;
  padding-bottom: 0.55rem;
  border-bottom: 2px solid var(--accent-light);
  letter-spacing: -0.02em;
}
.section-body { padding-bottom: 0.25rem; }
.section-body > p:first-child { margin-top: 0; }
.section-stack {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.section-stack > .panel,
.section-stack > article.panel {
  margin-bottom: 0;
}
.info-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.15rem;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.info-box--compact {
  padding: 0.85rem 1.05rem;
  font-size: 0.94rem;
  line-height: 1.52;
}
.info-box--compact p {
  margin: 0.45rem 0;
}
.info-box--compact p:first-child { margin-top: 0; }
.info-box--compact p:last-child { margin-bottom: 0; }
.info-box--compact .overlap-lead { margin: 0; }
.info-box--lead { font-size: 0.96rem; }
.info-box--subtle {
  background: #f7f9fb;
  border-color: #e4e9f0;
  box-shadow: none;
}
.info-box--availability .dash-list { margin: 0; }
.methodology-stack .info-box p { margin: 0.4rem 0; }
.methodology-stack .info-box p:first-of-type { margin-top: 0; }
.verify-card p, .verify-card ul { font-size: 0.94rem; }
.hp-card--compact { padding: 1rem 1.15rem; }
.hp-card-index {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  margin-bottom: 0.65rem;
}
.hp-card-main {
  display: grid;
  gap: 0.6rem 1.5rem;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}
.hp-kv--block { grid-column: 1 / -1; }
.hp-k {
  display: block;
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  margin-bottom: 0.22rem;
}
.hp-v { font-size: 0.93rem; color: var(--text); line-height: 1.45; }
.hp-v-ticker { font-weight: 700; color: var(--accent); font-size: 1.02rem; }
.hp-v-prose { font-size: 0.95rem; line-height: 1.5; }
.hp-card-extras {
  margin-top: 0.85rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border);
  font-size: 0.86rem;
  line-height: 1.45;
}
.hp-card-extras p { margin: 0.35rem 0; }
.hp-card-extras p:first-child { margin-top: 0; }
.hp-card-extras p:last-child { margin-bottom: 0; }
.dash-list--tight li { margin-bottom: 0.28rem; }
.evidence-panel .card-body-text { margin: 0; font-size: 0.95rem; line-height: 1.5; }
.ranked-card .evidence-panel {
  margin-top: 0.45rem;
  margin-bottom: 0.55rem;
}
.ranked-card .evidence-panel:last-child { margin-bottom: 0; }
.overlap-card .evidence-panel {
  margin-top: 0.45rem;
  margin-bottom: 0.55rem;
}
.overlap-card .evidence-panel:last-child { margin-bottom: 0; }
.section-body h3 {
  font-size: 1.12rem;
  font-weight: 600;
  color: var(--accent);
  margin: 1.1rem 0 0.5rem 0;
}
.section-body h3:first-child { margin-top: 0; }
.section-body .info-box > h3 {
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--accent);
  margin: 0 0 0.5rem 0;
}
.section-body .verify-card > h3 {
  font-size: 1.03rem;
  font-weight: 700;
  color: var(--accent);
  margin: 0 0 0.6rem 0;
  letter-spacing: -0.015em;
  text-transform: none;
}
.section-body h4 {
  font-size: 0.98rem;
  font-weight: 600;
  color: var(--text);
  margin: 1rem 0 0.45rem;
}
.h4-flush { margin-top: 1.35rem !important; }
.text-secondary { color: var(--muted); }
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.35rem;
  margin-bottom: 1.1rem;
  box-shadow: var(--shadow);
}
.panel h3 { margin-top: 0; }
.card-top {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem 1rem;
  margin-bottom: 0.85rem;
}
.card-top h3 {
  margin: 0;
  font-size: 1.14rem;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: -0.02em;
}
.card-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem 0.65rem;
}
.score-badge {
  flex-shrink: 0;
  display: inline-block;
  background: var(--accent);
  color: #fff;
  font-weight: 700;
  font-size: 1.15rem;
  padding: 0.38rem 0.85rem;
  border-radius: 8px;
  line-height: 1.2;
  box-shadow: 0 1px 3px rgba(26, 79, 124, 0.2);
}
.score-badge--sm {
  font-size: 0.9rem;
  padding: 0.22rem 0.62rem;
  border-radius: 7px;
}
.dir-tag {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 0.28rem 0.55rem;
  border-radius: 6px;
  line-height: 1.2;
}
.dir-bullish {
  background: #e9f4ee;
  color: #1a6b42;
  border: 1px solid #c5e3d4;
}
.dir-bearish {
  background: #fcefed;
  color: #9c2f2f;
  border: 1px solid #eec9c9;
}
.dir-mixed {
  background: #eef0f4;
  color: var(--muted);
  border: 1px solid var(--border);
}
.evidence-panel {
  background: #f7f9fb;
  border: 1px solid #e4e9f0;
  border-radius: 10px;
  padding: 0.9rem 1.05rem;
  margin: 0.65rem 0 1rem;
}
.evidence-panel h4 {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--muted);
  margin: 0 0 0.55rem 0;
}
.evidence-panel .dash-list { margin: 0; }
.evidence-panel--hero {
  background: rgba(255,255,255,.65);
  border-color: #dde5ee;
  margin-bottom: 0;
}
.evidence-panel--hero h4 { display: none; }
.anomaly-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 240px), 1fr));
  gap: 1.1rem;
}
.anomaly-grid .panel h3 { font-size: 1.02rem; }
.lead { color: var(--muted); margin-bottom: 1.1rem; font-size: 1.02rem; }
.hero-anomalies-wrap {
  margin: 0 0 2rem 0;
  padding-bottom: 2.25rem;
  border-bottom: 1px solid var(--border);
}
.dashboard-ai-strip {
  margin: 0 0 1.15rem 0;
  padding: 1rem 1.15rem 1.05rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  border-left: 3px solid var(--accent);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}
.dashboard-ai-strip__title {
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--muted);
  margin: 0 0 0.55rem 0;
}
.dashboard-ai-strip__body {
  font-size: 0.94rem;
  line-height: 1.55;
  color: var(--text);
  white-space: pre-wrap;
}
.anomaly-hero-card--minimal {
  min-height: 0;
}
.anomaly-hero-card--minimal .anomaly-hero-why {
  margin-top: 0.65rem;
  font-size: 0.88rem;
}
.what-changed-run {
  margin: 0 0 1.75rem 0;
  padding: 1.05rem 1.2rem 1.1rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent-light);
  border-radius: var(--radius);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}
.what-changed-run__title {
  font-size: 1.08rem;
  font-weight: 700;
  color: var(--accent);
  margin: 0 0 0.4rem 0;
  letter-spacing: -0.02em;
}
.what-changed-run__dek {
  font-size: 0.86rem;
  color: var(--muted);
  margin: 0 0 0.75rem 0;
  line-height: 1.45;
}
.what-changed-run__list {
  margin: 0;
  padding-left: 1.2rem;
  font-size: 0.94rem;
  line-height: 1.52;
  color: var(--text);
}
.what-changed-run__list li {
  margin-bottom: 0.5rem;
}
.what-changed-run__list li:last-child {
  margin-bottom: 0;
}
.executive-copy-block {
  margin: 0 0 2rem 0;
  padding: 1.35rem 1.45rem 1.25rem;
  background: #e9eef5;
  border: 1px solid #cdd6e3;
  border-radius: var(--radius);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.6);
}
.executive-copy-title {
  font-size: 1.12rem;
  font-weight: 700;
  color: var(--accent);
  margin: 0 0 0.45rem 0;
  letter-spacing: -0.02em;
}
.executive-copy-dek {
  font-size: 0.88rem;
  color: var(--muted);
  margin: 0 0 1rem 0;
  line-height: 1.45;
}
.executive-copy-list {
  margin: 0;
  padding-left: 1.25rem;
  line-height: 1.55;
  color: var(--text);
  font-size: 0.96rem;
}
.executive-copy-list li {
  margin-bottom: 0.55rem;
  padding-left: 0.15rem;
}
.executive-copy-list li:last-child {
  margin-bottom: 0;
}
.post-hero-split {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1.25rem;
  margin: 0 0 2rem 0;
  align-items: start;
}
.post-hero-split__main .executive-copy-block {
  margin-bottom: 0;
}
.post-hero-split .ask-claude-panel {
  margin-bottom: 0;
}
@media (min-width: 920px) {
  .post-hero-split {
    grid-template-columns: minmax(0, 1.15fr) minmax(280px, 0.95fr);
  }
}
.ask-claude-title {
  font-size: 1.12rem;
  font-weight: 700;
  color: var(--accent);
  margin: 0 0 0.45rem 0;
  letter-spacing: -0.02em;
}
.ask-claude-lead {
  font-size: 0.93rem;
  color: var(--text);
  margin: 0 0 0.5rem 0;
  line-height: 1.45;
}
.ask-claude-hint {
  font-size: 0.86rem;
  color: var(--muted);
  margin: 0 0 0.85rem 0;
  line-height: 1.45;
}
.ask-claude-hint code {
  font-size: 0.82em;
  background: #eef1f5;
  padding: 0.12rem 0.35rem;
  border-radius: 4px;
  border: 1px solid var(--border);
}
.ask-claude-subhead {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--muted);
  margin: 0 0 0.55rem 0;
}
.ask-claude-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem 0.5rem;
  margin: 0 0 1.05rem 0;
}
.ask-claude-chip {
  display: inline-block;
  margin: 0;
  padding: 0.38rem 0.75rem;
  font: inherit;
  font-size: 0.8rem;
  font-weight: 500;
  line-height: 1.35;
  color: var(--text);
  background: #f0f3f7;
  border: 1px solid #d0d8e4;
  border-radius: 999px;
  cursor: pointer;
  text-align: left;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
.ask-claude-chip:hover {
  background: var(--accent-light);
  border-color: #b8c8d8;
  color: var(--accent);
}
.ask-claude-chip:focus {
  outline: none;
}
.ask-claude-chip:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}
.ask-claude-chip--custom {
  background: #fff;
  border-style: dashed;
  color: var(--muted);
  font-weight: 600;
}
.ask-claude-chip--custom:hover {
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-light);
}
.ask-claude-label {
  display: block;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  margin-bottom: 0.4rem;
}
.ask-claude-input {
  width: 100%;
  margin: 0 0 0.75rem 0;
  padding: 0.65rem 0.85rem;
  font: inherit;
  font-size: 0.94rem;
  line-height: 1.45;
  color: var(--text);
  background: #fafbfc;
  border: 1px solid var(--border);
  border-radius: 8px;
  resize: vertical;
  min-height: 6rem;
  box-sizing: border-box;
}
.ask-claude-input:focus {
  outline: 2px solid var(--accent-light);
  outline-offset: 1px;
  border-color: var(--accent);
}
.ask-claude-submit {
  display: inline-block;
  margin-bottom: 1rem;
  padding: 0.45rem 1.1rem;
  font: inherit;
  font-size: 0.92rem;
  font-weight: 600;
  color: #fff;
  background: var(--accent);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  box-shadow: 0 1px 2px rgba(26, 79, 124, 0.25);
}
.ask-claude-submit:hover { filter: brightness(1.06); }
.ask-claude-submit:disabled { opacity: 0.65; cursor: not-allowed; }
.ask-claude-response-wrap { margin-top: 0.15rem; }
.ask-claude-response-label {
  display: block;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  margin-bottom: 0.4rem;
}
.ask-claude-response {
  min-height: 4rem;
  padding: 0.85rem 1rem;
  font-size: 0.92rem;
  line-height: 1.5;
  color: var(--text);
  background: #f7f9fb;
  border: 1px solid #e4e9f0;
  border-radius: 8px;
  white-space: pre-wrap;
  word-break: break-word;
}
.ask-claude-response--placeholder {
  color: var(--muted);
  font-style: italic;
}
.ask-claude-response--error {
  color: #7a3030;
  background: #faf4f4;
  border-color: #e5cfcf;
  font-style: normal;
}
.ask-claude-response--loading {
  color: var(--muted);
  font-style: italic;
}
.hero-anomalies-title {
  font-size: clamp(1.65rem, 3vw, 2rem);
  font-weight: 800;
  color: var(--accent);
  margin: 0 0 0.5rem 0;
  letter-spacing: -0.03em;
}
.hero-anomalies-dek {
  font-size: 1.08rem;
  color: var(--muted);
  margin: 0 0 1.5rem 0;
  max-width: 42rem;
  line-height: 1.5;
}
.hero-anomalies-empty { margin: 0; font-size: 1.05rem; }
.anomaly-hero-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
  gap: 1.35rem;
}
.anomaly-hero-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius);
  padding: 1.65rem 1.75rem 1.5rem;
  box-shadow: var(--shadow);
  font-size: 1.08rem;
  line-height: 1.52;
}
.top-card {
  cursor: pointer;
  transition: transform 0.15s ease;
}
.top-card:hover {
  transform: translateY(-3px);
}
.ranked-detail-anchor {
  scroll-margin-top: 0.85rem;
}
.anomaly-hero-card-head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.65rem 1rem;
  margin-bottom: 0.75rem;
}
.anomaly-hero-ticker {
  font-size: clamp(1.85rem, 3.5vw, 2.15rem);
  font-weight: 800;
  letter-spacing: -0.04em;
  color: var(--accent);
  line-height: 1.05;
}
.anomaly-hero-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem 0.65rem;
}
.anomaly-hero-card-title {
  font-size: 1.18rem;
  font-weight: 600;
  color: var(--text);
  margin: 0 0 0.95rem 0;
  line-height: 1.4;
}
.anomaly-hero-evidence {
  margin: 0 0 0 1.1rem;
  padding: 0;
  font-size: 1.02rem;
  list-style-type: disc;
}
.anomaly-hero-evidence li { margin-bottom: 0.48rem; }
.anomaly-hero-why {
  margin: 1.15rem 0 0 0;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  font-size: 0.98rem;
  color: var(--text);
  line-height: 1.52;
}
.anomaly-hero-why strong { color: var(--accent); font-weight: 700; }
.directional-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr));
  gap: 1.1rem;
}
.dir-col {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.15rem 1.2rem;
  box-shadow: var(--shadow);
}
.dir-col-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem 0.75rem;
  margin-bottom: 0.65rem;
}
.dir-col-head h3 {
  font-size: 1.02rem;
  font-weight: 700;
  color: var(--accent);
  margin: 0;
  letter-spacing: -0.015em;
}
.dir-col-tickers { margin: 0; line-height: 1.6; }
ul.dash-list { margin: 0.35rem 0; padding-left: 1.25rem; }
ul.dash-list li { margin-bottom: 0.4rem; }
.muted { color: var(--muted); font-style: italic; }
.footnote { font-size: 0.85rem; margin: 0; color: var(--muted); line-height: 1.5; }
@media (max-width: 640px) {
  .anomaly-hero-card-head { flex-direction: column; align-items: stretch; }
  .anomaly-hero-meta { justify-content: flex-start; }
}
@media (prefers-reduced-motion: reduce) {
  .top-card {
    transition: none;
  }
  .top-card:hover {
    transform: none;
  }
}
"""

def _parse_ranked_claude_json(raw: str) -> dict[str, Any] | None:
    """Best-effort extract a JSON object from model output (used for dashboard-wide summary JSON)."""
    t = (raw or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


_DASHBOARD_WIDE_CLAUDE_SYSTEM = """You write top-of-dashboard copy for a financial newsletter research workflow.

The user message is one combined prompt: task instructions plus a '=== EVIDENCE FOR THIS RUN ===' section—your only facts.

Strict rules:
- Use ONLY facts from the evidence block below. Do not invent tickers, numbers, dates, or events.
- No unsupported claims. No prediction language (avoid "will", "going to", "likely to rally", price targets).
- No exaggerated or promotional tone.
- Do not give trading recommendations or buy/sell/hold advice.

Produce JSON only (no markdown fences, no commentary) with exactly these keys:
{
  "key_insight": "<1-2 sentences, main takeaway for this run>",
  "what_changed_bullets": ["<bullet 1>", "<bullet 2>", "<bullet 3>"],
  "newsletter_insights_bullets": ["<bullet 1>", "<bullet 2>", "<bullet 3>", "<optional 4>", "<optional 5>"]
}

Requirements:
- key_insight: one or two tight sentences.
- what_changed_bullets: exactly 3 concise bullets (each one line).
- newsletter_insights_bullets: 3, 4, or 5 concise copy-ready bullets for a drafting desk (each one line).
All content must be grounded in the supplied structured data only."""


def _dashboard_wide_claude_context(signals: dict[str, Any]) -> str:
    """Compact factual bundle for dashboard-wide Claude (Key Insight + What Changed + Newsletter strip)."""
    from signal_logic import get_high_profile_congress_trades

    parts: list[str] = []

    qsig = signals.get("qualified_ranked_signals")
    if isinstance(qsig, list):
        ranked_top = [r for r in qsig if isinstance(r, dict)][:3]
    else:
        ranked_top = [r for r in (signals.get("ranked_signals") or []) if isinstance(r, dict)][:3]
    parts.append(
        "## Top qualified desk anomalies (tightened bar, up to 3; ordered by story_worthiness 0–5, "
        "then distinctiveness bonus, then desk score — strong desk ≠ strong newsletter story and vice versa)"
    )
    if not ranked_top:
        parts.append("(none)")
    for i, row in enumerate(ranked_top, start=1):
        ev = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        t = str(row.get("ticker", "")).strip().upper()
        sc = _desk_score_str(row.get("score", 0))
        d = str(row.get("direction") or ev.get("direction", "Mixed")).strip()
        try:
            db = float(row.get("distinctiveness_bonus") or 0.0)
        except (TypeError, ValueError):
            db = 0.0
        try:
            sw = float(row.get("story_worthiness_score") or 0.0)
        except (TypeError, ValueError):
            sw = 0.0
        parts.append(
            f"{i}. {t} | desk_score={sc}/10 | story_worthiness={sw:.1f}/5 | distinctiveness_bonus={db:.4f} | direction={d} "
            f"| congress_filings={ev.get('congress_filings', 0)} "
            f"| unique_reps={ev.get('unique_representatives', 0)} | insider_rows={ev.get('insider_rows', 0)} "
            f"| insider_buys={ev.get('insider_buys', 0)} insider_sells={ev.get('insider_sells', 0)} "
            f"| large_dollar_congress={'yes' if ev.get('large_congress_trade') else 'no'}"
        )
        if ev.get("contract_activity_count") is not None:
            parts.append(f"   contract_support_rows={ev.get('contract_activity_count')}")
        if ev.get("lobbying_activity_count") is not None:
            parts.append(f"   lobbying_support_rows={ev.get('lobbying_activity_count')}")
        wt = row.get("why_triggered") or []
        if isinstance(wt, list) and wt:
            parts.append("   why_triggered:")
            for line in wt[:6]:
                s = str(line).strip()
                if s:
                    parts.append(f"   - {s[:500]}")

    overlap = [o for o in (signals.get("overlap_signals") or []) if isinstance(o, dict)][:6]
    parts.append("\n## Strongest overlap signals (ranked list, head)")
    if not overlap:
        parts.append("(none)")
    for o in overlap:
        tk = str(o.get("ticker", "")).strip().upper()
        parts.append(
            f"- {tk}: congress_lines={o.get('congress_filing_count')} reps={o.get('unique_representatives')} "
            f"insider_rows={o.get('insider_rows')} buy_like={o.get('buy_like_count')} sell_like={o.get('sell_like_count')} "
            f"large_dollar_congress={'yes' if o.get('large_dollar_congress') else 'no'}"
        )
        if o.get("contract_activity_count") is not None:
            parts.append(f"  contract_support_rows={o.get('contract_activity_count')}")
        if o.get("lobbying_activity_count") is not None:
            parts.append(f"  lobbying_support_rows={o.get('lobbying_activity_count')}")

    cdf = signals.get("congress_df")
    if not isinstance(cdf, pd.DataFrame):
        cdf = pd.DataFrame()
    hp = get_high_profile_congress_trades(cdf)
    parts.append("\n## High-profile congressional trades (curated, up to 8)")
    if hp.empty:
        parts.append("(none)")
    else:
        for _, hr in hp.head(8).iterrows():
            rep = str(hr.get("Representative", ""))[:80]
            tkr = str(hr.get("Ticker", "")).strip().upper()
            txn = str(hr.get("Transaction", ""))[:100]
            rng = str(hr.get("Range", ""))[:140]
            parts.append(f"- {rep} | {tkr} | {txn} | Range: {rng}")

    ranked_all = [r for r in (signals.get("ranked_signals") or []) if isinstance(r, dict)]
    overlap_all = [o for o in (signals.get("overlap_signals") or []) if isinstance(o, dict)]
    dir_md = _directional_signal_summary_section(ranked_all, overlap_all)
    parts.append("\n## Directional summary (from MVP)")
    cap = 4000
    parts.append(dir_md if len(dir_md) <= cap else dir_md[:cap] + "\n…(truncated)")

    parts.append("\n## Run-level counts (supporting)")
    parts.append(
        f"congress_rows={signals.get('congress_row_count')} | insider_rows={signals.get('insider_row_count')} | "
        f"contracts_rows={signals.get('contracts_row_count')} | lobbying_rows={signals.get('lobbying_row_count')} | "
        f"overlap_symbols={len(signals.get('overlap') or [])} | "
        f"contracts_symbols_reinforced={signals.get('contracts_symbols_reinforced')} | "
        f"lobbying_symbols_reinforced={signals.get('lobbying_symbols_reinforced')}"
    )

    large = signals.get("large_trades")
    if isinstance(large, pd.DataFrame) and not large.empty and "Ticker" in large.columns:
        tick_sample = large["Ticker"].astype(str).str.strip().head(6).tolist()
        parts.append(f"large_dollar_congress_trade_rows={len(large)} sample_tickers={tick_sample}")

    return "\n".join(parts)


def _parse_dashboard_wide_claude_json(raw: str) -> dict[str, Any] | None:
    """Validate Claude dashboard JSON: key_insight, 3 what_changed bullets, 3–5 newsletter bullets."""
    obj = _parse_ranked_claude_json(raw)
    if not isinstance(obj, dict):
        return None
    ki = str(obj.get("key_insight") or "").strip()
    if len(ki) < 12:
        return None
    wc = obj.get("what_changed_bullets")
    if not isinstance(wc, list) or len(wc) != 3:
        return None
    wc_clean = [str(x).strip() for x in wc]
    if any(not x for x in wc_clean):
        return None
    nl = obj.get("newsletter_insights_bullets")
    if not isinstance(nl, list) or not (3 <= len(nl) <= 5):
        return None
    nl_clean = [str(x).strip() for x in nl]
    if any(not x for x in nl_clean):
        return None
    return {
        "key_insight": ki,
        "what_changed_bullets": wc_clean,
        "newsletter_insights_bullets": nl_clean,
    }


def _dashboard_summary_combined_user_prompt(evidence_block: str) -> str:
    """
    Single user turn for the dashboard summary Claude call: explicit combined task + evidence block.

    Key Insight, What Changed, and Newsletter strips are all produced from this one prompt + system JSON rules.
    """
    eb = (evidence_block or "").strip()
    return (
        "You are writing **three dashboard sections in one JSON object**. Use **only** facts that appear in the "
        "evidence block below—no invented tickers, counts, dates, or events.\n\n"
        "Deliver in that single JSON:\n"
        "1) **key_insight** — one or two sentences: the clearest takeaway for this run.\n"
        "2) **what_changed_bullets** — exactly three one-line bullets describing what stands out in this pull "
        "(specific to the data, not boilerplate).\n"
        "3) **newsletter_insights_bullets** — three to five one-line, copy-ready bullets for a drafting desk.\n\n"
        "Match the exact key names and array lengths required in your system message. Reply with JSON only "
        "(no markdown fences, no commentary).\n\n"
        "=== EVIDENCE FOR THIS RUN ===\n"
        f"{eb}\n"
    )


def try_apply_claude_dashboard_summaries(signals: dict[str, Any]) -> None:
    """
    **One** Claude call (cached via ``call_claude`` → ``output/claude_run_cache.json``): fills
    ``_dash_claude_key_insight``, ``_dash_claude_what_changed``, ``_dash_claude_newsletter`` when JSON validates;
    otherwise leaves keys unset (HTML uses deterministic fallbacks).
    """
    from claude_client import CLAUDE_MISSING_KEY_REPLY, call_claude

    if bool(signals.get("insider_only_fallback_mode")):
        return

    ctx = _dashboard_wide_claude_context(signals)
    if len(ctx.strip()) < 30:
        return

    user_combined = _dashboard_summary_combined_user_prompt(ctx)
    raw = call_claude(
        user_combined,
        system_prompt=_DASHBOARD_WIDE_CLAUDE_SYSTEM,
        max_tokens=2000,
    )
    if raw == CLAUDE_MISSING_KEY_REPLY or raw.startswith("[Assistant"):
        return

    parsed = _parse_dashboard_wide_claude_json(raw)
    if not parsed:
        return

    signals["_dash_claude_key_insight"] = parsed["key_insight"]
    signals["_dash_claude_what_changed"] = parsed["what_changed_bullets"]
    signals["_dash_claude_newsletter"] = parsed["newsletter_insights_bullets"]


def apply_dashboard_text_ai_enhancements(signals: SignalsBundle | dict[str, Any]) -> None:
    """
    Optional Claude pass: **up to three** calls—one per top anomaly hero card (insider watchlist or desk-ranked).

    Key Insight / What Changed / Newsletter are **not** re-called here (single combined call in
    ``try_apply_claude_dashboard_summaries``). No Claude for overlap, lower-ranked rows, or tables.
    On failure, hero cards fall back to deterministic HTML.
    """
    signals.pop("_ai_hero_card_texts", None)
    signals.pop("_ai_insider_hero_card_texts", None)
    signals.pop("_ai_hero_minimal_cards", None)
    signals.pop("_ai_insider_hero_minimal_cards", None)

    if os.environ.get("QQ_DISABLE_DASHBOARD_ENHANCEMENT", "").strip().lower() in ("1", "true", "yes"):
        return
    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return

    from claude_client import enhance_with_ai

    if bool(signals.get("insider_only_fallback_mode")):
        wl = [w for w in (signals.get("insider_fallback_watchlist") or []) if isinstance(w, dict)][:3]
        card_texts: list[str] = []
        for w in wl:
            if not isinstance(w, dict):
                card_texts.append("")
                continue
            tk = str(w.get("ticker", "")).strip().upper()
            raw_one = _collect_insider_watchlist_single_raw(w)
            ctx_one = {"insider_watchlist_card": _slim_insider_watchlist_for_context(w)}
            t_one, ok_one = enhance_with_ai(f"Top anomaly card: {tk}", raw_one, ctx_one)
            card_texts.append(t_one.strip() if ok_one and t_one.strip() else "")
        signals["_ai_insider_hero_card_texts"] = card_texts
        signals["_ai_insider_hero_minimal_cards"] = any(bool(x) for x in card_texts)
        return

    leaders = _top_ranked_hero_leaders(signals)[:3]
    card_texts_r: list[str] = []
    for row in leaders:
        if not isinstance(row, dict):
            card_texts_r.append("")
            continue
        tk = str(row.get("ticker", "")).strip().upper()
        raw_one = _collect_single_hero_card_raw(row)
        ctx_one = {"top_anomaly_card": _slim_ranked_for_context(row)}
        t_one, ok_one = enhance_with_ai(f"Top anomaly card: {tk}", raw_one, ctx_one)
        card_texts_r.append(t_one.strip() if ok_one and t_one.strip() else "")
    signals["_ai_hero_card_texts"] = card_texts_r
    signals["_ai_hero_minimal_cards"] = any(bool(x) for x in card_texts_r)


def generate_html_report(signals: SignalsBundle | dict[str, Any]) -> Path:
    """
    Write a standalone `output/research_dashboard.html` for local browser viewing.

    **Key Insight**, **What Changed**, and **Newsletter** share **one** combined Claude user prompt + system JSON
    spec (``try_apply_claude_dashboard_summaries``); **up to three** more calls optionally refine the top hero
    cards only (``apply_dashboard_text_ai_enhancements``). Responses are cached under ``output/claude_run_cache.json``.
    No Claude for ranked/overlap tables, directional HTML, or lower-ranked rows. Disable hero refinements with
    ``QQ_DISABLE_DASHBOARD_ENHANCEMENT=1``. ``ensure_ranked_distinctiveness`` / ``refresh_distinctiveness_on_ranked``
    attach a 0–1 distinctiveness bonus (see ``signal_logic``). ``ensure_dashboard_anomaly_views`` tightens hero /
    summary context tickers (full ``ranked_signals`` unchanged for tables).
    Below the hero anomalies, those strips precede **Ask AI Assistant**. The POST URL comes
    from QQ_CLAUDE_DASHBOARD_ENDPOINT or (if QQ_AUTO_LOCAL_CLAUDE) claude_proxy.py on 127.0.0.1 (CLAUDE_PROXY_PORT).
    Without a URL, submit shows an offline placeholder. Remaining order:
    summary → quick context → high-profile → ranked → overlap → directional →
    methodology → data availability → verification.
    """
    large = signals.get("large_trades", pd.DataFrame())
    top = signals.get("top_tickers", pd.DataFrame())
    overlap = list(signals.get("overlap") or [])
    overlap_signals = list(signals.get("overlap_signals") or [])
    ranked_signals = list(signals.get("ranked_signals") or [])

    if not isinstance(large, pd.DataFrame):
        large = pd.DataFrame()
    if not isinstance(top, pd.DataFrame):
        top = pd.DataFrame()

    congress_df = signals.get("congress_df", pd.DataFrame())
    if not isinstance(congress_df, pd.DataFrame):
        congress_df = pd.DataFrame()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    ensure_dashboard_anomaly_views(signals)
    try_apply_claude_dashboard_summaries(signals)
    apply_dashboard_text_ai_enhancements(signals)

    exec_html = _md_bullets_to_html(_executive_summary(signals))
    if not exec_html:
        exec_html = '<p class="muted">No executive summary available.</p>'

    anomalies_html = _html_top_anomalies_section(large, top, overlap, ranked_signals)
    hp_html = _html_high_profile_blocks(congress_df)
    ranked_html = _html_top_ranked_blocks(ranked_signals)
    overlap_html = _html_overlap_blocks(overlap, overlap_signals)
    _dir_md = _directional_signal_summary_section(ranked_signals, overlap_signals)
    _dir_m = re.search(r"##[^\n]+\n\n(.+?)\n\n###", _dir_md, re.DOTALL)
    dir_intro = (
        f"<p>{_format_inline_md(_dir_m.group(1).strip())}</p>" if _dir_m else ""
    )
    dir_buckets = _directional_buckets_html(ranked_signals, overlap_signals)
    if dir_intro.strip():
        dir_html = (
            f'<div class="section-stack directional-stack">'
            f'<div class="info-box info-box--compact">{dir_intro}</div>'
            f"{dir_buckets}</div>"
        )
    else:
        dir_html = f'<div class="section-stack directional-stack">{dir_buckets}</div>'
    meth_html = _methodology_body_html(_methodology_section())
    avail_html = _data_availability_list_html(_data_availability_section(signals))
    verify_html = _verification_body_html(_potential_next_step_verification_section(signals))

    def sec(sid: str, title: str, body: str) -> str:
        eid = html.escape(sid)
        et = html.escape(title)
        return (
            f'<section class="report-section" id="{eid}" aria-labelledby="{eid}-h">'
            f'<h2 class="section-title" id="{eid}-h">{et}</h2>'
            f'<div class="section-body">{body}</div></section>'
        )

    congress_warn_html = _html_congress_unavailable_banner(signals)
    hero_html = _html_top_anomalies_hero_section(signals)
    what_changed_html = _html_what_changed_this_run_section(signals)
    exec_copy_html = _html_executive_copy_block(signals)
    claude_endpoint = _resolve_claude_dashboard_endpoint()
    claude_ctx_b64 = _embed_claude_report_context_b64(signals, generated_utc=now)
    ask_claude_html = _html_ask_claude_panel(
        api_endpoint=claude_endpoint,
        context_b64=claude_ctx_b64,
    )
    post_hero_row = (
        f'<div class="post-hero-split">'
        f'<div class="post-hero-split__main">{exec_copy_html}</div>'
        f'<div class="post-hero-split__aside">{ask_claude_html}</div>'
        "</div>"
    )
    ask_claude_script = _html_ask_claude_script(backend_port=_qq_backend_port())
    quick_actions_html = _html_quick_actions_bar(
        include_what_changed=bool(what_changed_html.strip()),
    )
    key_insight_html = _html_key_insight_callout(signals)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Research Dashboard — Weekly Brief</title>
  <style>{_html_dashboard_styles()}</style>
</head>
<body>
  <header>
    <h1>Research Dashboard</h1>
    <p class="sub">Weekly alternative-data brief &mdash; generated {html.escape(now)}</p>
  </header>
  <main>
    {congress_warn_html}
    {key_insight_html}
    {quick_actions_html}
    {hero_html}
    {what_changed_html}
    {post_hero_row}
    {sec("executive-summary", "Executive summary", exec_html)}
    {sec("top-anomalies", "Quick context", anomalies_html)}
    {sec("high-profile", "High-profile congressional trades", hp_html)}
    {sec("top-ranked", "Top ranked signals", ranked_html)}
    {sec("overlap", "Strongest cross-dataset overlap signals", overlap_html)}
    {sec("directional", "Directional signal summary", dir_html)}
    {sec("methodology", "Methodology", meth_html)}
    {sec("data-availability", "Data availability notes", avail_html)}
    {sec("verification", "Potential next-step verification", verify_html)}
  </main>
{ask_claude_script}
{_html_top_card_scroll_script()}
</body>
</html>"""

    HTML_DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_DASHBOARD_PATH.write_text(doc, encoding="utf-8")
    return HTML_DASHBOARD_PATH


def _write_charts(signals: SignalsBundle | dict[str, Any]) -> tuple[bool, bool]:
    """Save both PNGs under output/charts/. Returns (congress_ok, insider_ok)."""
    c_df = signals.get("congress_df", pd.DataFrame())
    i_df = signals.get("insider_df", pd.DataFrame())
    if not isinstance(c_df, pd.DataFrame):
        c_df = pd.DataFrame()
    if not isinstance(i_df, pd.DataFrame):
        i_df = pd.DataFrame()

    ok_c = _save_top_ticker_bar_chart(
        c_df,
        CONGRESS_CHART_FILE,
        chart_title="Top 10 congressional tickers (row count)",
    )
    ok_i = _save_top_ticker_bar_chart(
        i_df,
        INSIDER_CHART_FILE,
        chart_title="Top 10 insider tickers (row count)",
    )
    return ok_c, ok_i


def generate_markdown_report(signals: SignalsBundle | dict[str, Any]) -> Path:
    """
    Build the weekly brief from real signal outputs and write `output/weekly_research_brief.md`.

    `signals` should include DataFrames/lists produced by `signal_logic` plus optional row counts for context.
    """
    large = signals.get("large_trades", pd.DataFrame())
    top = signals.get("top_tickers", pd.DataFrame())
    overlap = list(signals.get("overlap", []))
    overlap_signals = list(signals.get("overlap_signals", []))
    ranked_signals = list(signals.get("ranked_signals", []))

    if not isinstance(large, pd.DataFrame):
        large = pd.DataFrame()
    if not isinstance(top, pd.DataFrame):
        top = pd.DataFrame()

    congress_df = signals.get("congress_df", pd.DataFrame())
    if not isinstance(congress_df, pd.DataFrame):
        congress_df = pd.DataFrame()

    congress_chart_ok, insider_chart_ok = _write_charts(signals)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body = "\n".join(
        [
            "# Weekly Research Brief",
            "",
            f"_Generated: {now}_",
            "",
            "## Executive Summary",
            "",
            _executive_summary(signals),
            "",
            _data_availability_section(signals),
            _charts_section(congress_chart_ok=congress_chart_ok, insider_chart_ok=insider_chart_ok),
            _signal1_section(large),
            "",
            _signal2_section(top),
            "",
            _high_profile_section(congress_df),
            "",
            _strongest_overlap_section(overlap, overlap_signals),
            "",
            _top_ranked_section(ranked_signals),
            "",
            _directional_signal_summary_section(ranked_signals, overlap_signals),
            _methodology_section(),
            _potential_next_step_verification_section(signals),
        ]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(body, encoding="utf-8")
    return OUTPUT_PATH
