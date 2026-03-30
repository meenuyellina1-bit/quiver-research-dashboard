"""
Fetch Quiver live data, then inspect columns, preview, and missing values.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from quiver_api import (
    CONGRESS_TRADING_URL,
    INSIDER_TRADING_URL,
    fetch_government_contracts,
    fetch_lobbying_data,
    fetch_off_exchange_data,
    get_congress_trading,
    get_insider_trading,
)
from report_generator import (
    generate_html_report,
    generate_markdown_report,
)
from signal_logic import (
    attach_story_worthiness_to_ranked,
    get_cross_dataset_tickers,
    get_insider_only_watchlist,
    get_large_congress_trades,
    get_strongest_overlap_signals,
    get_top_congress_tickers,
    get_top_ranked_signals,
    refresh_distinctiveness_on_ranked,
)

API_KEY = os.getenv("QUIVER_API_TOKEN", "").strip()

# Set to True for token presence, endpoint URLs, and row counts (no secret values printed).
DEBUG = False

# Appended only when gov-contract rows exist for the same symbol as a ranked / overlap lead.
_CONTRACT_WHY_APPEND = (
    "This name also appears in federal contract data, suggesting potential alignment between policy exposure and business activity."
)
_CONTRACT_ANGLE_APPEND = (
    "This signal is reinforced by the company’s presence in recent government contract activity."
)

# Lobbying: same pattern—only merged into leads that already cleared ranked / overlap filters.
_LOBBYING_WHY_APPEND = (
    "This name also appears in Quiver lobbying data, which can make the political relevance more tangible than congressional trading alone."
)
_LOBBYING_ANGLE_APPEND = "The signal is reinforced by lobbying activity tied to the same ticker."


def _contracts_ticker_column(df: pd.DataFrame) -> str | None:
    for name in ("Ticker", "ticker", "Symbol"):
        if name in df.columns:
            return name
    return None


def _contracts_date_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "date" in c.lower():
            return c
    return None


def build_contracts_summary(
    contracts_df: pd.DataFrame,
    relevant_tickers: set[str],
) -> dict[str, dict]:
    """
    One entry per symbol (uppercase key) that appears in both the filtered contract feed
    and the relevant-ticker set (top-ranked + strongest overlap only).

    recent_flag: True if the latest parsed contract date is within the last 365 days (when a date column exists).
    """
    if contracts_df.empty or not relevant_tickers:
        return {}

    tcol = _contracts_ticker_column(contracts_df)
    if tcol is None:
        return {}

    rel = {t.strip().upper() for t in relevant_tickers if t and str(t).strip()}
    work = contracts_df.copy()
    work["_tk"] = work[tcol].astype(str).str.strip().str.upper()
    sub = work[work["_tk"].isin(rel)]
    if sub.empty:
        return {}

    dcol = _contracts_date_column(sub)
    summary: dict[str, dict] = {}
    for tk, grp in sub.groupby("_tk"):
        count = int(len(grp))
        recent_flag = False
        if dcol and dcol in grp.columns:
            dt = pd.to_datetime(grp[dcol], errors="coerce")
            mx = dt.max()
            if pd.notna(mx):
                mx_ts = pd.Timestamp(mx)
                if mx_ts.tzinfo is not None:
                    mx_ts = mx_ts.tz_convert(None)
                delta = pd.Timestamp.now().normalize() - mx_ts.normalize()
                recent_flag = bool(delta.days <= 365)
        summary[str(tk)] = {"count": count, "recent_flag": recent_flag}
    return summary


def apply_contract_support_to_ranked(
    ranked_signals: list[dict],
    contracts_summary: dict[str, dict],
) -> None:
    """Mutate in place: add evidence + append copy only when contracts reinforce the symbol."""
    for item in ranked_signals:
        k = str(item.get("ticker", "")).strip().upper()
        if not k or k not in contracts_summary:
            continue
        cs = contracts_summary[k]
        item.setdefault("evidence", {})
        item["evidence"]["contract_activity_count"] = cs["count"]
        item["why_may_matter"] = (item.get("why_may_matter") or "").rstrip() + " " + _CONTRACT_WHY_APPEND
        item["newsletter_angle"] = (item.get("newsletter_angle") or "").rstrip() + " " + _CONTRACT_ANGLE_APPEND


def apply_contract_support_to_overlap(
    overlap_signals: list[dict],
    contracts_summary: dict[str, dict],
) -> None:
    """Mutate in place: attach count + extend why (no separate contracts section)."""
    for row in overlap_signals:
        k = str(row.get("ticker", "")).strip().upper()
        if not k or k not in contracts_summary:
            continue
        row["contract_activity_count"] = contracts_summary[k]["count"]
        row["why_may_matter"] = (row.get("why_may_matter") or "").rstrip() + " " + _CONTRACT_WHY_APPEND


def _lobbying_amount_column(df: pd.DataFrame) -> str | None:
    """First column that looks like a dollar / spending field (summed per ticker when present)."""
    for c in df.columns:
        cl = c.lower()
        if any(x in cl for x in ("amount", "value", "spending", "payment", "fee", "cost", "total")):
            return c
    return None


def build_lobbying_summary(
    lobbying_df: pd.DataFrame,
    relevant_tickers: set[str],
) -> dict[str, dict]:
    """
    Per-symbol lobbying rows among top-ranked / overlap tickers only.
    amount: sum of numeric values in the best-matching spending column, if any; else None.
    """
    if lobbying_df.empty or not relevant_tickers:
        return {}

    tcol = _contracts_ticker_column(lobbying_df)
    if tcol is None:
        return {}

    rel = {t.strip().upper() for t in relevant_tickers if t and str(t).strip()}
    work = lobbying_df.copy()
    work["_tk"] = work[tcol].astype(str).str.strip().str.upper()
    sub = work[work["_tk"].isin(rel)]
    if sub.empty:
        return {}

    dcol = _contracts_date_column(sub)
    acol = _lobbying_amount_column(sub)
    summary: dict[str, dict] = {}
    for tk, grp in sub.groupby("_tk"):
        count = int(len(grp))
        recent_flag = False
        if dcol and dcol in grp.columns:
            dt = pd.to_datetime(grp[dcol], errors="coerce")
            mx = dt.max()
            if pd.notna(mx):
                mx_ts = pd.Timestamp(mx)
                if mx_ts.tzinfo is not None:
                    mx_ts = mx_ts.tz_convert(None)
                delta = pd.Timestamp.now().normalize() - mx_ts.normalize()
                recent_flag = bool(delta.days <= 365)
        amount_val: float | None = None
        if acol and acol in grp.columns:
            s = pd.to_numeric(grp[acol], errors="coerce").sum()
            if pd.notna(s) and float(s) != 0.0:
                amount_val = float(s)
        summary[str(tk)] = {"count": count, "recent_flag": recent_flag, "amount": amount_val}
    return summary


def apply_lobbying_support_to_ranked(
    ranked_signals: list[dict],
    lobbying_summary: dict[str, dict],
) -> None:
    for item in ranked_signals:
        k = str(item.get("ticker", "")).strip().upper()
        if not k or k not in lobbying_summary:
            continue
        ls = lobbying_summary[k]
        item.setdefault("evidence", {})
        item["evidence"]["lobbying_activity_count"] = ls["count"]
        if ls.get("amount") is not None:
            item["evidence"]["lobbying_amount"] = ls["amount"]
        item["why_may_matter"] = (item.get("why_may_matter") or "").rstrip() + " " + _LOBBYING_WHY_APPEND
        item["newsletter_angle"] = (item.get("newsletter_angle") or "").rstrip() + " " + _LOBBYING_ANGLE_APPEND


def apply_lobbying_support_to_overlap(
    overlap_signals: list[dict],
    lobbying_summary: dict[str, dict],
) -> None:
    for row in overlap_signals:
        k = str(row.get("ticker", "")).strip().upper()
        if not k or k not in lobbying_summary:
            continue
        ls = lobbying_summary[k]
        row["lobbying_activity_count"] = ls["count"]
        if ls.get("amount") is not None:
            row["lobbying_amount"] = ls["amount"]
        row["why_may_matter"] = (row.get("why_may_matter") or "").rstrip() + " " + _LOBBYING_WHY_APPEND


def _print_dataset_block(title: str, df) -> None:
    """Print column names and first 3 rows for one DataFrame."""
    print(f"=== {title} ===")
    print(f"Columns: {df.columns.tolist()}")
    print("Preview:")
    if df.empty:
        print("  (empty — no rows to show)")
    else:
        print(df.head(3).to_string())
    print()


def _print_missing_values(label: str, df) -> None:
    """Print missing value counts per column, one line each."""
    print(f"{label}")
    if df.empty and len(df.columns) == 0:
        print("  (no columns)")
        return
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        print(f"  {col}: {n_missing}")


def main() -> None:
    if not API_KEY:
        print("Missing QUIVER_API_TOKEN. Please set it in your environment before running.")
        sys.exit(1)

    congress_df = get_congress_trading(API_KEY)
    insider_df = get_insider_trading(API_KEY)

    congress_ok = not congress_df.empty
    insider_ok = not insider_df.empty
    insider_only_fallback_mode = not congress_ok and insider_ok
    insider_fallback_watchlist = (
        get_insider_only_watchlist(insider_df) if insider_only_fallback_mode else []
    )

    if DEBUG:
        print("--- DEBUG ---")
        print(f"Token present: {'yes' if API_KEY else 'no'}")
        print(f"Congress endpoint: {CONGRESS_TRADING_URL}")
        print(f"Insider endpoint:  {INSIDER_TRADING_URL}")
        print(f"Congress row count: {len(congress_df)}")
        print(f"Insider row count: {len(insider_df)}")
        print("-------------\n")

    # Case A: both have data — no extra message
    # Case B: congress yes, insider no
    if congress_ok and not insider_ok:
        print("Congress data loaded successfully, but insider data is unavailable.")
        print()
    # Case C: congress no, insider yes (insider-only fallback; no desk-ranked scores)
    elif insider_only_fallback_mode:
        print(
            "Congress data unavailable for this run. "
            "Dashboard generated in insider-only fallback mode."
        )
        print()
    # Case D: both empty
    elif not congress_ok and not insider_ok:
        print(
            "Both congress and insider datasets are unavailable. "
            "This is likely an API-side issue or configuration issue."
        )
        print()

    print(f"Congress rows: {len(congress_df)}")
    print(f"Insider rows: {len(insider_df)}")
    print("Patents: not fetched (endpoint not confirmed).")
    print()

    _print_dataset_block("CONGRESS DATA", congress_df)
    _print_dataset_block("INSIDER DATA", insider_df)

    print("=== MISSING VALUES ===")
    _print_missing_values("Congress:", congress_df)
    print()
    _print_missing_values("Insider:", insider_df)
    print()

    large_trades = get_large_congress_trades(congress_df)
    top_tickers = get_top_congress_tickers(congress_df)
    overlap = get_cross_dataset_tickers(congress_df, insider_df)
    overlap_signals = get_strongest_overlap_signals(congress_df, insider_df, overlap)
    ranked_signals = get_top_ranked_signals(congress_df, insider_df)

    # Symbols for local filtering after a single supporting-API pull (ranked + overlap; never widens the set).
    relevant_for_contracts: set[str] = set()
    for s in ranked_signals:
        relevant_for_contracts.add(str(s.get("ticker", "")).strip())
    for s in overlap_signals:
        relevant_for_contracts.add(str(s.get("ticker", "")).strip())
    relevant_for_contracts.discard("")

    # Supporting Quiver endpoints: at most one fetch each per run, only if desk-ranked anomalies exist.
    has_top_ranked_anomaly = any(
        isinstance(s, dict) and str(s.get("ticker", "")).strip() for s in ranked_signals
    )
    contracts_df = pd.DataFrame()
    lobbying_df = pd.DataFrame()
    off_exchange_df = pd.DataFrame()
    contracts_api_queried = False
    lobbying_api_queried = False
    off_exchange_api_queried = False
    contracts_summary: dict[str, dict] = {}
    lobbying_summary: dict[str, dict] = {}

    if has_top_ranked_anomaly:
        contracts_api_queried = True
        lobbying_api_queried = True
        off_exchange_api_queried = True
        contracts_df = fetch_government_contracts(API_KEY)
        lobbying_df = fetch_lobbying_data(API_KEY)
        off_exchange_df = fetch_off_exchange_data(API_KEY)
        contracts_summary = build_contracts_summary(contracts_df, relevant_for_contracts)
        apply_contract_support_to_ranked(ranked_signals, contracts_summary)
        apply_contract_support_to_overlap(overlap_signals, contracts_summary)
        lobbying_summary = build_lobbying_summary(lobbying_df, relevant_for_contracts)
        apply_lobbying_support_to_ranked(ranked_signals, lobbying_summary)
        apply_lobbying_support_to_overlap(overlap_signals, lobbying_summary)

    print(f"Contracts: {'queried, ' + str(len(contracts_df)) + ' row(s)' if contracts_api_queried else 'not queried'}")
    print(f"Lobbying: {'queried, ' + str(len(lobbying_df)) + ' row(s)' if lobbying_api_queried else 'not queried'}")
    print(
        f"Off-exchange: {'queried, ' + str(len(off_exchange_df)) + ' row(s)' if off_exchange_api_queried else 'not queried'}"
    )
    print()

    if ranked_signals:
        refresh_distinctiveness_on_ranked(ranked_signals, congress_df)
        attach_story_worthiness_to_ranked(ranked_signals, congress_df, overlap)

    print("=== LARGE TRADES ===")
    if large_trades.empty:
        print("  (no matching rows)")
    else:
        cols = [c for c in large_trades.columns if not str(c).startswith("_")]
        print(large_trades[cols].to_string())
    print()

    print("=== TOP TICKERS ===")
    if top_tickers.empty:
        print("  Ticker | Count")
        print("  (no data)")
    else:
        print("Ticker | Count | Conviction | Note (trimmed)")
        for _, row in top_tickers.iterrows():
            note = str(row.get("ConvictionNote", ""))[:72]
            cv = int(row["CongressConviction"]) if "CongressConviction" in top_tickers.columns else 0
            print(f"{row['Ticker']} | {int(row['Count'])} | {cv} | {note}")
    print()

    print("=== OVERLAP ===")
    print(f"Total overlap count: {len(overlap)}")
    print("First 20 overlap tickers (alphabetical):")
    if not overlap:
        print("  (none)")
    else:
        for t in overlap[:20]:
            print(f"  {t}")
    print("Strongest overlap (ranked; includes congressional conviction in score):")
    for row in overlap_signals[:8]:
        t = row.get("ticker", "")
        cc = row.get("congress_conviction_0_100")
        note = str(row.get("congress_conviction_note", ""))[:100]
        d = row.get("direction", "Mixed")
        print(f"  {t} | {d} | congress conviction {cc}/100 | {note}")
    print()

    print("=== TOP RANKED SIGNALS ===")
    if not ranked_signals:
        print("  (none)")
    else:
        for item in ranked_signals:
            print(f"  Ticker: {item['ticker']}")
            sc = item.get("score", 0)
            try:
                sc_f = float(sc)
            except (TypeError, ValueError):
                sc_f = 0.0
            print(f"  Score: {sc_f:.1f}/10")
            sw = item.get("story_worthiness_score")
            if sw is not None:
                try:
                    print(
                        f"  Story-worthiness (newsletter ordering layer): {float(sw):.1f}/5 "
                        "(separate from desk score; strong signal ≠ strong story and vice versa)"
                    )
                except (TypeError, ValueError):
                    pass
            comp = item.get("score_components")
            if isinstance(comp, dict) and comp:
                labels = [
                    ("congress_frequency", "Congress frequency"),
                    ("representative_breadth", "Rep breadth"),
                    ("high_dollar_trade", "High-dollar trade"),
                    ("insider_overlap", "Insider overlap"),
                    ("insider_bias", "Insider bias"),
                    ("recency", "Recency"),
                ]
                pen_labels = [
                    ("penalty_concentration", "Concentration penalty"),
                    ("penalty_mixed_signal", "Mixed insider penalty"),
                ]
                print("  Scoring breakdown:")
                for key, lab in labels:
                    if key not in comp:
                        continue
                    try:
                        cv = float(comp[key])
                    except (TypeError, ValueError):
                        continue
                    print(f"    - {lab}: {cv:.1f}")
                for key, lab in pen_labels:
                    if key not in comp:
                        continue
                    try:
                        pv = float(comp[key])
                    except (TypeError, ValueError):
                        continue
                    if pv <= 0:
                        continue
                    print(f"    - {lab}: -{pv:.1f}")
            print(f"  Direction: {item.get('direction', 'Mixed')}")
            if item.get("direction_note"):
                print(f"  Direction note: {item['direction_note']}")
            if item.get("congress_conviction_0_100") is not None:
                print(f"  Congressional conviction: {item['congress_conviction_0_100']}/100")
            if item.get("congress_conviction_note"):
                print(f"  Why conviction: {item['congress_conviction_note']}")
            print("  Why it triggered:")
            for line in item.get("why_triggered", []):
                print(f"    - {line}")
            ev = item.get("evidence", {})
            print("  Evidence:")
            print(f"    congressional filing count: {ev.get('congress_filings', 0)}")
            print(f"    unique representative count: {ev.get('unique_representatives', 0)}")
            print(f"    insider row count: {ev.get('insider_rows', 0)}")
            print(f"    buy-like count: {ev.get('insider_buys', 0)}")
            print(f"    sell-like count: {ev.get('insider_sells', 0)}")
            lg = ev.get("large_congress_trade", False)
            print(f"    whether there was a large-dollar congressional trade: {'yes' if lg else 'no'}")
            if "contract_activity_count" in ev:
                print(f"    contract activity count: {ev.get('contract_activity_count', 0)}")
            if "lobbying_activity_count" in ev:
                print(f"    lobbying activity count: {ev.get('lobbying_activity_count', 0)}")
            if "lobbying_amount" in ev:
                print(f"    lobbying amount (summed from API field): {ev.get('lobbying_amount')}")
            if "congress_conviction_0_100" in ev:
                print(f"    congressional conviction (ticker-level): {ev.get('congress_conviction_0_100')}/100")
            if ev.get("congress_conviction_note"):
                print(f"    why conviction is elevated: {ev.get('congress_conviction_note')}")
            if ev.get("direction"):
                print(f"    directional tilt: {ev.get('direction')}")
            if ev.get("direction_note"):
                print(f"    why this directional label: {ev.get('direction_note')}")
            if "congress_recency_factor" in ev:
                print(f"    congress recency factor (0-1): {ev.get('congress_recency_factor')}")
            if "insider_recency_factor" in ev:
                print(f"    insider recency factor (0-1): {ev.get('insider_recency_factor')}")
            print("  Why it may matter:")
            print(f"    - {item.get('why_may_matter', '')}")
            print("  Suggested newsletter angle:")
            print(f"    - {item.get('newsletter_angle', '')}")
            print()
    print()

    bundle = {
        "large_trades": large_trades,
        "top_tickers": top_tickers,
        "overlap": overlap,
        "overlap_signals": overlap_signals,
        "ranked_signals": ranked_signals,
        "congress_row_count": len(congress_df),
        "insider_row_count": len(insider_df),
        "contracts_row_count": len(contracts_df),
        "lobbying_row_count": len(lobbying_df),
        "off_exchange_row_count": len(off_exchange_df),
        "patents_row_count": 0,
        "patents_disabled": True,
        "contracts_symbols_reinforced": len(contracts_summary),
        "lobbying_symbols_reinforced": len(lobbying_summary),
        "had_core_tickers_for_support": bool(relevant_for_contracts),
        "contracts_api_queried": contracts_api_queried,
        "lobbying_api_queried": lobbying_api_queried,
        "off_exchange_api_queried": off_exchange_api_queried,
        "congress_df": congress_df,
        "insider_df": insider_df,
        "insider_only_fallback_mode": insider_only_fallback_mode,
        "insider_fallback_watchlist": insider_fallback_watchlist,
    }
    report_path = generate_markdown_report(bundle)
    print(f"Wrote research brief: {report_path.resolve()}")
    html_path = generate_html_report(bundle)
    print(f"Wrote HTML dashboard: {html_path.resolve()}")


if __name__ == "__main__":
    main()
