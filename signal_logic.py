"""
Signal helpers — large trades, frequency, overlap, clusters, insider activity, and ranked scores.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

# Substrings in `Range` that flag larger reported dollar bands (congress disclosures)
_LARGE_RANGE_MARKERS = (
    "500,001",
    "1,000,000",
    "5,000,000",
    "25,000,000",
)

_MIN_CLUSTER_REPS = 2
_TOP_RANKED = 5

# Name needles (lower rank = higher editorial priority). Longer/specific names first so "Nancy Pelosi" beats "Pelosi".
_HIGH_PROFILE_NAME_PRIORITY: tuple[tuple[str, int], ...] = (
    ("Nancy Pelosi", 0),
    ("Pelosi", 1),
    ("David H. McCormick", 2),
    ("McCormick", 3),
    ("Mitch McConnell", 4),
    ("Charles Schumer", 5),
    ("Chuck Schumer", 5),
    ("Hakeem Jeffries", 6),
    ("Steve Scalise", 7),
    ("John Thune", 8),
    ("Mike Johnson", 9),
    ("Kevin McCarthy", 10),
    ("Michael McCaul", 11),
    ("Dan Crenshaw", 12),
    ("Josh Gottheimer", 13),
    ("Tom Malinowski", 14),
    ("Dan Goldman", 15),
    ("Virginia Foxx", 16),
    ("Ro Khanna", 17),
)


def get_large_congress_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows where Range mentions one of the large-dollar markers.

    Up to 10 rows, ordered by **congressional conviction** (trade-level rule score: purchase tilt,
    high-profile member, recency — large Range is already implicit in the filter).
    Adds ``CongressConviction`` (0–100) and ``ConvictionNote`` for the brief.
    """
    if df.empty or "Range" not in df.columns:
        return pd.DataFrame()

    s = df["Range"].astype(str)
    mask = False
    for marker in _LARGE_RANGE_MARKERS:
        mask = mask | s.str.contains(marker, regex=False, na=False)

    out = df.loc[mask].copy()
    if out.empty:
        return out

    conv = out.apply(lambda r: compute_congress_conviction_for_row(r), axis=1)
    out["_conv"] = conv.map(lambda x: x[0]).astype(float)
    out["_conv_note"] = conv.map(lambda x: x[1]).astype(str)
    out = out.sort_values("_conv", ascending=False).head(10)
    out["CongressConviction"] = (out["_conv"] * 100).round().astype(int)
    out["ConvictionNote"] = out["_conv_note"]
    out = out.drop(columns=["_conv", "_conv_note"])
    return out.reset_index(drop=True)


def _congress_large_range_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask for rows whose Range includes a large-dollar marker (full table)."""
    if df.empty or "Range" not in df.columns:
        return pd.Series(False, index=df.index)
    s = df["Range"].astype(str)
    mask = False
    for marker in _LARGE_RANGE_MARKERS:
        mask = mask | s.str.contains(marker, regex=False, na=False)
    return mask


def get_top_congress_tickers(df: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    """
    Top tickers by **congressional conviction** (rule-based), not raw frequency alone.

    Tie-breaker: higher row count. Columns: Ticker, Count, CongressConviction, ConvictionNote
    """
    empty_cols = ["Ticker", "Count", "CongressConviction", "ConvictionNote"]
    if df.empty or "Ticker" not in df.columns:
        return pd.DataFrame(columns=empty_cols)

    tickers = df["Ticker"].dropna().astype(str).str.strip()
    tickers = tickers[tickers != ""]
    if tickers.empty:
        return pd.DataFrame(columns=empty_cols)

    rows: list[dict[str, Any]] = []
    for t in tickers.unique():
        conv_01, _comp, note = compute_congress_conviction_for_ticker(df, t)
        cnt = int((tickers == t).sum())
        rows.append(
            {
                "Ticker": t,
                "Count": cnt,
                "CongressConviction": int(round(conv_01 * 100)),
                "ConvictionNote": note,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["CongressConviction", "Count", "Ticker"],
        ascending=[False, False, True],
    ).head(top_n)
    return out.reset_index(drop=True)


def get_cross_dataset_tickers(df_congress: pd.DataFrame, df_insiders: pd.DataFrame) -> list[str]:
    """
    Tickers that appear in both congress and insider tables (set intersection).
    """
    if "Ticker" not in df_congress.columns or "Ticker" not in df_insiders.columns:
        return []

    c = df_congress["Ticker"].dropna().astype(str).str.strip()
    i = df_insiders["Ticker"].dropna().astype(str).str.strip()
    overlap = set(c.unique()) & set(i.unique())
    overlap.discard("")
    return sorted(overlap)


def get_congressional_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Congressional cluster signal: tickers with filings from 2+ different representatives.

    Columns: Ticker, UniqueRepresentatives
    """
    if df.empty or "Ticker" not in df.columns or "Representative" not in df.columns:
        return pd.DataFrame(columns=["Ticker", "UniqueRepresentatives"])

    work = df.copy()
    work["_tk"] = work["Ticker"].astype(str).str.strip()
    g = (
        work.groupby("_tk", dropna=False)["Representative"]
        .nunique()
        .reset_index(name="UniqueRepresentatives")
        .rename(columns={"_tk": "Ticker"})
    )
    out = g[g["UniqueRepresentatives"] >= _MIN_CLUSTER_REPS].sort_values(
        "UniqueRepresentatives", ascending=False
    )
    return out.reset_index(drop=True)


def _classify_insider_side_from_row(row: pd.Series) -> str:
    """
    Prefer SEC-style fields from live Quiver data:
    - AcquiredDisposedCode: A = acquisition (buy-like), D = disposition (sell-like)
    - TransactionCode: secondary hint (P/S/M etc.)
    """
    adc = row.get("AcquiredDisposedCode")
    if pd.notna(adc):
        adc_s = str(adc).strip().upper()
        if adc_s == "A":
            return "buy"
        if adc_s == "D":
            return "sell"

    tc = row.get("TransactionCode")
    if pd.notna(tc):
        tc_s = str(tc).strip().upper()
        # Common Form 4 transaction codes (not exhaustive; A/D above is authoritative when present)
        if tc_s in ("P", "P/A", "A", "G", "M", "C", "E", "I", "L"):
            return "buy"
        if tc_s in ("S", "D", "F"):
            return "sell"

    # Text fallback on Transaction / Type
    for col in ("Transaction", "Type", "transactionType", "TransactionType"):
        if col in row.index and pd.notna(row[col]):
            t = str(row[col]).lower()
            if any(x in t for x in ("purchase", "buy", "acquisition", "grant")):
                return "buy"
            if any(x in t for x in ("sale", "sell", "disposition")):
                return "sell"
    return "other"


def get_insider_activity_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insider activity by ticker: totals and buy/sell using AcquiredDisposedCode + TransactionCode.

    Columns: Ticker, TotalTrades, Buys, Sells, Other
    """
    empty = pd.DataFrame(columns=["Ticker", "TotalTrades", "Buys", "Sells", "Other"])
    if df.empty or "Ticker" not in df.columns:
        return empty

    work = df.copy()
    work["_side"] = work.apply(_classify_insider_side_from_row, axis=1)

    rows: list[dict[str, Any]] = []
    for ticker, grp in work.groupby(work["Ticker"].astype(str).str.strip()):
        if not ticker:
            continue
        side = grp["_side"]
        rows.append(
            {
                "Ticker": ticker,
                "TotalTrades": len(grp),
                "Buys": int((side == "buy").sum()),
                "Sells": int((side == "sell").sum()),
                "Other": int((side == "other").sum()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return empty
    return out.sort_values("TotalTrades", ascending=False).reset_index(drop=True)


def _insider_latest_dates_by_ticker(insider_df: pd.DataFrame) -> dict[str, pd.Timestamp | None]:
    """Best-effort max parsed date per ticker from common Quiver column names."""
    out: dict[str, pd.Timestamp | None] = {}
    if insider_df.empty or "Ticker" not in insider_df.columns:
        return out
    work = insider_df.copy()
    work["_tk"] = work["Ticker"].astype(str).str.strip().str.upper()
    work = work[work["_tk"] != ""]
    date_cols = [c for c in ("Date", "fileDate", "FileDate", "FilingDate") if c in work.columns]
    if not date_cols:
        return out
    for tk, grp in work.groupby("_tk"):
        mx: pd.Timestamp | None = None
        for c in date_cols:
            dt = pd.to_datetime(grp[c], errors="coerce")
            m = dt.max()
            if pd.isna(m):
                continue
            cur = pd.Timestamp(m)
            if cur.tzinfo is not None:
                cur = cur.tz_convert("UTC").replace(tzinfo=None)
            if mx is None or cur > mx:
                mx = cur
        out[str(tk)] = mx
    return out


def _days_since_timestamp(ts: pd.Timestamp | None) -> int | None:
    if ts is None or bool(pd.isna(ts)):
        return None
    tsn = pd.Timestamp(ts)
    if tsn.tzinfo is not None:
        tsn = tsn.tz_convert("UTC").replace(tzinfo=None)
    d = int((pd.Timestamp.now().normalize() - tsn.normalize()).days)
    return max(0, d)


def get_insider_only_watchlist(insider_df: pd.DataFrame, *, top_n: int = 3) -> list[dict[str, Any]]:
    """
    When congressional data is missing, surface up to ``top_n`` insider-led names without desk scoring.

    Selection (distinct tickers when possible):
    1) highest insider row count
    2) strongest buy/sell imbalance among the remainder
    3) most recent insider-dated filing among the remainder (falls back to volume if dates are sparse)
    """
    act = get_insider_activity_by_ticker(insider_df)
    if act.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, r in act.iterrows():
        tk = str(r.get("Ticker", "")).strip().upper()
        if not tk:
            continue
        tot = int(r.get("TotalTrades", 0) or 0)
        b = int(r.get("Buys", 0) or 0)
        s2 = int(r.get("Sells", 0) or 0)
        imb = abs(b - s2)
        rows.append(
            {
                "ticker": tk,
                "total_trades": tot,
                "buys": b,
                "sells": s2,
                "imbalance": imb,
                "imb_ratio": imb / max(tot, 1),
            }
        )

    latest_by_ticker = _insider_latest_dates_by_ticker(insider_df)
    used: set[str] = set()
    chosen: list[tuple[str, dict[str, Any]]] = []

    if not rows:
        return []

    first = max(rows, key=lambda x: (x["total_trades"], x["imbalance"], x["imb_ratio"]))
    chosen.append(("highest_volume", first))
    used.add(first["ticker"])

    cand2 = [x for x in rows if x["ticker"] not in used]
    if cand2 and len(chosen) < top_n:
        second = max(cand2, key=lambda x: (x["imbalance"], x["imb_ratio"], x["total_trades"]))
        chosen.append(("strongest_imbalance", second))
        used.add(second["ticker"])

    cand3 = [x for x in rows if x["ticker"] not in used]
    if cand3 and len(chosen) < top_n:

        def _rec_key(x: dict[str, Any]) -> tuple[int, int, int]:
            ds = _days_since_timestamp(latest_by_ticker.get(x["ticker"]))
            if ds is None:
                return (1, 99999, -x["total_trades"])
            return (0, ds, -x["total_trades"])

        third = min(cand3, key=_rec_key)
        chosen.append(("recent_activity", third))

    slot_labels = {
        "highest_volume": "Heaviest insider activity (row count) in this download.",
        "strongest_imbalance": "Strongest buy/sell skew among remaining names in this download.",
        "recent_activity": "Most recent insider-dated activity among remaining names (or next-busiest if dates are sparse).",
    }

    out_list: list[dict[str, Any]] = []
    for slot, row in chosen[:top_n]:
        tk = row["ticker"]
        b, s2, tot = row["buys"], row["sells"], row["total_trades"]
        if b > s2:
            skew = f"Buy-heavy in this pull ({b} buy-like vs {s2} sell-like rows)."
        elif s2 > b:
            skew = f"Sell-heavy in this pull ({s2} sell-like vs {b} buy-like rows)."
        else:
            skew = f"Balanced buy/sell tagging this pull ({b} buy-like / {s2} sell-like rows)."

        ds = _days_since_timestamp(latest_by_ticker.get(tk))
        if ds is not None:
            rec_note = f"Latest parsed insider date is about {ds} day(s) ago."
        else:
            rec_note = "Insider filing dates were sparse or unparsed for this symbol in the download."

        summary_line = f"{tk} — {tot} insider row(s) in this pull; {skew}"
        newsletter_line = (
            f"{tk}: {tot} insider row(s), {skew.rstrip('.')}; {rec_note.rstrip('.')}."
        )

        out_list.append(
            {
                "ticker": tk,
                "total_trades": tot,
                "buys": b,
                "sells": s2,
                "selection_slot": slot,
                "selection_note": slot_labels.get(slot, ""),
                "skew_summary": skew,
                "recency_note": rec_note,
                "title_line": f"{tk} — {tot} insider row(s)",
                "summary_line": summary_line,
                "newsletter_line": newsletter_line,
            }
        )

    return out_list


def _freq_rank(freq_series: pd.Series, ticker: str) -> int | None:
    """1-based rank in value_counts; None if ticker not present."""
    if ticker not in freq_series.index:
        return None
    rank = int(freq_series.index.get_loc(ticker)) + 1
    return rank


# ---------------------------------------------------------------------------
# Top-ranked desk score (0–10.0, one decimal) — transparent additive model, no ML.
# Six capped components are summed, then total is capped at 10.0 before rounding.
# ---------------------------------------------------------------------------


def _component_congress_frequency(freq_rank: int | None, filing_count: int, freq: pd.Series) -> float:
    """
    A. Congressional frequency (max 2.0).

    Top ranks by raw filing count in this pull; tail uses a percentile mapping so scores spread out.
    """
    if freq_rank is not None:
        if freq_rank == 1:
            return 2.0
        if freq_rank <= 3:
            return 1.7
        if freq_rank <= 5:
            return 1.4
        if freq_rank <= 10:
            return 1.0
    # Rank > 10 or unknown rank: scale by filing-count percentile among all tickers (0.2–0.9).
    if filing_count <= 0 or freq.empty:
        return 0.2
    all_counts = [int(v) for v in freq.values]
    n = len(all_counts)
    if n <= 0:
        return 0.2
    below = sum(1 for v in all_counts if v < filing_count)
    equal = sum(1 for v in all_counts if v == filing_count)
    pct = (below + 0.5 * equal) / n
    return 0.2 + 0.7 * pct


def _component_rep_breadth(unique_reps: int) -> float:
    """B. Unique representative breadth (max 1.8)."""
    ur = int(unique_reps)
    if ur >= 10:
        return 1.8
    if ur >= 8:
        return 1.5
    if ur >= 6:
        return 1.2
    if ur >= 4:
        return 0.9
    if ur >= 2:
        return 0.5
    if ur >= 1:
        return 0.2
    return 0.0


def _component_high_dollar_trade(sub_c: pd.DataFrame) -> float:
    """
    C. High-dollar congressional Range (max 1.7).

    Uses the largest dollar figure parsed from any Range cell for this ticker.
    """
    if sub_c.empty or "Range" not in sub_c.columns:
        return 0.0
    mx = 0
    for rng in sub_c["Range"]:
        mx = max(mx, _range_magnitude_score(rng))
    if mx >= 1_000_001:
        return 1.7
    if 500_001 <= mx <= 1_000_000:
        return 1.3
    if mx >= 100_001:
        return 0.7
    return 0.0


def _component_insider_overlap_volume(in_overlap: bool, insider_rows: int) -> float:
    """D. Insider overlap + volume (max 1.8). Requires ticker in cross-dataset overlap."""
    ir = int(insider_rows)
    if not in_overlap or ir <= 0:
        return 0.0
    if ir <= 2:
        return 0.5
    if ir <= 5:
        return 0.9
    if ir <= 15:
        return 1.2
    if ir <= 40:
        return 1.5
    return 1.8


def _component_insider_bias(buy_like: int, sell_like: int, insider_rows: int) -> float:
    """
    E. Insider directional bias (max 1.2).

    Bullish (buy-heavy) uses a higher ceiling than bearish, per desk preference; ties get a small floor.
    """
    itot = int(insider_rows)
    if itot <= 0:
        return 0.0
    b, s = int(buy_like), int(sell_like)
    net_bias_ratio = abs(b - s) / max(itot, 1)
    if b > s:
        bonus = 0.4 + 0.8 * net_bias_ratio
    elif s > b:
        bonus = 0.2 + 0.6 * net_bias_ratio
    else:
        bonus = 0.2
    return min(1.2, bonus)


def _latest_activity_datetime_congress_insider(
    c_with_t: pd.DataFrame,
    insider_df: pd.DataFrame,
    ticker: str,
) -> pd.Timestamp | None:
    """Most recent parsed timestamp across congress (Report/Transaction) and insider rows for ``ticker``."""
    t = str(ticker).strip()
    sub_c = c_with_t.loc[c_with_t["_t"] == t]
    latest_c = _latest_parsed_max(sub_c, ("ReportDate", "TransactionDate")) if not sub_c.empty else None

    latest_i: pd.Timestamp | None = None
    if not insider_df.empty and "Ticker" in insider_df.columns:
        sub_i = insider_df.loc[insider_df["Ticker"].astype(str).str.strip() == t]
        cols = tuple(x for x in ("Date", "fileDate", "FileDate") if x in sub_i.columns)
        if cols and not sub_i.empty:
            latest_i = _latest_parsed_max(sub_i, cols)

    opts = [x for x in (latest_c, latest_i) if x is not None and pd.notna(x)]
    if not opts:
        return None
    return max(pd.Timestamp(x) for x in opts)


def _component_recency_from_latest(latest: pd.Timestamp | None) -> float:
    """
    F. Recency (max 1.5) from the newest congress or insider date for the ticker.

    Day buckets are wall-clock days before “today” (local normalization).
    """
    if latest is None or pd.isna(latest):
        return 0.1
    ts = pd.Timestamp(latest)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    days = (pd.Timestamp.now().normalize() - ts.normalize()).days
    if days < 0:
        days = 0
    if days <= 3:
        return 1.5
    if days <= 7:
        return 1.2
    if days <= 14:
        return 0.8
    if days <= 30:
        return 0.4
    return 0.1


def _penalty_concentration(filing_count: int, unique_reps: int) -> float:
    """
    Many congressional lines for the ticker but only one or two distinct members → less breadth.

    Fires when unique_reps is 1–2 and filing count is “high” (≥4 rows this pull).
    """
    ur = int(unique_reps)
    cc = int(filing_count)
    if ur < 1 or ur > 2:
        return 0.0
    if cc >= 4:
        return 0.3
    return 0.0


def _penalty_mixed_insider_signal(buy_like: int, sell_like: int) -> float:
    """
    Insider activity is heavy on both buy-like and sell-like sides and nearly balanced → weaker direction read.

    “High” = at least 3 on each side. “Close” = |buy − sell| ≤ max(2, 15% of buy+sell).
    """
    b, s = int(buy_like), int(sell_like)
    if b < 3 or s < 3:
        return 0.0
    total_bs = b + s
    gap = abs(b - s)
    closeness_floor = max(2, int(0.15 * total_bs))
    if gap <= closeness_floor:
        return 0.2
    return 0.0


def _apply_score_penalties(
    base_total: float,
    penalty_concentration: float,
    penalty_mixed: float,
) -> tuple[float, float, float, float]:
    """
    Subtract capped penalties from base_total; floor at 0. Returns
    (adjusted_total, concentration_applied, mixed_applied, penalties_total_applied).
    """
    raw = float(penalty_concentration) + float(penalty_mixed)
    cap = min(0.5, raw)
    if raw <= 0:
        return max(0.0, float(base_total)), 0.0, 0.0, 0.0
    if raw <= 0.5:
        conc_a, mix_a = float(penalty_concentration), float(penalty_mixed)
    else:
        scale = 0.5 / raw
        conc_a = round(float(penalty_concentration) * scale, 2)
        mix_a = round(0.5 - conc_a, 2)
    adj = max(0.0, float(base_total) - cap)
    return adj, round(conc_a, 1), round(mix_a, 1), round(cap, 1)


def _score_ticker_additive(
    *,
    freq_rank: int | None,
    filing_count: int,
    freq: pd.Series,
    unique_reps: int,
    sub_c: pd.DataFrame,
    in_overlap: bool,
    insider_total: int,
    insider_buys: int,
    insider_sells: int,
    c_with_t: pd.DataFrame,
    insider_df: pd.DataFrame,
    ticker: str,
) -> tuple[float, dict[str, float]]:
    """
    Weighted additive desk score: six positive components, minus capped penalties (max 0.5),
    then cap total at 10.0, floor at 0, rounded to one decimal.

    Returns (final_score, component_map) with stable keys for reports and JSON.
    """
    parts = {
        "congress_frequency": _component_congress_frequency(freq_rank, filing_count, freq),
        "representative_breadth": _component_rep_breadth(unique_reps),
        "high_dollar_trade": _component_high_dollar_trade(sub_c),
        "insider_overlap": _component_insider_overlap_volume(in_overlap, insider_total),
        "insider_bias": _component_insider_bias(insider_buys, insider_sells, insider_total),
        "recency": _component_recency_from_latest(
            _latest_activity_datetime_congress_insider(c_with_t, insider_df, ticker)
        ),
    }
    base_total = sum(parts.values())
    p_c = _penalty_concentration(filing_count, unique_reps)
    p_m = _penalty_mixed_insider_signal(insider_buys, insider_sells)
    after_penalties, conc_applied, mix_applied, pen_total = _apply_score_penalties(
        base_total, p_c, p_m
    )
    capped = min(10.0, after_penalties)
    score = round(capped, 1)
    rounded_parts: dict[str, float] = {k: round(v, 1) for k, v in parts.items()}
    rounded_parts["penalty_concentration"] = conc_applied
    rounded_parts["penalty_mixed_signal"] = mix_applied
    rounded_parts["penalties_total"] = pen_total
    return score, rounded_parts


def _build_signal_narrative(
    ticker: str,
    *,
    large_rows: int,
    freq_rank: int | None,
    unique_reps: int,
    in_overlap: bool,
    congress_count: int,
    insider_total: int,
    insider_buys: int,
    insider_sells: int,
    has_large: bool,
    congress_recency_01: float = 0.25,
    insider_recency_01: float = 0.25,
) -> dict[str, Any]:
    """
    Newsletter-oriented copy: every claim ties to counts/flags passed in (no invented filings or prices).
    """
    triggers: list[str] = []
    if has_large:
        triggers.append(
            f"Our large-trade filter matched at least one congressional row for {ticker} "
            f"(Range text includes a $500k+ style band; {large_rows} such row(s) in this pull)."
        )
    if freq_rank is not None and freq_rank <= 10:
        triggers.append(
            f"{ticker} sits at rank #{freq_rank} by raw congressional line count in this download—enough repetition to matter for a lead."
        )
    if unique_reps >= 2:
        triggers.append(
            f"Disclosure breadth: {unique_reps} distinct representatives touched {ticker} here, so the signal is not a single-member blip."
        )
    if in_overlap:
        triggers.append(
            f"{ticker} also appears in the insider file for the same run, so congress and Form 4–style activity overlap on the symbol."
        )
    if insider_total > 0 and (insider_buys > 0 or insider_sells > 0):
        triggers.append(
            f"Insider direction is parseable this run ({insider_buys} rows coded buy-like via AcquiredDisposedCode A / related fields, "
            f"{insider_sells} sell-like via D / related)—the model uses that split in the score."
        )
    elif insider_total > 0:
        triggers.append(
            f"There are {insider_total} insider row(s) for {ticker}, but buy/sell codes were sparse; volume still nudged the rank."
        )

    # Recency nudge (matches _recency_factor_from_latest tiers: ≥0.65 ≈ filings within ~90 days or fresher).
    if congress_count > 0 and congress_recency_01 >= 0.65:
        triggers.append(
            "The composite score includes a **congress recency** bump: the newest **ReportDate / TransactionDate** rows for this "
            "ticker sit in a fresher band, so it ranks a bit ahead of otherwise similar names with older Capitol lines—useful for a Sunday sweep."
        )
    if insider_total > 0 and insider_recency_01 >= 0.65:
        triggers.append(
            "The composite score includes an **insider recency** bump: **Date / fileDate** fields for this symbol skew recent in "
            "this pull, so fresher Form 4–style activity gets extra weight versus stale insider rows."
        )

    if not triggers:
        triggers.append(
            "This ticker cleared the candidate pool on frequency or overlap with modest rule hits—worth a line if the desk needs a second-tier watch."
        )

    # Plain-English "so what" — only facts we actually have
    if in_overlap and insider_total >= 2 and congress_count >= 3:
        why_matter = (
            f"{ticker} is one of the names where Capitol disclosures and insider lines stack up in the same dataset slice. "
            "For a retail reader or writer, that is less a trading signal than a reading list: pull the underlying filings before you imply timing or motive."
        )
    elif in_overlap and insider_buys > insider_sells and insider_buys > 0:
        why_matter = (
            f"Insider tagging skews buy-like ({insider_buys} vs {insider_sells} sell-like) while congress still shows activity—"
            "useful only as a prompt to check whether news or fundamentals already priced the story."
        )
    elif unique_reps >= 3 and congress_count >= 2:
        why_matter = (
            f"Several members filed {ticker} in this window; that pattern is what policy and markets reporters usually flag before assigning causality."
        )
    elif has_large and congress_count >= 1:
        why_matter = (
            f"A wide reported Range on a congressional line is the classic hook for disclosure coverage on {ticker}—"
            "the number is self-reported bands, not market cap or performance."
        )
    else:
        why_matter = (
            f"{ticker} earned its slot from the blended score (congress count {congress_count}, reps {unique_reps}, insider rows {insider_total}). "
            "Treat it as a desk note to verify filings, not as confirmation of an edge."
        )

    # One sentence David can lift — must restate only what we measured
    if has_large and in_overlap:
        angle = (
            f"David can open with {ticker} as a rare overlap this pull: Capitol disclosures include a large-range line and the symbol also hits the insider feed."
        )
    elif freq_rank is not None and freq_rank <= 3 and congress_count > 0:
        angle = (
            f"David can frame {ticker} as the busiest congressional symbol in this snapshot ({congress_count} lines, rank #{freq_rank}) and work backward to who filed what."
        )
    elif unique_reps >= 3:
        angle = (
            f"David can note that {ticker} showed up across {unique_reps} members in one download—a cluster worth pairing with the week’s headlines, not a thesis."
        )
    elif insider_total > 0 and insider_buys != insider_sells:
        angle = (
            f"David can cite {ticker}’s insider split ({insider_buys} buy-like, {insider_sells} sell-like rows) as a disclosure detail readers can verify in the source data."
        )
    else:
        angle = (
            f"David can use {ticker} as a compact disclosure lead: {congress_count} congressional rows, {insider_total} insider rows, "
            f"{'with' if has_large else 'without'} a large-range hit this run."
        )

    return {
        "why_triggered": triggers,
        "why_may_matter": why_matter,
        "newsletter_angle": angle,
        "evidence": {
            "congress_filings": congress_count,
            "unique_representatives": unique_reps,
            "insider_rows": insider_total,
            "insider_buys": insider_buys,
            "insider_sells": insider_sells,
            "large_congress_trade": has_large,
            "congress_recency_factor": round(float(congress_recency_01), 3),
            "insider_recency_factor": round(float(insider_recency_01), 3),
        },
    }


def _latest_parsed_max(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Timestamp | None:
    """Newest timestamp among named columns (each column max, then max across columns)."""
    found: list[pd.Timestamp] = []
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_datetime(df[col], errors="coerce")
        mx = s.max()
        if pd.notna(mx):
            found.append(pd.Timestamp(mx))
    if not found:
        return None
    return max(found)


def _recency_factor_from_latest(latest: pd.Timestamp | None) -> float:
    """
    Map the newest known filing date to 0–1 using plain day-count tiers (easy to explain, no curves).

    Used for conviction scoring and for ranked / overlap recency bonuses: very recent → full credit,
    then step down at 45 / 90 / 180 / 365 days; missing dates → modest default so we do not over-weight noise.
    """
    if latest is None or pd.isna(latest):
        return 0.35
    ts = pd.Timestamp(latest)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    days = (pd.Timestamp.now().normalize() - ts.normalize()).days
    if days <= 14:
        return 1.0
    if days <= 45:
        return 0.85
    if days <= 90:
        return 0.65
    if days <= 180:
        return 0.45
    if days <= 365:
        return 0.25
    return 0.1


def _congress_recency_factor_for_ticker(c_with_t: pd.DataFrame, ticker: str) -> float:
    """
    0–1 bonus input for ranked scoring: fresher Capitol filings score higher (Sunday research workflow).

    Uses **ReportDate** and **TransactionDate** only when present; maps through ``_recency_factor_from_latest``.
    """
    sub = c_with_t.loc[c_with_t["_t"] == str(ticker).strip()]
    if sub.empty:
        return 0.25
    latest = _latest_parsed_max(sub, ("ReportDate", "TransactionDate"))
    return _recency_factor_from_latest(latest)


def _insider_recency_factor_for_ticker(insider_df: pd.DataFrame, ticker: str) -> float:
    """
    0–1 bonus for ranked scoring from insider **Date** / **fileDate** / **FileDate** when present.

    Same tier scale as congress recency so the two feeds are easy to explain side by side.
    """
    if insider_df.empty or "Ticker" not in insider_df.columns:
        return 0.25
    t = str(ticker).strip()
    sub = insider_df.loc[insider_df["Ticker"].astype(str).str.strip() == t]
    if sub.empty:
        return 0.25
    cols = tuple(c for c in ("Date", "fileDate", "FileDate") if c in sub.columns)
    if not cols:
        return 0.25
    latest = _latest_parsed_max(sub, cols)
    return _recency_factor_from_latest(latest)


def get_top_ranked_signals(
    congress_df: pd.DataFrame,
    insider_df: pd.DataFrame,
    *,
    top_n: int = _TOP_RANKED,
) -> list[dict[str, Any]]:
    """
    Top ``top_n`` tickers by additive desk score (0.0–10.0, one decimal) plus narrative fields for the report.

    Ordering here stays **desk-score-only**. A separate **Story-Worthiness** layer (0–5) is attached later for
    newsletter/hero/qualified ordering only—strong desk rank can still be a weak story, and the reverse.
    """
    if congress_df.empty or "Ticker" not in congress_df.columns:
        return []

    c = congress_df.copy()
    c["_t"] = c["Ticker"].astype(str).str.strip()

    mask_large = _congress_large_range_mask(c)
    large_by_ticker = c.loc[mask_large].groupby("_t").size() if mask_large.any() else pd.Series(dtype=int)

    freq = c["_t"].value_counts()
    clusters = get_congressional_clusters(c)
    rep_map = (
        clusters.set_index("Ticker")["UniqueRepresentatives"].to_dict()
        if not clusters.empty
        else {}
    )

    overlap_raw = get_cross_dataset_tickers(congress_df, insider_df)
    overlap_set = {x.strip() for x in overlap_raw if x and str(x).strip()}

    insider_act = get_insider_activity_by_ticker(insider_df)
    insider_lookup = (
        insider_act.set_index("Ticker").to_dict("index") if not insider_act.empty else {}
    )

    candidates: set[str] = set()
    candidates |= set(freq.head(15).index)
    candidates |= set(large_by_ticker.index)
    candidates |= set(rep_map.keys())
    candidates |= overlap_set
    candidates |= set(insider_lookup.keys())
    candidates.discard("")
    candidates.discard("nan")

    ranked: list[tuple[float, float, str, dict[str, Any]]] = []
    for ticker in sorted(candidates):
        if not ticker or ticker == "nan":
            continue
        lr = int(large_by_ticker.get(ticker, 0))
        fr = _freq_rank(freq, ticker)
        if ticker in rep_map:
            ur = int(rep_map[ticker])
        elif "Representative" in c.columns:
            ur = int(c.loc[c["_t"] == ticker, "Representative"].nunique())
        else:
            ur = 0

        in_ov = ticker in overlap_set
        ins = insider_lookup.get(ticker, {})
        itot = int(ins.get("TotalTrades", 0)) if ins else 0
        ibuy = int(ins.get("Buys", 0)) if ins else 0
        isell = int(ins.get("Sells", 0)) if ins else 0
        cc = int((c["_t"] == ticker).sum())
        sub_c = c.loc[c["_t"] == ticker]
        cb, cs, _co = _ticker_buy_sell_other_counts(sub_c)
        direction, direction_note = classify_directional_signal(
            congress_buys=cb,
            congress_sells=cs,
            insider_buys=ibuy,
            insider_sells=isell,
        )

        cr01 = _congress_recency_factor_for_ticker(c, ticker)
        ir01 = _insider_recency_factor_for_ticker(insider_df, ticker)

        score, score_components = _score_ticker_additive(
            freq_rank=fr,
            filing_count=cc,
            freq=freq,
            unique_reps=ur,
            sub_c=sub_c,
            in_overlap=in_ov,
            insider_total=itot,
            insider_buys=ibuy,
            insider_sells=isell,
            c_with_t=c,
            insider_df=insider_df,
            ticker=ticker,
        )

        narrative = _build_signal_narrative(
            ticker,
            large_rows=lr,
            freq_rank=fr,
            unique_reps=ur,
            in_overlap=in_ov,
            congress_count=cc,
            insider_total=itot,
            insider_buys=ibuy,
            insider_sells=isell,
            has_large=lr > 0,
            congress_recency_01=cr01,
            insider_recency_01=ir01,
        )

        conv_01, _conv_comp, conv_note = compute_congress_conviction_for_ticker(congress_df, ticker)
        conv_100 = int(round(conv_01 * 100))
        narrative["evidence"]["congress_conviction_0_100"] = conv_100
        narrative["evidence"]["congress_conviction_note"] = conv_note
        narrative["evidence"]["direction"] = direction
        narrative["evidence"]["direction_note"] = direction_note
        narrative["evidence"]["congress_buy_rows"] = cb
        narrative["evidence"]["congress_sell_rows"] = cs

        payload = {
            "ticker": ticker,
            "score": score,
            "score_components": score_components,
            "direction": direction,
            "direction_note": direction_note,
            "congress_conviction_0_100": conv_100,
            "congress_conviction_note": conv_note,
            **narrative,
            # Short line for quick terminal / legacy prints
            "explanation": " ".join(narrative["why_triggered"][:2]),
        }
        # Primary sort: desk score; secondary: congressional conviction (Capitol tie-break).
        ranked.append((score, conv_01, ticker, payload))

    ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [p for _, _, _, p in ranked[:top_n]]


def _overlap_rank_score(
    congress_count: int,
    unique_reps: int,
    insider_rows: int,
    buy_like: int,
    has_large_dollar: bool,
) -> float:
    """
    Sort overlaps for the report. Weights are intentional and easy to explain:
    - More congressional lines → more attention in this pull.
    - More distinct members → breadth, not a one-off.
    - Insider rows → second dataset confirms activity.
    - Buy-like (A-code) rows → directional read when codes exist.
    - Large-dollar Range on congress → headline-friendly disclosure band.
    """
    return (
        float(congress_count) * 1.0
        + float(unique_reps) * 3.0
        + float(insider_rows) * 1.2
        + float(buy_like) * 1.5
        + (15.0 if has_large_dollar else 0.0)
    )


def _overlap_why_may_matter(
    *,
    ticker: str,
    cc: int,
    ur: int,
    ir: int,
    buys: int,
    sells: int,
    large: bool,
) -> str:
    """One line, facts-only, for the overlap section."""
    if large and buys > sells and ir >= 2:
        return (
            f"{ticker} combines a wide congressional Range band with more buy-coded insider rows than sells in this slice—"
            "a natural place to point readers at primary filings before anyone implies timing."
        )
    if large:
        return (
            f"The congressional side includes a large-dollar Range for {ticker}; overlap with insider rows means two disclosure streams agree on the symbol this run."
        )
    if ur >= 3 and cc >= 2:
        return (
            f"{ur} members and {cc} congressional lines on {ticker}, plus insider activity, justify a short 'who is in this name?' note without over-reading price."
        )
    if ir >= 3:
        return (
            f"Insider volume ({ir} rows) on a symbol that also hits congress is a density signal—useful for prioritizing reads, not for ranking stocks."
        )
    return (
        f"{ticker} sits in the intersection of both feeds ({cc} congress lines, {ir} insider rows); treat that as a desk bookmark, not a thesis."
    )


def get_strongest_overlap_signals(
    congress_df: pd.DataFrame,
    insider_df: pd.DataFrame,
    overlap_tickers: list[str],
    *,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """
    Rank overlap tickers (congress ∩ insider) using filing counts, member breadth, insider rows,
    buy-like count, and a high-dollar congressional Range flag. Returns top ``limit`` dicts for the report.
    """
    if not overlap_tickers:
        return []

    c = congress_df.copy()
    c["_t"] = c["Ticker"].astype(str).str.strip()
    mask_large = _congress_large_range_mask(c)
    large_by_ticker = c.loc[mask_large].groupby("_t").size() if mask_large.any() else pd.Series(dtype=int)

    insider_act = get_insider_activity_by_ticker(insider_df)
    insider_lookup = insider_act.set_index("Ticker").to_dict("index") if not insider_act.empty else {}

    rows_out: list[dict[str, Any]] = []
    for raw in overlap_tickers:
        t = str(raw).strip()
        if not t:
            continue
        cc = int((c["_t"] == t).sum()) if not c.empty else 0
        if "Representative" in c.columns:
            ur = int(c.loc[c["_t"] == t, "Representative"].nunique())
        else:
            ur = 0
        ins = insider_lookup.get(t, {})
        ir = int(ins.get("TotalTrades", 0))
        buys = int(ins.get("Buys", 0))
        sells = int(ins.get("Sells", 0))
        lr = int(large_by_ticker.get(t, 0))
        has_large = lr > 0

        conv_01, _ccomp, conv_note = compute_congress_conviction_for_ticker(congress_df, t)
        conv_100 = int(round(conv_01 * 100))
        cr01 = _congress_recency_factor_for_ticker(c, t)
        ir01 = _insider_recency_factor_for_ticker(insider_df, t)
        # Same recency idea as top-ranked scoring: small additive bump so fresher overlap names float up.
        rank = (
            _overlap_rank_score(cc, ur, ir, buys, has_large)
            + 22.0 * conv_01
            + 5.0 * cr01
            + 5.0 * ir01
        )
        why = _overlap_why_may_matter(
            ticker=t, cc=cc, ur=ur, ir=ir, buys=buys, sells=sells, large=has_large
        )

        sub_c = c.loc[c["_t"] == t]
        cb, cs, _co = _ticker_buy_sell_other_counts(sub_c)
        direction, direction_note = classify_directional_signal(
            congress_buys=cb,
            congress_sells=cs,
            insider_buys=buys,
            insider_sells=sells,
        )

        rows_out.append(
            {
                "ticker": t,
                "direction": direction,
                "direction_note": direction_note,
                "congress_recency_factor": round(cr01, 3),
                "insider_recency_factor": round(ir01, 3),
                "congress_filing_count": cc,
                "unique_representatives": ur,
                "insider_rows": ir,
                "buy_like_count": buys,
                "sell_like_count": sells,
                "large_dollar_congress": has_large,
                "congress_conviction_0_100": conv_100,
                "congress_conviction_note": conv_note,
                "rank_score": rank,
                "why_may_matter": why,
            }
        )

    rows_out.sort(key=lambda x: (-x["rank_score"], x["ticker"]))
    return rows_out[:limit]


def get_overlap_ranked_for_report(
    congress_df: pd.DataFrame,
    insider_df: pd.DataFrame,
    overlap_tickers: list[str],
    *,
    limit: int = 15,
) -> list[tuple[str, float]]:
    """Backward-compatible (ticker, rank_score) pairs for any legacy callers."""
    detailed = get_strongest_overlap_signals(congress_df, insider_df, overlap_tickers, limit=limit)
    return [(d["ticker"], float(d["rank_score"])) for d in detailed]


def _high_profile_name_priority(rep: str) -> int:
    """Lower = more prominent for the case-study section."""
    name = str(rep).lower()
    best = 999
    for needle, rank in _HIGH_PROFILE_NAME_PRIORITY:
        if needle.lower() in name:
            best = min(best, rank)
    return best


def _md_escape_profile(val: Any) -> str:
    s = "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val)
    return s.replace("\n", " ").strip()


def _range_magnitude_score(range_str: Any) -> int:
    """
    Heuristic dollar scale from Range text (higher = larger reported band).
    Parses comma-separated integers from the string and takes the max.
    """
    s = str(range_str) if pd.notna(range_str) else ""
    parts = re.findall(r"[\d,]+", s)
    vals: list[int] = []
    for p in parts:
        try:
            vals.append(int(p.replace(",", "")))
        except ValueError:
            continue
    return max(vals) if vals else 0


def _congress_transaction_side_score(transaction: Any) -> int:
    """
    Sort key: purchases rank above sales when other factors tie.
    2 = clear purchase, 0 = clear sale, 1 = unclear / mixed.
    """
    t = str(transaction).lower() if pd.notna(transaction) else ""
    is_sale = "sale" in t or "sell" in t or "sold" in t
    is_buy = "purchase" in t or "buy" in t or "acquisition" in t
    if is_buy and not is_sale:
        return 2
    if is_sale and not is_buy:
        return 0
    return 1


# ---------------------------------------------------------------------------
# Congressional conviction score (rule-based, transparent — no ML)
#
# Ticker-level and trade-level scores summarize: purchase vs sale mix, large Range
# exposure, high-profile members, filing repetition, multi-member breadth, recency.
# Weights are fixed constants; each factor is normalized to ~0–1 before weighting.
# ---------------------------------------------------------------------------

# Ticker-level weights (must sum to 1.0).
_CONV_W_PURCHASE_SHARE = 0.18  # fraction of rows tagged buy-like vs all congress rows for symbol
_CONV_W_LARGE_SHARE = 0.20  # share of that symbol's rows hitting large-dollar Range markers
_CONV_W_HIGH_PROFILE = 0.17  # presence / intensity of watch-list members
_CONV_W_REPETITION = 0.15  # log-scaled row count (sustained attention, not raw rank)
_CONV_W_MULTI_REP = 0.18  # distinct representatives (breadth)
_CONV_W_RECENCY = 0.12  # how recent the latest filing is

# Trade-level (large-trade rows): used only to order which large trades to show first.
_CONV_ROW_W_LARGE = 0.35  # row already passed large-Range filter → full credit here
_CONV_ROW_W_PURCHASE = 0.25
_CONV_ROW_W_PROFILE = 0.25
_CONV_ROW_W_RECENCY = 0.15


def _congress_date_series_for_recency(df: pd.DataFrame) -> pd.Series:
    """Best-effort filing dates for recency (first available column wins)."""
    for col in ("ReportDate", "TransactionDate", "Date", "FilingDate"):
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def _row_recency_factor(row: pd.Series) -> float:
    for col in ("ReportDate", "TransactionDate", "Date", "FilingDate"):
        if col in row.index and pd.notna(row[col]):
            dt = pd.to_datetime(row[col], errors="coerce")
            if pd.notna(dt):
                return _recency_factor_from_latest(dt)
    return 0.35


def _ticker_buy_sell_other_counts(sub: pd.DataFrame) -> tuple[int, int, int]:
    """Counts of buy-like / sell-like / other congress rows from Transaction text."""
    if sub.empty or "Transaction" not in sub.columns:
        n = len(sub)
        return 0, 0, n
    buys = sells = other = 0
    for v in sub["Transaction"]:
        s = _congress_transaction_side_score(v)
        if s == 2:
            buys += 1
        elif s == 0:
            sells += 1
        else:
            other += 1
    return buys, sells, other


def classify_directional_signal(
    *,
    congress_buys: int,
    congress_sells: int,
    insider_buys: int,
    insider_sells: int,
) -> tuple[str, str]:
    """
    Rule-based Bullish / Bearish / Mixed from parsed disclosure text and insider codes only (not price predictions).

    - **Bullish:** Congress is purchase-heavy (strictly more buy-tagged rows than sell-tagged) *and* insiders are
      acquisition-heavy (strictly more buy-coded rows than sell-coded).
    - **Bearish:** Congress is sale-heavy *and* insiders are disposition-heavy.
    - **Mixed:** Conflicting tilts, ties, uncoded rows, or missing direction in either feed.
    """
    c_buy_heavy = congress_buys > congress_sells
    c_sell_heavy = congress_sells > congress_buys
    i_buy_heavy = insider_buys > insider_sells
    i_sell_heavy = insider_sells > insider_buys

    parseable_congress = congress_buys > 0 or congress_sells > 0
    parseable_insider = insider_buys > 0 or insider_sells > 0

    if c_buy_heavy and i_buy_heavy:
        return (
            "Bullish",
            "Both feeds align on parsed direction: more purchase-tagged congressional rows than sale-tagged, "
            "and more buy-coded (acquisition-like) insider rows than sell-coded.",
        )
    if c_sell_heavy and i_sell_heavy:
        return (
            "Bearish",
            "Both feeds align on parsed direction: more sale-tagged congressional rows than purchase-tagged, "
            "and more sell-coded (disposition-like) insider rows than buy-coded.",
        )

    parts: list[str] = []
    if not parseable_insider:
        parts.append("insider buy/sell codes were missing or all uncoded in this pull")
    elif c_buy_heavy and i_sell_heavy:
        parts.append("congress is purchase-heavy but insider rows skew sell-coded")
    elif c_sell_heavy and i_buy_heavy:
        parts.append("congress is sale-heavy but insider rows skew buy-coded")
    elif not parseable_congress:
        parts.append("congress Transaction text did not yield a clear purchase vs sale majority")
    elif (congress_buys == congress_sells and parseable_congress) or (insider_buys == insider_sells and parseable_insider):
        parts.append("at least one feed is tied on buy vs sell counts")
    else:
        parts.append("the two feeds do not jointly satisfy the bullish or bearish rules")

    return "Mixed", "Mixed / watchlist: " + "; ".join(parts) + "."


def _ticker_has_high_profile_congress_rep(congress_df: pd.DataFrame, ticker: str) -> bool:
    """True if any congressional row for ``ticker`` names a representative on the high-profile needle list."""
    if congress_df.empty or "Ticker" not in congress_df.columns or "Representative" not in congress_df.columns:
        return False
    t = str(ticker).strip()
    sub = congress_df.loc[congress_df["Ticker"].astype(str).str.strip() == t]
    if sub.empty:
        return False
    return bool(sub["Representative"].map(lambda r: _high_profile_name_priority(str(r)) < 900).any())


def compute_ranked_row_distinctiveness(
    row: dict[str, Any],
    *,
    congress_df: pd.DataFrame,
) -> tuple[float, dict[str, float]]:
    """
    Deterministic **distinctiveness bonus** in ``[0.0, 1.0]`` for ranking dashboard highlights.

    Each slice applies only when the structured ``evidence`` / ``direction`` fields (and congress frame
    for name checks) clearly support it—no guessed fundamentals.

    **Weighted sum (capped at 1.0)** — weights chosen so the maximum realistic stack hits 1.0:

    - ``high_profile_rep`` (0.18): ≥1 Capitol row for this ticker uses a high-profile representative name.
    - ``large_dollar_congress`` (0.18): ``evidence.large_congress_trade`` is true.
    - ``congress_insider_divergence`` (0.26): both feeds have parseable tilt (≥2 congress buy/sell rows,
      ≥3 insider buy+sell rows) and **opposite** leans (Capitol buy-heavy vs insider sell-heavy, or the reverse).
    - ``cross_feed_alignment`` (0.26): ``direction`` is Bullish or Bearish **and** ≥2 insider rows **and**
      ≥2 congress buy+sell rows (the label already encodes joint alignment in ``classify_directional_signal``).
    - ``representative_breadth_unusual`` (0.12): ≥6 unique representatives on ``evidence``.
    - ``supporting_contracts_or_lobbying`` (0.10): ``contract_activity_count`` or ``lobbying_activity_count``
      is present and > 0 on ``evidence``.

    **Primary uses:** ordering qualified / hero dashboard anomalies, newsletter ticker lines, and (separately)
    high-profile trade cards via ``_high_profile_trade_distinctiveness_score``.
    """
    ev = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    tk = str(row.get("ticker", "")).strip()

    W_HP = 0.18
    W_LARGE = 0.18
    W_DIV = 0.26
    W_ALIGN = 0.26
    W_BREADTH = 0.12
    W_SUPP = 0.10

    cb = int(ev.get("congress_buy_rows") or 0)
    cs = int(ev.get("congress_sell_rows") or 0)
    ibuy = int(ev.get("insider_buys") or 0)
    isell = int(ev.get("insider_sells") or 0)
    ins_tot = int(ev.get("insider_rows") or 0)
    ur = int(ev.get("unique_representatives") or 0)
    direction = str(row.get("direction", "Mixed")).strip()

    hp = 1.0 if (tk and _ticker_has_high_profile_congress_rep(congress_df, tk)) else 0.0
    lg = 1.0 if bool(ev.get("large_congress_trade")) else 0.0

    c_parse = (cb + cs) >= 2
    i_parse = (ibuy + isell) >= 3
    c_buy_heavy = cb > cs
    c_sell_heavy = cs > cb
    i_buy_heavy = ibuy > isell
    i_sell_heavy = isell > ibuy
    div = (
        1.0
        if (c_parse and i_parse and ((c_buy_heavy and i_sell_heavy) or (c_sell_heavy and i_buy_heavy)))
        else 0.0
    )

    align = (
        1.0
        if (direction in ("Bullish", "Bearish") and ins_tot >= 2 and (cb + cs) >= 2)
        else 0.0
    )

    br = 1.0 if ur >= 6 else 0.0

    sup = 0.0
    try:
        cac = ev.get("contract_activity_count")
        if cac is not None and int(cac) > 0:
            sup = 1.0
    except (TypeError, ValueError):
        pass
    if sup == 0.0:
        try:
            lac = ev.get("lobbying_activity_count")
            if lac is not None and int(lac) > 0:
                sup = 1.0
        except (TypeError, ValueError):
            pass

    bonus = min(
        1.0,
        W_HP * hp + W_LARGE * lg + W_DIV * div + W_ALIGN * align + W_BREADTH * br + W_SUPP * sup,
    )
    components = {
        "high_profile_rep": hp,
        "large_dollar_congress": lg,
        "congress_insider_divergence": div,
        "cross_feed_alignment": align,
        "representative_breadth_unusual": br,
        "supporting_contracts_or_lobbying": sup,
    }
    return round(bonus, 4), {k: round(v, 4) for k, v in components.items()}


def refresh_distinctiveness_on_ranked(
    ranked_signals: list[dict[str, Any]],
    congress_df: pd.DataFrame,
) -> None:
    """Mutates each ranked row in place: ``distinctiveness_bonus`` + ``distinctiveness_components``."""
    cdf = congress_df if isinstance(congress_df, pd.DataFrame) else pd.DataFrame()
    for row in ranked_signals:
        if not isinstance(row, dict):
            continue
        b, comp = compute_ranked_row_distinctiveness(row, congress_df=cdf)
        row["distinctiveness_bonus"] = b
        row["distinctiveness_components"] = comp


def compute_congress_conviction_for_ticker(
    congress_df: pd.DataFrame,
    ticker: str,
) -> tuple[float, dict[str, float], str]:
    """
    Rule-based congressional conviction for one symbol (0.0–1.0).

    Returns (score_0_1, component dict for transparency, one short sentence on why it is elevated).
    """
    if congress_df.empty or "Ticker" not in congress_df.columns:
        return 0.0, {}, "No congressional rows to score."

    t = str(ticker).strip()
    c = congress_df.copy()
    c["_t"] = c["Ticker"].astype(str).str.strip()
    sub = c[c["_t"] == t]
    n = int(len(sub))
    if n == 0:
        return 0.0, {}, "No congressional rows for this ticker."

    mask_large = _congress_large_range_mask(sub)
    large_rows = int(mask_large.sum())
    large_share = min(1.0, large_rows / n)

    buys, sells, _other = _ticker_buy_sell_other_counts(sub)
    # Purchase share among rows with a clear side; if none, use buy/(n) as soft tilt
    denom = buys + sells
    if denom > 0:
        purchase_share = buys / denom
    else:
        purchase_share = buys / n if n else 0.0

    if "Representative" in sub.columns:
        profile_hits = int(
            sub["Representative"].map(lambda r: _high_profile_name_priority(str(r)) < 900).sum()
        )
        ur = int(sub["Representative"].nunique())
    else:
        profile_hits = 0
        ur = 0

    high_profile = min(1.0, profile_hits / max(1, min(3, n)))

    repetition = min(1.0, math.log1p(n) / math.log1p(12))
    multi_rep = min(1.0, max(0.0, (ur - 1) / 5.0))

    dt = _congress_date_series_for_recency(sub)
    latest = dt.max() if not dt.empty else None
    recency = _recency_factor_from_latest(latest) if latest is not None and pd.notna(latest) else 0.35

    components = {
        "purchase_share": purchase_share,
        "large_share": large_share,
        "high_profile": high_profile,
        "repetition": repetition,
        "multi_rep": multi_rep,
        "recency": recency,
    }
    score = (
        _CONV_W_PURCHASE_SHARE * purchase_share
        + _CONV_W_LARGE_SHARE * large_share
        + _CONV_W_HIGH_PROFILE * high_profile
        + _CONV_W_REPETITION * repetition
        + _CONV_W_MULTI_REP * multi_rep
        + _CONV_W_RECENCY * recency
    )
    score = min(1.0, max(0.0, score))

    # Weighted contributions — pick the strongest drivers for a readable sentence
    contrib = [
        (_CONV_W_PURCHASE_SHARE * purchase_share, "purchase-titled activity outweighs sales in this pull"),
        (_CONV_W_LARGE_SHARE * large_share, "several lines include large-dollar Range bands"),
        (_CONV_W_HIGH_PROFILE * high_profile, "high-attention members show up in the filings"),
        (_CONV_W_REPETITION * repetition, "the symbol appears often in the disclosure set"),
        (_CONV_W_MULTI_REP * multi_rep, "multiple distinct representatives filed the name"),
        (_CONV_W_RECENCY * recency, "filings are relatively recent"),
    ]
    contrib.sort(key=lambda x: -x[0])
    strong = [msg for w, msg in contrib if w >= 0.06]
    if not strong:
        strong = [msg for w, msg in contrib if w > 0][:2]
    if not strong:
        sentence = "Conviction is modest here—few of the rule-based drivers fired strongly."
    else:
        sentence = "Congressional conviction is elevated because " + ", ".join(strong[:3]) + "."

    return score, components, sentence


def compute_congress_conviction_for_row(row: pd.Series) -> tuple[float, str]:
    """Single disclosure row (e.g. large-trade line): 0–1 score + short rationale."""
    tx = _congress_transaction_side_score(row.get("Transaction"))
    purchase = 1.0 if tx == 2 else (0.15 if tx == 0 else 0.55)
    prof = 1.0 if _high_profile_name_priority(str(row.get("Representative", ""))) < 900 else 0.0
    rec = _row_recency_factor(row)
    score = (
        _CONV_ROW_W_LARGE * 1.0
        + _CONV_ROW_W_PURCHASE * purchase
        + _CONV_ROW_W_PROFILE * prof
        + _CONV_ROW_W_RECENCY * rec
    )
    score = min(1.0, max(0.0, score))

    bits: list[str] = []
    if tx == 2:
        bits.append("a purchase-titled line")
    elif tx == 0:
        bits.append("a sale-titled line")
    if prof >= 1.0:
        bits.append("a high-profile member")
    if rec >= 0.65:
        bits.append("recent filing dates")
    bits.append("a large reported Range band")
    note = "Conviction on this row is driven by " + ", ".join(bits) + "."
    return score, note


def _explain_high_profile_trade(row: pd.Series) -> str:
    """Short Quiver-style note: name + dollar band + why readers care (not investment advice)."""
    rep = str(row.get("Representative", ""))
    ticker = str(row.get("Ticker", ""))
    txn = str(row.get("Transaction", ""))
    rng = str(row.get("Range", ""))
    mag = _range_magnitude_score(rng)
    side = _congress_transaction_side_score(txn)
    is_pelosi = "pelosi" in rep.lower()
    is_mccormick = "mccormick" in rep.lower()

    if is_pelosi and side == 2 and mag >= 500_000:
        return (
            "Pelosi filings with a wide **Range** and a **purchase** tag are the archetypal disclosure story: "
            "high name recognition plus a big reported band invites questions about timing and committee overlap—verify the filing, not the narrative."
        )
    if is_pelosi and side == 2:
        return (
            "A **purchase** from Pelosi stands out because retail and media attention concentrates on her book; "
            "use the **Range** to anchor the piece in hard numbers."
        )
    if is_mccormick and mag >= 100_000:
        return (
            "McCormick is on the short list of members whose trades get picked up nationally; "
            "pair the **Ticker** with the **Range** band so the conflict-of-interest frame stays factual."
        )
    if side == 2 and mag >= 1_000_000:
        return (
            "A **purchase** in a seven-figure **Range** band is rare in the feed—exactly the kind of row that fits a "
            "**name + dollar amount + conflict** headline if you cite the disclosure source."
        )
    if side == 2:
        return (
            f"**{rep}** is on the watch list for this MVP; a **purchase** in **{ticker}** is worth a line even when the "
            "**Range** is mid-sized because attention follows the name."
        )
    if side == 0:
        return (
            f"**Sales** from visible members still matter for completeness: readers compare them to earlier **purchases** "
            f"in **{ticker}**—good for a 'what changed' graf, not a bull/bear call."
        )
    return (
        f"**{rep}** draws above-average scrutiny; this **{ticker}** line plus the reported **Range** gives a concrete hook "
        "for a disclosure-driven paragraph without implying wrongdoing."
    )


def _high_profile_trade_distinctiveness_score(
    name_prio: int,
    purchase_score: int,
    range_mag: int,
    row_conviction_01: float,
) -> float:
    """
    Deterministic 0.0–1.0 score to **order** high-profile congressional trade cards by editorial rarity.

    Inputs are all already materialized in the high-profile pipeline (no extra API data):

    - **Name** (28%): lower ``name_prio`` (Pelosi-tier) scores higher.
    - **Purchase tilt** (22%): clear purchase > ambiguous > sale-titled.
    - **Range band** (22%): larger parsed ``Range`` magnitude → higher, capped at $500k-equivalent scale.
    - **Row conviction** (28%): ``compute_congress_conviction_for_row`` 0–1 score.

    Sum is capped at 1.0. Used only for sort order of the High-Profile Congressional Trades section.
    """
    name_part = max(0.0, min(1.0, (900.0 - float(min(int(name_prio), 899))) / 900.0))
    ps = int(purchase_score)
    if ps >= 2:
        purch_part = 1.0
    elif ps == 1:
        purch_part = 0.45
    else:
        purch_part = 0.15
    rm = int(range_mag)
    range_part = min(1.0, float(rm) / 500_000.0) if rm > 0 else 0.0
    conv_part = max(0.0, min(1.0, float(row_conviction_01)))
    return round(
        min(
            1.0,
            0.28 * name_part + 0.22 * purch_part + 0.22 * range_part + 0.28 * conv_part,
        ),
        4,
    )


def identify_high_profile_congress_trades(df: pd.DataFrame, *, max_rows: int = 12) -> pd.DataFrame:
    """
    Editorial short list: high-profile names only, then ranked primarily by **distinctiveness score**
    (notable name + purchase tilt + Range size + row conviction), then conviction, Range, purchase, name priority.

    Does not change numeric desk scores on ranked signals.
    """
    if df.empty or "Representative" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["_name_prio"] = work["Representative"].map(_high_profile_name_priority)
    hit = work["_name_prio"] < 900
    if not hit.any():
        return pd.DataFrame()

    subset = work.loc[hit].copy()
    subset["_purchase_score"] = subset["Transaction"].map(_congress_transaction_side_score) if "Transaction" in subset.columns else 1
    subset["_range_mag"] = subset["Range"].map(_range_magnitude_score) if "Range" in subset.columns else 0
    conv_pairs = subset.apply(lambda r: compute_congress_conviction_for_row(r), axis=1)
    subset["_row_conv"] = conv_pairs.map(lambda x: x[0]).astype(float)
    subset["_hp_distinctiveness"] = subset.apply(
        lambda r: _high_profile_trade_distinctiveness_score(
            int(r["_name_prio"]),
            int(r["_purchase_score"]),
            int(r["_range_mag"]),
            float(r["_row_conv"]),
        ),
        axis=1,
    )

    sort_cols = ["_hp_distinctiveness", "_row_conv", "_range_mag", "_purchase_score", "_name_prio"]
    ascending = [False, False, False, False, True]
    if "ReportDate" in subset.columns:
        subset = subset.assign(_rd=pd.to_datetime(subset["ReportDate"], errors="coerce"))
        sort_cols.append("_rd")
        ascending.append(False)

    subset = subset.sort_values(sort_cols, ascending=ascending, na_position="last")

    keep = [
        c
        for c in (
            "Representative",
            "Ticker",
            "Transaction",
            "Range",
            "ReportDate",
            "TransactionDate",
        )
        if c in subset.columns
    ]
    out = subset[keep].head(max_rows).copy()
    out["CongressConviction"] = (subset.loc[out.index, "_row_conv"] * 100).round().astype(int).values
    out["ConvictionNote"] = [compute_congress_conviction_for_row(row)[1] for _, row in out.iterrows()]
    out["Explanation"] = [_explain_high_profile_trade(row) for _, row in out.iterrows()]
    return out.reset_index(drop=True)


def _median_int_list(values: list[int]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    m = n // 2
    if n % 2 == 1:
        return float(s[m])
    return (s[m - 1] + s[m]) / 2.0


def tickers_with_high_profile_purchase(congress_df: pd.DataFrame) -> set[str]:
    """Tickers with at least one high-profile member row tagged as purchase this pull."""
    hp = identify_high_profile_congress_trades(congress_df, max_rows=48)
    if hp.empty or "Ticker" not in hp.columns:
        return set()
    out: set[str] = set()
    for _, row in hp.iterrows():
        if "Transaction" in hp.columns:
            if _congress_transaction_side_score(row.get("Transaction")) != 2:
                continue
        t = str(row.get("Ticker", "")).strip().upper()
        if t:
            out.add(t)
    return out


def _clear_insider_skew_for_dashboard(ibuy: int, isell: int) -> bool:
    """Same spirit as the dashboard insider-skew bullet: meaningful volume and a wide buy/sell gap."""
    if isell >= 5 and isell > ibuy:
        gap = isell - ibuy
        return gap >= max(4, isell // 5)
    if ibuy >= 5 and ibuy > isell:
        gap = ibuy - isell
        return gap >= max(4, ibuy // 5)
    return False


def count_dashboard_anomaly_conditions(
    row: dict[str, Any],
    *,
    median_congress_filings: float,
    overlap_tickers: set[str],
) -> int:
    """
    Count how many of the seven dashboard anomaly dimensions fire for this ranked row.

    Used for: (a) minimum 3-of-7 to appear in Key Insight / newsletter context, and
    (b) 4-of-7 (or exceptional) for hero cards.
    """
    ev = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    tk = str(row.get("ticker", "")).strip().upper()
    cf = int(ev.get("congress_filings") or 0)
    ur = int(ev.get("unique_representatives") or 0)
    ins = int(ev.get("insider_rows") or 0)
    ibuy = int(ev.get("insider_buys") or 0)
    isell = int(ev.get("insider_sells") or 0)

    n = 0
    if cf > median_congress_filings:
        n += 1
    if ur >= 4:
        n += 1
    if bool(ev.get("large_congress_trade")):
        n += 1
    if tk in overlap_tickers and ins >= 10:
        n += 1
    if _clear_insider_skew_for_dashboard(ibuy, isell):
        n += 1
    try:
        cr = float(ev.get("congress_recency_factor") or 0.0)
        ir = float(ev.get("insider_recency_factor") or 0.0)
    except (TypeError, ValueError):
        cr, ir = 0.0, 0.0
    if max(cr, ir) >= 0.65:
        n += 1
    c_sup = False
    try:
        cac = ev.get("contract_activity_count")
        if cac is not None and int(cac) > 0:
            c_sup = True
    except (TypeError, ValueError):
        pass
    if not c_sup:
        try:
            lac = ev.get("lobbying_activity_count")
            if lac is not None and int(lac) > 0:
                c_sup = True
        except (TypeError, ValueError):
            pass
    if c_sup:
        n += 1
    return n


def has_dashboard_exceptional_anomaly(
    row: dict[str, Any],
    *,
    high_profile_purchase_tickers: set[str],
    overlap_tickers: set[str],
) -> bool:
    """
    One-off hero bar: Pelosi / high-profile purchase ticker, large Capitol + insider overlap,
    or very broad reps with heavy insider rows.
    """
    ev = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    tk = str(row.get("ticker", "")).strip().upper()
    ins = int(ev.get("insider_rows") or 0)
    ur = int(ev.get("unique_representatives") or 0)
    if tk in high_profile_purchase_tickers:
        return True
    if bool(ev.get("large_congress_trade")) and ins >= 5 and tk in overlap_tickers:
        return True
    if ur >= 6 and ins >= 12:
        return True
    return False


# Widely recognized equity tickers (reader familiarity proxy for newsletter “story” fit—not a quality judgment).
STORY_RECOGNIZABLE_TICKERS: frozenset[str] = frozenset(
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
        "TSLA",
        "NFLX",
        "DIS",
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "C",
        "V",
        "MA",
        "BRK.B",
        "BRKB",
        "XOM",
        "CVX",
        "COP",
        "WMT",
        "HD",
        "MCD",
        "PG",
        "KO",
        "PEP",
        "UNH",
        "JNJ",
        "PFE",
        "MRK",
        "ABBV",
        "LLY",
        "TMO",
    }
)


def _story_peer_differentiator(row: dict[str, Any], peers: list[dict[str, Any]]) -> float:
    """
    1.0 if this symbol is editorially distinct **within the same run’s desk-ranked set** (not vs the whole market):
    sole (within tolerance) top desk score, rare large-dollar flag among peers, or unique max filings/insider depth.
    """
    if not peers:
        return 1.0
    self_sc = float(row.get("score") or 0.0)
    scores = [float(r.get("score") or 0.0) for r in peers if isinstance(r, dict)]
    if not scores:
        return 1.0
    mx_sc = max(scores)
    n_top = sum(1 for s in scores if s >= mx_sc - 0.05)
    if self_sc >= mx_sc - 0.05 and n_top == 1:
        return 1.0

    ev = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    large = bool(ev.get("large_congress_trade"))
    n_large = sum(
        1
        for r in peers
        if isinstance(r, dict) and bool((r.get("evidence") or {}).get("large_congress_trade"))
    )
    ln = len(peers)
    if large and n_large <= max(1, (ln + 2) // 3):
        return 1.0

    cf = int(ev.get("congress_filings") or 0)
    ins = int(ev.get("insider_rows") or 0)
    cfs = [int((r.get("evidence") or {}).get("congress_filings") or 0) for r in peers if isinstance(r, dict)]
    inss = [int((r.get("evidence") or {}).get("insider_rows") or 0) for r in peers if isinstance(r, dict)]
    if cfs and cf >= max(cfs) and cfs.count(cf) == 1:
        return 1.0
    if inss and ins >= max(inss) and inss.count(ins) == 1:
        return 1.0
    return 0.0


def compute_story_worthiness_score(
    row: dict[str, Any],
    *,
    congress_df: pd.DataFrame,
    overlap_tickers: set[str],
    ranked_peers: list[dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    """
    **Story-Worthiness Score (0.0–5.0)** — editorial / newsletter fit **separate** from the desk anomaly score.

    Desk score stays the technical composite; this layer answers “is this a compelling *story* for readers?”
    A name can be a **strong signal** but a **weak story** (e.g. obscure ticker, muddy direction), or the reverse.

    Six equally weighted binary inputs (each 0 or 1), scaled to 0–5:
    - ``high_profile_name``: high-profile congressional name on this ticker’s Capitol rows.
    - ``high_dollar_trade``: ``large_congress_trade`` in evidence.
    - ``clean_directional_narrative``: desk direction Bullish or Bearish (joint congress+insider parse rules).
    - ``cross_dataset_support``: overlap + filing depth, **or** contracts/lobbying counts on evidence.
    - ``recognizable_ticker``: symbol in ``STORY_RECOGNIZABLE_TICKERS`` (headline-name proxy).
    - ``peer_differentiator``: stands out vs other **ranked** names this run (see ``_story_peer_differentiator``).

    Used only for ordering / copy context on Top Anomalies, Key Insight, and Newsletter—**not** for reranking the
    main leaderboard table.
    """
    ev = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    tk = str(row.get("ticker", "")).strip().upper()
    direction = str(row.get("direction", "Mixed")).strip()

    hp = 1.0 if (tk and _ticker_has_high_profile_congress_rep(congress_df, tk)) else 0.0
    hd = 1.0 if bool(ev.get("large_congress_trade")) else 0.0
    nar = 1.0 if direction in ("Bullish", "Bearish") else 0.0

    cross = 0.0
    if tk in overlap_tickers:
        if int(ev.get("insider_rows") or 0) >= 3 and int(ev.get("congress_filings") or 0) >= 3:
            cross = 1.0
    try:
        if int(ev.get("contract_activity_count") or 0) > 0 or int(ev.get("lobbying_activity_count") or 0) > 0:
            cross = 1.0
    except (TypeError, ValueError):
        pass

    rec = 1.0 if tk in STORY_RECOGNIZABLE_TICKERS else 0.0
    diff = _story_peer_differentiator(row, ranked_peers)

    bits = {
        "high_profile_name": hp,
        "high_dollar_trade": hd,
        "clean_directional_narrative": nar,
        "cross_dataset_support": cross,
        "recognizable_ticker": rec,
        "peer_differentiator": diff,
    }
    total = min(5.0, 5.0 * sum(bits.values()) / 6.0)
    return round(total, 1), {k: round(v, 4) for k, v in bits.items()}


def attach_story_worthiness_to_ranked(
    ranked_signals: list[dict[str, Any]],
    congress_df: pd.DataFrame,
    overlap: list[Any],
) -> None:
    """Mutates each ranked row: ``story_worthiness_score`` (0–5) and ``story_worthiness_components``."""
    cdf = congress_df if isinstance(congress_df, pd.DataFrame) else pd.DataFrame()
    overlap_tickers = {str(x).strip().upper() for x in overlap if str(x).strip()}
    peers = [r for r in ranked_signals if isinstance(r, dict) and str(r.get("ticker", "")).strip()]
    for row in ranked_signals:
        if not isinstance(row, dict):
            continue
        if not str(row.get("ticker", "")).strip():
            continue
        sw, comp = compute_story_worthiness_score(
            row,
            congress_df=cdf,
            overlap_tickers=overlap_tickers,
            ranked_peers=peers,
        )
        row["story_worthiness_score"] = sw
        row["story_worthiness_components"] = comp


def compute_dashboard_anomaly_views(
    ranked_signals: list[dict[str, Any]],
    congress_df: pd.DataFrame,
    overlap: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split desk-ranked rows for dashboard copy only (does not change scoring or raw pulls).

    - ``qualified``: rows with ≥3 of seven anomaly conditions, **re-sorted** by
      ``story_worthiness_score`` (desc), then ``distinctiveness_bonus``, then desk ``score``, then conviction.
    - ``hero``: up to three of those (in that sort order) that also have ≥4 conditions **or** an exceptional hook.

    **Note:** desk strength and “story” diverge by design—high desk / low story-worthiness (and vice versa) is expected.

    Requires ``distinctiveness_bonus`` and ``story_worthiness_score`` on each row.
    """
    ranked = [r for r in ranked_signals if isinstance(r, dict) and str(r.get("ticker", "")).strip()]
    if not ranked:
        return [], []

    filings: list[int] = []
    for r in ranked:
        ev = r.get("evidence")
        if isinstance(ev, dict):
            filings.append(int(ev.get("congress_filings") or 0))
        else:
            filings.append(0)
    med_cf = _median_int_list(filings)

    overlap_tickers = {str(x).strip().upper() for x in overlap if str(x).strip()}
    hp_purchase = tickers_with_high_profile_purchase(congress_df)

    scored: list[tuple[dict[str, Any], int, bool]] = []
    for row in ranked:
        n = count_dashboard_anomaly_conditions(
            row,
            median_congress_filings=med_cf,
            overlap_tickers=overlap_tickers,
        )
        if n < 3:
            continue
        ex = has_dashboard_exceptional_anomaly(
            row,
            high_profile_purchase_tickers=hp_purchase,
            overlap_tickers=overlap_tickers,
        )
        scored.append((row, n, ex))

    def _sort_key(item: tuple[dict[str, Any], int, bool]) -> tuple[float, float, float, float, str]:
        row, _n, _ex = item
        try:
            sw = float(row.get("story_worthiness_score") or 0.0)
        except (TypeError, ValueError):
            sw = 0.0
        try:
            db = float(row.get("distinctiveness_bonus") or 0.0)
        except (TypeError, ValueError):
            db = 0.0
        try:
            sc = float(row.get("score") or 0.0)
        except (TypeError, ValueError):
            sc = 0.0
        conv = row.get("congress_conviction_0_100")
        try:
            conv_f = float(conv) / 100.0 if conv is not None else 0.0
        except (TypeError, ValueError):
            conv_f = 0.0
        tk = str(row.get("ticker", "")).strip().upper()
        return (-sw, -db, -sc, -conv_f, tk)

    scored.sort(key=_sort_key)

    qualified = [t[0] for t in scored]
    hero: list[dict[str, Any]] = []
    for row, n, ex in scored:
        if len(hero) >= 3:
            break
        if n >= 4 or ex:
            hero.append(row)
    return qualified, hero


def get_high_profile_congress_trades(df: pd.DataFrame, *, max_rows: int = 12) -> pd.DataFrame:
    """Backward-compatible alias for the report pipeline."""
    return identify_high_profile_congress_trades(df, max_rows=max_rows)


def format_high_profile_markdown(hp: pd.DataFrame) -> str:
    """Markdown items for ## High-Profile Congressional Trades (structured bullets)."""
    if hp.empty:
        return "_No trades matched the high-profile name list in this pull._\n"

    lines: list[str] = []
    for i, (_, row) in enumerate(hp.iterrows(), start=1):
        lines.append(f"### {i}.")
        lines.append("")
        lines.append(f"- Representative: {_md_escape_profile(row.get('Representative', ''))}")
        lines.append(f"- Ticker: {_md_escape_profile(row.get('Ticker', ''))}")
        lines.append(f"- Transaction: {_md_escape_profile(row.get('Transaction', ''))}")
        lines.append(f"- Range: {_md_escape_profile(row.get('Range', ''))}")
        if "ReportDate" in hp.columns:
            lines.append(f"- ReportDate: {_md_escape_profile(row.get('ReportDate', ''))}")
        if "TransactionDate" in hp.columns:
            lines.append(f"- TransactionDate: {_md_escape_profile(row.get('TransactionDate', ''))}")
        if "CongressConviction" in hp.columns:
            lines.append(f"- Congressional conviction (trade-level): {_md_escape_profile(row.get('CongressConviction', ''))}/100")
        if "ConvictionNote" in hp.columns:
            lines.append(f"- Why conviction on this line: {_md_escape_profile(row.get('ConvictionNote', ''))}")
        expl = row.get("Explanation", "")
        lines.append(f"- Why it matters: {_md_escape_profile(expl)}")
        lines.append("")
    return "\n".join(lines)
