# QQ — Congressional & Insider Research Dashboard

A Python pipeline that pulls **live** U.S. congressional stock disclosures and corporate insider data from the [Quiver Quantitative API](https://api.quiverquant.com/), scores cross-dataset signals, and produces a **Markdown research brief** plus a **standalone HTML dashboard** suitable for a financial newsletter workflow.

Signal rules and scores are **deterministic** (no ML in the core engine). Optional **Anthropic Claude** calls can refine dashboard copy; responses are cached on disk.

---

## What you get

| Output | Description |
|--------|-------------|
| `output/weekly_research_brief.md` | Narrative brief with tables and structured sections |
| `output/research_dashboard.html` | Full interactive dashboard (open in a browser) |
| `output/charts/` | Matplotlib PNGs when charts are generated |
| `output/claude_run_cache.json` | Deduped Claude responses (if AI is used) |

The dashboard includes ranked “desk” anomalies, overlap signals, high-profile trades, optional government-contract and lobbying reinforcement, **Key Insight**, **Newsletter-ready** copy, and an **Ask AI** panel when served through the local backend.

---

## Requirements

- **Python 3.10+**
- **Quiver API token** (required for data pulls)
- **Anthropic API key** (optional — for Claude-enhanced dashboard text and `/ask-ai`)

---

## Setup

```bash
pip install -r requirements.txt
```

Set your Quiver token **only in the environment** (do not commit it):

| Shell | Command |
|--------|---------|
| PowerShell | `$env:QUIVER_API_TOKEN='your_token'` |
| bash / zsh | `export QUIVER_API_TOKEN='your_token'` |

Optional — Claude features:

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Enables AI copy on the dashboard and the backend `/ask-ai` route |
| `ANTHROPIC_MODEL` | Override default model (see `claude_client.py`) |
| `QQ_DISABLE_CLAUDE_CACHE=1` | Skip reading/writing `output/claude_run_cache.json` |

---

## Run the pipeline

```bash
python main.py
```

This fetches congress and insider feeds, may query supporting endpoints (contracts, lobbying, off-exchange) when ranked signals exist, builds the signal bundle, and writes:

- `output/weekly_research_brief.md`
- `output/research_dashboard.html`

Open the HTML file directly, or use the Flask backend (below) for same-origin **Ask AI**.

### Insider-only fallback

If congressional data is empty but insider data exists, the run switches to **insider-only fallback mode**: desk-ranked scores are omitted and the dashboard highlights an insider watchlist instead.

---

## Local backend (optional)

`backend.py` serves the generated dashboard and proxies **POST `/ask-ai`** to Anthropic (see `claude_client.py`).

```bash
set ANTHROPIC_API_KEY=your_key
python backend.py
```

Default URL: `http://127.0.0.1:5000/dashboard` — override the port with `QQ_BACKEND_PORT`.

The backend expects `output/research_dashboard.html` to exist (run `main.py` first).

---

## Project layout

| File / directory | Role |
|------------------|------|
| `main.py` | Orchestrates API fetch, signal build, contract/lobbying merge, report generation |
| `quiver_api.py` | HTTP calls to Quiver endpoints |
| `signal_logic.py` | Scoring, overlap, ranked signals, distinctiveness, **Story-Worthiness** (newsletter ordering layer), dashboard anomaly views |
| `report_generator.py` | Markdown + HTML dashboard, charts, optional Claude wiring |
| `claude_client.py` | Anthropic Messages API + dashboard context cache |
| `claude_proxy.py` | Related helper for Claude usage |
| `backend.py` | Flask: static dashboard + `/ask-ai` |
| `test_insider_only.py` | Insider-only path tests |
| `signal_confirmation_map.html` | Standalone Chart.js demo (scatter) |
| `congress_insider_alignment_gauge.html` | Standalone Chart.js demo (alignment bars) |

---

## Scoring model (summary)

- **Desk score (0–10)** — composite from congressional frequency, breadth, large-dollar trades, insider overlap, bias, recency, penalties, etc. Leaderboard order in the main ranked table follows this score.
- **Story-Worthiness (0–5)** — separate editorial layer (high-profile names, dollar size, direction, cross-dataset support, recognizable tickers, differentiation within the run). Used to order **qualified** highlights, **Top Anomalies** hero cards, **Key Insight**, and **Newsletter-ready** copy — not to replace the desk score.
- **Strong desk ≠ strong story** (and vice versa) by design.

---

## Disclaimer

Outputs are **research aids**, not investment advice. Congressional and insider data are incomplete by nature; verify filings and do your own diligence before acting.
