"""
Quiver Quant API — live congress and insider feeds.

Base: https://api.quiverquant.com/beta
Auth: Authorization: Token <API_KEY>
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.quiverquant.com/beta"

# Full URLs (same paths the functions use — handy for debugging from main.py)
CONGRESS_TRADING_URL = f"{BASE_URL}/live/congresstrading"
# Exact URL matching the simple Colab request
INSIDER_TRADING_URL = "https://api.quiverquant.com/beta/live/insiders"
GOVERNMENT_CONTRACTS_URL = f"{BASE_URL}/live/govcontracts"
LOBBYING_URL = f"{BASE_URL}/live/lobbying"
OFF_EXCHANGE_URL = f"{BASE_URL}/live/offexchange"
PATENTS_URL = f"{BASE_URL}/live/patents"

# Stop waiting after this many seconds (network or slow server)
REQUEST_TIMEOUT = 30


def _fetch_live_beta_json_df(api_key: str, url: str, label: str) -> pd.DataFrame:
    """
    GET a Quiver beta live JSON endpoint (list → DataFrame). Same diagnostics as other live pulls.

    Never raises; returns empty DataFrame on failure.
    """
    print(f"{label}: requesting {url}")

    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Token {api_key}"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        print(f"{label}: request timed out after {REQUEST_TIMEOUT} seconds.")
        return pd.DataFrame()
    except requests.RequestException as e:
        print(f"{label}: network error — {e}")
        return pd.DataFrame()

    print(f"{label}: HTTP status {response.status_code}")

    if response.status_code != 200:
        sc = response.status_code
        if sc == 401:
            print(f"{label}: {_explain_http_status(401)}")
        elif sc == 403:
            print(f"{label}: {_explain_http_status(403)}")
        elif sc == 404:
            print(f"{label}: {_explain_http_status(404)}")
        elif sc == 500:
            print(f"{label}: {_explain_http_status(500)}")
        else:
            print(f"{label}: {_explain_http_status(sc)}")
        preview = response.text[:200] if response.text else ""
        print(f"Response preview (first 200 chars): {preview}")
        return pd.DataFrame()

    body = response.text
    if not body or not body.strip():
        print(f"{label}: empty response body.")
        return pd.DataFrame()

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        print(f"{label}: JSON parsing error: {e}")
        return pd.DataFrame()

    return pd.DataFrame(data)


def _headers(api_key: str) -> dict[str, str]:
    return {
        "accept": "application/json",
        "Authorization": f"Token {api_key}",
    }


def _insider_headers(api_key: str) -> dict[str, str]:
    """Minimal headers like a simple Colab call: Authorization plus optional Accept."""
    return {
        "Authorization": f"Token {api_key}",
        "accept": "application/json",
    }


def _explain_http_status(status_code: int) -> str:
    """Short, beginner-friendly text for common status codes."""
    messages = {
        401: "Unauthorized — check that your API key is correct and active.",
        403: "Forbidden — your key may not have access to this dataset.",
        404: "Not found — endpoint path may have changed; verify Quiver's API docs.",
        500: "Internal server error — problem on Quiver's side; try again later.",
    }
    return messages.get(status_code, f"Unexpected HTTP status ({status_code}).")


def _print_non_200(dataset_label: str, response: requests.Response) -> None:
    """Beginner-friendly status line, then a short body preview for debugging."""
    sc = response.status_code
    if sc == 401:
        print(f"{dataset_label}: {_explain_http_status(401)}")
    elif sc == 403:
        print(f"{dataset_label}: {_explain_http_status(403)}")
    elif sc == 404:
        print(f"{dataset_label}: {_explain_http_status(404)}")
    elif sc == 500:
        print(f"{dataset_label}: {_explain_http_status(500)}")
    else:
        print(f"{dataset_label}: {_explain_http_status(sc)}")
    preview = response.text[:300] if response.text else ""
    print(f"Response preview: {preview}")


def _parse_body_to_dataframe(response: requests.Response, dataset_label: str) -> pd.DataFrame:
    """
    Parse JSON body into a DataFrame.

    If the body is not usable JSON or not a list/dict we can turn into rows,
    print a clear message and return an empty DataFrame (no crash).
    """
    text = response.text.strip()
    if not text:
        print(f"{dataset_label}: empty response body.")
        return pd.DataFrame()

    # Valid JSON can be a quoted string (e.g. an error message from the API)
    if text.startswith('"') and text.endswith('"'):
        try:
            msg = json.loads(text)
            print(f"{dataset_label}: API returned a message instead of table data: {msg}")
        except json.JSONDecodeError as e:
            print(f"{dataset_label}: JSON parsing error: {e}")
        return pd.DataFrame()

    try:
        data: Any = json.loads(response.content)
    except json.JSONDecodeError as e:
        print(f"{dataset_label}: JSON parsing error: {e}")
        return pd.DataFrame()

    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    print(f"{dataset_label}: JSON was not a list or object we can load as rows.")
    return pd.DataFrame()


def _fetch_live_table(url: str, api_key: str, dataset_label: str) -> pd.DataFrame:
    """
    GET one live endpoint: print URL and status, handle errors, return DataFrame or empty.

    Never raises — invalid responses become an empty DataFrame.
    """
    print(f"{dataset_label}: requesting {url}")

    try:
        response = requests.get(
            url,
            headers=_headers(api_key),
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        print(f"{dataset_label}: request timed out after {REQUEST_TIMEOUT} seconds.")
        return pd.DataFrame()
    except requests.RequestException as e:
        print(f"{dataset_label}: network error — {e}")
        return pd.DataFrame()

    print(f"{dataset_label}: HTTP status {response.status_code}")

    if response.status_code != 200:
        _print_non_200(dataset_label, response)
        return pd.DataFrame()

    df = _parse_body_to_dataframe(response, dataset_label)
    return df


def get_congress_trading(api_key: str) -> pd.DataFrame:
    """
    GET /live/congresstrading — U.S. Congress stock trading disclosures.

    On any problem, prints what went wrong and returns an empty DataFrame.
    """
    df = _fetch_live_table(CONGRESS_TRADING_URL, api_key, "Congress trading")

    for col in ("ReportDate", "TransactionDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def get_insider_trading(api_key: str) -> pd.DataFrame:
    """
    GET /live/insiders — same pattern as a minimal Colab request (URL + Token + optional Accept).

    Uses response.json() and pd.DataFrame(data). No query params. No extra filtering here.
    """
    url = INSIDER_TRADING_URL
    label = "Insider trading"
    print(f"{label}: requesting {url}")

    try:
        response = requests.get(
            url,
            headers=_insider_headers(api_key),
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        print(f"{label}: request timed out after {REQUEST_TIMEOUT} seconds.")
        return pd.DataFrame()
    except requests.RequestException as e:
        print(f"{label}: network error — {e}")
        return pd.DataFrame()

    print(f"{label}: HTTP status {response.status_code}")

    if response.status_code != 200:
        _print_non_200(label, response)
        return pd.DataFrame()

    body = response.text
    if not body or not body.strip():
        print(f"{label}: empty response body.")
        return pd.DataFrame()

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        print(f"{label}: JSON parsing error: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    for col in df.columns:
        if "date" in col.lower() and df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def fetch_government_contracts(api_key: str) -> pd.DataFrame:
    """
    GET /live/govcontracts — federal contract activity (supporting context only in our pipeline).

    Used only to reinforce tickers that already appear in top-ranked or overlap signals.
    Never raises; returns an empty DataFrame on any failure.
    """
    url = GOVERNMENT_CONTRACTS_URL
    label = "Government contracts"
    print(f"{label}: requesting {url}")

    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Token {api_key}"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        print(f"{label}: request timed out after {REQUEST_TIMEOUT} seconds.")
        return pd.DataFrame()
    except requests.RequestException as e:
        print(f"{label}: network error — {e}")
        return pd.DataFrame()

    print(f"{label}: HTTP status {response.status_code}")

    if response.status_code != 200:
        sc = response.status_code
        if sc == 401:
            print(f"{label}: {_explain_http_status(401)}")
        elif sc == 403:
            print(f"{label}: {_explain_http_status(403)}")
        elif sc == 404:
            print(f"{label}: {_explain_http_status(404)}")
        elif sc == 500:
            print(f"{label}: {_explain_http_status(500)}")
        else:
            print(f"{label}: {_explain_http_status(sc)}")
        preview = response.text[:200] if response.text else ""
        print(f"Response preview (first 200 chars): {preview}")
        return pd.DataFrame()

    body = response.text
    if not body or not body.strip():
        print(f"{label}: empty response body.")
        return pd.DataFrame()

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        print(f"{label}: JSON parsing error: {e}")
        return pd.DataFrame()

    return pd.DataFrame(data)


def fetch_lobbying_data(api_key: str) -> pd.DataFrame:
    """
    GET /live/lobbying — lobbying disclosures (supporting context only in our pipeline).

    Reinforces tickers already flagged in top-ranked or overlap signals; never a standalone dump here.
    Never raises; returns an empty DataFrame on any failure.
    """
    df = _fetch_live_beta_json_df(api_key, LOBBYING_URL, "Lobbying")
    if df.empty:
        return df
    for col in df.columns:
        if "date" in col.lower() and df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def fetch_off_exchange_data(api_key: str) -> pd.DataFrame:
    """
    GET /live/offexchange — for data-coverage reporting only in this MVP (not merged into signals).

    Never raises; returns an empty DataFrame on any failure.
    """
    return _fetch_live_beta_json_df(api_key, OFF_EXCHANGE_URL, "Off-exchange")


def fetch_patents_data(api_key: str) -> pd.DataFrame:
    """
    Patents live endpoint is disabled: `/live/patents` is not confirmed (404 in the wild).

    Does not perform HTTP; returns an empty DataFrame so callers stay stable without log noise.
    """
    _ = api_key  # token unused until the endpoint is confirmed
    return pd.DataFrame()
