from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


TEXT_COLUMN = "review_text"


class DataSourceError(RuntimeError):
    pass


@dataclass
class GooglePlaceResult:
    place_id: str
    name: str | None = None
    url: str | None = None


def get_google_api_key(explicit_key: str | None = None) -> str:
    key = (explicit_key or "").strip()
    if key:
        return key
    env_key = (os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_PLACES_API_KEY") or "").strip()
    if env_key:
        return env_key
    raise DataSourceError(
        "Google API key tidak ditemukan. Set `GOOGLE_MAPS_API_KEY` / `GOOGLE_PLACES_API_KEY` "
        "atau isi manual di UI."
    )


def _raise_for_status(payload: Dict[str, Any], context: str) -> None:
    status = payload.get("status")
    if status and status != "OK":
        message = payload.get("error_message") or payload.get("info_messages") or ""
        raise DataSourceError(f"{context}: status={status}. {message}".strip())


def google_find_place_id_legacy(api_key: str, query: str) -> GooglePlaceResult:
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": query,
        "inputtype": "textquery",
        "fields": "place_id,name",
        "key": api_key,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    _raise_for_status(payload, "Google Find Place")
    candidates = payload.get("candidates") or []
    if not candidates:
        raise DataSourceError("Google Find Place: tidak menemukan place_id untuk query tersebut.")
    top = candidates[0]
    return GooglePlaceResult(place_id=str(top.get("place_id")), name=top.get("name"))


def google_fetch_reviews_legacy(api_key: str, place_id: str, language: str = "id") -> pd.DataFrame:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,url,rating,reviews",
        "language": language,
        "key": api_key,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    _raise_for_status(payload, "Google Place Details")

    result = payload.get("result") or {}
    reviews: List[Dict[str, Any]] = result.get("reviews") or []
    rows: List[Dict[str, Any]] = []
    for review in reviews:
        text = (review.get("text") or "").strip()
        if not text:
            continue
        rows.append(
            {
                TEXT_COLUMN: text,
                "rating": review.get("rating"),
                "date": review.get("time"),  # unix epoch seconds (if present)
                "author_name": review.get("author_name"),
                "source": "google_places_api",
                "place_name": result.get("name"),
                "place_url": result.get("url"),
            }
        )

    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], unit="s", errors="coerce").dt.date.astype("string")
    return df


def load_reviews_from_public_csv_url(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
    except Exception as exc:  # noqa: BLE001
        raise DataSourceError(f"Gagal membaca CSV dari URL: {exc}") from exc
    return df

