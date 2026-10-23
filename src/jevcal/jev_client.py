"""Cached, retrying client for Jev via OpenRouter's Decisions API.

Every request is cached on disk (SQLite), keyed by a SHA-256 hash of the exact
request body. Re-running an experiment therefore never re-calls the API, and the
raw responses stay available for the paper's reproducibility archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import requests

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"
RETRY_STATUS = {429, 500, 502, 503, 504}


class JevError(RuntimeError):
    pass


@dataclass
class JevAnswer:
    predicted_label: str
    class_order: list[str]
    probabilities: list[float]          # raw API values, in class_order
    confidence: float | None
    model_snapshot: str | None          # e.g. "typesafe/jev-1.13-20260917"
    request_id: str | None
    request_hash: str
    usage: dict = field(default_factory=dict)
    from_cache: bool = False
    latency_ms: float | None = None


def request_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_choice_payload(text: str, criteria: dict[str, str | None], instructions: str,
                         model: str = DEFAULT_MODEL, question: str = "label") -> dict:
    return {
        "model": model,
        "state": {"text": text},
        "questions": {question: {"type": "choice", "instructions": instructions, "criteria": criteria}},
    }


def parse_choice(raw: dict, question: str, class_order: list[str], key: str) -> JevAnswer:
    try:
        ans = raw["answers"][question]
        probs_map = ans["probabilities"]
    except (KeyError, TypeError) as e:
        raise JevError(f"Unexpected response shape: {json.dumps(raw)[:500]}") from e
    unknown = set(probs_map) - set(class_order)
    if unknown:
        raise JevError(f"Response has labels not in class_order: {sorted(unknown)[:5]}")
    return JevAnswer(
        predicted_label=ans.get("choice"),
        class_order=list(class_order),
        probabilities=[float(probs_map.get(c, 0.0)) for c in class_order],
        confidence=ans.get("confidence"),
        model_snapshot=raw.get("model"),
        request_id=raw.get("id"),
        request_hash=key,
        usage=raw.get("usage") or {},
    )


class JevClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 cache_path: str | Path = "cache/jev_cache.sqlite",
                 max_retries: int = 6, timeout: float = 60.0):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS responses (hash TEXT PRIMARY KEY, request TEXT, "
                "response TEXT, created_at REAL, latency_ms REAL)")
            self._db.commit()
        self._session = requests.Session()

    # ---- cache -----------------------------------------------------------------
    def _get(self, key: str):
        with self._lock:
            row = self._db.execute("SELECT response, latency_ms FROM responses WHERE hash=?", (key,)).fetchone()
        return (json.loads(row[0]), row[1]) if row else None

    def _put(self, key: str, payload: dict, raw: dict, latency_ms: float):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?)",
                             (key, json.dumps(payload, ensure_ascii=False), json.dumps(raw), time.time(), latency_ms))
            self._db.commit()

    # ---- network ---------------------------------------------------------------
    def _post(self, payload: dict) -> tuple[dict, float]:
        if not self.api_key:
            raise JevError("OPENROUTER_API_KEY is not set (put it in .env).")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._session.post(ENDPOINT, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last = str(e)
            else:
                latency = (time.perf_counter() - t0) * 1000
                if r.status_code == 200:
                    return r.json(), latency
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in RETRY_STATUS:
                    raise JevError(last)
            if attempt < self.max_retries:
                time.sleep(min(60, 2 ** attempt) + random.random())
        raise JevError(f"Gave up after {self.max_retries} retries. Last error: {last}")

    # ---- public ------------------------------------------------------------------
    def classify(self, text: str, criteria: dict[str, str | None], instructions: str,
                 question: str = "label") -> JevAnswer:
        """criteria: ordered {label: description or None}. Its key order is the class order."""
        payload = build_choice_payload(text, criteria, instructions, self.model, question)
        key = request_hash(payload)
        hit = self._get(key)
        if hit:
            raw, latency = hit
            ans = parse_choice(raw, question, list(criteria), key)
            ans.from_cache, ans.latency_ms = True, latency
            return ans
        raw, latency = self._post(payload)
        self._put(key, payload, raw, latency)
        ans = parse_choice(raw, question, list(criteria), key)
        ans.latency_ms = latency
        return ans

    def classify_many(self, texts: list[str], criteria: dict[str, str | None], instructions: str,
                      concurrency: int = 8, question: str = "label") -> list[JevAnswer | JevError]:
        """Order-preserving. Failed items come back as JevError so one bad call never kills a run."""
        def one(t):
            try:
                return self.classify(t, criteria, instructions, question)
            except JevError as e:
                return e
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(one, texts))
