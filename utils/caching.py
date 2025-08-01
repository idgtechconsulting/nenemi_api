"""Simple JSON file based caching utilities."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Tuple
from pydantic import HttpUrl

CACHE_FILE = os.path.join("data", "cache.json")
CACHE_TTL = 86400

cache: Dict[str, Tuple[float, Any]] = {}


def serialize_for_cache(value: Any) -> Any:
    if isinstance(value, list):
        return [serialize_for_cache(v) for v in value]
    if hasattr(value, "dict"):
        d = value.dict(exclude_none=True)
        for k, v in d.items():
            if isinstance(v, list):
                d[k] = [str(x) if isinstance(x, HttpUrl) else x for x in v]
            elif isinstance(v, HttpUrl):
                d[k] = str(v)
        return d
    return value


def save_cache() -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({k: (t, serialize_for_cache(v)) for k, (t, v) in cache.items()}, f)
    except Exception as e:
        print(f"Error saving cache: {e}")


def load_cache() -> None:
    global cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                raw = json.load(f)
                cache = {k: (t, v) for k, (t, v) in raw.items()}
        except Exception as e:
            print(f"Error loading cache: {e}")


def get_cache(key: str) -> Any:
    entry = cache.get(key)
    if entry and time.time() - entry[0] < CACHE_TTL:
        return entry[1]
    return None


def set_cache(key: str, value: Any) -> None:
    cache[key] = (time.time(), serialize_for_cache(value))
    save_cache()


load_cache()


def get_ai_cache(key: str) -> Any:
    return get_cache(f"ai:{key}")


def set_ai_cache(key: str, value: Any) -> None:
    set_cache(f"ai:{key}", value)
