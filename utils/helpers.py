"""Helper utilities used by services and API modules."""
from __future__ import annotations

import itertools
import json
import os
from datetime import datetime
from typing import Iterable, List, Optional

from models.trip import Airport, AirportsData

ROUTES_URL = "https://raw.githubusercontent.com/Jonty/airline-route-data/master/airline_routes.json"
DATA_DIR = "data"
ROUTES_FILE = os.path.join(DATA_DIR, "airline_routes.json")


def ensure_data_file() -> None:
    """Ensure airline route data is present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ROUTES_FILE):
        import requests
        r = requests.get(ROUTES_URL, timeout=10)
        r.raise_for_status()
        with open(ROUTES_FILE, "w") as f:
            f.write(r.text)


def load_airports(filepath: str) -> AirportsData:
    """Load airport data from disk."""
    with open(filepath) as f:
        raw = json.load(f)
    return {iata: Airport(**data) for iata, data in raw.items()}


ensure_data_file()
ROUTES_DATA: AirportsData = load_airports(ROUTES_FILE)


def chunked(iterable: Iterable, size: int) -> Iterable[List]:
    """Yield chunks from an iterable."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def get_routes_from_airport(origin: str, limit: Optional[int] = 5) -> List[str]:
    airport_info = ROUTES_DATA.get(origin)
    if not airport_info:
        return []
    destinations = list({r.iata for r in airport_info.routes})
    return destinations[:limit] if limit else destinations


def get_city_for_airport(iata: str) -> str:
    return ROUTES_DATA.get(iata, Airport(city_name=iata, iata=iata)).city_name


def parse_price(price: str) -> Optional[float]:
    if not price or not isinstance(price, str):
        return None
    if "unavailable" in price.lower():
        return None
    try:
        value = float(price.replace("$", "").replace(",", "").strip())
        return value if value > 0 else None
    except ValueError:
        return None


def parse_custom_date(date_str: str) -> str:
    try:
        return (
            datetime.strptime(date_str, "%I:%M %p on %a, %b %d")
            .replace(year=datetime.now().year)
            .strftime("%Y-%m-%d")
        )
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")
