"""Airbnb listing search using HasData API."""
from __future__ import annotations

import asyncio
from typing import List

import requests

from models.trip import AirbnbProperty, AirbnbResponse
from utils.caching import get_cache, set_cache

HASDATA_API_KEY = "569634eb-f56e-4a80-bc90-19d9d35cf9b0"
HASDATA_BASE_URL = "https://api.hasdata.com/scrape/airbnb/listing"
MAX_CONCURRENT_AIRBNB_CALLS = 15

airbnb_semaphore = asyncio.Semaphore(MAX_CONCURRENT_AIRBNB_CALLS)


async def async_get_airbnbs(
    city: str,
    check_in: str,
    check_out: str,
    guests: int = 1,
    limit: int = 10,
) -> List[AirbnbProperty]:
    """Return sorted Airbnb listings for the city and dates."""
    cache_key = f"airbnb:{city}:{check_in}:{check_out}:{guests}:{limit}"
    cached = get_cache(cache_key)
    if cached:
        return [AirbnbProperty(**p) for p in cached]

    async with airbnb_semaphore:
        headers = {"x-api-key": HASDATA_API_KEY, "Content-Type": "application/json"}
        params = {
            "location": city,
            "checkIn": check_in,
            "checkOut": check_out,
            "adults": guests,
            "limit": limit,
        }
        try:
            resp = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: requests.get(HASDATA_BASE_URL, headers=headers, params=params)
                ),
                timeout=25,
            )
            if resp.status_code == 200:
                parsed = AirbnbResponse(**resp.json())
                filtered = [p for p in parsed.properties if p.price.as_float()]
                sorted_props = sorted(filtered, key=lambda x: x.price.as_float())
                set_cache(cache_key, sorted_props)
                return sorted_props
            else:
                print(f"HasData error: {resp.text}")
        except Exception as e:
            print(f"Error fetching Airbnbs: {e}")
    return []
