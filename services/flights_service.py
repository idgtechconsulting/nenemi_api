"""Flight search logic using fast_flights library."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from fast_flights import FlightData, Passengers, Result, get_flights

from utils.caching import get_cache, set_cache
from utils.helpers import parse_price, parse_custom_date

MAX_CONCURRENT_FLIGHT_CALLS = 20

executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FLIGHT_CALLS)
semaphore = asyncio.Semaphore(MAX_CONCURRENT_FLIGHT_CALLS)


async def async_get_cheapest_flight(
    origin: str,
    destination: str,
    date: str,
    trip_type: str = "one-way",
    adults: int = 1,
) -> Optional[Dict[str, Any]]:
    """Return cheapest flight details using cached results when possible."""
    cache_key = f"flight:{trip_type}:{origin}:{destination}:{date}:{adults}"
    cached = get_cache(cache_key)
    if cached:
        return cached

    async with semaphore:
        loop = asyncio.get_running_loop()
        try:
            result: Result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: get_flights(
                        flight_data=[FlightData(date=date, from_airport=origin, to_airport=destination)],
                        trip=trip_type,
                        seat="economy",
                        passengers=Passengers(adults=adults),
                        fetch_mode="fallback",
                    ),
                ),
                timeout=25,
            )
            flights = result.flights or []
            valid = [f for f in flights if parse_price(f.price)]
            if not valid:
                return None
            cheapest = sorted(valid, key=lambda x: parse_price(x.price))[0]
            flight_data = {
                "origin": origin,
                "destination": destination,
                "price": parse_price(cheapest.price),
                "currency": "USD",
                "airline": cheapest.name,
                "flight_number": getattr(cheapest, "flight_number", None),
                "departure": parse_custom_date(cheapest.departure),
                "return": parse_custom_date(getattr(cheapest, "return_date", "")) if trip_type == "round-trip" else None,
                "duration": cheapest.duration,
                "stops": cheapest.stops,
            }
            set_cache(cache_key, flight_data)
            return flight_data
        except Exception as e:
            print(f"Error fetching {trip_type} flight {origin}->{destination}: {e}")
    return None
