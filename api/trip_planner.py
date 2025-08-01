"""FastAPI endpoints for trip planning."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from models.trip import TripRequest
from services.airbnb_service import async_get_airbnbs
from services.ai_service import (
    ai_generate_image,
    ai_generate_itinerary,
    ai_generate_tagline,
    ai_score_package,
)
from services.flights_service import async_get_cheapest_flight
from utils.helpers import (
    chunked,
    get_city_for_airport,
    get_routes_from_airport,
)

DEV_MODE = False
MAX_FLIGHTS_DEV = 3
MAX_AIRBNB_RESULTS = 5
NIGHT_RANGE = [3, 5, 7]
BATCH_SIZE = 10

router = APIRouter()


@router.post("/trip/plan")
async def plan_trip(payload: TripRequest):
    routes = get_routes_from_airport(payload.origin, None if not DEV_MODE else MAX_FLIGHTS_DEV)
    city_to_airports: Dict[str, List[str]] = {}
    for iata in routes:
        city = get_city_for_airport(iata)
        if not city:
            continue
        city_to_airports.setdefault(city, []).append(iata)

    async def process_city(city: str, airports: List[str]):
        cheapest_flight = None
        for airport in airports:
            try:
                outbound = await asyncio.wait_for(
                    async_get_cheapest_flight(payload.origin, airport, payload.departure_date, "one-way", payload.guests),
                    timeout=25,
                )
            except asyncio.TimeoutError:
                continue
            if not outbound:
                continue
            tasks = []
            for nights in NIGHT_RANGE:
                return_date = (
                    datetime.strptime(payload.departure_date, "%Y-%m-%d") + timedelta(days=nights)
                ).strftime("%Y-%m-%d")
                tasks.append(async_get_cheapest_flight(airport, payload.origin, return_date, "one-way", payload.guests))
            inbound_flights = [f for f in await asyncio.gather(*tasks) if f]
            combos = []
            for f in inbound_flights:
                try:
                    dep = datetime.strptime(outbound["departure"], "%Y-%m-%d")
                    ret = datetime.strptime(f["departure"], "%Y-%m-%d")
                    if ret > dep:
                        combos.append({"outbound": outbound, "inbound": f, "price": outbound["price"] + f["price"]})
                except Exception:
                    continue
            best_combo = min(combos, key=lambda x: x["price"], default=None)
            roundtrip = await async_get_cheapest_flight(payload.origin, airport, payload.departure_date, "round-trip", payload.guests)
            if roundtrip:
                try:
                    dep = datetime.strptime(roundtrip["departure"], "%Y-%m-%d")
                    ret = datetime.strptime(roundtrip.get("return") or "", "%Y-%m-%d")
                    if ret <= dep:
                        roundtrip = None
                except Exception:
                    roundtrip = None
            candidate = None
            if roundtrip and (not best_combo or roundtrip["price"] < best_combo["price"]):
                candidate = roundtrip
            elif best_combo:
                candidate = {**best_combo["outbound"], "return": best_combo["inbound"]["departure"], "price": best_combo["price"]}
            if candidate and candidate.get("return"):
                try:
                    dep = datetime.strptime(candidate["departure"], "%Y-%m-%d")
                    ret = datetime.strptime(candidate["return"], "%Y-%m-%d")
                    if ret > dep:
                        if not cheapest_flight or candidate["price"] < cheapest_flight["price"]:
                            cheapest_flight = candidate
                except Exception:
                    continue
        if not cheapest_flight:
            return None
        dep_date, ret_date = cheapest_flight["departure"], cheapest_flight["return"]
        try:
            airbnbs = await asyncio.wait_for(
                async_get_airbnbs(city, dep_date, ret_date, payload.guests, MAX_AIRBNB_RESULTS),
                timeout=25,
            )
        except asyncio.TimeoutError:
            return None
        if not airbnbs:
            return None
        best_airbnb = airbnbs[0]
        total_cost = cheapest_flight["price"] + (best_airbnb.price.as_float() or 0)
        if total_cost > payload.budget:
            return None
        nights = (
            datetime.strptime(ret_date, "%Y-%m-%d") - datetime.strptime(dep_date, "%Y-%m-%d")
        ).days
        return {
            "destination": cheapest_flight["destination"],
            "city": city,
            "flight": cheapest_flight,
            "airbnb": {
                "id": best_airbnb.id,
                "title": best_airbnb.title,
                "url": str(best_airbnb.url),
                "rating": best_airbnb.rating,
                "reviews": best_airbnb.reviews,
                "price": best_airbnb.price.as_float(),
                "photos": [str(p) for p in best_airbnb.photos],
            },
            "nights": max(nights, 1),
            "total_cost": total_cost,
        }

    async def enrich_package(pkg: dict):
        itinerary_task = asyncio.create_task(
            asyncio.wait_for(ai_generate_itinerary(pkg["city"], pkg["nights"], pkg["total_cost"], payload.budget), timeout=30)
        )
        itinerary = await itinerary_task
        tagline = await ai_generate_tagline(itinerary, pkg["city"])
        image_task = asyncio.create_task(
            asyncio.wait_for(ai_generate_image(pkg["city"], itinerary, tagline), timeout=90)
        )
        pkg["itinerary"], pkg["image"], pkg["tagline"] = itinerary, await image_task, tagline
        return pkg

    async def package_generator():
        seen_cities = set()
        packages = []
        for batch in chunked(city_to_airports.items(), BATCH_SIZE):
            results = await asyncio.gather(*[process_city(city, airports) for city, airports in batch])
            for package in filter(None, results):
                if package["city"] in seen_cities:
                    continue
                seen_cities.add(package["city"])
                packages.append(package)
        scored = [(await ai_score_package(pkg), pkg) for pkg in packages]
        top_packages = [pkg for _, pkg in sorted(scored, key=lambda x: x[0], reverse=True)[:3]]
        enriched = await asyncio.gather(*[enrich_package(pkg) for pkg in top_packages])
        for pkg in enriched:
            yield json.dumps(pkg) + "\n"
            await asyncio.sleep(0)

    return StreamingResponse(package_generator(), media_type="application/json")


app = FastAPI(title="Trip Package Planner API", version="5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
