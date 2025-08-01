import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
import requests, os, json, time
from typing import Optional, List, Dict, Tuple, Any
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from fast_flights import FlightData, Passengers, get_flights, Result
from concurrent.futures import ThreadPoolExecutor
import itertools
from openai import OpenAI

# ----------------------
# CONFIG
# ----------------------
ROUTES_URL = "https://raw.githubusercontent.com/Jonty/airline-route-data/master/airline_routes.json"
DATA_DIR = "data"
ROUTES_FILE = os.path.join(DATA_DIR, "airline_routes.json")
CACHE_FILE = os.path.join(DATA_DIR, "cache.json")

HASDATA_API_KEY = "569634eb-f56e-4a80-bc90-19d9d35cf9b0"
HASDATA_BASE_URL = "https://api.hasdata.com/scrape/airbnb/listing"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEV_MODE = False
MAX_FLIGHTS_DEV = 3
MAX_AIRBNB_RESULTS = 5
MAX_CONCURRENT_FLIGHT_CALLS = 20
MAX_CONCURRENT_AIRBNB_CALLS = 15
CACHE_TTL = 86400  # 1 day
NIGHT_RANGE = [3, 5, 7]  # fewer ranges to reduce calls
BATCH_SIZE = 10

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------
# MODELS
# ----------------------
class Carrier(BaseModel):
    iata: str
    name: str

class Route(BaseModel):
    iata: str
    km: Optional[float] = 0
    min: Optional[int] = 0
    carriers: List[Carrier] = []

class Airport(BaseModel):
    city_name: Optional[str] = None
    continent: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    display_name: Optional[str] = None
    elevation: Optional[float] = None
    iata: str
    icao: Optional[str] = None
    latitude: Optional[str] = None
    longitude: Optional[str] = None
    name: Optional[str] = None
    timezone: Optional[str] = None
    routes: List[Route] = []

AirportsData = Dict[str, Airport]

class PriceBreakdown(BaseModel):
    description: str
    price: str

class AirbnbPrice(BaseModel):
    originalPrice: Optional[str]
    discountedPrice: Optional[str] = None
    qualifier: Optional[str] = None
    breakdown: Optional[List[PriceBreakdown]] = []
    def as_float(self) -> Optional[float]:
        value = self.discountedPrice or self.originalPrice
        if not value: return None
        try:
            price_val = float(value.replace("$", "").replace(",", "").strip())
            return price_val if price_val > 0 else None
        except ValueError:
            return None

class AirbnbProperty(BaseModel):
    id: str
    url: HttpUrl
    title: str
    latitude: float
    longitude: float
    description: Optional[str] = None
    photos: List[HttpUrl] = []
    rating: Optional[float] = None
    reviews: Optional[int] = None
    badges: Optional[List[str]] = []
    price: AirbnbPrice

class AirbnbResponse(BaseModel):
    requestMetadata: Dict
    properties: List[AirbnbProperty]
    pagination: Optional[Dict] = None

class TripRequest(BaseModel):
    origin: str
    departure_date: str
    budget: int
    guests: Optional[int] = 1

# ----------------------
# DATA LOADING
# ----------------------
def ensure_data_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ROUTES_FILE):
        print("Downloading airline_routes.json...")
        r = requests.get(ROUTES_URL)
        r.raise_for_status()
        with open(ROUTES_FILE, "w") as f: f.write(r.text)

def load_airports(filepath) -> AirportsData:
    with open(filepath) as f:
        raw = json.load(f)
        return {iata: Airport(**data) for iata, data in raw.items()}

ensure_data_file()
routes_data: AirportsData = load_airports(ROUTES_FILE)

# ----------------------
# HELPERS
# ----------------------
def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk: break
        yield chunk

def get_routes_from_airport(origin: str, limit: int = 5) -> List[str]:
    airport_info = routes_data.get(origin)
    if not airport_info: return []
    destinations = list({r.iata for r in airport_info.routes})
    return destinations[:limit] if limit else destinations

def get_city_for_airport(iata: str) -> str:
    return routes_data.get(iata, Airport(city_name=iata, iata=iata)).city_name

def parse_price(price: str) -> Optional[float]:
    if not price or not isinstance(price, str): return None
    if "unavailable" in price.lower(): return None
    try:
        value = float(price.replace("$", "").replace(",", "").strip())
        return value if value > 0 else None
    except ValueError:
        return None

def parse_custom_date(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%I:%M %p on %a, %b %d").replace(year=datetime.now().year).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

def make_json_safe(obj):
    if isinstance(obj, BaseException):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return obj

# ----------------------
# CACHING
# ----------------------
cache: Dict[str, Tuple[float, Any]] = {}
def serialize_for_cache(value: Any):
    if isinstance(value, list): return [serialize_for_cache(v) for v in value]
    if hasattr(value, "dict"):
        d = value.dict(exclude_none=True)
        for k, v in d.items():
            if isinstance(v, list): d[k] = [str(x) if isinstance(x, HttpUrl) else x for x in v]
            elif isinstance(v, HttpUrl): d[k] = str(v)
        return d
    return value
def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({k: (t, serialize_for_cache(v)) for k, (t, v) in cache.items()}, f)
    except Exception as e: print(f"Error saving cache: {e}")
def load_cache():
    global cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                raw = json.load(f)
                cache = {k: (t, v) for k, (t, v) in raw.items()}
        except Exception as e: print(f"Error loading cache: {e}")
def get_cache(key: str):
    entry = cache.get(key)
    if entry and time.time() - entry[0] < CACHE_TTL: return entry[1]
    return None
def set_cache(key: str, value: Any):
    cache[key] = (time.time(), serialize_for_cache(value)); save_cache()
load_cache()

# ----------------------
# AI CACHING
# ----------------------
def get_ai_cache(key: str):
    return get_cache(f"ai:{key}")

def set_ai_cache(key: str, value: Any):
    set_cache(f"ai:{key}", value)

# ----------------------
# FLIGHT SERVICE
# ----------------------
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FLIGHT_CALLS)
semaphore = asyncio.Semaphore(MAX_CONCURRENT_FLIGHT_CALLS)
async def async_get_cheapest_flight(origin: str, destination: str, date: str, trip_type="one-way", adults: int = 1):
    cache_key = f"flight:{trip_type}:{origin}:{destination}:{date}:{adults}"
    cached = get_cache(cache_key)
    if cached: return cached
    async with semaphore:
        loop = asyncio.get_running_loop()
        try:
            result: Result = await asyncio.wait_for(loop.run_in_executor(
                executor,
                lambda: get_flights(
                    flight_data=[FlightData(date=date, from_airport=origin, to_airport=destination)],
                    trip=trip_type,
                    seat="economy",
                    passengers=Passengers(adults=adults),
                    fetch_mode="fallback"
                )
            ), timeout=25)
            flights = result.flights or []
            valid_flights = [f for f in flights if parse_price(f.price)]
            if not valid_flights: return None
            cheapest = sorted(valid_flights, key=lambda x: parse_price(x.price))[0]
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
                "stops": cheapest.stops
            }
            set_cache(cache_key, flight_data)
            return flight_data
        except Exception as e: 
            print(f"Error fetching {trip_type} flight {origin}->{destination}: {e}")
        return None

# ----------------------
# AIRBNB SERVICE
# ----------------------
airbnb_semaphore = asyncio.Semaphore(MAX_CONCURRENT_AIRBNB_CALLS)
async def async_get_airbnbs(city: str, check_in: str, check_out: str, guests: int = 1, limit: int = 10) -> List[AirbnbProperty]:
    cache_key = f"airbnb:{city}:{check_in}:{check_out}:{guests}:{limit}"
    cached = get_cache(cache_key)
    if cached: return [AirbnbProperty(**p) for p in cached]
    async with airbnb_semaphore:
        headers = {"x-api-key": HASDATA_API_KEY, "Content-Type": "application/json"}
        params = {"location": city, "checkIn": check_in, "checkOut": check_out, "adults": guests, "limit": limit}
        try:
            resp = await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(
                None, lambda: requests.get(HASDATA_BASE_URL, headers=headers, params=params)
            ), timeout=25)
            if resp.status_code == 200:
                parsed = AirbnbResponse(**resp.json())
                filtered = [p for p in parsed.properties if p.price.as_float()]
                sorted_props = sorted(filtered, key=lambda x: x.price.as_float())
                set_cache(cache_key, sorted_props)
                return sorted_props
            else: print(f"HasData error: {resp.text}")
        except Exception as e: print(f"Error fetching Airbnbs: {e}")
        return []

# ----------------------
# AI SERVICES
# ----------------------
async def ai_score_package(package):
    score = 0
    if package["nights"] >= 4: score += 2
    if package["airbnb"]["rating"]: score += package["airbnb"]["rating"]
    score += max(0, 5 - (package["total_cost"] / package["nights"] / 200))
    return score

async def ai_generate_itinerary(city, nights, total_cost, budget):
    cache_key = f"itinerary:{city}:{nights}:{total_cost}:{budget}"
    cached = get_ai_cache(cache_key)
    if cached: return cached
    remaining = max(budget - total_cost, 0)
    daily_budget = remaining // max(nights, 1)
    prompt = f"""
    Create a detailed {nights}-day itinerary for a trip to {city}.
    Flights + stay cost ${total_cost}, total budget ${budget}.
    That leaves about ${daily_budget} per day for food, attractions, and experiences.
    Suggest realistic activities, at least one free/low-cost daily.
    Format: Day-by-day plan with costs in parentheses.
    """
    response = await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a travel planner."},
                {"role": "user", "content": prompt}
            ]
        )
    ), timeout=30)
    itinerary = response.choices[0].message.content
    set_ai_cache(cache_key, itinerary)
    return itinerary

async def ai_generate_image(city, itinerary):
    # Cache key now uses itinerary to make unique
    cache_key = f"poster_image:{city}:{hash(itinerary)}"
    cached = get_ai_cache(cache_key)
    if cached:
        return cached

    # Step 1: Extract 3 key themes + taglines from the itinerary
    themes_prompt = f"""
    From this itinerary for a trip to {city}, extract 3 key trip themes (e.g., culture, food, adventure). 
    For each theme, create a catchy travel poster tagline (max 6 words).
    Return them as a JSON list of objects with 'theme' and 'tagline' fields.
    Itinerary:
    {itinerary}
    """
    themes_response = await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a travel copywriter making snappy poster captions."},
                {"role": "user", "content": themes_prompt}
            ],
            temperature=0.7
        )
    ), timeout=25)
    themes_json = themes_response.choices[0].message.content.strip()

    try:
        themes = json.loads(themes_json)
    except:
        # Fallback if parsing fails
        themes = [{"theme": city, "tagline": "Experience the Best of " + city}]

    # Step 2: Build the DALL·E poster prompt
    poster_panels = "\n".join(
        [f"Panel {i+1}: hyper-realistic photo representing {t['theme']}, with bold overlay text '{t['tagline']}'." for i, t in enumerate(themes[:3])]
    )
    prompt = f"""
    A 3-panel vertical travel poster for {city}.
    {poster_panels}
    Extremely realistic, cinematic photography style, vibrant colors, HDR details.
    Modern poster layout with bold typography for city and taglines.
    Looks like a high-end tourism ad. Photo-real collage composition.
    """

    # Step 3: Generate the image with DALL·E 3
    try:
        response = await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1792"  # vertical poster
            )
        ), timeout=90)
        image_url = response.data[0].url
        set_ai_cache(cache_key, image_url)
        return image_url
    except Exception as e:
        print(f"[AI Poster] Failed for {city}: {e}")
        return None

# ----------------------
# FASTAPI APP
# ----------------------
app = FastAPI(title="Trip Package Planner API", version="5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# ENDPOINT (STREAMING)
# ----------------------
@app.post("/trip/plan")
async def plan_trip(payload: TripRequest):
    routes = get_routes_from_airport(payload.origin, None if not DEV_MODE else MAX_FLIGHTS_DEV)
    city_to_airports: Dict[str, List[str]] = {}
    for iata in routes:
        city = get_city_for_airport(iata)
        if not city: continue
        city_to_airports.setdefault(city, []).append(iata)

    async def process_city(city, airports):
        cheapest_flight = None
        for airport in airports:
            try:
                outbound = await asyncio.wait_for(async_get_cheapest_flight(payload.origin, airport, payload.departure_date, "one-way", payload.guests), timeout=25)
            except asyncio.TimeoutError:
                continue
            if not outbound: continue
            tasks = []
            for nights in NIGHT_RANGE:
                return_date = (datetime.strptime(payload.departure_date, "%Y-%m-%d") + timedelta(days=nights)).strftime("%Y-%m-%d")
                tasks.append(async_get_cheapest_flight(airport, payload.origin, return_date, "one-way", payload.guests))
            inbound_flights = [f for f in await asyncio.gather(*tasks) if f]
            combos = []
            for f in inbound_flights:
                try:
                    dep = datetime.strptime(outbound["departure"], "%Y-%m-%d")
                    ret = datetime.strptime(f["departure"], "%Y-%m-%d")
                    if ret > dep:
                        combos.append({"outbound": outbound, "inbound": f, "price": outbound["price"] + f["price"]})
                except Exception: continue
            best_combo = min(combos, key=lambda x: x["price"], default=None)
            roundtrip = await async_get_cheapest_flight(payload.origin, airport, payload.departure_date, "round-trip", payload.guests)
            if roundtrip:
                try:
                    dep = datetime.strptime(roundtrip["departure"], "%Y-%m-%d")
                    ret = datetime.strptime(roundtrip.get("return") or "", "%Y-%m-%d")
                    if ret <= dep: roundtrip = None
                except Exception: roundtrip = None
            candidate = None
            if roundtrip and (not best_combo or roundtrip["price"] < best_combo["price"]): candidate = roundtrip
            elif best_combo: candidate = {**best_combo["outbound"], "return": best_combo["inbound"]["departure"], "price": best_combo["price"]}
            if candidate and candidate.get("return"):
                try:
                    dep = datetime.strptime(candidate["departure"], "%Y-%m-%d")
                    ret = datetime.strptime(candidate["return"], "%Y-%m-%d")
                    if ret > dep:
                        if not cheapest_flight or candidate["price"] < cheapest_flight["price"]: cheapest_flight = candidate
                except Exception: continue
        if not cheapest_flight: return None
        dep_date, ret_date = cheapest_flight["departure"], cheapest_flight["return"]
        try:
            airbnbs = await asyncio.wait_for(async_get_airbnbs(city, dep_date, ret_date, payload.guests, MAX_AIRBNB_RESULTS), timeout=25)
        except asyncio.TimeoutError:
            return None
        if not airbnbs: return None
        best_airbnb = airbnbs[0]
        total_cost = cheapest_flight["price"] + (best_airbnb.price.as_float() or 0)
        if total_cost > payload.budget: return None
        nights = (datetime.strptime(ret_date, "%Y-%m-%d") - datetime.strptime(dep_date, "%Y-%m-%d")).days
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
                "photos": [str(p) for p in best_airbnb.photos]
            },
            "nights": max(nights, 1),
            "total_cost": total_cost
        }
    
    async def enrich_package(pkg):
        itinerary_task = asyncio.create_task(asyncio.wait_for(ai_generate_itinerary(pkg["city"], pkg["nights"], pkg["total_cost"], payload.budget), timeout=30))
        itinerary = await itinerary_task
        image_task = asyncio.create_task(asyncio.wait_for(ai_generate_image(pkg["city"], itinerary), timeout=90))
        pkg["itinerary"], pkg["image"] = itinerary, await image_task
        return pkg


    async def package_generator():
        seen_cities = set()
        packages = []
        for batch in chunked(city_to_airports.items(), BATCH_SIZE):
            results = await asyncio.gather(*[process_city(city, airports) for city, airports in batch])
            for package in filter(None, results):
                if package["city"] in seen_cities: continue
                seen_cities.add(package["city"])
                packages.append(package)
        scored = [(await ai_score_package(pkg), pkg) for pkg in packages]
        top_packages = [pkg for _, pkg in sorted(scored, key=lambda x: x[0], reverse=True)[:3]]  # only top 3
        enriched = await asyncio.gather(*[enrich_package(pkg) for pkg in top_packages])
        for pkg in enriched:
            yield json.dumps(make_json_safe(pkg)) + "\n"
            await asyncio.sleep(0)

    return StreamingResponse(package_generator(), media_type="application/json")
