from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
import time
from typing import Optional, Any, Dict
from fastapi.middleware.cors import CORSMiddleware


# ----------------------
# CONFIG
# ----------------------

AMADEUS_API_KEY = "rAXRWAFFMtR7KwimuJ2sprbfhiJMAj2W"
AMADEUS_API_SECRET = "D6ZptSzflB3skuRG"
AMADEUS_BASE_URL = "https://test.api.amadeus.com"
TOKEN_URL = f"{AMADEUS_BASE_URL}/v1/security/oauth2/token"

# ----------------------
# TOKEN MANAGEMENT
# ----------------------
class AmadeusAuth:
    _token: Optional[str] = None
    _expiry: float = 0

    @classmethod
    def get_token(cls) -> str:
        """Returns a valid token, refreshing it if expired."""
        if not cls._token or time.time() >= cls._expiry:
            cls._refresh_token()
        return cls._token

    @classmethod
    def _refresh_token(cls) -> None:
        """Fetch a new token from Amadeus."""
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": AMADEUS_API_KEY,
            "client_secret": AMADEUS_API_SECRET,
        }
        response = requests.post(TOKEN_URL, headers=headers, data=data)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to authenticate with Amadeus")
        token_data = response.json()
        cls._token = token_data["access_token"]
        cls._expiry = time.time() + token_data["expires_in"] - 60  # Refresh 1 min early

# ----------------------
# REQUEST / RESPONSE MODELS
# ----------------------
class FlightInspirationRequest(BaseModel):
    origin: str = Field(..., description="Origin IATA code (e.g., JFK)")
    departure_date: Optional[str] = Field(None, description="Departure date or date range (YYYY-MM-DD or YYYY-MM-DD,YYYY-MM-DD)")
    max_price: Optional[int] = Field(None, description="Maximum price in USD")

class FlightInspirationResponse(BaseModel):
    data: Any
    meta: Optional[Dict] = None

# ----------------------
# FLIGHT SERVICE
# ----------------------
def get_flight_inspiration(origin: str, departure_date: Optional[str], max_price: Optional[int]) -> dict:
    """Fetch inspiration flights from Amadeus API."""
    token = AmadeusAuth.get_token()
    url = f"{AMADEUS_BASE_URL}/v1/shopping/flight-destinations"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"origin": origin}
    if departure_date:
        params["departureDate"] = departure_date
    if max_price:
        params["maxPrice"] = max_price

    print(f"Fetching flight inspirations for {origin} with params: {params}")

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json().get("errors", "Failed to fetch flight inspirations"))
    return response.json()

# ----------------------
# APP
# ----------------------
app = FastAPI(title="Flight Inspiration API", version="1.0")

# Add this CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/flights/inspiration", response_model=FlightInspirationResponse)
def fetch_flight_inspirations(payload: FlightInspirationRequest):
    """
    Fetch inspiration flights from a specific city.
    Date and budget are optional filters.
    """
    return get_flight_inspiration(payload.origin, payload.departure_date, payload.max_price)

