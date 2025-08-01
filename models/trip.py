"""Pydantic models used across the trip planner service."""
from __future__ import annotations

from typing import List, Optional, Dict
from pydantic import BaseModel, HttpUrl

class Carrier(BaseModel):
    """Airline carrier metadata."""
    iata: str
    name: str

class Route(BaseModel):
    """Represents a connection between airports."""
    iata: str
    km: Optional[float] = 0
    min: Optional[int] = 0
    carriers: List[Carrier] = []

class Airport(BaseModel):
    """Simplified airport information with routes."""
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
        if not value:
            return None
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
    """Incoming trip planning request."""
    origin: str
    departure_date: str
    budget: int
    guests: Optional[int] = 1
