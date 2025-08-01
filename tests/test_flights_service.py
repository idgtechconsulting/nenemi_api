import asyncio
from unittest.mock import patch, MagicMock

import pytest

from services.flights_service import async_get_cheapest_flight


@pytest.mark.asyncio
async def test_async_get_cheapest_flight(monkeypatch):
    mock_result = MagicMock()
    flight = MagicMock(price="$100", name="AA", departure="1:00 PM on Mon, Jan 1", duration="2h", stops=0)
    mock_result.flights = [flight]
    with patch("services.flights_service.get_flights", return_value=mock_result):
        res = await async_get_cheapest_flight("JFK", "LAX", "2024-01-01")
        assert res["price"] == 100
