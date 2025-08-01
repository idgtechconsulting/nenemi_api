import asyncio
from unittest.mock import patch, MagicMock

import pytest

from services.ai_service import ai_generate_itinerary, ai_generate_tagline, ai_generate_image, _PosterPipeline


@pytest.mark.asyncio
async def test_generate_itinerary(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Plan"))]
    with patch("services.ai_service.client.chat.completions.create", return_value=mock_resp):
        result = await ai_generate_itinerary("NYC", 3, 500, 1000)
        assert "Plan" in result


@pytest.mark.asyncio
async def test_generate_tagline(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Great Trip"))]
    with patch("services.ai_service.client.chat.completions.create", return_value=mock_resp):
        tag = await ai_generate_tagline("itinerary", "NYC")
        assert tag == "Great Trip"


@pytest.mark.asyncio
async def test_generate_image(monkeypatch):
    class DummyPipe:
        def __call__(self, prompt):
            class R:
                images = [MagicMock()]
            return R()

    monkeypatch.setattr(_PosterPipeline, "get_pipeline", lambda: DummyPipe())
    monkeypatch.setattr("services.ai_service._save_image", lambda img, city, tagline: "path.png")
    result = await ai_generate_image("NYC", "plan", "tag")
    assert result == "path.png"
