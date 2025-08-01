"""AI services for itinerary creation and poster image generation."""
from __future__ import annotations

import asyncio
import os
from hashlib import sha1
from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline
from openai import OpenAI

from utils.caching import get_ai_cache, set_ai_cache

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SD3_MODEL_PATH = os.getenv("SD3_MODEL_PATH", "stabilityai/stable-diffusion-3-medium")
_device = "cuda" if torch.cuda.is_available() else "cpu"


class _PosterPipeline:
    _pipe: Optional[DiffusionPipeline] = None

    @classmethod
    def get_pipeline(cls) -> DiffusionPipeline:
        if cls._pipe is None:
            cls._pipe = DiffusionPipeline.from_pretrained(SD3_MODEL_PATH, torch_dtype=torch.float16 if _device == "cuda" else torch.float32)
            cls._pipe.to(_device)
        return cls._pipe


def _save_image(image, city: str, tagline: str) -> str:
    images_dir = os.path.join("data", "images")
    os.makedirs(images_dir, exist_ok=True)
    filename = sha1(f"{city}-{tagline}".encode()).hexdigest() + ".png"
    path = os.path.join(images_dir, filename)
    image.save(path)
    return path


async def ai_score_package(package: dict) -> int:
    score = 0
    if package["nights"] >= 4:
        score += 2
    if package["airbnb"]["rating"]:
        score += package["airbnb"]["rating"]
    score += max(0, 5 - (package["total_cost"] / package["nights"] / 200))
    return score


async def ai_generate_itinerary(city: str, nights: int, total_cost: int, budget: int) -> str:
    cache_key = f"itinerary:{city}:{nights}:{total_cost}:{budget}"
    cached = get_ai_cache(cache_key)
    if cached:
        return cached
    remaining = max(budget - total_cost, 0)
    daily_budget = remaining // max(nights, 1)
    prompt = (
        f"Create a detailed {nights}-day itinerary for a trip to {city}.\n"
        f"Flights + stay cost ${total_cost}, total budget ${budget}.\n"
        f"That leaves about ${daily_budget} per day for food, attractions, and experiences.\n"
        "Suggest realistic activities, at least one free/low-cost daily.\n"
        "Format: Day-by-day plan with costs in parentheses."
    )
    response = await asyncio.wait_for(
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a travel planner."},
                    {"role": "user", "content": prompt},
                ],
            ),
        ),
        timeout=30,
    )
    itinerary = response.choices[0].message.content
    set_ai_cache(cache_key, itinerary)
    return itinerary


async def ai_generate_tagline(itinerary: str, city: str) -> str:
    prompt = (
        f"Create a catchy 6 word tagline for a travel poster about {city}.\n"
        f"Use the following itinerary for inspiration:\n{itinerary}"
    )
    response = await asyncio.wait_for(
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            ),
        ),
        timeout=20,
    )
    return response.choices[0].message.content.strip().strip("\n")


async def ai_generate_image(city: str, itinerary: str, tagline: str) -> Optional[str]:
    cache_key = f"poster_image:{city}:{hash(itinerary)}:{tagline}"
    cached = get_ai_cache(cache_key)
    if cached:
        return cached

    activities = " ".join(itinerary.splitlines()[:3])
    prompt = (
        f"A cinematic travel poster of {city}. {tagline}. Activities include: {activities}. "
        "Hyper realistic, vibrant colors, high detail, tourism advertisement style."
    )

    try:
        pipe = _PosterPipeline.get_pipeline()
        image = await asyncio.get_event_loop().run_in_executor(None, lambda: pipe(prompt).images[0])
        path = _save_image(image, city, tagline)
        set_ai_cache(cache_key, path)
        return path
    except Exception as e:
        print(f"[AI Poster] Failed for {city}: {e}")
        return None
