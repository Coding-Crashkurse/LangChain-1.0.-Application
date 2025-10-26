from __future__ import annotations

from pydantic import BaseModel, Field


class Weather(BaseModel):
    city: str = Field(..., description="Stadt")
    temperature: float = Field(..., description="Temperatur in °C")
    condition: str = Field(..., description="Beschreibung der Wetterlage")
