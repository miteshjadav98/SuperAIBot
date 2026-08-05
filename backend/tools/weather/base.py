"""The weather provider contract.

Everything above this line is provider-agnostic: the Personal Assistant, its
tools and the daily briefing know only about :class:`WeatherProvider`. Swapping
Open-Meteo for OpenWeather, or for the ``weather-mcp`` MCP server, is a new
implementation of this protocol plus one settings value — no agent code changes.

Two deliberate differences from :mod:`tools.email.base`:

* **Provider methods are async.** Email backends are blocking client libraries,
  so that protocol is synchronous and the tool layer offloads to a thread. Every
  weather source here is an HTTP call, and the briefing fans several sources out
  with ``asyncio.gather`` — making the protocol async keeps that fan-out real
  rather than three threads pretending to be concurrent.
* **Location is a string, resolved by the provider.** "Ahmedabad" means a
  geocoding step for one backend and a station id for another; nothing above
  this layer may assume coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


class WeatherError(RuntimeError):
    """An expected weather failure — surfaced to the model, not raised at the graph.

    Unknown place name, upstream timeout, provider not configured. The tool
    layer turns these into readable text the model can recover from ("I couldn't
    find that city — which one did you mean?").
    """


@dataclass
class Weather:
    """Current conditions, in the subset of fields every provider can supply.

    Kept to what the briefing actually shows. A provider with richer data (air
    quality, marine, radar) exposes it through its own tools rather than
    widening this contract for everyone.
    """

    location: str  # resolved, human-readable — "Ahmedabad, Gujarat, India"
    temperature_c: float
    feels_like_c: float
    condition: str  # plain English: "Sunny", "Light rain"
    high_c: float
    low_c: float
    rain_probability: int  # percent, 0-100, for the rest of today
    wind_kph: float
    alerts: list[str]  # severe-weather warnings; empty when the source has none

    def summary(self) -> str:
        """One line, the way the briefing wants to read it."""
        return (
            f"{self.temperature_c:.0f}°C | {self.condition} | "
            f"H {self.high_c:.0f}° L {self.low_c:.0f}° | "
            f"{self.rain_probability}% chance of rain"
        )


@runtime_checkable
class WeatherProvider(Protocol):
    """What a weather backend must do. One method — the MVP needs one fact."""

    name: str

    async def current(self, location: str) -> Weather:
        """Current conditions for ``location``.

        Raises :class:`WeatherError` for anything expected, including a place
        name that cannot be resolved.
        """
        ...


def describe_alerts(alerts: list[str]) -> Optional[str]:
    """Alerts as one line, or ``None`` when there are none to show.

    Worth knowing: Open-Meteo has no alert feed, and NOAA's (which the
    ``weather-mcp`` server wraps) is US-only — so outside the US this is empty
    whichever provider is configured. The field exists so a provider that *does*
    have alerts needs no contract change.
    """
    return "; ".join(alerts) if alerts else None
