"""Open-Meteo weather provider — free, no API key, no signup.

Two calls: geocoding turns "Ahmedabad" into coordinates, then the forecast
endpoint returns current conditions plus today's high/low and rain probability.

Why this and not the ``weather-mcp`` MCP server: that server is a wrapper over
these same Open-Meteo (and NOAA) endpoints, delivered over stdio — it would add
a Node subprocess to the deployment and 17 tools to the agent's tool list to
reach the same three numbers. When its extra tools (radar, marine, air quality)
become worth having, it implements :class:`~tools.weather.base.WeatherProvider`
as a sibling of this file.
"""

from __future__ import annotations

import httpx

from core.settings import settings
from tools.weather.base import Weather, WeatherError

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes. Open-Meteo returns a number; users read
# English. Grouped rather than exhaustive — "Slight or moderate thunderstorm"
# helps nobody decide whether to carry an umbrella.
_CONDITIONS = {
    0: "Clear",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Freezing fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    56: "Freezing drizzle",
    57: "Freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light showers",
    81: "Showers",
    82: "Violent showers",
    85: "Snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}


def _condition(code: int | None) -> str:
    return _CONDITIONS.get(int(code), "Unknown") if code is not None else "Unknown"


def _place_name(hit: dict) -> str:
    """"Ahmedabad, Gujarat, India" — enough for the user to catch a wrong match."""
    parts = [hit.get("name"), hit.get("admin1"), hit.get("country")]
    return ", ".join(p for p in parts if p)


class OpenMeteoProvider:
    """Open-Meteo, over plain HTTPS. No credentials to configure."""

    name = "openmeteo"

    def __init__(self, timeout: float | None = None) -> None:
        self._timeout = timeout or settings.weather_timeout_seconds

    async def current(self, location: str) -> Weather:
        location = (location or "").strip()
        if not location:
            raise WeatherError("No location given, so there is nowhere to look up.")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            place = await self._geocode(client, location)
            return await self._forecast(client, place)

    async def _geocode(self, client: httpx.AsyncClient, location: str) -> dict:
        data = await self._get(
            client,
            _GEOCODE_URL,
            {"name": location, "count": 1, "language": "en", "format": "json"},
        )
        results = data.get("results") or []
        if not results:
            raise WeatherError(
                f"Couldn't find a place called '{location}'. A city name, or "
                "'city, country', works best."
            )
        return results[0]

    async def _forecast(self, client: httpx.AsyncClient, place: dict) -> Weather:
        data = await self._get(
            client,
            _FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 1,
            },
        )

        current = data.get("current") or {}
        daily = data.get("daily") or {}

        def today(key: str, default: float = 0.0) -> float:
            values = daily.get(key) or []
            return float(values[0]) if values and values[0] is not None else default

        temperature = float(current.get("temperature_2m") or 0.0)
        return Weather(
            location=_place_name(place),
            temperature_c=temperature,
            feels_like_c=float(current.get("apparent_temperature") or temperature),
            condition=_condition(current.get("weather_code")),
            high_c=today("temperature_2m_max", temperature),
            low_c=today("temperature_2m_min", temperature),
            rain_probability=int(today("precipitation_probability_max")),
            wind_kph=float(current.get("wind_speed_10m") or 0.0),
            alerts=[],  # Open-Meteo has no alert feed; see base.describe_alerts
        )

    async def _get(self, client: httpx.AsyncClient, url: str, params: dict) -> dict:
        """One request, with every expected failure turned into WeatherError.

        A weather lookup is a nice-to-have inside a briefing: it must degrade to
        "couldn't get the weather" rather than take the whole run down.
        """
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise WeatherError("The weather service didn't respond in time.") from exc
        except httpx.HTTPStatusError as exc:
            raise WeatherError(
                f"The weather service returned {exc.response.status_code}."
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise WeatherError(f"Couldn't reach the weather service: {exc}") from exc
