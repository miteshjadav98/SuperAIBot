"""Weather backends, chosen by ``settings.weather_provider``.

    from tools.weather import get_provider
    weather = await get_provider().current("Ahmedabad")

Adding a backend: implement :class:`~tools.weather.base.WeatherProvider` in a
new module, add one line to ``_PROVIDERS``, set ``WEATHER_PROVIDER`` in .env.
"""

from __future__ import annotations

from functools import lru_cache

from core.settings import settings
from tools.weather.base import Weather, WeatherError, WeatherProvider, describe_alerts

__all__ = ["Weather", "WeatherError", "WeatherProvider", "describe_alerts", "get_provider"]

_PROVIDERS = {"openmeteo": "tools.weather.openmeteo:OpenMeteoProvider"}


@lru_cache(maxsize=1)
def get_provider() -> WeatherProvider:
    """The configured provider, built once per process."""
    target = _PROVIDERS.get(settings.weather_provider.lower())
    if target is None:
        raise WeatherError(
            f"Unknown weather provider '{settings.weather_provider}'. "
            f"Known providers: {', '.join(sorted(_PROVIDERS))}."
        )

    module_path, _, class_name = target.partition(":")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)()
