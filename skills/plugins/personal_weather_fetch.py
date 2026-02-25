"""
skills/plugins/personal_weather_fetch.py — Personal: Weather Fetch

Fetches current weather and a 3-day forecast using the Open-Meteo API.
Completely free, no API key required, no data leaves the machine beyond
the geo-coordinates of the requested location.

Risk: LOW — net:fetch capability required.
"""

from __future__ import annotations

import time
from typing import ClassVar

from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

# Open-Meteo — free, no API key, GDPR-friendly
_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}


class PersonalWeatherFetchSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="personal_weather_fetch",
        version="1.0.0",
        description=(
            "Fetch current weather conditions and 3-day forecast for a location. "
            "Uses Open-Meteo (free, no API key). Provide a city name or lat/lon coordinates."
        ),
        category="personal",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"net:fetch"}),
        requires_confirmation=False,
        timeout_seconds=20,
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name (e.g. 'London'), or 'lat,lon' (e.g. '51.5,-0.1').",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units (default celsius).",
                    "default": "celsius",
                },
            },
            "required": ["location"],
        },
    )

    async def validate(self, location: str, **_) -> None:
        if not location or not location.strip():
            raise SkillValidationError("location must be a non-empty string.")

    async def execute(
        self,
        location: str,
        units: str = "celsius",
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()

        try:
            import httpx

            temp_unit = "celsius" if units == "celsius" else "fahrenheit"
            temp_symbol = "°C" if temp_unit == "celsius" else "°F"

            # Resolve location to lat/lon
            lat: float
            lon: float
            resolved_name: str

            location = location.strip()
            if "," in location:
                # Try parsing as lat,lon
                try:
                    parts = location.split(",")
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    resolved_name = f"{lat},{lon}"
                except (ValueError, IndexError):
                    return SkillResult.fail(
                        self.manifest.name, call_id,
                        f"Could not parse coordinates: '{location}'. Use 'lat,lon' format.",
                    )
            else:
                # Geocode city name
                async with httpx.AsyncClient(timeout=10.0) as client:
                    geo_resp = await client.get(
                        _GEOCODE_URL,
                        params={"name": location, "count": 1, "language": "en", "format": "json"},
                    )
                    geo_resp.raise_for_status()
                    geo = geo_resp.json()

                if not geo.get("results"):
                    return SkillResult.fail(
                        self.manifest.name, call_id,
                        f"Location not found: '{location}'", "LookupError",
                    )
                result = geo["results"][0]
                lat = result["latitude"]
                lon = result["longitude"]
                resolved_name = f"{result.get('name', location)}, {result.get('country', '')}"

            # Fetch weather
            async with httpx.AsyncClient(timeout=10.0) as client:
                wx_resp = await client.get(
                    _WEATHER_URL,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": [
                            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                            "weather_code", "wind_speed_10m", "wind_direction_10m",
                            "precipitation", "cloud_cover",
                        ],
                        "daily": [
                            "temperature_2m_max", "temperature_2m_min",
                            "weather_code", "precipitation_sum", "wind_speed_10m_max",
                        ],
                        "temperature_unit": temp_unit,
                        "wind_speed_unit": "kmh",
                        "precipitation_unit": "mm",
                        "forecast_days": 4,
                        "timezone": "auto",
                    },
                )
                wx_resp.raise_for_status()
                wx = wx_resp.json()

            current = wx.get("current", {})
            daily = wx.get("daily", {})

            wmo = int(current.get("weather_code", 0))
            condition = _WMO_CODES.get(wmo, f"Code {wmo}")

            forecast = []
            dates = daily.get("time", [])
            for i, d in enumerate(dates[1:4], 1):  # next 3 days
                day_wmo = int(daily.get("weather_code", [0] * 10)[i])
                forecast.append({
                    "date": d,
                    "condition": _WMO_CODES.get(day_wmo, f"Code {day_wmo}"),
                    "temp_max": f"{daily.get('temperature_2m_max', [])[i]}{temp_symbol}",
                    "temp_min": f"{daily.get('temperature_2m_min', [])[i]}{temp_symbol}",
                    "precipitation_mm": daily.get("precipitation_sum", [])[i],
                    "wind_max_kmh": daily.get("wind_speed_10m_max", [])[i],
                })

            duration_ms = (time.monotonic() - t_start) * 1000
            return SkillResult.ok(
                skill_name=self.manifest.name,
                skill_call_id=call_id,
                output={
                    "location": resolved_name,
                    "coordinates": {"lat": lat, "lon": lon},
                    "current": {
                        "condition": condition,
                        "temperature": f"{current.get('temperature_2m')}{temp_symbol}",
                        "feels_like": f"{current.get('apparent_temperature')}{temp_symbol}",
                        "humidity": f"{current.get('relative_humidity_2m')}%",
                        "wind": f"{current.get('wind_speed_10m')} km/h",
                        "cloud_cover": f"{current.get('cloud_cover')}%",
                        "precipitation": f"{current.get('precipitation')} mm",
                    },
                    "forecast_3day": forecast,
                },
                duration_ms=duration_ms,
            )

        except (OSError, ValueError) as e:
            return SkillResult.fail(
                self.manifest.name, call_id,
                f"{type(e).__name__}: {e}", type(e).__name__,
                duration_ms=(time.monotonic() - t_start) * 1000,
            )
        except BaseException as e:
            return SkillResult.fail(
                self.manifest.name, call_id,
                f"{type(e).__name__}: {e}", type(e).__name__,
                duration_ms=(time.monotonic() - t_start) * 1000,
            )