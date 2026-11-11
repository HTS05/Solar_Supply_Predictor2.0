#Open-Meteo forecast client used by the dashboard.

from datetime import date

import pandas as pd
import requests


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CITY_COORDINATES = {
    "Bhopal": (23.2599, 77.4126),
    "Chennai": (13.0827, 80.2707),
    "Delhi": (28.6139, 77.2090),
    "Guwahati": (26.1445, 91.7362),
    "Jaipur": (26.9124, 75.7873),
    "Kolkata": (22.5726, 88.3639),
}


def get_forecast_weather(city, selected_date, timeout=30):
    #Fetch and aggregate Open-Meteo hourly weather for one future day.
    if city not in CITY_COORDINATES:
        raise ValueError(f"Unknown city: {city}")

    selected_date = pd.Timestamp(selected_date)
    today = pd.Timestamp(date.today())
    if selected_date < today or selected_date > today + pd.Timedelta(days=16):
        raise ValueError("Forecast API supports today through the next 16 days")

    latitude, longitude = CITY_COORDINATES[city]
    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m,precipitation,surface_pressure,shortwave_radiation",
            "timezone": "Asia/Kolkata",
            "forecast_days": 16,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    hourly = response.json().get("hourly", {})
    date_text = selected_date.strftime("%Y-%m-%d")
    indexes = [index for index, value in enumerate(hourly.get("time", [])) if value.startswith(date_text)]
    if not indexes:
        raise ValueError(f"No forecast returned for {city} on {selected_date.date()}")

    def values(name):
        return [hourly[name][index] for index in indexes if hourly[name][index] is not None]

    def average(name):
        items = [float(value) for value in values(name)]
        if not items:
            raise ValueError(f"Forecast returned no {name} value")
        return sum(items) / len(items)

    return {
        "Date": selected_date,
        "City": city,
        "Temperature": average("temperature_2m"),
        "Humidity": average("relative_humidity_2m"),
        "Cloud_Coverage": average("cloud_cover"),
        "Wind_Speed": average("wind_speed_10m") / 3.6,
        "Precipitation": max(0.0, sum(float(value) for value in values("precipitation"))),
        "Pressure": average("surface_pressure") / 10,
        "Corrected_Wind_Speed": average("wind_speed_10m") / 3.6,
        "GHI": max(0.0, sum(float(value) for value in values("shortwave_radiation")) / 1000),
    }
