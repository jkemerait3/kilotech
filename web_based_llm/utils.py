from pathlib import Path
import json
import openmeteo_requests
import requests_cache
from retry_requests import retry

def get_current_weather(coordinates: list[tuple[float, float]], location_names: list[str]) -> str:
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": [coord[0] for coord in coordinates],
        "longitude": [coord[1] for coord in coordinates],
        "models": "gfs_seamless",
        "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "weather_code", "wind_speed_10m"],
    }
    responses = openmeteo.weather_api(url, params=params)

    weather_summary = ""
    for i, response in enumerate(responses):
        location = location_names[i]
        # Process current data. The order of variables needs to be the same as requested.
        current = response.Current()
        current_temperature_2m = current.Variables(0).Value()
        current_relative_humidity_2m = current.Variables(1).Value()
        current_precipitation = current.Variables(2).Value()
        current_weather_code = current.Variables(3).Value()
        current_wind_speed_10m = current.Variables(4).Value()

        weather_summary += (f"At {location}, the current temperature is {current_temperature_2m}°C, "
                            f"humidity is {current_relative_humidity_2m}%, "
                            f"precipitation is {current_precipitation}mm, "
                            f"weather code is {current_weather_code}, "
                            f"and wind speed is {current_wind_speed_10m}km/h.\n")
    
    return weather_summary.strip()

# Ensure output directory exists
Path("output").mkdir(parents=True, exist_ok=True)
