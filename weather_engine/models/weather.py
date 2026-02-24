from datetime import date, datetime

from pydantic import BaseModel


class HourlyForecast(BaseModel):
    city_slug: str
    time: datetime
    model_run: datetime
    temperature_2m: float | None = None
    relative_humidity_2m: float | None = None
    dew_point_2m: float | None = None
    apparent_temperature: float | None = None
    pressure_msl: float | None = None
    surface_pressure: float | None = None
    cloud_cover: float | None = None
    cloud_cover_low: float | None = None
    cloud_cover_mid: float | None = None
    cloud_cover_high: float | None = None
    wind_speed_10m: float | None = None
    wind_direction_10m: float | None = None
    wind_gusts_10m: float | None = None
    precipitation: float | None = None
    rain: float | None = None
    snowfall: float | None = None
    snow_depth: float | None = None
    visibility: float | None = None
    cape: float | None = None
    shortwave_radiation: float | None = None
    direct_radiation: float | None = None
    diffuse_radiation: float | None = None
    uv_index: float | None = None


class DailyForecast(BaseModel):
    city_slug: str
    date: date
    model_run: datetime
    temperature_2m_max: float | None = None
    temperature_2m_min: float | None = None
    temperature_2m_mean: float | None = None
    apparent_temperature_max: float | None = None
    apparent_temperature_min: float | None = None
    precipitation_sum: float | None = None
    rain_sum: float | None = None
    snowfall_sum: float | None = None
    precipitation_hours: float | None = None
    precipitation_probability_max: float | None = None
    wind_speed_10m_max: float | None = None
    wind_gusts_10m_max: float | None = None
    wind_direction_10m_dominant: float | None = None
    shortwave_radiation_sum: float | None = None
    uv_index_max: float | None = None
    sunrise: str | None = None
    sunset: str | None = None


class EnsembleMember(BaseModel):
    city_slug: str
    time: datetime
    model: str
    member_id: int
    temperature_2m: float | None = None
    precipitation: float | None = None
    wind_speed_10m: float | None = None
    wind_gusts_10m: float | None = None
    cloud_cover: float | None = None
    pressure_msl: float | None = None
    relative_humidity_2m: float | None = None
    dew_point_2m: float | None = None
    shortwave_radiation: float | None = None
    cape: float | None = None


class Observation(BaseModel):
    city_slug: str
    date: date
    temperature_2m_max: float | None = None
    temperature_2m_min: float | None = None
    temperature_2m_mean: float | None = None
    precipitation_sum: float | None = None
    rain_sum: float | None = None
    snowfall_sum: float | None = None
    wind_speed_10m_max: float | None = None
    wind_gusts_10m_max: float | None = None
    wind_direction_10m_dominant: float | None = None
    shortwave_radiation_sum: float | None = None
    pressure_msl_mean: float | None = None
