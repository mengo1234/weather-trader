from pydantic import BaseModel


class City(BaseModel):
    slug: str
    name: str
    latitude: float
    longitude: float
    timezone: str
    country: str
