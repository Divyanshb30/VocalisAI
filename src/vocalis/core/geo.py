"""Airports, carriers and great-circle distance for the jurisdictions VocalisAI covers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Airport:
    iata: str
    city: str
    country: str  # ISO 3166-1 alpha-2
    lat: float
    lon: float


_AIRPORTS: dict[str, Airport] = {
    a.iata: a
    for a in [
        # India
        Airport("DEL", "Delhi", "IN", 28.5562, 77.1000),
        Airport("BOM", "Mumbai", "IN", 19.0896, 72.8656),
        Airport("BLR", "Bengaluru", "IN", 13.1986, 77.7066),
        Airport("MAA", "Chennai", "IN", 12.9941, 80.1709),
        Airport("CCU", "Kolkata", "IN", 22.6547, 88.4467),
        Airport("HYD", "Hyderabad", "IN", 17.2403, 78.4294),
        Airport("GOI", "Goa", "IN", 15.3808, 73.8314),
        Airport("COK", "Kochi", "IN", 10.1520, 76.4019),
        Airport("PNQ", "Pune", "IN", 18.5821, 73.9197),
        Airport("AMD", "Ahmedabad", "IN", 23.0772, 72.6347),
        Airport("JAI", "Jaipur", "IN", 26.8242, 75.8122),
        Airport("LKO", "Lucknow", "IN", 26.7606, 80.8893),
        Airport("IXC", "Chandigarh", "IN", 30.6735, 76.7885),
        Airport("SXR", "Srinagar", "IN", 33.9871, 74.7742),
        # UK
        Airport("LHR", "London Heathrow", "GB", 51.4700, -0.4543),
        Airport("LGW", "London Gatwick", "GB", 51.1537, -0.1821),
        Airport("MAN", "Manchester", "GB", 53.3537, -2.2750),
        Airport("EDI", "Edinburgh", "GB", 55.9500, -3.3725),
        Airport("BHX", "Birmingham", "GB", 52.4539, -1.7480),
        Airport("GLA", "Glasgow", "GB", 55.8719, -4.4331),
        # UAE
        Airport("DXB", "Dubai", "AE", 25.2532, 55.3657),
        Airport("DWC", "Dubai World Central", "AE", 24.8964, 55.1614),
        Airport("AUH", "Abu Dhabi", "AE", 24.4330, 54.6511),
        Airport("SHJ", "Sharjah", "AE", 25.3286, 55.5172),
        # EU / elsewhere (connections)
        Airport("CDG", "Paris", "FR", 49.0097, 2.5479),
        Airport("FRA", "Frankfurt", "DE", 50.0379, 8.5622),
        Airport("AMS", "Amsterdam", "NL", 52.3105, 4.7683),
        Airport("MAD", "Madrid", "ES", 40.4983, -3.5676),
        Airport("FCO", "Rome", "IT", 41.8003, 12.2389),
        Airport("DUB", "Dublin", "IE", 53.4264, -6.2499),
        Airport("JFK", "New York", "US", 40.6413, -73.7781),
        Airport("SIN", "Singapore", "SG", 1.3644, 103.9915),
        Airport("DOH", "Doha", "QA", 25.2731, 51.6081),
    ]
}

# Carrier IATA code -> (name, country of licence)
_CARRIERS: dict[str, tuple[str, str]] = {
    "6E": ("IndiGo", "IN"),
    "AI": ("Air India", "IN"),
    "IX": ("Air India Express", "IN"),
    "QP": ("Akasa Air", "IN"),
    "SG": ("SpiceJet", "IN"),
    "BA": ("British Airways", "GB"),
    "VS": ("Virgin Atlantic", "GB"),
    "U2": ("easyJet", "GB"),
    "EK": ("Emirates", "AE"),
    "FZ": ("flydubai", "AE"),
    "G9": ("Air Arabia", "AE"),
    "EY": ("Etihad Airways", "AE"),
    "LH": ("Lufthansa", "DE"),
    "AF": ("Air France", "FR"),
    "KL": ("KLM", "NL"),
    "QR": ("Qatar Airways", "QA"),
}

EU_COUNTRIES = frozenset(
    "AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split()
    # EEA + Switzerland apply EU261 as well
    + ["IS", "NO", "LI", "CH"]
)


class UnknownAirportError(KeyError):
    pass


def airport(iata: str) -> Airport:
    try:
        return _AIRPORTS[iata.upper()]
    except KeyError as exc:
        raise UnknownAirportError(iata) from exc


def carrier_name(code: str) -> str:
    return _CARRIERS.get(code.upper(), (code.upper(), ""))[0]


def carrier_country(code: str) -> str | None:
    entry = _CARRIERS.get(code.upper())
    return entry[1] if entry else None


def great_circle_km(origin: str, destination: str) -> float:
    """Haversine distance, the method EU261/UK261 Article 7(4) refers to."""
    a, b = airport(origin), airport(destination)
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * math.asin(math.sqrt(h))
