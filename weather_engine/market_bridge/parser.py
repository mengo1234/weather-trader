"""Parse Polymarket weather questions into structured queries."""
import logging
import re
from datetime import date, datetime, timedelta

logger = logging.getLogger(__name__)

# City aliases → slugs
CITY_ALIASES = {
    "new york": "nyc", "nyc": "nyc", "new york city": "nyc", "manhattan": "nyc",
    "miami": "miami", "miami beach": "miami",
    "chicago": "chicago",
    "los angeles": "los_angeles", "la": "los_angeles",
    "dallas": "dallas",
    "atlanta": "atlanta",
    "seattle": "seattle",
    "london": "london",
    "paris": "paris", "parigi": "paris",
    "ankara": "ankara",
    "seoul": "seoul",
    "toronto": "toronto",
    "sao paulo": "sao_paulo", "são paulo": "sao_paulo", "san paolo": "sao_paulo",
    "buenos aires": "buenos_aires",
    "wellington": "wellington",
    "roma": "roma", "rome": "roma",
    "milano": "milano", "milan": "milano",
    "napoli": "napoli", "naples": "napoli",
    "berlin": "berlin", "berlino": "berlin",
    "madrid": "madrid",
    "amsterdam": "amsterdam",
    "cesena": "cesena",
    "bologna": "bologna",
    "vipiteno": "vipiteno", "sterzing": "vipiteno",
}

# Variable detection patterns
VARIABLE_PATTERNS = {
    "temperature_2m_max": [
        r"(?:high|highest|max|maximum)\s*(?:temp|temperature)",
        r"temperature.*(?:high|max|reach|exceed|above|over)",
        r"how (?:hot|warm)",
        r"temp(?:erature)?.*(?:above|over|exceed|reach)",
    ],
    "temperature_2m_min": [
        r"(?:low|lowest|min|minimum)\s*(?:temp|temperature)",
        r"temperature.*(?:low|min|drop|below|under)",
        r"how (?:cold|cool)",
    ],
    "precipitation_sum": [
        r"(?:rain|rainfall|precipitation|precip)",
        r"(?:snow|snowfall)",
        r"(?:inches|mm)\s*of\s*(?:rain|snow|precip)",
    ],
    "wind_speed_10m_max": [
        r"(?:wind|winds|gusts?)\s*(?:speed)?",
        r"(?:mph|km/h|knots)",
    ],
    "snow_depth": [
        r"snow\s*depth",
        r"snow\s*accumulation",
        r"(?:inches|cm)\s*of\s*snow\s*on\s*ground",
    ],
    "visibility": [
        r"visibility",
        r"(?:fog|foggy|mist)",
    ],
    "pressure_msl": [
        r"(?:pressure|barometric|barometer)",
        r"(?:hpa|mbar|millibar)",
    ],
}

# Day-of-week mapping for relative date parsing
_WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

# Date parsing patterns (order matters: more specific first)
DATE_PATTERNS = [
    (r"(\d{4}-\d{2}-\d{2})", "iso"),
    (r"on\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)", "month_day"),
    (r"(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)", "month_day"),
    (r"(today|tomorrow|day\s+after\s+tomorrow)", "relative"),
    (r"(?:this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", "weekday"),
    (r"(?:end\s+of|this)\s+(week|month)", "relative_period"),
]


def parse_market_question(question: str, outcomes: list[str]) -> dict | None:
    """Parse a Polymarket question into structured weather query.

    Returns: {city_slug, variable, target_date, thresholds: [(low, high, label), ...]}
    """
    q_lower = question.lower()

    # Detect city
    city_slug = _detect_city(q_lower)
    if city_slug is None:
        logger.warning("Could not detect city in: %s", question)
        return None

    # Detect variable
    variable = _detect_variable(q_lower)

    # Detect date
    target_date = _detect_date(q_lower)
    if target_date is None:
        target_date = str(date.today())

    # Parse outcome thresholds
    thresholds = _parse_outcome_thresholds(outcomes, variable)

    return {
        "city_slug": city_slug,
        "variable": variable,
        "target_date": target_date,
        "thresholds": thresholds,
    }


def _detect_city(text: str) -> str | None:
    # Sort by length descending so "new york city" matches before "new york"
    for alias in sorted(CITY_ALIASES.keys(), key=len, reverse=True):
        # Use word boundary to avoid matching "la" inside "Atlantis"
        if re.search(r"\b" + re.escape(alias) + r"\b", text):
            return CITY_ALIASES[alias]
    return None


def _detect_variable(text: str) -> str:
    for variable, patterns in VARIABLE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return variable
    # Default to max temperature
    return "temperature_2m_max"


def _detect_date(text: str) -> str | None:
    for pattern, fmt in DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1)
            if fmt == "iso":
                return raw
            elif fmt == "relative":
                today = date.today()
                raw_l = raw.lower()
                if "day after tomorrow" in raw_l:
                    return str(today + timedelta(days=2))
                elif "tomorrow" in raw_l:
                    return str(today + timedelta(days=1))
                return str(today)
            elif fmt == "weekday":
                return _parse_next_weekday(raw.lower())
            elif fmt == "relative_period":
                return _parse_relative_period(raw.lower())
            elif fmt == "month_day":
                return _parse_month_day(raw)
    return None


def _parse_next_weekday(day_name: str) -> str:
    """Parse 'next Monday', 'this Friday' etc."""
    today = date.today()
    target_wd = _WEEKDAY_MAP.get(day_name)
    if target_wd is None:
        return str(today)
    days_ahead = target_wd - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return str(today + timedelta(days=days_ahead))


def _parse_relative_period(period: str) -> str:
    """Parse 'end of week', 'end of month', 'this week'."""
    today = date.today()
    if "month" in period:
        # End of current month
        if today.month == 12:
            return str(date(today.year + 1, 1, 1) - timedelta(days=1))
        return str(date(today.year, today.month + 1, 1) - timedelta(days=1))
    else:
        # End of week (Sunday)
        days_to_sunday = 6 - today.weekday()
        return str(today + timedelta(days=days_to_sunday))


def _parse_month_day(text: str) -> str | None:
    # Remove ordinal suffixes
    cleaned = re.sub(r"(\d+)(?:st|nd|rd|th)", r"\1", text)
    try:
        for fmt in ["%B %d, %Y", "%B %d %Y", "%B %d", "%b %d, %Y", "%b %d %Y", "%b %d"]:
            try:
                dt = datetime.strptime(cleaned.strip(), fmt)
                if dt.year < 2000:
                    dt = dt.replace(year=date.today().year)
                return str(dt.date())
            except ValueError:
                continue
    except Exception:
        pass
    return None


def _parse_outcome_thresholds(outcomes: list[str], variable: str) -> list[tuple[float, float, str]]:
    """Parse outcome labels like '32-33F', '31F or below', '36F or above' into (low, high, label)."""
    thresholds = []

    for outcome in outcomes:
        low, high = _extract_range(outcome, variable)
        thresholds.append((low, high, outcome))

    return thresholds


def _is_celsius(outcome: str) -> bool:
    """Check if outcome uses Celsius."""
    return bool(re.search(r"°\s*[Cc]|celsius", outcome, re.IGNORECASE))


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _extract_range(outcome: str, variable: str) -> tuple[float, float]:
    """Extract numeric range from outcome text. Converts Celsius to Fahrenheit if detected."""
    outcome_lower = outcome.lower().strip()
    celsius = _is_celsius(outcome)

    # "X or below" / "X or less" / "under X" / "below X"
    match = re.search(r"(\d+(?:\.\d+)?)\s*[°fFcC]*\s*or\s*(?:below|less|under|lower)", outcome_lower)
    if match:
        high = float(match.group(1))
        return (-100.0, _c_to_f(high) if celsius else high)

    match = re.search(r"(?:below|under|less than)\s*(\d+(?:\.\d+)?)", outcome_lower)
    if match:
        high = float(match.group(1))
        return (-100.0, _c_to_f(high) if celsius else high)

    # "X or above" / "X or more" / "over X" / "above X"
    match = re.search(r"(\d+(?:\.\d+)?)\s*[°fFcC]*\s*or\s*(?:above|more|over|higher)", outcome_lower)
    if match:
        low = float(match.group(1))
        return (_c_to_f(low) if celsius else low, 200.0)

    match = re.search(r"(?:above|over|more than|exceeds?)\s*(\d+(?:\.\d+)?)", outcome_lower)
    if match:
        low = float(match.group(1))
        return (_c_to_f(low) if celsius else low, 200.0)

    # "X-Y" range (e.g., "32-33F", "32°F - 33°F", "10-11°C")
    match = re.search(r"(\d+(?:\.\d+)?)\s*[°fFcC]*\s*[-–—to]+\s*(\d+(?:\.\d+)?)", outcome_lower)
    if match:
        lo, hi = float(match.group(1)), float(match.group(2))
        if celsius:
            return (_c_to_f(lo), _c_to_f(hi))
        return (lo, hi)

    # Single number (exact match, +/- 0.5)
    match = re.search(r"(\d+(?:\.\d+)?)", outcome_lower)
    if match:
        val = float(match.group(1))
        if celsius:
            val = _c_to_f(val)
        return (val - 0.5, val + 0.5)

    # Fallback
    return (-100.0, 200.0)
