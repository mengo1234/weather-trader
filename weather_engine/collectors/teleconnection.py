"""Teleconnection index collector — NOAA CPC global climate indices.

Collects ONI (ENSO), NAO, AO, PNA indices. Global data (not per-city):
only the first city triggers the actual fetch.
"""
import logging
import re
from datetime import date, datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# Default NOAA CPC URLs
_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
_NAO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table"
_AO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/ao.shtml"
_PNA_URL = "https://www.cpc.ncep.noaa.gov/data/teledoc/pna.shtml"

# Season-to-month mapping for ONI (center month of 3-month window)
_SEASON_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def collect_teleconnections(db, cities: list[dict] | None = None, config=None) -> int:
    """Collect teleconnection indices from NOAA CPC.
    
    Global data — fetches once regardless of city count.
    Returns total rows inserted/updated.
    """
    total = 0
    
    oni_url = getattr(config, "oni_url", _ONI_URL) if config else _ONI_URL
    nao_url = getattr(config, "nao_url", _NAO_URL) if config else _NAO_URL
    ao_url = getattr(config, "ao_url", _AO_URL) if config else _AO_URL
    pna_url = getattr(config, "pna_url", _PNA_URL) if config else _PNA_URL
    
    fetchers = [
        ("oni", oni_url, _parse_oni),
        ("nao", nao_url, _parse_nao),
        ("ao", ao_url, _parse_ao),
        ("pna", pna_url, _parse_pna),
    ]
    
    with httpx.Client(timeout=30) as client:
        for index_name, url, parser in fetchers:
            try:
                resp = client.get(url)
                resp.raise_for_status()
                records = parser(resp.text)
                n = _store_records(db, index_name, records)
                total += n
                logger.info("Teleconnection %s: %d records stored", index_name, n)
            except Exception as e:
                logger.warning("Teleconnection %s fetch/parse failed: %s", index_name, e)
    
    return total


def _store_records(db, index_name: str, records: list[tuple[date, float]]) -> int:
    """Store parsed records into teleconnection_indices table."""
    n = 0
    now = datetime.now(timezone.utc)
    for d, value in records:
        try:
            db.execute(
                """INSERT OR REPLACE INTO teleconnection_indices
                (index_name, date, value, collected_at)
                VALUES (?, ?, ?, ?)""",
                [index_name, d, value, now],
            )
            n += 1
        except Exception as e:
            logger.debug("Failed to store %s/%s: %s", index_name, d, e)
    return n


def _parse_oni(text: str) -> list[tuple[date, float]]:
    """Parse ONI ASCII table from NOAA CPC.
    
    Format: SEAS  YEAR  TOTAL  ANOM
    Example:
    DJF   1950  24.72 -1.53
    JFM   1950  24.43 -1.34
    """
    records = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("SEAS") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            season = parts[0].upper()
            year = int(parts[1])
            anom = float(parts[3])
            month = _SEASON_MONTH.get(season)
            if month is None:
                continue
            # Use 15th of the center month as the date
            d = date(year, month, 15)
            records.append((d, anom))
        except (ValueError, IndexError):
            continue
    return records


def _parse_nao(text: str) -> list[tuple[date, float]]:
    """Parse NAO monthly table from NOAA CPC.
    
    Format: year followed by 12 monthly values.
    Example:
    1950  -0.06   0.62  -0.39  -2.02  ...
    """
    records = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
            if year < 1900 or year > 2100:
                continue
            for month_idx in range(12):
                val_str = parts[1 + month_idx]
                val = float(val_str)
                # Skip missing values (often -99.99 or similar)
                if abs(val) > 90:
                    continue
                d = date(year, month_idx + 1, 15)
                records.append((d, val))
        except (ValueError, IndexError):
            continue
    return records


def _parse_ao(html: str) -> list[tuple[date, float]]:
    """Parse AO daily index from NOAA CPC HTML page.
    
    Extracts year, month, day, AO_value from <pre> or table-like data.
    Common format in the page body:
    year month day  ao_value
    """
    records = []
    # Try to extract data from <pre> tags or plain text
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", "\n", html)
    
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            if year < 1900 or year > 2100 or month < 1 or month > 12 or day < 1 or day > 31:
                continue
            # AO value may be in column 3 or later
            val = float(parts[3]) if len(parts) > 3 else float(parts[2])
            if abs(val) > 20:  # AO typically -5 to +5
                continue
            d = date(year, month, day)
            records.append((d, val))
        except (ValueError, IndexError):
            continue
    return records


def _parse_pna(html: str) -> list[tuple[date, float]]:
    """Parse PNA monthly index from NOAA CPC HTML page.
    
    Similar structure to NAO — year + 12 monthly values embedded in HTML.
    """
    records = []
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", "\n", html)
    
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
            if year < 1900 or year > 2100:
                continue
            for month_idx in range(12):
                val_str = parts[1 + month_idx]
                val = float(val_str)
                # Skip missing/placeholder values
                if abs(val) > 90:
                    continue
                d = date(year, month_idx + 1, 15)
                records.append((d, val))
        except (ValueError, IndexError):
            continue
    return records


def get_latest_index(db, index_name: str) -> tuple[date, float] | None:
    """Get the most recent value for a teleconnection index."""
    row = db.execute(
        """SELECT date, value FROM teleconnection_indices
        WHERE index_name = ? ORDER BY date DESC LIMIT 1""",
        [index_name],
    ).fetchone()
    if row:
        return (row[0], float(row[1]))
    return None


def get_index_values(db, index_name: str, last_n_months: int = 12) -> list[tuple[date, float]]:
    """Get recent values for a teleconnection index."""
    rows = db.execute(
        """SELECT date, value FROM teleconnection_indices
        WHERE index_name = ?
        ORDER BY date DESC LIMIT ?""",
        [index_name, last_n_months],
    ).fetchall()
    return [(r[0], float(r[1])) for r in rows]
