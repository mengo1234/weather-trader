"""Constants: colors, city mappings, utility conversions."""

# Colori tema scuro (restyling: navy + cyan + amber)
BG = "#080b14"
CARD = "#111827"
CARD_HOVER = "#182235"
ACCENT = "#25d0ff"
ACCENT2 = "#7c8cff"
GREEN = "#2ee6a6"
RED = "#ff6b7a"
YELLOW = "#ffd166"
ORANGE = "#ff9f43"
TEXT = "#eef3ff"
TEXT_DIM = "#9ba8c7"
UV_COLORS = ["#4ade80", "#fde047", "#fb923c", "#f87171", "#c084fc"]
SURFACE_1 = "#101827"
SURFACE_2 = "#162033"
SURFACE_3 = "#1b2740"
OUTLINE = "#2a3756"
OUTLINE_SOFT = "#22304d"

# Typography tokens (global type scale)
FONT_UI = "Segoe UI"
TYPE_2XS = 9
TYPE_XS = 10
TYPE_SM = 11
TYPE_MD = 12
TYPE_LG = 14
TYPE_XL = 20
TYPE_2XL = 22
TYPE_DISPLAY = 54
TYPE_DISPLAY_COMPACT = 48
TYPE_METRIC = 18
TYPE_METRIC_LG = 23

API_BASE = "http://localhost:8321"

# Città con mercati REALI su Polymarket (ordinate: italiane, europee, mondo)
CITY_SLUGS = [
    # Italiane (previsioni reali, mercati simulati/test)
    "roma", "milano", "napoli", "bologna", "cesena", "vipiteno",
    # Europee (con mercati reali Polymarket)
    "london", "paris", "ankara", "berlin", "madrid", "amsterdam",
    # USA (con mercati reali Polymarket)
    "nyc", "miami", "chicago", "los_angeles", "dallas", "atlanta", "seattle",
    # Mondo (con mercati reali Polymarket)
    "seoul", "toronto", "sao_paulo", "buenos_aires", "wellington",
]
CITY_NAMES = [
    "Roma", "Milano", "Napoli", "Bologna", "Cesena", "Vipiteno",
    "Londra", "Parigi", "Ankara", "Berlino", "Madrid", "Amsterdam",
    "New York", "Miami", "Chicago", "Los Angeles", "Dallas", "Atlanta", "Seattle",
    "Seoul", "Toronto", "San Paolo", "Buenos Aires", "Wellington",
]
CITY_MAP = dict(zip(CITY_SLUGS, CITY_NAMES))


# Betting config
INITIAL_BANKROLL = 1000.0
MAX_BET_PCT = 0.05           # Max 5% of bankroll per bet
KELLY_FRACTION = 0.25        # Kelly/4 (conservative)
EDGE_GOOD_THRESHOLD = 0.05   # 5%+ edge = green
SIGNAL_BET_THRESHOLD = 0.6   # Signal > 60% → SCOMMETTI
SIGNAL_CAUTION_THRESHOLD = 0.3  # Signal > 30% → CAUTELA


def f2c(f):
    """Fahrenheit → Celsius."""
    return (f - 32) * 5 / 9
