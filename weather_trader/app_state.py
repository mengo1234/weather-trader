"""Application state management."""


class AppState:
    """Centralized app state: current city, section, caches."""

    def __init__(self):
        self.current_city: str = "nyc"
        self.current_section: int = 0  # 0=Dashboard, 1=Previsioni, 2=Mercati, 3=Storico, 4=Sistema, 5=Mappa, 6=Guida
        self.forecast_days: int = 7

        self.cache: dict = {
            "forecast": None,
            "metrics": None,
            "markets": None,
            "stats": None,
            "anomalies": None,
            "climate": None,
            "health": None,
            "verification": None,
        }

        # Status bar widgets — set by main_layout before section creation
        self.status_dot = None
        self.status_text = None
        self.status_hint_text = None
        self.sb_db_text = None
        self.sb_update_text = None
        self.sb_context_text = None

    def invalidate_cache(self):
        """Clear all caches (e.g. on city change)."""
        for k in self.cache:
            self.cache[k] = None
