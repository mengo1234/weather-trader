<div align="center">

<img src="icon-512.png" width="128" alt="Weather Trader icon">

# Weather Trader

**Forecast intelligence meets prediction market dashboard**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776ab?logo=python&logoColor=white)](https://python.org)
[![Flet 0.80](https://img.shields.io/badge/Flet-0.80-02569B?logo=flutter&logoColor=white)](https://flet.dev)
[![Material 3](https://img.shields.io/badge/Material-3-6750a4?logo=materialdesign&logoColor=white)](https://m3.material.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

<div align="center">

### Download

[![Windows](https://img.shields.io/badge/Windows-Download-0078D4?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/mengo1234/weather-trader/releases/latest/download/WeatherTrader-windows.exe)
[![macOS](https://img.shields.io/badge/macOS-Download-000000?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/mengo1234/weather-trader/releases/latest/download/WeatherTrader-macos.dmg)
[![Linux](https://img.shields.io/badge/Linux-Download-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://github.com/mengo1234/weather-trader/releases/latest/download/weather-trader-linux)

</div>

---

## Features

| Section | Description |
|---------|-------------|
| **Dashboard** | Real-time overview with KPI cards, portfolio equity chart, and live market signals |
| **Previsioni** | Multi-model ensemble forecasts (GFS, ECMWF, ICON) with confidence intervals and accuracy tracking |
| **Mercati** | Polymarket integration — live odds, edge detection, Kelly-sized bet suggestions |
| **Mappa** | Interactive weather map powered by Flet Map with multi-layer overlays |
| **Storico** | Full trade history, P&L curves, drawdown analysis, and win-rate statistics |
| **Sistema** | Backend health monitoring, API latency, data pipeline status |
| **Guida** | Built-in user guide with searchable documentation |

## Highlights

- **24 cities** tracked worldwide — Italy, Europe, USA, and global
- **Polymarket** real-time odds integration for weather prediction markets
- **Ensemble forecasting** combining GFS, ECMWF, ICON models
- **Kelly criterion** position sizing (conservative Kelly/4) with bankroll management
- **Edge detection** — automatic identification of mispriced weather markets
- **P&L tracking** with equity curves, drawdown stats, and trade journal
- **Interactive map** with temperature, precipitation, and wind overlays
- **Dark theme** with navy + cyan + amber Material 3 design system
- **Cross-platform** — runs natively on Windows, macOS, and Linux

## Screenshots

<div align="center">

| Dashboard | Previsioni | Mercati |
|:---------:|:----------:|:-------:|
| ![Dashboard](docs/screenshots/dashboard.png) | ![Previsioni](docs/screenshots/previsioni.png) | ![Mercati](docs/screenshots/mercati.png) |

| Mappa | Storico | Sistema |
|:-----:|:-------:|:-------:|
| ![Mappa](docs/screenshots/mappa.png) | ![Storico](docs/screenshots/storico.png) | ![Sistema](docs/screenshots/sistema.png) |

</div>

## Download

| Platform | Download | Notes |
|----------|----------|-------|
| **Windows** | [WeatherTrader-windows.exe](https://github.com/mengo1234/weather-trader/releases/latest/download/WeatherTrader-windows.exe) | Windows 10+ |
| **macOS** | [WeatherTrader-macos.dmg](https://github.com/mengo1234/weather-trader/releases/latest/download/WeatherTrader-macos.dmg) | macOS 12+ (Intel & Apple Silicon) |
| **Linux** | [weather-trader-linux](https://github.com/mengo1234/weather-trader/releases/latest/download/weather-trader-linux) | Ubuntu 22.04+, Fedora 38+ |

## Quick Start

Run from source:

```bash
git clone https://github.com/mengo1234/weather-trader.git
cd weather-trader
uv sync
uv run python app_new.py
```

## Backend Setup

Weather Trader requires the **weather-engine** backend running on `localhost:8321`.

The backend provides forecast data, market odds, and trade execution. Start it before launching the desktop app:

```bash
cd ../weather-engine
uv run uvicorn weather_engine.main:app --host 0.0.0.0 --port 8321
```

Or use the included launcher script that starts both:

```bash
./launch.sh
```

## Architecture

```
weather-trader/
├── app_new.py                  # Entry point
├── pyproject.toml              # Project config & dependencies
├── launch.sh                   # Backend + app launcher
├── icon-*.png                  # App icons (48/128/256/512)
├── icon.svg                    # Vector source icon
└── weather_trader/             # Main package
    ├── __init__.py
    ├── main_layout.py          # Layout, navigation, status bar
    ├── api_client.py           # HTTP client (httpx → backend)
    ├── app_state.py            # Global application state
    ├── constants.py            # Colors, cities, config
    ├── logging_config.py       # Logging setup
    ├── logic/
    │   ├── pnl_tracker.py      # P&L tracking & persistence
    │   └── risk_manager.py     # Kelly criterion & bankroll mgmt
    ├── sections/
    │   ├── dashboard.py        # Dashboard KPIs & charts
    │   ├── previsioni.py       # Ensemble forecast view
    │   ├── mercati.py          # Market odds & bet signals
    │   ├── mappa.py            # Interactive weather map
    │   ├── storico.py          # Trade history & analytics
    │   ├── sistema.py          # System health monitoring
    │   └── guida.py            # Built-in user guide
    └── widgets/
        ├── charts.py           # Chart components
        ├── confidence.py       # Confidence interval display
        ├── distribution.py     # Distribution visualization
        ├── factory.py          # Widget factory & helpers
        ├── pnl_widgets.py      # P&L display widgets
        └── weather_map.py      # Map tile layer & markers
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI Framework | [Flet](https://flet.dev) 0.80 (Flutter-based Python UI) |
| Design System | Material 3 — dark theme |
| HTTP Client | [httpx](https://www.python-httpx.org) (async) |
| Maps | [flet-map](https://pypi.org/project/flet-map) (OpenStreetMap tiles) |
| Backend | weather-engine (FastAPI + Uvicorn) |
| Forecasts | GFS, ECMWF, ICON ensemble |
| Markets | Polymarket API |
| Packaging | [uv](https://docs.astral.sh/uv) + PyInstaller |
| CI/CD | GitHub Actions (auto-build on tag) |

## License

[MIT](LICENSE) — see [LICENSE](LICENSE) for details.
