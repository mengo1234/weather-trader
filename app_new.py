"""Weather Trader — Professional Desktop App (Flet 0.80+).

Slim entry point. All logic is in the weather_trader package.
"""

import flet as ft

from weather_trader.main_layout import main

ft.app(target=main)
