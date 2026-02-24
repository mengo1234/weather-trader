"""Guida (Tutorial/Guide) section: interactive how-to, formulas, sources, glossary."""

import flet as ft

from weather_trader.constants import (
    ACCENT, ACCENT2, BG, CARD, GREEN, ORANGE, OUTLINE_SOFT, RED, SURFACE_2, SURFACE_3, TEXT, TEXT_DIM, YELLOW,
)
from weather_trader.app_state import AppState
from weather_trader.widgets.factory import (
    make_card, make_section_title, make_verdict_banner, make_badge, pad,
)


def create_guida(page: ft.Page, state: AppState, safe_update):
    """Factory for the Guida (Tutorial/Guide) section — interactive & comprehensive."""

    # ------------------------------------------------------------------
    # expand/collapse state for accordion sections
    # ------------------------------------------------------------------
    _expanded: dict[str, bool] = {}
    _container_ref: list[ft.Container | None] = [None]

    def _rebuild():
        """Rebuild the guida content in-place and call safe_update."""
        if _container_ref[0] is not None:
            _container_ref[0].content = _build_all()
        safe_update()

    # ------------------------------------------------------------------
    # helper widgets
    # ------------------------------------------------------------------
    def _guide_step(num, title, text, icon=ft.Icons.LOOKS_ONE, color=ACCENT):
        """Un passo del tutorial."""
        return ft.Container(
            content=ft.Row([
                ft.Container(
                    content=ft.Text(str(num), size=16, color=BG, weight=ft.FontWeight.BOLD),
                    bgcolor=color, border_radius=20, width=32, height=32,
                    alignment=ft.Alignment(0, 0),
                ),
                ft.Column([
                    ft.Text(title, size=14, color=TEXT, weight=ft.FontWeight.BOLD),
                    ft.Text(text, size=11, color=TEXT_DIM),
                ], spacing=2, expand=True),
            ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START),
            padding=pad(h=12, v=10),
            bgcolor=ft.Colors.with_opacity(0.05, color),
            border=ft.Border(
                left=ft.BorderSide(3, color),
                top=ft.BorderSide(1, ft.Colors.with_opacity(0.12, color)),
                right=ft.BorderSide(1, ft.Colors.with_opacity(0.30, OUTLINE_SOFT)),
                bottom=ft.BorderSide(1, ft.Colors.with_opacity(0.30, OUTLINE_SOFT)),
            ),
            border_radius=12,
        )

    def _guide_term(term, definition, color=ACCENT):
        """Termine del glossario."""
        return ft.Container(
            content=ft.Row([
                ft.Container(
                    content=ft.Text(term, size=11, color=BG, weight=ft.FontWeight.BOLD),
                    bgcolor=color, border_radius=999, padding=pad(h=8, v=3),
                    width=140,
                ),
                ft.Text(definition, size=11, color=TEXT_DIM, expand=True),
            ], spacing=8),
            padding=pad(h=8, v=6),
            border_radius=10,
            bgcolor=ft.Colors.with_opacity(0.14, SURFACE_2),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.30, OUTLINE_SOFT)),
        )

    def _formula_box(title, formula, explanation, color=ACCENT2):
        """Box con formula statistica, codice monospaced e spiegazione."""
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.FUNCTIONS, size=16, color=color),
                    ft.Text(title, size=12, color=color, weight=ft.FontWeight.BOLD),
                ], spacing=6),
                ft.Container(
                    content=ft.Text(formula, size=13, color=YELLOW, font_family="monospace",
                                      weight=ft.FontWeight.BOLD),
                    bgcolor=ft.Colors.with_opacity(0.12, YELLOW),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.16, YELLOW)),
                    border_radius=10, padding=pad(h=12, v=8),
                ),
                ft.Text(explanation, size=10, color=TEXT_DIM, italic=True),
            ], spacing=6),
            padding=pad(h=12, v=10),
            bgcolor=ft.Colors.with_opacity(0.04, color),
            border=ft.Border(
                left=ft.BorderSide(2, color),
                top=ft.BorderSide(1, ft.Colors.with_opacity(0.10, color)),
                right=ft.BorderSide(1, ft.Colors.with_opacity(0.25, OUTLINE_SOFT)),
                bottom=ft.BorderSide(1, ft.Colors.with_opacity(0.25, OUTLINE_SOFT)),
            ),
            border_radius=12,
        )

    def _source_row(name, models, variables, freq, table, color=ACCENT):
        """Riga fonte dati con badge."""
        return ft.Container(
            content=ft.Row([
                ft.Container(
                    content=ft.Text(name, size=11, color=BG, weight=ft.FontWeight.BOLD),
                    bgcolor=color, border_radius=999, padding=pad(h=8, v=3),
                    width=120,
                ),
                ft.Column([
                    ft.Row([
                        make_badge(f"{models} modelli", ACCENT2) if models else ft.Container(width=0),
                        make_badge(f"{variables} var", ACCENT),
                        make_badge(freq, GREEN),
                    ], spacing=4, wrap=True),
                    ft.Text(f"Tabella: {table}", size=9, color=TEXT_DIM),
                ], spacing=2, expand=True),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.START),
            padding=pad(h=6, v=6),
            border_radius=10,
            bgcolor=ft.Colors.with_opacity(0.14, SURFACE_2),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.25, OUTLINE_SOFT)),
        )

    def _accordion(key, title, icon, color, content_builder):
        """Sezione espandibile/collassabile con animazione."""
        is_open = _expanded.get(key, False)

        def toggle(e):
            _expanded[key] = not _expanded.get(key, False)
            _rebuild()

        header = ft.Container(
            content=ft.Row([
                ft.Icon(icon, size=20, color=color),
                ft.Text(title, size=15, color=TEXT, weight=ft.FontWeight.BOLD, expand=True),
                ft.Container(
                    content=ft.Text(
                        "APERTO" if is_open else "CLICCA PER APRIRE",
                        size=9, color=color if is_open else TEXT_DIM,
                        weight=ft.FontWeight.BOLD,
                    ),
                    bgcolor=ft.Colors.with_opacity(0.12, color) if is_open else ft.Colors.with_opacity(0.06, TEXT_DIM),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.18, color if is_open else TEXT_DIM)),
                    border_radius=999, padding=pad(h=8, v=3),
                ),
                ft.Icon(
                    ft.Icons.KEYBOARD_ARROW_UP if is_open else ft.Icons.KEYBOARD_ARROW_DOWN,
                    size=20, color=color if is_open else TEXT_DIM,
                ),
            ], spacing=8),
            padding=pad(h=16, v=12),
            bgcolor=ft.Colors.with_opacity(0.05, color) if is_open else ft.Colors.with_opacity(0.10, SURFACE_3),
            border_radius=ft.BorderRadius(12, 12, 0 if is_open else 12, 0 if is_open else 12),
            on_click=toggle,
            ink=True,
        )

        controls = [header]
        if is_open:
            controls.append(ft.Container(
                content=content_builder(),
                padding=pad(h=16, v=12),
            ))

        return ft.Container(
            content=ft.Column(controls, spacing=0),
            bgcolor=ft.Colors.with_opacity(0.24, SURFACE_2),
            border_radius=14,
            border=ft.Border.all(
                1,
                ft.Colors.with_opacity(0.18, color if is_open else OUTLINE_SOFT),
            ),
            shadow=ft.BoxShadow(
                spread_radius=0, blur_radius=12,
                color=ft.Colors.with_opacity(0.12, "black"),
                offset=ft.Offset(0, 4),
            ),
            animate=ft.Animation(200, ft.AnimationCurve.EASE_IN_OUT),
        )

    def _stat_counter(label, value, color=ACCENT):
        """Mini stat counter per le headline."""
        return ft.Container(
            content=ft.Column([
                ft.Text(str(value), size=28, color=color, weight=ft.FontWeight.BOLD),
                ft.Text(label, size=9, color=TEXT_DIM, text_align=ft.TextAlign.CENTER),
            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=ft.Colors.with_opacity(0.08, color),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.16, color)),
            border_radius=14,
            padding=pad(h=16, v=12),
            expand=True,
        )

    # ------------------------------------------------------------------
    # content builders for each accordion section
    # ------------------------------------------------------------------
    def _build_how_it_works():
        return ft.Column([
            ft.Text(
                "Questa app analizza dati da 12 fonti diverse, 23 modelli meteorologici e 80+ variabili "
                "per trovare momenti in cui i mercati di previsione (Polymarket) hanno un prezzo sbagliato. "
                "Il motore di cross-referencing incrocia TUTTO: ensemble, deterministici, marine, alluvioni, "
                "qualita' dell'aria, trend climatici — e produce un punteggio composito che ti dice "
                "quanto possiamo fidarci della previsione.",
                size=12, color=TEXT,
            ),
            ft.Container(height=12),
            _guide_step(1, "Raccolta Dati Multi-Source",
                "Ogni 3-8 ore, il sistema interroga Open-Meteo per raccogliere dati da 15 modelli ensemble "
                "(ECMWF, GFS, ICON, GEM, BOM, CMA, MeteoFrance, NCEP, KNMI, UKMO, JMA, DWD...) "
                "e 8 modelli deterministici. In piu': marine, flood, air quality, seasonal, climate.", GREEN),
            ft.Container(height=6),
            _guide_step(2, "Cross-Referencing (10 Check)",
                "Il motore incrocia tutte le fonti con 10 analisi: "
                "accordo modelli, stabilita' atmosferica (CAPE), pattern pressione, "
                "umidita' suolo, coerenza variabili, dati marine, flood/precipitazione, "
                "trend climatici, AQI, accordo deterministici. Ogni check produce uno score 0-100.", ACCENT),
            ft.Container(height=6),
            _guide_step(3, "Probabilita' KDE a 4 Fonti",
                "La probabilita' di ogni outcome e' calcolata con Kernel Density Estimation "
                "che miscela 4 fonti: Ensemble (pesato per affidabilita' modello), Storico (25 anni), "
                "Deterministici (8 modelli pesati), Cross-Reference Adjustment. "
                "I pesi cambiano per orizzonte: vicino = ensemble domina, lontano = storico domina.", ACCENT2),
            ft.Container(height=6),
            _guide_step(4, "Confidenza a 12 Fattori",
                "Lo score di confidenza (0-100) usa 12 fattori pesati: "
                "accordo ensemble (15%), accuratezza modello (12%), decadimento temporale (8%), "
                "complessita' mercato (5%), sample size (8%), allineamento stagionale (6%), "
                "convergenza forecast (8%), accordo deterministici (10%), stabilita' atmosferica (8%), "
                "coerenza cross-variabile (7%), affidabilita' modello (8%), trend climatico (5%).", YELLOW),
            ft.Container(height=6),
            _guide_step(5, "Segnale di Betting (7 Gate)",
                "Il segnale passa 7 controlli: (1) qualita' dati sufficiente, "
                "(2) confidenza >= 35, (3) edge effettivo >= 2%, "
                "(4) orizzonte <= 7 giorni, (5) spread ensemble ragionevole, "
                "(6) liquidita' mercato >= $2000 (sotto = NON SCOMMETTERE, sotto $5000 = sizing dimezzato), "
                "(7) convergenza modelli (converging → edge +1%, diverging → penalita' confidenza). "
                "Se tutti passano → SCOMMETTI. Se qualcuno fallisce → CAUTELA o NON SCOMMETTERE.", RED),
            ft.Container(height=6),
            _guide_step(6, "Kelly Adattivo + Position Sizing",
                "L'importo e' calcolato con Kelly adattivo che scala per 3 fattori: "
                "(1) Confidenza: 80% conf = 0.8x Kelly. "
                "(2) Signal: SCOMMETTI = half-Kelly, CAUTELA = fifth-Kelly, NON SCOMMETTERE = 0. "
                "(3) Liquidita': <$2000 = 0, <$5000 = dimezzato. "
                "Floor 0.5% (sotto non vale la pena), Ceiling 8% del bankroll. "
                "Esposizione totale limitata al 50% del bankroll.", GREEN),
            ft.Container(height=6),
            _guide_step(7, "Line Movement + Timing",
                "Il sistema traccia il prezzo di mercato nel tempo (ogni 30 min durante lo scan). "
                "Analizzando l'evoluzione dell'edge produce un timing signal: "
                "SCOMMETTI_ORA = edge stabile/crescente con buona confidenza, "
                "ASPETTA = edge in rapida crescita (il mercato si muove verso di noi, attendi il picco), "
                "SBRIGATI = edge in calo (il mercato si allontana, scommetti ora o perdi l'occasione), "
                "TROPPO_TARDI = edge sotto 2% e in calo.", ACCENT2),
            ft.Container(height=6),
            _guide_step(8, "Telegram Interattivo",
                "Il bot Telegram permette di scommettere direttamente dal telefono: "
                "/bet = mostra la miglior opportunita' con sizing adattivo e timing, "
                "/bet <citta'> = opportunita' per citta' specifica, "
                "/confirm = conferma e registra la scommessa proposta (valida 5 min), "
                "/portfolio = posizioni aperte ed esposizione, "
                "/health = stato circuit breaker e freschezza dati.", ACCENT),
        ], spacing=4)

    def _build_sources():
        return ft.Column([
            ft.Text("Il sistema raccoglie dati da 12 fonti indipendenti, 23+ modelli e 80+ variabili. "
                     "Ogni fonte viene incrociata con le altre per verificare coerenza e affidabilita'.",
                     size=12, color=TEXT),
            ft.Container(height=12),

            ft.Text("MODELLI ENSEMBLE (15 modelli)", size=12, color=ACCENT, weight=ft.FontWeight.BOLD),
            ft.Text("Ogni modello genera decine di 'membri' con condizioni iniziali leggermente diverse. "
                     "La dispersione tra i membri indica l'incertezza della previsione.",
                     size=10, color=TEXT_DIM, italic=True),
            ft.Container(height=4),
            _source_row("Ensemble", "15", "15", "4-8h", "ensemble_members", ACCENT),
            ft.Container(
                content=ft.Column([
                    ft.Text("Batch 1 (ogni 4h):", size=10, color=TEXT, weight=ft.FontWeight.BOLD),
                    ft.Row([
                        make_badge("ICON", ACCENT), make_badge("GFS", ACCENT),
                        make_badge("ECMWF IFS025", ACCENT), make_badge("GEM", ACCENT),
                        make_badge("ECMWF IFS04", ACCENT),
                    ], spacing=4, wrap=True),
                    ft.Text("Batch 2 (ogni 6h):", size=10, color=TEXT, weight=ft.FontWeight.BOLD),
                    ft.Row([
                        make_badge("BOM Access", ACCENT2), make_badge("CMA GRAPES", ACCENT2),
                        make_badge("MeteoFrance", ACCENT2), make_badge("NCEP GFS025", ACCENT2),
                        make_badge("NCEP GEFS05", ACCENT2),
                    ], spacing=4, wrap=True),
                    ft.Text("Batch 3 (ogni 8h):", size=10, color=TEXT, weight=ft.FontWeight.BOLD),
                    ft.Row([
                        make_badge("DWD ICON EU", GREEN), make_badge("DWD ICON D2", GREEN),
                        make_badge("KNMI", GREEN), make_badge("UKMO 10km", GREEN),
                        make_badge("JMA GSM", GREEN),
                    ], spacing=4, wrap=True),
                    ft.Container(height=4),
                    ft.Text("Variabili: temperatura, precipitazione, vento, raffiche, "
                            "copertura nuvolosa, pressione al mare, umidita' relativa, punto di rugiada, "
                            "radiazione solare, CAPE, pressione superficiale, temperatura suolo, "
                            "umidita' suolo, visibilita', profondita' neve",
                            size=10, color=TEXT_DIM),
                ], spacing=4),
                padding=pad(h=12, v=8),
            ),

            ft.Container(height=8),
            ft.Text("MODELLI DETERMINISTICI (8 modelli)", size=12, color=ACCENT2, weight=ft.FontWeight.BOLD),
            ft.Text("A differenza degli ensemble, questi producono una singola previsione 'best guess'. "
                     "Confrontarli tra loro rivela dove c'e' accordo e dove c'e' incertezza.",
                     size=10, color=TEXT_DIM, italic=True),
            ft.Container(height=4),
            _source_row("Deterministici", "8", "11", "3h", "deterministic_forecasts", ACCENT2),
            ft.Container(
                content=ft.Row([
                    make_badge("Best Match", ACCENT2), make_badge("ECMWF", ACCENT2),
                    make_badge("GFS", ACCENT2), make_badge("ICON", ACCENT2),
                    make_badge("GEM", ACCENT2), make_badge("MeteoFrance", ACCENT2),
                    make_badge("JMA", ACCENT2), make_badge("UKMO", ACCENT2),
                ], spacing=4, wrap=True),
                padding=pad(h=12, v=6),
            ),

            ft.Container(height=8),
            ft.Text("ALTRE FONTI", size=12, color=GREEN, weight=ft.FontWeight.BOLD),
            ft.Container(height=4),
            _source_row("Forecast", "1", "25+", "1h", "forecasts_hourly/daily", GREEN),
            _source_row("Storico", None, "10", "on-demand", "observations", GREEN),
            _source_row("Climate Normals", None, "8", "on-demand", "climate_normals", GREEN),
            _source_row("Air Quality", None, "14", "3h", "air_quality", YELLOW),
            _source_row("Seasonal", None, "3", "12h", "seasonal_forecast", YELLOW),
            _source_row("Climate Proj.", "3", "5", "giornaliero", "climate_indicators", YELLOW),
            _source_row("Marine", None, "5", "6h", "marine_data", ORANGE),
            _source_row("Flood", None, "3", "6h", "flood_data", ORANGE),
            _source_row("Model Accuracy", None, "4", "giornaliero", "model_accuracy", RED),
            _source_row("Cross-Reference", None, "10 score", "3h", "cross_reference_scores", RED),

            ft.Container(height=8),
            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.VERIFIED, size=16, color=GREEN),
                    ft.Text("Citta' costiere con dati marine: NYC, Miami, Los Angeles, Seattle, London, "
                            "Napoli, San Paolo, Buenos Aires, Wellington, Amsterdam, Toronto",
                            size=10, color=TEXT_DIM),
                ], spacing=6),
                padding=pad(h=8, v=6),
                bgcolor=ft.Colors.with_opacity(0.06, GREEN),
                border_radius=6,
            ),
        ], spacing=4)

    def _build_formulas():
        return ft.Column([
            ft.Text("Queste sono le formule matematiche usate dal sistema per calcolare "
                     "probabilita', posizionamento e rischio. Non serve capirle tutte — "
                     "il sistema le calcola per te — ma aiutano a capire PERCHE' certi numeri escono.",
                     size=12, color=TEXT),
            ft.Container(height=12),

            ft.Text("PROBABILITA'", size=12, color=ACCENT, weight=ft.FontWeight.BOLD),
            ft.Container(height=6),
            _formula_box(
                "Kernel Density Estimation (KDE)",
                "f(x) = (1/nh) * SUM[ K((x - xi) / h) ]",
                "Stima la distribuzione di probabilita' dai dati ensemble. "
                "K e' il kernel gaussiano, h la bandwidth (calcolata con regola di Scott). "
                "Piu' membri dell'ensemble cadono in un range, piu' e' probabile.",
            ),
            ft.Container(height=6),
            _formula_box(
                "Blending a 4 Fonti (pesi dinamici)",
                "P = w_ens * P_ensemble + w_hist * P_historical + w_det * P_deterministic + w_xref * Adj_crossref",
                "La probabilita' finale e' una media pesata di 4 fonti. "
                "I pesi cambiano per orizzonte: "
                "0-2gg → (45%, 10%, 35%, 10%) | "
                "3-5gg → (40%, 20%, 25%, 15%) | "
                "6-8gg → (30%, 35%, 15%, 20%) | "
                "8+gg → (20%, 45%, 10%, 25%)",
            ),
            ft.Container(height=6),
            _formula_box(
                "Bandwidth Modulation (CAPE)",
                "h_adj = h_scott * (1 + (35 - stability_score) / 100)",
                "Se l'atmosfera e' instabile (CAPE alto → stability score basso), "
                "la bandwidth KDE viene allargata, producendo una distribuzione piu' incerta. "
                "CAPE < 500 J/kg = stabile, CAPE > 2000 = convettivo.",
            ),
            ft.Container(height=6),
            _formula_box(
                "Bootstrap Confidence Interval (90%)",
                "CI = [percentile(5%, P*), percentile(95%, P*)]",
                "500 campionamenti con sostituzione dall'ensemble. Per ciascuno si ricalcola "
                "il KDE e la probabilita'. I percentili 5% e 95% danno l'intervallo di confidenza.",
            ),

            ft.Container(height=12),
            ft.Text("BETTING E RISCHIO", size=12, color=GREEN, weight=ft.FontWeight.BOLD),
            ft.Container(height=6),
            _formula_box(
                "Expected Value (EV)",
                "EV = P_nostra - P_mercato",
                "Se noi stimiamo 60% e il mercato prezza 45%, l'EV e' +15%. "
                "Scommettendo ripetutamente su EV positivo, nel lungo periodo si guadagna.",
                GREEN,
            ),
            ft.Container(height=6),
            _formula_box(
                "Kelly Adattivo",
                "f = base_kelly * (conf/100) * signal_mult * liq_adj   |   0.5% <= f <= 8%",
                "Base Kelly = (P_nostra - P_mercato) / (1 - P_mercato). "
                "Scalato per: confidenza (80% conf = 0.8x), "
                "signal (SCOMMETTI = x0.50, CAUTELA = x0.20, NON_SCOMMETTERE = 0), "
                "liquidita' (<$2000 = 0, <$5000 = x0.50). "
                "Floor 0.5% (sotto non vale la pena), Ceiling 8% del bankroll.",
                GREEN,
            ),
            ft.Container(height=6),
            _formula_box(
                "Edge Effettivo (con bias correction)",
                "Edge_eff = Edge - |Bias| * 0.01",
                "L'edge grezzo viene ridotto in base al bias storico del modello. "
                "Se il modello tende a sovrastimare di 2 gradi, l'edge viene penalizzato.",
                GREEN,
            ),

            ft.Container(height=12),
            ft.Text("ACCURATEZZA E VERIFICA", size=12, color=YELLOW, weight=ft.FontWeight.BOLD),
            ft.Container(height=6),
            _formula_box(
                "MAE (Mean Absolute Error)",
                "MAE = (1/n) * SUM |forecast_i - observed_i|",
                "Errore medio assoluto in gradi. MAE 1.5 = le previsioni sbagliano in media di 1.5 gradi. "
                "Usato per calcolare il peso di ciascun modello (MAE basso = piu' peso).",
                YELLOW,
            ),
            ft.Container(height=6),
            _formula_box(
                "RMSE (Root Mean Squared Error)",
                "RMSE = SQRT( (1/n) * SUM (forecast_i - observed_i)^2 )",
                "Come il MAE ma penalizza di piu' gli errori grandi. "
                "Se MAE e RMSE sono simili, gli errori sono costanti. "
                "Se RMSE >> MAE, ci sono errori grossi occasionali.",
                YELLOW,
            ),
            ft.Container(height=6),
            _formula_box(
                "Model Reliability Score",
                "R_model = max(0.2, 1 - MAE_model / (1.2 * MAE_worst))",
                "Ogni modello riceve uno score 0-1 basato sulla sua MAE recente (30 giorni). "
                "I modelli piu' accurati pesano di piu' nel blending deterministico.",
                YELLOW,
            ),

            ft.Container(height=12),
            ft.Text("CROSS-REFERENCE", size=12, color=RED, weight=ft.FontWeight.BOLD),
            ft.Container(height=6),
            _formula_box(
                "Composite Cross-Reference Score",
                "Score = SUM(w_i * check_i)   per i = 1..10",
                "Media pesata di 10 sub-check: "
                "Model Agreement (20%), Atm. Stability (10%), Pressure (10%), "
                "Soil Moisture (5%), Cross-Variable (12%), Marine (5%), "
                "Flood-Precip (5%), Climate Trend (8%), AQI (5%), Deterministic (20%). "
                "Score > 70 = alta fiducia, < 40 = dati contrastanti.",
                RED,
            ),
            ft.Container(height=6),
            _formula_box(
                "Brier Score (Calibrazione)",
                "BS = (1/n) * SUM (P_forecast_i - O_i)^2",
                "Misura quanto le probabilita' previste sono calibrate rispetto ai risultati reali. "
                "BS = 0 e' perfetto, BS = 0.25 e' come tirare a caso. "
                "Usato per verificare che le nostre probabilita' siano affidabili.",
                RED,
            ),
        ], spacing=4)

    def _build_cross_reference():
        return ft.Column([
            ft.Text("Il motore di cross-referencing incrocia TUTTE le fonti per produrre uno score composito 0-100. "
                     "Se le fonti concordano → alta fiducia. Se si contraddicono → l'app segnala i conflitti.",
                     size=12, color=TEXT),
            ft.Container(height=12),

            _guide_step(1, "Accordo Modelli (peso: 20%)",
                "Confronta tutti i 15 modelli ensemble + 8 deterministici. "
                "Quanti concordano sulla direzione (sopra/sotto media)? "
                "100 = tutti d'accordo, 0 = meta' dicono l'opposto. "
                "Include penalita' per spread inter-modello elevato.", ACCENT),
            ft.Container(height=6),
            _guide_step(2, "Stabilita' Atmosferica (10%)",
                "Usa il CAPE (Convective Available Potential Energy) dall'ensemble. "
                "CAPE basso (<500 J/kg) = atmosfera stabile = previsione affidabile. "
                "CAPE alto (>2000 J/kg) = temporali = imprevedibile. "
                "Cross-referenzia con copertura nuvolosa e precipitazione.", ACCENT),
            ft.Container(height=6),
            _guide_step(3, "Pattern Pressione (10%)",
                "Analizza la tendenza della pressione al livello del mare. "
                "Pressione in salita = tempo stabile = piu' prevedibile. "
                "Pressione in forte discesa = perturbazione in arrivo = meno prevedibile. "
                "Confronta tra modelli la concordanza sulla tendenza barica.", ACCENT2),
            ft.Container(height=6),
            _guide_step(4, "Bias Umidita' Suolo (5%)",
                "Suolo molto umido → temperature piu' moderate, piu' precipitazione. "
                "Suolo secco + previsione fredda → pattern anomalo → meno fiducia. "
                "Cross-referenzia temperatura e umidita' del suolo dai dati ensemble.", ACCENT2),
            ft.Container(height=6),
            _guide_step(5, "Coerenza Cross-Variabile (12%)",
                "Verifica che le variabili siano internamente coerenti: "
                "Precipitazione alta con umidita' bassa? Conflitto. "
                "Neve prevista con temperatura > 4 gradi C? Conflitto. "
                "Vento forte con pressione stabile? Anomalia. "
                "Score = % di check di coerenza superati.", YELLOW),
            ft.Container(height=6),
            _guide_step(6, "Influenza Marine (5%)",
                "Solo per citta' costiere (11 citta'). "
                "Onde alte + vento forte = coerente → boost fiducia. "
                "Onde calme + vento forte previsto = conflitto → penalita'. "
                "Citta' non costiere ricevono uno score neutro (50).", YELLOW),
            ft.Container(height=6),
            _guide_step(7, "Coerenza Flood/Precipitazione (5%)",
                "Incrocia portata fiumi con precipitazione prevista. "
                "Portata alta + molta pioggia prevista = coerente → fiducia. "
                "Portata bassa + pioggia estrema = conflitto → penalita'.", ORANGE),
            ft.Container(height=6),
            _guide_step(8, "Allineamento Trend Climatico (8%)",
                "Confronta la previsione con la norma climatica (25 anni di dati). "
                "Entro 1 deviazione standard dalla norma → bonus. "
                "Oltre 2.5 deviazioni standard → previsione 'estrema' → penalita'. "
                "Verifica anche se i modelli climatici (EC-Earth, FGOALS, HiRAM) concordano.", ORANGE),
            ft.Container(height=6),
            _guide_step(9, "Correlazione AQI-Meteo (5%)",
                "AQI alto spesso indica pattern meteo estremi → piu' incertezza. "
                "Cross-referenzia PM10 con visibilita': se PM10 alto ma visibilita' buona, "
                "il dato AQI potrebbe essere sospetto → penalita'.", RED),
            ft.Container(height=6),
            _guide_step(10, "Accordo Deterministici Pesato (20%)",
                "Confronta gli 8 modelli deterministici, pesati per affidabilita' storica. "
                "Modelli con MAE basso pesano di piu'. "
                "Spread basso tra modelli pesati → alta fiducia nel consensus.", RED),
        ], spacing=4)

    def _build_glossary():
        return ft.Column([
            ft.Text("TEMPERATURE E METEO", size=12, color=ACCENT, weight=ft.FontWeight.BOLD),
            ft.Container(height=4),
            _guide_term("Temp Max/Min", "Temperatura massima e minima prevista. Dati in Fahrenheit dall'API, convertiti in Celsius nell'app."),
            _guide_term("Precipitazione", "Millimetri di pioggia previsti. 0 = sereno, 1-5 = leggera, >10 = forte, >30 = alluvionale."),
            _guide_term("Vento", "Velocita' del vento a 10m. <15 km/h = calmo, 15-40 = moderato, >50 = forte, >80 = tempesta."),
            _guide_term("CAPE", "Convective Available Potential Energy (J/kg). Misura l'energia per temporali. <500 = stabile, >2000 = convettivo."),
            _guide_term("Pressione MSL", "Pressione al livello del mare (hPa). ~1013 = normale, >1025 = alta (sereno), <1000 = bassa (tempesta)."),
            _guide_term("Soil Moisture", "Umidita' del suolo (0-1). <0.1 = secco, 0.1-0.3 = normale, >0.3 = saturo."),
            _guide_term("Snow Depth", "Profondita' neve al suolo in cm."),
            _guide_term("Visibility", "Visibilita' in metri. >10000 = ottima, 1000-5000 = moderata, <1000 = nebbia."),
            _guide_term("UV Index", "Radiazione ultravioletta. 1-2 basso, 3-5 moderato, 6-7 alto, 8+ molto alto."),
            _guide_term("Wave Height", "Altezza onde (m). Solo citta' costiere. >3m = mare agitato."),
            _guide_term("River Discharge", "Portata fiumi (m3/s). Confrontato con media storica per rischio alluvione."),

            ft.Container(height=8),
            ft.Text("MODELLI E PREVISIONI", size=12, color=ACCENT2, weight=ft.FontWeight.BOLD),
            ft.Container(height=4),
            _guide_term("Ensemble", "15 modelli meteo con decine di membri ciascuno. La dispersione indica incertezza.", ACCENT2),
            _guide_term("Deterministici", "8 modelli che danno una sola previsione 'best guess'. Confrontati per trovare il consenso.", ACCENT2),
            _guide_term("Spread", "Deviazione standard tra i membri ensemble. Basso = modelli concordano, Alto = incertezza.", ACCENT2),
            _guide_term("Bias", "Errore sistematico del modello. Bias +2 = sovrastima in media di 2 gradi. Viene corretto automaticamente.", ACCENT2),
            _guide_term("MAE", "Mean Absolute Error. Errore medio in gradi. MAE 1 = sbaglia in media di 1 grado. Piu' basso = meglio.", ACCENT2),
            _guide_term("RMSE", "Root Mean Squared Error. Come MAE ma penalizza errori grandi. Se RMSE >> MAE = outlier.", ACCENT2),
            _guide_term("KDE", "Kernel Density Estimation. Metodo statistico per stimare la distribuzione di probabilita' dai dati.", ACCENT2),
            _guide_term("Convergenza", "Se previsioni successive si avvicinano tra loro (converging = piu' affidabile) o si allontanano (diverging).", ACCENT2),
            _guide_term("Cross-Reference", "Score 0-100 che incrocia 10 check su tutte le fonti dati. Piu' alto = piu' fiducia.", ACCENT2),
            _guide_term("Climate Normal", "Valore medio storico (25 anni) per quel giorno dell'anno. Usato come riferimento.", ACCENT2),

            ft.Container(height=8),
            ft.Text("SCOMMESSE E SOLDI", size=12, color=GREEN, weight=ft.FontWeight.BOLD),
            ft.Container(height=4),
            _guide_term("Edge", "Vantaggio sul mercato. Edge = P_nostra - P_mercato. >5% = interessante, >10% = forte.", GREEN),
            _guide_term("Edge Effettivo", "Edge corretto per bias. Questo e' il numero VERO su cui basarsi.", GREEN),
            _guide_term("EV", "Expected Value. Guadagno medio per euro scommesso. EV +5% = 5 centesimi per euro.", GREEN),
            _guide_term("Confidenza", "Score 0-100 con 12 fattori pesati. Sopra 70 = FORTE, 50-70 = BUONO, 30-50 = DEBOLE, <30 = EVITA.", GREEN),
            _guide_term("Kelly (f*)", "Frazione ottimale del bankroll. Noi usiamo Kelly/4 per sicurezza.", GREEN),
            _guide_term("Bankroll", "Portafoglio totale. Inizia con $1000. Max 5% per singola scommessa.", GREEN),
            _guide_term("ROI", "Return on Investment. ROI +10% = guadagnato $10 ogni $100 scommessi.", GREEN),
            _guide_term("Win Rate", "% scommesse vinte. WR 55%+ e' buono per betting a lungo termine.", GREEN),
            _guide_term("P&L", "Profit & Loss totale. Positivo = profitto, negativo = perdita.", GREEN),
            _guide_term("Sharpe Ratio", "Rendimento medio / volatilita'. >1 = buono, >2 = eccellente. Misura il rendimento per unita' di rischio.", GREEN),
            _guide_term("Max Drawdown", "Peggior calo dal picco. DD 20% = al massimo hai perso 20% dal punto piu' alto.", GREEN),

            _guide_term("Kelly Adattivo", "Kelly che scala per confidenza, signal e liquidita'. Sostituisce il Kelly/4 fisso.", GREEN),
            _guide_term("Liquidita'", "Volume disponibile sul mercato ($). Sotto $2000 = non scommettere. Sotto $5000 = sizing dimezzato.", GREEN),
            _guide_term("Line Movement", "Evoluzione del prezzo di mercato nel tempo. Tracciato ogni 30 min durante lo scan.", GREEN),
            _guide_term("Esposizione", "Somma degli stake di tutte le scommesse aperte. Limitata al 50% del bankroll.", GREEN),

            ft.Container(height=8),
            ft.Text("SEGNALI BETTING", size=12, color=YELLOW, weight=ft.FontWeight.BOLD),
            ft.Container(height=4),
            _guide_term("SCOMMETTI", "Tutti i 7 gate passati. Dati OK, confidenza alta, edge buono, previsione vicina, modelli d'accordo, liquidita' OK.", GREEN),
            _guide_term("CAUTELA", "Opportunita' ma qualcosa non e' perfetto. Importo ridotto a Kelly * 0.20 (un quinto).", YELLOW),
            _guide_term("NON SCOMMETTERE", "Uno o piu' gate critici falliti (edge troppo basso, liquidita' insufficiente, ecc). Stai fermo.", RED),

            ft.Container(height=8),
            ft.Text("SEGNALI TIMING", size=12, color=ORANGE, weight=ft.FontWeight.BOLD),
            ft.Container(height=4),
            _guide_term("SCOMMETTI_ORA", "Edge stabile o crescente + buona confidenza. Momento giusto per piazzare.", GREEN),
            _guide_term("ASPETTA", "Edge in rapida crescita — il mercato si muove verso di noi. Aspetta il picco.", YELLOW),
            _guide_term("SBRIGATI", "Edge in calo — il mercato si allontana. Scommetti ora o perdi l'occasione.", ORANGE),
            _guide_term("TROPPO_TARDI", "Edge sotto 2% e in calo. Hai perso la finestra, non inseguire.", RED),

            ft.Container(height=8),
            ft.Text("COMANDI TELEGRAM", size=12, color=ACCENT, weight=ft.FontWeight.BOLD),
            ft.Container(height=4),
            _guide_term("/bet", "Mostra la miglior opportunita' con sizing adattivo, confidenza, signal e timing.", ACCENT),
            _guide_term("/bet <citta'>", "Opportunita' per una citta' specifica (es. /bet roma).", ACCENT),
            _guide_term("/confirm", "Conferma e registra la scommessa proposta (valida 5 minuti).", ACCENT),
            _guide_term("/portfolio", "Posizioni aperte, esposizione totale, P&L realizzato.", ACCENT),
            _guide_term("/health", "Stato circuit breaker e freschezza dati per citta'.", ACCENT),
        ], spacing=4)

    def _build_sections_guide():
        return ft.Column([
            ft.Text("DASHBOARD", size=13, color=ACCENT, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Il centro di comando. In alto: verdetto del giorno (buon giorno per scommettere?). "
                "Card grande: meteo di oggi per la citta' selezionata. 8 caselle: dati chiave. "
                "Portafoglio, migliori opportunita', anomalie meteo (z-score alto = occasione).",
                size=11, color=TEXT_DIM,
            ),
            ft.Container(height=6),

            ft.Text("PREVISIONI", size=13, color=ACCENT, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Meteo dei prossimi 7-14 giorni. Grafico barre = temperatura. "
                "Bande intorno = incertezza ensemble. Tabella dettagliata giorno per giorno. "
                "Sezione Ensemble: accordo tra i 15 modelli. "
                "Sezione Climate Normals: confronto con la media storica.",
                size=11, color=TEXT_DIM,
            ),
            ft.Container(height=6),

            ft.Text("MERCATI", size=13, color=ACCENT2, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Mercati Polymarket attivi. Ogni card: domanda, bucket, prezzo mercato, "
                "nostra stima, edge. La stella = miglior puntata. "
                "'Dettagli analisi' = confidenza (12 fattori), segnale betting, convergenza, "
                "cross-reference (10 check), e tutti i conflitti rilevati tra le fonti.",
                size=11, color=TEXT_DIM,
            ),
            ft.Container(height=6),

            ft.Text("MAPPA", size=13, color=GREEN, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Mappa interattiva con 4 layer: Temperatura, Precipitazione, Confidenza, Esposizione. "
                "Clicca su una citta' per i dettagli. I colori indicano il livello di opportunita'.",
                size=11, color=TEXT_DIM,
            ),
            ft.Container(height=6),

            ft.Text("STORICO", size=13, color=YELLOW, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Dati storici: precisione previsioni, anomalie, scommesse piazzate. "
                "ROI per Citta' = dove guadagni e dove perdi. Concentrati sulle citta' verdi. "
                "Metriche avanzate: Sharpe ratio, Sortino ratio, profit factor, equity curve.",
                size=11, color=TEXT_DIM,
            ),
            ft.Container(height=6),

            ft.Text("SISTEMA", size=13, color=RED, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Stato tecnico: server connesso? Dati aggiornati? Quanti record nel DB? "
                "Ultimo aggiornamento di ogni fonte? Se qualcosa e' rosso, i dati potrebbero "
                "non essere affidabili. Ora include anche lo stato di tutte e 12 le fonti dati.",
                size=11, color=TEXT_DIM,
            ),
            ft.Container(height=6),

            ft.Text("TELEGRAM BOT", size=13, color=ORANGE, weight=ft.FontWeight.BOLD),
            ft.Text(
                "Scommetti direttamente dal telefono! /bet mostra la miglior opportunita' "
                "con sizing adattivo (Kelly che scala per confidenza e liquidita'), "
                "timing signal (SCOMMETTI_ORA / ASPETTA / SBRIGATI / TROPPO_TARDI) "
                "e conferma con /confirm. /portfolio per vedere le posizioni aperte, "
                "/health per controllare che i dati siano freschi.",
                size=11, color=TEXT_DIM,
            ),
        ], spacing=4)

    def _build_mistakes():
        return ft.Column([
            make_verdict_banner(
                "NON scommettere su mercati con meno di $2000 di liquidita'. "
                "Il sistema lo blocca automaticamente (Gate 6). Sotto $5000 il sizing viene dimezzato. "
                "Mercati sottili = slippage, manipolazione, impossibilita' di uscire.",
                RED, ft.Icons.BLOCK),
            ft.Container(height=6),
            make_verdict_banner(
                "NON scommettere su mercati a piu' di 7 giorni. "
                "Le previsioni meteo oltre una settimana sono quasi inutili. "
                "La confidenza decade esponenzialmente: giorno 1 = 100, giorno 7 = 45, giorno 10+ = 25.",
                RED, ft.Icons.BLOCK),
            ft.Container(height=6),
            make_verdict_banner(
                "NON inseguire le perdite. Se perdi 3 scommesse di fila, fermati. "
                "Non aumentare le puntate per recuperare — la matematica ti frega.",
                RED, ft.Icons.BLOCK),
            ft.Container(height=6),
            make_verdict_banner(
                "NON ignorare il segnale. Se dice NON SCOMMETTERE, non scommettere. "
                "Anche se l'edge sembra buono, c'e' un motivo preciso per cui 1+ dei 5 gate e' fallito.",
                RED, ft.Icons.BLOCK),
            ft.Container(height=6),
            make_verdict_banner(
                "NON scommettere piu' del suggerito. Il Kelly frazionario (f*/4) esiste per proteggerti. "
                "Il Kelly pieno e' troppo aggressivo — un Kelly/4 ha lo stesso rendimento a lungo termine "
                "con meta' della varianza.",
                ORANGE, ft.Icons.WARNING_AMBER),
            ft.Container(height=6),
            make_verdict_banner(
                "NON fidarti di una singola fonte. Il sistema usa 12 fonti per un motivo. "
                "Se il cross-reference score e' basso (<40) o ci sono conflitti, "
                "significa che le fonti si contraddicono. Stai fermo.",
                ORANGE, ft.Icons.WARNING_AMBER),
            ft.Container(height=6),
            make_verdict_banner(
                "NON scommettere su citta' dove lo spread ensemble e' rosso (alto). "
                "15 modelli non concordano = la nostra previsione potrebbe essere sbagliata.",
                ORANGE, ft.Icons.WARNING_AMBER),
            ft.Container(height=6),
            make_verdict_banner(
                "SI' punta sulle anomalie! Quando il meteo fa qualcosa di insolito (z-score > 2), "
                "i mercati spesso non reagiscono in tempo. Il sistema con 23 modelli lo vede prima.",
                GREEN, ft.Icons.THUMB_UP),
            ft.Container(height=6),
            make_verdict_banner(
                "SI' sfrutta i conflitti! Quando il cross-reference segnala un conflitto "
                "(es. flood alto + precipitazione bassa), spesso il mercato sta prezzando solo una fonte. "
                "Se tu hai 12 fonti, hai un vantaggio informativo.",
                GREEN, ft.Icons.THUMB_UP),
        ], spacing=4)

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def _build_all():
        """Inner build that returns the full column content."""
        return ft.Column([
            make_section_title("Guida Interattiva", ft.Icons.SCHOOL, YELLOW),
            ft.Container(height=4),

            # --- headline stats ---
            ft.ResponsiveRow([
                ft.Column([_stat_counter("Fonti Dati", "12", ACCENT)], col={"xs": 6, "sm": 4, "md": 2}),
                ft.Column([_stat_counter("Modelli Meteo", "23", ACCENT2)], col={"xs": 6, "sm": 4, "md": 2}),
                ft.Column([_stat_counter("Variabili", "80+", GREEN)], col={"xs": 6, "sm": 4, "md": 2}),
                ft.Column([_stat_counter("Gate Signal", "7", YELLOW)], col={"xs": 6, "sm": 4, "md": 2}),
                ft.Column([_stat_counter("Check Cross-Ref", "10", RED)], col={"xs": 6, "sm": 4, "md": 2}),
                ft.Column([_stat_counter("Comandi TG", "5", ORANGE)], col={"xs": 6, "sm": 4, "md": 2}),
            ], spacing=8, run_spacing=8),

            ft.Container(height=12),

            # --- accordion sections ---
            _accordion("how", "Come Funziona (8 Step)",
                        ft.Icons.PLAY_CIRCLE, GREEN, _build_how_it_works),
            ft.Container(height=6),
            _accordion("sources", "Fonti Dati (12 Fonti, 23 Modelli, 80+ Variabili)",
                        ft.Icons.STORAGE, ACCENT, _build_sources),
            ft.Container(height=6),
            _accordion("formulas", "Formule Statistiche (KDE, Kelly, MAE, RMSE, Brier...)",
                        ft.Icons.FUNCTIONS, ACCENT2, _build_formulas),
            ft.Container(height=6),
            _accordion("crossref", "Motore Cross-Reference (10 Check Incrociati)",
                        ft.Icons.COMPARE_ARROWS, RED, _build_cross_reference),
            ft.Container(height=6),
            _accordion("glossary", "Glossario Completo (55+ Termini)",
                        ft.Icons.MENU_BOOK, ACCENT, _build_glossary),
            ft.Container(height=6),
            _accordion("sections", "Come Leggere Ogni Sezione dell'App",
                        ft.Icons.MAP, ACCENT2, _build_sections_guide),
            ft.Container(height=6),
            _accordion("mistakes", "Errori da NON Fare (9 Regole d'Oro)",
                        ft.Icons.ERROR_OUTLINE, RED, _build_mistakes),

            ft.Container(height=20),
        ], scroll=ft.ScrollMode.AUTO, spacing=0)

    def build():
        container = ft.Container(content=_build_all(), expand=True, padding=pad(h=4, v=4))
        _container_ref[0] = container
        return container

    # ------------------------------------------------------------------
    # load (static content, nothing to fetch)
    # ------------------------------------------------------------------
    def load():
        """La guida e' statica, non serve caricare nulla."""
        pass

    return build, load
