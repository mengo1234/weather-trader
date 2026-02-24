import logging
import re
from datetime import date, datetime, timezone

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Query

from weather_engine.analysis.climate_normals import get_historical_values_for_doy
from weather_engine.analysis.probability import estimate_multiple_outcomes
from weather_engine.db import get_db, get_city
from weather_engine.market_bridge.ev_calculator import calculate_ev, kelly_criterion
from weather_engine.market_bridge.parser import parse_market_question
from weather_engine.market_bridge.predictor import predict_outcomes
from weather_engine.models.market import (
    BetImportRequest, BetRecordRequest, BetResolveRequest,
    BettingRecommendation, MarketQuery, OutcomePrediction,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["market"])

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

WEATHER_KEYWORDS_STRICT = [
    "temperature", "°f", "°c", "fahrenheit", "celsius",
    "hurricane landfall", "hurricane form", "hurricane make",
    "category 4", "category 5", "tropical storm",
    "hottest year", "record high", "record low",
    "arctic sea ice", "sea ice extent",
    "heat wave", "heatwave", "polar vortex",
    "tornado", "snowfall total", "rainfall total",
    "wildfire", "drought", "flooding",
    "natural disaster",
]
# Parole che escludono falsi positivi
WEATHER_EXCLUDE = [
    "bafta", "oscar", "rugby", "nfl", "nba", "nhl", "mlb",
    "super rugby", "rent freeze", "hiring freeze", "asset freeze",
    "political storm", "firestorm", "brainstorm",
    "ai model", "cryptocurrency", "bitcoin",
]


@router.post("/predict")
def predict_market(query: MarketQuery):
    db = get_db()

    # Parse the market question
    parsed = parse_market_question(query.question, query.outcomes)
    if parsed is None:
        raise HTTPException(400, f"Could not parse market question: {query.question}")

    city_slug = parsed["city_slug"]
    variable = parsed["variable"]
    target_date_str = parsed["target_date"]
    thresholds = parsed["thresholds"]

    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    target_date = date.fromisoformat(target_date_str)

    # Get predictions
    predictions = predict_outcomes(db, city_slug, variable, target_date, thresholds)
    if predictions is None:
        raise HTTPException(404, f"No ensemble data for {city_slug} on {target_date_str}")

    # Compute EV and recommendations for each outcome
    outcome_predictions = []
    best_bet = None
    best_edge = 0.0

    for pred, market_price in zip(predictions, query.outcome_prices):
        edge = pred.blended_prob - market_price
        confidence = 1.0 - pred.ensemble_spread / 10.0  # Normalize spread to confidence
        confidence = max(0.0, min(1.0, confidence))

        op = OutcomePrediction(
            outcome=pred.outcome,
            market_price=market_price,
            our_probability=pred.blended_prob,
            edge=round(edge, 4),
            confidence=round(confidence, 3),
        )
        outcome_predictions.append(op)

        if edge > best_edge:
            best_edge = edge
            best_bet = pred.outcome

    # Calculate Kelly for best bet
    kelly = 0.0
    ev = 0.0
    if best_bet and best_edge > 0.05:
        for op in outcome_predictions:
            if op.outcome == best_bet:
                kelly = kelly_criterion(op.our_probability, op.market_price)
                ev = calculate_ev(op.our_probability, op.market_price)
                break

    recommendation = BettingRecommendation(
        market_question=query.question,
        city=city_slug,
        variable=variable,
        date=target_date_str,
        outcomes=outcome_predictions,
        best_bet=best_bet if best_edge > 0.05 else None,
        kelly_fraction=round(kelly, 4),
        suggested_size_pct=round(min(kelly * 0.5, 0.05), 4),  # Half-Kelly, capped at 5%
        expected_value=round(ev, 4),
        reasoning=_build_reasoning(outcome_predictions, best_bet, best_edge),
    )

    # Serialize per-source probabilities for attribution
    import json
    source_probs_json = None
    try:
        source_probs_json = json.dumps([
            {
                "outcome": est.outcome,
                "ensemble_prob": est.ensemble_prob,
                "historical_prob": est.historical_prob,
                "deterministic_prob": est.deterministic_prob,
                "analog_prob": est.analog_prob,
                "bma_prob": est.bma_prob,
            }
            for est in predictions
        ])
    except Exception:
        pass

    # Log prediction
    prediction_id = None
    try:
        db.execute(
            """INSERT INTO market_predictions
            (market_id, question, city_slug, variable, target_date, predicted_at, outcomes_json, best_bet, edge, kelly_fraction, source_probs_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                query.market_id, query.question, city_slug, variable, target_date,
                datetime.now(timezone.utc),
                json.dumps([op.model_dump() for op in outcome_predictions]),
                best_bet, best_edge, kelly,
                source_probs_json,
            ],
        )
        pid_row = db.execute("SELECT currval('market_pred_seq')").fetchone()
        if pid_row:
            prediction_id = pid_row[0]
    except Exception as e:
        logger.warning("Failed to log prediction: %s", e)

    return {
        "recommendation": recommendation.model_dump(),
        "prediction_id": prediction_id,
        "metadata": {
            "city": city,
            "target_date": target_date_str,
            "variable": variable,
            "market_id": query.market_id,
        },
    }


@router.get("/scan")
def scan_markets(min_edge: float = Query(0.0, description="Filtro edge minimo")):
    """Scansiona Polymarket per mercati meteo, analizza automaticamente e restituisce raccomandazioni.

    I mercati Polymarket sono strutturati come eventi con piu' sub-mercati Yes/No
    (es. "31°F or below" Yes/No, "32-33°F" Yes/No, ecc.).
    Raggruppiamo per evento e analizziamo come mercato multi-outcome.
    """
    import json as _json

    # 1. Fetch mercati da Gamma API
    raw_markets = _fetch_gamma_weather_markets()
    if not raw_markets:
        return {"markets": [], "n_scanned": 0, "n_parseable": 0, "message": "Nessun mercato meteo trovato su Polymarket"}

    # 2. Raggruppa mercati per event_title (ogni evento = un mercato multi-outcome)
    events_map: dict[str, list[dict]] = {}
    for market in raw_markets:
        event_title = market.get("event_title", "")
        if not event_title:
            continue
        events_map.setdefault(event_title, []).append(market)

    # 3. Per ogni evento, estrai outcome e prezzi "Yes" (probabilita' per ciascun bucket)
    db = get_db()
    results = []
    n_parseable = 0

    for event_title, markets in events_map.items():
        # Estrai bucket: ogni sub-mercato ha outcome ["Yes","No"] con il prezzo "Yes"
        outcome_labels = []
        outcome_prices = []

        for m in markets:
            q = m.get("question", "")
            raw_prices = m.get("outcomePrices", [])
            if isinstance(raw_prices, str):
                try:
                    raw_prices = _json.loads(raw_prices)
                except (ValueError, TypeError):
                    raw_prices = []

            # Il prezzo "Yes" e' la probabilita' implicita del mercato per quel bucket
            yes_price = float(raw_prices[0]) if raw_prices else 0.0

            # Estrai il bucket label dalla question
            # "Will the highest temperature in NYC be between 32-33°F on Feb 20?" → "32-33°F"
            # "Will the highest temperature in NYC be 31°F or below on Feb 20?" → "31°F or below"
            label = _extract_bucket_label(q)
            if label and yes_price > 0.001:
                outcome_labels.append(label)
                outcome_prices.append(yes_price)

        if len(outcome_labels) < 2:
            continue

        # Usa il titolo dell'evento come domanda
        question = event_title

        # Prova a parsare
        parsed = parse_market_question(question, outcome_labels)
        if parsed is None:
            continue

        n_parseable += 1
        city_slug = parsed["city_slug"]
        variable = parsed["variable"]
        target_date_str = parsed["target_date"]
        thresholds = parsed["thresholds"]

        city = get_city(city_slug, db)
        if city is None:
            continue

        # Controlla se i bucket sono in °C (citta' europee/asiatiche)
        # I nostri dati sono sempre in °F, quindi convertiamo le soglie
        is_celsius = any("°C" in label for _, _, label in thresholds)
        if is_celsius:
            thresholds = [
                (_celsius_to_fahrenheit(low), _celsius_to_fahrenheit(high), label)
                for low, high, label in thresholds
            ]

        target_date = date.fromisoformat(target_date_str)

        # Genera previsioni
        predictions = predict_outcomes(db, city_slug, variable, target_date, thresholds, fast=True)
        if predictions is None:
            continue

        # Calcola EV e raccomandazioni
        outcome_predictions = []
        best_bet = None
        best_edge_val = 0.0

        for pred, market_price in zip(predictions, outcome_prices):
            edge = pred.blended_prob - market_price
            confidence = 1.0 - pred.ensemble_spread / 10.0
            confidence = max(0.0, min(1.0, confidence))

            op = OutcomePrediction(
                outcome=pred.outcome,
                market_price=market_price,
                our_probability=pred.blended_prob,
                edge=round(edge, 4),
                confidence=round(confidence, 3),
            )
            outcome_predictions.append(op)

            if edge > best_edge_val:
                best_edge_val = edge
                best_bet = pred.outcome

        # Kelly e EV per best bet
        kelly = 0.0
        ev = 0.0
        if best_bet and best_edge_val > 0.05:
            for op in outcome_predictions:
                if op.outcome == best_bet:
                    kelly = kelly_criterion(op.our_probability, op.market_price)
                    ev = calculate_ev(op.our_probability, op.market_price)
                    break

        # Filtro edge minimo
        if min_edge > 0 and best_edge_val < min_edge:
            continue

        total_volume = sum(float(m.get("volume", 0) or 0) for m in markets)
        total_liquidity = sum(float(m.get("liquidity", 0) or 0) for m in markets)

        # Record line movement snapshot for timing analysis
        try:
            from weather_engine.analysis.line_tracker import record_line_snapshot
            cond_id = markets[0].get("condition_id", "")
            if cond_id and best_bet:
                best_op = next((o for o in outcome_predictions if o.outcome == best_bet), None)
                if best_op:
                    record_line_snapshot(
                        db, cond_id, city_slug, variable, target_date,
                        our_prob=best_op.our_probability,
                        market_price=best_op.market_price,
                        edge=best_op.edge,
                        confidence=best_op.confidence * 100,
                        signal="SCAN",
                    )
        except Exception as e:
            logger.debug("Line snapshot failed for %s: %s", event_title, e)

        results.append({
            "market": {
                "question": question,
                "condition_id": markets[0].get("condition_id", ""),
                "event_title": event_title,
                "volume": total_volume,
                "liquidity": total_liquidity,
                "end_date": markets[0].get("endDateIso", markets[0].get("end_date_iso", "")),
                "n_buckets": len(outcome_labels),
            },
            "recommendation": BettingRecommendation(
                market_question=question,
                city=city_slug,
                variable=variable,
                date=target_date_str,
                outcomes=outcome_predictions,
                best_bet=best_bet if best_edge_val > 0.05 else None,
                kelly_fraction=round(kelly, 4),
                suggested_size_pct=round(min(kelly * 0.5, 0.05), 4),
                expected_value=round(ev, 4),
                reasoning=_build_reasoning(outcome_predictions, best_bet, best_edge_val),
            ).model_dump(),
            "metadata": {
                "city": city,
                "target_date": target_date_str,
                "variable": variable,
            },
        })

    # Ordina per edge decrescente
    results.sort(key=lambda r: max((o.get("edge", 0) for o in r["recommendation"]["outcomes"]), default=0), reverse=True)

    return {
        "markets": results,
        "n_scanned": len(raw_markets),
        "n_events": len(events_map),
        "n_parseable": n_parseable,
        "n_with_data": len(results),
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }


def _extract_bucket_label(question: str) -> str | None:
    """Estrai il bucket label dalla domanda Polymarket.

    Es: "Will the highest temperature in NYC be between 32-33°F on Feb 20?" → "32-33°F"
         "Will the highest temperature in NYC be 31°F or below on Feb 20?" → "31°F or below"
         "Will the highest temperature in London be between 10-11°C on Feb 20?" → "10-11°C"
    """
    q = question.strip()

    # Rileva unita' (°F o °C)
    unit = "°F"
    if "°C" in q or "ºC" in q or "°c" in q:
        unit = "°C"

    # "between X-Y°F/°C"
    m = re.search(r"be\s+between\s+(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*°?\s*[FfCc]?", q)
    if m:
        return f"{m.group(1)}-{m.group(2)}{unit}"

    # "X°F or below/above"
    m = re.search(r"be\s+(\d+(?:\.\d+)?)\s*°?\s*[FfCc]?\s+or\s+(below|above|lower|higher)", q, re.IGNORECASE)
    if m:
        return f"{m.group(1)}{unit} or {m.group(2)}"

    # "at least X"
    m = re.search(r"at\s+least\s+(\d+(?:\.\d+)?)\s*°?\s*[FfCc]?", q, re.IGNORECASE)
    if m:
        return f"{m.group(1)}{unit} or above"

    # Fallback: pattern numerico con gradi
    m = re.search(r"(\d+(?:\.\d+)?)\s*°\s*[FfCc]", q)
    if m:
        return f"{m.group(1)}{unit}"

    return None


def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


@router.get("/opportunities")
def get_opportunities():
    """Genera analisi opportunita' basate sui dati meteo reali delle nostre citta'.

    Simula mercati Polymarket per le prossime date con ensemble data disponibile.
    Utile quando Polymarket non ha mercati meteo attivi.
    """
    db = get_db()
    from weather_engine.db import get_cities

    all_cities = get_cities(db)
    results = []

    for city in all_cities[:8]:  # Top 8 citta' (include europee)
        slug = city["slug"]

        # Cerca le date con dati ensemble disponibili (solo prime 2)
        date_rows = db.execute(
            """SELECT DISTINCT time::DATE as d FROM ensemble_members
            WHERE city_slug = ? AND time::DATE >= CURRENT_DATE
            ORDER BY d LIMIT 2""",
            [slug],
        ).fetchall()

        for row in date_rows:
            target_date = row[0]

            for variable in ["temperature_2m_max"]:
                # Genera thresholds simulati basati sull'ensemble
                ens_stats = db.execute(
                    """SELECT AVG(val), STDDEV(val), MIN(val), MAX(val) FROM (
                        SELECT MAX(temperature_2m) as val
                        FROM ensemble_members
                        WHERE city_slug = ? AND time::DATE = ? AND temperature_2m IS NOT NULL
                        GROUP BY model, member_id
                    )""",
                    [slug, target_date],
                ).fetchone()

                if not ens_stats or ens_stats[0] is None:
                    continue

                mean_val = ens_stats[0]
                std_val = ens_stats[1] or 2.0
                min_val = ens_stats[2]
                max_val = ens_stats[3]

                # Crea bucket di temperatura realistici
                low_bound = round(mean_val - 2 * std_val)
                mid_low = round(mean_val - std_val)
                mid_high = round(mean_val + std_val)
                high_bound = round(mean_val + 2 * std_val)

                thresholds = [
                    (-100.0, float(low_bound), f"{low_bound:.0f}°F or below"),
                    (float(low_bound), float(mid_low), f"{low_bound:.0f}-{mid_low:.0f}°F"),
                    (float(mid_low), float(mid_high), f"{mid_low:.0f}-{mid_high:.0f}°F"),
                    (float(mid_high), float(high_bound), f"{mid_high:.0f}-{high_bound:.0f}°F"),
                    (float(high_bound), 200.0, f"{high_bound:.0f}°F or above"),
                ]

                predictions = predict_outcomes(db, slug, variable, target_date, thresholds, fast=True)
                if predictions is None:
                    continue

                # Simula prezzi di mercato (leggermente diversi dalle nostre probabilita')
                import random
                random.seed(hash(f"{slug}{target_date}{variable}"))

                outcome_predictions = []
                best_bet = None
                best_edge_val = 0.0

                for pred in predictions:
                    # Simula prezzo mercato con rumore
                    noise = random.uniform(-0.08, 0.08)
                    simulated_price = max(0.02, min(0.95, pred.blended_prob + noise))

                    edge = pred.blended_prob - simulated_price
                    confidence = 1.0 - pred.ensemble_spread / 10.0
                    confidence = max(0.0, min(1.0, confidence))

                    op = OutcomePrediction(
                        outcome=pred.outcome,
                        market_price=round(simulated_price, 3),
                        our_probability=pred.blended_prob,
                        edge=round(edge, 4),
                        confidence=round(confidence, 3),
                    )
                    outcome_predictions.append(op)

                    if edge > best_edge_val:
                        best_edge_val = edge
                        best_bet = pred.outcome

                kelly = 0.0
                ev = 0.0
                if best_bet and best_edge_val > 0.03:
                    for op in outcome_predictions:
                        if op.outcome == best_bet:
                            kelly = kelly_criterion(op.our_probability, op.market_price)
                            ev = calculate_ev(op.our_probability, op.market_price)
                            break

                results.append({
                    "market": {
                        "question": f"Highest temperature in {city['name']} on {target_date}?",
                        "condition_id": f"sim-{slug}-{target_date}",
                        "event_title": f"Weather - {city['name']}",
                        "volume": 0,
                        "liquidity": 0,
                        "end_date": str(target_date),
                        "simulated": True,
                    },
                    "recommendation": BettingRecommendation(
                        market_question=f"Highest temperature in {city['name']} on {target_date}?",
                        city=slug,
                        variable=variable,
                        date=str(target_date),
                        outcomes=outcome_predictions,
                        best_bet=best_bet if best_edge_val > 0.03 else None,
                        kelly_fraction=round(kelly, 4),
                        suggested_size_pct=round(min(kelly * 0.5, 0.05), 4),
                        expected_value=round(ev, 4),
                        reasoning=_build_reasoning(outcome_predictions, best_bet, best_edge_val),
                    ).model_dump(),
                    "metadata": {
                        "city": city,
                        "target_date": str(target_date),
                        "variable": variable,
                    },
                })

    # Ordina per edge max decrescente
    results.sort(key=lambda r: max((o.get("edge", 0) for o in r["recommendation"]["outcomes"]), default=0), reverse=True)

    return {
        "opportunities": results[:20],  # Top 20
        "n_cities": len(all_cities),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _fetch_gamma_weather_markets() -> list[dict]:
    """Fetch mercati meteo da Polymarket Gamma API.

    Strategia multipla:
    1. Cerca per tag_slug=weather (paginato)
    2. Genera slug prevedibili per mercati temperatura giornaliera delle nostre citta'
    3. Filtra e deduplicata
    """
    weather_events = []
    seen_slugs = set()

    with httpx.Client(timeout=15) as client:
        # 1. Tag search paginato
        for offset in range(0, 500, 100):
            try:
                resp = client.get(
                    f"{GAMMA_API_BASE}/events",
                    params={"limit": 100, "offset": offset, "tag_slug": "weather", "active": "true", "closed": "false"},
                )
                resp.raise_for_status()
                events = resp.json()
                if not isinstance(events, list) or not events:
                    break
                for ev in events:
                    if isinstance(ev, dict):
                        slug = ev.get("slug", "")
                        if slug not in seen_slugs:
                            seen_slugs.add(slug)
                            weather_events.append(ev)
            except Exception as e:
                logger.warning("Gamma tag search failed at offset %d: %s", offset, e)
                break

        # 2. Cerca mercati giornalieri per le nostre citta' (slug prevedibili)
        from weather_engine.db import get_cities
        db_cities = get_cities()
        city_slug_map = {
            "nyc": "nyc", "miami": "miami", "chicago": "chicago",
            "los_angeles": "los-angeles", "dallas": "dallas",
            "atlanta": "atlanta", "seattle": "seattle",
            "london": "london", "paris": "paris",
            "ankara": "ankara", "seoul": "seoul",
            "toronto": "toronto", "sao_paulo": "sao-paulo",
            "buenos_aires": "buenos-aires", "wellington": "wellington",
            "roma": "rome", "milano": "milan", "napoli": "naples",
            "berlin": "berlin", "madrid": "madrid", "amsterdam": "amsterdam",
            "cesena": "cesena", "bologna": "bologna", "vipiteno": "vipiteno",
        }

        # Genera slug per oggi e domani
        from datetime import timedelta
        today = date.today()
        for day_offset in range(0, 3):  # oggi, domani, dopodomani
            d = today + timedelta(days=day_offset)
            month_name = d.strftime("%B").lower()
            day_num = d.day
            year = d.year

            for db_slug, poly_slug in city_slug_map.items():
                event_slug = f"highest-temperature-in-{poly_slug}-on-{month_name}-{day_num}-{year}"
                if event_slug in seen_slugs:
                    continue

                try:
                    resp = client.get(f"{GAMMA_API_BASE}/events", params={"slug": event_slug})
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, list) and data:
                        for ev in data:
                            if isinstance(ev, dict):
                                seen_slugs.add(event_slug)
                                weather_events.append(ev)
                                logger.info("Found weather market: %s", ev.get("title", event_slug))
                except Exception:
                    pass

        # 3. Anche mercati temperature mensili
        for month_slug in [f"february-2026-temperature-increase-c", f"march-2026-temperature-increase-c"]:
            if month_slug not in seen_slugs:
                try:
                    resp = client.get(f"{GAMMA_API_BASE}/events", params={"slug": month_slug})
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, list) and data:
                        for ev in data:
                            if isinstance(ev, dict):
                                seen_slugs.add(month_slug)
                                weather_events.append(ev)
                except Exception:
                    pass

        # Estrai mercati dagli eventi
        all_markets = []
        seen_ids = set()

        for ev in weather_events:
            for market in ev.get("markets", []):
                if not isinstance(market, dict):
                    continue
                market_id = market.get("conditionId", "") or market.get("condition_id", "") or str(market.get("id", ""))
                if not market_id or market_id in seen_ids:
                    continue
                seen_ids.add(market_id)

                market["event_title"] = ev.get("title", "")
                market["condition_id"] = market_id
                all_markets.append(market)

    # Solo attivi non risolti
    active = [m for m in all_markets if not m.get("resolved", False) and m.get("active", True)]
    logger.info("Gamma scan: %d weather events, %d unique markets, %d active", len(weather_events), len(all_markets), len(active))
    return active


@router.post("/analyze")
def analyze_market(query: MarketQuery):
    """Analyze a market: predict outcomes + compute confidence + betting signal.

    Returns prediction, confidence scoring, and betting signal in one call.
    Frontend calls this instead of computing confidence/signal locally.
    """
    db = get_db()

    # Parse
    parsed = parse_market_question(query.question, query.outcomes)
    if parsed is None:
        raise HTTPException(400, f"Could not parse market question: {query.question}")

    city_slug = parsed["city_slug"]
    variable = parsed["variable"]
    target_date_str = parsed["target_date"]
    thresholds = parsed["thresholds"]

    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    target_date = date.fromisoformat(target_date_str)

    # Predictions
    predictions = predict_outcomes(db, city_slug, variable, target_date, thresholds)
    if predictions is None:
        raise HTTPException(404, f"No ensemble data for {city_slug} on {target_date_str}")

    # Outcome predictions + best bet
    outcome_predictions = []
    best_bet = None
    best_edge = 0.0

    for pred, market_price in zip(predictions, query.outcome_prices):
        edge = pred.blended_prob - market_price
        op = OutcomePrediction(
            outcome=pred.outcome,
            market_price=market_price,
            our_probability=pred.blended_prob,
            edge=round(edge, 4),
            confidence=round(1.0 - min(1.0, max(0.0, pred.ensemble_spread / 10.0)), 3),
        )
        outcome_predictions.append(op)
        if edge > best_edge:
            best_edge = edge
            best_bet = pred.outcome

    # Kelly + EV
    kelly = 0.0
    ev = 0.0
    if best_bet and best_edge > 0.05:
        for op in outcome_predictions:
            if op.outcome == best_bet:
                kelly = kelly_criterion(op.our_probability, op.market_price)
                ev = calculate_ev(op.our_probability, op.market_price)
                break

    # Confidence scoring (server-side)
    from weather_engine.analysis.confidence import calculate_confidence, get_seasonal_alignment
    from weather_engine.analysis.betting_signal import calculate_betting_signal
    from weather_engine.analysis.accuracy import compute_accuracy_from_db
    from weather_engine.analysis.convergence import check_convergence

    # Get ensemble data for target date
    ens_row = db.execute(
        """SELECT AVG(temperature_2m) as mean_temp,
                STDDEV(temperature_2m) as std_temp,
                COUNT(DISTINCT member_id) as n_members
        FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ? AND temperature_2m IS NOT NULL""",
        [city_slug, target_date],
    ).fetchone()

    ensemble_data = {}
    if ens_row and ens_row[0] is not None:
        ensemble_data = {
            "ensemble_mean": ens_row[0],
            "ensemble_std": ens_row[1] or 0,
            "n_members": ens_row[2],
        }

    # Accuracy
    acc_metrics = compute_accuracy_from_db(db, city_slug, variable)
    acc_data = acc_metrics.model_dump() if acc_metrics else None

    horizon_days = max(0, (target_date - date.today()).days)

    # Seasonal alignment
    forecast_temp = ensemble_data.get("ensemble_mean") if ensemble_data else None
    seasonal_align = get_seasonal_alignment(db, city_slug, forecast_temp)

    # Convergence
    conv = check_convergence(db, city_slug, target_date=target_date)
    conv_trend = conv.get("trend")

    # Cross-reference scores
    cross_ref_data = {}
    try:
        from weather_engine.analysis.cross_reference import compute_full_cross_reference
        cross_ref_data = compute_full_cross_reference(db, city_slug, target_date, horizon_days)
    except Exception as e:
        logger.debug("Cross-reference computation failed: %s", e)

    # Model reliability
    model_reliability = {}
    try:
        from weather_engine.analysis.model_tracker import get_model_reliability
        model_reliability = get_model_reliability(db, city_slug, variable, horizon_days)
    except Exception:
        pass

    # Regime info
    regime_info = None
    try:
        from weather_engine.analysis.ensemble_analysis import analyze_ensemble
        ea = analyze_ensemble(db, city_slug, target_date)
        if ea:
            regime_info = {
                "is_bimodal": ea.regime.is_bimodal,
                "n_regimes": ea.regime.n_regimes,
                "dominant_regime_weight": ea.regime.dominant_regime_weight,
                "bimodality_coefficient": ea.regime.bimodality_coefficient,
            }
    except Exception:
        pass

    # Extreme tail score
    extreme_tail_score = None
    try:
        from weather_engine.analysis.extreme_value import analyze_extremes
        if forecast_temp is not None:
            ext = analyze_extremes(db, city_slug, variable, forecast_temp)
            extreme_tail_score = ext.tail_score
    except Exception:
        pass

    # Drift status
    drift_status = None
    try:
        from weather_engine.analysis.drift import detect_drift
        drifts = detect_drift(db, city_slug, variable)
        if drifts:
            # Use worst status
            statuses = [d.status for d in drifts]
            if "alert" in statuses:
                drift_status = "alert"
            elif "degrading" in statuses:
                drift_status = "degrading"
            elif "improving" in statuses:
                drift_status = "improving"
            else:
                drift_status = "stable"
    except Exception:
        pass

    conf = calculate_confidence(
        ensemble_data, acc_data, n_outcomes=len(outcome_predictions),
        days_ahead=horizon_days, seasonal_alignment=seasonal_align,
        convergence_trend=conv_trend,
        cross_ref=cross_ref_data,
        model_reliability=model_reliability,
        regime_info=regime_info,
        extreme_tail_score=extreme_tail_score,
        drift_status=drift_status,
        db=db,
        city_slug=city_slug,
        variable=variable,
    )

    bias = acc_data.get("bias", 0) if acc_data else 0
    ens_std_val = ensemble_data.get("ensemble_std", 5)

    # Compute spread trajectory signal
    spread_boost = 0
    try:
        from weather_engine.analysis.spread_tracker import get_spread_trajectory, spread_trajectory_signal
        trajectory = get_spread_trajectory(db, city_slug, target_date, variable)
        traj_signal = spread_trajectory_signal(trajectory)
        spread_boost = traj_signal.get("signal_boost", 0)
    except Exception:
        pass

    # Get total liquidity from query outcome_prices context (default high for analyze)
    total_liquidity = getattr(query, '_liquidity', 50000.0)

    signal = calculate_betting_signal(
        conf, best_edge, bias=bias, ens_std=ens_std_val, days_ahead=horizon_days,
        convergence_trend=conv_trend,
        spread_signal_boost=spread_boost,
        liquidity=total_liquidity,
    )

    recommendation = BettingRecommendation(
        market_question=query.question,
        city=city_slug,
        variable=variable,
        date=target_date_str,
        outcomes=outcome_predictions,
        best_bet=best_bet if best_edge > 0.05 else None,
        kelly_fraction=round(kelly, 4),
        suggested_size_pct=round(min(kelly * 0.5, 0.05), 4),
        expected_value=round(ev, 4),
        reasoning=_build_reasoning(outcome_predictions, best_bet, best_edge),
    )

    return {
        "recommendation": recommendation.model_dump(),
        "confidence": conf,
        "betting_signal": signal,
        "convergence": conv,
        "cross_reference": cross_ref_data,
        "regime_info": regime_info,
        "drift_status": drift_status,
        "extreme_tail_score": extreme_tail_score,
        "metadata": {
            "city": city,
            "target_date": target_date_str,
            "variable": variable,
            "horizon_days": horizon_days,
            "source_count": cross_ref_data.get("source_count", 0),
            "conflicts": cross_ref_data.get("conflicts", []),
        },
    }


@router.post("/bets/record")
def record_bet(req: BetRecordRequest):
    """Record a new bet."""
    db = get_db()
    td = date.fromisoformat(req.target_date) if req.target_date else None
    db.execute(
        """INSERT INTO bets (timestamp, market_question, outcome, stake, odds, our_prob, edge, confidence,
           city_slug, target_date, market_prediction_id, variable, confidence_scores_json, cross_ref_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [datetime.now(timezone.utc), req.market_question, req.outcome, req.stake, req.odds,
         req.our_prob, req.edge, req.confidence, req.city_slug, td,
         req.market_prediction_id, req.variable,
         req.confidence_scores_json, req.cross_ref_json],
    )
    return {"status": "recorded"}


@router.get("/bets/list")
def list_bets(status: str | None = None, limit: int = 50):
    """List bets, optionally filtered by status."""
    db = get_db()
    if status:
        rows = db.execute(
            "SELECT * FROM bets WHERE status = ? ORDER BY timestamp DESC LIMIT ?",
            [status, limit],
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM bets ORDER BY timestamp DESC LIMIT ?",
            [limit],
        ).fetchall()

    cols = ["id", "timestamp", "market_question", "outcome", "stake", "odds", "our_prob",
            "edge", "confidence", "city_slug", "target_date", "status", "pnl", "resolved_at", "resolution_source"]
    return {"bets": [dict(zip(cols, r)) for r in rows]}


@router.post("/bets/{bet_id}/resolve")
def resolve_bet(bet_id: int, req: BetResolveRequest):
    """Manually resolve a bet."""
    db = get_db()
    bet = db.execute("SELECT stake, odds FROM bets WHERE id = ?", [bet_id]).fetchone()
    if bet is None:
        raise HTTPException(404, f"Bet {bet_id} not found")

    stake, odds = bet
    status = "won" if req.won else "lost"
    pnl = (stake * (odds - 1)) if req.won else -stake

    db.execute(
        "UPDATE bets SET status = ?, pnl = ?, resolved_at = ?, resolution_source = 'manual' WHERE id = ?",
        [status, pnl, datetime.now(timezone.utc), bet_id],
    )
    return {"status": status, "pnl": pnl}


@router.post("/bets/auto-resolve")
def auto_resolve():
    """Auto-resolve pending bets based on actual weather observations."""
    from weather_engine.market_bridge.pnl_resolver import auto_resolve_bets
    result = auto_resolve_bets()
    # Score newly resolved bets
    if result.get("resolved", 0) > 0:
        try:
            from weather_engine.analysis.scoring import score_all_unscored
            db = get_db()
            scoring = score_all_unscored(db)
            result["scoring"] = scoring
        except Exception as e:
            logger.warning("Scoring after auto-resolve failed: %s", e)
    return result


@router.post("/bets/import")
def import_bets(req: BetImportRequest):
    """Import bets from JSON (one-shot migration from local pnl_data.json)."""
    db = get_db()
    imported = 0
    for b in req.bets:
        td = date.fromisoformat(b.target_date) if b.target_date else None
        ts = b.timestamp or datetime.now(timezone.utc).isoformat()
        resolved_at = datetime.now(timezone.utc) if b.status in ("won", "lost") else None
        db.execute(
            """INSERT INTO bets (timestamp, market_question, outcome, stake, odds, our_prob,
               edge, confidence, city_slug, target_date, status, pnl, resolved_at, resolution_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [ts, b.market_question, b.outcome, b.stake, b.odds, b.our_prob,
             b.edge, b.confidence, b.city_slug, td, b.status, b.pnl,
             resolved_at, "import" if b.status != "pending" else None],
        )
        imported += 1
    return {"status": "imported", "count": imported}


@router.get("/bets/stats")
def bet_stats():
    """Get overall betting statistics."""
    from weather_engine.market_bridge.pnl_resolver import get_bet_stats
    return get_bet_stats()


@router.get("/bets/metrics")
def get_advanced_metrics():
    """Get advanced betting metrics: Sharpe, Sortino, drawdown, profit factor, etc."""
    from statistics import mean
    from weather_engine.backtesting.metrics import (
        sharpe_ratio, sortino_ratio, max_drawdown, profit_factor, calmar_ratio,
    )

    db = get_db()
    rows = db.execute(
        "SELECT stake, pnl, status, timestamp FROM bets WHERE status IN ('won', 'lost') ORDER BY timestamp ASC LIMIT 500"
    ).fetchall()

    if not rows:
        return {
            "sharpe_ratio": 0, "sortino_ratio": 0, "max_drawdown": 0,
            "profit_factor": 0, "calmar_ratio": 0,
            "avg_win": 0, "avg_loss": 0, "expectancy": 0,
            "n_trades": 0, "equity_curve": [],
        }

    returns = []
    pnls = []
    equity_curve = [1000.0]  # Start from initial bankroll
    running = 1000.0

    for stake, pnl, status, ts in rows:
        if stake and stake > 0:
            returns.append(pnl / stake)
        pnls.append(pnl or 0)
        running += (pnl or 0)
        equity_curve.append(round(running, 2))

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]

    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if equity_curve[0] > 0 else 0
    md = max_drawdown(equity_curve)

    return {
        "sharpe_ratio": round(sharpe_ratio(returns), 3),
        "sortino_ratio": round(sortino_ratio(returns), 3),
        "max_drawdown": round(md, 4),
        "profit_factor": round(profit_factor(pnls), 3),
        "calmar_ratio": round(calmar_ratio(total_return, md), 3),
        "avg_win": round(mean(wins), 4) if wins else 0,
        "avg_loss": round(mean(losses), 4) if losses else 0,
        "expectancy": round(mean(returns), 4) if returns else 0,
        "n_trades": len(rows),
        "equity_curve": equity_curve,
    }


@router.post("/backtest")
def run_backtest(
    city: str = Query("nyc", description="City slug"),
    start_date: str = Query("", description="Start date YYYY-MM-DD"),
    end_date: str = Query("", description="End date YYYY-MM-DD"),
    min_edge: float = Query(0.05, description="Minimum edge"),
    min_confidence: float = Query(0.60, description="Minimum confidence"),
    kelly_fraction: float = Query(0.25, description="Kelly multiplier"),
):
    """Run backtest with configurable parameters."""
    from weather_engine.backtesting.engine import BacktestEngine

    db = get_db()
    engine = BacktestEngine(db)

    # Default dates: last 30 days
    if not start_date:
        from datetime import timedelta
        start_date = str(date.today() - timedelta(days=30))
    if not end_date:
        end_date = str(date.today() - timedelta(days=1))

    result = engine.run(
        city_slug=city,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
        min_edge=min_edge,
        min_confidence=min_confidence,
        kelly_multiplier=kelly_fraction,
    )

    trades_list = [
        {
            "date": str(t.date),
            "city": t.city_slug,
            "outcome": t.outcome,
            "our_prob": round(t.our_prob, 3),
            "market_price": round(t.market_price, 3),
            "edge": round(t.edge, 4),
            "stake": t.stake,
            "odds": t.odds,
            "won": t.won,
            "pnl": t.pnl,
        }
        for t in result.trades
    ]

    return {
        "total_pnl": round(result.total_pnl, 2),
        "win_rate": round(result.win_rate, 3),
        "sharpe_ratio": round(result.sharpe_ratio, 3),
        "max_drawdown": round(result.max_drawdown, 4),
        "profit_factor": round(result.profit_factor, 3),
        "n_trades": result.n_trades,
        "avg_edge": round(result.avg_edge, 4),
        "equity_curve": [round(x, 2) for x in result.equity_curve],
        "drawdown_curve": [round(x, 4) for x in result.drawdown_curve],
        "trades": trades_list,
    }


@router.get("/portfolio/analyze")
def analyze_portfolio_endpoint():
    """Portfolio analysis: correlations, CVaR, optimal position sizing."""
    try:
        from weather_engine.analysis.portfolio import analyze_portfolio
        db = get_db()
        result = analyze_portfolio(db)
        return {
            "positions": [
                {
                    "bet_id": p.bet_id,
                    "city_slug": p.city_slug,
                    "variable": p.variable,
                    "target_date": p.target_date,
                    "outcome": p.outcome,
                    "probability": p.probability,
                    "edge": p.edge,
                    "kelly_fraction": p.kelly_fraction,
                    "allocated_fraction": p.allocated_fraction,
                    "allocated_amount": p.allocated_amount,
                }
                for p in result.positions
            ],
            "total_exposure": result.total_exposure,
            "portfolio_cvar": result.portfolio_cvar,
            "sharpe_estimate": result.sharpe_estimate,
            "n_positions": result.n_positions,
            "correlation_summary": result.correlation_summary,
        }
    except Exception as e:
        logger.error("Portfolio analysis failed: %s", e)
        return {
            "positions": [],
            "total_exposure": 0,
            "portfolio_cvar": 0,
            "sharpe_estimate": 0,
            "n_positions": 0,
            "correlation_summary": {},
            "error": str(e),
        }


@router.get("/bets/scores")
def get_scores(
    city: str | None = Query(None, description="Filter by city slug"),
    variable: str | None = Query(None, description="Filter by variable"),
    days: int = Query(90, description="Days to look back"),
):
    """Get aggregate prediction scores with breakdowns."""
    from weather_engine.analysis.scoring import get_aggregate_scores
    db = get_db()
    return get_aggregate_scores(db, city_slug=city, variable=variable, days_back=days)


@router.get("/calibration")
def get_calibration(variable: str = Query("temperature_2m_max", description="Weather variable")):
    """Get reliability diagram data for a variable."""
    from weather_engine.analysis.recalibration import get_reliability_data
    db = get_db()
    return get_reliability_data(db, variable=variable)


@router.get("/feedback-report")
def get_feedback_report(days: int = Query(30, description="Period in days")):
    """Get the latest feedback report, or generate one on-demand."""
    from weather_engine.analysis.feedback_report import generate_feedback_report
    db = get_db()
    period_end = date.today()
    period_start = period_end - __import__("datetime").timedelta(days=days)
    return generate_feedback_report(db, period_start, period_end)


@router.get("/source-attribution")
def get_source_attribution_endpoint(days: int = Query(90, description="Days to look back")):
    """Get per-source Brier scores and ranking."""
    from weather_engine.analysis.scoring import get_source_attribution
    db = get_db()
    return get_source_attribution(db, days_back=days)


@router.get("/system/health")
def system_health():
    """System health: circuit breaker states, data staleness per city, last collection times."""
    db = get_db()
    result = {"circuit_breakers": [], "staleness": [], "status": "ok"}

    # Circuit breaker states
    try:
        from weather_engine.resilience.circuit_breaker import get_all_states
        result["circuit_breakers"] = get_all_states(db)
        open_circuits = [c for c in result["circuit_breakers"] if c["state"] == "open"]
        if open_circuits:
            result["status"] = "degraded"
    except Exception as e:
        result["circuit_breaker_error"] = str(e)

    # Data staleness per city
    try:
        from weather_engine.resilience.staleness import check_data_staleness
        from weather_engine.db import get_cities
        cities = get_cities(db)
        for city in cities[:10]:
            try:
                staleness = check_data_staleness(db, city["slug"])
                staleness["city_slug"] = city["slug"]
                result["staleness"].append(staleness)
            except Exception:
                pass

        # Mark degraded if any city has very stale data
        for s in result["staleness"]:
            if s.get("staleness_score", 100) < 25:
                result["status"] = "degraded"
                break
    except Exception as e:
        result["staleness_error"] = str(e)

    # Last collection times
    try:
        last_collections = db.execute(
            """SELECT collector, MAX(finished_at) as last_run, COUNT(*) FILTER (WHERE status = 'error') as recent_errors
            FROM collection_log
            WHERE started_at >= NOW() - INTERVAL '24 hours'
            GROUP BY collector
            ORDER BY last_run DESC"""
        ).fetchall()
        result["last_collections"] = [
            {"collector": r[0], "last_run": str(r[1]) if r[1] else None, "recent_errors_24h": r[2]}
            for r in last_collections
        ]
    except Exception:
        result["last_collections"] = []

    return result


def _build_reasoning(outcomes: list[OutcomePrediction], best_bet: str | None, best_edge: float) -> str:
    parts = []
    for op in outcomes:
        parts.append(f"{op.outcome}: market={op.market_price:.0%} vs ours={op.our_probability:.0%} (edge={op.edge:+.1%})")

    if best_bet and best_edge > 0.05:
        parts.append(f"\nBest opportunity: {best_bet} with {best_edge:.1%} edge")
    else:
        parts.append("\nNo significant edge found (all < 5%)")

    return "\n".join(parts)
