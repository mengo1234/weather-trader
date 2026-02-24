"""Cross-reference engine: incrocia TUTTE le fonti dati per scoring composito.

10 sub-funzioni + orchestratore che produce un composite score 0-100
e identifica conflitti tra fonti.
"""
import logging
from datetime import date, timedelta

import numpy as np

logger = logging.getLogger(__name__)


def compute_full_cross_reference(db, city_slug: str, target_date: date, days_ahead: int = 1) -> dict:
    """Orchestratore: chiama tutti i check e aggrega."""
    results = {}
    source_count = 0
    conflicts = []

    checks = [
        ("model_agreement", compute_model_agreement),
        ("atmospheric_stability", compute_atmospheric_stability),
        ("pressure_patterns", analyze_pressure_patterns),
        ("soil_moisture_bias", compute_soil_moisture_bias),
        ("cross_variable_consistency", check_cross_variable_consistency),
        ("marine_influence", compute_marine_confidence),
        ("flood_precip_consistency", check_flood_precip_consistency),
        ("climate_trend_alignment", check_climate_trend_alignment),
        ("aqi_weather_correlation", compute_aqi_uncertainty),
        ("deterministic_agreement", compute_deterministic_agreement),
        ("teleconnection_alignment", compute_teleconnection_alignment),
        ("ensemble_regime_score", compute_ensemble_regime_score),
        ("extreme_value_score", compute_extreme_value_score),
    ]

    for key, func in checks:
        try:
            result = func(db, city_slug, target_date)
            score = result.get("score", 50.0)
            results[key] = score
            if result.get("has_data", False):
                source_count += 1
            if result.get("conflict"):
                conflicts.append(result["conflict"])
        except Exception as e:
            logger.debug("Cross-ref %s failed for %s/%s: %s", key, city_slug, target_date, e)
            results[key] = 50.0  # neutral on failure

    # Composite: weighted average of all 13 sub-scores
    weights = {
        "model_agreement": 0.16,
        "atmospheric_stability": 0.08,
        "pressure_patterns": 0.08,
        "soil_moisture_bias": 0.04,
        "cross_variable_consistency": 0.10,
        "marine_influence": 0.04,
        "flood_precip_consistency": 0.04,
        "climate_trend_alignment": 0.06,
        "aqi_weather_correlation": 0.04,
        "deterministic_agreement": 0.16,
        "teleconnection_alignment": 0.08,
        "ensemble_regime_score": 0.06,
        "extreme_value_score": 0.06,
    }

    # Try learned weights
    try:
        from weather_engine.analysis.weight_learner import load_weights
        learned = load_weights(db, "cross_ref", "all")
        if learned and set(learned.keys()) == set(weights.keys()):
            weights = learned
    except Exception:
        pass

    composite = sum(results[k] * weights[k] for k in results)
    composite = round(min(100, max(0, composite)), 1)

    return {
        **results,
        "composite_score": composite,
        "source_count": source_count,
        "conflicts": conflicts,
    }


def compute_model_agreement(db, city_slug: str, target_date: date) -> dict:
    """Compare 15 ensemble models + 8 deterministic models.

    Score: 100 if all agree on direction, 0 if half say opposite.
    """
    # Ensemble model means
    ens_rows = db.execute(
        """SELECT model, AVG(temperature_2m) as mean_temp
        FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ? AND temperature_2m IS NOT NULL
        GROUP BY model""",
        [city_slug, target_date],
    ).fetchall()

    # Deterministic model values
    det_rows = db.execute(
        """SELECT model, temp_max FROM deterministic_forecasts
        WHERE city_slug = ? AND date = ? AND temp_max IS NOT NULL""",
        [city_slug, target_date],
    ).fetchall()

    all_values = [r[1] for r in ens_rows] + [r[1] for r in det_rows]
    if len(all_values) < 2:
        return {"score": 50.0, "has_data": False}

    overall_mean = np.mean(all_values)
    # Fraction agreeing on direction (above/below mean)
    above = sum(1 for v in all_values if v >= overall_mean)
    below = len(all_values) - above
    agreement_ratio = max(above, below) / len(all_values)

    # Inter-model spread
    inter_std = float(np.std(all_values))

    # Score: high agreement + low spread = high score
    agreement_score = agreement_ratio * 100
    spread_penalty = min(30, inter_std * 5)
    score = max(0, min(100, agreement_score - spread_penalty))

    n_models = len(set(r[0] for r in ens_rows)) + len(set(r[0] for r in det_rows))
    conflict = None
    if agreement_ratio < 0.6:
        conflict = f"Model disagreement: only {agreement_ratio:.0%} of {n_models} models agree on direction"

    return {"score": round(score, 1), "has_data": True, "n_models": n_models, "conflict": conflict}


def compute_atmospheric_stability(db, city_slug: str, target_date: date) -> dict:
    """Use CAPE from ensemble data to assess atmospheric stability.

    CAPE < 500 → stable → more predictable (high score)
    CAPE > 2000 → convective → less predictable (low score)
    """
    row = db.execute(
        """SELECT AVG(cape), MAX(cape) FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ? AND cape IS NOT NULL""",
        [city_slug, target_date],
    ).fetchone()

    if row is None or row[0] is None:
        return {"score": 50.0, "has_data": False}

    avg_cape = float(row[0])
    max_cape = float(row[1])

    # Low CAPE = stable = high score; high CAPE = convective = low score
    if avg_cape < 200:
        score = 90.0
    elif avg_cape < 500:
        score = 75.0
    elif avg_cape < 1000:
        score = 55.0
    elif avg_cape < 2000:
        score = 35.0
    else:
        score = 15.0

    conflict = None
    if max_cape > 2500:
        conflict = f"High CAPE ({max_cape:.0f} J/kg) indicates convective instability"

    return {"score": score, "has_data": True, "avg_cape": avg_cape, "conflict": conflict}


def analyze_pressure_patterns(db, city_slug: str, target_date: date) -> dict:
    """Analyze pressure trends from ensemble and deterministic data.

    Rising pressure → stable → more predictable
    Falling pressure → perturbation → less predictable
    """
    # Get pressure for target date and day before
    prev_date = target_date - timedelta(days=1)

    rows = db.execute(
        """SELECT time::DATE as d, AVG(pressure_msl) as avg_p
        FROM ensemble_members
        WHERE city_slug = ? AND time::DATE IN (?, ?) AND pressure_msl IS NOT NULL
        GROUP BY time::DATE ORDER BY d""",
        [city_slug, prev_date, target_date],
    ).fetchall()

    if len(rows) < 2:
        # Try deterministic
        det_rows = db.execute(
            """SELECT date, AVG(pressure_msl) as avg_p FROM deterministic_forecasts
            WHERE city_slug = ? AND date IN (?, ?) AND pressure_msl IS NOT NULL
            GROUP BY date ORDER BY date""",
            [city_slug, prev_date, target_date],
        ).fetchall()
        if len(det_rows) < 2:
            return {"score": 50.0, "has_data": False}
        rows = det_rows

    p_prev = float(rows[0][1])
    p_target = float(rows[1][1])
    delta = p_target - p_prev

    # Rising = stable = high score; falling = unstable = low score
    if delta > 3:
        score = 85.0
    elif delta > 1:
        score = 70.0
    elif delta > -1:
        score = 55.0  # stable
    elif delta > -3:
        score = 35.0
    else:
        score = 20.0

    # Check model agreement on pressure trend
    det_pressures = db.execute(
        """SELECT model, pressure_msl FROM deterministic_forecasts
        WHERE city_slug = ? AND date = ? AND pressure_msl IS NOT NULL""",
        [city_slug, target_date],
    ).fetchall()

    conflict = None
    if len(det_pressures) >= 3:
        vals = [r[1] for r in det_pressures]
        if np.std(vals) > 5:
            conflict = f"Pressure disagreement: std={np.std(vals):.1f} hPa across {len(vals)} models"

    return {"score": round(score, 1), "has_data": True, "pressure_delta": delta, "conflict": conflict}


def compute_soil_moisture_bias(db, city_slug: str, target_date: date) -> dict:
    """Cross-reference soil moisture with temperature and precipitation forecasts.

    Wet soil → moderate temps, more precip likelihood
    Dry soil + cold forecast → anomaly → less confident
    """
    # Soil moisture from ensemble
    sm_row = db.execute(
        """SELECT AVG(soil_moisture) FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ? AND soil_moisture IS NOT NULL""",
        [city_slug, target_date],
    ).fetchone()

    if sm_row is None or sm_row[0] is None:
        # Try deterministic
        sm_row = db.execute(
            """SELECT AVG(soil_moisture) FROM deterministic_forecasts
            WHERE city_slug = ? AND date = ? AND soil_moisture IS NOT NULL""",
            [city_slug, target_date],
        ).fetchone()
        if sm_row is None or sm_row[0] is None:
            return {"score": 50.0, "has_data": False}

    soil_moisture = float(sm_row[0])

    # Get temperature forecast for context
    temp_row = db.execute(
        """SELECT AVG(temperature_2m) FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ? AND temperature_2m IS NOT NULL""",
        [city_slug, target_date],
    ).fetchone()

    score = 50.0
    conflict = None

    if temp_row and temp_row[0] is not None:
        avg_temp = float(temp_row[0])
        # Dry soil (< 0.1) + cold forecast (< 32F) = anomaly
        if soil_moisture < 0.1 and avg_temp < 32:
            score = 35.0
            conflict = f"Dry soil ({soil_moisture:.2f}) with cold temps ({avg_temp:.0f}F) — unusual pattern"
        elif soil_moisture > 0.3 and avg_temp > 90:
            score = 40.0
            conflict = f"Wet soil ({soil_moisture:.2f}) with high temps ({avg_temp:.0f}F) — humidity risk"
        elif 0.1 <= soil_moisture <= 0.3:
            score = 65.0  # normal range
        else:
            score = 50.0

    return {"score": score, "has_data": True, "soil_moisture": soil_moisture, "conflict": conflict}


def check_cross_variable_consistency(db, city_slug: str, target_date: date) -> dict:
    """Verify internal consistency between forecast variables.

    Checks: precip vs humidity, wind vs pressure, snow vs temperature.
    """
    row = db.execute(
        """SELECT AVG(precipitation) as precip, AVG(relative_humidity_2m) as rh,
                AVG(wind_speed_10m) as wind, AVG(pressure_msl) as pressure,
                AVG(temperature_2m) as temp, AVG(snow_depth) as snow
        FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ? AND temperature_2m IS NOT NULL""",
        [city_slug, target_date],
    ).fetchone()

    if row is None or row[4] is None:
        return {"score": 50.0, "has_data": False}

    precip, rh, wind, pressure, temp, snow = [float(v) if v is not None else None for v in row]

    checks_passed = 0
    checks_total = 0
    conflicts = []

    # Check 1: High precip should have high humidity
    if precip is not None and rh is not None:
        checks_total += 1
        if precip > 2 and rh < 40:
            conflicts.append(f"Precip={precip:.1f}mm but RH={rh:.0f}% (too dry)")
        else:
            checks_passed += 1

    # Check 2: Snow should require cold temps
    if snow is not None and temp is not None:
        checks_total += 1
        if snow > 0.5 and temp > 40:  # > 40F and snow
            conflicts.append(f"Snow={snow:.1f}cm but temp={temp:.0f}F (too warm)")
        else:
            checks_passed += 1

    # Check 3: Strong wind usually accompanies pressure change
    if wind is not None and pressure is not None:
        checks_total += 1
        # Very high wind with very stable pressure is unusual
        if wind > 30 and 1010 < pressure < 1020:
            conflicts.append(f"Wind={wind:.0f}km/h but pressure stable ({pressure:.0f}hPa)")
        else:
            checks_passed += 1

    # Check 4: Temperature and humidity coherence
    if temp is not None and rh is not None:
        checks_total += 1
        # Very hot and very humid is physically possible but stressful
        if temp > 100 and rh > 80:
            conflicts.append(f"Extreme heat index: temp={temp:.0f}F, RH={rh:.0f}%")
        else:
            checks_passed += 1

    if checks_total == 0:
        return {"score": 50.0, "has_data": False}

    consistency_ratio = checks_passed / checks_total
    score = round(consistency_ratio * 100, 1)

    return {
        "score": score,
        "has_data": True,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "conflict": conflicts[0] if conflicts else None,
    }


def compute_marine_confidence(db, city_slug: str, target_date: date) -> dict:
    """For coastal cities: cross-reference marine data with wind forecasts.

    Waves + wind coherence boosts confidence.
    Non-coastal cities get a neutral score.
    """
    from weather_engine.collectors.marine import COASTAL_CITIES

    if city_slug not in COASTAL_CITIES:
        return {"score": 50.0, "has_data": False}

    marine_row = db.execute(
        """SELECT wave_height_max, swell_wave_height_max, ocean_current_velocity
        FROM marine_data WHERE city_slug = ? AND date = ?""",
        [city_slug, target_date],
    ).fetchone()

    if marine_row is None:
        return {"score": 50.0, "has_data": False}

    wave_h = marine_row[0]
    swell_h = marine_row[1]

    # Get wind forecast
    wind_row = db.execute(
        """SELECT AVG(wind_speed_10m) FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ? AND wind_speed_10m IS NOT NULL""",
        [city_slug, target_date],
    ).fetchone()

    wind_speed = float(wind_row[0]) if wind_row and wind_row[0] else None

    conflict = None
    score = 50.0

    if wave_h is not None and wind_speed is not None:
        # High waves + strong wind = coherent
        if wave_h > 2 and wind_speed > 20:
            score = 75.0  # coherent stormy conditions
        elif wave_h < 0.5 and wind_speed > 25:
            score = 30.0
            conflict = f"Calm waves ({wave_h:.1f}m) but strong wind forecast ({wind_speed:.0f}km/h)"
        elif wave_h > 3 and wind_speed < 10:
            score = 35.0
            conflict = f"High waves ({wave_h:.1f}m) but light wind ({wind_speed:.0f}km/h)"
        else:
            score = 60.0

    return {"score": score, "has_data": True, "conflict": conflict}


def check_flood_precip_consistency(db, city_slug: str, target_date: date) -> dict:
    """Cross-reference flood discharge with precipitation forecast.

    High discharge + high precip = coherent
    Low discharge + extreme precip = conflict
    """
    flood_row = db.execute(
        """SELECT river_discharge, river_discharge_mean, river_discharge_max
        FROM flood_data WHERE city_slug = ? AND date = ?""",
        [city_slug, target_date],
    ).fetchone()

    if flood_row is None:
        return {"score": 50.0, "has_data": False}

    discharge = float(flood_row[0]) if flood_row[0] else None
    discharge_mean = float(flood_row[1]) if flood_row[1] else None

    # Get precipitation forecast
    precip_row = db.execute(
        """SELECT SUM(precipitation) FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ? AND precipitation IS NOT NULL
        GROUP BY model, member_id""",
        [city_slug, target_date],
    ).fetchall()

    if not precip_row or discharge is None:
        return {"score": 50.0, "has_data": discharge is not None}

    avg_precip = float(np.mean([r[0] for r in precip_row]))

    conflict = None
    # High discharge + high precip = coherent, boost confidence
    if discharge_mean and discharge > discharge_mean * 1.5 and avg_precip > 10:
        score = 70.0  # coherent high-risk pattern
    elif discharge_mean and discharge < discharge_mean * 0.5 and avg_precip > 20:
        score = 30.0
        conflict = f"Low discharge ({discharge:.0f}) but high precip forecast ({avg_precip:.1f}mm)"
    elif discharge_mean and discharge > discharge_mean * 2 and avg_precip < 2:
        score = 35.0
        conflict = f"High discharge ({discharge:.0f}) but minimal precip ({avg_precip:.1f}mm)"
    else:
        score = 55.0

    return {"score": round(score, 1), "has_data": True, "conflict": conflict}


def check_climate_trend_alignment(db, city_slug: str, target_date: date) -> dict:
    """Compare forecast vs climate trend (climate_indicators + climate_normals).

    Forecast in line with decadal trend → bonus.
    Forecast counter to trend → small penalty (weather is variable).
    """
    doy = target_date.timetuple().tm_yday

    # Climate normal
    normal_row = db.execute(
        """SELECT temperature_2m_max_mean, temperature_2m_max_std FROM climate_normals
        WHERE city_slug = ? AND day_of_year = ?""",
        [city_slug, doy],
    ).fetchone()

    if normal_row is None or normal_row[0] is None:
        return {"score": 50.0, "has_data": False}

    normal_temp = float(normal_row[0])
    normal_std = float(normal_row[1]) if normal_row[1] else 3.0

    # Climate indicator trend (from climate models)
    trend_row = db.execute(
        """SELECT AVG(temp_max) FROM climate_indicators
        WHERE city_slug = ? AND date BETWEEN ? AND ?""",
        [city_slug, target_date, target_date + timedelta(days=30)],
    ).fetchone()

    # Current forecast
    fc_row = db.execute(
        """SELECT AVG(temperature_2m) FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ? AND temperature_2m IS NOT NULL""",
        [city_slug, target_date],
    ).fetchone()

    if fc_row is None or fc_row[0] is None:
        return {"score": 50.0, "has_data": False}

    forecast_temp = float(fc_row[0])
    deviation_from_normal = abs(forecast_temp - normal_temp) / max(normal_std, 1.0)

    # Within 1 std = aligned, within 2 std = acceptable, > 2 std = counter-trend
    if deviation_from_normal < 1.0:
        score = 75.0
    elif deviation_from_normal < 1.5:
        score = 60.0
    elif deviation_from_normal < 2.0:
        score = 45.0
    else:
        score = 30.0

    # Bonus if climate models also support direction
    if trend_row and trend_row[0] is not None:
        climate_trend = float(trend_row[0])
        # If forecast and climate trend both above/below normal → extra boost
        fc_above = forecast_temp > normal_temp
        trend_above = climate_trend > normal_temp
        if fc_above == trend_above:
            score = min(100, score + 10)

    conflict = None
    if deviation_from_normal > 2.5:
        conflict = f"Forecast {forecast_temp:.0f}F deviates {deviation_from_normal:.1f} std from climate normal {normal_temp:.0f}F"

    return {"score": round(score, 1), "has_data": True, "conflict": conflict}


def compute_aqi_uncertainty(db, city_slug: str, target_date: date) -> dict:
    """AQI-weather correlation for uncertainty estimation.

    High AQI → extreme weather patterns → more uncertainty.
    Cross-ref PM10 with visibility for data quality check.
    """
    aqi_row = db.execute(
        """SELECT AVG(aqi_european), AVG(pm10), AVG(us_aqi)
        FROM air_quality WHERE city_slug = ? AND forecast_date = ?""",
        [city_slug, target_date],
    ).fetchone()

    if aqi_row is None or aqi_row[0] is None:
        return {"score": 50.0, "has_data": False}

    aqi = float(aqi_row[0])
    pm10 = float(aqi_row[1]) if aqi_row[1] else None

    # Check visibility from ensemble for cross-validation
    vis_row = db.execute(
        """SELECT AVG(visibility) FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ? AND visibility IS NOT NULL""",
        [city_slug, target_date],
    ).fetchone()

    conflict = None

    # Low AQI = good air = stable = high score
    if aqi < 50:
        score = 70.0
    elif aqi < 100:
        score = 55.0
    elif aqi < 150:
        score = 40.0
    else:
        score = 25.0

    # Cross-ref: high PM10 but high visibility → suspect AQI data
    if pm10 is not None and vis_row and vis_row[0] is not None:
        visibility = float(vis_row[0])
        if pm10 > 50 and visibility > 20000:  # PM10 high but visibility > 20km
            score = max(score - 10, 10)
            conflict = f"PM10={pm10:.0f} but visibility={visibility:.0f}m — AQI data may be suspect"

    return {"score": round(score, 1), "has_data": True, "conflict": conflict}


def compute_deterministic_agreement(db, city_slug: str, target_date: date) -> dict:
    """Compare 8 deterministic models, weighted by historical accuracy.

    Uses model_tracker weights for reliability-adjusted agreement.
    """
    from weather_engine.analysis.model_tracker import get_model_weights

    rows = db.execute(
        """SELECT model, temp_max FROM deterministic_forecasts
        WHERE city_slug = ? AND date = ? AND temp_max IS NOT NULL""",
        [city_slug, target_date],
    ).fetchall()

    if len(rows) < 2:
        return {"score": 50.0, "has_data": False}

    values = {r[0]: r[1] for r in rows}

    # Get reliability weights
    weights = get_model_weights(db, city_slug, "temperature_2m_max")

    # Weighted mean and spread
    if weights:
        weighted_vals = []
        total_w = 0
        for model, val in values.items():
            w = weights.get(model, 0.05)  # default small weight for unknown models
            weighted_vals.append(val * w)
            total_w += w
        weighted_mean = sum(weighted_vals) / total_w if total_w > 0 else np.mean(list(values.values()))
    else:
        weighted_mean = np.mean(list(values.values()))

    # Spread: std of all model values
    spread = float(np.std(list(values.values())))

    # Score: low spread = high agreement
    if spread < 1.0:
        score = 90.0
    elif spread < 2.0:
        score = 75.0
    elif spread < 3.5:
        score = 55.0
    elif spread < 5.0:
        score = 35.0
    else:
        score = 15.0

    conflict = None
    if spread > 5.0:
        conflict = f"Deterministic models diverge: spread={spread:.1f}F across {len(values)} models"

    return {
        "score": round(score, 1),
        "has_data": True,
        "n_models": len(values),
        "spread": round(spread, 2),
        "weighted_mean": round(weighted_mean, 1),
        "conflict": conflict,
    }


def compute_teleconnection_alignment(db, city_slug: str, target_date: date) -> dict:
    """Check teleconnection indices for alignment with forecast conditions.

    ENSO+ + warm SE-US = aligned, NAO+ + mild Europe = aligned,
    AO- + cold mid-lat = aligned.
    """
    try:
        # Get latest indices
        oni_row = db.execute(
            "SELECT value FROM teleconnection_indices WHERE index_name = 'oni' ORDER BY date DESC LIMIT 1"
        ).fetchone()
        nao_row = db.execute(
            "SELECT value FROM teleconnection_indices WHERE index_name = 'nao' ORDER BY date DESC LIMIT 1"
        ).fetchone()
        ao_row = db.execute(
            "SELECT value FROM teleconnection_indices WHERE index_name = 'ao' ORDER BY date DESC LIMIT 1"
        ).fetchone()

        if oni_row is None and nao_row is None and ao_row is None:
            return {"score": 50.0, "has_data": False}

        # Get city info for region
        city_row = db.execute(
            "SELECT country, latitude FROM cities WHERE slug = ?", [city_slug]
        ).fetchone()
        if city_row is None:
            return {"score": 50.0, "has_data": False}

        country = city_row[0]
        lat = float(city_row[1])

        # Get forecast temperature
        fc_row = db.execute(
            """SELECT AVG(temperature_2m) FROM ensemble_members
            WHERE city_slug = ? AND time::DATE = ? AND temperature_2m IS NOT NULL""",
            [city_slug, target_date],
        ).fetchone()

        # Get climate normal for reference
        doy = target_date.timetuple().tm_yday
        normal_row = db.execute(
            "SELECT temperature_2m_max_mean FROM climate_normals WHERE city_slug = ? AND day_of_year = ?",
            [city_slug, doy],
        ).fetchone()

        forecast_warm = False
        if fc_row and fc_row[0] is not None and normal_row and normal_row[0] is not None:
            forecast_warm = float(fc_row[0]) > float(normal_row[0])

        score = 50.0
        alignments = 0

        # ENSO: positive ONI + warm southern US / cool northern
        if oni_row and oni_row[0] is not None:
            oni = float(oni_row[0])
            if country == "US" and lat < 35:
                if oni > 0.5 and forecast_warm:
                    score += 10
                    alignments += 1
                elif oni < -0.5 and not forecast_warm:
                    score += 8
                    alignments += 1

        # NAO: positive = mild European winter
        if nao_row and nao_row[0] is not None:
            nao = float(nao_row[0])
            is_european = country in ("GB", "FR", "DE", "IT", "ES", "NL", "TR")
            if is_european:
                if nao > 0.5 and forecast_warm:
                    score += 10
                    alignments += 1
                elif nao < -0.5 and not forecast_warm:
                    score += 8
                    alignments += 1

        # AO: negative = cold mid-latitudes
        if ao_row and ao_row[0] is not None:
            ao = float(ao_row[0])
            is_midlat = 30 < lat < 60
            if is_midlat:
                if ao < -1.0 and not forecast_warm:
                    score += 10
                    alignments += 1
                elif ao > 1.0 and forecast_warm:
                    score += 8
                    alignments += 1

        score = min(100, max(0, score))
        return {"score": round(score, 1), "has_data": True, "alignments": alignments}

    except Exception as e:
        logger.debug("Teleconnection alignment failed for %s: %s", city_slug, e)
        return {"score": 50.0, "has_data": False}


def compute_ensemble_regime_score(db, city_slug: str, target_date: date) -> dict:
    """Score based on ensemble regime analysis.

    Unimodal tight = 80, bimodal with dominant >75% = 60, true bimodal = 30.
    """
    try:
        from weather_engine.analysis.ensemble_analysis import analyze_ensemble

        analysis = analyze_ensemble(db, city_slug, target_date)
        if analysis is None:
            return {"score": 50.0, "has_data": False}

        regime = analysis.regime
        if not regime.is_bimodal:
            # Unimodal: score based on spread
            if analysis.std < 2.0:
                score = 85.0
            elif analysis.std < 4.0:
                score = 75.0
            else:
                score = 65.0
        elif regime.dominant_regime_weight > 0.75:
            score = 60.0
        else:
            score = 30.0

        return {"score": round(score, 1), "has_data": True, "is_bimodal": regime.is_bimodal}

    except Exception as e:
        logger.debug("Ensemble regime score failed for %s: %s", city_slug, e)
        return {"score": 50.0, "has_data": False}


def compute_extreme_value_score(db, city_slug: str, target_date: date) -> dict:
    """Score based on how extreme the forecast is relative to historical extremes.

    Forecast near mean = high score, forecast in GEV tail = low score.
    """
    try:
        from weather_engine.analysis.extreme_value import analyze_extremes

        # Get forecast value
        fc_row = db.execute(
            """SELECT AVG(temperature_2m) FROM ensemble_members
            WHERE city_slug = ? AND time::DATE = ? AND temperature_2m IS NOT NULL""",
            [city_slug, target_date],
        ).fetchone()

        if fc_row is None or fc_row[0] is None:
            return {"score": 50.0, "has_data": False}

        forecast_value = float(fc_row[0])
        result = analyze_extremes(db, city_slug, "temperature_2m_max", forecast_value)
        return {"score": round(result.tail_score, 1), "has_data": True}

    except Exception as e:
        logger.debug("Extreme value score failed for %s: %s", city_slug, e)
        return {"score": 50.0, "has_data": False}


def store_cross_reference(db, city_slug: str, target_date: date, scores: dict) -> None:
    """Store computed cross-reference scores to DB."""
    db.execute(
        """INSERT OR REPLACE INTO cross_reference_scores
        (city_slug, target_date,
         model_agreement, atmospheric_stability, pressure_patterns,
         soil_moisture_bias, cross_variable_consistency, marine_influence,
         flood_precip_consistency, climate_trend_alignment,
         aqi_weather_correlation, deterministic_agreement,
         teleconnection_alignment, ensemble_regime_score, extreme_value_score,
         composite_score, source_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            city_slug, target_date,
            scores.get("model_agreement", 50),
            scores.get("atmospheric_stability", 50),
            scores.get("pressure_patterns", 50),
            scores.get("soil_moisture_bias", 50),
            scores.get("cross_variable_consistency", 50),
            scores.get("marine_influence", 50),
            scores.get("flood_precip_consistency", 50),
            scores.get("climate_trend_alignment", 50),
            scores.get("aqi_weather_correlation", 50),
            scores.get("deterministic_agreement", 50),
            scores.get("teleconnection_alignment", 50),
            scores.get("ensemble_regime_score", 50),
            scores.get("extreme_value_score", 50),
            scores.get("composite_score", 50),
            scores.get("source_count", 0),
        ],
    )


def run_cross_reference_update(db) -> dict:
    """Run cross-reference for all cities, for the next 7 days."""
    from datetime import date as date_type
    from weather_engine.db import get_cities

    cities = get_cities(db)
    today = date_type.today()
    total = 0

    for city in cities:
        slug = city["slug"]
        for days_ahead in range(0, 7):
            target = today + timedelta(days=days_ahead)
            try:
                scores = compute_full_cross_reference(db, slug, target, days_ahead)
                store_cross_reference(db, slug, target, scores)
                total += 1
            except Exception as e:
                logger.debug("Cross-ref failed for %s/%s: %s", slug, target, e)

    logger.info("Cross-reference update: %d city-date combinations computed", total)
    return {"computed": total}
