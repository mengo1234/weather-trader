#!/bin/bash
# Weather Trader launcher
# Avvia il server backend se non è già in esecuzione, poi l'app desktop

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$(dirname "$SCRIPT_DIR")/weather-engine"
APP_DIR="$SCRIPT_DIR"
LOG_DIR="/tmp"

# Avvia weather-engine se non risponde
if ! curl -s http://localhost:8321/health > /dev/null 2>&1; then
    echo "Avvio weather-engine..."
    cd "$ENGINE_DIR"
    nohup uv run uvicorn weather_engine.main:app --host 0.0.0.0 --port 8321 > "$LOG_DIR/weather-engine.log" 2>&1 &
    # Aspetta che il server sia pronto
    for i in $(seq 1 15); do
        sleep 1
        if curl -s http://localhost:8321/health > /dev/null 2>&1; then
            break
        fi
    done
fi

# Avvia la desktop app
cd "$APP_DIR"
exec uv run python app_new.py 2> "$LOG_DIR/desktop-app.log"
