# Distribution Guide (Linux / Windows / macOS)

This project has two parts:

- `desktop-app` (Flet UI)
- `weather-engine` (FastAPI backend)

## What "complete" means

For a friend to run the app with all features:

1. Desktop app launches
2. Backend starts locally (`localhost:8321`)
3. UI connects automatically
4. (Optional) Ollama local AI can be installed from inside the app

The desktop app now includes **backend auto-bootstrap**:

- it checks `http://localhost:8321/health`
- if backend is not running, it looks for a sibling `weather-engine/`
- then it starts it in background

## Portable folder layout (recommended for friends)

Distribute these folders together:

```text
WeatherTrader/
├── desktop-app/
└── weather-engine/
```

This works best when `weather-engine/.venv` is already prepared on the target OS.

## Launchers

- Linux/macOS shell: `desktop-app/launch.sh` or `desktop-app/launch.command`
- Windows: `desktop-app/launch.bat` or `desktop-app/launch.ps1`

`app_new.py` also auto-starts backend if launched directly.

## Packaging strategy by OS

### Windows

- Build the Flet app on Windows via GitHub Actions (`build-release.yml`)
- Ship the generated `.exe`
- For full offline/local backend support, also ship the `weather-engine` folder (or create a separate backend bundle)

### macOS

- Build `.app` + `.dmg` on macOS via GitHub Actions
- Same note as Windows for backend bundle
- Users may need to allow unsigned app in macOS Security settings

### Linux

- Build native binary on Ubuntu runner via GitHub Actions
- Ship binary plus `weather-engine` folder for full features
- Test on target distro (GTK/runtime compatibility can vary)

## Backend runtime requirements

The backend needs Python deps from `weather-engine/pyproject.toml`.

Recommended prep before sharing:

```bash
cd weather-engine
uv sync
```

## AI (Ollama)

AI is optional:

- Without Ollama: the in-app assistant can still use fallback analysis
- With Ollama: users can install Ollama + model from inside the app (with confirmation and logs)

## Troubleshooting

### UI opens but no data

- Check backend health in app (`Sistema`) or logs:
  - desktop log: OS temp dir (e.g. `/tmp/desktop-app.log`)
  - backend log: OS temp dir (e.g. `/tmp/weather-engine.log`)

### Backend not auto-starting

Check:

- sibling folder exists (`../weather-engine`)
- `.venv` or `uv`/`python` is available
- local port `8321` is not blocked/in use

### Packaged app but no backend

That is expected unless you also distribute backend runtime assets.
The desktop binary alone is only the UI layer.
