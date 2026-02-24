@echo off
setlocal
cd /d "%~dp0"

REM Weather Trader desktop launcher (Windows)
REM Backend is auto-started by app_new.py if ../weather-engine is present.

where uv >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  uv run python app_new.py
  exit /b %ERRORLEVEL%
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  python app_new.py
  exit /b %ERRORLEVEL%
)

echo [ERROR] Neither 'uv' nor 'python' found in PATH.
echo Install Python 3.12+ and uv, then retry.
pause
exit /b 1
