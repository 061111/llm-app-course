@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM ============================================================
REM  Double-click to start the studio and open the page.
REM  Keep this window open while you use it. Ctrl+C stops it.
REM ============================================================

if not exist ".venv\Scripts\python.exe" (
  echo [x] .venv not found in this folder.
  echo     Run this once first:  uv sync --extra web
  echo.
  pause
  exit /b 1
)

if not exist ".env" (
  echo [!] .env not found. Fill in the keys, then run this again.
  echo.
  pause
  exit /b 1
)

netstat -ano | findstr /r /c:":8000 .*LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo [i] Already running on port 8000 - opening the page.
  start http://127.0.0.1:8000
  exit /b 0
)

echo [i] Starting... the page opens by itself in about 3 seconds.
echo     URL: http://127.0.0.1:8000
echo     Close this window or press Ctrl+C to stop.
echo.

REM The page cannot open until the server is up, so wait ~3s in a minimized window.
start "" /min cmd /c "ping -n 4 127.0.0.1 >nul & start http://127.0.0.1:8000"

".venv\Scripts\python.exe" -m studio --port 8000

echo.
echo [i] Server stopped.
pause
