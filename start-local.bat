@echo off
rem Double-click to start the backend and the website, then open http://localhost:3000
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-local.ps1" %*
pause
