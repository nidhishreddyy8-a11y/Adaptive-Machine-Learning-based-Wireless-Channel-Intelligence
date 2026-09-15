@echo off
title Adaptive ML - Wireless Channel Intelligence Server
color 0A
cls
echo.
echo  ============================================================
echo   ADAPTIVE ML - WIRELESS CHANNEL INTELLIGENCE
echo   DeepMIMO 3.5 GHz ^| Flask Backend ^| Port 5000
echo  ============================================================
echo.
echo  [INFO] Starting server... Please wait.
echo.

cd /d "C:\Users\HP\OneDrive\Desktop\Adaptive ML"

:: Check if port 5000 is already in use and kill it
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":5000 "') do (
    echo  [INFO] Freeing port 5000 (PID: %%a)...
    taskkill /PID %%a /F >nul 2>&1
)

echo  [OK]  Port 5000 is ready.
echo  [OK]  Loading 7 ML models from .\models\
echo.
echo  ============================================================
echo   Server URL : http://127.0.0.1:5000
echo   Network    : http://192.168.31.125:5000  (LAN access)
echo   Stop       : Close this window or press CTRL+C
echo  ============================================================
echo.

:: Open browser after 3 second delay (in background)
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:5000"

:: Launch Flask
"C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe" app/app.py

echo.
echo  [INFO] Server stopped. Press any key to close.
pause >nul
