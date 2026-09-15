@echo off
title Adaptive ML - Service Installer
color 0B
cls
echo.
echo  ================================================================
echo   ADAPTIVE ML - WIRELESS CHANNEL INTELLIGENCE
echo   Windows Auto-Start Service Installer
echo  ================================================================
echo.

:: Check for admin privileges
net session >nul 2>&1
if %errorlevel% NEQ 0 (
    echo  [ERROR] This installer requires Administrator privileges.
    echo.
    echo  Please RIGHT-CLICK this file and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo  [OK] Running with Administrator privileges.
echo.

set "TASK_NAME=AdaptiveML_ChannelIntelligence"
set "TASK_XML=%~dp0adaptive_ml_task.xml"
set "PYTHON_EXE=C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe"
set "APP_DIR=C:\Users\HP\OneDrive\Desktop\Adaptive ML"

echo  [INFO] Registering scheduled task: %TASK_NAME%
echo.

:: Remove existing task if present
schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

:: Register new task from XML
schtasks /Create /XML "%TASK_XML%" /TN "%TASK_NAME%" /F

if %errorlevel% EQU 0 (
    echo.
    echo  ================================================================
    echo   [SUCCESS] Auto-start service installed!
    echo  ================================================================
    echo.
    echo   The Flask server will now automatically start:
    echo    - Every time you LOG IN to Windows (10 sec after login)
    echo    - Running silently in the background
    echo    - Auto-restarts if it crashes (up to 3 times)
    echo.
    echo   Access the app at: http://127.0.0.1:5000
    echo.
    echo   To STOP the service:   Run uninstall_service.bat
    echo   To manually start now: Run start_server.bat
    echo  ================================================================
    echo.
    
    :: Ask to start server right now
    set /p START_NOW="Start the server now? (Y/N): "
    if /i "%START_NOW%"=="Y" (
        echo.
        echo  [INFO] Starting server in background...
        schtasks /Run /TN "%TASK_NAME%"
        timeout /t 3 /nobreak >nul
        start http://127.0.0.1:5000
        echo  [OK] Server started! Browser opening...
    )
) else (
    echo.
    echo  [ERROR] Failed to register task. Try running as Administrator.
)

echo.
pause
