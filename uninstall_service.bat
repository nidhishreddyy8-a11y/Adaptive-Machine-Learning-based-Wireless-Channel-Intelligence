@echo off
title Adaptive ML - Service Uninstaller
color 0C
cls
echo.
echo  ================================================================
echo   ADAPTIVE ML - Remove Auto-Start Service
echo  ================================================================
echo.

net session >nul 2>&1
if %errorlevel% NEQ 0 (
    echo  [ERROR] Requires Administrator privileges. Right-click ^> Run as administrator.
    pause
    exit /b 1
)

set "TASK_NAME=AdaptiveML_ChannelIntelligence"

:: Kill running Flask process on port 5000
echo  [INFO] Stopping any running server on port 5000...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":5000 "') do (
    taskkill /PID %%a /F >nul 2>&1
)

:: Delete the scheduled task
schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

if %errorlevel% EQU 0 (
    echo  [OK] Auto-start service removed successfully.
) else (
    echo  [INFO] Task was not registered (already removed or never installed).
)

echo.
echo  The server will no longer auto-start on login.
echo  You can still launch it manually using start_server.bat
echo.
pause
