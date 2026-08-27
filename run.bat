@echo off
setlocal EnableExtensions

cd /d "%~dp0"
title Laser Shooter

set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "VENV_PYTHONW=%~dp0.venv\Scripts\pythonw.exe"
set "REQUESTED_MODE=%~1"

echo.
echo ========================================
echo  Laser Shooter - Windows Launcher
echo ========================================
echo.

call :find_python
if not defined PYTHON_EXE call :install_python
if not defined PYTHON_EXE goto :python_error

echo [OK] Python: "%PYTHON_EXE%"

if not exist "%VENV_PYTHON%" (
    echo [SETUP] Creating the virtual environment...
    "%PYTHON_EXE%" -m venv "%~dp0.venv"
    if errorlevel 1 goto :venv_error
)

"%VENV_PYTHON%" -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1
if errorlevel 1 goto :venv_version_error

"%VENV_PYTHON%" -c "import cv2, numpy, tkinter; assert hasattr(cv2, 'aruco')" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Installing dependencies...
    "%VENV_PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto :dependency_error

    "%VENV_PYTHON%" -m pip install --only-binary=:all: -r "%~dp0requirements.txt"
    if errorlevel 1 goto :dependency_error
) else (
    echo [OK] Dependencies are already installed.
)

echo [OK] Setup is complete.

if /i "%REQUESTED_MODE%"=="setup" exit /b 0
if /i "%REQUESTED_MODE%"=="gui" goto :gui_once
if /i "%REQUESTED_MODE%"=="calibrate" goto :calibrate_once
if /i "%REQUESTED_MODE%"=="main" goto :main_once
if not "%REQUESTED_MODE%"=="" goto :usage_error
goto :gui_once

:gui_once
if not exist "%VENV_PYTHONW%" goto :venv_error
start "" "%VENV_PYTHONW%" "%~dp0gui.py"
exit /b 0

:main_once
"%VENV_PYTHON%" "%~dp0main.py"
exit /b %ERRORLEVEL%

:calibrate_once
"%VENV_PYTHON%" "%~dp0red_difference.py"
exit /b %ERRORLEVEL%

:find_python
set "PYTHON_EXE="

for /f "delims=" %%P in ('py -3.13 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
if defined PYTHON_EXE exit /b 0

if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
    exit /b 0
)

if exist "%LocalAppData%\Programs\Python\Python314\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python314\python.exe"
    exit /b 0
)

if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
    exit /b 0
)

if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python311\python.exe"
    exit /b 0
)

if exist "%LocalAppData%\Programs\Python\Python310\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python310\python.exe"
    exit /b 0
)

for /f "delims=" %%P in ('py -c "import sys; assert sys.version_info.major == 3 and sys.version_info.minor in range(10, 20); print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
exit /b 0

:install_python
where winget >nul 2>&1
if errorlevel 1 exit /b 0

echo [SETUP] Python 3.13 was not found. Installing it with winget...
winget install --exact --id Python.Python.3.13 --scope user --accept-package-agreements --accept-source-agreements
if errorlevel 1 exit /b 0

call :find_python
exit /b 0

:python_error
echo.
echo [ERROR] Python 3.10 or newer could not be found or installed.
echo Install Python 3.13 and run this file again:
echo https://www.python.org/downloads/windows/
echo.
pause
exit /b 1

:venv_error
echo.
echo [ERROR] Failed to create the virtual environment.
echo Check the messages above and try again.
echo.
pause
exit /b 1

:venv_version_error
echo.
echo [ERROR] The existing .venv uses an unsupported Python version.
echo Delete the .venv directory and run this file again.
echo.
pause
exit /b 1

:dependency_error
echo.
echo [ERROR] Failed to install dependencies.
echo Check the internet connection and the messages above.
echo.
pause
exit /b 1

:usage_error
echo.
echo [ERROR] Unknown option: %REQUESTED_MODE%
echo Usage: run.bat [setup^|gui^|calibrate^|main]
echo.
pause
exit /b 2
