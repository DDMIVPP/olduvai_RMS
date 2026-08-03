@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON_EXE="
set "PYTHON_ARGS="

echo Looking for 64-bit Python 3.12...

where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') * 8 == 64 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3.12"
    )
)

if not defined PYTHON_EXE (
    where python3.12 >nul 2>&1
    if not errorlevel 1 (
        python3.12 -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') * 8 == 64 else 1)" >nul 2>&1
        if not errorlevel 1 set "PYTHON_EXE=python3.12"
    )
)

if not defined PYTHON_EXE (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') * 8 == 64 else 1)" >nul 2>&1
        if not errorlevel 1 set "PYTHON_EXE=python"
    )
)

if not defined PYTHON_EXE (
    if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
)

if not defined PYTHON_EXE (
    if exist "%ProgramFiles%\Python312\python.exe" set "PYTHON_EXE=%ProgramFiles%\Python312\python.exe"
)

if not defined PYTHON_EXE (
    echo.
    echo A 64-bit Python 3.12 interpreter was not found.
    echo Install Python 3.12 64-bit, then run this file again.
    echo During Python installation, selecting "Add python.exe to PATH" is recommended.
    pause
    exit /b 1
)

"%PYTHON_EXE%" %PYTHON_ARGS% --version
if errorlevel 1 (
    echo.
    echo The selected Python interpreter could not be started.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Creating the project virtual environment .venv ...
    "%PYTHON_EXE%" %PYTHON_ARGS% -m venv ".venv"
    if errorlevel 1 (
        echo.
        echo Failed to create .venv.
        pause
        exit /b 1
    )
)

set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
"%VENV_PYTHON%" -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') * 8 == 64 else 1)"
if errorlevel 1 (
    echo.
    echo The existing .venv does not use 64-bit Python 3.12.
    echo Rename or remove that .venv, then run this installer again.
    pause
    exit /b 1
)

echo.
echo Upgrading pip in .venv ...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo.
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

echo.
echo Installing the pinned packages from requirements.txt ...
"%VENV_PYTHON%" -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 (
    echo.
    echo Installation failed. Check the internet connection and requirements.txt.
    pause
    exit /b 1
)

echo.
echo Verifying required imports ...
"%VENV_PYTHON%" -c "import matplotlib,numpy,openpyxl,pandas,scipy,sklearn,statsmodels; print('Dependency import check passed.')"
if errorlevel 1 (
    echo.
    echo Import verification failed.
    pause
    exit /b 1
)

echo.
echo Installation completed successfully.
echo Run the analysis with:
echo   .venv\Scripts\python.exe olduvai_RMS.py
pause
