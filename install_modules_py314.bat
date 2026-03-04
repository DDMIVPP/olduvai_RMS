@echo off
setlocal
cd /d "%~dp0"

echo Checking Python version...
py --version
echo.

echo Upgrading pip...
py -m pip install --upgrade pip
if errorlevel 1 (
    echo.
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

echo.
echo Installing required Python modules from requirements.txt ...
py -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 (
    echo.
    echo Installation failed.
    echo Please make sure requirements.txt is in the same folder as this .bat file.
    pause
    exit /b 1
)

echo.
echo Installation completed successfully.
pause
