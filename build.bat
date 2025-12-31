@echo off
setlocal

echo ========================================
echo  Value Ledger - Windows Build Script
echo ========================================
echo.

:: Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Python found
python --version

:: Install the package in editable mode
echo.
echo [2/4] Installing Value Ledger...
pip install -e . >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to install package
    pause
    exit /b 1
)
echo       Done

:: Install dev dependencies
echo.
echo [3/4] Installing dev dependencies...
pip install -e ".[dev]" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Dev dependencies failed, continuing anyway...
) else (
    echo       Done
)

:: Run tests
echo.
echo [4/4] Running tests...
pytest tests/ -q
if errorlevel 1 (
    echo.
    echo WARNING: Some tests failed
) else (
    echo.
    echo All tests passed!
)

echo.
echo ========================================
echo  Build complete!
echo ========================================
echo.
echo Usage:
echo   python -m value_ledger.cli --help
echo   python -m value_ledger.cli demo
echo.

pause
