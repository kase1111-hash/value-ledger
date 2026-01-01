@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo  Value Ledger - Assembly Script
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

echo [1/7] Python found
python --version
echo.

:: Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo [2/7] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo       Done
) else (
    echo [2/7] Virtual environment already exists
)
echo.

:: Activate virtual environment
echo [3/7] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo       Done
echo.

:: Upgrade pip
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo       Done
echo.

:: Install the package in editable mode with dev dependencies
echo [5/7] Installing Value Ledger with dev dependencies...
pip install -e ".[dev]" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to install package
    pause
    exit /b 1
)
echo       Done
echo.

:: Install pre-commit hooks
echo [6/7] Setting up pre-commit hooks...
if exist ".pre-commit-config.yaml" (
    pre-commit install >nul 2>&1
    if errorlevel 1 (
        echo       WARNING: Pre-commit setup failed, continuing...
    ) else (
        echo       Done
    )
) else (
    echo       Skipped (no .pre-commit-config.yaml found)
)
echo.

:: Run tests
echo [7/7] Running tests...
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
echo  Assembly complete!
echo ========================================
echo.
echo Virtual environment: .venv
echo.
echo To activate manually:
echo   .venv\Scripts\activate.bat
echo.
echo To start the application:
echo   start.bat
echo.
echo Or use the CLI directly:
echo   python -m value_ledger.cli --help
echo   value-ledger --help
echo.

pause
