@echo off
setlocal EnableDelayedExpansion

:: Value Ledger - Startup Script
:: Usage: start.bat [command] [options]
:: Examples:
::   start.bat              - Show help
::   start.bat demo         - Run demo workflow
::   start.bat stats        - Show ledger statistics
::   start.bat query        - Query entries

:: Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat >nul 2>&1
)

:: Check if value_ledger is importable
python -c "import value_ledger" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Value Ledger is not installed
    echo.
    echo Please run assemble.bat first to set up the environment
    echo Or install manually: pip install -e .
    pause
    exit /b 1
)

:: Run CLI with passed arguments, or show help if none
if "%~1"=="" (
    python -m value_ledger.cli --help
) else (
    python -m value_ledger.cli %*
)
