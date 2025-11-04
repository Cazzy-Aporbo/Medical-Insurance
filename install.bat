@echo off
REM Medical Insurance Cost Prediction - Windows Installation Script

echo ==============================================
echo Medical Insurance Cost Prediction Installer
echo ==============================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

REM Check pip
echo.
echo Checking pip installation...
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: pip not found, installing...
    python -m ensurepip --upgrade
)
echo pip is available

REM Upgrade pip
echo.
echo Upgrading pip to latest version...
python -m pip install --upgrade pip --quiet

REM Ask about virtual environment
echo.
set /p VENV="Create virtual environment? (recommended) [Y/n]: "
if /i "%VENV%"=="n" goto :install
if /i "%VENV%"=="" set VENV=y

if /i "%VENV%"=="y" (
    echo Creating virtual environment...
    python -m venv venv
    
    REM Activate virtual environment
    call venv\Scripts\activate.bat
    echo Virtual environment created and activated
)

:install
REM Install requirements
echo.
echo Installing required packages...
echo This may take a few minutes...
pip install -r requirements.txt --quiet

if %errorlevel% neq 0 (
    echo Error during package installation
    pause
    exit /b 1
)
echo All packages installed successfully!

REM Verify installation
echo.
echo Verifying installation...
python -c "import sys; import pandas as pd; import numpy as np; import matplotlib; import seaborn as sns; import sklearn; import scipy; print(f'\nPython: {sys.version.split()[0]}'); print(f'pandas: {pd.__version__}'); print(f'numpy: {np.__version__}'); print(f'matplotlib: {matplotlib.__version__}'); print(f'seaborn: {sns.__version__}'); print(f'scikit-learn: {sklearn.__version__}'); print(f'scipy: {scipy.__version__}'); print('\n[OK] All packages verified successfully!')"

if %errorlevel% neq 0 (
    echo Error: Package verification failed
    pause
    exit /b 1
)

REM Check data file
echo.
if exist insurance.csv (
    echo [OK] Dataset (insurance.csv) found
) else (
    echo [Warning] Dataset (insurance.csv) not found
    echo           Please download the dataset or ensure it's in the project directory
)

REM Display usage instructions
echo.
echo ==============================================
echo Installation Complete!
echo ==============================================
echo.
echo To run the analysis:
echo.
echo   Foundation Analysis:
echo     python medical_costs_beginner.py
echo.
echo   Intermediate Analysis:
echo     python medical_costs_intermediate.py
echo.
echo   Exceptional Analysis:
echo     python medical_costs_exceptional.py
echo.
echo   Ethical ^& Privacy Analysis:
echo     python ethical_privacy.py
echo.

if /i "%VENV%"=="y" (
    echo Remember to activate the virtual environment before running:
    echo   venv\Scripts\activate.bat
    echo.
)

echo For more information, see README.md or REQUIREMENTS.md
echo.
pause
