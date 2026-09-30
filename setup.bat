@echo off
REM PharmaLens — Environment Setup Script
REM Run this from the project root directory

echo ============================================
echo   PharmaLens — Setting Up Environment
echo ============================================

REM Use Python 3.12
set PYTHON=C:\Users\ASUS\AppData\Local\Programs\Python\Python312\python.exe

echo.
echo [1/6] Creating virtual environment...
%PYTHON% -m venv venv
call venv\Scripts\activate.bat

echo.
echo [2/6] Upgrading pip and build tools...
python -m pip install --upgrade pip setuptools wheel

echo.
echo [3/6] Installing core scientific packages...
pip install numpy pandas scipy matplotlib seaborn scikit-learn tqdm joblib requests jupyter jupyterlab ipywidgets

echo.
echo [4/6] Installing ML and XAI packages...
pip install xgboost shap optuna captum

echo.
echo [5/6] Installing PyTorch (CPU)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo [5b/6] Installing PyTorch Geometric...
pip install torch-geometric

echo.
echo [5c/6] Installing RDKit...
REM rdkit-pypi was renamed to rdkit for Python 3.12+
pip install rdkit

echo.
echo [6/6] Installing PharmaLens in editable mode...
pip install -e .

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo If RDKit failed, try:  conda install -c conda-forge rdkit
echo.
echo To start working:
echo   1. Activate:  venv\Scripts\activate
echo   2. Launch:    jupyter lab notebooks/
echo   3. Run notebooks in order: 01 through 14
echo.
pause
