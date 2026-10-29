@echo off
chcp 65001 >nul
rem 一次性环境安装（幂等）
cd /d "%~dp0.."
echo [1/3] create venv...
if not exist .venv python -m venv .venv
echo [2/3] install core deps (aliyun mirror)...
.venv\Scripts\python.exe -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
.venv\Scripts\python.exe -m pip install akshare baostock pandas pyarrow scikit-learn lightgbm optuna statsmodels pyyaml requests "transformers==4.46.3" accelerate "certifi==2024.8.30"
echo [3/3] install GPU torch ~2.4GB (aliyun direct link, Python 3.12)...
.venv\Scripts\python.exe -m pip install "https://mirrors.aliyun.com/pytorch-wheels/cu121/torch-2.5.1%%2Bcu121-cp312-cp312-win_amd64.whl"
if errorlevel 1 (
    echo [TIP] GPU torch failed - try CPU: pip install torch ; or curl resume-download the URL then install locally
)
echo smoke test...
.venv\Scripts\python.exe tests\smoke_test.py
pause
