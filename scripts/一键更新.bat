@echo off
chcp 65001 >nul
rem 每日双轨预测一键更新（周一至周五 17:30 由计划任务调用，亦可双击）
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe (
    echo [ERROR] venv not found - run scripts\安装环境.bat first
    exit /b 1
)
if not exist reports\runs mkdir reports\runs
set LOG=reports\runs\last_run.log
echo ===== ONE-KEY UPDATE START %date% %time% ===== >> "%LOG%"

echo [1/5] daily bars increment (baostock)...
.venv\Scripts\python.exe -m src.data.fetch_baostock >> "%LOG%" 2>&1
if errorlevel 1 echo   [WARN] bars update failed, using old data

echo [2/5] rebuild feature panel...
.venv\Scripts\python.exe -m src.features.build >> "%LOG%" 2>&1
if errorlevel 1 (
    echo   [ERROR] panel build failed - abort
    goto :end
)

echo [3/5] news + sentiment (optional)...
.venv\Scripts\python.exe -m src.data.fetch_news >> "%LOG%" 2>&1
if errorlevel 1 (
    echo   [INFO] news skipped
) else (
    .venv\Scripts\python.exe -m src.features.sentiment >> "%LOG%" 2>&1
    if errorlevel 1 echo   [INFO] sentiment skipped
)

echo [4/5] daily dual-track report (GPU ~1-2 min)...
.venv\Scripts\python.exe -m src.report.daily >> "%LOG%" 2>&1
if errorlevel 1 (
    echo   [ERROR] report failed - see reports\runs\last_run.log
    goto :end
)

echo [5/5] DONE. Report: reports\ (latest dated .md)

:end
echo ===== END %time% ===== >> "%LOG%"
