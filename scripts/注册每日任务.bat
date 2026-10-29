@echo off
chcp 65001 >nul
rem 注册每日计划任务（周一至周五 18:45，静默 pythonw，先增量数据再出报告）
rem 18:45 而非 17:30：baostock 当日日线在 17:30 尚未落库，17:30 跑会得到前一交易日的陈旧面板。
schtasks /Create /TN "StockDailyPredict" /TR "\"%~dp0..\.venv\Scripts\pythonw.exe\" \"%~dp0..\run_daily.pyw\"" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 18:45 /F
if errorlevel 1 (
    echo 注册失败 - 请右键以管理员身份运行本脚本重试
) else (
    echo 注册成功：每个交易日 18:45 自动更新数据并生成双轨报告
    echo 取消：schtasks /Delete /TN "StockDailyPredict" /F
)
pause
