"""运行日志：把每次运行的结果追加到 reports/runs/*.log。

- gui_run.log：完整管线（GUI / 命令行 run_full）的运行摘要与进度
- last_run.log：每日任务（run_daily / 一键更新）的运行摘要
UTF-8 追加；写日志失败绝不影响主流程。
"""
from pathlib import Path

import pandas as pd

from src.utils.config import ROOT


def append_run(log_name: str, source: str, payload: dict) -> None:
    try:
        path = ROOT / "reports" / "runs" / log_name
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"\n===== RUN {pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | {source} ====="]
        for k, v in payload.items():
            lines.append(f"- {k}: {v}")
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:  # noqa: BLE001
        pass
