"""17:30 静默日常任务：先增量数据/重建面板，再生成双轨报告。

修复（2026-10-26）：旧版直接调用 daily.main()，不抓数据、不重建面板，导致计划任务
每天只重算同一份陈旧面板。现在与完整管线共用 step_data，并在数据未更新时重试。
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.update import panel_last_date, step_data  # noqa: E402
from src.report.daily import main  # noqa: E402
from src.report.ledger import stale_note  # noqa: E402


def refresh(max_tries: int = 6, wait_s: int = 360) -> str:
    """Retry until the panel reaches today (or the deadline).

    2026-10-27 incident: baostock published the T-day daily bar only around
    19:00, after the 18:45 start plus 3x4min retries, so the report was built on
    yesterday's panel.  Six retries with 6-minute spacing keep trying until
    ~19:15-19:35 while still producing a report if the data never arrives.
    """
    note = ""
    for i in range(max_tries):
        step_data()
        note = stale_note(panel_last_date())
        if not note:
            return ""
        print(f"[run_daily] 第 {i+1}/{max_tries} 次数据更新后仍陈旧：{note}", flush=True)
        if i + 1 < max_tries:
            time.sleep(wait_s)
    return note


def main_entry() -> None:
    note = refresh()
    main()
    if note:
        log = ROOT / "reports" / "runs" / "data_stale.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')} {note}\n")


if __name__ == "__main__":
    main_entry()
