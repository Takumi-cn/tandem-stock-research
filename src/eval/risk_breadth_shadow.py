"""Read-only breadth shadow logger for RISK-BREADTH-OVERLAY-001.

Run after the daily production report (e.g. 19:00) to append one row per
trading day.  It never writes latest.json, the model, the ledger, the 55%
threshold, or any production file; the only output is
reports/risk_breadth_shadow/history.csv.

Rule (pre-registered): entry_allowed = (p_up >= 0.55) and (breadth_T >= 0.40),
where breadth_T is the fraction of the 101-stock panel with pct_chg > 0 at T
close.  A false entry_allowed means "research shadow says no new entry", not a
production signal.

Run:
  .venv\\Scripts\\python.exe -m src.eval.risk_breadth_shadow --dry-run
  .venv\\Scripts\\python.exe -m src.eval.risk_breadth_shadow
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.utils.config import ROOT

PANEL = ROOT / "data" / "parquet" / "panel.parquet"
LATEST = ROOT / "reports" / "latest.json"
OUT = ROOT / "reports" / "risk_breadth_shadow"
HIST = OUT / "history.csv"
START = pd.Timestamp("2026-10-28")
BREADTH_MIN = 0.40
PUP_MIN = 0.55


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="signal date T; default latest.json T")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    panel = pd.read_parquet(PANEL, columns=["date", "code", "pct_chg"])
    panel["date"] = pd.to_datetime(panel["date"])
    latest = json.loads(LATEST.read_text(encoding="utf-8"))
    T = pd.Timestamp(a.date) if a.date else pd.Timestamp(latest["T"])
    day = panel[panel["date"] == T]
    if len(day) == 0:
        raise SystemExit(f"no panel rows for T={T.date()}")
    breadth = float((day["pct_chg"] > 0).mean())
    short = latest.get("short", {})
    p_up = float(short.get("win_prob", float("nan")))
    code = str(short.get("code", ""))
    allowed = bool(pd.notna(p_up) and p_up >= PUP_MIN and breadth >= BREADTH_MIN)
    row = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "T": str(T.date()),
        "phase": "forward" if T >= START else "pre_start",
        "breadth": round(breadth, 4),
        "p_up": round(p_up, 4) if pd.notna(p_up) else None,
        "signal_code": code,
        "entry_allowed": allowed,
        "rule": f"p_up>={PUP_MIN} and breadth>={BREADTH_MIN}",
        "note": "research shadow only; no production write",
    }
    if a.dry_run:
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if HIST.exists():
        old = pd.read_csv(HIST, dtype={"signal_code": str})
        if (old["T"].astype(str) == row["T"]).any():
            print(json.dumps({"skipped": "date already recorded", "T": row["T"]}, ensure_ascii=False))
            return
        old = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        old = pd.DataFrame([row])
    old.to_csv(HIST, index=False, encoding="utf-8-sig")
    print(json.dumps({"appended": row, "path": str(HIST)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
