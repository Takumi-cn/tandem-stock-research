"""Regenerate arms A/B on the 21 non-quick folds used by MICRO-005-ENC-002.

The saved oos_pred_A/B parquets came from a --quick run (7 folds, 262 days) and
cannot be paired with the encoder's 21-fold OOS predictions.  This script keeps
the frozen LightGBM configuration, the same complete-case frame and seeds
(0,1,2), and writes separate 21-fold artifacts; the original minimal-test files
are left untouched.

Run:
  .venv\\Scripts\\python.exe -m src.eval.micro005_baseline_21f
"""
from __future__ import annotations

import json
import time

import pandas as pd

from src.eval.micro005_minimal import (DATA as MIN_DATA, build_frame, complete_case,
                                       daily_metrics, folds, log, run_arm)
from src.utils.config import ROOT

OUT = ROOT / "reports" / "micro005_encoder_002"
DATA = ROOT / "data" / "research" / "micro005_encoder_002"


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    df, info = build_frame()
    df = complete_case(df)
    DATA.mkdir(parents=True, exist_ok=True)
    df[["date", "code"]].drop_duplicates().to_parquet(DATA / "base_frame_keys.parquet", index=False)
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    fl = folds(dates, False)
    log(f"BASELINE21 frame rows={len(df):,} dates={len(dates)} folds={len(fl)}")
    a = run_arm(df, "daily", fl)
    log(f"BASELINE21 arm A done rows={len(a):,} oos_dates={a['date'].nunique()}")
    b = run_arm(df, "daily_intraday", fl)
    log(f"BASELINE21 arm B done rows={len(b):,} oos_dates={b['date'].nunique()}")
    a.to_parquet(DATA / "oos_pred_A_21f.parquet", index=False)
    b.to_parquet(DATA / "oos_pred_B_21f.parquet", index=False)
    da, db = daily_metrics(a), daily_metrics(b)
    da.to_csv(OUT / "daily_metrics_A_21f.csv", index=False, encoding="utf-8-sig")
    db.to_csv(OUT / "daily_metrics_B_21f.csv", index=False, encoding="utf-8-sig")
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "folds": len(fl),
        "rows_a": int(len(a)), "rows_b": int(len(b)),
        "frame_rows": int(len(df)), "frame_dates": int(len(dates)),
        "dates_a": int(a["date"].nunique()), "dates_b": int(b["date"].nunique()),
        "ic_a": float(da["ic"].mean()), "ic_b": float(db["ic"].mean()),
        "top1_a": float(da["top1_alpha"].mean()), "top1_b": float(db["top1_alpha"].mean()),
        "frame_info": info, "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    (OUT / "baseline_21f.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("BASELINE21_DONE " + json.dumps({k: meta[k] for k in ("folds", "ic_a", "ic_b")}))


if __name__ == "__main__":
    main()
