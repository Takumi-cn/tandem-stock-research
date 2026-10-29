"""Parallel rolling full-null worker for IND-REL-MICRO-003.

Each worker owns a disjoint run range and writes immutable null batches. It does
not change the frozen model, features, folds, costs or thresholds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NIGHT_N_JOBS", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval import industry_rel_micro_003 as h  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=10)
    args = ap.parse_args()
    h.OUT.mkdir(parents=True, exist_ok=True)
    h.LOGS.mkdir(parents=True, exist_ok=True)
    h.STAGES.mkdir(parents=True, exist_ok=True)
    panel = h.pd.read_parquet(h.RESEARCH / "features_panel.parquet")
    panel["date"] = h.pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype(str)
    dates = h.pd.DatetimeIndex(sorted(panel["date"].unique()))
    codes = sorted(panel["code"].unique())
    folds, signal_dates = h.prepare_folds(panel, codes, dates)
    G = h.build_gross(signal_dates, codes)
    rows, batch, batch_id = [], [], 0
    t0 = time.time()
    for rep in range(args.start, args.end):
        m = h.null_rep(folds, len(codes), signal_dates, G, 990001 + rep * 7919)
        m["run"] = rep
        m["worker"] = args.worker
        batch.append(m)
        if len(batch) >= args.batch_size or rep == args.end - 1:
            f = h.OUT / f"rolling_full_null_batch_w{args.worker}_{batch_id:03d}.parquet"
            h.pd.DataFrame(batch).to_parquet(f, index=False)
            rows.extend(batch)
            batch = []
            batch_id += 1
        done = rep - args.start + 1
        if done % 10 == 0:
            print(f"worker={args.worker} {done}/{args.end-args.start} elapsed={time.time()-t0:.1f}s", flush=True)
    h.atomic_json(h.OUT / f"null_worker_{args.worker}.json", {
        "worker": args.worker, "start": args.start, "end": args.end,
        "completed": len(rows), "elapsed_s": round(time.time() - t0, 1),
        "n_jobs": h.MODEL_PARAMS["n_jobs"],
    })
    print("NULL_WORKER_DONE", json.dumps({"worker": args.worker, "n": len(rows), "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
