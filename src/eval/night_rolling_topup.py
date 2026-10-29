"""Top up the pre-registered rolling label-permutation null in parallel.

Research-only helper for NIGHT-IND-CERT-002. It does not change any rule; it
only finishes the Level C rolling-null count (200) with deterministic run seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NIGHT_N_JOBS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval import night_ind_cert_002 as h  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=3)
    args = ap.parse_args()
    h.OUT.mkdir(parents=True, exist_ok=True)
    h.LOGS.mkdir(parents=True, exist_ok=True)
    h.STAGES.mkdir(parents=True, exist_ok=True)
    ctx = h.load_context()
    folds, panel_rank, finite_groups = h.prepare_rolling(ctx)
    rows, batch = [], []
    batch_id = 0
    t0 = time.time()
    for rep in range(args.start, args.end):
        rng = h.np.random.default_rng(880001 + rep * 104729)
        rank_perm = panel_rank.copy()
        for g in finite_groups:
            rank_perm[g] = rng.permutation(panel_rank[g])
        sm = h._rolling_scores_impl(ctx, folds, h.MAIN_SEED, rank_perm)
        p = h.portfolio_from_scores(sm, ctx, h.COST_BPS, False)
        m = p["metrics"]
        batch.append({
            "run": rep, "ann": m["ann"], "sharpe": m["sharpe"], "mdd": m["mdd"],
            "turnover": float(p["turnover"].mean()),
            "first_ann": float(h.perf(p["net"].values[:43])["ann"]),
            "second_ann": float(h.perf(p["net"].values[43:])["ann"]),
        })
        if len(batch) >= args.batch_size or rep == args.end - 1:
            f = h.OUT / f"full_null_rolling_batch_w{args.worker}_{batch_id:03d}.parquet"
            h.pd.DataFrame(batch).to_parquet(f, index=False)
            rows.extend(batch)
            batch = []
            batch_id += 1
        done = rep - args.start + 1
        if done % 3 == 0:
            print(f"worker={args.worker} {done}/{args.end-args.start} rep={rep} elapsed={time.time()-t0:.1f}s", flush=True)
    h.atomic_json(h.OUT / f"topup_worker_{args.worker}.json", {
        "worker": args.worker, "start": args.start, "end": args.end,
        "completed": len(rows), "elapsed_s": round(time.time() - t0, 1),
        "n_jobs": h.N_JOBS,
    })
    print("TOPUP_DONE", json.dumps({"worker": args.worker, "n": len(rows), "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
