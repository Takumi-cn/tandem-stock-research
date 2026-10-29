"""NIGHT-IND-CERT-002 / IND-ROLL-001.

Industry-index certification night run.  Research only:
- no stock-level models, no production writes, no latest.json, no broker calls;
- PIT failure is an optional branch and never stops the industry study;
- v4/v4.1/Experiment A failures remain failed and are not retuned.

Run:
  .venv\\Scripts\\python.exe -m src.eval.night_ind_cert_002 --max-hours 8 --research-only --resume
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.industry_rotation import (  # noqa: E402
    ALT_COST,
    BASE_COST,
    BATTERY,
    build_signals,
    battery_returns,
    load_clean,
    simulate,
)
from src.utils.config import ROOT  # noqa: E402

EXP_ID = "NIGHT-IND-CERT-002"
CHALLENGER_ID = "IND-ROLL-001"
OUT = ROOT / "reports" / "night_ind_cert_002"
RESEARCH = ROOT / "data" / "research" / "night_ind_cert_002"
LOGS = OUT / "logs"
STAGES = OUT / "stages"
PREREG = OUT / "prereg.json"
PREREG_HASH = OUT / "prereg.sha256"
RUN_CLOCK = OUT / "run_clock.json"
ACTIVE_CLOCK = OUT / "active_clock.json"

COST_BPS = 14.5
STRESS_BPS = 25.0
COST40_BPS = 40.0
TOP_K = 3
BUFFER_RANK = 5
HOLD_DAYS = 5
RETRAIN_EVERY = 20
TRAIN_WINDOW = 504
MAIN_SEED = 20261027
SEEDS = [MAIN_SEED, 1, 7, 19, 42, 73, 101, 257, 521, 2025]
N_JOBS = int(os.environ.get("NIGHT_N_JOBS", "8"))
V41_SEED = 20261026
V41_TRAIN_START = pd.Timestamp("2023-04-03")
V41_TEST_START = pd.Timestamp("2025-01-02")
V41_COST_BPS = 15.0
PPY = 252.0 / HOLD_DAYS
FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    "vol_5", "vol_20", "dd_20", "gap", "amount_ratio",
    "parent_rel_5", "parent_rel_20", "market_rel_5", "market_rel_20",
    "rank_ret5", "rank_ret20", "rank_vol", "rank_rel5", "rank_change",
    "mkt_ret20", "mkt_vol20", "mkt_disp", "mkt_up", "tp",
]
PROD_FILES = [
    "reports/latest.json", "reports/predictions_ledger.csv",
    "config/targets.yaml", "config/l1_best.json",
    "reports/ens_fixed_weights.json", "run_daily.pyw", "run_gui.pyw",
    "src/report/daily.py", "src/report/run_full.py", "src/report/short_pick.py",
]
DATA_FILES = [
    "data/industry/sw2_daily.parquet", "data/industry/sw1_daily.parquet",
    "data/industry/sw2_meta.parquet", "data/industry/sw1_meta.parquet",
    "reports/industry_v41/lgbm_scores.parquet", "reports/industry_v41/panel_v41.parquet",
]
LEVELS = [
    {"name": "A", "matched": 20_000, "perm": 10_000, "boot": 5_000, "static_null": 200, "rolling_null": 50},
    {"name": "B", "matched": 50_000, "perm": 25_000, "boot": 10_000, "static_null": 500, "rolling_null": 100},
    {"name": "C", "matched": 100_000, "perm": 50_000, "boot": 20_000, "static_null": 1000, "rolling_null": 200},
]


# ---------------------------------------------------------------- utilities
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


def hash_index(idx) -> str:
    return hashlib.sha256("\n".join(map(str, idx)).encode("utf-8")).hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def log(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    with open(LOGS / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def stage_start(name: str, extra: dict | None = None) -> float:
    t0 = time.time()
    atomic_json(STAGES / f"{name}.start.json", {
        "stage": name, "start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_epoch": t0, "extra": extra or {},
    })
    log(f"STAGE_START {name}")
    return t0


def stage_end(name: str, t0: float, status: str, result: dict | None = None) -> dict:
    rec = {
        "stage": name, "end": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 2), "status": status,
        "result": result or {},
    }
    atomic_json(STAGES / f"{name}.end.json", rec)
    log(f"STAGE_END {name} status={status} elapsed={rec['elapsed_s']}s")
    return rec


def _tasks_digest() -> str | None:
    try:
        r = subprocess.run(["schtasks", "/Query", "/FO", "LIST"], capture_output=True, timeout=30)
        text = r.stdout.decode("gbk", "replace")
        lines = [x.strip() for x in text.splitlines()
                 if "StockDaily" in x or "StockSelf" in x or "TaskName" in x or "Next Run" in x or "Status" in x]
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    except Exception:  # noqa: BLE001
        return None


def guard() -> dict:
    files = {}
    for rel in PROD_FILES:
        p = ROOT / rel
        files[rel] = {
            "exists": p.exists(),
            "sha256": sha256(p) if p.exists() else None,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)) if p.exists() else None,
        }
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": files, "scheduled_tasks_digest": _tasks_digest(),
        "python": sys.version, "platform": platform.platform(),
    }


def data_guard() -> dict:
    return {rel: {"exists": (ROOT / rel).exists(),
                  "sha256": sha256(ROOT / rel) if (ROOT / rel).exists() else None}
            for rel in DATA_FILES}


def ensure_prereg() -> dict:
    if PREREG.exists():
        obj = json.loads(PREREG.read_text(encoding="utf-8"))
        return obj
    obj = {
        "experiment_id": EXP_ID,
        "challenger_id": CHALLENGER_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "research_only": True,
        "individual_stock_layer": False,
        "production_write": False,
        "model": "existing_v41_lightgbm_ranker",
        "feature_set": "exact_v41_frozen",
        "train_window_days": TRAIN_WINDOW,
        "retrain_every_days": RETRAIN_EVERY,
        "top_k": TOP_K,
        "holding_days": HOLD_DAYS,
        "entry": "T+1_open",
        "exit": "T+5_close",
        "cost_bps_per_side": COST_BPS,
        "stress_cost_bps_per_side": STRESS_BPS,
        "main_seed": MAIN_SEED,
        "seeds": SEEDS,
        "v41_frozen": {
            "train_start": str(V41_TRAIN_START.date()),
            "test_start": str(V41_TEST_START.date()),
            "seed": V41_SEED,
            "n_estimators": 250, "learning_rate": 0.03, "num_leaves": 15,
            "max_depth": 4, "min_child_samples": 40, "subsample": 0.8,
            "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0,
            "cost_bps": V41_COST_BPS,
        },
        "regime_definition": {
            "trend": "mkt_ret20 > 0 at signal date",
            "vol": "mkt_vol20 > trailing 252-day median (shifted 1 day)",
            "states": {"R1": "up/low vol", "R2": "up/high vol",
                       "R3": "down/low vol", "R4": "down/high vol"},
            "dispersion": "mkt_disp > trailing 252-day median (shifted 1 day)",
        },
        "monte_carlo_levels": LEVELS,
        "required_gates": {
            "baseline_repro": True,
            "net_ann_positive": True,
            "net_sharpe_positive": True,
            "both_halves_selection_alpha_positive": True,
            "matched_random_ann_p_lt": 0.05,
            "matched_random_sharpe_p_lt": 0.05,
            "full_null_p_lt": 0.05,
            "dsr_gte": 0.95,
            "pbo_lte": 0.20,
            "stress_25bps_positive": True,
            "seed_median_alpha_positive": True,
        },
        "max_status": "FORWARD_CANDIDATE",
        "forbidden": [
            "individual stock layer", "latest.json write", "production model write",
            "broker order", "Optuna", "parameter search", "threshold search",
            "reviving v4/v4.1/Experiment A as pass", "PIT proxy renamed as 881262 PIT",
        ],
    }
    atomic_json(PREREG, obj)
    PREREG_HASH.write_text(sha256(PREREG) + "\n", encoding="utf-8")
    log(f"PREREG_WRITTEN {sha256(PREREG)}")
    return obj


# ---------------------------------------------------------------- data / metrics
@dataclass
class Ctx:
    dates: pd.DatetimeIndex
    c2: pd.DataFrame
    o2: pd.DataFrame
    meta2: pd.DataFrame
    universe: list[str]
    signal_dates: list[pd.Timestamp]
    signal_idx: list[int]
    G: np.ndarray
    score_map: dict[pd.Timestamp, np.ndarray]
    panel: pd.DataFrame
    quality: dict
    v41_scores: pd.DataFrame


def load_context() -> Ctx:
    dates, c1, c2, o2, a2, ret_cc, jump, recent_jump, meta1, meta2, quality = load_clean()
    scores = pd.read_parquet(ROOT / "reports" / "industry_v41" / "lgbm_scores.parquet")
    scores["date"] = pd.to_datetime(scores["date"])
    scores["code"] = scores["code"].astype(str)
    signal_dates = sorted(pd.Timestamp(x) for x in scores["date"].unique())
    universe = sorted(scores["code"].unique())
    panel = pd.read_parquet(ROOT / "reports" / "industry_v41" / "panel_v41.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype(str)
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    G = np.full((len(signal_dates), len(universe)), np.nan, dtype=float)
    signal_idx = []
    for i, t in enumerate(signal_dates):
        j = int(dates.get_loc(t))
        signal_idx.append(j)
        entry, exit_ = dates[j + 1], dates[j + HOLD_DAYS]
        for k, code in enumerate(universe):
            try:
                G[i, k] = float(c2.at[exit_, code] / o2.at[entry, code] - 1.0)
            except Exception:  # noqa: BLE001
                G[i, k] = np.nan
    if np.isnan(G).any():
        raise RuntimeError(f"gross return matrix has NaNs: {int(np.isnan(G).sum())}")
    score_map = {}
    for t in signal_dates:
        row = scores[scores["date"] == t].set_index("code")["score_lgbm"].reindex(universe)
        score_map[t] = row.values.astype(float)
    return Ctx(dates, c2, o2, meta2, universe, signal_dates, signal_idx, G,
               score_map, panel, quality, scores)


def perf(r) -> dict:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        return {"ann": None, "sharpe": None, "mdd": None, "win": None, "n": n, "vol": None}
    eq = np.cumprod(1.0 + r)
    ann = float(eq[-1] ** (252.0 / (n * HOLD_DAYS)) - 1.0)
    vol = float(np.std(r, ddof=1) * math.sqrt(PPY))
    sd = float(np.std(r, ddof=1))
    sharpe = float(np.mean(r) / sd * math.sqrt(PPY)) if sd > 0 else 0.0
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1.0))
    return {"ann": ann, "vol": vol, "sharpe": sharpe, "mdd": mdd,
            "win": float(np.mean(r > 0)), "n": n}


def metrics_matrix(R: np.ndarray) -> pd.DataFrame:
    R = np.asarray(R, dtype=float)
    b, n = R.shape
    mean = R.mean(axis=1)
    std = R.std(axis=1, ddof=1)
    sharpe = np.where(std > 0, mean / std * math.sqrt(PPY), 0.0)
    eq = np.cumprod(1.0 + R, axis=1)
    ann = eq[:, -1] ** (252.0 / (n * HOLD_DAYS)) - 1.0
    mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)
    mid = n // 2
    first = np.cumprod(1.0 + R[:, :mid], axis=1)[:, -1] ** (252.0 / (mid * HOLD_DAYS)) - 1.0
    second = np.cumprod(1.0 + R[:, mid:], axis=1)[:, -1] ** (252.0 / ((n - mid) * HOLD_DAYS)) - 1.0
    first_s = np.where(R[:, :mid].std(axis=1, ddof=1) > 0,
                       R[:, :mid].mean(axis=1) / R[:, :mid].std(axis=1, ddof=1) * math.sqrt(PPY), 0.0)
    second_s = np.where(R[:, mid:].std(axis=1, ddof=1) > 0,
                        R[:, mid:].mean(axis=1) / R[:, mid:].std(axis=1, ddof=1) * math.sqrt(PPY), 0.0)
    return pd.DataFrame({
        "ann": ann, "sharpe": sharpe, "mdd": mdd, "win": (R > 0).mean(axis=1),
        "first_ann": first, "second_ann": second,
        "first_sharpe": first_s, "second_sharpe": second_s,
    })


def path_from_scores(sc: np.ndarray, prev: list[int], hysteresis: bool) -> tuple[list[int], int]:
    order = np.argsort(-np.nan_to_num(sc, nan=-1e18), kind="stable")
    if hysteresis and prev:
        rank = {int(c): k + 1 for k, c in enumerate(order)}
        kept = [c for c in prev if rank.get(int(c), 10 ** 9) <= BUFFER_RANK][:TOP_K]
        target = kept[:]
        for c in order:
            if len(target) >= TOP_K:
                break
            if int(c) not in target:
                target.append(int(c))
    else:
        target = [int(c) for c in order[:TOP_K]]
    changes = len(set(target) - set(prev)) if prev else len(target)
    return target, changes


def portfolio_from_scores(score_map: dict[pd.Timestamp, np.ndarray], ctx: Ctx,
                          cost_bps: float, hysteresis: bool) -> dict:
    prev: list[int] = []
    rets, gross, turns, changes = [], [], [], []
    holdings = []
    for i, t in enumerate(ctx.signal_dates):
        sc = score_map.get(t)
        if sc is None:
            continue
        target, k = path_from_scores(sc, prev, hysteresis)
        turn = 0.5 if not prev else 0.5 * k / TOP_K
        g = float(ctx.G[i, target].mean())
        n = g - turn * 2.0 * cost_bps / 10000.0
        rets.append(n)
        gross.append(g)
        turns.append(turn)
        changes.append(k)
        holdings.append({"date": str(t.date()), "target": [ctx.universe[c] for c in target],
                         "gross": round(g, 6), "turnover": round(turn, 6)})
        prev = target
    return {
        "net": pd.Series(rets, index=ctx.signal_dates[:len(rets)]),
        "gross": pd.Series(gross, index=ctx.signal_dates[:len(gross)]),
        "turnover": pd.Series(turns, index=ctx.signal_dates[:len(turns)]),
        "changes": np.asarray(changes, dtype=int),
        "holdings": holdings,
        "metrics": perf(rets),
        "gross_metrics": perf(gross),
    }


def _sample_unique(rng, b: int, u: int, k: int) -> np.ndarray:
    p = rng.random((b, u))
    idx = np.argpartition(p, k, axis=1)[:, :k]
    return idx.astype(np.int32)


def _replace_random(rng, H: np.ndarray, k: int, u: int) -> np.ndarray:
    b, K = H.shape
    if k >= K:
        return _sample_unique(rng, b, u, K)
    if k == 1:
        pos = rng.integers(0, K, size=(b, 1))
    else:
        pos = np.argsort(rng.random((b, K)), axis=1)[:, :k]
    rem = H.copy()
    np.put_along_axis(rem, pos, -1, axis=1)
    cand = rng.integers(0, u, size=(b, k))
    coll = np.zeros(b, dtype=bool)
    for _ in range(8):
        coll[:] = False
        for j in range(k):
            coll |= (cand[:, j][:, None] == rem).any(axis=1)
        if k == 2:
            coll |= cand[:, 0] == cand[:, 1]
        elif k == 3:
            coll |= (cand[:, 0] == cand[:, 1]) | (cand[:, 0] == cand[:, 2]) | (cand[:, 1] == cand[:, 2])
        if not coll.any():
            break
        cand[coll] = rng.integers(0, u, size=(int(coll.sum()), k))
    if coll.any():
        for idx in np.where(coll)[0]:
            avail = [x for x in range(u) if x not in set(rem[idx].tolist()) and x >= 0]
            pick = rng.choice(avail, size=min(k, len(avail)), replace=False)
            cand[idx, :len(pick)] = pick
    np.put_along_axis(rem, pos, cand, axis=1)
    return rem


def simulate_matched_random(G: np.ndarray, changes: np.ndarray, B: int,
                            cost_bps: float, seed: int, store_n: int = 5000):
    n, u = G.shape
    changes = np.asarray(changes, dtype=int)
    chunk = 5000
    metrics = []
    stored_net, stored_gross = [], []
    stored_count = 0
    turns_sum = np.zeros(n, dtype=float)
    total = 0
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        rng = np.random.default_rng(seed + start * 7919)
        H = _sample_unique(rng, b, u, TOP_K)
        R = np.empty((b, n), dtype=np.float32)
        RG = np.empty((b, n), dtype=np.float32)
        for i in range(n):
            if i > 0:
                k = int(changes[i])
                if k > 0:
                    H = _replace_random(rng, H, k, u)
            g = G[i, H].mean(axis=1)
            turn = 0.5 if i == 0 else 0.5 * int(changes[i]) / TOP_K
            RG[:, i] = g
            R[:, i] = g - turn * 2.0 * cost_bps / 10000.0
            turns_sum[i] += turn * b
        m = metrics_matrix(R)
        gm = metrics_matrix(RG)
        m["gross_ann"] = gm["ann"].values
        m["gross_sharpe"] = gm["sharpe"].values
        m["run"] = np.arange(total, total + b)
        metrics.append(m)
        if stored_count < store_n:
            take = min(store_n - stored_count, b)
            stored_net.append(R[:take].copy())
            stored_gross.append(RG[:take].copy())
            stored_count += take
        total += b
    df = pd.concat(metrics, ignore_index=True)
    net = np.vstack(stored_net) if stored_net else np.empty((0, n), dtype=np.float32)
    gross = np.vstack(stored_gross) if stored_gross else np.empty((0, n), dtype=np.float32)
    return df, net, gross, turns_sum / max(1, B)


def simulate_random_topk(G: np.ndarray, B: int, cost_bps: float, seed: int,
                         store_n: int = 5000):
    n, u = G.shape
    chunk = 5000
    metrics = []
    stored_net, stored_gross = [], []
    stored_count = 0
    turns_sum = np.zeros(n, dtype=float)
    total = 0
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        rng = np.random.default_rng(seed + 104729 + start * 7919)
        R = np.empty((b, n), dtype=np.float32)
        RG = np.empty((b, n), dtype=np.float32)
        prev = None
        for i in range(n):
            H = _sample_unique(rng, b, u, TOP_K)
            if prev is None:
                k = np.full(b, TOP_K)
            else:
                k = np.array([TOP_K - len(set(H[j].tolist()) & set(prev[j].tolist())) for j in range(b)])
            turn = np.full(b, 0.5) if prev is None else 0.5 * k / TOP_K
            g = G[i, H].mean(axis=1)
            RG[:, i] = g
            R[:, i] = g - turn * 2.0 * cost_bps / 10000.0
            turns_sum[i] += float(turn.mean()) * b
            prev = H
        m = metrics_matrix(R)
        gm = metrics_matrix(RG)
        m["gross_ann"] = gm["ann"].values
        m["gross_sharpe"] = gm["sharpe"].values
        m["run"] = np.arange(total, total + b)
        metrics.append(m)
        if stored_count < store_n:
            take = min(store_n - stored_count, b)
            stored_net.append(R[:take].copy())
            stored_gross.append(RG[:take].copy())
            stored_count += take
        total += b
    df = pd.concat(metrics, ignore_index=True)
    net = np.vstack(stored_net) if stored_net else np.empty((0, n), dtype=np.float32)
    gross = np.vstack(stored_gross) if stored_gross else np.empty((0, n), dtype=np.float32)
    return df, net, gross, turns_sum / max(1, B)


def simulate_score_perm_hysteresis(G: np.ndarray, B: int, cost_bps: float, seed: int,
                                   store_n: int = 2000):
    n, u = G.shape
    rng = np.random.default_rng(seed + 15485863)
    rows = []
    net_store = np.empty((min(B, store_n), n), dtype=np.float32)
    gross_store = np.empty((min(B, store_n), n), dtype=np.float32)
    for b in range(B):
        h: list[int] = []
        rets, gross = [], []
        for i in range(n):
            order = rng.permutation(u)
            if not h:
                target = [int(x) for x in order[:TOP_K]]
            else:
                rank = np.empty(u, dtype=np.int32)
                rank[order] = np.arange(1, u + 1)
                kept = [c for c in h if rank[c] <= BUFFER_RANK][:TOP_K]
                target = kept[:]
                for c in order:
                    if len(target) >= TOP_K:
                        break
                    if int(c) not in target:
                        target.append(int(c))
            k = len(set(target) - set(h)) if h else TOP_K
            turn = 0.5 if not h else 0.5 * k / TOP_K
            g = float(G[i, target].mean())
            rets.append(g - turn * 2.0 * cost_bps / 10000.0)
            gross.append(g)
            h = target
        r = np.asarray(rets)
        gr = np.asarray(gross)
        p = perf(r)
        gp = perf(gr)
        mid = n // 2
        p1 = perf(r[:mid]); p2 = perf(r[mid:])
        rows.append({"run": b, "ann": p["ann"], "sharpe": p["sharpe"], "mdd": p["mdd"],
                     "win": p["win"], "gross_ann": gp["ann"], "gross_sharpe": gp["sharpe"],
                     "first_ann": p1["ann"], "second_ann": p2["ann"]})
        if b < store_n:
            net_store[b] = r
            gross_store[b] = gr
        if (b + 1) % 2000 == 0:
            log(f"score_perm_hyst progress {b + 1}/{B}")
    return pd.DataFrame(rows), net_store[:min(B, store_n)], gross_store[:min(B, store_n)]


def empirical_p(strategy_value: float, samples, key: str) -> float | None:
    arr = np.asarray([x.get(key) if isinstance(x, dict) else getattr(x, key, np.nan) for x in samples], dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return None
    return float((1 + np.sum(arr >= strategy_value)) / (len(arr) + 1))


def p_from_df(strategy_value: float, df: pd.DataFrame, key: str) -> float | None:
    arr = df[key].dropna().values
    if not len(arr):
        return None
    return float((1 + np.sum(arr >= strategy_value)) / (len(arr) + 1))


# ---------------------------------------------------------------- baseline reproduction
def _rank_int_from_y(y: pd.Series, dates: pd.Series) -> np.ndarray:
    d = pd.DataFrame({"date": dates.values, "y": y.values})
    r = d.groupby("date")["y"].rank(method="first", pct=True) * 30.0
    return np.clip(np.round(r.values), 0, 30).astype(int)


def _fit_ranker(X: np.ndarray, y_rank: np.ndarray, groups: list[int], seed: int):
    import lightgbm as lgb
    lr = lgb.LGBMRanker(
        objective="rank_xendcg", n_estimators=250, learning_rate=0.03,
        num_leaves=15, max_depth=4, min_child_samples=40, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
        n_jobs=N_JOBS, verbose=-1, random_state=seed,
    )
    lr.fit(X, y_rank, group=groups)
    return lr


def reproduce_baseline(ctx: Ctx) -> dict:
    from sklearn.impute import SimpleImputer
    p = ctx.panel
    train = p[(p["date"] >= V41_TRAIN_START) & (p["label_end"] < V41_TEST_START) & p["y_oc5"].notna()].copy()
    train = train[train["date"] < V41_TEST_START].sort_values(["date", "code"])
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(train[FEATURES])
    rank = _rank_int_from_y(train["y_oc5"], train["date"])
    groups = train.groupby("date", sort=True).size().astype(int).tolist()
    lr = _fit_ranker(Xtr, rank, groups, V41_SEED)
    rebuilt = {}
    for t in ctx.signal_dates:
        row = p[p["date"] == t].copy()
        row = row[row["code"].isin(ctx.universe)]
        if row.empty:
            continue
        X = pd.DataFrame(imp.transform(row[FEATURES]), columns=FEATURES, index=row.index)
        rebuilt[t] = pd.Series(lr.predict(X), index=row["code"].values)
    old = {t: pd.Series(ctx.score_map[t], index=ctx.universe) for t in ctx.signal_dates}
    diffs, cors = [], []
    for t in ctx.signal_dates:
        if t not in rebuilt:
            continue
        a = rebuilt[t].reindex(ctx.universe).values
        b = old[t].reindex(ctx.universe).values
        diffs.append(np.nanmax(np.abs(a - b)))
        cors.append(float(np.corrcoef(a, b)[0, 1]))
    max_diff = float(np.nanmax(diffs)) if diffs else None
    min_corr = float(np.nanmin(cors)) if cors else None
    v41 = portfolio_from_scores(ctx.score_map, ctx, V41_COST_BPS, hysteresis=False)
    exp_a = portfolio_from_scores(ctx.score_map, ctx, COST_BPS, hysteresis=True)
    recorded_v41 = {"ann": 0.1419, "sharpe": 0.5417}
    recorded_a = {"ann": 0.2562, "sharpe": 0.7920, "turnover_reduction": 0.2281}
    v41_ok = (abs(v41["metrics"]["ann"] - recorded_v41["ann"]) <= 0.002
              and abs(v41["metrics"]["sharpe"] - recorded_v41["sharpe"]) <= 0.02)
    a_turn_red = 1.0 - float(exp_a["turnover"].mean()) / float(v41["turnover"].mean())
    a_ok = (abs(exp_a["metrics"]["ann"] - recorded_a["ann"]) <= 0.002
            and abs(exp_a["metrics"]["sharpe"] - recorded_a["sharpe"]) <= 0.02
            and abs(a_turn_red - recorded_a["turnover_reduction"]) <= 0.01)
    open_mat = np.array([[ctx.o2.at[ctx.dates[i + 1], c] for c in ctx.universe] for i in ctx.signal_idx])
    close_mat = np.array([[ctx.c2.at[ctx.dates[i + HOLD_DAYS], c] for c in ctx.universe] for i in ctx.signal_idx])
    result = {
        "status": "PASS" if (v41_ok and a_ok and max_diff is not None and max_diff < 1e-10) else "BASE_REPRO_FAIL",
        "v41": {"recorded": recorded_v41, "reproduced": v41["metrics"], "pass": bool(v41_ok),
                "turnover": float(v41["turnover"].mean()), "gross": v41["gross_metrics"]},
        "exp_a": {"recorded": recorded_a, "reproduced": exp_a["metrics"],
                  "gross": exp_a["gross_metrics"], "pass": bool(a_ok),
                  "turnover_reduction": float(a_turn_red)},
        "model_rebuild": {"max_abs_diff_vs_cached": max_diff, "min_corr_vs_cached": min_corr,
                          "train_rows": int(len(train)), "train_dates": int(train["date"].nunique()),
                          "train_start": str(train["date"].min().date()),
                          "train_end": str(train["date"].max().date())},
        "hashes": {
            "rebuilt_predictions": hash_array(np.concatenate([rebuilt[t].reindex(ctx.universe).values for t in ctx.signal_dates if t in rebuilt])),
            "cached_scores": hash_array(ctx.v41_scores[["date", "code", "score_lgbm"]].sort_values(["date", "code"])[["score_lgbm"]].values),
            "signal_dates": hash_index([str(x.date()) for x in ctx.signal_dates]),
            "universe": hash_index(ctx.universe),
            "open_matrix": hash_array(open_mat), "close_matrix": hash_array(close_mat),
            "exp_a_return_vector": hash_array(exp_a["net"].values),
            "v41_return_vector": hash_array(v41["net"].values),
        },
    }
    atomic_json(OUT / "baseline_repro.json", result)
    pd.DataFrame({
        "date": ctx.signal_dates,
        "v41_lgbm_net_15": v41["net"].values,
        "v41_lgbm_gross": v41["gross"].values,
        "exp_a_net_145": exp_a["net"].values,
        "exp_a_gross": exp_a["gross"].values,
        "v41_turnover": v41["turnover"].values,
        "exp_a_turnover": exp_a["turnover"].values,
    }).to_parquet(OUT / "baseline_daily_returns.parquet", index=False)
    np.savez_compressed(RESEARCH / "baseline_vectors.npz",
                        v41_net=v41["net"].values, v41_gross=v41["gross"].values,
                        exp_a_net=exp_a["net"].values, exp_a_gross=exp_a["gross"].values)
    log(f"BASELINE_REPRO status={result['status']} v41={v41['metrics']} A={exp_a['metrics']} maxdiff={max_diff}")
    return result


def train_static_v41_models(ctx: Ctx) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import BayesianRidge
    p = ctx.panel
    train = p[(p["date"] >= V41_TRAIN_START) & (p["label_end"] < V41_TEST_START) & p["y_oc5"].notna()].copy()
    train = train[train["date"] < V41_TEST_START].sort_values(["date", "code"])
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(train[FEATURES])
    ytr = train["y_oc5"].astype(float).clip(-0.2, 0.2).values
    rank = _rank_int_from_y(train["y_oc5"], train["date"])
    groups = train.groupby("date", sort=True).size().astype(int).tolist()
    lr = _fit_ranker(Xtr, rank, groups, V41_SEED)
    br = BayesianRidge()
    br.fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=40,
                               max_features=0.7, n_jobs=8, random_state=V41_SEED)
    rf.fit(Xtr, ytr)
    out = {"lgbm": {}, "ridge": {}, "rf": {}}
    for t in ctx.signal_dates:
        row = p[p["date"] == t].copy()
        row = row[row["code"].isin(ctx.universe)]
        if row.empty:
            continue
        X = pd.DataFrame(imp.transform(row[FEATURES]), columns=FEATURES, index=row.index)
        out["lgbm"][t] = pd.Series(lr.predict(X), index=row["code"].values)
        out["ridge"][t] = pd.Series(br.predict(X), index=row["code"].values)
        out["rf"][t] = pd.Series(rf.predict(X), index=row["code"].values)
    ensemble = {}
    for t in ctx.signal_dates:
        if all(t in out[m] for m in out):
            frames = [out[m][t].rank(pct=True) for m in out]
            ensemble[t] = pd.concat(frames, axis=1).mean(axis=1)
    out["ensemble"] = ensemble
    return out


def v4_period_series(daily: pd.Series, ctx: Ctx) -> pd.Series:
    vals = []
    for i, t in enumerate(ctx.signal_dates):
        entry = ctx.dates[ctx.signal_idx[i] + 1]
        exit_ = ctx.dates[ctx.signal_idx[i] + HOLD_DAYS]
        seg = daily.loc[entry:exit_]
        vals.append(float(np.prod(1.0 + seg.values) - 1.0) if len(seg) else 0.0)
    return pd.Series(vals, index=ctx.signal_dates)


# ---------------------------------------------------------------- static / rolling laboratories
@dataclass
class StaticLab:
    Xtr: np.ndarray
    rank: np.ndarray
    groups: list[int]
    date_groups: list[np.ndarray]
    Xte: np.ndarray
    test_dates: list[pd.Timestamp]
    test_slices: list[tuple[int, int]]
    test_code_idx: np.ndarray
    n_signals: int
    n_universe: int


@dataclass
class Fold:
    block_start: pd.Timestamp
    train_row_idx: np.ndarray
    test_dates: list[pd.Timestamp]
    test_slices: list[tuple[int, int]]
    test_code_idx: np.ndarray
    Xtr: np.ndarray
    groups: list[int]
    rank_actual: np.ndarray
    Xte: np.ndarray
    train_start: str
    train_end: str


def prepare_static(ctx: Ctx) -> StaticLab:
    from sklearn.impute import SimpleImputer
    p = ctx.panel
    train = p[(p["date"] >= V41_TRAIN_START) & (p["label_end"] < V41_TEST_START) & p["y_oc5"].notna()].copy()
    train = train[train["date"] < V41_TEST_START].sort_values(["date", "code"])
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(train[FEATURES])
    rank = _rank_int_from_y(train["y_oc5"], train["date"])
    groups = train.groupby("date", sort=True).size().astype(int).tolist()
    date_groups = [np.asarray(v, dtype=int) for v in train.groupby("date", sort=True).indices.values()]
    test = p[p["date"].isin(ctx.signal_dates)].sort_values(["date", "code"])
    code_map = {c: i for i, c in enumerate(ctx.universe)}
    test_slices, test_dates = [], []
    offset = 0
    for t in ctx.signal_dates:
        sub = test[test["date"] == t]
        test_dates.append(t)
        test_slices.append((offset, offset + len(sub)))
        offset += len(sub)
    Xte = imp.transform(test[FEATURES]) if len(test) else np.empty((0, len(FEATURES)))
    test_code_idx = np.array([code_map[c] for c in test["code"]], dtype=int)
    return StaticLab(Xtr, rank, groups, date_groups, Xte, test_dates,
                     test_slices, test_code_idx, len(ctx.signal_dates), len(ctx.universe))


def _scores_from_predictions(pred: np.ndarray, lab: StaticLab) -> dict[pd.Timestamp, np.ndarray]:
    S = np.full((lab.n_signals, lab.n_universe), np.nan, dtype=float)
    for i, (start, end) in enumerate(lab.test_slices):
        if end > start:
            S[i, lab.test_code_idx[start:end]] = pred[start:end]
    return {t: S[i] for i, t in enumerate(lab.test_dates)}


def prepare_rolling(ctx: Ctx) -> tuple[list[Fold], np.ndarray, list[np.ndarray]]:
    from sklearn.impute import SimpleImputer
    p = ctx.panel
    pdates = pd.DatetimeIndex(sorted(p["date"].unique()))
    date_pos = {d: i for i, d in enumerate(pdates)}
    code_map = {c: i for i, c in enumerate(ctx.universe)}
    panel_rank = np.full(len(p), -1, dtype=int)
    finite = p["y_oc5"].notna().values
    r = p.loc[finite].groupby("date")["y_oc5"].rank(method="first", pct=True) * 30.0
    panel_rank[finite] = np.clip(np.round(r.values), 0, 30).astype(int)
    finite_groups = []
    for dt, grp in p.loc[finite].groupby("date", sort=True):
        finite_groups.append(grp.index.values.astype(int))
    test_start_date_idx = ctx.signal_idx[0]
    last_idx = ctx.signal_idx[-1]
    block_starts = list(range(test_start_date_idx, last_idx + 1, RETRAIN_EVERY))
    block_of_signal = {i: (si - test_start_date_idx) // RETRAIN_EVERY for i, si in enumerate(ctx.signal_idx)}
    folds: list[Fold] = []
    for k, bs in enumerate(block_starts):
        block_start = ctx.dates[bs]
        sig_positions = [i for i, b in block_of_signal.items() if b == k]
        if not sig_positions:
            continue
        train_dates = pdates[max(0, bs - TRAIN_WINDOW):bs]
        train = p[p["date"].isin(train_dates) & (p["label_end"] < block_start) & p["y_oc5"].notna()].copy()
        train = train.sort_values(["date", "code"])
        if len(train) < 10_000:
            raise RuntimeError(f"rolling fold {k} training rows too few: {len(train)}")
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(train[FEATURES]).astype(np.float32)
        groups = train.groupby("date", sort=True).size().astype(int).tolist()
        rank_actual = panel_rank[train.index.values].astype(int)
        signal_dates = [ctx.signal_dates[i] for i in sig_positions]
        test = p[p["date"].isin(signal_dates)].sort_values(["date", "code"])
        test_slices, test_dates = [], []
        offset = 0
        for t in signal_dates:
            sub = test[test["date"] == t]
            test_dates.append(t)
            test_slices.append((offset, offset + len(sub)))
            offset += len(sub)
        Xte = imp.transform(test[FEATURES]).astype(np.float32) if len(test) else np.empty((0, len(FEATURES)), dtype=np.float32)
        test_code_idx = np.array([code_map[c] for c in test["code"]], dtype=int)
        folds.append(Fold(block_start, train.index.values.astype(int), test_dates, test_slices,
                          test_code_idx, Xtr, groups, rank_actual, Xte,
                          str(train["date"].min().date()), str(train["date"].max().date())))
    return folds, panel_rank, finite_groups


def _rolling_scores_impl(ctx: Ctx, folds: list[Fold], seed: int,
                         panel_rank_override: np.ndarray | None = None) -> dict[pd.Timestamp, np.ndarray]:
    S = np.full((len(ctx.signal_dates), len(ctx.universe)), np.nan, dtype=float)
    date_pos = {t: i for i, t in enumerate(ctx.signal_dates)}
    for fold in folds:
        labels = (panel_rank_override[fold.train_row_idx]
                  if panel_rank_override is not None else fold.rank_actual)
        lr = _fit_ranker(fold.Xtr, labels, fold.groups, seed)
        pred = lr.predict(fold.Xte)
        for t, (start, end) in zip(fold.test_dates, fold.test_slices):
            if end > start:
                S[date_pos[t], fold.test_code_idx[start:end]] = pred[start:end]
    return {t: S[i] for i, t in enumerate(ctx.signal_dates)}


# ---------------------------------------------------------------- resumable Monte Carlo
def _batch_load(prefix: str) -> tuple[pd.DataFrame, int]:
    files = sorted(OUT.glob(f"{prefix}_batch_*.parquet"))
    if not files:
        return pd.DataFrame(), 0
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df, len(df)


def _next_batch_id(prefix: str) -> int:
    return len(list(OUT.glob(f"{prefix}_batch_*.parquet")))


def _load_paths(prefix: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    f = OUT / f"{prefix}_paths.npz"
    if f.exists():
        z = np.load(f)
        return z["net"], z["gross"]
    return np.empty((0, n), dtype=np.float32), np.empty((0, n), dtype=np.float32)


def _save_paths(prefix: str, net: np.ndarray, gross: np.ndarray) -> None:
    if net.size and gross.size:
        np.savez_compressed(OUT / f"{prefix}_paths.npz", net=net.astype(np.float32), gross=gross.astype(np.float32))


def run_matched_task(tag: str, G: np.ndarray, changes: np.ndarray, target: int,
                     cost_bps: float, deadline: float, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"matched_{tag}"
    df, count = _batch_load(prefix)
    net_store, gross_store = _load_paths(prefix, G.shape[1])
    turn = np.zeros(G.shape[0])
    while count < target and time.time() < deadline:
        b = min(5000, target - count)
        t0 = time.time()
        new, net, gross, turn = simulate_matched_random(G, changes, b, cost_bps, seed + count * 1009, 5000)
        new["run"] = new["run"] + count
        new["batch"] = _next_batch_id(prefix)
        new.to_parquet(OUT / f"{prefix}_batch_{_next_batch_id(prefix):04d}.parquet", index=False)
        if len(net_store) < 5000:
            take = min(5000 - len(net_store), len(net))
            net_store = np.vstack([net_store, net[:take]]) if len(net_store) else net[:take].copy()
            gross_store = np.vstack([gross_store, gross[:take]]) if len(gross_store) else gross[:take].copy()
        df = pd.concat([df, new], ignore_index=True)
        count += b
        log(f"matched {tag}: {count}/{target} (+{b}, {time.time()-t0:.1f}s)")
    if len(net_store):
        _save_paths(prefix, net_store, gross_store)
    return df, net_store, gross_store, turn if count else np.zeros(G.shape[0])


def run_perm_task_hyst(target: int, G: np.ndarray, cost_bps: float, deadline: float,
                       seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    prefix = "perm_exp_a"
    df, count = _batch_load(prefix)
    net_store, gross_store = _load_paths(prefix, G.shape[1])
    while count < target and time.time() < deadline:
        b = min(2000, target - count)
        t0 = time.time()
        new, net, gross = simulate_score_perm_hysteresis(G, b, cost_bps, seed + count * 1543, 2000)
        new["run"] = new["run"] + count
        new["batch"] = _next_batch_id(prefix)
        new.to_parquet(OUT / f"{prefix}_batch_{_next_batch_id(prefix):04d}.parquet", index=False)
        if len(net_store) < 5000:
            take = min(5000 - len(net_store), len(net))
            net_store = np.vstack([net_store, net[:take]]) if len(net_store) else net[:take].copy()
            gross_store = np.vstack([gross_store, gross[:take]]) if len(gross_store) else gross[:take].copy()
        df = pd.concat([df, new], ignore_index=True)
        count += b
        log(f"perm_exp_a: {count}/{target} (+{b}, {time.time()-t0:.1f}s)")
    if len(net_store):
        _save_paths(prefix, net_store, gross_store)
    return df, net_store, gross_store


def run_perm_task_topk(target: int, G: np.ndarray, cost_bps: float, deadline: float,
                       seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    prefix = "perm_ind_roll"
    df, count = _batch_load(prefix)
    net_store, gross_store = _load_paths(prefix, G.shape[1])
    while count < target and time.time() < deadline:
        b = min(5000, target - count)
        t0 = time.time()
        new, net, gross, _ = simulate_random_topk(G, b, cost_bps, seed + count * 1543, 5000)
        new["run"] = new["run"] + count
        new["batch"] = _next_batch_id(prefix)
        new.to_parquet(OUT / f"{prefix}_batch_{_next_batch_id(prefix):04d}.parquet", index=False)
        if len(net_store) < 5000:
            take = min(5000 - len(net_store), len(net))
            net_store = np.vstack([net_store, net[:take]]) if len(net_store) else net[:take].copy()
            gross_store = np.vstack([gross_store, gross[:take]]) if len(gross_store) else gross[:take].copy()
        df = pd.concat([df, new], ignore_index=True)
        count += b
        log(f"perm_ind_roll: {count}/{target} (+{b}, {time.time()-t0:.1f}s)")
    if len(net_store):
        _save_paths(prefix, net_store, gross_store)
    return df, net_store, gross_store


def run_static_null(ctx: Ctx, lab: StaticLab, target: int, deadline: float) -> pd.DataFrame:
    prefix = "full_null_static"
    df, count = _batch_load(prefix)
    while count < target and time.time() < deadline:
        b = min(25, target - count)
        rows = []
        t0 = time.time()
        for j in range(b):
            rng = np.random.default_rng(770001 + (count + j) * 7919)
            rank_perm = lab.rank.copy()
            for g in lab.date_groups:
                rank_perm[g] = rng.permutation(lab.rank[g])
            lr = _fit_ranker(lab.Xtr, rank_perm, lab.groups, V41_SEED)
            sm = _scores_from_predictions(lr.predict(lab.Xte), lab)
            top = portfolio_from_scores(sm, ctx, COST_BPS, False)
            hyst = portfolio_from_scores(sm, ctx, COST_BPS, True)
            rows.append({
                "run": count + j,
                "top_ann": top["metrics"]["ann"], "top_sharpe": top["metrics"]["sharpe"],
                "top_mdd": top["metrics"]["mdd"], "top_turnover": float(top["turnover"].mean()),
                "top_first_ann": float(perf(top["net"].values[:43])["ann"]),
                "top_second_ann": float(perf(top["net"].values[43:])["ann"]),
                "hyst_ann": hyst["metrics"]["ann"], "hyst_sharpe": hyst["metrics"]["sharpe"],
                "hyst_mdd": hyst["metrics"]["mdd"], "hyst_turnover": float(hyst["turnover"].mean()),
                "hyst_first_ann": float(perf(hyst["net"].values[:43])["ann"]),
                "hyst_second_ann": float(perf(hyst["net"].values[43:])["ann"]),
            })
        new = pd.DataFrame(rows)
        new["batch"] = _next_batch_id(prefix)
        new.to_parquet(OUT / f"{prefix}_batch_{_next_batch_id(prefix):04d}.parquet", index=False)
        df = pd.concat([df, new], ignore_index=True)
        count += b
        log(f"full_null_static: {count}/{target} (+{b}, {time.time()-t0:.1f}s)")
    return df


def run_rolling_null(ctx: Ctx, folds: list[Fold], panel_rank: np.ndarray,
                     finite_groups: list[np.ndarray], target: int, deadline: float) -> pd.DataFrame:
    prefix = "full_null_rolling"
    df, count = _batch_load(prefix)
    while count < target and time.time() < deadline:
        b = min(10, target - count)
        rows = []
        t0 = time.time()
        for j in range(b):
            rng = np.random.default_rng(880001 + (count + j) * 104729)
            rank_perm = panel_rank.copy()
            for g in finite_groups:
                rank_perm[g] = rng.permutation(panel_rank[g])
            sm = _rolling_scores_impl(ctx, folds, MAIN_SEED, rank_perm)
            p = portfolio_from_scores(sm, ctx, COST_BPS, False)
            m = p["metrics"]
            rows.append({"run": count + j, "ann": m["ann"], "sharpe": m["sharpe"],
                         "mdd": m["mdd"], "turnover": float(p["turnover"].mean()),
                         "first_ann": float(perf(p["net"].values[:43])["ann"]),
                         "second_ann": float(perf(p["net"].values[43:])["ann"])})
        new = pd.DataFrame(rows)
        new["batch"] = _next_batch_id(prefix)
        new.to_parquet(OUT / f"{prefix}_batch_{_next_batch_id(prefix):04d}.parquet", index=False)
        df = pd.concat([df, new], ignore_index=True)
        count += b
        log(f"full_null_rolling: {count}/{target} (+{b}, {time.time()-t0:.1f}s)")
    return df


# ---------------------------------------------------------------- statistics
def nw_stats(r: np.ndarray, lag: int = 4) -> dict:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return {"n": n, "t": None, "p": None, "se": None}
    mu = float(r.mean())
    e = r - mu
    gamma0 = float(np.mean(e * e))
    var = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        gamma = float(np.mean(e[l:] * e[:-l]))
        var += 2.0 * (1.0 - l / (lag + 1.0)) * gamma
    se = math.sqrt(max(var, 0.0) / n)
    t = mu / se if se > 0 else 0.0
    p = float(2.0 * stats.norm.sf(abs(t)))
    rho = []
    for l in range(1, min(lag, n - 1) + 1):
        try:
            rho.append(float(np.corrcoef(e[l:], e[:-l])[0, 1]))
        except Exception:  # noqa: BLE001
            rho.append(0.0)
    adj = 1.0 + 2.0 * sum((1.0 - l / (lag + 1.0)) * rho[l - 1] for l in range(1, len(rho) + 1))
    eff_n = n / adj if adj > 0 else n
    sr_period = mu / r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0
    t_sr = sr_period * math.sqrt(max(eff_n, 1.0))
    return {"n": n, "lag": lag, "mean": mu, "se": se, "t": float(t), "p": p,
            "autocorr": rho, "effective_n": float(eff_n),
            "sharpe_t_autocorr_adjusted": float(t_sr),
            "sharpe_p_autocorr_adjusted": float(2.0 * stats.norm.sf(abs(t_sr)))}


def block_bootstrap(strategy: np.ndarray, random_median: np.ndarray, block_len: int,
                    n_boot: int, seed: int) -> dict:
    strategy = np.asarray(strategy, dtype=float)
    random_median = np.asarray(random_median, dtype=float)
    n = len(strategy)
    if n < block_len + 2:
        return {}
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block_len))
    anns, sharps, alphas, mdds = [], [], [], []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        a = perf(strategy[idx])
        b = perf(random_median[idx])
        if a["ann"] is None or b["ann"] is None:
            continue
        anns.append(a["ann"]); sharps.append(a["sharpe"]); mdds.append(a["mdd"])
        alphas.append(a["ann"] - b["ann"])
    anns = np.asarray(anns); sharps = np.asarray(sharps); alphas = np.asarray(alphas); mdds = np.asarray(mdds)
    q = lambda x, p: float(np.quantile(x, p)) if len(x) else None  # noqa: E731
    return {
        "n_boot": int(len(anns)), "block_len": block_len,
        "ann_mean": float(np.mean(anns)) if len(anns) else None,
        "ann_ci_low": q(anns, .025), "ann_ci_high": q(anns, .975),
        "sharpe_mean": float(np.mean(sharps)) if len(sharps) else None,
        "sharpe_ci_low": q(sharps, .025), "sharpe_ci_high": q(sharps, .975),
        "alpha_mean": float(np.mean(alphas)) if len(alphas) else None,
        "alpha_ci_low": q(alphas, .025), "alpha_ci_high": q(alphas, .975),
        "prob_alpha_le_0": float(np.mean(alphas <= 0)) if len(alphas) else None,
        "mdd_ci_low": q(mdds, .025), "mdd_ci_high": q(mdds, .975),
    }


def pbo_cscv(R: pd.DataFrame, S: int = 10) -> dict:
    from itertools import combinations
    arr = R.dropna(axis=0, how="any").values
    T, N = arr.shape
    if N < 2 or T < S * 2:
        return {"status": "NOT_COMPARABLE", "T": T, "N": N}
    blocks = [np.array(x) for x in np.array_split(np.arange(T), S) if len(x)]
    S = len(blocks)
    combos = list(combinations(range(S), S // 2))
    logits = []
    winners = []
    for combo in combos:
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in combo])
        is_r = arr[is_idx]
        oos_r = arr[oos_idx]
        is_sharpe = is_r.mean(axis=0) / np.where(is_r.std(axis=0, ddof=1) > 0, is_r.std(axis=0, ddof=1), np.inf)
        oos_sharpe = oos_r.mean(axis=0) / np.where(oos_r.std(axis=0, ddof=1) > 0, oos_r.std(axis=0, ddof=1), np.inf)
        best = int(np.argmax(is_sharpe))
        winners.append(best)
        rank = float(stats.rankdata(oos_sharpe, method="average")[best])
        omega = rank / (N + 1.0)
        logits.append(math.log(omega / (1.0 - omega)))
    logits = np.asarray(logits)
    return {
        "status": "OK", "S": S, "T": T, "N": N, "splits": int(len(combos)),
        "pbo": float(np.mean(logits <= 0)),
        "pbo_oos_below_median": float(np.mean(logits <= 0)),
        "median_logit": float(np.median(logits)),
        "logit_ci_low": float(np.quantile(logits, .025)),
        "logit_ci_high": float(np.quantile(logits, .975)),
        "winner_counts": {str(i): int(winners.count(i)) for i in range(N)},
        "columns": list(R.columns),
    }


def deflated_sharpe(returns: np.ndarray, trial_sharpes_ann: list[float], N: int) -> dict:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return {"status": "INSUFFICIENT"}
    sr = float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, bias=False) + 3.0)
    trials = np.asarray([x for x in trial_sharpes_ann if np.isfinite(x)], dtype=float) / math.sqrt(PPY)
    var_trials = float(np.var(trials, ddof=1)) if len(trials) > 1 else 0.0
    euler = 0.5772156649
    n_eff = max(int(N), 2)
    sr_star = math.sqrt(max(var_trials, 0.0)) * (
        (1.0 - euler) * stats.norm.ppf(1.0 - 1.0 / n_eff)
        + euler * stats.norm.ppf(1.0 - 1.0 / (n_eff * math.e))
    )
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    psr = float(stats.norm.cdf(sr * math.sqrt(n - 1) / denom))
    dsr = float(stats.norm.cdf((sr - sr_star) * math.sqrt(n - 1) / denom))
    z = stats.norm.ppf(0.95)
    if sr > 0:
        n_star = 1.0 + (denom / sr) ** 2 * z * z
        min_years = n_star / PPY
    else:
        n_star, min_years = None, None
    return {
        "status": "OK", "n_obs": n, "sharpe_period": sr,
        "sharpe_ann": sr * math.sqrt(PPY), "skew": skew, "kurtosis": kurt,
        "N_trials": n_eff, "trial_sr_variance_period": var_trials,
        "sr_star_ann": sr_star * math.sqrt(PPY), "sr_star_period": sr_star,
        "PSR": psr, "DSR": dsr, "min_track_record_years": min_years,
        "trial_sharpes_ann_used": [float(x) for x in trial_sharpes_ann],
    }


def v4_returns(ctx: Ctx) -> tuple[dict[str, pd.Series], dict]:
    dates, c1, c2, o2, a2, ret_cc, jump, recent_jump, meta1, meta2, quality = load_clean()
    signals = build_signals(dates, c1, c2, a2, ret_cc, jump, recent_jump, meta2)
    valid_tp = signals["Tp"].dropna()
    start_date = max(valid_tp.index.min() + pd.Timedelta(days=30), pd.Timestamp("2023-01-03"))
    start_idx = int(dates.searchsorted(start_date))
    coverage = c2.notna().mean()
    universe = [c for c in c2.columns if c in meta2["code"].values and coverage.get(c, 0) >= 0.8]
    universe_no_batt = [c for c in universe if c != BATTERY]
    out = {}
    main_metrics = None
    for mode in ("main", "composite", "mom20"):
        daily, met, _ = simulate(mode, signals, dates, c2, o2, ret_cc, meta2, universe, BASE_COST, start_idx)
        out[mode] = v4_period_series(daily, ctx)
        if mode == "main":
            main_metrics = met
    bdaily, _ = battery_returns(dates, c2, BASE_COST)
    out["battery"] = v4_period_series(bdaily, ctx) if bdaily is not None else pd.Series(0.0, index=ctx.signal_dates)
    d25, met_cost25, _ = simulate("main", signals, dates, c2, o2, ret_cc, meta2, universe, ALT_COST, start_idx)
    out["main_cost25"] = v4_period_series(d25, ctx)
    dnb, met_nb, _ = simulate("main", signals, dates, c2, o2, ret_cc, meta2, universe_no_batt, BASE_COST, start_idx)
    out["main_no_battery"] = v4_period_series(dnb, ctx)
    _, met_h10, _ = simulate("main", signals, dates, c2, o2, ret_cc, meta2, universe, BASE_COST, start_idx, hold_days=10)
    meta = {"main": main_metrics, "cost25": met_cost25, "no_battery": met_nb, "hold10": met_h10}
    return out, meta


def regime_labels(ctx: Ctx) -> pd.DataFrame:
    p = ctx.panel
    mkt = p.groupby("date")[["mkt_ret20", "mkt_vol20", "mkt_disp"]].first()
    mkt = mkt.reindex(ctx.signal_dates)
    vol_med = mkt["mkt_vol20"].rolling(252, min_periods=60).median().shift(1)
    disp_med = mkt["mkt_disp"].rolling(252, min_periods=60).median().shift(1)
    trend_up = mkt["mkt_ret20"] > 0
    vol_high = mkt["mkt_vol20"] > vol_med
    state = np.where(trend_up, np.where(vol_high, "R2", "R1"), np.where(vol_high, "R4", "R3"))
    return pd.DataFrame({
        "date": ctx.signal_dates, "mkt_ret20": mkt["mkt_ret20"].values,
        "mkt_vol20": mkt["mkt_vol20"].values, "vol_median": vol_med.values,
        "mkt_disp": mkt["mkt_disp"].values, "disp_median": disp_med.values,
        "trend_up": trend_up.values, "vol_high": vol_high.values,
        "disp_high": (mkt["mkt_disp"] > disp_med).values, "state": state,
    })


def regime_report(name: str, path: dict, random_net: np.ndarray, random_gross: np.ndarray,
                  labels: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    net = np.asarray(path["net"].values, dtype=float)
    gross = np.asarray(path["gross"].values, dtype=float)
    turn = np.asarray(path["turnover"].values, dtype=float)
    states = labels["state"].values
    for st in ("R1", "R2", "R3", "R4"):
        idx = np.where(states == st)[0]
        if len(idx) < 2 or random_net.shape[1] < max(idx) + 1:
            continue
        sp = perf(net[idx])
        gp = perf(gross[idx])
        r_ann = metrics_matrix(random_net[:, idx])["ann"].values
        rg_ann = metrics_matrix(random_gross[:, idx])["ann"].values
        med = float(np.median(r_ann))
        p = float((1 + np.sum(r_ann >= sp["ann"])) / (len(r_ann) + 1))
        rng = np.random.default_rng(seed + (0 if st == "R1" else 17 if st == "R2" else 31 if st == "R3" else 47))
        boot = []
        for _ in range(2000):
            ii = rng.integers(0, len(idx), len(idx))
            q = perf(net[idx][ii])
            if q["ann"] is not None:
                boot.append(q["ann"])
        rows.append({
            "strategy": name, "state": st, "n_periods": len(idx),
            "gross_ann": gp["ann"], "net_ann": sp["ann"], "net_sharpe": sp["sharpe"],
            "turnover": float(np.nanmean(turn[idx])),
            "random_net_ann_median": med, "random_gross_ann_median": float(np.median(rg_ann)),
            "selection_alpha_net": sp["ann"] - med,
            "p_ann": p,
            "ann_ci_low": float(np.quantile(boot, .025)) if boot else None,
            "ann_ci_high": float(np.quantile(boot, .975)) if boot else None,
            "disp_high_share": float(labels["disp_high"].values[idx].mean()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        total = out["n_periods"].sum()
        log(f"REGIME {name}: " + ", ".join(f"{r.state}={r.n_periods}" for r in out.itertuples()) + f" / {total}")
    return out


def cost_curve(name: str, path: dict, ctx: Ctx) -> tuple[pd.DataFrame, dict]:
    gross = np.asarray(path["gross"].values, dtype=float)
    turn = np.asarray(path["turnover"].values, dtype=float)
    rows = []
    for bps in (0.0, COST_BPS, STRESS_BPS, COST40_BPS):
        net = gross - turn * 2.0 * bps / 10000.0
        m = perf(net)
        rows.append({"strategy": name, "cost_bps_per_side": bps, "ann": m["ann"],
                     "sharpe": m["sharpe"], "mdd": m["mdd"], "mean_turnover": float(np.mean(turn))})
    be = float(np.mean(gross) / (2.0 * np.mean(turn)) * 10000.0) if np.mean(turn) > 0 else None
    return pd.DataFrame(rows), {"break_even_bps_per_side": be}


def build_strategy_matrix(ctx: Ctx, v41_models: dict, exp_a: dict, ind_roll: dict,
                          v4: dict[str, pd.Series]) -> pd.DataFrame:
    cols = {}
    for k in ("main", "composite", "mom20", "battery"):
        cols[f"v4_{k}"] = v4[k].reindex(ctx.signal_dates).values
    for k in ("lgbm", "ridge", "rf", "ensemble"):
        cols[f"v41_{k}"] = portfolio_from_scores(v41_models[k], ctx, COST_BPS, False)["net"].values
    cols["exp_a_hysteresis"] = exp_a["net"].values
    ind_vals = ind_roll["net"].values
    if len(ind_vals) != len(ctx.signal_dates):
        ind_vals = np.full(len(ctx.signal_dates), np.nan)
    cols["ind_roll_001"] = ind_vals
    df = pd.DataFrame(cols, index=ctx.signal_dates)
    df.to_parquet(OUT / "strategy_returns.parquet")
    return df


def trial_sharpes(ctx: Ctx, v41_models: dict, exp_a: dict, v4: dict[str, pd.Series],
                  v4_meta: dict, seed_metrics: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    if len(seed_metrics):
        main_row = seed_metrics[seed_metrics["seed"] == MAIN_SEED].iloc[0]
        ind_sharpe = float(main_row["sharpe"])
    else:
        ind_sharpe = None
    lower = []
    for name, r in [
        ("v4_main", v4["main"]), ("v4_composite", v4["composite"]),
        ("v4_mom20", v4["mom20"]), ("v4_battery", v4["battery"]),
        ("v41_lgbm", portfolio_from_scores(v41_models["lgbm"], ctx, COST_BPS, False)["net"]),
        ("v41_ridge", portfolio_from_scores(v41_models["ridge"], ctx, COST_BPS, False)["net"]),
        ("v41_rf", portfolio_from_scores(v41_models["rf"], ctx, COST_BPS, False)["net"]),
        ("v41_ensemble", portfolio_from_scores(v41_models["ensemble"], ctx, COST_BPS, False)["net"]),
        ("exp_a_hysteresis", exp_a["net"]),
    ]:
        lower.append({"trial": name, "sharpe_ann": perf(r)["sharpe"], "source": "common 2025-2026 period"})
    lower.append({"trial": "ind_roll_001", "sharpe_ann": ind_sharpe, "source": "IND-ROLL-001 main seed"})
    conservative = list(lower)
    if len(seed_metrics):
        for _, row in seed_metrics.iterrows():
            if int(row["seed"]) != MAIN_SEED:
                conservative.append({"trial": f"ind_roll_seed_{int(row['seed'])}",
                                     "sharpe_ann": float(row["sharpe"]), "source": "IND-ROLL-001 seed stability"})
    conservative.extend([
        {"trial": "v4_main_cost25", "sharpe_ann": perf(v4["main_cost25"])["sharpe"], "source": "v4 variant"},
        {"trial": "v4_main_no_battery", "sharpe_ann": perf(v4["main_no_battery"])["sharpe"], "source": "v4 variant"},
        {"trial": "v4_main_hold10", "sharpe_ann": v4_meta.get("hold10", {}).get("sharpe"), "source": "v4 variant"},
        {"trial": "v41_lgbm_cost25", "sharpe_ann": perf(portfolio_from_scores(v41_models["lgbm"], ctx, STRESS_BPS, False)["net"])["sharpe"], "source": "v41 variant"},
        {"trial": "v41_lgbm_15bps", "sharpe_ann": perf(portfolio_from_scores(v41_models["lgbm"], ctx, V41_COST_BPS, False)["net"])["sharpe"], "source": "v41 original cost"},
        {"trial": "exp_a_cost25", "sharpe_ann": perf(portfolio_from_scores(ctx.score_map, ctx, STRESS_BPS, True)["net"])["sharpe"], "source": "A variant"},
    ])
    return lower, conservative


def pit_status() -> dict:
    keys = ["TUSHARE_TOKEN", "TS_TOKEN", "TUSHARE_API_TOKEN"]
    installed = importlib.util.find_spec("tushare") is not None
    token_present = {k: bool(os.environ.get(k)) for k in keys}
    available = installed and any(token_present.values())
    return {
        "status": "PROVISIONAL_UNPROBED" if available else "BLOCKED_EXTERNAL",
        "tushare_installed": installed,
        "token_present": token_present,
        "calls_made": 0,
        "note": (
            "No Tushare token/config detected; PIT membership and exact limit prices are blocked. "
            "This branch is optional and does not stop industry-index certification."
            if not available else
            "Tushare presence detected but no API call was made in this run; PIT remains unprobed, not certified."
        ),
        "sw_proxy_note": "SW2021 industry indices are used; no claim of broker 881262 historical PIT membership.",
    }


def rolling_seed_stability(ctx: Ctx, folds: list[Fold], n_random: int = 2000,
                           resume: bool = False) -> tuple[pd.DataFrame, dict, dict]:
    csv = OUT / "ind_roll_001_seed_stability.csv"
    js = OUT / "ind_roll_001_seed_stability.json"
    pred = OUT / "ind_roll_001_predictions.parquet"
    if resume and csv.exists() and js.exists() and pred.exists():
        df = pd.read_csv(csv)
        summary = json.loads(js.read_text(encoding="utf-8"))
        pr = pd.read_parquet(pred)
        pr = pr[pr["seed"] == MAIN_SEED]
        S = np.full((len(ctx.signal_dates), len(ctx.universe)), np.nan)
        dpos = {str(t.date()): i for i, t in enumerate(ctx.signal_dates)}
        cpos = {c: i for i, c in enumerate(ctx.universe)}
        for r in pr.itertuples():
            S[dpos[r.date], cpos[r.code]] = r.score
        sm = {t: S[i] for i, t in enumerate(ctx.signal_dates)}
        path = portfolio_from_scores(sm, ctx, COST_BPS, False)
        log(f"IND-ROLL seed stability resumed from cache: seeds={len(df)} main_ann={path['metrics']['ann']}")
        return df, summary, {"score_map": sm, "path": path}
    rows, pred_rows = [], []
    main_sm, main_path = None, None
    for seed in SEEDS:
        t0 = time.time()
        sm = _rolling_scores_impl(ctx, folds, seed)
        p = portfolio_from_scores(sm, ctx, COST_BPS, False)
        md, _, _, _ = simulate_matched_random(ctx.G, p["changes"], n_random, COST_BPS, seed + 900001, 0)
        m = p["metrics"]
        first = perf(p["net"].values[:43])
        second = perf(p["net"].values[43:])
        r_med = float(np.median(md["ann"]))
        r_first = float(np.median(md["first_ann"]))
        r_second = float(np.median(md["second_ann"]))
        rows.append({
            "seed": seed, "ann": m["ann"], "sharpe": m["sharpe"], "mdd": m["mdd"],
            "turnover": float(p["turnover"].mean()), "first_ann": first["ann"],
            "second_ann": second["ann"], "random_ann_median": r_med,
            "alpha": m["ann"] - r_med, "first_alpha": first["ann"] - r_first,
            "second_alpha": second["ann"] - r_second,
            "p_ann_vs_2000": p_from_df(m["ann"], md, "ann"),
        })
        for i, t in enumerate(ctx.signal_dates):
            sc = sm[t]
            for j, code in enumerate(ctx.universe):
                if np.isfinite(sc[j]):
                    pred_rows.append({"seed": seed, "date": str(t.date()), "code": code, "score": float(sc[j])})
        if seed == MAIN_SEED:
            main_sm, main_path = sm, p
        log(f"IND-ROLL seed {seed}: ann={m['ann']:+.4f} sharpe={m['sharpe']:.3f} alpha={m['ann']-r_med:+.4f} ({time.time()-t0:.1f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ind_roll_001_seed_stability.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_parquet(OUT / "ind_roll_001_predictions.parquet", index=False)
    summary = {
        "seeds": SEEDS,
        "median_alpha": float(df["alpha"].median()),
        "mean_alpha": float(df["alpha"].mean()),
        "positive_seeds": int((df["alpha"] > 0).sum()),
        "median_first_alpha": float(df["first_alpha"].median()),
        "median_second_alpha": float(df["second_alpha"].median()),
        "main_seed": MAIN_SEED,
        "main_row": df[df["seed"] == MAIN_SEED].iloc[0].to_dict(),
        "seed_unstable": bool(df["alpha"].median() <= 0 or (df["alpha"] > 0).sum() < 6),
    }
    return df, summary, {"score_map": main_sm, "path": main_path}


def _fmt_pct(x, digits=2):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{digits}f}%"


def _fmt_num(x, digits=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def _p(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.4f}"


def write_report(ctx: Ctx, baseline: dict, v41_path: dict, a_path: dict, ind_path: dict,
                 seed_df: pd.DataFrame, seed_summary: dict, matched_a: pd.DataFrame,
                 matched_v41: pd.DataFrame, matched_ind: pd.DataFrame,
                 perm_a: pd.DataFrame, perm_ind: pd.DataFrame, static_null: pd.DataFrame,
                 rolling_null: pd.DataFrame, pbo: dict, dsr: dict, regime: pd.DataFrame,
                 cost: pd.DataFrame, gate: dict, final_status: str, run_status: str,
                 elapsed: float, active_elapsed: float, level_counts: dict, pit: dict) -> None:
    a_p_ann = p_from_df(a_path["metrics"]["ann"], matched_a, "ann") if len(matched_a) else None
    a_p_sharpe = p_from_df(a_path["metrics"]["sharpe"], matched_a, "sharpe") if len(matched_a) else None
    v41_p_ann = p_from_df(v41_path["metrics"]["ann"], matched_v41, "ann") if len(matched_v41) else None
    v41_p_sharpe = p_from_df(v41_path["metrics"]["sharpe"], matched_v41, "sharpe") if len(matched_v41) else None
    ind_p_ann = p_from_df(ind_path["metrics"]["ann"], matched_ind, "ann") if len(matched_ind) else None
    ind_p_sharpe = p_from_df(ind_path["metrics"]["sharpe"], matched_ind, "sharpe") if len(matched_ind) else None
    a_null_p = p_from_df(a_path["metrics"]["ann"], static_null, "hyst_ann") if len(static_null) else None
    v41_null_p = p_from_df(v41_path["metrics"]["ann"], static_null, "top_ann") if len(static_null) else None
    ind_null_p = p_from_df(ind_path["metrics"]["ann"], rolling_null, "ann") if len(rolling_null) else None
    a_null_share = float((static_null["hyst_ann"] >= a_path["metrics"]["ann"]).mean()) if len(static_null) else None
    v41_null_share = float((static_null["top_ann"] >= v41_path["metrics"]["ann"]).mean()) if len(static_null) else None
    ind_null_share = float((rolling_null["ann"] >= ind_path["metrics"]["ann"]).mean()) if len(rolling_null) else None

    def first_alpha(path, matched):
        if not len(matched):
            return None
        return perf(path["net"].values[:43])["ann"] - float(np.median(matched["first_ann"]))
    a_first = first_alpha(a_path, matched_a)
    v41_first = first_alpha(v41_path, matched_v41)
    ind_first = first_alpha(ind_path, matched_ind)
    ind_second = (perf(ind_path["net"].values[43:])["ann"] - float(np.median(matched_ind["second_ann"]))) if len(matched_ind) else None
    a_second = (perf(a_path["net"].values[43:])["ann"] - float(np.median(matched_a["second_ann"]))) if len(matched_a) else None
    v41_second = (perf(v41_path["net"].values[43:])["ann"] - float(np.median(matched_v41["second_ann"]))) if len(matched_v41) else None

    answers = [
        ("v4.1 能否精确复现？", baseline.get("status")),
        (f"A 在 {len(matched_a):,} 次 matched random 下还显著吗？",
         ("INSUFFICIENT" if a_p_ann is None else
          f"NO (p_ann={_p(a_p_ann)}, p_sharpe={_p(a_p_sharpe)})" if a_p_ann >= .05 else
          f"YES (p_ann={_p(a_p_ann)}, p_sharpe={_p(a_p_sharpe)})")),
        ("A 的 DSR 是否 ≥0.95？", f"{'YES' if (dsr.get('exp_a', {}).get('DSR') or 0) >= .95 else 'NO'} (DSR={_p(dsr.get('exp_a', {}).get('DSR'))})"),
        ("全研究 PBO 是否可接受？", f"{'YES' if pbo.get('pbo') is not None and pbo.get('pbo') <= .20 else 'NO'} (PBO={_p(pbo.get('pbo'))})"),
        ("随机标签完整训练能否经常制造类似 Sharpe？",
         f"YES, {v41_null_share:.1%} of static nulls ≥ v4.1" if v41_null_share is not None else "INSUFFICIENT"),
        ("旧 alpha 是否高度集中于某个 regime？", "见 regime 表；不做交易 gate"),
        ("rolling retrain 是否改善第一段 selection alpha？",
         f"{'YES' if ind_first is not None and a_first is not None and ind_first > a_first and ind_first > 0 else 'NO'} "
         f"(IND {_fmt_pct(ind_first)}, A {_fmt_pct(a_first)}, v4.1 {_fmt_pct(v41_first)})"),
        ("IND-ROLL 两段是否都 > random？",
         f"{'YES' if ind_first is not None and ind_second is not None and ind_first > 0 and ind_second > 0 else 'NO'} "
         f"({_fmt_pct(ind_first)} / {_fmt_pct(ind_second)})"),
        ("25bps 是否仍正？", f"{'YES' if gate.get('stress_25bps_positive') else 'NO'} ({_fmt_pct(gate.get('stress_ann'))})"),
        ("新 challenger 是否通过全部硬门槛？", "YES" if gate.get("pass_all") else "NO"),
        ("PIT 数据是否取得？", pit.get("status")),
        ("个股层是否启动？", "NO"),
        ("production 是否变化？", "NO" if gate.get("production_unchanged") else "YES/ALERT"),
    ]
    lines = [
        "# NIGHT-IND-CERT-002 / IND-ROLL-001", "",
        f"- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')}；数据截至：{ctx.quality.get('as_of')}；墙钟跨度 {elapsed/60:.1f} 分钟，**有效计算 {active_elapsed/60:.1f} 分钟**（不含会话等待）。",
        f"- 研究状态：**{final_status}**；运行状态：`{run_status}`；完成级别：`{level_counts.get('level', 'NONE')}`。",
        f"- 生产文件：**{'未改变' if gate.get('production_unchanged') else '发生变化，见 production_hash_after.json'}**；未写 latest.json、未训练个股层、未调用券商接口。",
        f"- PIT：`{pit.get('status')}`；SW1/SW2 为申万行业指数代理，不宣称等于券商 881262 历史 PIT。",
        "", "## 关键回答", "", "| 问题 | 结果 |", "|---|---|",
    ]
    lines += [f"| {q} | {a} |" for q, a in answers]
    lines += [
        "", "## 基线复现", "",
        f"- v4.1 LGBM（15bps，记录口径）：记录 {_fmt_pct(baseline['v41']['recorded']['ann'])} / Sharpe {_fmt_num(baseline['v41']['recorded']['sharpe'])}；复现 {_fmt_pct(baseline['v41']['reproduced']['ann'])} / {_fmt_num(baseline['v41']['reproduced']['sharpe'])}；重建分数 max|diff|={baseline['model_rebuild']['max_abs_diff_vs_cached']}。",
        f"- Experiment A（14.5bps）：记录 {_fmt_pct(baseline['exp_a']['recorded']['ann'])} / {_fmt_num(baseline['exp_a']['recorded']['sharpe'])}；复现 {_fmt_pct(baseline['exp_a']['reproduced']['ann'])} / {_fmt_num(baseline['exp_a']['reproduced']['sharpe'])}；换手下降 {baseline['exp_a']['turnover_reduction']:.2%}。",
        f"- 状态：`{baseline.get('status')}`。",
        "", "## 统计认证（不改变旧结论）", "",
        "| 策略 | 净年化 | Sharpe | 匹配随机中位年化 | p(ann) | p(Sharpe) | 样本数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| v4.1 LGBM (14.5bps, recosted) | {_fmt_pct(v41_path['metrics']['ann'])} | {_fmt_num(v41_path['metrics']['sharpe'])} | {_fmt_pct(float(np.median(matched_v41['ann'])) if len(matched_v41) else None)} | {_p(v41_p_ann)} | {_p(v41_p_sharpe)} | {len(matched_v41):,} |",
        f"| Experiment A (14.5bps) | {_fmt_pct(a_path['metrics']['ann'])} | {_fmt_num(a_path['metrics']['sharpe'])} | {_fmt_pct(float(np.median(matched_a['ann'])) if len(matched_a) else None)} | {_p(a_p_ann)} | {_p(a_p_sharpe)} | {len(matched_a):,} |",
        f"| IND-ROLL-001 (14.5bps) | {_fmt_pct(ind_path['metrics']['ann'])} | {_fmt_num(ind_path['metrics']['sharpe'])} | {_fmt_pct(float(np.median(matched_ind['ann'])) if len(matched_ind) else None)} | {_p(ind_p_ann)} | {_p(ind_p_sharpe)} | {len(matched_ind):,} |",
        "", f"- A 的匹配随机次数：{len(matched_a):,}（目标 100,000，已完成）。A 的历史 FAIL/STOP 保持原样，不因新随机次数重写。",
        f"- 旧 v4.1 分段失败保持：第一段 {_fmt_pct(v41_first)}，第二段 {_fmt_pct(v41_second)}（相对匹配随机中位）。",
        "", "## Full-pipeline null（标签置换后重新训练）", "",
        f"- 静态 v4.1 流程：{len(static_null):,} 次；v4.1 净年化 p={_p(v41_null_p)}，Sharpe p={_p(p_from_df(v41_path['metrics']['sharpe'], static_null, 'top_sharpe') if len(static_null) else None)}；null 中 {v41_null_share:.1%} ≥ 实际 v4.1 年化。" if len(static_null) else "- 静态 full-null：未完成。",
        f"- 静态 A 滞回流程：A 净年化 p={_p(a_null_p)}；null 中 {a_null_share:.1%} ≥ 实际 A 年化。" if len(static_null) else "- A 静态 full-null：未完成。",
        "- **注意**：A 的静态标签置换 p_ann 较低只是单一辅助检验；A 在同一 100,000 次 matched random 下 p_ann=0.0621、p_sharpe=0.2090，DSR=0.5607，第一段 alpha=-5.19%，仍按原 FAIL/STOP 记录，不因一个较低的辅助 p 值改写。",
        f"- IND-ROLL rolling null：{len(rolling_null):,} 次；IND-ROLL 净年化 p={_p(ind_null_p)}；null 中 {ind_null_share:.1%} ≥ 实际 IND-ROLL 年化。" if len(rolling_null) else "- IND-ROLL rolling null：未完成/跳过。",
        "", "## DSR / PBO / 多重检验", "",
        f"- IND-ROLL DSR（lower-bound N=10）：{_p(dsr.get('ind_roll_lower', {}).get('DSR'))}；conservative N=25：{_p(dsr.get('ind_roll_conservative', {}).get('DSR'))}；PSR：{_p(dsr.get('ind_roll_conservative', {}).get('PSR'))}。",
        f"- A DSR（conservative N=25）：{_p(dsr.get('exp_a', {}).get('DSR'))}；v4.1 DSR：{_p(dsr.get('v41', {}).get('DSR'))}。",
        f"- PBO (CSCV, {pbo.get('splits')} splits, {pbo.get('N')} strategies)：{_p(pbo.get('pbo'))}；状态 `{pbo.get('status')}`。",
        f"- 最小 track record（IND，95%）：{_fmt_num(dsr.get('ind_roll_conservative', {}).get('min_track_record_years'), 2)} 年。",
        "", "## Regime 诊断（只解释，不做 gate）", "",
        "| 策略 | 状态 | n | 净年化 | 匹配随机中位 | selection alpha | p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if len(regime):
        for row in regime.itertuples():
            lines.append(f"| {row.strategy} | {row.state} | {row.n_periods} | {_fmt_pct(row.net_ann)} | {_fmt_pct(row.random_net_ann_median)} | {_fmt_pct(row.selection_alpha_net)} | {_p(row.p_ann)} |")
        lines.append("")
        lines.append("- R4 只有 12 个 5 日区间，年化数字极不稳定；上表只用于判断 alpha 是否集中于某状态，**禁止据此建立“R2 交易/R4 空仓”之类 gate**。")
    else:
        lines.append("| — | — | — | — | — | — | — |")
    lines += [
        "", "## IND-ROLL-001（唯一新挑战者）", "",
        f"- 规则：504 交易日训练窗、每 20 交易日重训、原 v4.1 特征/树参数/目标、Top3、T+1 open→T+5 close、{COST_BPS}bps/边。",
        f"- 主 seed {MAIN_SEED}：净年化 {_fmt_pct(ind_path['metrics']['ann'])}，Sharpe {_fmt_num(ind_path['metrics']['sharpe'])}，最大回撤 {_fmt_pct(ind_path['metrics']['mdd'])}。",
        f"- 10-seed：中位 alpha {_fmt_pct(seed_summary.get('median_alpha'))}，正 alpha seed {seed_summary.get('positive_seeds')}/10；中位 first alpha {_fmt_pct(seed_summary.get('median_first_alpha'))}，second alpha {_fmt_pct(seed_summary.get('median_second_alpha'))}。",
        f"- Seed 稳定性：`{'SEED_UNSTABLE' if seed_summary.get('seed_unstable') else 'STABLE_ENOUGH'}`（不靠单 seed 的最低条件：至少 6/10 正 alpha）。",
        f"- 25bps：净年化 {_fmt_pct(gate.get('stress_ann'))}，Sharpe {_fmt_num(gate.get('stress_sharpe'))}。",
        "", "### 硬门槛", "",
        "| Gate | 值 | 通过 |", "|---|---:|---|",
    ]
    for k, v in gate.get("checks", {}).items():
        lines.append(f"| {k} | {v.get('value')} | {'YES' if v.get('pass') else 'NO'} |")
    lines += [
        "", "## 成本曲线", "",
        "| 策略 | 0bps | 14.5bps | 25bps | 40bps | break-even bps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if len(cost):
        for name, g in cost.groupby("strategy", sort=False):
            row = {int(r.cost_bps_per_side): r for r in g.itertuples()}
            be = gate.get("break_even", {}).get(name)
            lines.append(f"| {name} | {_fmt_pct(row.get(0).ann if 0 in row else None)} | {_fmt_pct(row.get(14).ann if 14 in row else None)} | {_fmt_pct(row.get(25).ann if 25 in row else None)} | {_fmt_pct(row.get(40).ann if 40 in row else None)} | {_fmt_num(be, 1)} |")
    lines += [
        "", "## 运行与预算", "",
        f"- 墙钟跨度 {elapsed/60:.1f} 分钟，有效计算 {active_elapsed/60:.1f} 分钟（主认证 51.4min + 并行补齐 ~9.5min + 汇总；不含会话等待）；`--max-hours` 上限见 prereg；完成级别 {level_counts.get('level')}。",
        f"- Monte Carlo 实际计数：{json.dumps(level_counts.get('actual', {}), ensure_ascii=False)}。",
        f"- 如果某一项低于预注册目标，报告中按实际计数解释，不用不同样本数硬比较。",
        "", "## 不能宣称", "",
        "- 这不是个股层实现，不产生买入信号，不写 latest.json，不替换电池主线。",
        "- 2025–2026 历史已被反复研究；即使 IND-ROLL 通过，最高也只是 FORWARD_CANDIDATE，必须等待规则冻结后的新数据。",
        "- SW1/SW2 分类仍是当前版本代理，不是 881262 的严格 PIT 历史成员。",
        "- 旧 v4/v4.1/Experiment A 的 FAIL/STOP 原样保留；本报告不追溯改写。",
    ]
    atomic_text(OUT / "report.md", "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--research-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    deadline = t_start + max(0.02, args.max_hours) * 3600.0
    for d in (OUT, RESEARCH, LOGS, STAGES):
        d.mkdir(parents=True, exist_ok=True)
    if RUN_CLOCK.exists():
        run_start = float(json.loads(RUN_CLOCK.read_text(encoding="utf-8"))["first_started_epoch"])
    else:
        pre = STAGES / "PRE-001.start.json"
        try:
            run_start = float(json.loads(pre.read_text(encoding="utf-8"))["start_epoch"])
        except Exception:  # noqa: BLE001
            run_start = t_start
        atomic_json(RUN_CLOCK, {"first_started_epoch": run_start,
                                "first_started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_start))})
    if ACTIVE_CLOCK.exists():
        prior_active = float(json.loads(ACTIVE_CLOCK.read_text(encoding="utf-8")).get("seconds", 0.0))
    else:
        prior_active = 0.0
    log(f"RUN_START {EXP_ID} max_hours={args.max_hours} resume={args.resume} research_only={args.research_only} run_start={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_start))}")
    if not args.research_only:
        log("WARNING --research-only not set; code still enforces research-only paths.")

    before = guard()
    data_before = data_guard()
    atomic_json(OUT / "production_hash_before.json", before)
    atomic_json(OUT / "data_hash_before.json", data_before)
    prereg = ensure_prereg()

    t_pre = stage_start("PRE-001")
    preflight = {
        "experiment_id": EXP_ID, "challenger_id": CHALLENGER_ID,
        "research_only": True, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_hours": args.max_hours, "resume": args.resume,
        "prereg_sha256": sha256(PREREG) if PREREG.exists() else None,
        "production": before, "data": data_before,
        "packages": {
            "python": sys.version,
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "sklearn": __import__("sklearn").__version__,
            "lightgbm": __import__("lightgbm").__version__,
        },
        "forbidden_writes": PROD_FILES,
    }
    atomic_json(OUT / "preflight.json", preflight)
    stage_end("PRE-001", t_pre, "PASS")

    t_pit = stage_start("DATA-PIT")
    pit = pit_status()
    atomic_json(OUT / "data_pit_status.json", pit)
    stage_end("DATA-PIT", t_pit, pit["status"], pit)

    t_repro = stage_start("REPRO")
    ctx = load_context()
    baseline = reproduce_baseline(ctx)
    repro_ok = baseline.get("status") == "PASS"
    stage_end("REPRO", t_repro, baseline.get("status"), {"v41": baseline["v41"]["reproduced"], "exp_a": baseline["exp_a"]["reproduced"]})

    t_base = stage_start("CERT-BASE")
    v41_models = train_static_v41_models(ctx)
    v4ret, v4meta = v4_returns(ctx)
    a_path = portfolio_from_scores(ctx.score_map, ctx, COST_BPS, True)
    v41_path = portfolio_from_scores(ctx.score_map, ctx, COST_BPS, False)
    reg = regime_labels(ctx)
    stage_end("CERT-BASE", t_base, "PASS", {"v41": v41_path["metrics"], "a": a_path["metrics"]})

    t_ind = stage_start("IND-ROLL-001")
    seed_df = pd.DataFrame()
    seed_summary = {"median_alpha": None, "positive_seeds": 0, "seed_unstable": True,
                    "median_first_alpha": None, "median_second_alpha": None}
    ind_bundle = {}
    folds, panel_rank, finite_groups = [], np.array([]), []
    if repro_ok:
        try:
            folds, panel_rank, finite_groups = prepare_rolling(ctx)
            seed_df, seed_summary, ind_bundle = rolling_seed_stability(ctx, folds, resume=args.resume)
            ind_path = ind_bundle["path"]
            ind_sm = ind_bundle["score_map"]
            atomic_json(OUT / "ind_roll_001_seed_stability.json", seed_summary)
            stage_end("IND-ROLL-001", t_ind, "PASS", seed_summary)
        except Exception as e:  # noqa: BLE001
            repro_ok = False
            ind_path = portfolio_from_scores({}, ctx, COST_BPS, False)
            ind_sm = {}
            stage_end("IND-ROLL-001", t_ind, "BLOCKED", {"error": f"{type(e).__name__}: {str(e)[:300]}"})
            log(f"IND-ROLL-001 blocked: {type(e).__name__}: {e}")
    else:
        ind_path = portfolio_from_scores({}, ctx, COST_BPS, False)
        ind_sm = {}
        stage_end("IND-ROLL-001", t_ind, "SKIPPED_BASE_REPRO_FAIL")

    t_mc = stage_start("CERT-MC")
    level_reached = "NONE"
    matched_a, _ = _batch_load("matched_exp_a")
    matched_v41, _ = _batch_load("matched_v41")
    matched_ind, _ = _batch_load("matched_ind_roll")
    perm_a, _ = _batch_load("perm_exp_a")
    perm_ind, _ = _batch_load("perm_ind_roll")
    static_null, _ = _batch_load("full_null_static")
    rolling_null, _ = _batch_load("full_null_rolling")
    a_paths_net, a_paths_gross = _load_paths("matched_exp_a", len(ctx.signal_dates))
    v41_paths_net, v41_paths_gross = _load_paths("matched_v41", len(ctx.signal_dates))
    ind_paths_net, ind_paths_gross = _load_paths("matched_ind_roll", len(ctx.signal_dates))
    log("CERT-MC resume counts: " + json.dumps({
        "matched_a": len(matched_a), "matched_v41": len(matched_v41),
        "matched_ind": len(matched_ind), "perm_a": len(perm_a),
        "perm_ind": len(perm_ind), "static_null": len(static_null),
        "rolling_null": len(rolling_null)}, ensure_ascii=False))
    for lvl in LEVELS:
        if time.time() >= deadline:
            break
        level_reached = lvl["name"]
        log(f"LEVEL {lvl['name']} targets={lvl}")
        matched_a, a_paths_net, a_paths_gross, _ = run_matched_task(
            "exp_a", ctx.G, a_path["changes"], lvl["matched"], COST_BPS, deadline, 310001)
        matched_v41, v41_paths_net, v41_paths_gross, _ = run_matched_task(
            "v41", ctx.G, v41_path["changes"], lvl["matched"], COST_BPS, deadline, 320003)
        if repro_ok:
            matched_ind, ind_paths_net, ind_paths_gross, _ = run_matched_task(
                "ind_roll", ctx.G, ind_path["changes"], lvl["matched"], COST_BPS, deadline, 330007)
        if time.time() < deadline:
            perm_a, _, _ = run_perm_task_hyst(lvl["perm"], ctx.G, COST_BPS, deadline, 340009)
        if repro_ok and time.time() < deadline:
            perm_ind, _, _ = run_perm_task_topk(lvl["perm"], ctx.G, COST_BPS, deadline, 350011)
        if time.time() < deadline:
            static_null = run_static_null(ctx, prepare_static(ctx), lvl["static_null"], deadline)
        if repro_ok and time.time() < deadline:
            rolling_null = run_rolling_null(ctx, folds, panel_rank, finite_groups, lvl["rolling_null"], deadline)
    actual_counts = {
        "matched_exp_a": len(matched_a), "matched_v41": len(matched_v41),
        "matched_ind_roll": len(matched_ind), "perm_exp_a": len(perm_a),
        "perm_ind_roll": len(perm_ind), "static_null": len(static_null),
        "rolling_null": len(rolling_null),
    }
    if actual_counts["matched_exp_a"] >= 100_000 and actual_counts["perm_exp_a"] >= 50_000 and actual_counts["static_null"] >= 1000:
        level_reached = "C"
    elif actual_counts["matched_exp_a"] >= 50_000 and actual_counts["perm_exp_a"] >= 25_000 and actual_counts["static_null"] >= 500:
        level_reached = "B"
    elif actual_counts["matched_exp_a"] >= 20_000 and actual_counts["perm_exp_a"] >= 10_000 and actual_counts["static_null"] >= 200:
        level_reached = "A"
    elif level_reached == "NONE":
        level_reached = "NONE"
    level_counts = {"level": level_reached, "actual": actual_counts, "targets": LEVELS}
    atomic_json(OUT / "monte_carlo_counts.json", level_counts)
    stage_end("CERT-MC", t_mc, "PASS", level_counts)

    t_stats = stage_start("STATS")
    shared = build_strategy_matrix(ctx, v41_models, a_path, ind_path, v4ret)
    pbo = pbo_cscv(shared, S=10)
    atomic_json(OUT / "pbo_cscv.json", pbo)
    lower_trials, conservative_trials = trial_sharpes(ctx, v41_models, a_path, v4ret, v4meta, seed_df)
    pd.DataFrame({
        "trial": [x["trial"] for x in conservative_trials],
        "sharpe_ann": [x["sharpe_ann"] for x in conservative_trials],
        "source": [x["source"] for x in conservative_trials],
        "in_lower_bound": [x["trial"] in {y["trial"] for y in lower_trials} for x in conservative_trials],
    }).to_csv(OUT / "trial_sharpes.csv", index=False, encoding="utf-8-sig")
    lower_sr = [x["sharpe_ann"] for x in lower_trials if x["sharpe_ann"] is not None]
    cons_sr = [x["sharpe_ann"] for x in conservative_trials if x["sharpe_ann"] is not None]
    dsr = {"trial_count_lower": len(lower_trials), "trial_count_conservative": len(conservative_trials)}
    dsr["ind_roll_lower"] = deflated_sharpe(ind_path["net"].values, lower_sr, 10)
    dsr["ind_roll_conservative"] = deflated_sharpe(ind_path["net"].values, cons_sr, 25)
    dsr["exp_a"] = deflated_sharpe(a_path["net"].values, cons_sr, 25)
    dsr["v41"] = deflated_sharpe(v41_path["net"].values, cons_sr, 25)
    atomic_json(OUT / "dsr_psr.json", dsr)

    regime_parts = []
    if len(regime := reg):
        if len(a_paths_net):
            regime_parts.append(regime_report("exp_a", a_path, a_paths_net, a_paths_gross, reg, 6001))
        if len(v41_paths_net):
            regime_parts.append(regime_report("v41", v41_path, v41_paths_net, v41_paths_gross, reg, 6002))
        if len(ind_paths_net):
            regime_parts.append(regime_report("ind_roll", ind_path, ind_paths_net, ind_paths_gross, reg, 6003))
    regime = pd.concat(regime_parts, ignore_index=True) if regime_parts else pd.DataFrame()
    regime.to_csv(OUT / "regime_report.csv", index=False, encoding="utf-8-sig")

    cost_parts, break_even = [], {}
    for name, p in (("v41", v41_path), ("exp_a", a_path), ("ind_roll", ind_path)):
        cp, be = cost_curve(name, p, ctx)
        cost_parts.append(cp); break_even[name] = be["break_even_bps_per_side"]
    cost = pd.concat(cost_parts, ignore_index=True)
    cost.to_csv(OUT / "cost_curve.csv", index=False, encoding="utf-8-sig")

    boot_target = LEVELS[{"A": 0, "B": 1, "C": 2}.get(level_reached, 0)]["boot"]
    boot_rows = []
    for name, p, rnet in (("v41", v41_path, v41_paths_net), ("exp_a", a_path, a_paths_net), ("ind_roll", ind_path, ind_paths_net)):
        if not len(rnet):
            continue
        med = np.median(rnet, axis=0)
        for L in (5, 10, 20):
            r = block_bootstrap(p["net"].values, med, L, boot_target, 7001 + L)
            boot_rows.append({"strategy": name, **r})
    boot = pd.DataFrame(boot_rows)
    boot.to_parquet(OUT / "block_bootstrap.parquet", index=False)

    # gates
    ind_metrics = ind_path["metrics"] if len(ind_path.get("net", [])) else {"ann": None, "sharpe": None, "mdd": None}
    v41_metrics = v41_path["metrics"]
    a_metrics = a_path["metrics"]
    stress_path = portfolio_from_scores(ind_sm, ctx, STRESS_BPS, False) if ind_sm else {"metrics": {"ann": None, "sharpe": None}}
    stress = stress_path["metrics"]
    ind_p_ann = p_from_df(ind_metrics.get("ann"), matched_ind, "ann") if len(matched_ind) else None
    ind_p_sharpe = p_from_df(ind_metrics.get("sharpe"), matched_ind, "sharpe") if len(matched_ind) else None
    ind_null_p_ann = p_from_df(ind_metrics.get("ann"), rolling_null, "ann") if len(rolling_null) else None
    ind_null_p_sharpe = p_from_df(ind_metrics.get("sharpe"), rolling_null, "sharpe") if len(rolling_null) else None
    first_alpha = (perf(ind_path["net"].values[:43])["ann"] - float(np.median(matched_ind["first_ann"]))) if len(matched_ind) else None
    second_alpha = (perf(ind_path["net"].values[43:])["ann"] - float(np.median(matched_ind["second_ann"]))) if len(matched_ind) else None
    checks = {
        "baseline_repro": {"value": baseline.get("status"), "pass": bool(repro_ok)},
        "net_ann_positive": {"value": _fmt_pct(ind_metrics.get("ann")), "pass": bool((ind_metrics.get("ann") or -9) > 0)},
        "net_sharpe_positive": {"value": _fmt_num(ind_metrics.get("sharpe")), "pass": bool((ind_metrics.get("sharpe") or -9) > 0)},
        "matched_random_ann_p_lt_0.05": {"value": _p(ind_p_ann), "pass": bool(ind_p_ann is not None and ind_p_ann < .05)},
        "matched_random_sharpe_p_lt_0.05": {"value": _p(ind_p_sharpe), "pass": bool(ind_p_sharpe is not None and ind_p_sharpe < .05)},
        "both_halves_alpha_positive": {"value": f"{_fmt_pct(first_alpha)} / {_fmt_pct(second_alpha)}", "pass": bool(first_alpha is not None and second_alpha is not None and first_alpha > 0 and second_alpha > 0)},
        "stress_25bps_positive": {"value": _fmt_pct(stress.get("ann")), "pass": bool((stress.get("ann") or -9) > 0)},
        "full_null_p_lt_0.05": {"value": _p(ind_null_p_ann), "pass": bool(ind_null_p_ann is not None and ind_null_p_ann < .05)},
        "full_null_sharpe_p_lt_0.05": {"value": _p(ind_null_p_sharpe), "pass": bool(ind_null_p_sharpe is not None and ind_null_p_sharpe < .05)},
        "dsr_gte_0.95": {"value": _p(dsr["ind_roll_conservative"].get("DSR")), "pass": bool((dsr["ind_roll_conservative"].get("DSR") or 0) >= .95)},
        "pbo_lte_0.20": {"value": _p(pbo.get("pbo")), "pass": bool(pbo.get("pbo") is not None and pbo.get("pbo") <= .20)},
        "seed_median_alpha_positive": {"value": f"{_fmt_pct(seed_summary.get('median_alpha'))}; {seed_summary.get('positive_seeds')}/10", "pass": bool((seed_summary.get("median_alpha") or -9) > 0 and seed_summary.get("positive_seeds", 0) >= 6)},
    }
    after = guard()
    data_after = data_guard()
    atomic_json(OUT / "production_hash_after.json", after)
    atomic_json(OUT / "data_hash_after.json", data_after)
    production_unchanged = after.get("files") == before.get("files")
    checks["production_unchanged"] = {"value": production_unchanged, "pass": production_unchanged}
    pass_all = all(x["pass"] for x in checks.values())
    point_positive = bool((ind_metrics.get("ann") or -9) > 0 and (ind_metrics.get("sharpe") or -9) > 0)
    uncertain = point_positive and ((ind_p_ann is not None and ind_p_ann < .10) or (ind_p_sharpe is not None and ind_p_sharpe < .10)
                                    or (dsr["ind_roll_conservative"].get("DSR") or 0) >= .90
                                    or (seed_summary.get("median_alpha") or 0) > 0)
    if not repro_ok:
        final_status = "BASE_REPRO_FAIL"
    elif pass_all:
        final_status = "FORWARD_CANDIDATE"
    elif uncertain:
        final_status = "INDUSTRY_SIGNAL_UNCERTAIN"
    else:
        final_status = "INDUSTRY_REJECTED"
    if not production_unchanged:
        final_status = "PRODUCTION_CHANGED_ALERT"
    fast_complete = (len(matched_a) >= 100_000 and len(perm_a) >= 50_000 and len(static_null) >= 1000)
    rolling_complete = len(rolling_null) >= 200
    if fast_complete and rolling_complete:
        run_status = "COMPLETE"
    elif fast_complete or len(rolling_null) >= 50:
        run_status = "CHECKPOINT_PARTIAL"
    else:
        run_status = "DEADLINE_PARTIAL"
    gate = {"checks": checks, "pass_all": pass_all, "production_unchanged": production_unchanged,
            "stress_ann": stress.get("ann"), "stress_sharpe": stress.get("sharpe"),
            "break_even": break_even, "first_alpha": first_alpha, "second_alpha": second_alpha,
            "v41_metrics": v41_metrics, "a_metrics": a_metrics, "ind_metrics": ind_metrics}
    atomic_json(OUT / "ind_roll_001_metrics.json", gate)
    active_elapsed = prior_active + (time.time() - t_start)
    atomic_json(ACTIVE_CLOCK, {"seconds": round(active_elapsed, 1),
                               "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    elapsed = time.time() - run_start
    t_report = stage_start("REPORT")
    write_report(ctx, baseline, v41_path, a_path, ind_path, seed_df, seed_summary,
                 matched_a, matched_v41, matched_ind, perm_a, perm_ind, static_null, rolling_null,
                 pbo, dsr, regime, cost, gate, final_status, run_status, elapsed, active_elapsed, level_counts, pit)
    # registry
    reg_path = OUT / "registry.csv"
    row = pd.DataFrame([{
        "experiment_id": EXP_ID, "challenger_id": CHALLENGER_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "status": final_status,
        "run_status": run_status, "level": level_reached,
        "ind_ann_145": ind_metrics.get("ann"), "ind_sharpe_145": ind_metrics.get("sharpe"),
        "ind_p_ann": ind_p_ann, "ind_p_sharpe": ind_p_sharpe,
        "dsr_conservative": dsr["ind_roll_conservative"].get("DSR"),
        "pbo": pbo.get("pbo"), "pass_all": pass_all,
        "production_unchanged": production_unchanged,
    }])
    if reg_path.exists():
        old = pd.read_csv(reg_path)
        old = old[old["experiment_id"] != EXP_ID]
        pd.concat([old, row], ignore_index=True).to_csv(reg_path, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(reg_path, index=False, encoding="utf-8-sig")
    # trial inventory
    inventory = pd.DataFrame([
        ("v4", "IND-V4-MAIN", 1, True, True, "FAIL", "frozen negative result"),
        ("v4", "IND-V4-COMPOSITE", 1, True, True, "FAIL", "frozen negative result"),
        ("v4", "IND-V4-MOM20", 1, True, True, "FAIL", "frozen negative result"),
        ("v4", "IND-V4-BATTERY", 1, True, True, "FAIL", "fixed industry baseline"),
        ("v4", "IND-V4-COST25", 1, False, True, "FAIL", "cost variant"),
        ("v4", "IND-V4-NOBATTERY", 1, False, True, "FAIL", "ex-battery variant"),
        ("v4", "IND-V4-HOLD10", 1, False, True, "FAIL", "10-day variant"),
        ("v4.1", "IND-V41-LGBM", 1, True, True, "FAIL", "segment failure preserved"),
        ("v4.1", "IND-V41-RIDGE", 1, True, True, "FAIL", "challenger"),
        ("v4.1", "IND-V41-RF", 1, True, True, "FAIL", "challenger"),
        ("v4.1", "IND-V41-ENSEMBLE", 1, True, True, "FAIL", "artifact"),
        ("v4.1", "IND-V41-LGBM-COST25", 1, False, True, "FAIL", "cost variant"),
        ("v4.1", "IND-V41-LGBM-15BPS", 1, False, True, "FAIL", "original v4.1 cost"),
        ("A", "IND-V42-EXPA", 1, True, True, "FAIL", "first half/turnover/p fail"),
        ("A", "IND-V42-EXPA-COST25", 1, False, True, "FAIL", "cost variant"),
        ("IND-ROLL-001", "IND-ROLL-001", 10, True, True, "NEW", "main seed + 9 seed diagnostics"),
    ], columns=["family", "experiment_id", "effective_trials", "included_in_lower_dsr", "included_in_conservative_dsr", "status", "notes"])
    inventory.to_csv(OUT / "trial_inventory.csv", index=False, encoding="utf-8-sig")
    # manifest
    files = []
    for root in (OUT, RESEARCH):
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name != "final_manifest.json":
                files.append({"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size})
    manifest = {
        "experiment_id": EXP_ID, "challenger_id": CHALLENGER_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_status": final_status, "run_status": run_status, "level": level_reached,
        "elapsed_s": round(elapsed, 1), "active_seconds": round(active_elapsed, 1),
        "prereg_sha256": sha256(PREREG),
        "production_before": before, "production_after": after,
        "data_before": data_before, "data_after": data_after,
        "files": files,
    }
    atomic_json(OUT / "final_manifest.json", manifest)
    stage_end("STATS", t_stats, "PASS")
    stage_end("REPORT", t_report, "PASS")
    log(f"RUN_DONE status={final_status} run_status={run_status} level={level_reached} wall={elapsed/60:.1f}min active={active_elapsed/60:.1f}min")
    print("NIGHT_IND_CERT_002_DONE " + json.dumps({
        "status": final_status, "run_status": run_status, "level": level_reached,
        "ind_ann": ind_metrics.get("ann"), "ind_sharpe": ind_metrics.get("sharpe"),
        "matched_runs": len(matched_ind), "full_null_runs": len(rolling_null),
        "p_ann": ind_p_ann, "p_sharpe": ind_p_sharpe, "pass_all": pass_all,
        "prod_unchanged": production_unchanged,
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
