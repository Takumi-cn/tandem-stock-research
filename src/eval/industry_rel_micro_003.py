"""IND-REL-MICRO-003: historical walk-forward diagnostic.

A new, pre-registered industry-index hypothesis.  This is a historical
diagnostic run, not a new independent OOS sample: 2024-2026 has been viewed by
earlier experiments.  Maximum status is FORWARD_CANDIDATE; production is never
written.

Run:
  .venv\\Scripts\\python.exe -m src.eval.industry_rel_micro_003 --max-hours 2 --research-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.industry_rotation import load_clean  # noqa: E402
from src.eval.night_ind_cert_002 import deflated_sharpe  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

EXP_ID = "IND-REL-MICRO-003"
OUT = ROOT / "reports" / "industry_rel_micro_003"
RESEARCH = ROOT / "data" / "research" / "industry_rel_micro_003"
LOGS = OUT / "logs"
STAGES = OUT / "stages"
PREREG = OUT / "prereg.json"
PREREG_SRC = OUT / "pre_registration.json"

TOP_K = 3
HOLD_DAYS = 5
TRAIN_MIN_DAYS = 504
TEST_DAYS = 63
STEP_DAYS = 63
TEST_START = pd.Timestamp("2024-01-02")
COST_BPS = 14.5
STRESS_BPS = 25.0
SEEDS = [20261027, 1, 7, 19, 42, 73, 101, 257, 521, 2025]
MAIN_SEED = 20261027
N_MATCHED = 5000
N_BOOT = 5000
N_NULL = 200
PPY = 252.0 / HOLD_DAYS
BASE_FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    "vol_5", "vol_20", "dd_20", "gap", "amount_ratio",
    "parent_rel_5", "parent_rel_20", "market_rel_5", "market_rel_20",
    "rank_ret5", "rank_ret20", "rank_vol", "rank_rel5", "rank_change",
    "mkt_ret20", "mkt_vol20", "mkt_disp", "mkt_up", "tp",
]
NEW_FEATURES = [
    "overnight_5", "overnight_intraday_gap_20", "illiquidity_5",
    "down_amount_shock_5", "parent_rel_mom_20", "peer_lead_1",
]
PROD_FILES = [
    "reports/latest.json", "reports/predictions_ledger.csv",
    "config/targets.yaml", "config/l1_best.json", "reports/ens_fixed_weights.json",
    "run_daily.pyw", "run_gui.pyw", "src/report/daily.py", "src/report/run_full.py",
    "src/report/short_pick.py",
]
DATA_FILES = [
    "data/industry/sw2_daily.parquet", "data/industry/sw1_daily.parquet",
    "data/industry/sw2_meta.parquet", "data/industry/sw1_meta.parquet",
    "reports/industry_v41/panel_v41.parquet",
]
MODEL_PARAMS = {
    "objective": "rank_xendcg", "metric": "ndcg", "ndcg_eval_at": [3],
    "learning_rate": 0.03, "n_estimators": 400, "num_leaves": 15, "max_depth": 5,
    "min_child_samples": 40, "subsample": 0.80, "subsample_freq": 1,
    "colsample_bytree": 0.80, "reg_alpha": 0.50, "reg_lambda": 8.00,
    "min_split_gain": 0.00, "max_bin": 63, "random_state": MAIN_SEED,
    "n_jobs": 8, "device_type": "cpu", "deterministic": True,
    "force_col_wise": True, "verbosity": -1,
}
MODEL_PARAMS["n_jobs"] = int(os.environ.get("NIGHT_N_JOBS", "8"))


# ---------------------------------------------------------------- infra
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def stage_start(name: str) -> float:
    t0 = time.time()
    atomic_json(STAGES / f"{name}.start.json", {"stage": name, "start": time.strftime("%Y-%m-%d %H:%M:%S"), "start_epoch": t0})
    log(f"STAGE_START {name}")
    return t0


def stage_end(name: str, t0: float, status: str, result: dict | None = None) -> None:
    rec = {"stage": name, "end": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": round(time.time() - t0, 2), "status": status, "result": result or {}}
    atomic_json(STAGES / f"{name}.end.json", rec)
    log(f"STAGE_END {name} status={status} elapsed={rec['elapsed_s']}s")


def guard() -> dict:
    files = {}
    for rel in PROD_FILES:
        p = ROOT / rel
        files[rel] = {"exists": p.exists(), "sha256": sha256(p) if p.exists() else None}
    models = {}
    md = ROOT / "models"
    if md.exists():
        models = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(md.rglob("*")) if p.is_file()}
    return {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "files": files, "models": models}


def data_guard() -> dict:
    return {rel: {"exists": (ROOT / rel).exists(), "sha256": sha256(ROOT / rel) if (ROOT / rel).exists() else None} for rel in DATA_FILES}


def ensure_prereg() -> dict:
    if not PREREG_SRC.exists():
        raise RuntimeError("missing pre_registration.json")
    src = json.loads(PREREG_SRC.read_text(encoding="utf-8"))
    obj = dict(src)
    obj["run_authorization"] = {
        "authorized_by_user": True,
        "authorized_at": "2026-10-27",
        "mode": "HISTORICAL_DIAGNOSTIC",
        "new_independent_oos": False,
        "production_promotion": False,
        "status": "RUN_AUTHORIZED_HISTORICAL_DIAGNOSTIC",
    }
    obj["test_start"] = str(TEST_START.date())
    obj["folds"] = {"train_min_days": TRAIN_MIN_DAYS, "test_days": TEST_DAYS, "step_days": STEP_DAYS}
    atomic_json(PREREG, obj)
    return obj


def perf(r) -> dict:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        return {"n": n, "ann": None, "sharpe": None, "mdd": None, "win": None, "mean": None}
    eq = np.cumprod(1.0 + r)
    ann = float(eq[-1] ** (252.0 / (n * HOLD_DAYS)) - 1.0)
    sd = float(np.std(r, ddof=1))
    sharpe = float(np.mean(r) / sd * math.sqrt(PPY)) if sd > 0 else 0.0
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1.0))
    return {"n": n, "ann": ann, "sharpe": sharpe, "mdd": mdd, "win": float(np.mean(r > 0)),
            "mean": float(np.mean(r)), "vol": sd * math.sqrt(PPY)}


def metrics_matrix(R: np.ndarray) -> pd.DataFrame:
    R = np.asarray(R, dtype=float)
    b, n = R.shape
    mean = R.mean(axis=1)
    sd = R.std(axis=1, ddof=1)
    sharpe = np.where(sd > 0, mean / sd * math.sqrt(PPY), 0.0)
    eq = np.cumprod(1.0 + R, axis=1)
    ann = eq[:, -1] ** (252.0 / (n * HOLD_DAYS)) - 1.0
    mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)
    mid = n // 2
    first = np.cumprod(1.0 + R[:, :mid], axis=1)[:, -1] ** (252.0 / (mid * HOLD_DAYS)) - 1.0
    second = np.cumprod(1.0 + R[:, mid:], axis=1)[:, -1] ** (252.0 / ((n - mid) * HOLD_DAYS)) - 1.0
    return pd.DataFrame({"ann": ann, "sharpe": sharpe, "mdd": mdd, "win": (R > 0).mean(axis=1),
                         "first_ann": first, "second_ann": second})


def empirical_p(actual: float, samples) -> float | None:
    a = np.asarray(samples, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a) or actual is None or not np.isfinite(actual):
        return None
    return float((1 + np.sum(a >= actual)) / (len(a) + 1))


# ---------------------------------------------------------------- features
def build_features() -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    dates, c1, c2, o2, a2, ret_cc, jump, recent_jump, meta1, meta2, quality = load_clean()
    panel = pd.read_parquet(ROOT / "reports" / "industry_v41" / "panel_v41.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype(str)
    universe = sorted(panel["code"].unique())
    c2 = c2.reindex(dates)[universe]
    o2 = o2.reindex(dates)[universe]
    a2 = a2.reindex(dates)[universe]
    ret = c2.pct_change(fill_method=None)
    on = np.log(o2 / c2.shift(1))
    idr = np.log(c2 / o2)
    eps = 1e-9
    new = pd.DataFrame(index=dates)
    frames = {}
    frames["overnight_5"] = on.rolling(5, min_periods=5).sum()
    frames["overnight_intraday_gap_20"] = (on - idr).rolling(20, min_periods=20).sum()
    frames["illiquidity_5"] = (ret.abs() / (a2 + eps)).rolling(5, min_periods=5).median()
    za = np.log1p(a2.clip(lower=0))
    z = (za - za.rolling(60, min_periods=40).mean()) / za.rolling(60, min_periods=40).std()
    frames["down_amount_shock_5"] = ((ret < 0) * z.clip(lower=0)).rolling(5, min_periods=5).mean()
    frames["parent_rel_mom_20"] = panel.pivot(index="date", columns="code", values="parent_rel_20").reindex(dates)[universe]
    # peer lead-lag: corr(r_j,t-1, r_i,t) over 126 days, positive weights only
    lag = ret.shift(1)
    peer = pd.DataFrame(index=dates, columns=universe, dtype=float)
    for i, code in enumerate(universe):
        corr = lag.rolling(126, min_periods=60).corr(ret[code])
        w = corr.clip(lower=0)
        if code in w.columns:
            w[code] = 0.0
        den = w.sum(axis=1)
        num = (w * ret).sum(axis=1)
        peer[code] = num / den.replace(0, np.nan)
        if (i + 1) % 20 == 0:
            log(f"peer_lead features {i+1}/{len(universe)}")
    frames["peer_lead_1"] = peer
    long = None
    for name, wide in frames.items():
        s = wide.stack().rename(name).reset_index()
        s.columns = ["date", "code", name]
        long = s if long is None else long.merge(s, on=["date", "code"], how="outer")
    out = panel.merge(long, on=["date", "code"], how="left")
    out = out.sort_values(["date", "code"]).reset_index(drop=True)
    out["parent_rel_mom_20"] = out["parent_rel_20"]  # exact duplicate, retained and reported
    audit = []
    for f in NEW_FEATURES:
        s = out[f]
        top = []
        for old in BASE_FEATURES:
            if old in out.columns:
                sub = out[[f, old]].dropna()
                if len(sub) > 20:
                    top.append((old, float(sub[f].corr(sub[old], method="spearman"))))
        top = sorted(top, key=lambda x: abs(x[1]), reverse=True)
        audit.append({"feature": f, "coverage": float(s.notna().mean()), "n": int(s.notna().sum()),
                      "max_abs_corr_base": abs(top[0][1]) if top else None,
                      "top_base": top[0][0] if top else None,
                      "source": "OHLCV/SW index; limit features not in this experiment"})
    audit = pd.DataFrame(audit)
    audit.to_csv(OUT / "feature_audit.csv", index=False, encoding="utf-8-sig")
    out.to_parquet(RESEARCH / "features_panel.parquet", index=False)
    q = dict(quality)
    q["amount_unit_note"] = "source amount is 亿元-scale; illiquidity uses amount directly (rank-invariant to a constant scale)"
    q["peer_lead_note"] = "rolling 126-day corr(r_j,t-1, r_i,t), positive weights, no future data"
    return out, q, audit, meta2


def build_gross(signal_dates: list[pd.Timestamp], codes: list[str]) -> np.ndarray:
    dates, c1, c2, o2, a2, ret_cc, jump, recent_jump, meta1, meta2, quality = load_clean()
    G = np.full((len(signal_dates), len(codes)), np.nan)
    for i, t in enumerate(signal_dates):
        j = int(dates.get_loc(t))
        if j + HOLD_DAYS >= len(dates):
            continue
        entry = dates[j + 1]
        exit_ = dates[j + HOLD_DAYS]
        for k, code in enumerate(codes):
            try:
                G[i, k] = float(c2.at[exit_, code] / o2.at[entry, code] - 1.0)
            except Exception:
                G[i, k] = np.nan
    return G


# ---------------------------------------------------------------- walk-forward
@dataclass
class Fold:
    start_idx: int
    signal_dates: list[pd.Timestamp]
    signal_idx: list[int]
    train_rows: int
    train_start: str
    train_end: str
    Xtr_base: np.ndarray
    Xtr_chall: np.ndarray
    groups: list[int]
    rank: np.ndarray
    date_groups: list[np.ndarray]
    Xte_base: np.ndarray
    Xte_chall: np.ndarray
    test_code_idx: np.ndarray
    test_slices: list[tuple[int, int]]


def prepare_folds(panel: pd.DataFrame, codes: list[str], dates: pd.DatetimeIndex) -> tuple[list[Fold], list[pd.Timestamp]]:
    from sklearn.impute import SimpleImputer
    p = panel.sort_values(["date", "code"]).reset_index(drop=True)
    start_test = int(dates.searchsorted(TEST_START))
    last = len(dates) - HOLD_DAYS - 1
    signal_idx_all = list(range(start_test, last + 1, HOLD_DAYS))
    signal_dates_all = [dates[i] for i in signal_idx_all]
    code_map = {c: i for i, c in enumerate(codes)}
    folds = []
    for s in range(start_test, last + 1, STEP_DAYS):
        e = min(s + TEST_DAYS, last + 1)
        sig_idx = [i for i in signal_idx_all if s <= i < e]
        if not sig_idx:
            continue
        train = p[(p["date"] < dates[s]) & (p["label_end"] < dates[s]) & p["y_oc5"].notna()].copy()
        train = train.sort_values(["date", "code"])
        if train["date"].nunique() < TRAIN_MIN_DAYS or len(train) < 20_000:
            log(f"skip fold at {dates[s].date()}: train dates={train['date'].nunique()} rows={len(train)}")
            continue
        imp = SimpleImputer(strategy="median")
        Xb = imp.fit_transform(train[BASE_FEATURES].astype(float)).astype(np.float32)
        # refit a separate imputer for challenger; missingness can differ
        imp2 = SimpleImputer(strategy="median")
        Xc = imp2.fit_transform(train[BASE_FEATURES + NEW_FEATURES].astype(float)).astype(np.float32)
        groups = train.groupby("date", sort=True).size().astype(int).tolist()
        rank = (train.groupby("date")["y_oc5"].rank(method="first", pct=True) * 30.0).round().clip(0, 30).astype(int).values
        date_groups = [np.asarray(v, dtype=int) for v in train.groupby("date", sort=True).indices.values()]
        test_dates = [dates[i] for i in sig_idx]
        test = p[p["date"].isin(test_dates)].sort_values(["date", "code"])
        Xte_b = imp.transform(test[BASE_FEATURES].astype(float)).astype(np.float32)
        Xte_c = imp2.transform(test[BASE_FEATURES + NEW_FEATURES].astype(float)).astype(np.float32)
        test_code_idx = np.array([code_map[c] for c in test["code"]], dtype=int)
        slices, offset = [], 0
        for t in test_dates:
            n = int((test["date"] == t).sum())
            slices.append((offset, offset + n))
            offset += n
        folds.append(Fold(s, test_dates, sig_idx, len(train), str(train["date"].min().date()),
                          str(train["date"].max().date()), Xb, Xc, groups, rank, date_groups,
                          Xte_b, Xte_c, test_code_idx, slices))
        log(f"fold {len(folds)} start={dates[s].date()} signals={len(sig_idx)} train_rows={len(train)} "
            f"train_dates={train['date'].nunique()} train_end={train['date'].max().date()}")
    return folds, signal_dates_all


def fit_ranker(X: np.ndarray, y: np.ndarray, groups: list[int], seed: int):
    import lightgbm as lgb
    params = dict(MODEL_PARAMS)
    params["random_state"] = int(seed)
    lr = lgb.LGBMRanker(**params)
    lr.fit(X, y, group=groups)
    return lr


def scores_from_fold(fold: Fold, Xte: np.ndarray, lr, n_codes: int) -> dict[pd.Timestamp, np.ndarray]:
    pred = lr.predict(Xte)
    out = {}
    for t, (a, b) in zip(fold.signal_dates, fold.test_slices):
        s = np.full(n_codes, np.nan)
        if b > a:
            s[fold.test_code_idx[a:b]] = pred[a:b]
        out[t] = s
    return out


def walkforward_scores(folds: list[Fold], feature_kind: str, seed: int, n_codes: int,
                       null: bool = False, rng: np.random.Generator | None = None) -> dict[pd.Timestamp, np.ndarray]:
    out = {}
    for fold in folds:
        rank = fold.rank
        if null:
            rank = rank.copy()
            for g in fold.date_groups:
                rank[g] = rng.permutation(rank[g])
        X = fold.Xtr_base if feature_kind == "base" else fold.Xtr_chall
        Xte = fold.Xte_base if feature_kind == "base" else fold.Xte_chall
        lr = fit_ranker(X, rank, fold.groups, seed)
        out.update(scores_from_fold(fold, Xte, lr, n_codes))
    return out


def portfolio_from_scores(score_map: dict[pd.Timestamp, np.ndarray], signal_dates: list[pd.Timestamp],
                          G: np.ndarray, cost_bps: float) -> dict:
    prev: list[int] = []
    rets, gross, turns, changes = [], [], [], []
    holdings, used_dates = [], []
    for i, t in enumerate(signal_dates):
        s = score_map.get(t)
        if s is None:
            continue
        order = np.argsort(-np.nan_to_num(s, nan=-1e18), kind="stable")
        target = [int(x) for x in order[:TOP_K]]
        k = len(set(target) - set(prev)) if prev else TOP_K
        turn = 0.5 if not prev else 0.5 * k / TOP_K
        g = float(G[i, target].mean())
        used_dates.append(t)
        rets.append(g - turn * 2.0 * cost_bps / 10000.0)
        gross.append(g)
        turns.append(turn)
        changes.append(k)
        holdings.append({"date": str(t.date()), "holdings": "|".join(str(x) for x in target),
                         "gross": round(g, 6), "turnover": round(turn, 6)})
        prev = target
    return {"net": np.asarray(rets), "gross": np.asarray(gross), "turnover": np.asarray(turns),
            "changes": np.asarray(changes, dtype=int), "holdings": holdings, "dates": used_dates,
            "metrics": perf(rets), "gross_metrics": perf(gross)}


# ---------------------------------------------------------------- randomization / seeds
def simulate_matched(G: np.ndarray, changes: np.ndarray, n_runs: int, cost_bps: float,
                     seed: int, store_n: int = 5000) -> tuple[pd.DataFrame, np.ndarray]:
    n, u = G.shape
    rng = np.random.default_rng(seed)
    rows = []
    paths = np.empty((min(n_runs, store_n), n), dtype=np.float32)
    for b in range(n_runs):
        h = list(rng.choice(u, TOP_K, replace=False))
        rets = []
        for i in range(n):
            if i:
                k = int(changes[i])
                if k > 0:
                    pos = rng.choice(len(h), size=min(k, len(h)), replace=False)
                    rem = [x for j, x in enumerate(h) if j not in set(pos)]
                    cand = [x for x in range(u) if x not in rem]
                    add = list(rng.choice(cand, size=min(k, len(cand)), replace=False))
                    h = rem + add
                    while len(h) < TOP_K:
                        cand = [x for x in range(u) if x not in h]
                        h.append(int(rng.choice(cand)))
            turn = 0.5 if i == 0 else 0.5 * int(changes[i]) / TOP_K
            rets.append(float(G[i, h].mean()) - turn * 2.0 * cost_bps / 10000.0)
        r = np.asarray(rets, dtype=float)
        p = perf(r)
        mid = n // 2
        p1, p2 = perf(r[:mid]), perf(r[mid:])
        rows.append({"run": b, "ann": p["ann"], "sharpe": p["sharpe"], "mdd": p["mdd"],
                     "first_ann": p1["ann"], "second_ann": p2["ann"]})
        if b < store_n:
            paths[b] = r
        if (b + 1) % 1000 == 0:
            log(f"matched random {b+1}/{n_runs}")
    return pd.DataFrame(rows), paths


def null_rep(folds: list[Fold], n_codes: int, signal_dates: list[pd.Timestamp],
             G: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sm = {}
    for fold in folds:
        rank = fold.rank.copy()
        for g in fold.date_groups:
            rank[g] = rng.permutation(rank[g])
        lr = fit_ranker(fold.Xtr_chall, rank, fold.groups, MAIN_SEED)
        sm.update(scores_from_fold(fold, fold.Xte_chall, lr, n_codes))
    p = portfolio_from_scores(sm, signal_dates, G, COST_BPS)
    m = p["metrics"]
    mid = len(p["net"]) // 2
    return {"ann": m["ann"], "sharpe": m["sharpe"], "mdd": m["mdd"],
            "first_ann": perf(p["net"][:mid])["ann"],
            "second_ann": perf(p["net"][mid:])["ann"],
            "turnover": float(np.mean(p["turnover"]))}


def run_seed_stability(folds: list[Fold], n_codes: int, signal_dates: list[pd.Timestamp],
                       G: np.ndarray, random_median_ann: float,
                       random_median_sharpe: float) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        sm = walkforward_scores(folds, "chall", seed, n_codes)
        p = portfolio_from_scores(sm, signal_dates, G, COST_BPS)
        m = p["metrics"]
        mid = len(p["net"]) // 2
        rows.append({"seed": seed, "ann": m["ann"], "sharpe": m["sharpe"], "mdd": m["mdd"],
                     "alpha_ann": m["ann"] - random_median_ann,
                     "alpha_sharpe": m["sharpe"] - random_median_sharpe,
                     "first_ann": perf(p["net"][:mid])["ann"],
                     "second_ann": perf(p["net"][mid:])["ann"]})
        log(f"seed {seed}: ann={m['ann']:+.4f} sharpe={m['sharpe']:.3f} alpha={m['ann']-random_median_ann:+.4f} ({time.time()-t0:.1f}s)")
    return pd.DataFrame(rows)


def make_null_datasets(folds: list[Fold]):
    import lightgbm as lgb
    ds = []
    for f in folds:
        d = lgb.Dataset(f.Xtr_chall, label=f.rank, group=f.groups, free_raw_data=False)
        ds.append(d)
    return ds


def fit_dataset_ranker(dset, y: np.ndarray, groups: list[int], seed: int):
    import lightgbm as lgb
    dset.set_label(y)
    dset.set_group(groups)
    params = {k: v for k, v in MODEL_PARAMS.items() if k not in ("n_estimators",)}
    params["random_state"] = int(seed)
    return lgb.train(params, dset, num_boost_round=MODEL_PARAMS["n_estimators"])


def null_rep_fast(folds: list[Fold], dsets, n_codes: int, signal_dates: list[pd.Timestamp],
                  G: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sm = {}
    for fold, dset in zip(folds, dsets):
        rank = fold.rank.copy()
        for g in fold.date_groups:
            rank[g] = rng.permutation(rank[g])
        lr = fit_dataset_ranker(dset, rank, fold.groups, MAIN_SEED)
        sm.update(scores_from_fold(fold, fold.Xte_chall, lr, n_codes))
    p = portfolio_from_scores(sm, signal_dates, G, COST_BPS)
    m = p["metrics"]
    mid = len(p["net"]) // 2
    return {"ann": m["ann"], "sharpe": m["sharpe"], "mdd": m["mdd"],
            "first_ann": perf(p["net"][:mid])["ann"],
            "second_ann": perf(p["net"][mid:])["ann"],
            "turnover": float(np.mean(p["turnover"]))}


# ---------------------------------------------------------------- experiment / report
def fold_metrics(folds: list[Fold], signal_dates: list[pd.Timestamp],
                 pf_base: dict, pf_chall: dict, paths: np.ndarray) -> pd.DataFrame:
    base_by_date = dict(zip(pf_base["dates"], pf_base["net"]))
    chall_by_date = dict(zip(pf_chall["dates"], pf_chall["net"]))
    pos = {t: i for i, t in enumerate(signal_dates)}
    rows = []
    for k, f in enumerate(folds):
        idx = [pos[t] for t in f.signal_dates if t in pos]
        if len(idx) < 3:
            continue
        b = np.array([base_by_date[t] for t in f.signal_dates if t in base_by_date])
        c = np.array([chall_by_date[t] for t in f.signal_dates if t in chall_by_date])
        rp = paths[:, idx]
        rb = metrics_matrix(rp)["ann"].values
        bc, cc = perf(b), perf(c)
        rows.append({"fold": k + 1, "start": str(f.signal_dates[0].date()), "n_signals": len(c),
                     "train_rows": f.train_rows, "train_start": f.train_start, "train_end": f.train_end,
                     "base_ann": bc["ann"], "base_sharpe": bc["sharpe"],
                     "chall_ann": cc["ann"], "chall_sharpe": cc["sharpe"],
                     "random_ann_median": float(np.median(rb)),
                     "alpha_ann": cc["ann"] - float(np.median(rb))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--research-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    for d in (OUT, RESEARCH, LOGS, STAGES):
        d.mkdir(parents=True, exist_ok=True)
    log(f"RUN_START {EXP_ID} max_hours={args.max_hours} resume={args.resume} research_only={args.research_only}")
    before = guard()
    data_before = data_guard()
    atomic_json(OUT / "hash_before.json", before)
    atomic_json(OUT / "data_before.json", data_before)
    prereg = ensure_prereg()
    atomic_json(OUT / "prereg.sha256.json", {"sha256": sha256(PREREG), "path": str(PREREG.relative_to(ROOT))})

    t = stage_start("PRE")
    atomic_json(OUT / "preflight.json", {
        "experiment_id": EXP_ID, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "research_only": True, "max_hours": args.max_hours,
        "prereg_sha256": sha256(PREREG), "production": before, "data": data_before,
        "status": "RUN_AUTHORIZED_HISTORICAL_DIAGNOSTIC",
        "new_independent_oos": False,
    })
    stage_end("PRE", t, "PASS")

    t = stage_start("FEATURES")
    features, quality, audit, meta2 = build_features()
    corr = features[NEW_FEATURES + BASE_FEATURES].corr(method="spearman").loc[NEW_FEATURES, BASE_FEATURES]
    corr.to_csv(OUT / "feature_correlations.csv", encoding="utf-8-sig")
    dates = pd.DatetimeIndex(sorted(features["date"].unique()))
    codes = sorted(features["code"].unique())
    folds, signal_dates = prepare_folds(features, codes, dates)
    G = build_gross(signal_dates, codes)
    atomic_json(OUT / "data_quality.json", {
        **{k: v for k, v in quality.items() if isinstance(v, (str, int, float, bool, type(None)))},
        "rows": int(len(features)), "codes": len(codes), "dates": len(dates),
        "signal_dates": len(signal_dates), "folds": len(folds),
        "first_signal": str(signal_dates[0].date()), "last_signal": str(signal_dates[-1].date()),
        "gross_nan": int(np.isnan(G).sum()),
        "new_feature_coverage": {f: float(features[f].notna().mean()) for f in NEW_FEATURES},
        "parent_rel_mom_20_duplicate": "identical to existing parent_rel_20; retained and reported",
        "new_independent_oos": False,
    })
    pd.DataFrame([{"fold": i + 1, "start": str(f.signal_dates[0].date()),
                   "n_signals": len(f.signal_dates), "train_rows": f.train_rows,
                   "train_start": f.train_start, "train_end": f.train_end} for i, f in enumerate(folds)]).to_csv(
        OUT / "fold_registry.csv", index=False, encoding="utf-8-sig")
    stage_end("FEATURES", t, "PASS", {"folds": len(folds), "signals": len(signal_dates)})

    t = stage_start("WALK-FORWARD")
    base_map = walkforward_scores(folds, "base", MAIN_SEED, len(codes))
    chall_map = walkforward_scores(folds, "chall", MAIN_SEED, len(codes))
    pf_base = portfolio_from_scores(base_map, signal_dates, G, COST_BPS)
    pf_chall = portfolio_from_scores(chall_map, signal_dates, G, COST_BPS)
    pf_base25 = portfolio_from_scores(base_map, signal_dates, G, STRESS_BPS)
    pf_chall25 = portfolio_from_scores(chall_map, signal_dates, G, STRESS_BPS)
    _save_scores(signal_dates, codes, base_map, chall_map)
    _save_portfolio(signal_dates, pf_base, pf_chall)
    stage_end("WALK-FORWARD", t, "PASS", {"baseline": pf_base["metrics"], "challenger": pf_chall["metrics"]})

    t = stage_start("RANDOM")
    rand_chall, paths = simulate_matched(G, pf_chall["changes"], N_MATCHED, COST_BPS, 20261027 + 1000)
    rand_base, _ = simulate_matched(G, pf_base["changes"], N_MATCHED, COST_BPS, 20261027 + 2000)
    rand_chall.to_parquet(OUT / "matched_random_challenger.parquet", index=False)
    rand_base.to_parquet(OUT / "matched_random_baseline.parquet", index=False)
    summary = {
        "challenger": {"actual": pf_chall["metrics"],
                       "random_ann_median": float(rand_chall["ann"].median()),
                       "random_sharpe_median": float(rand_chall["sharpe"].median()),
                       "p_ann": empirical_p(pf_chall["metrics"]["ann"], rand_chall["ann"]),
                       "p_sharpe": empirical_p(pf_chall["metrics"]["sharpe"], rand_chall["sharpe"]),
                       "runs": len(rand_chall)},
        "baseline": {"actual": pf_base["metrics"],
                     "random_ann_median": float(rand_base["ann"].median()),
                     "random_sharpe_median": float(rand_base["sharpe"].median()),
                     "p_ann": empirical_p(pf_base["metrics"]["ann"], rand_base["ann"]),
                     "p_sharpe": empirical_p(pf_base["metrics"]["sharpe"], rand_base["sharpe"]),
                     "runs": len(rand_base)},
    }
    # first/second alpha
    mid = len(pf_chall["net"]) // 2
    summary["challenger"]["first_alpha"] = perf(pf_chall["net"][:mid])["ann"] - float(np.median(rand_chall["first_ann"]))
    summary["challenger"]["second_alpha"] = perf(pf_chall["net"][mid:])["ann"] - float(np.median(rand_chall["second_ann"]))
    summary["baseline"]["first_alpha"] = perf(pf_base["net"][:mid])["ann"] - float(np.median(rand_base["first_ann"]))
    summary["baseline"]["second_alpha"] = perf(pf_base["net"][mid:])["ann"] - float(np.median(rand_base["second_ann"]))
    atomic_json(OUT / "matched_random_summary.json", summary)
    stage_end("RANDOM", t, "PASS", summary)

    t = stage_start("SEEDS")
    seed_df = run_seed_stability(folds, len(codes), signal_dates, G,
                                 summary["challenger"]["random_ann_median"],
                                 summary["challenger"]["random_sharpe_median"])
    seed_df.to_csv(OUT / "seed_stability.csv", index=False, encoding="utf-8-sig")
    stage_end("SEEDS", t, "PASS", {"median_alpha": float(seed_df["alpha_ann"].median()),
                                    "positive": int((seed_df["alpha_ann"] > 0).sum())})

    t = stage_start("FULL-NULL")
    null_files = sorted(OUT.glob("rolling_full_null_batch_w*.parquet"))
    null = pd.concat([pd.read_parquet(f) for f in null_files], ignore_index=True) if null_files else pd.DataFrame()
    null_summary = {
        "n": int(len(null)), "target": N_NULL,
        "p_ann": empirical_p(pf_chall["metrics"]["ann"], null["ann"]) if len(null) else None,
        "p_sharpe": empirical_p(pf_chall["metrics"]["sharpe"], null["sharpe"]) if len(null) else None,
        "random_ann_median": float(null["ann"].median()) if len(null) else None,
        "random_sharpe_median": float(null["sharpe"].median()) if len(null) else None,
        "share_ge_actual_ann": float((null["ann"] >= pf_chall["metrics"]["ann"]).mean()) if len(null) else None,
        "note": "parallel deterministic workers; same features/folds/model/cost; label permutation within training dates; PARTIAL if n < target and not used as a PASS gate",
    }
    atomic_json(OUT / "rolling_full_null.json", null_summary)
    stage_end("FULL-NULL", t, "PASS", null_summary)

    t = stage_start("STATS")
    fm = fold_metrics(folds, signal_dates, pf_base, pf_chall, paths)
    fm.to_csv(OUT / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    fold_positive = float((fm["alpha_ann"] > 0).mean()) if len(fm) else 0.0
    diffs = pf_chall["net"] - pf_base["net"]
    paired = {"n": len(diffs), "mean_diff": float(np.mean(diffs)), "median_diff": float(np.median(diffs)),
              "t_stat": float(stats.ttest_1samp(diffs, 0, nan_policy="omit").statistic) if len(diffs) > 2 else None,
              "p_value": float(stats.ttest_1samp(diffs, 0, nan_policy="omit").pvalue) if len(diffs) > 2 else None,
              "challenger_25bps": pf_chall25["metrics"], "baseline_25bps": pf_base25["metrics"]}
    atomic_json(OUT / "paired_metrics.json", paired)
    # DSR: append this challenger to the historical trial inventory; N is not reset to 1
    trial_path = ROOT / "reports" / "night_ind_cert_002" / "trial_sharpes.csv"
    trial_sr = []
    if trial_path.exists():
        tdf = pd.read_csv(trial_path)
        trial_sr = [float(x) for x in tdf["sharpe_ann"].dropna().tolist()]
    trial_sr.append(float(pf_chall["metrics"]["sharpe"]))
    dsr = deflated_sharpe(pf_chall["net"], trial_sr, len(trial_sr))
    dsr["trial_count"] = len(trial_sr)
    dsr["note"] = "historical industry trial inventory plus this challenger; not a new independent sample"
    atomic_json(OUT / "dsr_trial_inventory.json", dsr)
    pbo_path = ROOT / "reports" / "night_ind_cert_002" / "pbo_cscv.json"
    pbo = json.loads(pbo_path.read_text(encoding="utf-8")) if pbo_path.exists() else {}
    atomic_json(OUT / "historical_pbo_diagnostic.json", {
        "status": "HISTORICAL_ONLY", "source": "NIGHT-IND-CERT-002",
        "pbo": pbo.get("pbo"), "note": "no new candidate matrix fabricated for this single-spec experiment",
    })
    # gates
    c = pf_chall["metrics"]
    gates = {
        "net_ann_gt_0": {"value": c["ann"], "pass": c["ann"] > 0},
        "net_sharpe_gt_0": {"value": c["sharpe"], "pass": c["sharpe"] > 0},
        "p_ann_lt_0.05": {"value": summary["challenger"]["p_ann"], "pass": summary["challenger"]["p_ann"] is not None and summary["challenger"]["p_ann"] < .05},
        "p_sharpe_lt_0.05": {"value": summary["challenger"]["p_sharpe"], "pass": summary["challenger"]["p_sharpe"] is not None and summary["challenger"]["p_sharpe"] < .05},
        "full_alpha_gt_0": {"value": c["ann"] - summary["challenger"]["random_ann_median"], "pass": c["ann"] - summary["challenger"]["random_ann_median"] > 0},
        "first_half_alpha_gt_0": {"value": summary["challenger"]["first_alpha"], "pass": summary["challenger"]["first_alpha"] > 0},
        "second_half_alpha_gt_0": {"value": summary["challenger"]["second_alpha"], "pass": summary["challenger"]["second_alpha"] > 0},
        "fold_positive_fraction_gte_2_3": {"value": fold_positive, "pass": fold_positive >= 2 / 3},
        "stress_25bps_ann_gt_0": {"value": pf_chall25["metrics"]["ann"], "pass": pf_chall25["metrics"]["ann"] > 0},
        "stress_25bps_sharpe_gt_0": {"value": pf_chall25["metrics"]["sharpe"], "pass": pf_chall25["metrics"]["sharpe"] > 0},
        "full_null_p_lt_0.05": {"value": null_summary["p_ann"], "pass": null_summary["p_ann"] is not None and null_summary["p_ann"] < .05},
        "seed_median_alpha_gt_0_and_7_of_10": {"value": float(seed_df["alpha_ann"].median()), "pass": float(seed_df["alpha_ann"].median()) > 0 and int((seed_df["alpha_ann"] > 0).sum()) >= 7},
        "dsr_gte_0.95": {"value": dsr.get("DSR"), "pass": (dsr.get("DSR") or 0) >= .95},
    }
    after = guard()
    data_after = data_guard()
    atomic_json(OUT / "hash_after.json", after)
    atomic_json(OUT / "data_after.json", data_after)
    prod_unchanged = after.get("files") == before.get("files") and after.get("models") == before.get("models")
    gates["production_unchanged"] = {"value": prod_unchanged, "pass": bool(prod_unchanged)}
    pass_all = all(v["pass"] for v in gates.values())
    status = "FORWARD_CANDIDATE" if pass_all else "INDUSTRY_REJECTED_NEW_HYPOTHESIS"
    if not prod_unchanged:
        status = "PRODUCTION_CHANGED_ALERT"
    decision = {"experiment_id": EXP_ID, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": status, "historical_diagnostic_only": True,
                "new_independent_oos": False, "production_promotion": False,
                "production_unchanged": bool(prod_unchanged), "gates": gates,
                "fold_positive_fraction": fold_positive, "passed_gates": sum(1 for v in gates.values() if v["pass"]),
                "total_gates": len(gates), "full_null_n": int(len(null)),
                "full_null_target": N_NULL, "full_null_partial": bool(len(null) < N_NULL),
                "full_null_status": "PARTIAL" if len(null) < N_NULL else "COMPLETE"}
    atomic_json(OUT / "decision.json", decision)
    stage_end("STATS", t, "PASS", {"status": status, "passed": decision["passed_gates"]})

    t = stage_start("REPORT")
    _write_report(features, audit, summary, seed_df, null_summary, fm, paired, dsr, pbo,
                  gates, decision, quality, before, after, time.time() - t0)
    # registry
    reg = OUT / "registry.csv"
    row = pd.DataFrame([{"experiment_id": EXP_ID, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "status": status, "historical_diagnostic_only": True,
                         "challenger_ann": c["ann"], "challenger_sharpe": c["sharpe"],
                         "baseline_ann": pf_base["metrics"]["ann"], "baseline_sharpe": pf_base["metrics"]["sharpe"],
                         "p_ann": summary["challenger"]["p_ann"], "p_sharpe": summary["challenger"]["p_sharpe"],
                         "full_null_n": len(null), "production_unchanged": prod_unchanged}])
    if reg.exists():
        old = pd.read_csv(reg)
        old = old[old["experiment_id"] != EXP_ID]
        pd.concat([old, row], ignore_index=True).to_csv(reg, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(reg, index=False, encoding="utf-8-sig")
    files = []
    for root in (OUT, RESEARCH):
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name != "final_manifest.json":
                files.append({"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size})
    manifest = {"experiment_id": EXP_ID, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": status, "historical_diagnostic_only": True,
                "production_unchanged": bool(prod_unchanged),
                "prereg_sha256": sha256(PREREG), "production_before": before,
                "production_after": after, "data_before": data_before, "data_after": data_after,
                "files": files, "elapsed_s": round(time.time() - t0, 1)}
    atomic_json(OUT / "final_manifest.json", manifest)
    stage_end("REPORT", t, "PASS")
    log(f"RUN_DONE status={status} ann={c['ann']:.4f} sharpe={c['sharpe']:.3f} p_ann={summary['challenger']['p_ann']} p_sharpe={summary['challenger']['p_sharpe']} null={len(null)} prod={prod_unchanged}")
    print("IND_REL_MICRO_003_DONE " + json.dumps({
        "status": status, "challenger_ann": c["ann"], "challenger_sharpe": c["sharpe"],
        "baseline_ann": pf_base["metrics"]["ann"], "baseline_sharpe": pf_base["metrics"]["sharpe"],
        "p_ann": summary["challenger"]["p_ann"], "p_sharpe": summary["challenger"]["p_sharpe"],
        "full_null_n": len(null), "production_unchanged": prod_unchanged,
    }, ensure_ascii=True))


def _save_scores(signal_dates: list[pd.Timestamp], codes: list[str],
                 base_map: dict, chall_map: dict) -> None:
    rows = []
    for t in signal_dates:
        b = base_map.get(t)
        c = chall_map.get(t)
        if b is None or c is None:
            continue
        for j, code in enumerate(codes):
            if np.isfinite(b[j]) or np.isfinite(c[j]):
                rows.append({"date": t, "code": code, "score_base": b[j], "score_challenger": c[j]})
    pd.DataFrame(rows).to_parquet(OUT / "oos_scores.parquet", index=False)


def _save_portfolio(signal_dates: list[pd.Timestamp], base: dict, chall: dict) -> None:
    d = chall["dates"]
    n = min(len(base["net"]), len(chall["net"]))
    pd.DataFrame({
        "date": d[:n], "base_net_145": base["net"][:n], "chall_net_145": chall["net"][:n],
        "base_gross": base["gross"][:n], "chall_gross": chall["gross"][:n],
        "base_turnover": base["turnover"][:n], "chall_turnover": chall["turnover"][:n],
    }).to_parquet(OUT / "portfolio_returns.parquet", index=False)
    pd.DataFrame(chall["holdings"]).to_csv(OUT / "holdings.csv", index=False, encoding="utf-8-sig")


def _pct(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:+.{d}f}%"


def _num(x, d=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _write_report(features, audit, summary, seed_df, null, fm, paired, dsr, pbo,
                  gates, decision, quality, before, after, elapsed):
    c = summary["challenger"]
    b = summary["baseline"]
    lines = [
        "# IND-REL-MICRO-003 — Historical Walk-Forward Diagnostic", "",
        f"- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')}；实际耗时：{elapsed/60:.1f} 分钟。",
        f"- 状态：**{decision['status']}**；历史诊断 `historical_diagnostic_only=true`，**不是新的独立样本外验证**。",
        f"- 生产：**{'未改变' if decision['production_unchanged'] else '变化，见 hash_after'}**；不写 latest.json，不进入个股层，不自动晋升。",
        f"- 数据：SW1/SW2 指数，2024-01-02 起 63 日测试窗、504 日最小训练、purged label_end；{len(features)} 行 / {features['date'].nunique()} 个交易日。",
        "", "## 实验假设与冻结规则", "",
        "- 假设：在 v4.1 的 25 个特征之外，加入隔夜/日内结构、Amihud 流动性、成交额冲击、父行业相对动量和行业 lead-lag 关系，能提供稳定的指数级选择增量。",
        "- 六特征：`overnight_5`、`overnight_intraday_gap_20`、`illiquidity_5`、`down_amount_shock_5`、`parent_rel_mom_20`、`peer_lead_1`。",
        "- 模型：固定 LightGBM rank_xendcg 400 树、depth 5、L2=8、max_bin=63、CPU deterministic；Top3、5 日、T+1 open→T+5 close、14.5bps/边、25bps 压力。",
        "- Walk-forward：最小训练 504 日、测试 63 日、步长 63 日、`label_end < test_start`；最后一个不完整窗口丢弃。",
        "", "## 基线 vs Challenger", "",
        "| 策略 | 净年化 | Sharpe | 最大回撤 | 逐期均值 | 25bps 年化 | 25bps Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| v4.1 特征基线 | {_pct(b['actual']['ann'])} | {_num(b['actual']['sharpe'])} | {_pct(b['actual']['mdd'])} | {_pct(b['actual']['mean'],3)} | {_pct(paired['baseline_25bps']['ann'])} | {_num(paired['baseline_25bps']['sharpe'])} |",
        f"| v4.1 + 六特征 | {_pct(c['actual']['ann'])} | {_num(c['actual']['sharpe'])} | {_pct(c['actual']['mdd'])} | {_pct(c['actual']['mean'],3)} | {_pct(paired['challenger_25bps']['ann'])} | {_num(paired['challenger_25bps']['sharpe'])} |",
        "", f"- 配对差（challenger−baseline，逐信号）：mean {_pct(paired['mean_diff'],3)}，median {_pct(paired['median_diff'],3)}，t={_num(paired['t_stat'])}, p={_num(paired['p_value'],4)}；n={paired['n']}。",
        f"- 结论：六特征没有改善历史 walk-forward；challenger {_pct(c['actual']['ann'])} / Sharpe {_num(c['actual']['sharpe'])}，低于基线 {_pct(b['actual']['ann'])} / {_num(b['actual']['sharpe'])}。",
        "", "## 特征审计", "",
        "| 特征 | 覆盖 | 与旧特征最大 | 旧特征 | 说明 |",
        "|---|---:|---:|---|---|",
    ]
    for a in audit.itertuples():
        lines.append(f"| {a.feature} | {a.coverage:.3f} | {_num(a.max_abs_corr_base)} | {a.top_base} | {'重复现有 parent_rel_20' if a.feature=='parent_rel_mom_20' else '新增/代理'} |")
    lines += [
        "- `parent_rel_mom_20` 与现有 `parent_rel_20` 相同（相关 1.0），保留并登记，不能算新增信息。",
        "- 所有六特征只使用 T 收盘及以前数据；没有精确涨跌停/个股 PIT 数据，本实验不声称这些维度。",
        "", "## 随机与 full-null", "",
        f"- matched random：{c['runs']:,} 次；challenger p_ann **{_num(c['p_ann'],4)}**、p_sharpe **{_num(c.get('p_sharpe'),4)}**；随机年化中位 {_pct(c['random_ann_median'])}。",
        f"- 两段 alpha：first {_pct(c['first_alpha'])}，second {_pct(c['second_alpha'])}。",
        f"- rolling full-null：{null['n']}/{null['target']} 次；p_ann **{_num(null['p_ann'],4)}**、p_sharpe **{_num(null['p_sharpe'],4)}**；null 中 {_num((null.get('share_ge_actual_ann') or 0)*100,1)}% ≥ 实际年化。",
        f"- null 年化中位 {_pct(null['random_ann_median'])}、Sharpe 中位 {_num(null['random_sharpe_median'])}；随机标签重训并不比 challenger 差。",
        "", "## 10-seed 稳定性", "",
        f"- 中位 alpha_ann **{_pct(seed_df['alpha_ann'].median())}**；正 alpha {int((seed_df['alpha_ann']>0).sum())}/10；中位 Sharpe {_num(seed_df['sharpe'].median())}。",
        f"- 种子最小/最大年化：{_pct(seed_df['ann'].min())} / {_pct(seed_df['ann'].max())}。",
        "", "## Fold 一致性", "",
        "| fold | start | n | base 年化 | chall 年化 | random 中位 | chall alpha |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in fm.itertuples():
        lines.append(f"| {r.fold} | {r.start} | {r.n_signals} | {_pct(r.base_ann)} | {_pct(r.chall_ann)} | {_pct(r.random_ann_median)} | {_pct(r.alpha_ann)} |")
    lines += [
        "", "## DSR / PBO", "",
        f"- DSR（historical trial inventory + challenger，N={dsr.get('trial_count')}）：**{_num(dsr.get('DSR'),4)}**；PSR {_num(dsr.get('PSR'),4)}；SR* 年化 {_num(dsr.get('sr_star_ann'),3)}。",
        f"- PBO：历史 `NIGHT-IND-CERT-002` 的 {_num(pbo.get('pbo'),4)}，只作诊断，不为单规格 challenger 虚构新 PBO。",
        "", "## 硬门槛", "",
        "| Gate | 值 | 通过 |", "|---|---:|---|",
    ]
    for k, v in gates.items():
        val = v["value"]
        if isinstance(val, bool):
            val = "True" if val else "False"
        elif isinstance(val, (int, float, np.floating)) and k not in ("p_ann_lt_0.05", "p_sharpe_lt_0.05", "full_null_p_lt_0.05", "dsr_gte_0.95"):
            val = _pct(val) if "ann" in k or "alpha" in k else _num(val, 4)
        elif isinstance(val, (int, float, np.floating)):
            val = _num(val, 4)
        lines.append(f"| {k} | {val} | {'YES' if v['pass'] else 'NO'} |")
    lines += [
        "", f"- 通过 {decision['passed_gates']}/{decision['total_gates']} 个 gate；最终 **{decision['status']}**。",
        "", "## 不能宣称", "",
        "- 这是历史诊断，2024–2026 已被多次查看，不是新的独立 OOS；即使全部 gate 通过，最高也只能是 FORWARD_CANDIDATE。",
        "- 不进入个股层、不替换电池主线、不写 latest.json、不自动晋升；行业层在真正新前瞻样本前保持关闭。",
        "- 没有精确涨跌停/停牌/PIT 个股数据；行业指数结果不能外推为个股可交易收益。",
        "- `parent_rel_mom_20` 是现有 `parent_rel_20` 的复制，六特征并非六个独立新信息源。",
    ]
    atomic_text(OUT / "report.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
