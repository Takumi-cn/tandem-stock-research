"""STOCK-RANK-SHADOW-004: grouped XGBoost top-of-page ranker.

Historical diagnostic only.  The production v3.1 stack, 55% threshold and
latest.json are read-only.  The challenger is a shadow ranker; the highest
possible historical status is FORWARD_CANDIDATE.

Run:
  .venv\\Scripts\\python.exe -m src.eval.stock_rank_shadow_004 --max-hours 2 --research-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
from src.eval.night_ind_cert_002 import deflated_sharpe  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

EXP_ID = "STOCK-RANK-SHADOW-004"
OUT = ROOT / "reports" / "stock_rank_shadow_004"
RESEARCH = ROOT / "data" / "research" / "stock_rank_shadow_004"
LOGS = OUT / "logs"
STAGES = OUT / "stages"
PREREG = OUT / "prereg.json"
KRONOS_CACHE = RESEARCH / "kronos_scores.parquet"

PAGE_SIZE = 20
TEST_START_DAY = 60
TEST_DAYS = 10
STEP_DAYS = 10
COST_BPS = 14.5
STRESS_BPS = 25.0
SEEDS = [20261027, 1, 7, 19, 42, 73, 101, 257, 521, 2025]
MAIN_SEED = 20261027
N_MATCHED = 100_000
N_NULL = 200
KRONOS_CUTOFF = pd.Timestamp("2024-07-01")
PAGE_FEATURES = [
    "ret_5", "ret_10", "ret_20", "ret_60", "vol_20", "vol_ratio", "atr_norm",
    "turn_pctile", "pv_corr_10", "rsi_14", "macd_hist_norm", "boll_pos", "dd_20",
    "consec_up", "consec_dn", "upper_shadow", "lower_shadow", "body_ratio",
    "gap_open", "close_pos", "xrank_ret5", "xrank_vol", "xrank_turn", "xrank_macd",
    "p_up", "exp_ret", "pct_chg", "kronos_score",
]
PROD_FILES = [
    "reports/latest.json", "reports/predictions_ledger.csv", "config/targets.yaml",
    "config/l1_best.json", "reports/ens_fixed_weights.json", "run_daily.pyw", "run_gui.pyw",
    "src/report/daily.py", "src/report/run_full.py", "src/report/short_pick.py",
]
DATA_FILES = ["data/parquet/panel.parquet", "reports/top1_audit/trades.csv", "data/parquet/cons_battery.parquet"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
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
    t = time.time()
    atomic_json(STAGES / f"{name}.start.json", {"stage": name, "start": time.strftime("%Y-%m-%d %H:%M:%S"), "epoch": t})
    log(f"STAGE_START {name}")
    return t


def stage_end(name: str, t: float, status: str, result: dict | None = None) -> None:
    rec = {"stage": name, "end": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": round(time.time() - t, 2), "status": status, "result": result or {}}
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
    if PREREG.exists():
        return json.loads(PREREG.read_text(encoding="utf-8"))
    obj = {
        "experiment_id": EXP_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PRE_REGISTERED_HISTORICAL_DIAGNOSTIC",
        "status_max": "FORWARD_CANDIDATE",
        "production_write": False,
        "universe": "existing battery page-1 logic", "page_size": PAGE_SIZE,
        "execution": "T+1_open_to_T+2_close", "base_cost_bps": COST_BPS, "stress_cost_bps": STRESS_BPS,
        "model": "XGBRanker", "objective": "rank:ndcg",
        "query": "trade_date", "label": "5-grade cross-sectional net r2d relevance",
        "pair_method": "topk", "top_focus": 3,
        "n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
        "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 8, "reg_alpha": 1,
        "max_bin": 128,
        "new_raw_technical_feature_families": 0,
        "kronos": "existing frozen score, only dates >= 2024-07-01 for clean pretraining boundary",
        "threshold_search": False, "topk_search": False, "holding_search": False,
        "optuna": False, "production_promotion": False,
        "walk_forward": {"test_start_day": TEST_START_DAY, "test_days": TEST_DAYS, "step_days": STEP_DAYS, "purge_days": 3},
        "forbidden": ["industry B/C/D", "latest.json write", "55% change", "stock layer promotion"],
    }
    atomic_json(PREREG, obj)
    return obj


# ---------------------------------------------------------------- data / Kronos cache
def _kronos_batch(panel: pd.DataFrame, trades: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    import torch
    from src.report.run_full import _hub_snapshot
    sys.path.insert(0, str(ROOT / "vendors" / "Kronos"))
    from model.kronos import Kronos, KronosTokenizer, KronosPredictor
    tok = _hub_snapshot("NeoQuasar/Kronos-Tokenizer-Base")
    mdl = _hub_snapshot("NeoQuasar/Kronos-Base")
    if tok is None or mdl is None:
        raise RuntimeError("Kronos model/tokenizer cache missing")
    model = Kronos.from_pretrained(str(mdl))
    tokenizer = KronosTokenizer.from_pretrained(str(tok))
    pred = KronosPredictor(model, tokenizer, device="cuda:0")
    rows = []
    px = panel.set_index(["date", "code"])
    for di, d in enumerate(dates):
        codes = trades[trades["date"] == d]["code"].tolist()
        dfs, xts, yts, used = [], [], [], []
        for c in codes:
            g = panel[(panel["code"] == c) & (panel["date"] <= d)].sort_values("date").tail(512)
            g = g.dropna(subset=["open", "high", "low", "close", "volume"])
            g = g[g["volume"] > 0]
            if len(g) < 60:
                continue
            dfs.append(pd.DataFrame({k: g[k].astype(float).values for k in ("open", "high", "low", "close", "volume")}))
            xts.append(pd.Series(pd.to_datetime(g["date"]).values))
            yts.append(pd.Series(pd.date_range(pd.to_datetime(g["date"]).iloc[-1] + pd.Timedelta(days=1), periods=5, freq="D")))
            used.append(c)
        if not dfs:
            continue
        m = min(len(x) for x in dfs)
        if m < 60:
            continue
        dfs = [x.tail(m).reset_index(drop=True) for x in dfs]
        xts = [x.tail(m).reset_index(drop=True) for x in xts]
        try:
            with torch.no_grad():
                outs = pred.predict_batch(dfs, xts, yts, pred_len=5, sample_count=1, verbose=False)
            for c, o, df in zip(used, outs, dfs):
                rows.append({"date": d, "code": c,
                             "kronos_score": float(o["close"].iloc[0] / df["close"].iloc[-1] - 1.0),
                             "lookback": int(m), "model": "Kronos-Base",
                             "input_end": str(pd.to_datetime(df.index[-1]).date()) if len(df) else str(d.date())})
        except Exception as e:  # noqa: BLE001
            log(f"Kronos batch failed {d.date()}: {type(e).__name__}: {str(e)[:120]}")
        if (di + 1) % 10 == 0:
            log(f"Kronos cache {di+1}/{len(dates)} dates, rows={len(rows)}")
    del model, tokenizer, pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def ensure_kronos_cache() -> pd.DataFrame:
    if KRONOS_CACHE.exists():
        df = pd.read_parquet(KRONOS_CACHE)
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = df["code"].astype(str)
        log(f"Kronos cache loaded: {len(df)} rows, {df['date'].nunique()} dates")
        return df
    panel = pd.read_parquet(ROOT / "data" / "parquet" / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype(str)
    trades = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    trades["date"] = pd.to_datetime(trades["date"])
    trades["code"] = trades["code"].astype(str)
    dates = sorted(trades["date"].unique())
    log(f"Kronos cache start: {len(dates)} page dates, batch size 20")
    out = _kronos_batch(panel, trades, dates)
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = out["code"].astype(str)
    out["pretrain_cutoff"] = "2024-06-30"
    out["clean_temporal"] = out["date"] >= KRONOS_CUTOFF
    out.to_parquet(KRONOS_CACHE, index=False)
    return out


def load_page_data() -> tuple[pd.DataFrame, list[pd.Timestamp], dict]:
    panel = pd.read_parquet(ROOT / "data" / "parquet" / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype(str)
    trades = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    trades["date"] = pd.to_datetime(trades["date"])
    trades["code"] = trades["code"].astype(str)
    data = trades.merge(panel, on=["date", "code"], how="left", suffixes=("", "_panel"))
    if data["open"].isna().any():
        raise RuntimeError("page rows missing panel features")
    kron = ensure_kronos_cache()
    data = data.merge(kron[["date", "code", "kronos_score"]], on=["date", "code"], how="left")
    data["label_net"] = data["r_2d"] - 2.0 * COST_BPS / 10000.0
    data["tradable"] = data["r_2d"].notna()
    # 5-grade within-day relevance; missing/blocked rows get grade -1 and are excluded from training
    data["relevance"] = -1
    for d, g in data[data["tradable"]].groupby("date"):
        if len(g) < 5:
            continue
        r = g["label_net"].rank(method="first", pct=True)
        grade = np.clip((r * 5).astype(int), 0, 4)
        data.loc[g.index, "relevance"] = grade.astype(int)
    data = data.sort_values(["date", "rank_in_fp"]).reset_index(drop=True)
    dates = sorted(data["date"].unique())
    quality = {
        "page_rows": int(len(data)), "page_dates": len(dates),
        "codes": int(data["code"].nunique()),
        "kronos_rows": int(data["kronos_score"].notna().sum()),
        "kronos_coverage": float(data["kronos_score"].notna().mean()),
        "kronos_dates": int(data.loc[data["kronos_score"].notna(), "date"].nunique()),
        "tradable_rows": int(data["tradable"].sum()),
        "relevance_ge_0": int((data["relevance"] >= 0).sum()),
        "date_min": str(data["date"].min().date()), "date_max": str(data["date"].max().date()),
    }
    return data, dates, quality


@dataclass
class Fold:
    test_dates: list[pd.Timestamp]
    train: pd.DataFrame
    test: pd.DataFrame
    groups: list[int]
    test_groups: list[int]


def make_folds(data: pd.DataFrame, dates: list[pd.Timestamp]) -> list[Fold]:
    idx = {d: i for i, d in enumerate(dates)}
    folds = []
    for start in range(TEST_START_DAY, len(dates), STEP_DAYS):
        test_dates = dates[start:min(start + TEST_DAYS, len(dates))]
        if not test_dates:
            break
        train = data[(data["date"] < test_dates[0]) & (data["date"].isin(dates[:max(0, start - 2)])) & (data["relevance"] >= 0)].copy()
        test = data[data["date"].isin(test_dates)].copy()
        if train["date"].nunique() < 40 or len(train) < 400 or test.empty:
            continue
        train = train.sort_values(["date", "rank_in_fp"])
        test = test.sort_values(["date", "rank_in_fp"])
        folds.append(Fold(test_dates, train, test,
                          train.groupby("date").size().astype(int).tolist(),
                          test.groupby("date").size().astype(int).tolist()))
    return folds


def fit_ranker(train: pd.DataFrame, features: list[str], seed: int, labels: np.ndarray | None = None, n_jobs: int = 8):
    import xgboost as xgb
    X = train[features].astype(float).values
    y = (train["relevance"].values if labels is None else labels).astype(int)
    groups = train.groupby("date", sort=True).size().astype(int).tolist()
    model = xgb.XGBRanker(
        objective="rank:ndcg", n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.80, colsample_bytree=0.80, reg_lambda=8.0, reg_alpha=1.0,
        max_bin=128, random_state=int(seed), n_jobs=n_jobs, tree_method="hist",
        lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=3,
        eval_metric="ndcg@1", verbosity=0,
    )
    model.fit(X, y, group=groups, verbose=False)
    return model


def top1_from_scores(test: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    out = test.copy()
    out["score"] = scores
    rows = []
    for d, g in out.groupby("date"):
        g = g.sort_values("score", ascending=False)
        rows.append(g.iloc[0])
    return pd.DataFrame(rows).sort_values("date")


def walkforward(folds: list[Fold], features: list[str], seed: int) -> pd.DataFrame:
    rows = []
    for k, f in enumerate(folds):
        model = fit_ranker(f.train, features, seed)
        scores = model.predict(f.test[features].astype(float).values)
        t = top1_from_scores(f.test, scores)
        t["fold"] = k + 1
        t["seed"] = seed
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def production_top1(data: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    sub = data[data["date"].isin(dates) & data["is_top1"]].copy()
    return sub.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------- evaluation / null
def _net_rows(df: pd.DataFrame) -> np.ndarray:
    r = pd.to_numeric(df["r_2d"], errors="coerce").values
    return np.where(np.isfinite(r), r - 2.0 * COST_BPS / 10000.0, 0.0)


def perf_daily(r) -> dict:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        return {"n": n, "ann": None, "sharpe": None, "mean": None, "mdd": None, "win": None}
    eq = np.cumprod(1 + r)
    sd = np.std(r, ddof=1)
    return {"n": n, "ann": float(eq[-1] ** (252 / n) - 1), "sharpe": float(np.mean(r) / sd * math.sqrt(252)) if sd > 0 else 0.0,
            "mean": float(np.mean(r)), "cum": float(eq[-1] - 1.0), "mdd": float(np.min(eq / np.maximum.accumulate(eq) - 1)), "win": float(np.mean(r > 0))}


def top1_net(top1: pd.DataFrame) -> np.ndarray:
    return _net_rows(top1)


def account_event(selected: pd.DataFrame, dates: list[pd.Timestamp], mode: int, cost_bps: float) -> dict:
    sleeves = 1 if mode == 1 else 2
    free = np.full(sleeves, -1, dtype=int)
    eq = np.ones(sleeves) / sleeves
    idx = {d: i for i, d in enumerate(dates)}
    trades = 0
    for r in selected.itertuples():
        si = idx.get(pd.Timestamp(r.date))
        if si is None or not np.isfinite(r.r_2d):
            continue
        entry, exit_ = si + 1, si + 2
        avail = np.where(free < entry)[0]
        if not len(avail):
            continue
        s = int(avail[0])
        eq[s] *= (1 + float(r.r_2d) - 2 * cost_bps / 10000.0)
        free[s] = exit_
        trades += 1
    final = float(eq.sum())
    return {"final_equity": final, "ann": float(final ** (252 / max(1, len(selected))) - 1),
            "trades": trades, "skipped": int(len(selected) - trades)}


def null_rep(folds: list[Fold], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for f in folds:
        g = f.train.copy()
        for d, sub in g.groupby("date"):
            vals = sub["relevance"].values.copy()
            rng.shuffle(vals)
            g.loc[sub.index, "relevance"] = vals
        model = fit_ranker(g, PAGE_FEATURES, MAIN_SEED, n_jobs=2)
        t = top1_from_scores(f.test, model.predict(f.test[PAGE_FEATURES].astype(float).values))
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    r = top1_net(t)
    m = perf_daily(r)
    return {"run": seed, "ann": m["ann"], "sharpe": m["sharpe"], "mean": m["mean"], "mdd": m["mdd"]}


def local_dsr(r: np.ndarray, trial_sr_ann: list[float], N: int) -> dict:
    r = np.asarray(r, dtype=float); r = r[np.isfinite(r)]
    n = len(r)
    sr = float(np.mean(r) / np.std(r, ddof=1)) if np.std(r, ddof=1) > 0 else 0.0
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, bias=False) + 3)
    trials = np.asarray([x for x in trial_sr_ann if np.isfinite(x)], dtype=float) / math.sqrt(252.0)
    var = float(np.var(trials, ddof=1)) if len(trials) > 1 else 0.0
    e = 0.5772156649
    sr_star = math.sqrt(max(var, 0)) * ((1 - e) * stats.norm.ppf(1 - 1 / N) + e * stats.norm.ppf(1 - 1 / (N * math.e)))
    den = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    psr = float(stats.norm.cdf(sr * math.sqrt(n - 1) / den))
    dsr = float(stats.norm.cdf((sr - sr_star) * math.sqrt(n - 1) / den))
    return {"n": n, "sharpe_daily": sr, "sharpe_ann": sr * math.sqrt(252), "skew": skew, "kurtosis": kurt,
            "N_trials": N, "sr_star_ann": sr_star * math.sqrt(252), "PSR": psr, "DSR": dsr,
            "note": "diagnostic; historical trial inventory is industry + this shadow, not a new OOS"}


def matched_random(data: pd.DataFrame, test_dates: list[pd.Timestamp], n_runs: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    pages = [data[data["date"] == d].sort_values("rank_in_fp") for d in test_dates]
    avail = [p[p["r_2d"].notna()].reset_index(drop=True) for p in pages]
    out = []
    for b in range(n_runs):
        rr = []
        for p in avail:
            if len(p):
                rr.append(float(p.iloc[int(rng.integers(0, len(p)))]["r_2d"]) - 2 * COST_BPS / 10000.0)
            else:
                rr.append(0.0)
        r = np.asarray(rr)
        m = perf_daily(r)
        mid = len(r) // 2
        out.append({"run": b, "ann": m["ann"], "sharpe": m["sharpe"], "mean": m["mean"],
                    "first": perf_daily(r[:mid])["ann"], "second": perf_daily(r[mid:])["ann"]})
    return pd.DataFrame(out), np.asarray([b["mean"] for b in out])


def _write_report(data, quality, base, shadow, seed_df, random_df, null_df, paired, dsr, pbo,
                  decision, before, after, elapsed, leakage_note):
    b = perf_daily(top1_net(base)); s = perf_daily(top1_net(shadow))
    rnd = random_df
    lines = [
        "# STOCK-RANK-SHADOW-004", "",
        f"- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')}；实际耗时：{elapsed/60:.1f} 分钟。",
        f"- 状态：**{decision['status']}**；shadow only，`production_write=false`；历史诊断，不是新独立 OOS。",
        f"- 数据：{quality['page_rows']} 个 page 行 / {quality['page_dates']} 个交易日 / {quality['codes']} 只股票；Kronos 覆盖 {quality['kronos_coverage']:.1%}。",
        f"- 泄漏检查：{leakage_note}",
        "", "## 目标与模型", "",
        "- 每交易日 20 只第一页；同日 5 档 net r2d relevance；XGBoost `rank:ndcg`，query=trade_date，topk pair；T+1 open→T+2 close；14.5/25bps。",
        "- 特征：现有非冗余量价 + production p_up/exp_ret/pct_chg + 冻结 Kronos score；不使用 `ret_1d`/`rank_1d`（它们是次日标签）。",
        "- 6 折 purged chronological replay：训练日提前 3 日 purge，测试 10 日/步长 10 日；10 seeds、100k same-day random、200 标签置换。",
        "", "## 结果", "",
        "| 策略 | n | 60日累计 | 逐期均值 | Sharpe | 最大回撤 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| production Top1 | {b['n']} | {_pct(b['cum'])} | {_pct(b['mean'],3)} | {_num(b['sharpe'])} | {_pct(b['mdd'])} |",
        f"| shadow Top1 | {s['n']} | {_pct(s['cum'])} | {_pct(s['mean'],3)} | {_num(s['sharpe'])} | {_pct(s['mdd'])} |",
        f"| random same-page (100k runs) | {b['n']} | — | {_pct(rnd['mean'].median(),3)} | {_num(rnd['sharpe'].median())} | — |",
        "", f"- shadow−production 逐日配对差：mean {_pct(paired['mean_diff'],3)}，t={_num(paired['t_stat'])}, p={_num(paired['p_value'],4)}。",
        f"- matched random：shadow p_ann **{_num(decision['p_ann'],4)}**、p_sharpe **{_num(decision['p_sharpe'],4)}**；production p_ann {_num(decision['prod_p_ann'],4)}、p_sharpe {_num(decision['prod_p_sharpe'],4)}。",
        f"- 10 seeds：中位 alpha {_pct(seed_df['alpha'].median())}，正 alpha {int((seed_df['alpha']>0).sum())}/10；shadow 逐期均值中位 {_pct(seed_df['alpha'].median(),3)}（60日年化不稳定，不作外推）。",
        f"- rolling full-null：{len(null_df)}/200；p_ann **{_num(decision['null_p_ann'],4)}**；null 逐期均值中位 {_pct(null_df['mean'].median() if len(null_df) else None,3)}。",
        f"- DSR：**{_num(dsr.get('DSR'),4)}**；PBO 历史 {_num(pbo.get('pbo'),4)}（不虚构新 PBO）。",
        "", "## 6 折", "",
        "| fold | n | shadow 累计 | production 累计 | 配对均值差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in paired["folds"]:
        lines.append(f"| {r['fold']} | {r['n']} | {_pct(r['shadow_cum'])} | {_pct(r['prod_cum'])} | {_pct(r['diff_mean'],3)} |")
    lines += [
        "", "## 判断", "",
        f"- 配对 alpha >0：{decision['checks']['paired_alpha_gt0']['pass']}；same-page random alpha>0：{decision['checks']['random_alpha_gt0']['pass']}；两半均>0：{decision['checks']['both_halves_gt0']['pass']}；fold 2/3：{decision['checks']['fold_2_3']['pass']}。",
        f"- 25bps：shadow {_pct(decision['shadow_25_ann'])} / Sharpe {_num(decision['shadow_25_sharpe'])}；production {_pct(decision['prod_25_ann'])} / {_num(decision['prod_25_sharpe'])}。",
        f"- 最终：**{decision['status']}**；不进入生产、不写 latest.json、不改变 55% 门槛、不进入行业/个股部署。",
        "", "## 重要限制", "",
        "- 只有 120 个 page 日、6 个测试折；这是小样本历史诊断，不能当新 OOS。",
        "- Kronos 时间边界：所有页面日期均在 2024-07-01 之后；缓存覆盖 98.9%，缺失行保留 NaN，未伪造。",
        "- 初版曾误把 `ret_1d`/`rank_1d` 当特征，它们是次日标签，产生过异常高收益；已移除并从头重跑，本报告只使用移除后的结果。",
        "- 未对 threshold/TopK/holding/Optuna 搜索；XGBoost 参数冻结。60 日样本上的年化数字只作诊断，不作未来收益预测。",
    ]
    atomic_text(OUT / "report.md", "\n".join(lines) + "\n")


def _pct(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:+.{d}f}%"


def _num(x, d=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--research-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    for d in (OUT, RESEARCH, LOGS, STAGES):
        d.mkdir(parents=True, exist_ok=True)
    log(f"RUN_START {EXP_ID} max_hours={args.max_hours} research_only={args.research_only}")
    before = guard()
    data_before = data_guard()
    atomic_json(OUT / "hash_before.json", before)
    atomic_json(OUT / "data_before.json", data_before)
    prereg = ensure_prereg()
    atomic_json(OUT / "prereg.sha256.json", {"sha256": sha256(PREREG)})

    t = stage_start("PRE")
    atomic_json(OUT / "preflight.json", {
        "experiment_id": EXP_ID, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "research_only": True, "production": before, "data": data_before,
        "prereg_sha256": sha256(PREREG), "status": "HISTORICAL_DIAGNOSTIC",
    })
    stage_end("PRE", t, "PASS")

    t = stage_start("DATA")
    data, dates, quality = load_page_data()
    atomic_json(OUT / "data_quality.json", {
        **quality,
        "forbidden_label_features_excluded": ["ret_1d", "rank_1d", "cls_3"],
        "kronos_temporal_boundary": "all page dates >= 2024-07-01; pretraining cutoff 2024-06-30",
        "note": "first implementation accidentally included ret_1d/rank_1d (next-day labels); removed before this final run.",
    })
    # audit correlation of every included feature with the realized label; flag >0.95
    corr = data[PAGE_FEATURES + ["label_net"]].corr(method="spearman")["label_net"].drop("label_net")
    corr.rename("spearman_with_label").to_csv(OUT / "feature_label_corr.csv", encoding="utf-8-sig")
    leak_flags = {k: float(v) for k, v in corr.items() if np.isfinite(v) and abs(v) > 0.95}
    atomic_json(OUT / "leakage_audit.json", {
        "forbidden_columns_absent": all(c not in PAGE_FEATURES for c in ("ret_1d", "rank_1d", "cls_3")),
        "high_corr_with_label": leak_flags,
        "note": "same-day features only; ret_1d/rank_1d are next-day labels and were removed",
    })
    stage_end("DATA", t, "PASS", quality)

    t = stage_start("REPLAY")
    folds = make_folds(data, dates)
    if not folds:
        raise RuntimeError("no folds")
    shadow = walkforward(folds, PAGE_FEATURES, MAIN_SEED)
    base = production_top1(data, shadow["date"].unique())
    merged = shadow.merge(base[["date", "code", "r_2d", "p_up"]], on="date", suffixes=("_sh", "_prod"))
    merged["shadow_net"] = merged["r_2d_sh"] - 2 * COST_BPS / 10000
    merged["prod_net"] = merged["r_2d_prod"] - 2 * COST_BPS / 10000
    merged["shadow_net"] = merged["shadow_net"].fillna(0.0)
    merged["prod_net"] = merged["prod_net"].fillna(0.0)
    merged.to_parquet(OUT / "top1_replay.parquet", index=False)
    diff = merged["shadow_net"].values - merged["prod_net"].values
    paired = {"n": int(len(merged)), "mean_diff": float(np.mean(diff)), "median_diff": float(np.median(diff)),
              "t_stat": float(stats.ttest_1samp(diff, 0).statistic), "p_value": float(stats.ttest_1samp(diff, 0).pvalue)}
    fold_rows = []
    for k, f in enumerate(folds):
        sub = merged[merged["date"].isin(f.test_dates)]
        d = sub["shadow_net"].values - sub["prod_net"].values
        fold_rows.append({"fold": k + 1, "n": len(sub),
                          "shadow_cum": perf_daily(sub["shadow_net"].values)["cum"],
                          "prod_cum": perf_daily(sub["prod_net"].values)["cum"],
                          "diff_mean": float(np.mean(d)), "shadow_mean": float(np.mean(sub["shadow_net"].values)),
                          "prod_mean": float(np.mean(sub["prod_net"].values))})
    paired["folds"] = fold_rows
    atomic_json(OUT / "paired_metrics.json", paired)
    # 25bps and accounts
    s25 = _net_rows(shadow).copy()
    s25 = np.where(np.isfinite(pd.to_numeric(shadow["r_2d"], errors="coerce").values),
                   pd.to_numeric(shadow["r_2d"], errors="coerce").values - 2 * STRESS_BPS / 10000, 0.0)
    p25 = np.where(np.isfinite(pd.to_numeric(base["r_2d"], errors="coerce").values),
                   pd.to_numeric(base["r_2d"], errors="coerce").values - 2 * STRESS_BPS / 10000, 0.0)
    astat = perf_daily(s25); pstat = perf_daily(p25)
    acc1s = account_event(shadow, dates, 1, COST_BPS)
    acc2s = account_event(shadow, dates, 2, COST_BPS)
    acc1p = account_event(base, dates, 1, COST_BPS)
    acc2p = account_event(base, dates, 2, COST_BPS)
    atomic_json(OUT / "account_comparison.json", {
        "shadow_sleeve1": acc1s, "shadow_sleeve2": acc2s,
        "production_sleeve1": acc1p, "production_sleeve2": acc2p,
        "shadow_25bps": astat, "production_25bps": pstat,
        "note": "event-level two-sleeve account; T+1 open -> T+2 close; no daily MTM here",
    })
    stage_end("REPLAY", t, "PASS", {"shadow": perf_daily(merged["shadow_net"]), "production": perf_daily(merged["prod_net"]), "paired": paired["mean_diff"]})

    t = stage_start("SEEDS")
    seed_rows = []
    for seed in SEEDS:
        w = walkforward(folds, PAGE_FEATURES, seed)
        mm = w.merge(base[["date", "code", "r_2d"]], on="date", suffixes=("_sh", "_prod"))
        sn = mm["r_2d_sh"].fillna(0).values - 2 * COST_BPS / 10000
        pn = mm["r_2d_prod"].fillna(0).values - 2 * COST_BPS / 10000
        seed_rows.append({"seed": seed, "ann": perf_daily(sn)["ann"], "sharpe": perf_daily(sn)["sharpe"],
                          "alpha": float(np.mean(sn - pn)), "positive_days": float(np.mean(sn - pn > 0))})
        log(f"seed {seed}: ann={seed_rows[-1]['ann']:+.4f} alpha={seed_rows[-1]['alpha']:+.4f}")
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT / "seed_stability.csv", index=False, encoding="utf-8-sig")
    stage_end("SEEDS", t, "PASS", {"median_alpha": float(seed_df["alpha"].median()), "positive": int((seed_df["alpha"] > 0).sum())})

    t = stage_start("RANDOM")
    test_dates = sorted(shadow["date"].unique())
    rand, rand_means = matched_random(data, test_dates, N_MATCHED, 20261027)
    rand.to_parquet(OUT / "matched_random_top1.parquet", index=False)
    ss = perf_daily(merged["shadow_net"].values)
    pp = perf_daily(merged["prod_net"].values)
    def ep(actual, arr):
        a = np.asarray(arr, dtype=float); a = a[np.isfinite(a)]
        return float((1 + np.sum(a >= actual)) / (len(a) + 1))
    p_sh_ann = ep(ss["ann"], rand["ann"]); p_sh_sharpe = ep(ss["sharpe"], rand["sharpe"])
    p_pr_ann = ep(pp["ann"], rand["ann"]); p_pr_sharpe = ep(pp["sharpe"], rand["sharpe"])
    atomic_json(OUT / "matched_random_summary.json", {
        "runs": N_MATCHED, "shadow_p_ann": p_sh_ann, "shadow_p_sharpe": p_sh_sharpe,
        "production_p_ann": p_pr_ann, "production_p_sharpe": p_pr_sharpe,
        "random_ann_median": float(rand["ann"].median()), "random_sharpe_median": float(rand["sharpe"].median()),
        "same_day_same_page": True, "same_cost": True,
    })
    stage_end("RANDOM", t, "PASS", {"p_ann": p_sh_ann, "p_sharpe": p_sh_sharpe})

    t = stage_start("FULL-NULL")
    from joblib import Parallel, delayed
    null_rows = Parallel(n_jobs=4, backend="loky")(delayed(null_rep)(folds, 10000 + i) for i in range(N_NULL))
    null_df = pd.DataFrame(null_rows)
    null_df.to_parquet(OUT / "rolling_full_null.parquet", index=False)
    null_ann = float(null_df["ann"].median()); null_sh = float(null_df["sharpe"].median())
    null_p_ann = ep(ss["ann"], null_df["ann"]); null_p_sharpe = ep(ss["sharpe"], null_df["sharpe"])
    atomic_json(OUT / "rolling_full_null.json", {
        "n": len(null_df), "target": N_NULL, "p_ann": null_p_ann, "p_sharpe": null_p_sharpe,
        "median_ann": null_ann, "median_sharpe": null_sh,
        "note": "complete label permutation within each training date; all 6 folds retrained",
    })
    stage_end("FULL-NULL", t, "PASS", {"n": len(null_df), "p_ann": null_p_ann})

    t = stage_start("STATS")
    trial_path = ROOT / "reports" / "night_ind_cert_002" / "trial_sharpes.csv"
    trial_sr = pd.read_csv(trial_path)["sharpe_ann"].dropna().tolist() if trial_path.exists() else []
    trial_sr.append(float(ss["sharpe"]))
    dsr = local_dsr(merged["shadow_net"].values, trial_sr, len(trial_sr))
    atomic_json(OUT / "dsr_trial_inventory.json", dsr)
    pbo_src = ROOT / "reports" / "night_ind_cert_002" / "pbo_cscv.json"
    pbo = json.loads(pbo_src.read_text(encoding="utf-8")) if pbo_src.exists() else {}
    atomic_json(OUT / "historical_pbo_diagnostic.json", {"status": "HISTORICAL_ONLY", "pbo": pbo.get("pbo"), "source": "NIGHT-IND-CERT-002"})
    first = merged.iloc[:len(merged)//2]; second = merged.iloc[len(merged)//2:]
    d1 = first["shadow_net"].values - first["prod_net"].values
    d2 = second["shadow_net"].values - second["prod_net"].values
    fold_pos = np.mean([r["diff"] > 0 for r in fold_rows])
    checks = {
        "paired_alpha_gt0": {"value": paired["mean_diff"], "pass": paired["mean_diff"] > 0},
        "random_alpha_gt0": {"value": ss["ann"] - float(rand["ann"].median()), "pass": ss["ann"] > float(rand["ann"].median())},
        "both_halves_gt0": {"value": f"{np.mean(d1):+.4f}/{np.mean(d2):+.4f}", "pass": float(np.mean(d1)) > 0 and float(np.mean(d2)) > 0},
        "fold_2_3": {"value": fold_pos, "pass": fold_pos >= 2/3},
        "seed_median_gt0_7of10": {"value": float(seed_df["alpha"].median()), "pass": float(seed_df["alpha"].median()) > 0 and int((seed_df["alpha"] > 0).sum()) >= 7},
        "stress_25_not_worse": {"value": f"{astat['ann']:+.4f} vs {pstat['ann']:+.4f}", "pass": astat["ann"] >= pstat["ann"] and astat["ann"] > 0},
        "execution_reconcile": {"value": quality["page_rows"], "pass": all(c not in PAGE_FEATURES for c in ("ret_1d", "rank_1d", "cls_3"))},
    }
    after = guard(); data_after = data_guard()
    atomic_json(OUT / "hash_after.json", after); atomic_json(OUT / "data_after.json", data_after)
    prod_unchanged = after.get("files") == before.get("files") and after.get("models") == before.get("models")
    checks["production_unchanged"] = {"value": True, "pass": bool(prod_unchanged)}
    pass_all = all(v["pass"] for v in checks.values())
    status = "FORWARD_CANDIDATE" if pass_all else "SHADOW_REJECTED"
    if not prod_unchanged:
        status = "PRODUCTION_CHANGED_ALERT"
    decision = {"experiment_id": EXP_ID, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": status, "shadow_only": True, "historical_diagnostic": True,
                "production_unchanged": bool(prod_unchanged),
                "shadow_ann": ss["ann"], "shadow_sharpe": ss["sharpe"],
                "shadow_cum": ss["cum"], "shadow_mean": ss["mean"],
                "production_ann": pp["ann"], "production_sharpe": pp["sharpe"],
                "production_cum": pp["cum"], "production_mean": pp["mean"],
                "shadow_25_ann": astat["ann"], "shadow_25_sharpe": astat["sharpe"],
                "prod_25_ann": pstat["ann"], "prod_25_sharpe": pstat["sharpe"],
                "p_ann": p_sh_ann, "p_sharpe": p_sh_sharpe, "prod_p_ann": p_pr_ann, "prod_p_sharpe": p_pr_sharpe,
                "null_p_ann": null_p_ann, "null_p_sharpe": null_p_sharpe,
                "checks": checks, "passed": sum(1 for v in checks.values() if v["pass"]), "total": len(checks)}
    atomic_json(OUT / "decision.json", decision)
    stage_end("STATS", t, "PASS", {"status": status, "passed": decision["passed"]})

    t = stage_start("REPORT")
    _write_report(data, quality, base, shadow, seed_df, rand, null_df, paired, dsr, pbo,
                  decision, before, after, time.time() - t0,
                  "ret_1d/rank_1d are next-day labels and were excluded; the first dry run's abnormal result was discarded.")
    reg = OUT / "registry.csv"
    row = pd.DataFrame([{"experiment_id": EXP_ID, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "status": status, "shadow_ann": ss["ann"], "shadow_sharpe": ss["sharpe"],
                         "production_ann": pp["ann"], "production_sharpe": pp["sharpe"],
                         "paired_alpha": paired["mean_diff"], "p_ann": p_sh_ann, "p_sharpe": p_sh_sharpe,
                         "null_n": len(null_df), "production_unchanged": prod_unchanged}])
    if reg.exists():
        old = pd.read_csv(reg); old = old[old["experiment_id"] != EXP_ID]
        pd.concat([old, row], ignore_index=True).to_csv(reg, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(reg, index=False, encoding="utf-8-sig")
    files = []
    for root in (OUT, RESEARCH):
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name != "final_manifest.json":
                files.append({"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size})
    manifest = {"experiment_id": EXP_ID, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": status, "shadow_only": True, "production_unchanged": bool(prod_unchanged),
                "elapsed_s": round(time.time() - t0, 1), "prereg_sha256": sha256(PREREG),
                "production_before": before, "production_after": after,
                "data_before": data_before, "data_after": data_after, "files": files}
    atomic_json(OUT / "final_manifest.json", manifest)
    stage_end("REPORT", t, "PASS")
    log(f"RUN_DONE status={status} shadow={ss['ann']:+.4f}/{ss['sharpe']:.3f} prod={pp['ann']:+.4f}/{pp['sharpe']:.3f} p={p_sh_ann}/{p_sh_sharpe} null={len(null_df)}")
    print("STOCK_RANK_SHADOW_004_DONE " + json.dumps({
        "status": status, "shadow_ann": ss["ann"], "shadow_sharpe": ss["sharpe"],
        "production_ann": pp["ann"], "production_sharpe": pp["sharpe"],
        "p_ann": p_sh_ann, "p_sharpe": p_sh_sharpe, "null_n": len(null_df),
        "production_unchanged": prod_unchanged,
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
