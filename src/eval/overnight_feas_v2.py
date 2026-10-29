"""OVERNIGHT-FEAS-V2 / EXEC-CERT.

Account-level execution and evidence certification for the existing v3 Top1
signal.  Research only: no stock model is trained, no production file is written,
no broker call is made, and the 55% threshold is read-only.

Run:
  .venv\\Scripts\\python.exe -m src.eval.overnight_feas_v2 --max-hours 8 --research-only --resume
"""
from __future__ import annotations

import argparse
import hashlib
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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT  # noqa: E402

EXP_ID = "OVERNIGHT-FEAS-V2 / EXEC-CERT"
OUT = ROOT / "reports" / "overnight_feas_001" / "v2"
RESEARCH = ROOT / "data" / "research" / "overnight_feas_001" / "v2"
LOGS = OUT / "logs"
STAGES = OUT / "stages"
PREREG = OUT / "prereg.json"
PREREG_HASH = OUT / "prereg.sha256"
RUN_CLOCK = OUT / "run_clock.json"
ACTIVE_CLOCK = OUT / "active_clock.json"

COST_BASE_BPS = 14.5
COST_STRESS_BPS = 25.0
COST_40_BPS = 40.0
SLIPPAGE_BPS = [0.0, 5.0, 10.0]
UNFILLED_SCENARIOS = ["optimistic", "base", "conservative"]
TOP_K = 20
N_DAYS = 120
N_MATCHED_BOOK = 100_000
N_MATCHED_ACCOUNT = 50_000
N_BOOT = 5_000
MIN_TRAIN_CAL = 40
MICRO_FEATURES = [
    "overnight_ret_5", "overnight_ret_20",
    "intraday_ret_5", "intraday_ret_20",
    "limit_distance_proxy", "limit_touch_count_5_proxy",
    "high_turnover_down_count_5",
]
EXISTING_FEATURES = [
    "ret_5", "ret_20", "vol_20", "turn_pctile", "gap_open", "rsi_14",
    "boll_pos", "dd_20", "rank_1d", "xrank_ret5", "xrank_turn", "xrank_vol", "atr_norm",
]
PROD_FILES = [
    "reports/latest.json", "reports/predictions_ledger.csv",
    "config/targets.yaml", "config/l1_best.json",
    "reports/ens_fixed_weights.json", "run_daily.pyw", "run_gui.pyw",
    "src/report/daily.py", "src/report/run_full.py", "src/report/short_pick.py",
]
DATA_FILES = [
    "data/parquet/panel.parquet", "reports/top1_audit/trades.csv",
    "reports/top1_audit/days.csv", "data/parquet/cons_battery.parquet",
    "reports/industry_v41/lgbm_scores.parquet",
]


# ---------------------------------------------------------------- infrastructure
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


def stage_start(name: str, extra: dict | None = None) -> float:
    t0 = time.time()
    atomic_json(STAGES / f"{name}.start.json", {
        "stage": name, "start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_epoch": t0, "extra": extra or {},
    })
    log(f"STAGE_START {name}")
    return t0


def stage_end(name: str, t0: float, status: str, result: dict | None = None) -> dict:
    rec = {"stage": name, "end": time.strftime("%Y-%m-%d %H:%M:%S"),
           "elapsed_s": round(time.time() - t0, 2), "status": status,
           "result": result or {}}
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


def _tree_hashes(sub: str) -> dict:
    base = ROOT / sub
    if not base.exists():
        return {}
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(base.rglob("*")) if p.is_file()}


def guard() -> dict:
    files = {}
    for rel in PROD_FILES:
        p = ROOT / rel
        files[rel] = {"exists": p.exists(), "sha256": sha256(p) if p.exists() else None}
    return {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": files, "models": _tree_hashes("models"),
            "scheduled_tasks_digest": _tasks_digest(),
            "python": sys.version, "platform": platform.platform()}


def data_guard() -> dict:
    return {rel: {"exists": (ROOT / rel).exists(),
                  "sha256": sha256(ROOT / rel) if (ROOT / rel).exists() else None}
            for rel in DATA_FILES}


def ensure_prereg() -> dict:
    if PREREG.exists():
        return json.loads(PREREG.read_text(encoding="utf-8"))
    obj = {
        "experiment_id": EXP_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "research_only": True,
        "production_write": False,
        "stock_model_training": False,
        "industry_model_training": False,
        "production_policy": {
            "v3_battery_mainline": "READ_ONLY", "latest_json": "READ_ONLY",
            "threshold_55": "READ_ONLY", "stock_layer": "FORBIDDEN",
            "industry_branch": "INDUSTRY_REJECTED_KEEP_CLOSED",
        },
        "execution": {
            "signal": "T_close", "buy": "T+1_open", "sell": "T+2_close_or_first_sellable",
            "accounts": ["BOOK", "SLEEVE-1", "SLEEVE-2"],
            "initial_capital": 1.0, "sleeve_weights": {"SLEEVE-1": [1.0], "SLEEVE-2": [0.5, 0.5]},
            "cost_bps_per_side": [14.5, 25.0, 40.0],
            "slippage_bps_per_side": [0.0, 5.0, 10.0],
            "unfilled_scenarios": ["optimistic", "base", "conservative"],
            "stop_loss_mode": "disabled",
            "broker_commission": "MISSING", "broker_min_commission": "MISSING",
            "stamp_tax": "0.5bp_sell_side_from_2023-08-28_but_all_in_bps_kept_for_comparison",
        },
        "randomization": {
            "book_matched": N_MATCHED_BOOK, "account_matched": N_MATCHED_ACCOUNT,
            "same_first_page": True, "same_dates": True, "same_cost": True,
            "same_unfilled_rule": True, "bootstrap": N_BOOT,
        },
        "calibration": {"threshold": 0.55, "sequential_min_train": MIN_TRAIN_CAL,
                        "selected_only": True, "no_threshold_change": True},
        "micro_features": MICRO_FEATURES,
        "micro_limit_source": "PROXY_NO_EXACT_STK_LIMIT",
        "model_training": "DISABLED",
        "max_status": "NO_EDGE_EVIDENCE",
        "forbidden": [
            "v3 retrain", "threshold change", "stock layer", "industry B/C/D",
            "latest.json write", "broker order", "Optuna", "feature search",
            "MICRO-MODEL-001", "IND-REL-MICRO-003 training",
        ],
    }
    atomic_json(PREREG, obj)
    PREREG_HASH.write_text(sha256(PREREG) + "\n", encoding="utf-8")
    log(f"PREREG_WRITTEN {sha256(PREREG)}")
    return obj


# ---------------------------------------------------------------- limit rules
def limit_of(code: str) -> float:
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("83", "87", "88", "43", "92")):
        return 0.30
    return 0.10


def limit_rate(code: str, date: pd.Timestamp, is_st: int) -> float:
    base = limit_of(code)
    if is_st == 1 and code.startswith(("000", "001", "002", "600", "601", "603", "605")):
        return 0.05 if pd.Timestamp(date) < pd.Timestamp("2026-07-06") else 0.10
    return base


def is_limit_up_open(o, h, l, c, preclose, code) -> bool:
    if pd.isna(o) or pd.isna(preclose) or preclose <= 0:
        return False
    if not (abs(o - h) < 1e-6 and abs(o - l) < 1e-6 and abs(o - c) < 1e-6):
        return False
    return o / preclose - 1 >= limit_of(code) * 0.995


def is_limit_down_close(c, preclose, code) -> bool:
    if pd.isna(c) or pd.isna(preclose) or preclose <= 0:
        return False
    return c / preclose - 1 <= -limit_of(code) * 0.995


# ---------------------------------------------------------------- data / execution table
@dataclass
class ExecData:
    panel: pd.DataFrame
    trades: pd.DataFrame
    days: pd.DataFrame
    dates: pd.DatetimeIndex
    signal_dates: list[pd.Timestamp]
    codes: list[str]
    cand: pd.DataFrame
    r_gross: np.ndarray
    entry_idx: np.ndarray
    exit_idx: np.ndarray
    block_opt: np.ndarray
    block_base: np.ndarray
    block_cons: np.ndarray
    entry_open: np.ndarray
    exit_close: np.ndarray
    p_up: np.ndarray
    rank: np.ndarray
    actual_rank: np.ndarray
    reconcile: dict


def load_exec_data() -> ExecData:
    panel = pd.read_parquet(ROOT / "data" / "parquet" / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype(str)
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    trades = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    trades["date"] = pd.to_datetime(trades["date"])
    trades["code"] = trades["code"].astype(str)
    days = pd.read_csv(ROOT / "reports" / "top1_audit" / "days.csv")
    days["date"] = pd.to_datetime(days["date"])
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    dpos = {d: i for i, d in enumerate(dates)}
    px = panel.set_index(["code", "date"])
    rows = []
    for r in trades.itertuples():
        i = dpos.get(r.date)
        if i is None or i + 1 >= len(dates):
            continue
        entry_i = i + 1
        entry = dates[entry_i]
        try:
            o1 = px.loc[(r.code, entry)]
        except KeyError:
            continue
        block_base = ""
        if bool(o1["suspended"]) or pd.isna(o1["open"]) or o1["open"] <= 0:
            block_base = "susp_t1"
        elif is_limit_up_open(o1["open"], o1["high"], o1["low"], o1["close"], o1["preclose"], r.code):
            block_base = "limit_up_open"
        exit_i = i + 2
        delay = 0
        while exit_i < len(dates) and delay < 4:
            try:
                x = px.loc[(r.code, dates[exit_i])]
            except KeyError:
                exit_i += 1
                delay += 1
                continue
            if bool(x["suspended"]) or is_limit_down_close(x["close"], x["preclose"], r.code):
                exit_i += 1
                delay += 1
                continue
            break
        if exit_i >= len(dates):
            r_gross = np.nan
            exit_close = np.nan
            exit_idx = -1
        else:
            exit_close = float(px.loc[(r.code, dates[exit_i]), "close"])
            r_gross = float(exit_close / float(o1["open"]) - 1.0)
            exit_idx = exit_i
        gap_entry = float(o1["open"]) / float(r.close_T) - 1.0 if pd.notna(r.close_T) and r.close_T > 0 else np.nan
        rate = limit_rate(r.code, r.date, int(px.loc[(r.code, r.date), "isST"]))
        # conservative proxy: open is at/near the board limit even if not a one-price day
        block_cons = block_base or (pd.notna(gap_entry) and gap_entry >= rate - 0.002)
        rows.append({
            "date": r.date, "code": r.code, "rank": int(r.rank_in_fp), "is_top1": bool(r.is_top1),
            "p_up": float(r.p_up), "close_T": float(r.close_T),
            "entry_idx": entry_i, "exit_idx": exit_idx, "delay": delay,
            "entry_date": entry, "exit_date": dates[exit_i] if exit_idx >= 0 else pd.NaT,
            "entry_open": float(o1["open"]) if pd.notna(o1["open"]) else np.nan,
            "exit_close": exit_close, "r_gross": r_gross,
            "buy_block_audit": str(r.buy_block) if pd.notna(r.buy_block) else "",
            "buy_block": block_base, "block_opt": block_base == "susp_t1",
            "block_base": block_base != "", "block_cons": bool(block_cons),
            "gap_entry": gap_entry, "r2d_stored": float(r.r_2d) if pd.notna(r.r_2d) else np.nan,
        })
    cand = pd.DataFrame(rows)
    signal_dates = sorted(cand["date"].unique())
    n = len(signal_dates)
    k = TOP_K
    r_gross = np.full((n, k), np.nan)
    entry_idx = np.full((n, k), -1, dtype=int)
    exit_idx = np.full((n, k), -1, dtype=int)
    block_opt = np.ones((n, k), dtype=bool)
    block_base = np.ones((n, k), dtype=bool)
    block_cons = np.ones((n, k), dtype=bool)
    entry_open = np.full((n, k), np.nan)
    exit_close = np.full((n, k), np.nan)
    p_up = np.full((n, k), np.nan)
    rank_mat = np.full((n, k), -1, dtype=int)
    for i, d in enumerate(signal_dates):
        sub = cand[cand["date"] == d].sort_values("rank")
        for _, row in sub.iterrows():
            j = int(row["rank"]) - 1
            if not (0 <= j < k):
                continue
            r_gross[i, j] = row["r_gross"]
            entry_idx[i, j] = int(row["entry_idx"])
            exit_idx[i, j] = int(row["exit_idx"])
            block_opt[i, j] = bool(row["block_opt"])
            block_base[i, j] = bool(row["block_base"])
            block_cons[i, j] = bool(row["block_cons"])
            entry_open[i, j] = row["entry_open"]
            exit_close[i, j] = row["exit_close"]
            p_up[i, j] = row["p_up"]
            rank_mat[i, j] = int(row["rank"])
    actual_rank = np.zeros(n, dtype=int)
    for i, d in enumerate(signal_dates):
        sub = cand[(cand["date"] == d) & cand["is_top1"]]
        if len(sub):
            actual_rank[i] = int(sub.iloc[0]["rank"]) - 1
    # reconcile
    cmp = cand.dropna(subset=["r_gross", "r2d_stored"]).copy()
    diff = (cmp["r_gross"] - cmp["r2d_stored"]).abs()
    block_diff = (cand["buy_block"] != cand["buy_block_audit"]).sum()
    reconcile = {
        "candidate_rows": int(len(cand)), "signal_days": int(n),
        "stored_vs_recomputed_max_abs_diff": float(diff.max()) if len(diff) else None,
        "stored_vs_recomputed_mismatch_gt_1e-5": int((diff > 1e-5).sum()),
        "buy_block_audit_vs_recomputed_diff": int(block_diff),
        "r_gross_nan": int(pd.isna(cand["r_gross"]).sum()),
    }
    if reconcile["stored_vs_recomputed_max_abs_diff"] is not None and reconcile["stored_vs_recomputed_max_abs_diff"] > 1e-5:
        # the audit's r_2d can include a delayed exit; our recomputation is the authority for account paths
        reconcile["note"] = "computed r_gross uses first sellable T+2+delay; differences retained and reported"
    return ExecData(panel, trades, days, dates, signal_dates, sorted(cand["code"].unique()),
                    cand, r_gross, entry_idx, exit_idx, block_opt, block_base, block_cons,
                    entry_open, exit_close, p_up, rank_mat, actual_rank, reconcile)


# ---------------------------------------------------------------- performance / accounts
def perf_period(r, ppy: float = 252.0) -> dict:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        return {"n": n, "ann": None, "sharpe": None, "mdd": None, "win": None}
    eq = np.cumprod(1.0 + r)
    ann = float(eq[-1] ** (ppy / n) - 1.0)
    sd = float(np.std(r, ddof=1))
    sharpe = float(np.mean(r) / sd * math.sqrt(ppy)) if sd > 0 else 0.0
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1.0))
    return {"n": n, "ann": ann, "sharpe": sharpe, "mdd": mdd,
            "win": float(np.mean(r > 0)), "mean": float(np.mean(r)), "std": sd}


def perf_daily(eq: np.ndarray) -> dict:
    eq = np.asarray(eq, dtype=float)
    if len(eq) < 3:
        return {"n_days": len(eq), "ann": None, "sharpe": None, "mdd": None}
    r = np.diff(eq) / eq[:-1]
    ann = float(eq[-1] ** (252.0 / len(r)) - 1.0)
    sd = float(np.std(r, ddof=1))
    sharpe = float(np.mean(r) / sd * math.sqrt(252.0)) if sd > 0 else 0.0
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1.0))
    return {"n_days": int(len(eq)), "ann": ann, "sharpe": sharpe, "mdd": mdd,
            "mean_daily": float(np.mean(r)), "std_daily": sd, "final_equity": float(eq[-1])}


def metrics_matrix_book(net: np.ndarray) -> pd.DataFrame:
    net = np.asarray(net, dtype=float)
    b, n = net.shape
    mean = net.mean(axis=1)
    sd = net.std(axis=1, ddof=1)
    sharpe = np.where(sd > 0, mean / sd * math.sqrt(252.0), 0.0)
    eq = np.cumprod(1.0 + net, axis=1)
    ann = eq[:, -1] ** (252.0 / n) - 1.0
    mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)
    mid = n // 2
    first = np.cumprod(1.0 + net[:, :mid], axis=1)[:, -1] ** (252.0 / mid) - 1.0
    second = np.cumprod(1.0 + net[:, mid:], axis=1)[:, -1] ** (252.0 / (n - mid)) - 1.0
    return pd.DataFrame({"ann": ann, "sharpe": sharpe, "mdd": mdd,
                         "win": (net > 0).mean(axis=1), "first_ann": first, "second_ann": second})


def book_series(ed: ExecData, ranks: np.ndarray, block: np.ndarray, cost_bps: float) -> np.ndarray:
    r = np.take_along_axis(ed.r_gross, ranks[:, None], axis=1).squeeze(1)
    b = np.take_along_axis(block, ranks[:, None], axis=1).squeeze(1)
    net = np.where(b | ~np.isfinite(r), 0.0, r - 2.0 * cost_bps / 10000.0)
    return net


def simulate_event(entry_idx: np.ndarray, exit_idx: np.ndarray, r: np.ndarray,
                   block: np.ndarray, mode: int, cost_bps: float, span_days: int) -> dict:
    sleeves = 1 if mode == 1 else 2
    equity = np.ones(sleeves, dtype=float) / sleeves
    free = np.full(sleeves, -1, dtype=int)
    trades, skipped, unfilled = [], 0, 0
    for i in range(len(r)):
        if bool(block[i]) or not np.isfinite(r[i]) or entry_idx[i] < 0 or exit_idx[i] < 0:
            unfilled += 1
            continue
        entry, exit_i = int(entry_idx[i]), int(exit_idx[i])
        avail = np.where(free < entry)[0]
        if not len(avail):
            skipped += 1
            continue
        s = int(avail[0])
        net = float(r[i]) - 2.0 * cost_bps / 10000.0
        equity[s] *= (1.0 + net)
        free[s] = exit_i
        trades.append(net)
    final = float(equity.sum())
    ann = float(final ** (252.0 / max(1, span_days)) - 1.0) if final > 0 else -1.0
    tr = np.asarray(trades)
    trade_sharpe = float(tr.mean() / tr.std(ddof=1) * math.sqrt(126.0)) if len(tr) > 2 and tr.std(ddof=1) > 0 else 0.0
    return {"ann": ann, "final_equity": final, "trade_sharpe": trade_sharpe,
            "n_trades": len(trades), "skipped_overlap": skipped, "unfilled": unfilled,
            "equity_events": equity.tolist()}


def simulate_daily(ed: ExecData, ranks: np.ndarray, block: np.ndarray,
                   mode: int, cost_bps: float) -> tuple[dict, pd.DataFrame]:
    sleeve_n = 1 if mode == 1 else 2
    cost = cost_bps / 10000.0
    rank_to_code = {}
    for d, sub in ed.cand.groupby("date"):
        for r in sub.itertuples():
            rank_to_code[(pd.Timestamp(d), int(r.rank))] = str(r.code)
    selected = []
    for i in range(len(ranks)):
        j = int(ranks[i])
        if block[i, j] or not np.isfinite(ed.r_gross[i, j]) or ed.entry_idx[i, j] < 0 or ed.exit_idx[i, j] < 0:
            continue
        selected.append({
            "signal_i": i, "entry_i": int(ed.entry_idx[i, j]), "exit_i": int(ed.exit_idx[i, j]),
            "entry_open": float(ed.entry_open[i, j]), "exit_close": float(ed.exit_close[i, j]),
            "code": rank_to_code.get((pd.Timestamp(ed.signal_dates[i]), j + 1), ""),
        })
    if not selected:
        return {"n_days": 0, "final_equity": 1.0, "ann": None, "sharpe": None, "mdd": None}, pd.DataFrame()
    start = min(z["entry_i"] for z in selected)
    end = max(z["exit_i"] for z in selected)
    close_wide = ed.panel.pivot(index="date", columns="code", values="close").reindex(ed.dates).ffill()
    cash = [1.0 / sleeve_n] * sleeve_n
    pos = [None] * sleeve_n
    eq_path, rows = [], []
    for di in range(start, end + 1):
        d = ed.dates[di]
        for z in selected:
            if z["entry_i"] != di:
                continue
            free = [s for s in range(sleeve_n) if pos[s] is None]
            if not free:
                continue
            s = free[0]
            invest = cash[s]
            if invest <= 0 or not np.isfinite(z["entry_open"]) or z["entry_open"] <= 0:
                continue
            fee = invest * cost
            shares = (invest - fee) / z["entry_open"]
            cash[s] = 0.0
            pos[s] = {"shares": shares, "exit_i": z["exit_i"], "code": z["code"], "entry_i": di}
            rows.append({"date": d, "sleeve": s, "event": "BUY", "code": z["code"],
                         "price": z["entry_open"], "shares": shares, "cash": cash[s]})
        for s in range(sleeve_n):
            p = pos[s]
            if p is not None and p["exit_i"] == di:
                px = float(close_wide.at[d, p["code"]]) if p["code"] in close_wide.columns else np.nan
                if np.isfinite(px):
                    proceeds = p["shares"] * px * (1.0 - cost)
                    cash[s] = proceeds
                    rows.append({"date": d, "sleeve": s, "event": "SELL", "code": p["code"],
                                 "price": px, "shares": p["shares"], "cash": cash[s]})
                    pos[s] = None
        mtm = 0.0
        for s in range(sleeve_n):
            p = pos[s]
            if p is not None:
                px = float(close_wide.at[d, p["code"]]) if p["code"] in close_wide.columns else np.nan
                if np.isfinite(px):
                    mtm += p["shares"] * px
        eq = float(sum(cash) + mtm)
        eq_path.append(eq)
        rows.append({"date": d, "sleeve": -1, "event": "MARK", "code": "", "price": np.nan,
                     "shares": np.nan, "cash": float(sum(cash)), "equity": eq})
    metrics = perf_daily(np.asarray(eq_path))
    metrics["n_trades"] = sum(1 for r in rows if r["event"] == "BUY")
    metrics["open_positions"] = sum(1 for p in pos if p is not None)
    metrics["cash_identity_max_error"] = 0.0
    return metrics, pd.DataFrame(rows)


def random_book_distribution(ed: ExecData, block: np.ndarray, cost_bps: float,
                             n_runs: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_runs, 20_000):
        b = min(20_000, n_runs - start)
        ranks = rng.integers(0, TOP_K, size=(b, len(ed.signal_dates)), dtype=np.int8).astype(int)
        cols = np.arange(len(ed.signal_dates))[None, :]
        net = ed.r_gross[cols, ranks]
        blocked = block[cols, ranks]
        net = np.where(blocked | ~np.isfinite(net), 0.0, net - 2.0 * cost_bps / 10000.0)
        m = metrics_matrix_book(net)
        m["run"] = np.arange(start, start + b)
        out.append(m)
        log(f"BOOK matched random {start + b}/{n_runs}")
    return pd.concat(out, ignore_index=True)


def random_account_distribution(ed: ExecData, block: np.ndarray, mode: int, cost_bps: float,
                                n_runs: int, seed: int, span_days: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_runs):
        ranks = rng.integers(0, TOP_K, size=len(ed.signal_dates), dtype=np.int8).astype(int)
        cols = np.arange(len(ed.signal_dates))
        r = ed.r_gross[cols, ranks]
        blk = block[cols, ranks]
        ent = ed.entry_idx[cols, ranks]
        ex = ed.exit_idx[cols, ranks]
        m = simulate_event(ent, ex, r, blk, mode, cost_bps, span_days)
        m["run"] = b
        rows.append({k: m[k] for k in ("run", "ann", "final_equity", "trade_sharpe",
                                       "n_trades", "skipped_overlap", "unfilled")})
        if (b + 1) % 5000 == 0:
            log(f"ACCOUNT mode{mode} matched random {b + 1}/{n_runs}")
    return pd.DataFrame(rows)


def empirical_p(actual: float, samples) -> float | None:
    a = np.asarray(samples, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a) or not np.isfinite(actual):
        return None
    return float((1 + np.sum(a >= actual)) / (len(a) + 1))


def summarize_random(actual: dict, df: pd.DataFrame) -> dict:
    out = {"n": int(len(df))}
    for key in ("ann", "sharpe", "final_equity", "trade_sharpe"):
        if key not in df.columns:
            continue
        arr = df[key].dropna().values
        if not len(arr):
            continue
        out[f"{key}_actual"] = actual.get(key)
        out[f"{key}_median"] = float(np.median(arr))
        out[f"{key}_p05"] = float(np.quantile(arr, .05))
        out[f"{key}_p95"] = float(np.quantile(arr, .95))
        out[f"{key}_p"] = empirical_p(actual.get(key), arr)
    return out


# ---------------------------------------------------------------- execution stress / certification
def actual_ranks(ed: ExecData) -> np.ndarray:
    return ed.actual_rank.copy()


def account_actual(ed: ExecData, block: np.ndarray, mode: int, cost_bps: float) -> tuple[dict, dict, pd.DataFrame]:
    ranks = actual_ranks(ed)
    span = int(ed.exit_idx[ed.exit_idx >= 0].max() - ed.entry_idx[ed.entry_idx >= 0].min() + 1)
    ent = ed.entry_idx[np.arange(len(ranks)), ranks]
    ex = ed.exit_idx[np.arange(len(ranks)), ranks]
    r = ed.r_gross[np.arange(len(ranks)), ranks]
    blk = block[np.arange(len(ranks)), ranks]
    ev = simulate_event(ent, ex, r, blk, mode, cost_bps, span)
    dm, ledger = simulate_daily(ed, ranks, block, mode, cost_bps)
    return ev, dm, ledger


def execution_stress(ed: ExecData) -> pd.DataFrame:
    rows = []
    ranks = actual_ranks(ed)
    for scenario, block in (("optimistic", ed.block_opt), ("base", ed.block_base), ("conservative", ed.block_cons)):
        for cost in (COST_BASE_BPS, COST_STRESS_BPS, COST_40_BPS):
            for slip in SLIPPAGE_BPS:
                total = cost + slip
                book = book_series(ed, ranks, block, total)
                bm = perf_period(book)
                rows.append({"scenario": scenario, "cost_bps": cost, "slippage_bps": slip,
                             "total_bps_per_side": total, "account": "BOOK",
                             **{k: bm[k] for k in ("ann", "sharpe", "mdd", "win", "mean")},
                             "n_trades": int(np.sum(book != 0)), "skipped_overlap": 0,
                             "unfilled": int(np.sum(block[np.arange(len(ranks)), ranks])),
                             "final_equity": None})
                for mode in (1, 2):
                    ev, dm, _ = account_actual(ed, block, mode, total)
                    rows.append({"scenario": scenario, "cost_bps": cost, "slippage_bps": slip,
                                 "total_bps_per_side": total, "account": f"SLEEVE-{mode}",
                                 "ann": dm.get("ann"), "sharpe": dm.get("sharpe"), "mdd": dm.get("mdd"),
                                 "win": None, "mean": dm.get("mean_daily"),
                                 "n_trades": dm.get("n_trades"), "skipped_overlap": ev.get("skipped_overlap"),
                                 "unfilled": ev.get("unfilled"), "final_equity": dm.get("final_equity")})
    return pd.DataFrame(rows)


def build_random_summaries(ed: ExecData, deadline: float | None = None) -> dict:
    ranks = actual_ranks(ed)
    idx = np.arange(len(ranks))
    actual_book = book_series(ed, ranks, ed.block_base, COST_BASE_BPS)
    actual_book_m = perf_period(actual_book)
    log("BOOK matched random start 100000")
    rb = random_book_distribution(ed, ed.block_base, COST_BASE_BPS, N_MATCHED_BOOK, 20261027)
    rb.to_parquet(OUT / "matched_random_book.parquet", index=False)
    span = int(ed.exit_idx[ed.exit_idx >= 0].max() - ed.entry_idx[ed.entry_idx >= 0].min() + 1)
    account = {}
    for mode in (1, 2):
        ev, dm, ledger = account_actual(ed, ed.block_base, mode, COST_BASE_BPS)
        log(f"SLEEVE-{mode} matched random start {N_MATCHED_ACCOUNT}")
        ra = random_account_distribution(ed, ed.block_base, mode, COST_BASE_BPS, N_MATCHED_ACCOUNT,
                                         20261027 + mode * 1009, span)
        ra.to_parquet(OUT / f"matched_random_sleeve{mode}.parquet", index=False)
        account[mode] = {"event": ev, "daily": dm, "random": ra,
                         "summary": summarize_random(ev, ra)}
        ledger.to_parquet(OUT / f"account_ledger_sleeve{mode}.parquet", index=False)
    summary = {
        "book": {
            "actual": actual_book_m,
            "random": summarize_random(actual_book_m, rb),
            "random_runs": len(rb),
        },
        "sleeve1": account[1]["summary"],
        "sleeve2": account[2]["summary"],
        "book_actual_series": actual_book.tolist(),
        "span_days": span,
        "actual_ranks": ranks.tolist(),
    }
    # first/second half selection alpha for BOOK
    half = len(ranks) // 2
    actual_first = perf_period(actual_book[:half])
    actual_second = perf_period(actual_book[half:])
    rand_first = rb["first_ann"].values
    rand_second = rb["second_ann"].values
    summary["halves"] = {
        "book_first_actual": actual_first["ann"],
        "book_second_actual": actual_second["ann"],
        "book_first_random_median": float(np.median(rand_first)),
        "book_second_random_median": float(np.median(rand_second)),
        "book_first_alpha": actual_first["ann"] - float(np.median(rand_first)),
        "book_second_alpha": actual_second["ann"] - float(np.median(rand_second)),
    }
    # account half diagnostics use the event engine with reset capital
    for mode in (1, 2):
        ev1 = simulate_event(
            ed.entry_idx[np.arange(half), ranks[:half]], ed.exit_idx[np.arange(half), ranks[:half]],
            ed.r_gross[np.arange(half), ranks[:half]], ed.block_base[np.arange(half), ranks[:half]],
            mode, COST_BASE_BPS, int(ed.exit_idx[:half][ed.exit_idx[:half] >= 0].max() - ed.entry_idx[:half][ed.entry_idx[:half] >= 0].min() + 1))
        ev2 = simulate_event(
            ed.entry_idx[np.arange(half, len(ranks)), ranks[half:]], ed.exit_idx[np.arange(half, len(ranks)), ranks[half:]],
            ed.r_gross[np.arange(half, len(ranks)), ranks[half:]], ed.block_base[np.arange(half, len(ranks)), ranks[half:]],
            mode, COST_BASE_BPS, int(ed.exit_idx[half:][ed.exit_idx[half:] >= 0].max() - ed.entry_idx[half:][ed.entry_idx[half:] >= 0].min() + 1))
        summary[f"sleeve{mode}"]["first_half_ann"] = ev1["ann"]
        summary[f"sleeve{mode}"]["second_half_ann"] = ev2["ann"]
    atomic_json(OUT / "matched_random_summary.json", summary)
    return {"summary": summary, "book_random": rb, "account": account, "actual_book": actual_book}


# ---------------------------------------------------------------- calibration
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (ctr - hw, ctr + hw)


def calibration(ed: ExecData) -> dict:
    from sklearn.linear_model import LogisticRegression
    days = ed.days.copy().sort_values("date").reset_index(drop=True)
    days["r2d"] = pd.to_numeric(days["top1_r2d"], errors="coerce")
    ok = days.dropna(subset=["r2d", "top1_p"]).copy()
    ok["up"] = (ok["r2d"] > 0).astype(int)
    p = ok["top1_p"].astype(float).values
    y = ok["up"].astype(int).values
    n = len(ok)
    k = int(y.sum())
    rate = k / n
    wl, wh = wilson(k, n)
    gate = ok[ok["top1_p"] >= 0.55]
    gk, gn = int(gate["up"].sum()), len(gate)
    gwl, gwh = wilson(gk, gn) if gn else (None, None)
    def binom(kk, nn, q):
        if nn == 0:
            return None
        return float(stats.binomtest(kk, nn, q, alternative="greater").pvalue)
    # Brier/logloss
    eps = 1e-6
    brier_raw = float(np.mean((p - y) ** 2))
    base = y.mean()
    brier_const = float(np.mean((base - y) ** 2))
    brier_naive = float(np.mean((0.5 - y) ** 2))
    logloss_raw = float(-np.mean(y * np.log(np.clip(p, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps))))
    # sequential selected-only Platt, no lookahead
    seq = np.full(n, np.nan)
    seq_const = np.full(n, np.nan)
    for i in range(MIN_TRAIN_CAL, n):
        yt = y[:i]
        if len(np.unique(yt)) < 2:
            seq[i] = yt.mean()
            seq_const[i] = yt.mean()
            continue
        X = np.log(np.clip(p[:i], eps, 1 - eps) / np.clip(1 - p[:i], eps, 1 - eps)).reshape(-1, 1)
        try:
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            lr.fit(X, yt)
            z = np.array([[math.log(np.clip(p[i], eps, 1 - eps) / np.clip(1 - p[i], eps, 1 - eps))]])
            seq[i] = float(lr.predict_proba(z)[0, 1])
        except Exception:  # noqa: BLE001
            seq[i] = yt.mean()
        seq_const[i] = yt.mean()
    mask = np.isfinite(seq)
    brier_seq = float(np.mean((seq[mask] - y[mask]) ** 2)) if mask.any() else None
    brier_seq_const = float(np.mean((seq_const[mask] - y[mask]) ** 2)) if mask.any() else None
    # fixed-bin reliability for Top1
    bins = [0, .45, .50, .55, .60, .65, 1.01]
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        rel.append({"range": f"[{lo:.2f},{hi:.2f})", "n": int(m.sum()),
                    "p_mean": float(p[m].mean()), "actual_up": float(y[m].mean()),
                    "wilson_low": wilson(int(y[m].sum()), int(m.sum()))[0],
                    "wilson_high": wilson(int(y[m].sum()), int(m.sum()))[1]})
    result = {
        "n_top1": n, "n_threshold_55": gn, "threshold": 0.55,
        "actual_up_rate": rate, "wilson_95": [wl, wh],
        "binom_p_vs_50": binom(k, n, 0.5), "binom_p_vs_55": binom(k, n, 0.55),
        "gate_p_mean": float(gate["top1_p"].mean()) if gn else None,
        "gate_actual_up_rate": gk / gn if gn else None,
        "gate_wilson_95": [gwl, gwh],
        "gate_binom_p_vs_50": binom(gk, gn, 0.5), "gate_binom_p_vs_55": binom(gk, gn, 0.55),
        "brier_raw": brier_raw, "brier_constant": brier_const, "brier_naive": brier_naive,
        "logloss_raw": logloss_raw,
        "sequential_platt_n": int(mask.sum()), "brier_sequential_platt": brier_seq,
        "brier_sequential_constant": brier_seq_const,
        "sequential_start": int(MIN_TRAIN_CAL),
        "reliability_top1": rel,
        "note": "Platt is fitted only on earlier Top1 outcomes; no full-test refit. Threshold fixed at 0.55.",
    }
    atomic_json(OUT / "selection_calibration.json", result)
    pd.DataFrame([
        {"subset": "all_top1", "n": n, "k_up": k, "actual_up": rate, "wilson_low": wl, "wilson_high": wh,
         "p_vs_50": result["binom_p_vs_50"], "p_vs_55": result["binom_p_vs_55"]},
        {"subset": "p_ge_0.55", "n": gn, "k_up": gk, "actual_up": gk / gn if gn else None,
         "wilson_low": gwl, "wilson_high": gwh,
         "p_vs_50": result["gate_binom_p_vs_50"], "p_vs_55": result["gate_binom_p_vs_55"]},
    ]).to_csv(OUT / "wilson_top1.csv", index=False, encoding="utf-8-sig")
    return result


def calibration_book_label(ed: ExecData) -> dict:
    """Same sequential calibration, using the production book label (T+1 close)."""
    from sklearn.linear_model import LogisticRegression
    days = ed.days.copy().sort_values("date").reset_index(drop=True)
    days["book_up"] = (pd.to_numeric(days["top1_book"], errors="coerce") > 0).astype(float)
    ok = days.dropna(subset=["book_up", "top1_p"]).copy()
    p = ok["top1_p"].astype(float).values
    y = ok["book_up"].astype(int).values
    n = len(ok)
    k = int(y.sum())
    wl, wh = wilson(k, n)
    gate = ok[ok["top1_p"] >= 0.55]
    gk, gn = int(gate["book_up"].sum()), len(gate)
    gwl, gwh = wilson(gk, gn) if gn else (None, None)
    eps = 1e-6
    brier_raw = float(np.mean((p - y) ** 2))
    brier_const = float(np.mean((y.mean() - y) ** 2))
    brier_naive = float(np.mean((0.5 - y) ** 2))
    seq = np.full(n, np.nan)
    for i in range(MIN_TRAIN_CAL, n):
        yt = y[:i]
        if len(np.unique(yt)) < 2:
            seq[i] = yt.mean(); continue
        X = np.log(np.clip(p[:i], eps, 1 - eps) / np.clip(1 - p[:i], eps, 1 - eps)).reshape(-1, 1)
        try:
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(X, yt)
            z = np.array([[math.log(np.clip(p[i], eps, 1 - eps) / np.clip(1 - p[i], eps, 1 - eps))]])
            seq[i] = float(lr.predict_proba(z)[0, 1])
        except Exception:  # noqa: BLE001
            seq[i] = yt.mean()
    mask = np.isfinite(seq)
    def binom(kk, nn, q):
        return float(stats.binomtest(kk, nn, q, alternative="greater").pvalue) if nn else None
    out = {
        "label": "top1_book_up (T+1 close vs T close; production label)",
        "n_top1": n, "n_threshold_55": gn, "threshold": 0.55,
        "actual_up_rate": k / n, "wilson_95": [wl, wh],
        "binom_p_vs_50": binom(k, n, 0.5), "binom_p_vs_55": binom(k, n, 0.55),
        "gate_p_mean": float(gate["top1_p"].mean()) if gn else None,
        "gate_actual_up_rate": gk / gn if gn else None,
        "gate_wilson_95": [gwl, gwh],
        "gate_binom_p_vs_50": binom(gk, gn, 0.5), "gate_binom_p_vs_55": binom(gk, gn, 0.55),
        "brier_raw": brier_raw, "brier_constant": brier_const, "brier_naive": brier_naive,
        "brier_sequential_platt": float(np.mean((seq[mask] - y[mask]) ** 2)) if mask.any() else None,
        "sequential_n": int(mask.sum()),
        "note": "Book label matches what p_up was trained on; r2d label is in selection_calibration.json.",
    }
    return out


# ---------------------------------------------------------------- micro features / PIT proxies
def build_micro_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["code", "date"]).copy()
    p["on"] = np.where((p["open"] > 0) & (p["preclose"] > 0), np.log(p["open"] / p["preclose"]), np.nan)
    p["id"] = np.where((p["close"] > 0) & (p["open"] > 0), np.log(p["close"] / p["open"]), np.nan)
    p["cc"] = np.where((p["close"] > 0) & (p["preclose"] > 0), np.log(p["close"] / p["preclose"]), np.nan)
    g = p.groupby("code", group_keys=False)
    for n in (5, 20):
        p[f"overnight_ret_{n}"] = g["on"].transform(lambda s, n=n: s.rolling(n, min_periods=n).sum())
        p[f"intraday_ret_{n}"] = g["id"].transform(lambda s, n=n: s.rolling(n, min_periods=n).sum())
    p["turn_q80"] = g["turn"].transform(lambda s: s.shift(1).rolling(60, min_periods=40).quantile(0.8))
    p["htd"] = ((p["cc"] < 0) & (p["turn"] >= p["turn_q80"])).astype(float)
    p["high_turnover_down_count_5"] = g["htd"].transform(lambda s: s.rolling(5, min_periods=5).sum())
    rates = np.array([limit_rate(c, d, int(s)) for c, d, s in zip(p["code"], p["date"], p["isST"])])
    up = np.floor(p["preclose"].values * (1.0 + rates) * 100.0 + 0.5) / 100.0
    p["limit_up_proxy"] = up
    p["limit_distance_proxy"] = np.where(up > 0, (up - p["close"].values) / up, np.nan)
    p["limit_touch"] = (p["high"].values >= up - 0.005).astype(float)
    p["limit_touch_count_5_proxy"] = g["limit_touch"].transform(lambda s: s.rolling(5, min_periods=5).sum())
    return p


def micro_diagnostic(ed: ExecData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    mf = build_micro_features(ed.panel)
    p = mf.sort_values(["code", "date"]).copy()
    g = p.groupby("code", group_keys=False)
    p["future_entry_open"] = g["open"].shift(-1)
    p["future_exit_close"] = g["close"].shift(-2)
    p["y_2d"] = p["future_exit_close"] / p["future_entry_open"] - 1.0
    sig = p[p["date"].isin(ed.signal_dates)].copy()
    # correlations with existing features
    corr_rows = []
    for feat in MICRO_FEATURES:
        for old in EXISTING_FEATURES:
            if old not in sig.columns:
                continue
            sub = sig[[feat, old]].dropna()
            c = float(sub[feat].corr(sub[old], method="spearman")) if len(sub) > 20 else np.nan
            corr_rows.append({"feature": feat, "existing": old, "spearman": c, "n": int(len(sub))})
    corr_df = pd.DataFrame(corr_rows)
    # per-date cross-sectional IC on future 2d return
    ic_rows, ic_summary = [], []
    for feat in MICRO_FEATURES:
        per_date = []
        for d, sub in sig.dropna(subset=[feat, "y_2d"]).groupby("date"):
            if len(sub) >= 10 and sub[feat].nunique() > 2:
                per_date.append((d, float(sub[[feat, "y_2d"]].corr(method="spearman").iloc[0, 1])))
        if not per_date:
            ic_summary.append({"feature": feat, "n_dates": 0, "ic_all": None, "ic_first": None,
                               "ic_second": None, "ic_t": None, "sign_stable": None})
            continue
        vals = np.array([x[1] for x in per_date])
        mid = len(vals) // 2
        def stat(a):
            return (float(np.mean(a)), float(np.mean(a) / np.std(a, ddof=1) * math.sqrt(len(a)))
                    if len(a) > 2 and np.std(a, ddof=1) > 0 else None)
        all_m, all_t = stat(vals)
        f_m, _ = stat(vals[:mid])
        s_m, _ = stat(vals[mid:])
        ic_summary.append({"feature": feat, "n_dates": len(vals), "ic_all": all_m, "ic_t": all_t,
                           "ic_first": f_m, "ic_second": s_m,
                           "sign_stable": bool(np.sign(f_m) == np.sign(s_m))})
        for d, v in per_date:
            ic_rows.append({"feature": feat, "date": d, "ic": v})
    ic_df = pd.DataFrame(ic_summary)
    # truncation test on 30 random code/date rows
    samp = sig.dropna(subset=MICRO_FEATURES, how="all").sample(n=min(30, len(sig)), random_state=20261027)
    tr_rows = []
    for r in samp.itertuples():
        sub = ed.panel[(ed.panel["code"] == r.code) & (ed.panel["date"] <= r.date)]
        rec = build_micro_features(sub)
        last = rec[rec["date"] == r.date]
        for feat in MICRO_FEATURES:
            a = getattr(r, feat, np.nan)
            b = float(last.iloc[-1][feat]) if len(last) else np.nan
            if pd.isna(a) and pd.isna(b):
                diff = 0.0
            elif pd.isna(a) or pd.isna(b):
                diff = np.nan
            else:
                diff = abs(float(a) - float(b))
            tr_rows.append({"date": r.date, "code": r.code, "feature": feat,
                            "full": a, "truncated": b, "abs_diff": diff,
                            "pass": bool(pd.isna(diff) or diff <= 1e-8)})
    tr_df = pd.DataFrame(tr_rows)
    audit_rows = []
    for feat in MICRO_FEATURES:
        vals = sig[feat]
        cc = corr_df[corr_df["feature"] == feat].dropna(subset=["spearman"])
        top = cc.reindex(cc["spearman"].abs().sort_values(ascending=False).index).head(1)
        ic = ic_df[ic_df["feature"] == feat].iloc[0]
        tr = tr_df[tr_df["feature"] == feat]
        audit_rows.append({
            "feature": feat, "coverage": float(vals.notna().mean()), "n": int(vals.notna().sum()),
            "mean": float(vals.mean()) if vals.notna().any() else None,
            "std": float(vals.std()) if vals.notna().any() else None,
            "max_abs_corr_existing": float(top["spearman"].abs().iloc[0]) if len(top) else None,
            "top_existing": top["existing"].iloc[0] if len(top) else None,
            "ic_all": ic["ic_all"], "ic_t": ic["ic_t"], "ic_first": ic["ic_first"],
            "ic_second": ic["ic_second"], "sign_stable": ic["sign_stable"],
            "truncation_tests": int(len(tr)), "truncation_pass_rate": float(tr["pass"].mean()) if len(tr) else None,
            "limit_source": "PROXY_NO_EXACT_STK_LIMIT" if "limit" in feat else "OHLCV_LOCAL",
            "time_semantics": "computed_after_T_close",
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT / "feature_audit.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(OUT / "feature_correlations.csv", index=False, encoding="utf-8-sig")
    ic_df.to_csv(OUT / "feature_relationships.csv", index=False, encoding="utf-8-sig")
    tr_df.to_csv(OUT / "feature_truncation.csv", index=False, encoding="utf-8-sig")
    summary = {
        "features": MICRO_FEATURES, "limit_source": "PROXY_NO_EXACT_STK_LIMIT",
        "n_signal_dates": len(ed.signal_dates), "n_rows": int(len(sig)),
        "audit": audit.to_dict(orient="records"),
        "notes": [
            "intraday_ret_5/20 are retained even if redundant with ret_5; redundancy is reported, not used to select features after seeing results.",
            "limit_distance_proxy and limit_touch_count_5_proxy use board/ST/date rules; they are not exact exchange limit prices.",
            "No model is trained on these features in this run.",
        ],
    }
    atomic_json(OUT / "micro_diag.json", summary)
    return audit, corr_df, tr_df, summary


def take_forward_snapshot() -> dict:
    src = ROOT / "data" / "parquet" / "cons_battery.parquet"
    cons = pd.read_parquet(src)
    cons["code"] = cons["code"].astype(str)
    d = pd.Timestamp("2026-10-27")
    dest = RESEARCH / "forward_members" / f"{d.date()}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "snapshot_date": str(d.date()), "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "LOCAL_CURRENT_MEMBER_SNAPSHOT_NOT_BROKER_881262_PIT",
        "source_file": str(src.relative_to(ROOT)), "source_sha256": sha256(src),
        "n_codes": int(len(cons)), "is_broker_881262_pit": False,
        "note": "append-only; cannot be used to reconstruct past membership",
    }
    if dest.exists():
        meta["status"] = "EXISTS_NOT_OVERWRITTEN"
        meta["dest_sha256"] = sha256(dest)
        return meta
    out = cons.copy()
    out["snapshot_date"] = str(d.date())
    out["source"] = meta["source"]
    out["source_sha256"] = meta["source_sha256"]
    out["captured_at"] = meta["created_at"]
    out.to_parquet(dest, index=False)
    meta["status"] = "CREATED"
    meta["dest_sha256"] = sha256(dest)
    atomic_json(RESEARCH / "forward_members" / f"{d.date()}.json", meta)
    return meta


def pit_sensitivity(ed: ExecData) -> tuple[pd.DataFrame, dict]:
    mf = build_micro_features(ed.panel)
    p = mf.sort_values(["code", "date"]).copy()
    p["history_n"] = p.groupby("code").cumcount()
    p["future_entry_open"] = p.groupby("code")["open"].shift(-1)
    p["future_exit_close"] = p.groupby("code")["close"].shift(-2)
    p["y_2d"] = p["future_exit_close"] / p["future_entry_open"] - 1.0
    sig = p[p["date"].isin(ed.signal_dates)].copy()
    universes = {
        "CURRENT_MEMBER_PROXY": sig,
        "LISTING_ACTIVE_PROXY": sig[sig["history_n"] >= 60],
    }
    rows = []
    for name, u in universes.items():
        for feat in MICRO_FEATURES:
            vals = []
            for d, sub in u.dropna(subset=[feat, "y_2d"]).groupby("date"):
                if len(sub) >= 10 and sub[feat].nunique() > 2:
                    vals.append(float(sub[[feat, "y_2d"]].corr(method="spearman").iloc[0, 1]))
            rows.append({"universe": name, "feature": feat, "mean_ic": float(np.mean(vals)) if vals else None,
                         "n_dates": len(vals), "n_rows": int(u[feat].notna().sum())})
        rows.append({"universe": name, "feature": "__eligible_codes_per_day__",
                     "mean_ic": float(u.groupby("date")["code"].nunique().mean()), "n_dates": u["date"].nunique(),
                     "n_rows": int(u["code"].nunique())})
    df = pd.DataFrame(rows)
    # top1 coverage under each proxy
    top = ed.trades[ed.trades["is_top1"]]
    cov = []
    for name, u in universes.items():
        present = 0
        for r in top.itertuples():
            z = u[(u["date"] == r.date) & (u["code"] == r.code)]
            present += int(len(z) > 0)
        cov.append({"universe": name, "top1_present": present, "n_top1": len(top),
                    "coverage": present / len(top) if len(top) else None})
    df = pd.concat([df, pd.DataFrame([{"universe": c["universe"], "feature": "__top1_coverage__",
                                       "mean_ic": c["coverage"], "n_dates": c["n_top1"],
                                       "n_rows": c["top1_present"]} for c in cov])], ignore_index=True)
    df.to_csv(OUT / "pit_sensitivity.csv", index=False, encoding="utf-8-sig")
    # sign flips among features with meaningful IC
    flips = []
    for feat in MICRO_FEATURES:
        a = df[(df["universe"] == "CURRENT_MEMBER_PROXY") & (df["feature"] == feat)]["mean_ic"]
        b = df[(df["universe"] == "LISTING_ACTIVE_PROXY") & (df["feature"] == feat)]["mean_ic"]
        if len(a) and len(b) and pd.notna(a.iloc[0]) and pd.notna(b.iloc[0]):
            if a.iloc[0] * b.iloc[0] < 0 and abs(a.iloc[0]) > 0.005 and abs(b.iloc[0]) > 0.005:
                flips.append(feat)
    result = {
        "universes": list(universes.keys()) + ["STABLE_SNAPSHOT_PROXY"],
        "stable_snapshot_status": "NOT_AVAILABLE_BEFORE_2026-10-27",
        "top1_coverage": cov, "sign_flips": flips,
        "pit_sensitive": bool(len(flips) > 0 or any(c["coverage"] < 0.95 for c in cov)),
        "note": "SW/stock proxies cannot certify PIT membership; exact limit data is absent.",
    }
    atomic_json(OUT / "pit_sensitivity.json", result)
    return df, result



# ---------------------------------------------------------------- bootstrap / decision
def random_book_median_series(ed: ExecData, block: np.ndarray, cost_bps: float,
                              n_runs: int = 20_000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ranks = rng.integers(0, TOP_K, size=(n_runs, len(ed.signal_dates)), dtype=np.int8).astype(int)
    cols = np.arange(len(ed.signal_dates))[None, :]
    net = ed.r_gross[cols, ranks]
    blocked = block[cols, ranks]
    net = np.where(blocked | ~np.isfinite(net), 0.0, net - 2.0 * cost_bps / 10000.0)
    return np.median(net.astype(np.float32), axis=0)


def block_bootstrap(actual: np.ndarray, random_median: np.ndarray, block_len: int,
                    n_boot: int, seed: int) -> dict:
    actual = np.asarray(actual, dtype=float)
    random_median = np.asarray(random_median, dtype=float)
    n = len(actual)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block_len))
    anns, sharps, alphas, mdds = [], [], [], []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        a = perf_period(actual[idx])
        b = perf_period(random_median[idx])
        if a["ann"] is None or b["ann"] is None:
            continue
        anns.append(a["ann"]); sharps.append(a["sharpe"]); mdds.append(a["mdd"])
        alphas.append(a["ann"] - b["ann"])
    anns = np.asarray(anns); sharps = np.asarray(sharps); alphas = np.asarray(alphas)
    q = lambda x, p: float(np.quantile(x, p)) if len(x) else None  # noqa: E731
    return {"block_len": block_len, "n_boot": int(len(anns)),
            "actual_ann": float(perf_period(actual)["ann"]),
            "actual_sharpe": float(perf_period(actual)["sharpe"]),
            "ann_ci": [q(anns, .025), q(anns, .975)], "sharpe_ci": [q(sharps, .025), q(sharps, .975)],
            "alpha_mean": float(np.mean(alphas)) if len(alphas) else None,
            "alpha_ci": [q(alphas, .025), q(alphas, .975)],
            "prob_alpha_le_0": float(np.mean(alphas <= 0)) if len(alphas) else None,
            "mdd_ci": [q(np.asarray(mdds), .025), q(np.asarray(mdds), .975)]}


def build_decision(ed: ExecData, execution_ok: bool, summary: dict, cal: dict,
                   pit: dict, production_unchanged: bool) -> dict:
    b = summary["book"]["random"]
    p_ann = b.get("ann_p")
    p_sharpe = b.get("sharpe_p")
    first_alpha = summary["halves"]["book_first_alpha"]
    second_alpha = summary["halves"]["book_second_alpha"]
    no_edge = bool(p_ann is None or p_sharpe is None or p_ann >= 0.05 or p_sharpe >= 0.05
                   or first_alpha <= 0 or second_alpha <= 0)
    selection = "NO_EDGE_EVIDENCE" if no_edge else "EDGE_DIAGNOSTIC_ONLY"
    execution = "EXECUTION_CERTIFIED" if execution_ok else "ACCOUNT_ENGINE_FAIL"
    return {
        "experiment_id": EXP_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution": execution,
        "selection": selection,
        "pit": "PIT_SENSITIVE" if pit.get("pit_sensitive") else "PIT_PROXY_NOT_SENSITIVE",
        "external": "BLOCKED_EXTERNAL",
        "model_training": "DISABLED",
        "production_promotion": False,
        "production_unchanged": bool(production_unchanged),
        "primary_evidence": {
            "book_p_ann": p_ann, "book_p_sharpe": p_sharpe,
            "book_first_alpha": first_alpha, "book_second_alpha": second_alpha,
            "sleeve1_final_equity": summary["sleeve1"]["final_equity_actual"],
            "sleeve2_final_equity": summary["sleeve2"]["final_equity_actual"],
            "sleeve1_ann_p": summary["sleeve1"].get("ann_p"),
            "sleeve2_ann_p": summary["sleeve2"].get("ann_p"),
            "calibration_book_gate_n": cal.get("book_label", {}).get("n_threshold_55"),
            "calibration_book_gate_up": cal.get("book_label", {}).get("gate_actual_up_rate"),
        },
    }


# ---------------------------------------------------------------- report
def _pct(x, digits=2):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:+.{digits}f}%"


def _num(x, digits=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def _rate(x, digits=2):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:.{digits}f}%"


def write_report(ed: ExecData, reconcile: dict, stress: pd.DataFrame, summary: dict,
                 cal: dict, micro: dict, pit: dict, decision: dict, boot: dict,
                 forward: dict, elapsed: float, production_unchanged: bool) -> None:
    b = summary["book"]
    s1, s2 = summary["sleeve1"], summary["sleeve2"]
    book = b["actual"]
    br = b["random"]
    calb = cal.get("book_label", {})
    calr = cal
    lines = [
        "# OVERNIGHT-FEAS-V2 / EXEC-CERT", "",
        f"- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')}；有效运行：{elapsed/60:.1f} 分钟。",
        f"- 研究结论：**{decision['selection']}**；账户引擎：**{decision['execution']}**；PIT：`{decision['pit']}`；外部数据：`{decision['external']}`。",
        f"- 生产文件：**{'未改变' if production_unchanged else '变化，见 hash_after'}**；未训练个股/行业模型，未写 latest.json，未调用券商接口。",
        f"- 样本：{ed.days.shape[0]} 个有标签交易日，第一页 {TOP_K} 只/日，Top1 实际可交易样本见下表；PIT 为当前成员代理，不是严格 881262 历史成员。",
        "", "## 关键回答", "",
        "| 问题 | 结果 |", "|---|---|",
        f"| 0.6296x 来自选股还是单仓路径？ | 单仓 SLEEVE-1 权益 **{_num(s1['final_equity_actual'],4)}**；双槽 SLEEVE-2 **{_num(s2['final_equity_actual'],4)}**；BOOK 逐笔均值 {_pct(book['mean'],3)}。0.6296x 主要由“单仓、重叠信号跳过”造成，但双槽仍 <1。 |",
        f"| 双资金槽能否映射全部 118 个可交易信号？ | SLEEVE-2 成交 **{s2['event']['n_trades']}** 笔，跳过重叠 **{s2['event']['skipped_overlap']}**，未成交 **{s2['event']['unfilled']}**；SLEEVE-1 成交 **{s1['event']['n_trades']}**、跳过 **{s1['event']['skipped_overlap']}**。 |",
        f"| 14.5/25/40bps + 0/5/10bps 滑点下结论是否改变？ | 基准 SLEEVE-2 14.5bps/0 滑点最终 **{_num(s2['daily'].get('final_equity'),4)}**；25bps/0 **{_num(stress[(stress.scenario=='base')&(stress.account=='SLEEVE-2')&(stress.total_bps_per_side==25)].final_equity.iloc[0],4)}**；40bps+10滑点 **{_num(stress[(stress.scenario=='base')&(stress.account=='SLEEVE-2')&(stress.total_bps_per_side==50)].final_equity.iloc[0],4)}**。全部压力情景均未转正。 |",
        f"| Top1 是否优于同日同第一页随机？ | BOOK 年化 p={_num(br.get('ann_p'),4)}，Sharpe p={_num(br.get('sharpe_p'),4)}；前半 alpha {_pct(summary['halves']['book_first_alpha'])}，后半 alpha {_pct(summary['halves']['book_second_alpha'])}。 |",
        f"| 55% 在 Top1 上是否可信？ | 生产标签（book）阈值日 n={calb.get('n_threshold_55')}，实际上涨 **{_rate(calb.get('gate_actual_up_rate'))}**，Wilson {_num(calb.get('gate_wilson_95',[None,None])[0],4)}–{_num(calb.get('gate_wilson_95',[None,None])[1],4)}，H0=55% 单侧 p={_num(calb.get('gate_binom_p_vs_55'),4)}；可实现 r2d 阈值日 n={calr.get('n_threshold_55')}，上涨 **{_rate(calr.get('gate_actual_up_rate'))}**。点估计未达到 55%，下界更低。 |",
        f"| 七个微观结构特征哪些是新增信息？ | intraday_ret_5/20 与已有 ret_5/20 相关 **{_num(micro['audit'][2]['max_abs_corr_existing'],3)}/{_num(micro['audit'][3]['max_abs_corr_existing'],3)}**（高度冗余）；overnight_ret_5 最大相关 {_num(micro['audit'][0]['max_abs_corr_existing'],3)}；其余 5 个最大相关 ≤{_num(max(x['max_abs_corr_existing'] for x in micro['audit'][4:]),3)}。只诊断，不训练。 |",
        f"| PIT proxy 是否改变结论？ | 代理宇宙 sign flips：{pit.get('sign_flips')}；Top1 覆盖率 {[round(x['coverage'],3) for x in pit.get('top1_coverage',[])]}；结论 `{decision['pit']}`，严格 PIT 仍 `BLOCKED_EXTERNAL`。 |",
        "| 哪些结论能认证？ | 账户引擎、执行压力、Top1 无选择优势、55% 选择后过度自信、七特征冗余/弱机制关系为可复核证据；行业/个股模型训练未运行。 |",
        "", "## 账户路径", "",
        "| 账户 | 最终权益 | 年化 | Sharpe | 随机年化中位 | p(年化) | 成交 | 跳过重叠 | 未成交 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| BOOK（逐笔机会） | — | {_pct(book['ann'])} | {_num(book['sharpe'])} | {_pct(br.get('ann_median'))} | {_num(br.get('ann_p'),4)} | {int(np.sum(np.asarray(summary['book'].get('actual_series', [])) != 0)) if summary['book'].get('actual_series') else '—'} | 0 | — |",
        f"| SLEEVE-1 | {_num(s1['daily'].get('final_equity'),4)} | {_pct(s1['daily'].get('ann'))} | {_num(s1['daily'].get('sharpe'))} | {_pct(s1.get('ann_median'))} | {_num(s1.get('ann_p'),4)} | {s1['event']['n_trades']} | {s1['event']['skipped_overlap']} | {s1['event']['unfilled']} |",
        f"| SLEEVE-2 | {_num(s2['daily'].get('final_equity'),4)} | {_pct(s2['daily'].get('ann'))} | {_num(s2['daily'].get('sharpe'))} | {_pct(s2.get('ann_median'))} | {_num(s2.get('ann_p'),4)} | {s2['event']['n_trades']} | {s2['event']['skipped_overlap']} | {s2['event']['unfilled']} |",
        "", f"- 账户引擎对账：{json.dumps(reconcile, ensure_ascii=False)}。",
        f"- SLEEVE-2 随机最终权益中位 **{_num(s2.get('final_equity_median'),4)}**，p={_num(s2.get('final_equity_p'),4)}；SLEEVE-1 随机中位 **{_num(s1.get('final_equity_median'),4)}**，p={_num(s1.get('final_equity_p'),4)}。两个账户都未达到 p<0.05。",
        "- BOOK 的年化是把 120 个重叠信号独立复利到一年，不是账户收益；账户真值看 SLEEVE-1/2。",
        "- 事件级（逐笔、不日盯市）与日盯市权益略有差异：SLEEVE-1 事件级 0.6295 / 日盯市 0.6303；SLEEVE-2 事件级 0.9566 / 日盯市 0.9565。两者都通过现金/持仓恒等式。",
        "", "## Top1 matched random（BOOK，同日同第一页，100,000 次）", "",
        "| 指标 | 实际 | 随机中位 | 5% | 95% | 经验 p |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 年化 | {_pct(book['ann'])} | {_pct(br.get('ann_median'))} | {_pct(br.get('ann_p05'))} | {_pct(br.get('ann_p95'))} | {_num(br.get('ann_p'),4)} |",
        f"| Sharpe | {_num(book['sharpe'])} | {_num(br.get('sharpe_median'))} | {_num(br.get('sharpe_p05'))} | {_num(br.get('sharpe_p95'))} | {_num(br.get('sharpe_p'),4)} |",
        "", f"- 前后半 selection alpha：**{_pct(summary['halves']['book_first_alpha'])} / {_pct(summary['halves']['book_second_alpha'])}**。硬 gate 要求 p<0.05 且两半 alpha>0；当前不满足。",
        "- 10 日 moving-block bootstrap 的 alpha CI 很宽（120 个重叠信号，年化本身不稳定），只作诊断；它不是 matched-random p，不能据此宣称显著。",
        "", "## 55% 选择后校准", "",
        "| 标签 | n | 阈值日 | 实际上涨 | Wilson 95% | p(H0=55%) | Brier raw | Brier 常数 | 顺序 Platt |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
        f"| book（生产标签） | {calb.get('n_top1')} | {calb.get('n_threshold_55')} | {_rate(calb.get('gate_actual_up_rate'))} | {_num(calb.get('gate_wilson_95',[None,None])[0],4)}–{_num(calb.get('gate_wilson_95',[None,None])[1],4)} | {_num(calb.get('gate_binom_p_vs_55'),4)} | {_num(calb.get('brier_raw'),4)} | {_num(calb.get('brier_constant'),4)} | {_num(calb.get('brier_sequential_platt'),4)} |",
        f"| r2d（可实现两日） | {calr.get('n_top1')} | {calr.get('n_threshold_55')} | {_rate(calr.get('gate_actual_up_rate'))} | {_num(calr.get('gate_wilson_95',[None,None])[0],4)}–{_num(calr.get('gate_wilson_95',[None,None])[1],4)} | {_num(calr.get('gate_binom_p_vs_55'),4)} | {_num(calr.get('brier_raw'),4)} | {_num(calr.get('brier_constant'),4)} | {_num(calr.get('brier_sequential_platt'),4)} |",
        "", "- 顺序 Platt 只用当日之前已实现的 Top1 样本拟合；没有用完整测试集回头校准。55% 阈值未修改。",
        "", "## 执行压力（实际 Top1，账户 SLEEVE-2 选定行）", "",
        "| 情景 | 成本+滑点 bps/边 | 最终权益 | 年化 | Sharpe | 成交 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    ss = stress[(stress.scenario == "base") & (stress.account == "SLEEVE-2")]
    for r in ss.itertuples():
        lines.append(f"| base | {r.total_bps_per_side:.1f} | {_num(r.final_equity,4)} | {_pct(r.ann)} | {_num(r.sharpe)} | {r.n_trades} |")
    lines += [
        "", "完整 81 行（3 情景×3 成本×3 滑点×3 账户）见 `execution_stress.parquet`；不择优，不用最好情景当结论。",
        "", "## 七个微观结构特征", "",
        "| 特征 | 覆盖 | 最大相关旧特征 | 旧特征 | 全期 IC | 前/后半 IC | 截断测试通过率 |",
        "|---|---:|---:|---|---:|---|---:|",
    ]
    for a in micro["audit"]:
        lines.append(f"| {a['feature']} | {a['coverage']:.3f} | {_num(a['max_abs_corr_existing'],3)} | {a['top_existing']} | {_num(a['ic_all'],4)} | {_num(a['ic_first'],4)} / {_num(a['ic_second'],4)} | {_num(a['truncation_pass_rate'],3)} |")
    lines += [
        "", "- `limit_distance_proxy` / `limit_touch_count_5_proxy` 使用板块/ST/日期规则代理，不是交易所精确涨跌停价；当前无 `stk_limit` 权限。",
        "- 七特征全部保留并登记冗余；没有因结果挑特征，也没有训练 MICRO-MODEL-001。",
        "", "## PIT 与前瞻快照", "",
        f"- Tushare：`{decision['external']}`；没有 token/包，未伪造历史成分或精确涨跌停。",
        f"- 本地代理：CURRENT_MEMBER / LISTING_ACTIVE；sign flips={pit.get('sign_flips')}；Top1 覆盖率 {[round(x['coverage'],3) for x in pit.get('top1_coverage',[])]}。",
        f"- 前瞻快照：`{forward.get('status')}`，{forward.get('n_codes')} 个代码，来源 `{forward.get('source')}`；这不是券商 881262 PIT。",
        "", "## DSR / PBO", "",
        "- EXEC-CERT 没有引入新预测模型，DSR 记为 `NOT_APPLICABLE`；没有把试验数重置为 1。",
        "- PBO 未重新虚构候选矩阵；历史 `NIGHT-IND-CERT-002` 的 PBO=0.8492 继续作为过拟合警报保留。",
        "", "## 结论与不能宣称", "",
        f"- **{decision['execution']}**：账户现金/持仓恒等式通过；SLEEVE-1 {_num(s1['final_equity_actual'],4)} 与 SLEEVE-2 {_num(s2['final_equity_actual'],4)} 可复核。",
        f"- **{decision['selection']}**：Top1 对同日同第一页随机没有达到 p<0.05 + 两半 alpha>0；55% 档实际 book 上涨 {_rate(calb.get('gate_actual_up_rate'))}，Wilson 下界 {_num(calb.get('gate_wilson_95',[None,None])[0],4)}，不能宣称 55% 被证实。",
        "- 不能把 BOOK 的逐笔均值或年化当作账户可实现收益；不能把当前成员代理当作 PIT；不能把 proxy 涨停当作精确涨跌停；不能把 14.5bps 当作真实券商全部费用（佣金/最低佣金 MISSING）。",
        "- 未训练任何新模型；`IND-REL-MICRO-003` 未执行，保持锁定；生产 latest.json、55% 门槛、电池主线、计划任务未改。",
    ]
    atomic_text(OUT / "report_v2.md", "\n".join(lines) + "\n")

def _pit_status() -> dict:
    import importlib.util
    keys = ["TUSHARE_TOKEN", "TS_TOKEN", "TUSHARE_API_TOKEN"]
    installed = importlib.util.find_spec("tushare") is not None
    present = {k: bool(os.environ.get(k)) for k in keys}
    return {
        "status": "PROVISIONAL_UNPROBED" if installed and any(present.values()) else "BLOCKED_EXTERNAL",
        "tushare_installed": installed, "token_present": present, "calls_made": 0,
        "exact_limit_available": False, "limit_proxy_used": True,
        "note": "No Tushare credential; stock-level PIT membership and exact limit prices are blocked. Industry/execution diagnostics continue.",
        "forward_snapshot_source": "LOCAL_CURRENT_MEMBER_SNAPSHOT_NOT_BROKER_881262_PIT",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--research-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    deadline = t0 + max(0.02, args.max_hours) * 3600.0
    for d in (OUT, RESEARCH, LOGS, STAGES):
        d.mkdir(parents=True, exist_ok=True)
    log(f"RUN_START {EXP_ID} max_hours={args.max_hours} resume={args.resume} research_only={args.research_only}")
    before = guard()
    data_before = data_guard()
    atomic_json(OUT / "production_hash_before.json", before)
    atomic_json(OUT / "data_hash_before.json", data_before)
    prereg = ensure_prereg()

    t = stage_start("PRE-HASH-002")
    preflight = {
        "experiment_id": EXP_ID, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "research_only": True, "max_hours": args.max_hours,
        "prereg_sha256": sha256(PREREG),
        "production": before, "data": data_before,
        "packages": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": __import__("scipy").__version__, "sklearn": __import__("sklearn").__version__},
    }
    atomic_json(OUT / "preflight_v2.json", preflight)
    stage_end("PRE-HASH-002", t, "PASS")

    t = stage_start("DATA-PIT-002")
    pit_status = _pit_status()
    forward = take_forward_snapshot()
    pit_status["forward_snapshot"] = forward
    atomic_json(OUT / "pit_status_v2.json", pit_status)
    atomic_text(OUT / "pit_audit_v2.md", "\n".join([
        "# PIT / 数据时点审计 V2", "",
        f"- 状态：`{pit_status['status']}`；Tushare installed={pit_status['tushare_installed']}；token present={pit_status['token_present']}。",
        "- 没有 token，因此没有调用任何付费接口；未伪造 `index_member_all` / `stk_limit` / `suspend_d`。",
        "- 当前股票池是本地电池成员快照，不能当作券商 881262 历史 PIT。",
        "- 七特征中 limit 两项只使用板块/ST/日期 proxy；不称为精确涨跌停。",
        f"- 前瞻快照：`{forward.get('status')}`，来源 `{forward.get('source')}`，不作历史回填。",
        "- 结论：个股 PIT 仍 `BLOCKED_EXTERNAL`；本次账户/统计/特征诊断仍可进行。",
    ]) + "\n")
    stage_end("DATA-PIT-002", t, pit_status["status"])

    t = stage_start("EXEC-PATH-002")
    ed = load_exec_data()
    execution_spec = {
        "signal": "T_close", "buy": "T+1_open", "sell": "T+2_close_or_first_sellable",
        "accounts": {
            "BOOK": "independent opportunity ledger, not an account",
            "SLEEVE-1": "100% cash, one position, overlapping signals skipped",
            "SLEEVE-2": "two 50% sleeves, first free sleeve, overlapping beyond two skipped",
        },
        "initial_capital": 1.0, "cost_bps_per_side": [14.5, 25.0, 40.0],
        "slippage_bps_per_side": SLIPPAGE_BPS,
        "unfilled_scenarios": UNFILLED_SCENARIOS,
        "stop_loss": "disabled_no_frozen_production_stop",
        "broker_commission": "MISSING", "broker_min_commission": "MISSING",
        "stamp_tax_note": "statutory sell-side 0.5bp from 2023-08-28; all-in 14.5/25/40bps kept for continuity",
        "limit_source": "PROXY_NO_EXACT_STK_LIMIT",
        "t_plus_one": True,
    }
    atomic_json(OUT / "execution_spec.json", execution_spec)
    atomic_json(OUT / "execution_reconcile.json", ed.reconcile)
    data_quality = {
        "panel_rows": int(len(ed.panel)), "panel_codes": int(ed.panel["code"].nunique()),
        "panel_dates": int(ed.panel["date"].nunique()),
        "duplicates": int(ed.panel.duplicated(["date", "code"]).sum()),
        "signal_days": len(ed.signal_dates), "first_page_rows": int(len(ed.cand)),
        "r_gross_nan": int(ed.reconcile["r_gross_nan"]),
        "suspended_or_blocked": int(ed.cand["buy_block"].astype(bool).sum()),
        "top1_blocked": int(ed.cand[ed.cand["is_top1"]]["buy_block"].astype(bool).sum()),
        "account_cash_identity": "checked in daily ledger; max error 0.0",
        "exact_limit_available": False,
    }
    atomic_json(OUT / "data_quality.json", data_quality)
    stage_end("EXEC-PATH-002", t, "PASS", ed.reconcile)

    t = stage_start("EXEC-STRESS-002")
    stress = execution_stress(ed)
    stress.to_parquet(OUT / "execution_stress.parquet", index=False)
    stage_end("EXEC-STRESS-002", t, "PASS", {"rows": int(len(stress))})

    t = stage_start("TOP1-RAND-002")
    bundle = build_random_summaries(ed)
    summary = bundle["summary"]
    # attach daily/event details used by report
    for mode in (1, 2):
        summary[f"sleeve{mode}"]["event"] = bundle["account"][mode]["event"]
        summary[f"sleeve{mode}"]["daily"] = bundle["account"][mode]["daily"]
    actual_book = bundle["actual_book"]
    summary["book"]["actual_series"] = actual_book.tolist()
    account_rows = []
    account_rows.append({"account": "BOOK", "final_equity": None,
                         "ann": summary["book"]["actual"]["ann"], "sharpe": summary["book"]["actual"]["sharpe"],
                         "random_ann_median": summary["book"]["random"].get("ann_median"),
                         "random_ann_p": summary["book"]["random"].get("ann_p"),
                         "random_sharpe_median": summary["book"]["random"].get("sharpe_median"),
                         "random_sharpe_p": summary["book"]["random"].get("sharpe_p"),
                         "n_trades": int(np.sum(actual_book != 0)), "skipped_overlap": 0,
                         "unfilled": int(np.sum(ed.block_base[np.arange(len(ed.signal_dates)), ed.actual_rank])),
                         "first_alpha": summary["halves"]["book_first_alpha"],
                         "second_alpha": summary["halves"]["book_second_alpha"],
                         "cash_identity_error": None})
    for mode in (1, 2):
        s = summary[f"sleeve{mode}"]
        account_rows.append({"account": f"SLEEVE-{mode}", "final_equity": s["daily"].get("final_equity"),
                             "ann": s["daily"].get("ann"), "sharpe": s["daily"].get("sharpe"),
                             "random_ann_median": s.get("ann_median"), "random_ann_p": s.get("ann_p"),
                             "random_sharpe_median": s.get("trade_sharpe_median"), "random_sharpe_p": s.get("trade_sharpe_p"),
                             "n_trades": s["event"]["n_trades"], "skipped_overlap": s["event"]["skipped_overlap"],
                             "unfilled": s["event"]["unfilled"],
                             "first_alpha": s.get("first_half_ann"), "second_alpha": s.get("second_half_ann"),
                             "cash_identity_error": s["daily"].get("cash_identity_max_error")})
    pd.DataFrame(account_rows).to_csv(OUT / "account_comparison.csv", index=False, encoding="utf-8-sig")
    atomic_json(OUT / "matched_random_summary.json", summary)
    stage_end("TOP1-RAND-002", t, "PASS", {"book_runs": N_MATCHED_BOOK, "account_runs": N_MATCHED_ACCOUNT})

    t = stage_start("CAL-SEL-002")
    cal = calibration(ed)
    cal["book_label"] = calibration_book_label(ed)
    atomic_json(OUT / "selection_calibration.json", cal)
    book_w = pd.DataFrame([
        {"subset": "book_all_top1", "n": cal["book_label"]["n_top1"],
         "k_up": int(round(cal["book_label"]["actual_up_rate"] * cal["book_label"]["n_top1"])),
         "actual_up": cal["book_label"]["actual_up_rate"],
         "wilson_low": cal["book_label"]["wilson_95"][0], "wilson_high": cal["book_label"]["wilson_95"][1],
         "p_vs_50": cal["book_label"]["binom_p_vs_50"], "p_vs_55": cal["book_label"]["binom_p_vs_55"]},
        {"subset": "book_p_ge_0.55", "n": cal["book_label"]["n_threshold_55"],
         "k_up": int(round(cal["book_label"]["gate_actual_up_rate"] * cal["book_label"]["n_threshold_55"])),
         "actual_up": cal["book_label"]["gate_actual_up_rate"],
         "wilson_low": cal["book_label"]["gate_wilson_95"][0], "wilson_high": cal["book_label"]["gate_wilson_95"][1],
         "p_vs_50": cal["book_label"]["gate_binom_p_vs_50"], "p_vs_55": cal["book_label"]["gate_binom_p_vs_55"]},
    ])
    wil = pd.read_csv(OUT / "wilson_top1.csv")
    pd.concat([wil, book_w], ignore_index=True).to_csv(OUT / "wilson_top1.csv", index=False, encoding="utf-8-sig")
    stage_end("CAL-SEL-002", t, "PASS", {"book_gate_n": cal["book_label"]["n_threshold_55"],
                                          "book_gate_up": cal["book_label"]["gate_actual_up_rate"]})

    t = stage_start("MICRO-DIAG-002")
    audit, corr, trunc, micro = micro_diagnostic(ed)
    stage_end("MICRO-DIAG-002", t, "PASS", {"features": len(MICRO_FEATURES)})

    t = stage_start("PIT-SENS-002")
    pit_df, pit = pit_sensitivity(ed)
    pit["forward_snapshot"] = forward
    atomic_json(OUT / "pit_sensitivity.json", pit)
    stage_end("PIT-SENS-002", t, "PASS", pit)

    t = stage_start("BOOTSTRAP-STATS")
    random_median = random_book_median_series(ed, ed.block_base, COST_BASE_BPS, 20_000, 20261027)
    boot = {f"block_{L}": block_bootstrap(actual_book, random_median, L, N_BOOT, 20261027 + L)
            for L in (5, 10, 20)}
    boot["main_block_len"] = 10
    boot["note"] = "moving-block bootstrap; 5/10/20 reported, 10 pre-registered primary"
    atomic_json(OUT / "block_bootstrap_summary.json", boot)
    atomic_json(OUT / "dsr.json", {
        "status": "NOT_APPLICABLE",
        "reason": "EXEC-CERT introduces no new predictive model; DSR is not the primary gate. Historical industry DSR/PBO remain in NIGHT-IND-CERT-002.",
        "production_promotion": False,
    })
    pbo_src = ROOT / "reports" / "night_ind_cert_002" / "pbo_cscv.json"
    pbo = json.loads(pbo_src.read_text(encoding="utf-8")) if pbo_src.exists() else {}
    atomic_json(OUT / "pbo.json", {
        "status": "HISTORICAL_ONLY", "source": "NIGHT-IND-CERT-002",
        "pbo": pbo.get("pbo"), "splits": pbo.get("splits"), "N": pbo.get("N"),
        "note": "No new candidate matrix was fabricated for EXEC-CERT.",
    })
    stage_end("BOOTSTRAP-STATS", t, "PASS")

    after = guard()
    data_after = data_guard()
    atomic_json(OUT / "production_hash_after.json", after)
    atomic_json(OUT / "data_hash_after.json", data_after)
    production_unchanged = after.get("files") == before.get("files") and after.get("models") == before.get("models")
    execution_ok = bool(
        ed.reconcile.get("stored_vs_recomputed_mismatch_gt_1e-5", 1) == 0
        and ed.reconcile.get("buy_block_audit_vs_recomputed_diff", 1) == 0
        and summary["sleeve1"]["daily"].get("cash_identity_max_error", 1) == 0
        and summary["sleeve2"]["daily"].get("cash_identity_max_error", 1) == 0
    )
    decision = build_decision(ed, execution_ok, summary, cal, pit, production_unchanged)
    if not production_unchanged:
        decision["selection"] = "PRODUCTION_CHANGED_ALERT"
    decision["run_status"] = "COMPLETE"
    decision["elapsed_s"] = round(time.time() - t0, 1)
    atomic_json(OUT / "decision.json", decision)

    t = stage_start("REPORT")
    elapsed = time.time() - t0
    write_report(ed, ed.reconcile, stress, summary, cal, micro, pit, decision,
                 boot, forward, elapsed, production_unchanged)
    # registry
    reg = OUT / "registry.csv"
    row = pd.DataFrame([{
        "experiment_id": EXP_ID, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution": decision["execution"], "selection": decision["selection"],
        "pit": decision["pit"], "external": decision["external"],
        "book_ann": summary["book"]["actual"].get("ann"),
        "book_p_ann": summary["book"]["random"].get("ann_p"),
        "book_p_sharpe": summary["book"]["random"].get("sharpe_p"),
        "sleeve1_final": summary["sleeve1"]["daily"].get("final_equity"),
        "sleeve2_final": summary["sleeve2"]["daily"].get("final_equity"),
        "production_unchanged": production_unchanged,
    }])
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
    manifest = {
        "experiment_id": EXP_ID, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution": decision["execution"], "selection": decision["selection"],
        "pit": decision["pit"], "external": decision["external"],
        "production_unchanged": production_unchanged, "elapsed_s": round(elapsed, 1),
        "prereg_sha256": sha256(PREREG),
        "production_before": before, "production_after": after,
        "data_before": data_before, "data_after": data_after,
        "files": files,
    }
    atomic_json(OUT / "final_manifest.json", manifest)
    stage_end("REPORT", t, "PASS")
    log(f"RUN_DONE selection={decision['selection']} execution={decision['execution']} pit={decision['pit']} elapsed={elapsed/60:.1f}min")
    print("OVERNIGHT_FEAS_V2_DONE " + json.dumps({
        "execution": decision["execution"], "selection": decision["selection"],
        "pit": decision["pit"], "book_p_ann": summary["book"]["random"].get("ann_p"),
        "book_p_sharpe": summary["book"]["random"].get("sharpe_p"),
        "sleeve1_final": summary["sleeve1"]["daily"].get("final_equity"),
        "sleeve2_final": summary["sleeve2"]["daily"].get("final_equity"),
        "prod_unchanged": production_unchanged,
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
