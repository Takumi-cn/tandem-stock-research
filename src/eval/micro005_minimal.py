"""MICRO-005-MINIMAL: does 5-minute information add anything to daily features?

Pre-registered, deliberately small experiment.  One fixed model family
(LightGBM, no tuning), one target (`label__page_resid_T1_T2`), one comparison:

    A = feat__ daily + market whitelist          (v2 contract, 33 features)
    B = A + feat__ intraday / intraday5 / rel_*  (v2 contract, +58 features)

Both arms see exactly the same rows, folds, seeds and hyper-parameters.  The
2021-2026 panel is a *diagnostic* sample: it has already influenced earlier
TANDEM decisions, so nothing here can certify or promote a model.  A positive
result only earns the right to train the compact encoder (arm C) as a shadow;
a negative result stops the search.

Gates (frozen in prereg.json before the run):
  G1 primary   mean daily rank-IC(B) - IC(A) > 0 and date-block bootstrap
               one-sided p < 0.05
  G2 secondary mean top-1 page alpha(B) - (A) > 0 and bootstrap p < 0.10
  G3 null      mean OOS IC(B) exceeds the within-date label-permutation
               refit null at p < 0.05
Verdicts:
  G1 fail                      -> NO_INCREMENTAL_INTRADAY_INFO (stop, no encoder)
  G1 pass, G2/G3 fail          -> WEAK_INCREMENTAL_ONLY
  G1+G2+G3 pass                -> INTRADAY_CANDIDATE_SHADOW_ONLY

Run:
  .venv\\Scripts\\python.exe -m src.eval.micro005_minimal
  .venv\\Scripts\\python.exe -m src.eval.micro005_minimal --quick
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.feature_contract_v2 import (  # noqa: E402
    DAILY_ARM, FULL_ARM, INTRADAY_ARM, TARGET, LABELS, FORBIDDEN, VERSION as CONTRACT_VERSION,
    metadata as contract_metadata, select_X, validate_frame, write_contract,
)
from src.utils.config import ROOT  # noqa: E402

EXP = "TANDEM-FWD-MICRO-005-MINIMAL"
OUT = ROOT / "reports" / "micro005_minimal"
DATA = ROOT / "data" / "research" / "micro005_minimal"
LOGS = OUT / "logs"
COST = 0.00145
PROD_FILES = [
    "reports/latest.json", "reports/predictions_ledger.csv", "config/targets.yaml",
    "config/l1_best.json", "reports/ens_fixed_weights.json", "run_daily.pyw", "run_gui.pyw",
    "src/report/daily.py", "src/report/run_full.py", "src/report/short_pick.py",
]
DATA_FILES = ["data/parquet/panel.parquet", "data/parquet/cons_battery.parquet"]

LGB_PARAMS = dict(objective="regression", n_estimators=300, learning_rate=0.03, num_leaves=15,
                  min_child_samples=200, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                  reg_lambda=5.0, n_jobs=4, random_state=0, deterministic=True, verbose=-1)
SEEDS = (0, 1, 2)
TRAIN_MIN_DAYS = 400
TEST_BLOCK = 40
PURGE_DAYS = 5


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


def guard(files: list[str]) -> dict:
    out = {}
    for rel in files:
        p = ROOT / rel
        out[rel] = {"exists": p.exists(), "sha256": sha256(p) if p.exists() else None}
    md = ROOT / "models"
    if md.exists():
        out["_models"] = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(md.rglob("*")) if p.is_file()}
    return out


def ensure_prereg(quick: bool) -> dict:
    p = OUT / "prereg.json"
    obj = {
        "experiment_id": EXP, "created_at": "2026-10-27", "status": "PRE_REGISTERED",
        "contract_version": CONTRACT_VERSION,
        "hypothesis": ("Five-minute intraday structure available at the T close contains incremental "
                       "cross-sectional information about a stock's T+1-open to T+2-close net return "
                       "relative to the other stocks on the same battery page, after the existing "
                       "TANDEM daily features are accounted for."),
        "target": TARGET,
        "label_definition": "net exec (T+1 open -> T+2 close, 14.5bps/side) minus same-page mean net exec",
        "arms": {"A": DAILY_ARM, "B": FULL_ARM},
        "model": {"family": "lightgbm.LGBMRegressor", "params": LGB_PARAMS, "seeds": list(SEEDS),
                  "no_tuning": True, "no_early_stopping": True},
        "folds": {"train_min_days": TRAIN_MIN_DAYS, "test_block_days": TEST_BLOCK,
                  "purge_days": PURGE_DAYS, "scheme": "expanding window, chronological, purged"},
        "gates": {
            "G1_primary": "mean_daily_rank_ic_B - mean_daily_rank_ic_A > 0 and date-block bootstrap one-sided p < 0.05",
            "G2_secondary": "mean_top1_page_alpha_B - mean_top1_page_alpha_A > 0 and bootstrap one-sided p < 0.10",
            "G3_null": "mean OOS IC(B) exceeds within-date label-permutation refit null at p < 0.05",
            "G4_temporal": "delta IC is positive in at least 3 of 4 chronological OOS blocks",
        },
        "verdicts": {
            "NO_INCREMENTAL_INTRADAY_INFO": "G1 fails: stop, do not train an encoder",
            "WEAK_INCREMENTAL_ONLY": "G1 passes but G2/G3 fail: no encoder, no selection claim",
            "INTRADAY_CANDIDATE_SHADOW_ONLY": "all gates pass: encoder may be trained, shadow only",
        },
        "sample_note": ("2021-10..2026-10 panel is HISTORICAL_DIAGNOSTIC_ONLY: it has already influenced "
                        "earlier TANDEM decisions.  No result here may promote a model or touch production."),
        "nulls": {"matched_random_draws_per_date": 2000,
                  "permutation_refit_iterations": 30,
                  "bootstrap_iterations": 5000},
        "production_policy": {"latest_json": "READ_ONLY", "threshold_55": "READ_ONLY",
                              "ensemble": "READ_ONLY", "industry": "CLOSED"},
    }
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old.get("model") != obj["model"] or old.get("arms") != obj["arms"]:
            raise SystemExit("prereg.json exists with different arms/model; refusing to overwrite")
        return old
    atomic_json(p, obj)
    return obj


def build_frame() -> tuple[pd.DataFrame, dict]:
    page = pd.read_parquet(DATA / "page_dataset.parquet")
    intra = pd.read_parquet(DATA / "intraday_features.parquet")
    page["date"] = pd.to_datetime(page["date"]); page["code"] = page["code"].astype(str)
    intra["date"] = pd.to_datetime(intra["date"]); intra["code"] = intra["code"].astype(str)
    rename = {c: f"feat__{c}" for c in
              ["ret_5", "ret_10", "ret_20", "ret_60", "exret_5", "exret_20", "vol_20", "vol_ratio",
               "atr_norm", "vol_spike", "turn_pctile", "pv_corr_10", "rsi_14", "macd_hist_norm",
               "boll_pos", "dd_20", "new_high_20", "consec_up", "consec_dn", "upper_shadow",
               "lower_shadow", "body_ratio", "gap_open", "close_pos", "xrank_ret5", "xrank_vol",
               "xrank_turn", "xrank_macd", "pct_chg"]}
    missing_daily = [c for c in rename if c not in page.columns]
    page = page.rename(columns=rename)
    df = page.merge(intra, on=["date", "code"], how="left", suffixes=("", "_intra"))
    # page-relative versions of a few intraday features (T-close information)
    rel_src = ["intra_ret", "intra_late_vol_share", "intra_vol_z_mean",
               "intra5_ret_mean", "intra5_close_pos_mean"]
    for c in rel_src:
        col = f"feat__{c}"
        if col in df.columns:
            df[f"feat__rel_{c}"] = df[col] - df.groupby("date")[col].transform("mean")
    info = {"missing_daily_renames": missing_daily,
            "rows": len(df),
            "rows_with_intraday": int(df["feat__intra_ret"].notna().sum()) if "feat__intra_ret" in df else 0,
            "dates": int(df["date"].nunique())}
    return df, info


def complete_case(df: pd.DataFrame) -> pd.DataFrame:
    need = [c for c in FULL_ARM if c in df.columns]
    miss = [c for c in FULL_ARM if c not in df.columns]
    if miss:
        raise SystemExit(f"contract columns missing: {miss[:5]} ... ({len(miss)})")
    sub = df.dropna(subset=need + [TARGET]).copy()
    sub = sub[~sub["entry_blocked"].astype(bool)]
    return sub


def folds(dates: pd.DatetimeIndex, quick: bool) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    i = TRAIN_MIN_DAYS
    step = TEST_BLOCK * (3 if quick else 1)
    while i < len(dates):
        out.append((dates[i], dates[min(i + TEST_BLOCK - 1, len(dates) - 1)]))
        i += step
    return out


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, arm: str, seed: int) -> np.ndarray:
    import lightgbm as lgb
    Xtr = select_X(train, arm).to_numpy(dtype=np.float32)
    Xte = select_X(test, arm).to_numpy(dtype=np.float32)
    y = train[TARGET].to_numpy(dtype=np.float64)
    params = dict(LGB_PARAMS); params["random_state"] = seed
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr, y)
    return m.predict(Xte)


def run_arm(df: pd.DataFrame, arm: str, fold_list, seeds=SEEDS) -> pd.DataFrame:
    preds = []
    all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    for (te_start, te_end) in fold_list:
        # purge in *trading* days: the label of a T signal is realised at T+2
        j = int(all_dates.get_loc(te_start))
        cutoff = all_dates[max(0, j - PURGE_DAYS)]
        tr = df[df["date"] <= cutoff]
        te = df[(df["date"] >= te_start) & (df["date"] <= te_end)]
        if len(tr) < 500 or len(te) == 0:
            continue
        p = np.zeros(len(te))
        for s in seeds:
            p += fit_predict(tr, te, arm, s)
        p /= len(seeds)
        out = te[["date", "code", "rank_in_fp", TARGET, "label__exec_T1_T2"]].copy()
        out["pred"] = p
        out["arm"] = arm
        out["fold"] = f"{te_start.date()}"
        preds.append(out)
    return pd.concat(preds, ignore_index=True)


def daily_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in pred.groupby("date"):
        if len(g) < 8:
            continue
        ic = stats.spearmanr(g["pred"], g[TARGET]).statistic
        top1 = g.loc[g["pred"].idxmax()]
        top3 = g.nlargest(3, "pred")[TARGET].mean()
        q = pd.qcut(g["pred"].rank(method="first"), 4, labels=False)
        spread = g[TARGET][q == 3].mean() - g[TARGET][q == 0].mean()
        rows.append({"date": d, "n": len(g), "ic": ic, "top1_alpha": top1[TARGET],
                     "top1_code": top1["code"], "top3_alpha": top3, "q4_minus_q1": spread})
    return pd.DataFrame(rows)


def boot_p(diff: np.ndarray, n_boot: int = 5000, block: int = 5, seed: int = 42,
           alternative: str = "greater") -> dict:
    """Date-block bootstrap of the mean of `diff`; one-sided p under H0 mean<=0."""
    diff = np.asarray([x for x in diff if np.isfinite(x)], dtype=float)
    if len(diff) == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    n = len(diff)
    nblocks = int(np.ceil(n / block))
    starts = np.arange(0, n, block)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(starts, nblocks, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in pick])
        means[b] = diff[idx].mean()
    mean = float(diff.mean())
    p = float((means <= 0).mean()) if alternative == "greater" else float((means >= 0).mean())
    return {"n": int(n), "mean": mean,
            "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "p_one_sided": p, "block": block, "n_boot": n_boot}


def matched_random(pred: pd.DataFrame, n_draw: int = 2000, seed: int = 11) -> dict:
    """Null: pick a uniformly random page name each date instead of the model's top1."""
    rng = np.random.default_rng(seed)
    per_date = [g[TARGET].to_numpy() for _, g in pred.groupby("date")]
    if not per_date:
        return {}
    draws = np.empty(n_draw)
    for b in range(n_draw):
        draws[b] = np.mean([arr[rng.integers(len(arr))] for arr in per_date])
    actual = float(pred.groupby("date").apply(lambda g: g.loc[g["pred"].idxmax(), TARGET]).mean())
    return {"actual_top1_mean": actual, "random_mean_median": float(np.median(draws)),
            "p_model_le_random": float((draws >= actual).mean()), "n_draw": n_draw}


def permutation_refit(df: pd.DataFrame, fold_list, arm: str, n_iter: int, seed: int = 23,
                      observed_ic: float | None = None) -> dict:
    """Full-pipeline null: shuffle the target within each date, refit, recompute OOS IC."""
    rng = np.random.default_rng(seed)
    if observed_ic is None:
        r0 = run_arm(df, arm, fold_list, seeds=(0,))
        observed_ic = float(daily_metrics(r0)["ic"].mean())
    null = []
    base = df.copy()
    for it in range(n_iter):
        shuffled = base.copy()
        shuffled[TARGET] = shuffled.groupby("date")[TARGET].transform(
            lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
        r = run_arm(shuffled, arm, fold_list, seeds=(0,))
        null.append(daily_metrics(r)["ic"].mean())
        if (it + 1) % 5 == 0:
            log(f"    permutation {it+1}/{n_iter} null_ic_mean={np.mean(null):+.5f}")
    null = np.asarray(null)
    obs = float(observed_ic)
    return {"observed_ic": obs, "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, .95)),
            "p_value": float((null >= obs).mean()), "n_iter": int(n_iter)}


def rule_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-training page rules: the T-day top gainer vs the weakest name on the page.

    The 1,210-day page audit found a stable *within-page reversal* (rank IC +0.037),
    so "buy the weakest name on the page" is a real baseline any intraday model
    has to beat; otherwise a 5-minute encoder could look useful merely by
    rediscovering reversal.
    """
    rows = []
    for d, g in df.groupby("date"):
        g = g.sort_values("rank_in_fp")
        rows.append({"date": d, "pct_top1": float(g.iloc[0][TARGET]),
                     "pct_bottom1": float(g.iloc[-1][TARGET]),
                     "pct_mid": float(g.iloc[len(g) // 2][TARGET])})
    return pd.DataFrame(rows)


def single_feature_ic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in INTRADAY_ARM:
        if c not in df.columns:
            continue
        ics = []
        for _, g in df.groupby("date"):
            if len(g) < 10 or g[c].nunique() < 3:
                continue
            v = stats.spearmanr(g[c], g[TARGET], nan_policy="omit").statistic
            if np.isfinite(v):
                ics.append(v)
        if ics:
            rows.append({"feature": c, "mean_ic": float(np.mean(ics)), "ic_std": float(np.std(ics)),
                         "t": float(np.mean(ics) / (np.std(ics) / np.sqrt(len(ics)) + 1e-12)),
                         "n_days": len(ics)})
    return pd.DataFrame(rows).sort_values("mean_ic", ascending=False)


def importance(df: pd.DataFrame, fold_list, arm: str) -> pd.DataFrame:
    import lightgbm as lgb
    te_start, te_end = fold_list[-1]
    tr = df[df["date"] < te_start - pd.Timedelta(days=PURGE_DAYS)]
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(select_X(tr, arm).to_numpy(dtype=np.float32), tr[TARGET].to_numpy())
    return pd.DataFrame({"feature": select_X(df, arm).columns, "gain": m.booster_.feature_importance("gain")}
                        ).sort_values("gain", ascending=False)


def report(dec: dict, a_daily: pd.DataFrame, b_daily: pd.DataFrame, delta: pd.DataFrame,
           null: dict, rand: dict, imp: pd.DataFrame, single: pd.DataFrame, rules: pd.DataFrame,
           info: dict, gates: dict, elapsed: float, frame_note: dict) -> None:
    def pct(x, d=3):
        return "—" if x is None or not np.isfinite(x) else f"{x*100:+.{d}f}%"

    def num(x, d=4):
        return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"

    lines = [
        "# TANDEM-FWD-MICRO-005 · Minimal Increment Test", "",
        f"- 生成 {time.strftime('%Y-%m-%d %H:%M:%S')}｜耗时 {elapsed/60:.1f} 分钟｜契约 `{CONTRACT_VERSION}`",
        f"- 结论：**`{dec['verdict']}`**；训练状态 `{dec['model_training']}`；生产只读。",
        f"- 样本：{frame_note['n_rows']:,} 个 page 行、{frame_note['n_dates']} 个交易日"
        f"（{frame_note['date_min']} ~ {frame_note['date_max']}），OOS {frame_note['n_oos_dates']} 日。",
        "- **历史诊断，不是新 OOS**：2021–2026 已被本项目反复查看，任何结果都不能用于晋升。",
        "", "## 预注册门槛与结果", "",
        "| 门槛 | 规则 | 结果 | 通过 |", "|---|---|---|---|",
        f"| G1 主检验 | IC(B)−IC(A)>0 且 block-bootstrap 单侧 p<0.05 | ΔIC {num(dec['delta_ic'])}, "
        f"p={num(dec['delta_ic_p'],4)} | {'✅' if gates['G1'] else '❌'} |",
        f"| G2 次检验 | Top1 page alpha(B)−(A)>0 且 p<0.10 | ΔTop1 {pct(dec['delta_top1'])}, "
        f"p={num(dec['delta_top1_p'],4)} | {'✅' if gates['G2'] else '❌'} |",
        f"| G3 零分布 | IC(B) 超过同日内标签置换重训零分布 p<0.05 | p={num(dec['perm_p'],4)} | "
        f"{'✅' if gates['G3'] else '❌'} |",
        f"| G4 分段 | 4 个时序块中至少 3 块 ΔIC>0 | "
        f"{' / '.join(f'{x:+.4f}' for x in gates.get('G4_blocks', []))} | "
        f"{'✅' if gates.get('G4') else '❌'} |",
        "", "## 两个 arm（同一行、同一折、同一种子、同一超参）", "",
        "| 指标 | A 仅日线 | B 日线+日内 | B−A |", "|---|---:|---:|---:|",
        f"| 日均 rank IC | {num(a_daily['ic'].mean(),5)} | {num(b_daily['ic'].mean(),5)} | "
        f"{num(dec['delta_ic'],5)} |",
        f"| IC 标准差 | {num(a_daily['ic'].std(),4)} | {num(b_daily['ic'].std(),4)} | — |",
        f"| ICIR（IC/std） | {num(a_daily['ic'].mean()/a_daily['ic'].std(),3)} | "
        f"{num(b_daily['ic'].mean()/b_daily['ic'].std(),3)} | — |",
        f"| Top1 page alpha 均值 | {pct(a_daily['top1_alpha'].mean())} | {pct(b_daily['top1_alpha'].mean())} | "
        f"{pct(dec['delta_top1'])} |",
        f"| Top1 alpha>0 天数占比 | {a_daily['top1_alpha'].gt(0).mean():.1%} | "
        f"{b_daily['top1_alpha'].gt(0).mean():.1%} | — |",
        f"| Top3 page alpha 均值 | {pct(a_daily['top3_alpha'].mean())} | {pct(b_daily['top3_alpha'].mean())} | — |",
        f"| Q4−Q1 组内价差 | {pct(a_daily['q4_minus_q1'].mean())} | {pct(b_daily['q4_minus_q1'].mean())} | — |",
        "", "## 零训练规则基线（同样本、同标签）", "",
        "| 规则 | Top1 page alpha 均值 | >0 天数占比 |", "|---|---:|---:|",
        f"| 当日涨幅第 1（现规则） | {pct(rules['pct_top1'].mean())} | {rules['pct_top1'].gt(0).mean():.1%} |",
        f"| 当日涨幅第 20（页内反转） | {pct(rules['pct_bottom1'].mean())} | {rules['pct_bottom1'].gt(0).mean():.1%} |",
        f"| 页内中位 | {pct(rules['pct_mid'].mean())} | {rules['pct_mid'].gt(0).mean():.1%} |",
        "", "## 配对差与零分布", "",
        f"- ΔIC 按日：均值 {num(dec['delta_ic'],5)}，block-bootstrap 95% CI "
        f"[{num(dec['delta_ic_ci'][0],5)}, {num(dec['delta_ic_ci'][1],5)}]，单侧 p={num(dec['delta_ic_p'],4)}（块长 5 日）。",
        f"- ΔTop1 按日：均值 {pct(dec['delta_top1'])}，95% CI [{pct(dec['delta_top1_ci'][0])}, "
        f"{pct(dec['delta_top1_ci'][1])}]，单侧 p={num(dec['delta_top1_p'],4)}。",
        f"- 同日同页随机选一只：随机 Top1 均值中位 {pct(rand.get('random_mean_median'))}；"
        f"B 的 Top1 均值 {pct(rand.get('actual_top1_mean'))}，p={num(rand.get('p_model_le_random'),4)}。",
        f"- 重训置换零分布（{null.get('n_iter')} 次，B）：观测 IC {num(null.get('observed_ic'),5)}，"
        f"null 均值 {num(null.get('null_mean'),5)}、95 分位 {num(null.get('null_p95'),5)}，"
        f"p={num(null.get('p_value'),4)}。",
        "", "## 日内特征单独的信息量（Top 12 / Bottom 5）", "",
        "| 特征 | 日均 IC | t | 天数 |", "|---|---:|---:|---:|",
    ]
    for _, r in single.head(12).iterrows():
        lines.append(f"| `{r['feature']}` | {num(r['mean_ic'],5)} | {num(r['t'],2)} | {int(r['n_days'])} |")
    lines.append("| … | | | |")
    for _, r in single.tail(5).iterrows():
        lines.append(f"| `{r['feature']}` | {num(r['mean_ic'],5)} | {num(r['t'],2)} | {int(r['n_days'])} |")
    # redundancy audit (from the intraday feature build)
    red = []
    aud = OUT / "intraday_feature_audit.json"
    if aud.exists():
        try:
            aj = json.loads(aud.read_text(encoding="utf-8"))
            ma = aj.get("max_abs_corr_with_daily", {}) or {}
            red = sorted(ma.items(), key=lambda kv: -abs(kv[1] or 0))[:8]
        except Exception:  # noqa: BLE001
            red = []
    lines += ["", "## 与现有日线特征的冗余（日内特征，Top 8 最大相关）", "",
              "| 日内特征 | 与日线最大 |corr| |", "|---|---:|"]
    for name, v in red:
        lines.append(f"| `{name}` | {num(v,3)} |")
    lines += ["", "- 冗余是**报告**出来的，不是隐藏的：>0.95 的特征会从 arm 中剔除"
              "（`feat__intra_close_pos` 与日线 `close_pos` 相关 0.998，已剔除）。",
              "- 量能类特征（`intra_vol_z_*`）与换手分位相关约 0.8，说明其“新信息”主要是日内节奏而非量级；"
              "真正新增的是日内形态（收盘位置、VWAP 偏离、尾盘 30 分钟、收益自相关）。"]
    lines += ["", "## 特征重要性（B，最后一折 gain Top 15）", "", "| 特征 | gain |", "|---|---:|"]
    for _, r in imp.head(15).iterrows():
        lines.append(f"| `{r['feature']}` | {r['gain']:.1f} |")
    # trial family inventory: this is not the first hypothesis in this family
    fam = []
    reg_path = ROOT / "reports" / "experiment_registry.csv"
    if reg_path.exists():
        try:
            rr = pd.read_csv(reg_path)
            if "exp_id" in rr.columns:
                fam = rr[["exp_id", "status"]].tail(8).values.tolist()
        except Exception:  # noqa: BLE001
            fam = []
    lines += ["", "## 研究试验族与多重检验", "",
              f"- 这是个股选择族的第 {max(1, len(fam))} 个登记假设；此前的失败试验不会被重置："]
    for e, st in fam:
        lines.append(f"  - `{e}` → {st}")
    lines += ["- 因此本实验的门槛不是“p<0.05 就晋升”，而是 G1–G4 全过才拿到 **shadow 资格**；"
              "即便如此也不能进入生产，必须等 ≥250 个真前瞻交易日。",
              "", "## 判读", "",
        f"- {dec['interpretation']}",
        "- 只有 G1 通过才允许训练紧凑 encoder；G1 不通过就停止这条线，不在同一段历史上换模型重试。",
        "- 即使三个门槛都通过，也只得到 shadow 资格：需要 ≥250 个真前瞻交易日与 25bps 压力，才谈晋升。",
        "", "## 不能宣称", "",
        "- 不能宣称任何可交易优势：本实验只回答“5 分钟信息是否比日线多一点点”，不是账户收益。",
        "- 不能把这 2021–2026 结果当作新 OOS、不能改 55% 门槛、不能改生产集成或仓位。",
        "- 若 G1 通过，也不代表 encoder 会有用；encoder 必须独立预注册、独立验证。",
        "", "## 数据与工程", "",
        (lambda fm: f"- 5m 覆盖：{fm.get('codes')} 只 / {fm.get('rows'):,} 根 / {fm.get('trading_days')} 日"
                   f"（{fm.get('first_day')} ~ {fm.get('last_day')}）；异常 bar 日 {fm.get('days_with_irregular_bar_count')} 个；"
                   f"与面板成交量比中位 {[(u.get('volume_ratio_median') if isinstance(u, dict) else None) for u in (fm.get('unit_checks') or [])][:1]}")
        if isinstance(info.get("five_min"), dict) and info["five_min"].get("rows") else "- 5m 覆盖：审计文件缺失",
        f"- 契约：`reports/micro005_minimal/feature_contract_v2.json`（feat__/label__ 命名 + available_at/source/version_hash）。",
        f"- 生产文件与模型树 before/after 哈希：{'一致' if dec['production_unchanged'] else '**不一致（需调查）**'}。",
    ]
    atomic_text(OUT / "report.md", "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    for d in (OUT, DATA, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    log(f"RUN_START {EXP} quick={args.quick}")
    prereg = ensure_prereg(args.quick)
    write_contract(OUT / "feature_contract_v2.json")
    before = guard(PROD_FILES + DATA_FILES)
    atomic_json(OUT / "hash_before.json", before)

    df, info = build_frame()
    df = complete_case(df)
    frame_note = {"n_rows": len(df), "n_dates": int(df["date"].nunique()),
                  "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date())}
    log(f"frame rows={len(df):,} dates={frame_note['n_dates']} "
        f"({frame_note['date_min']}..{frame_note['date_max']}) "
        f"intraday_rows={info['rows_with_intraday']:,}")
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    fl = folds(dates, args.quick)
    log(f"folds={len(fl)} first_test={fl[0][0].date()} last_test={fl[-1][1].date()}")

    a = run_arm(df, "daily", fl)
    b = run_arm(df, "daily_intraday", fl)
    a.to_parquet(DATA / "oos_pred_A.parquet", index=False)
    b.to_parquet(DATA / "oos_pred_B.parquet", index=False)
    da, db = daily_metrics(a), daily_metrics(b)
    da.to_csv(OUT / "daily_metrics_A.csv", index=False, encoding="utf-8-sig")
    db.to_csv(OUT / "daily_metrics_B.csv", index=False, encoding="utf-8-sig")
    m = da.merge(db, on="date", suffixes=("_A", "_B"))
    delta = pd.DataFrame({"date": m["date"], "delta_ic": m["ic_B"] - m["ic_A"],
                          "delta_top1": m["top1_alpha_B"] - m["top1_alpha_A"],
                          "delta_top3": m["top3_alpha_B"] - m["top3_alpha_A"]})
    delta.to_csv(OUT / "paired_daily_delta.csv", index=False, encoding="utf-8-sig")
    frame_note["n_oos_dates"] = int(len(m))

    bi = boot_p(delta["delta_ic"].to_numpy(), prereg["nulls"]["bootstrap_iterations"])
    bt = boot_p(delta["delta_top1"].to_numpy(), prereg["nulls"]["bootstrap_iterations"])
    rand = matched_random(b, prereg["nulls"]["matched_random_draws_per_date"])
    log(f"A ic={da['ic'].mean():+.5f} top1={da['top1_alpha'].mean():+.5f} | "
        f"B ic={db['ic'].mean():+.5f} top1={db['top1_alpha'].mean():+.5f} | "
        f"dIC={bi['mean']:+.5f} p={bi['p_one_sided']:.4f}")
    n_perm = 3 if args.quick else prereg["nulls"]["permutation_refit_iterations"]
    perm = permutation_refit(df, fl, "daily_intraday", n_perm, observed_ic=float(db["ic"].mean()))
    log(f"permutation null p={perm['p_value']:.4f} obs={perm['observed_ic']:+.5f}")
    imp = importance(df, fl, "daily_intraday")
    imp.to_csv(OUT / "importance_B.csv", index=False, encoding="utf-8-sig")
    single = single_feature_ic(df)
    single.to_csv(OUT / "single_feature_ic.csv", index=False, encoding="utf-8-sig")
    rules = rule_baselines(df)
    rules.to_csv(OUT / "rule_baselines.csv", index=False, encoding="utf-8-sig")

    gates = {"G1": bool(bi["mean"] > 0 and bi["p_one_sided"] < 0.05),
             "G2": bool(bt["mean"] > 0 and bt["p_one_sided"] < 0.10),
             "G3": bool(perm["p_value"] < 0.05)}
    blocks = np.array_split(np.arange(len(delta)), 4)
    gates["G4_blocks"] = [float(delta["delta_ic"].iloc[b].mean()) for b in blocks if len(b)]
    gates["G4"] = bool(sum(1 for x in gates["G4_blocks"] if x > 0) >= 3)
    b_top1 = float(db["top1_alpha"].mean())
    rule_beat = ("低于" if b_top1 < float(rules["pct_bottom1"].mean()) else "不低于")
    if not gates["G1"]:
        verdict, training = "NO_INCREMENTAL_INTRADAY_INFO", "DISABLED_GATE_FAILED"
        interp = ("G1 未通过：加入 5 分钟信息后，样本外排序 IC 没有可辨识的增量。"
                  "按预注册停止规则，不训练 encoder、不换第二个模型、不再在同一段历史上搜索；"
                  "若要继续，只能靠真实前瞻样本积累。"
                  f"同日零训练规则对照：B 的 Top1 均值 {b_top1*100:+.3f}%，{rule_beat}"
                  f"“页内最弱一只”的 {float(rules['pct_bottom1'].mean())*100:+.3f}%。")
    elif not (gates["G2"] and gates["G3"] and gates["G4"]):
        verdict, training = "WEAK_INCREMENTAL_ONLY", "DISABLED_GATE_FAILED"
        interp = ("G1 通过但 G2/G3/G4 未全过：5 分钟信息可能改善排序，但没有转化成 Top1 选择、"
                  "无法区别于重训零分布、或分段不稳定，因此不训练 encoder、不宣称选择优势。"
                  f"B 的 Top1 均值 {b_top1*100:+.3f}%，{rule_beat}页内反转规则 "
                  f"{float(rules['pct_bottom1'].mean())*100:+.3f}%。")
    else:
        verdict, training = "INTRADAY_CANDIDATE_SHADOW_ONLY", "ENCODER_SMOKE_ALLOWED"
        interp = ("四个门槛均通过：5 分钟信息在样本外有增量，但仍只是历史诊断。"
                  "只允许训练 ≤300k 参数的紧凑 encoder 作为 shadow，不得进入生产。"
                  f"B 的 Top1 均值 {b_top1*100:+.3f}%，{rule_beat}页内反转规则 "
                  f"{float(rules['pct_bottom1'].mean())*100:+.3f}%。")

    after = guard(PROD_FILES + DATA_FILES)
    prod_unchanged = {k: v for k, v in before.items() if k in PROD_FILES} == \
                     {k: v for k, v in after.items() if k in PROD_FILES} and \
                     before.get("_models") == after.get("_models")
    five_min = {}
    aud = OUT / "5m_data_audit.json"
    if aud.exists():
        five_min = json.loads(aud.read_text(encoding="utf-8"))
    dec = {
        "experiment_id": EXP, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": verdict, "model_training": training,
        "gates": gates, "delta_ic": bi["mean"], "delta_ic_p": bi["p_one_sided"],
        "delta_ic_ci": bi["ci95"], "delta_top1": bt["mean"], "delta_top1_p": bt["p_one_sided"],
        "delta_top1_ci": bt["ci95"], "perm_p": perm["p_value"],
        "ic_A": float(da["ic"].mean()), "ic_B": float(db["ic"].mean()),
        "top1_A": float(da["top1_alpha"].mean()), "top1_B": float(db["top1_alpha"].mean()),
        "n_oos_dates": frame_note["n_oos_dates"], "n_rows": len(df),
        "rule_top1_alpha": float(rules["pct_top1"].mean()),
        "rule_bottom1_alpha": float(rules["pct_bottom1"].mean()),
        "rule_mid_alpha": float(rules["pct_mid"].mean()),
        "production_unchanged": bool(prod_unchanged), "production_write": False,
        "sample_status": "HISTORICAL_DIAGNOSTIC_ONLY", "interpretation": interp,
    }
    atomic_json(OUT / "permutation_null.json", perm)
    atomic_json(OUT / "matched_random.json", rand)
    dec["encoder"] = "NOT_TRAINED_GATE_FAILED" if verdict != "INTRADAY_CANDIDATE_SHADOW_ONLY" else "ALLOWED_SMOKE_ONLY"
    dec["next_step"] = ("stop the 5m-feature search; keep accumulating true forward 5m + evidence; "
                        "any encoder needs a new preregistration and new forward samples")
    atomic_json(OUT / "decision.json", dec)
    report(dec, da, db, delta, perm, rand, imp, single, rules, {"five_min": five_min}, gates,
           time.time() - t0, frame_note)
    # registry
    reg = ROOT / "reports" / "experiment_registry.csv"
    # The project registry schema is exp_id/date/scope/status/conclusion; keep it.
    row = pd.DataFrame([{
        "exp_id": EXP, "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "scope": "MICRO-005 5m 增量检验（仅日线 vs 日线+日内）", "status": verdict,
        "conclusion": (f"icA={dec['ic_A']:.5f};icB={dec['ic_B']:.5f};dIC={dec['delta_ic']:+.5f}"
                       f"(p={dec['delta_ic_p']:.4f});dTop1={dec['delta_top1']:+.5f}"
                       f"(p={dec['delta_top1_p']:.4f});perm_p={dec['perm_p']:.4f};"
                       f"n_oos={dec['n_oos_dates']};rule_top1={dec['rule_top1_alpha']:+.5f};"
                       f"rule_bottom1={dec['rule_bottom1_alpha']:+.5f}")}])
    if reg.exists():
        old = pd.read_csv(reg)
        if "exp_id" in old.columns:
            old = old[old["exp_id"] != EXP]
            pd.concat([old, row], ignore_index=True)[old.columns.tolist()].to_csv(
                reg, index=False, encoding="utf-8-sig")
        else:
            row.to_csv(reg.with_name("experiment_registry_micro005.csv"), index=False, encoding="utf-8-sig")
    else:
        row.to_csv(reg, index=False, encoding="utf-8-sig")
    files = []
    for root in (OUT, DATA):
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name not in ("final_manifest.json",):
                files.append({"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size})
    atomic_json(OUT / "final_manifest.json", {
        "experiment_id": EXP, "created_at": dec["created_at"], "verdict": verdict,
        "training": training, "elapsed_s": round(time.time() - t0, 1),
        "prereg_sha256": sha256(OUT / "prereg.json"), "contract_sha256": sha256(OUT / "feature_contract_v2.json"),
        "production_unchanged": bool(prod_unchanged), "files": files})
    log(f"RUN_DONE verdict={verdict} training={training} dIC={bi['mean']:+.5f} p={bi['p_one_sided']:.4f}")
    print("MICRO005_MINIMAL_DONE " + json.dumps({k: dec[k] for k in
          ("verdict", "model_training", "delta_ic", "delta_ic_p", "delta_top1", "delta_top1_p",
           "perm_p", "production_unchanged")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
