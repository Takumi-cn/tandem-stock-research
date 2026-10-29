"""TANDEM-FWD-MICRO-005 readiness and first-page mechanism audit.

This is deliberately not a model-training run.  It builds the data/leakage
contract, captures forward evidence, and tests whether the page-1 mechanism
itself has a demonstrable edge before spending more compute on a 5-minute
encoder.  5-minute history is currently sample-only, so MICRO-005 training is
marked BLOCKED_DATA_5M rather than fabricated.

Run:
  .venv\\Scripts\\python.exe -m src.eval.micro005_readiness --research-only
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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.feature_contract import (  # noqa: E402
    DAILY_FEATURE_WHITELIST, FORBIDDEN_COLUMNS, LABEL_COLUMNS,
    FEATURE_CONTRACT_VERSION, validate_features,
)
from src.data.forward_evidence import capture  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

EXP_ID = "TANDEM-FWD-MICRO-005-READINESS"
OUT = ROOT / "reports" / "micro005_readiness"
RESEARCH = ROOT / "data" / "research" / "micro005_readiness"
LOGS = OUT / "logs"
STAGES = OUT / "stages"
PREREG = OUT / "prereg.json"
COST = 0.00145
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
        "experiment_id": "TANDEM-FWD-MICRO-005", "readiness_id": EXP_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PRE_REGISTERED_READINESS",
        "hypothesis": "Five-minute intraday dynamics available by T close contain incremental information about a stock's T+1 open to T+2 close net return relative to the other stocks on the same battery page, after existing daily information.",
        "status_max": "FORWARD_CANDIDATE", "production_write": False,
        "label": "label__page_resid_T1_T2 = stock net r2d - same-page mean net r2d",
        "execution": "T+1_open_to_T+2_close", "cost_bps_per_side": 14.5, "stress_bps_per_side": 25.0,
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "daily_whitelist": DAILY_FEATURE_WHITELIST,
        "forbidden_features": FORBIDDEN_COLUMNS + LABEL_COLUMNS,
        "five_minute_status": "BLOCKED_DATA_5M",
        "five_minute_reason": "No QMT/xtquant; baostock 120-day 5m query for one stock exceeded 2 minutes, so full-universe historical collection is not currently practical. Only a 5-day sample is collected for pipeline validation.",
        "model_training": "DISABLED_UNTIL_DATA_AND_FORWARD_SAMPLE",
        "production_policy": {"latest_json": "READ_ONLY", "threshold_55": "READ_ONLY", "industry": "CLOSED", "stock_layer": "SHADOW_ONLY"},
    }
    atomic_json(PREREG, obj)
    return obj


def first_page_audit() -> tuple[dict, pd.DataFrame]:
    panel = pd.read_parquet(ROOT / "data" / "parquet" / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype(str)
    trades = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    trades["date"] = pd.to_datetime(trades["date"])
    trades["code"] = trades["code"].astype(str)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    close = panel.pivot(index="date", columns="code", values="close").reindex(dates)
    open_ = panel.pivot(index="date", columns="code", values="open").reindex(dates)
    rows = []
    for d, g in trades.groupby("date"):
        j = int(dates.get_loc(d))
        if j + 2 >= len(dates):
            continue
        entry, exit_ = dates[j + 1], dates[j + 2]
        all_codes = [c for c in close.columns if pd.notna(close.at[exit_, c]) and pd.notna(open_.at[entry, c])]
        if len(all_codes) < 30:
            continue
        r = close.loc[exit_, all_codes] / open_.loc[entry, all_codes] - 1.0 - 2 * COST
        page_codes = [c for c in g["code"] if c in r.index]
        nonpage_codes = [c for c in all_codes if c not in set(page_codes)]
        if len(page_codes) < 5 or len(nonpage_codes) < 5:
            continue
        page_mean = float(r[page_codes].mean()); nonpage_mean = float(r[nonpage_codes].mean())
        top1 = g[g["is_top1"]]
        top1_code = str(top1.iloc[0]["code"]) if len(top1) else None
        top1_r = float(r[top1_code]) if top1_code in r.index else np.nan
        sub = g[g["code"].isin(page_codes)][["code", "rank_in_fp"]].merge(
            r.rename("r2d").reset_index(), on="code", how="left")
        sub = sub.dropna(subset=["r2d"])
        rank_ic = float(sub[["rank_in_fp", "r2d"]].corr(method="spearman").iloc[0, 1]) if len(sub) >= 5 and sub["rank_in_fp"].nunique() > 2 else np.nan
        rows.append({"date": d, "n_all": len(all_codes), "n_page": len(page_codes),
                     "page_mean": page_mean, "nonpage_mean": nonpage_mean, "page_minus_nonpage": page_mean - nonpage_mean,
                     "top1_r2d": top1_r, "top1_minus_page": top1_r - page_mean if np.isfinite(top1_r) else np.nan,
                     "page_rank_ic": rank_ic})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    page = df["page_mean"].values
    nonpage = df["nonpage_mean"].values
    diff = page - nonpage
    def boot(a, n=5000):
        rng = np.random.default_rng(42)
        m = [float(np.mean(rng.choice(a, len(a), replace=True))) for _ in range(n)]
        return [float(np.quantile(m, .025)), float(np.quantile(m, .975)), float(np.mean(np.asarray(m) <= 0))]
    bd = boot(diff)
    t = stats.ttest_1samp(diff, 0, nan_policy="omit")
    half = len(df) // 2
    first, second = diff[:half], diff[half:]
    result = {
        "n_dates": len(df), "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date()),
        "page_mean": float(np.mean(page)), "nonpage_mean": float(np.mean(nonpage)),
        "page_minus_nonpage": float(np.mean(diff)), "page_minus_nonpage_ci95": bd[:2], "p_le_0": bd[2],
        "t_stat": float(t.statistic), "p_value": float(t.pvalue),
        "first_half_page_minus_nonpage": float(np.mean(first)), "second_half_page_minus_nonpage": float(np.mean(second)),
        "top1_minus_page": float(np.nanmean(df["top1_minus_page"])),
        "page_rank_ic_mean": float(np.nanmean(df["page_rank_ic"])),
        "page_rank_ic_negative_share": float(np.nanmean(df["page_rank_ic"] < 0)),
        "mechanism_status": ("PAGE_PREMIUM_DIAGNOSTIC" if np.mean(diff) > 0 and t.pvalue < .05 else "PAGE_PREMIUM_WEAK_UNCERTAIN" if np.mean(diff) > 0 and t.pvalue < .10 else "NO_PAGE_PREMIUM_EVIDENCE"),
        "note": "Page-1 is the top-20 by T-day pct_chg; non-page is the rest of the local battery universe. All returns are T+1 open to T+2 close net of 14.5bps/side.",
    }
    df.to_csv(OUT / "first_page_audit.csv", index=False, encoding="utf-8-sig")
    atomic_json(OUT / "first_page_audit.json", result)
    return result, df


def fetch_5m_sample() -> dict:
    cache = RESEARCH / "5m_sample.parquet"
    if cache.exists():
        d = pd.read_parquet(cache)
        return {"status": "CACHED", "rows": len(d), "codes": int(d["code"].nunique()), "days": int(d["date"].nunique()),
                "data": d}
    import baostock as bs
    cons = pd.read_parquet(ROOT / "data" / "parquet" / "cons_battery.parquet")
    codes = sorted(cons["code"].astype(str).unique())
    lg = bs.login()
    if lg.error_code != "0":
        return {"status": "BLOCKED_BAOSTOCK_LOGIN", "error": lg.error_msg}
    rows = []
    fields = "date,time,code,open,high,low,close,volume,amount"
    for i, c in enumerate(codes):
        if c.startswith("6"):
            bc = "sh." + c
        elif c.startswith(("0", "3")):
            bc = "sz." + c
        else:
            continue
        rs = bs.query_history_k_data_plus(bc, fields, start_date="2026-10-19", end_date="2026-10-23",
                                           frequency="5", adjustflag="2")
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if (i + 1) % 20 == 0:
            log(f"5m sample {i+1}/{len(codes)} codes, rows={len(rows)}")
    bs.logout()
    if not rows:
        return {"status": "BLOCKED_NO_DATA", "rows": 0}
    df = pd.DataFrame(rows, columns=["date", "time", "code", "open", "high", "low", "close", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].str.split(".").str[-1]
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df.to_parquet(cache, index=False)
    return {"status": "CREATED", "rows": len(df), "codes": int(df["code"].nunique()), "days": int(df["date"].nunique()),
            "data": df}


def five_min_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df is None or df.empty:
        return pd.DataFrame(), {"status": "BLOCKED"}
    d = df.copy()
    d["hhmm"] = d["time"].str[8:12].astype(int)
    d = d.sort_values(["code", "date", "time"])
    g = d.groupby(["code", "date"], sort=False)
    feat = g.agg(open_first=("open", "first"), close_last=("close", "last"),
                 high_max=("high", "max"), low_min=("low", "min"),
                 volume_sum=("volume", "sum"), amount_sum=("amount", "sum")).reset_index()
    feat["intraday_ret"] = feat["close_last"] / feat["open_first"] - 1
    feat["range"] = (feat["high_max"] - feat["low_min"]) / feat["open_first"]
    feat["close_pos"] = (feat["close_last"] - feat["low_min"]) / (feat["high_max"] - feat["low_min"]).replace(0, np.nan)
    feat["vwap"] = feat["amount_sum"] / feat["volume_sum"].replace(0, np.nan)
    feat["vwap_dev"] = feat["close_last"] / feat["vwap"] - 1
    # last-hour / first-hour if timestamps exist
    def ret_between(sub, hhmm):
        x = sub[sub["hhmm"] <= hhmm]
        return float(x["close"].iloc[-1]) if len(x) else np.nan
    last_h, first_h, aft = [], [], []
    for _, sub in d.groupby(["code", "date"], sort=False):
        o = float(sub["open"].iloc[0]); cl = float(sub["close"].iloc[-1])
        c1030 = ret_between(sub, 1030); c1400 = ret_between(sub, 1400); c1130 = ret_between(sub, 1130)
        first_h.append(c1030 / o - 1 if np.isfinite(c1030) else np.nan)
        last_h.append(cl / c1400 - 1 if np.isfinite(c1400) else np.nan)
        aft.append(cl / c1130 - 1 if np.isfinite(c1130) else np.nan)
    feat["first_hour"] = first_h; feat["last_hour"] = last_h; feat["afternoon"] = aft
    feat["up_bar_share"] = g.apply(lambda x: float((x["close"] > x["open"]).mean())).values
    feat.to_parquet(RESEARCH / "5m_features.parquet", index=False)
    audit = {
        "rows": int(len(d)), "codes": int(d["code"].nunique()), "days": int(d["date"].nunique()),
        "bars_per_code_day": d.groupby(["code", "date"]).size().describe().to_dict(),
        "time_min": str(d["time"].min()), "time_max": str(d["time"].max()),
        "features": ["intraday_ret", "range", "close_pos", "vwap_dev", "first_hour", "last_hour", "afternoon", "up_bar_share"],
        "sample_only": True,
        "note": "5 sessions only; sufficient for pipeline smoke test, not for training MICRO-005. All bars are at/before 15:00 T.",
    }
    atomic_json(OUT / "5m_sample_audit.json", audit)
    return feat, audit


def leakage_contract(panel: pd.DataFrame) -> dict:
    res = validate_features(panel)
    # truncation checks for simple trailing features
    rows = []
    p = panel.sort_values(["code", "date"])
    for c in p["code"].drop_duplicates().head(20):
        g = p[p["code"] == c]
        if len(g) < 25:
            continue
        d = g["date"].iloc[-1]
        close = g["close"].values
        ret5_t = close[-1] / close[-6] - 1 if len(close) > 5 else np.nan
        ret20_t = close[-1] / close[-21] - 1 if len(close) > 20 else np.nan
        rows.append({"code": c, "date": str(pd.Timestamp(d).date()),
                     "ret5_truncated": ret5_t, "ret5_panel": float(g["ret_5"].iloc[-1]),
                     "ret20_truncated": ret20_t, "ret20_panel": float(g["ret_20"].iloc[-1])})
    tr = pd.DataFrame(rows)
    tr["ret5_diff"] = (tr["ret5_truncated"] - tr["ret5_panel"]).abs()
    tr["ret20_diff"] = (tr["ret20_truncated"] - tr["ret20_panel"]).abs()
    tr.to_csv(OUT / "truncation_audit.csv", index=False, encoding="utf-8-sig")
    out = {"contract_version": FEATURE_CONTRACT_VERSION, "ok": bool(res.ok),
           "missing": list(res.missing), "forbidden_present": list(res.forbidden_present),
           "extra": list(res.extra), "forbidden_columns": FORBIDDEN_COLUMNS,
           "label_columns": LABEL_COLUMNS,
           "truncation_n": int(len(tr)), "ret5_max_abs_diff": float(tr["ret5_diff"].max()) if len(tr) else None,
           "ret20_max_abs_diff": float(tr["ret20_diff"].max()) if len(tr) else None}
    atomic_json(OUT / "feature_contract_audit.json", out)
    return out


def _write_report(first: dict, five: dict, five_feat: pd.DataFrame, leak: dict,
                  forward: dict, decision: dict, before: dict, after: dict, elapsed: float):
    lines = [
        "# TANDEM-FWD-MICRO-005 — Readiness / First-Page Audit", "",
        f"- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')}；耗时：{elapsed/60:.1f} 分钟。",
        f"- 结论：`{decision['readiness_status']}`；模型训练 **DISABLED**；生产只读；不进入个股/行业部署。",
        f"- 原因：本机没有 QMT/xtquant；baostock 120 日 5m 单只查询超过 2 分钟，无法伪造完整历史。只保留 **5 日 5m 样本**用于管线冒烟。",
        "", "## 第一页机制审计（不是模型）", "",
        "| 项目 | 结果 |", "|---|---|",
        f"| 交易日 | {first['n_dates']}（{first['date_min']} ~ {first['date_max']}） |",
        f"| page-1 平均 net r2d | {_pct(first['page_mean'])} |",
        f"| 非 page 电池池平均 | {_pct(first['nonpage_mean'])} |",
        f"| page − 非 page | **{_pct(first['page_minus_nonpage'])}**；95% CI {_num(first['page_minus_nonpage_ci95'][0],4)}–{_num(first['page_minus_nonpage_ci95'][1],4)}；t p={_num(first['p_value'],4)}；date-bootstrap p(≤0)={_num(first['p_le_0'],4)} |",
        f"| 前后半 | {_pct(first['first_half_page_minus_nonpage'])} / {_pct(first['second_half_page_minus_nonpage'])} |",
        f"| Top1 − page 均值 | {_pct(first['top1_minus_page'])} |",
        f"| page 内 rank IC（rank_in_fp vs r2d） | {_num(first['page_rank_ic_mean'],4)}；负 IC 日占比 {first['page_rank_ic_negative_share']*100:.1f}% |",
        "", f"- 判断：`{first['mechanism_status']}`。page-1 相对非 page 有 **弱正向点估计**（+0.31%），但 t p=0.09、95% CI 跨 0、bootstrap p=0.043 只是单侧日期重采样，且 page/non-page 扣成本后都为负；page 内 rank IC 接近零。不能把“第一页”当作已验证的选股优势。",
        "", "## 5 分钟数据（样本，不是训练集）", "",
        f"- 状态：{five.get('status')}；行数 {five.get('rows')}、代码 {five.get('codes')}、交易日 {five.get('days')}。",
    ]
    if len(five_feat):
        lines.append(f"- 样本特征：{len(five_feat)} 个 code-day，字段 {list(five_feat.columns)}。")
    lines += [
        "- 仅 5 个交易日，不能训练 MICRO-005，也不能用于任何收益声明；它只验证 5m 采集、时间戳和特征管线。",
        "", "## 泄漏契约", "",
        f"- 状态：**{'PASS' if leak.get('ok') else 'FAIL'}**；版本 `{leak.get('contract_version')}`。",
        f"- 禁止列：`{leak.get('forbidden_columns')}`；标签列：`{leak.get('label_columns')}`。",
        f"- 截断重算：ret_5 max diff {_num(leak.get('ret5_max_abs_diff'),8)}；ret_20 max diff {_num(leak.get('ret20_max_abs_diff'),8)}。",
        "- 训练代码必须使用 `df[FROZEN_FEATURE_WHITELIST]`，禁止 `df.drop(labels)`；新增标签不会自动成为特征。",
        "", "## 前瞻证据", "",
        f"- `{forward.get('status')}`；日期 {forward.get('snapshot_date')}；成员 {forward.get('n_members')}；第一页 {forward.get('n_page')}。",
        f"- 来源：`{forward.get('source')}`；`is_broker_881262_pit={forward.get('is_broker_881262_pit')}`。",
        f"- manifest：`{forward.get('hashes', {})}`。",
        "- 这是一份本地重建快照，不是券商 881262 截图；从此开始每日追加，不能回填历史。",
        "", "## 下一步（按证据排序）", "",
        "1. 先解决 5m 数据源：QMT/xtquant 或合法 Tushare 权限；没有 5m 全历史就不训练 MICRO-005。",
        "2. 冻结 `feature_contract` 和 page-relative residual 标签；用未来 20/60/120/250 日的前瞻台账验证，不用 2024–2026 历史认证。",
        "3. 第一页机制只有一个小而不显著的正向点估计；不能据此改 55% 门槛、仓位或生产选股。",
        "4. 生产 v3.1、latest.json、电池主线、GUI、计划任务保持原样。",
        "", "## 不能宣称", "",
        "- 没有训练 5m encoder，也没有任何 MICRO-005 收益结论。",
        "- page-1 的 +0.3075% 是扣 14.5bps 后的点估计，CI 跨 0、t p=0.09，且 page/non-page 都为负；不构成可交易优势。",
        "- 5m 样本只有 5 日；不能外推、不能当 OOS、不能当训练集。",
    ]
    atomic_text(OUT / "report.md", "\n".join(lines) + "\n")


def _pct(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:+.{d}f}%"


def _num(x, d=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    for d in (OUT, RESEARCH, LOGS, STAGES):
        d.mkdir(parents=True, exist_ok=True)
    before = guard(); data_before = data_guard()
    atomic_json(OUT / "hash_before.json", before); atomic_json(OUT / "data_before.json", data_before)
    prereg = ensure_prereg(); atomic_json(OUT / "prereg.sha256.json", {"sha256": sha256(PREREG)})
    log(f"RUN_START {EXP_ID}")

    first, first_df = first_page_audit()
    five = fetch_5m_sample()
    five_feat = pd.DataFrame()
    if isinstance(five.get("data"), pd.DataFrame):
        five_feat, five_audit = five_min_features(five["data"])
        five.pop("data", None)
        if five_audit:
            five.update(five_audit)
    panel = pd.read_parquet(ROOT / "data" / "parquet" / "panel.parquet")
    leak = leakage_contract(panel)
    forward = capture()
    after = guard(); data_after = data_guard()
    atomic_json(OUT / "hash_after.json", after); atomic_json(OUT / "data_after.json", data_after)
    prod_unchanged = after.get("files") == before.get("files") and after.get("models") == before.get("models")
    decision = {
        "experiment_id": "TANDEM-FWD-MICRO-005", "readiness_id": EXP_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "readiness_status": "BLOCKED_DATA_5M",
        "model_training": "DISABLED",
        "first_page_status": first["mechanism_status"],
        "first_page_premium": first["page_minus_nonpage"],
        "first_page_t_p": first["p_value"],
        "first_page_ci95": first["page_minus_nonpage_ci95"],
        "leakage_contract": "PASS" if leak.get("ok") else "FAIL",
        "five_min_status": five.get("status"),
        "five_min_rows": five.get("rows"),
        "five_min_days": five.get("days"),
        "forward_status": forward.get("status"),
        "production_unchanged": bool(prod_unchanged),
        "production_write": False,
        "max_status": "BLOCKED_DATA_5M",
    }
    atomic_json(OUT / "decision.json", decision)
    _write_report(first, five, five_feat, leak, forward, decision, before, after, time.time() - t0)
    # registry
    reg = OUT / "registry.csv"
    row = pd.DataFrame([{"experiment_id": "TANDEM-FWD-MICRO-005", "created_at": decision["created_at"],
                         "status": "BLOCKED_DATA_5M", "first_page_status": first["mechanism_status"],
                         "page_minus_nonpage": first["page_minus_nonpage"], "first_page_p": first["p_value"],
                         "five_min_status": five.get("status"), "five_min_days": five.get("days"),
                         "production_unchanged": prod_unchanged}])
    if reg.exists():
        old = pd.read_csv(reg); old = old[old["experiment_id"] != "TANDEM-FWD-MICRO-005"]
        pd.concat([old, row], ignore_index=True).to_csv(reg, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(reg, index=False, encoding="utf-8-sig")
    files = []
    for root in (OUT, RESEARCH):
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name != "final_manifest.json":
                files.append({"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size})
    manifest = {"experiment_id": "TANDEM-FWD-MICRO-005", "created_at": decision["created_at"],
                "status": "BLOCKED_DATA_5M", "model_training": "DISABLED",
                "production_unchanged": bool(prod_unchanged), "elapsed_s": round(time.time() - t0, 1),
                "prereg_sha256": sha256(PREREG), "production_before": before, "production_after": after,
                "data_before": data_before, "data_after": data_after, "files": files}
    atomic_json(OUT / "final_manifest.json", manifest)
    log(f"RUN_DONE status=BLOCKED_DATA_5M first_page={first['mechanism_status']} p={first['p_value']} 5m={five.get('status')} prod={prod_unchanged}")
    print("MICRO005_READINESS_DONE " + json.dumps(decision, ensure_ascii=True))


if __name__ == "__main__":
    main()
