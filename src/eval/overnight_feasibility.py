"""OVERNIGHT-FEAS-001：一夜可行性研究（研究沙盒，不改生产）。

执行 PRE → DATA → QA → EXEC → DIAG → 条件 MICRO → REPORT。
只写 reports/overnight_feas_001 与 data/research/overnight_feas_001。
不写 latest.json、不改模型/配置/计划任务、不发交易信号、不调用券商接口。

用法：
  .venv\\Scripts\\python.exe -m src.eval.overnight_feasibility --max-hours 8 --experiment-id OVERNIGHT-FEAS-001 --research-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import PARQUET_DIR, ROOT  # noqa: E402

OUT = ROOT / "reports" / "overnight_feas_001"
RESEARCH = ROOT / "data" / "research" / "overnight_feas_001"
COST = 0.00145
COST25 = 0.0025
PROD_FILES = [
    "reports/latest.json", "reports/predictions_ledger.csv", "config/targets.yaml",
    "config/l1_best.json", "reports/ens_fixed_weights.json", "run_daily.pyw", "run_gui.pyw",
    "src/report/daily.py", "src/report/run_full.py", "src/report/short_pick.py",
]
EXISTING_FEATURES = ["ret_5", "ret_20", "vol_20", "turn_pctile", "pv_corr_10", "dd_20", "boll_pos", "rsi_14"]
NEW_FEATURES = ["overnight_sum_5", "intraday_sum_5", "overnight_intraday_gap_20", "down_turnover_shock_5"]


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


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    with open(OUT / "logs" / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def stage(name: str, start: float) -> None:
    atomic_json(OUT / "stages" / f"{name}.start.json", {"stage": name, "start": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": round(time.time() - start, 2)})


def stage_end(name: str, t0: float, status: str, result: dict | None = None) -> None:
    atomic_json(OUT / "stages" / f"{name}.end.json", {"stage": name, "end": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": round(time.time() - t0, 2), "status": status, "result": result or {}})


def guard() -> dict:
    files = {}
    for rel in PROD_FILES:
        p = ROOT / rel
        files[rel] = {"exists": p.exists(), "sha256": sha256(p) if p.exists() else None,
                      "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)) if p.exists() else None}
    return {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "files": files, "scheduled_tasks_digest": _tasks_digest()}


def _tasks_digest() -> str | None:
    try:
        r = subprocess.run(["schtasks", "/Query", "/FO", "LIST"], capture_output=True, timeout=30)
        text = r.stdout.decode("gbk", "replace")
        lines = [x.strip() for x in text.splitlines() if "StockDaily" in x or "StockSelf" in x or "TaskName" in x or "Next Run" in x or "Status" in x]
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    except Exception:  # noqa: BLE001
        return None


def preflight() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    RESEARCH.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    before = guard()
    atomic_json(OUT / "prod_guard_before.json", before)
    panel = PARQUET_DIR / "panel.parquet"
    manifest = ROOT / "data" / "manifest.json"
    baseline = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version, "platform": platform.platform(),
        "panel": {"path": str(panel), "exists": panel.exists(), "sha256": sha256(panel) if panel.exists() else None},
        "manifest": json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None,
        "prod_guard": before,
        "research_only": True,
    }
    atomic_json(OUT / "baseline_manifest.json", baseline)
    atomic_json(OUT / "preflight.json", {"status": "PASS", "baseline": baseline, "guard": before})
    return baseline


def data_access(baseline: dict) -> dict:
    import importlib.util
    token_keys = ["TUSHARE_TOKEN", "TS_TOKEN", "TUSHARE_API_TOKEN"]
    tokens = {k: bool(os.environ.get(k)) for k in token_keys}
    tushare = importlib.util.find_spec("tushare") is not None
    local_manifest = baseline.get("manifest") or {}
    # 本地官方 SW 数据存在性/日期；不把当前成分当 PIT
    sw2 = ROOT / "data" / "industry" / "sw2_daily.parquet"
    as_of = local_manifest.get("as_of")
    if sw2.exists():
        try:
            as_of = str(pd.read_parquet(sw2, columns=["date"])["date"].max().date())
        except Exception:  # noqa: BLE001
            pass
    status = "BLOCKED_PIT" if (not tushare or not any(tokens.values())) else "PROVISIONAL"
    payload = {
        "status": status, "tushare_installed": tushare, "token_present": tokens,
        "local_sw2_as_of": as_of, "point_in_time": False,
        "classification_version": "SW2021_PROXY (not broker 881262 PIT)",
        "notes": [
            "Tushare index_member_all/stock_basic/stk_limit/suspend_d 未探测：无包或无凭证；不打印 token。",
            "本地 SW2 指数历史可作为价格/指数级研究；当前成分不是历史时点成员。",
            "因此本实验标记 BLOCKED_PIT；只做非 PIT 的数据/执行/机制诊断，不把结论称为正式历史认证。",
        ],
    }
    atomic_json(OUT / "data_access.json", payload)
    return payload


def qa() -> dict:
    panel_path = PARQUET_DIR / "panel.parquet"
    p = pd.read_parquet(panel_path)
    p["date"] = pd.to_datetime(p["date"])
    duplicated = int(p.duplicated(["code", "date"]).sum())
    neg = {c: int((p[c] <= 0).sum()) for c in ("open", "high", "low", "close") if c in p.columns}
    # log identity: C_t / C_{t-1} = (O_t/C_{t-1})*(C_t/O_t)
    valid = p[["open", "close", "preclose"]].gt(0).all(axis=1) & ~p["suspended"].astype(bool)
    q = p.loc[valid, ["open", "close", "preclose"]].astype(float)
    err = (np.log(q["close"] / q["preclose"]) - (np.log(q["open"] / q["preclose"]) + np.log(q["close"] / q["open"]))).abs()
    log_identity = {"n": int(len(err)), "max_abs_error": round(float(err.max()), 10) if len(err) else None,
                    "mean_abs_error": round(float(err.mean()), 10) if len(err) else None}
    # label identity: ret_1d = next close / close - 1
    p = p.sort_values(["code", "date"])
    p["_next_close"] = p.groupby("code")["close"].shift(-1)
    p["_ret_chk"] = p["_next_close"] / p["close"] - 1
    m = p["ret_1d"].notna() & p["_ret_chk"].notna()
    label_err = (p.loc[m, "ret_1d"] - p.loc[m, "_ret_chk"]).abs()
    label_identity = {"n": int(m.sum()), "max_abs_error": round(float(label_err.max()), 10) if m.any() else None}
    # leakage/truncation test (existing test, isolated subprocess)
    leakage = {"status": "SKIPPED", "stdout": ""}
    try:
        r = subprocess.run([sys.executable, "-m", "tests.test_leakage"], cwd=ROOT, capture_output=True, timeout=600)
        out = (r.stdout or b"").decode("utf-8", "replace")
        leakage = {"status": "PASS" if r.returncode == 0 and "LEAKAGE_AUDIT_OK" in out else "FAIL",
                   "returncode": r.returncode, "stdout_tail": out[-500:]}
    except Exception as e:  # noqa: BLE001
        leakage = {"status": "FAIL", "error": f"{type(e).__name__}: {str(e)[:160]}"}
    trades = ROOT / "reports" / "top1_audit" / "trades.csv"
    trade_codes = pd.read_csv(trades, dtype={"code": str})["code"] if trades.exists() else pd.Series(dtype=str)
    long_excluded = "000927" not in set(trade_codes.astype(str))
    status = "PASS"
    warnings = []
    if duplicated:
        status = "ENGINEERING_FIX_REQUIRED"; warnings.append("duplicate code/date")
    if label_identity["max_abs_error"] not in (None, 0) and label_identity["max_abs_error"] > 1e-6:
        status = "ENGINEERING_FIX_REQUIRED"; warnings.append("ret_1d identity mismatch")
    if leakage["status"] != "PASS":
        status = "ENGINEERING_FIX_REQUIRED"; warnings.append("leakage audit failed")
    if not long_excluded:
        status = "ENGINEERING_FIX_REQUIRED"; warnings.append("000927 appears in short audit trades")
    if log_identity["max_abs_error"] and log_identity["max_abs_error"] > 1e-5:
        warnings.append("log identity error >1e-5; check adjustment/preclose")
    report = {
        "status": status, "panel_path": str(panel_path), "rows": int(len(p)), "codes": int(p["code"].nunique()),
        "date_min": str(p["date"].min().date()), "date_max": str(p["date"].max().date()),
        "duplicates": duplicated, "nonpositive_prices": neg,
        "log_identity": log_identity, "label_identity": label_identity,
        "leakage_test": leakage, "long_target_excluded_from_short_trades": long_excluded,
        "warnings": warnings,
    }
    atomic_json(OUT / "qa_report.json", report)
    return report


def execution() -> dict:
    p = pd.read_parquet(PARQUET_DIR / "panel.parquet")
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["code", "date"]).reset_index(drop=True)
    dates = sorted(p["date"].unique())
    date_idx = {d: i for i, d in enumerate(dates)}
    px = p.set_index(["code", "date"])[["open", "close", "suspended"]]
    tr = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    tr["date"] = pd.to_datetime(tr["date"])
    top = tr[tr["is_top1"].astype(str).str.lower() == "true"].copy()
    top["buy_block"] = top["buy_block"].fillna("").astype(str)
    top["r_2d"] = pd.to_numeric(top["r_2d"], errors="coerce")
    # 独立价格重算 + 账户级（单仓、不重叠）模拟
    rec_rows, account = [], []
    cash, last_exit, skipped_overlap = 1.0, None, 0
    mismatches = 0
    for _, r in top.sort_values("date").iterrows():
        d, code = r["date"], str(r["code"])
        i = date_idx.get(d)
        if i is None or i + 2 >= len(dates):
            continue
        entry, exit_ = dates[i + 1], dates[i + 2]
        try:
            ov = float(px.loc[(code, entry), "open"])
            cv = float(px.loc[(code, exit_), "close"])
            recomputed = cv / ov - 1 if ov > 0 else np.nan
        except Exception:  # noqa: BLE001
            recomputed = np.nan
        stored = float(r["r_2d"]) if pd.notna(r["r_2d"]) else np.nan
        diff = abs(stored - recomputed) if pd.notna(stored) and pd.notna(recomputed) else np.nan
        if pd.notna(diff) and diff > 1e-5:
            mismatches += 1
        rec_rows.append({"date": d, "code": code, "entry_open": round(ov, 4) if pd.notna(ov) else None,
                         "exit_close": round(cv, 4) if pd.notna(cv) else None,
                         "stored_r2d": stored, "recomputed_r2d": recomputed, "diff": diff, "buy_block": r["buy_block"]})
        if r["buy_block"] or pd.isna(stored):
            continue
        if last_exit is not None and (i + 1) <= last_exit:
            skipped_overlap += 1
            continue
        net = stored - 2 * COST
        cash *= 1 + net
        last_exit = i + 2
        account.append({"date": d, "code": code, "r2d": stored, "net_145": round(net, 6),
                        "equity": round(cash, 6), "entry": str(entry.date()), "exit": str(exit_.date())})
    rec = pd.DataFrame(rec_rows)
    acc = pd.DataFrame(account)
    rec.to_csv(OUT / "execution_reconcile.csv", index=False, encoding="utf-8-sig")
    acc.to_csv(OUT / "account_curve.csv", index=False, encoding="utf-8-sig")
    tradeable = top[top["buy_block"].eq("") & top["r_2d"].notna()]
    ideal_net = float(tradeable["r_2d"].mean() - 2 * COST) if len(tradeable) else None
    # existing audit metrics compare
    audit_metrics = json.loads((ROOT / "reports" / "top1_audit" / "metrics.json").read_text(encoding="utf-8")) if (ROOT / "reports" / "top1_audit" / "metrics.json").exists() else {}
    reconcile_ok = bool(mismatches == 0 and len(acc) > 0 and cash > 0)
    status = "EXECUTION_RECONCILED" if reconcile_ok else "EXECUTION_BLOCKED"
    report = {
        "status": status, "top1_rows": int(len(top)), "tradeable": int(len(tradeable)),
        "blocked": int((top["buy_block"] != "").sum()), "mismatches": mismatches,
        "reconcile_rows": int(len(rec)), "account_trades": int(len(acc)),
        "skipped_overlap_signals": skipped_overlap, "final_equity_x": round(cash, 6),
        "account_min_equity": round(float(acc["equity"].min()), 6) if len(acc) else None,
        "ideal_net_per_trade_145bps": round(ideal_net, 6) if ideal_net is not None else None,
        "audit_metrics_r2d_net": audit_metrics.get("p0", {}).get("r2d", {}).get("net_145") if isinstance(audit_metrics, dict) else None,
        "execution_rules": "T signal close → T+1 open buy → T+2 close sell; one unit, no overlapping positions, no leverage; 14.5bps/side",
        "cash_identity_ok": bool(cash > 0 and (acc["equity"] > 0).all() if len(acc) else False),
    }
    (OUT / "execution_report.md").write_text("\n".join([
        "# EXEC-001 账户级执行审计", "",
        f"- 状态：{status}",
        f"- Top1 记录 {len(top)}；可买 {len(tradeable)}；买入受阻 {int((top['buy_block'] != '').sum())}",
        f"- T+1 open → T+2 close；单仓不重叠；跳过重叠信号 {skipped_overlap}",
        f"- 独立价格重算不一致 {mismatches} 条；账户交易 {len(acc)} 笔；最终权益 {cash:.4f}x；最小权益 {report['account_min_equity']}",
        f"- 14.5bps/边全信号理想均值 {ideal_net:.4%}" if ideal_net is not None else "- 无可交易样本",
        f"- 现金恒等式：{'通过' if report['cash_identity_ok'] else '未通过'}；审计报告 r2d 净均值 {report['audit_metrics_r2d_net']}",
        "", "口径：研究沙盒，非实盘；未计入券商最低佣金/印花税的具体期间差异，14.5/25bps 为统一对照成本。",
    ]) + "\n", encoding="utf-8")
    return report


def _new_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values(["code", "date"]).copy()
    g = p.groupby("code", group_keys=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        p["_on"] = np.log(p["open"] / p["preclose"])
        p["_id"] = np.log(p["close"] / p["open"])
    p["overnight_sum_5"] = g["_on"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    p["intraday_sum_5"] = g["_id"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    p["overnight_intraday_gap_20"] = g["_on"].transform(lambda s: s.rolling(20, min_periods=10).sum()) - g["_id"].transform(lambda s: s.rolling(20, min_periods=10).sum())
    p["_turn_med60"] = g["turn"].transform(lambda s: s.rolling(60, min_periods=20).median().shift(1))
    p["_down_shock"] = ((p["pct_chg"] < 0) & (p["turn"] > p["_turn_med60"])).astype(float)
    p["down_turnover_shock_5"] = p.groupby("code")["_down_shock"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    return p


def diagnostics() -> dict:
    p = pd.read_parquet(PARQUET_DIR / "panel.parquet")
    p["date"] = pd.to_datetime(p["date"])
    p = _new_features(p)
    # 保存研究特征（仅研究目录）
    keep = p[["date", "code"] + NEW_FEATURES].copy()
    keep.to_parquet(RESEARCH / "features_diag.parquet", index=False)
    rows = []
    for c in NEW_FEATURES:
        s = p[c]
        corr = {}
        for old in EXISTING_FEATURES:
            if old in p.columns:
                m = p[[c, old]].dropna()
                corr[old] = round(float(m.corr().iloc[0, 1]), 4) if len(m) > 100 else None
        rows.append({"feature": c, "n": int(s.notna().sum()), "missing_rate": round(float(s.isna().mean()), 4),
                     "std": round(float(s.std()), 6) if s.notna().any() else None,
                     "max_abs_corr_existing": max([abs(v) for v in corr.values() if v is not None], default=None),
                     "corr": corr})
    feature_audit = pd.DataFrame(rows)
    feature_audit.to_csv(OUT / "feature_audit.csv", index=False, encoding="utf-8-sig")
    # 第一页分组诊断：仅解释，不改变生产规则
    tr = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    tr["date"] = pd.to_datetime(tr["date"])
    dates = sorted(p["date"].unique()); idx = {d: i for i, d in enumerate(dates)}
    date_set = sorted(tr["date"].unique())
    buy = p[p["suspended"].eq(0) & p["isST"].eq(0) & p["code"].ne("000927")]
    bucket_rows = []
    for d in date_set:
        i = idx.get(d)
        if i is None or i + 2 >= len(dates): continue
        g = buy[buy["date"] == d].sort_values("pct_chg", ascending=False)
        if len(g) < 80: continue
        for name, sub in (("1-20", g.iloc[:20]), ("21-40", g.iloc[20:40]), ("41-60", g.iloc[40:60]), ("bottom20", g.iloc[-20:])):
            rr = []
            for _, r in sub.iterrows():
                c = str(r["code"])
                try:
                    o = float(p.loc[(p["code"] == c) & (p["date"] == dates[i + 1]), "open"].iloc[0])
                    cl = float(p.loc[(p["code"] == c) & (p["date"] == dates[i + 2]), "close"].iloc[0])
                    if o > 0: rr.append(cl / o - 1)
                except Exception:  # noqa: BLE001
                    pass
            if rr:
                bucket_rows.append({"date": d, "bucket": name, "n": len(rr), "up_rate": float(np.mean([x > 0 for x in rr])), "mean_r2d": float(np.mean(rr))})
    bdf = pd.DataFrame(bucket_rows)
    bucket = bdf.groupby("bucket").agg(days=("date", "nunique"), n_mean=("n", "mean"), up_rate=("up_rate", "mean"), mean_r2d=("mean_r2d", "mean")).reset_index()
    bucket["mean_r2d_net_145"] = bucket["mean_r2d"] - 2 * COST
    bucket.to_csv(OUT / "bucket_diagnostic.csv", index=False, encoding="utf-8-sig")
    # MICRO prereg: exact up_limit unavailable -> no model comparison
    pre = {
        "experiment_id": "MICRO-001", "status": "SKIPPED", "reason": "BLOCKED_PIT_AND_UP_LIMIT_UNAVAILABLE",
        "baseline": "v3 short_pick LightGBM p_up Top1 (production label; not modified)",
        "candidate_features": ["overnight_sum_5", "intraday_sum_5", "overnight_intraday_gap_20", "limit_touch_count_5", "down_turnover_shock_5"],
        "available_features": NEW_FEATURES,
        "unavailable_features": ["limit_touch_count_5 (no exact Tushare stk_limit; proxy forbidden)"],
        "rule": "No model training because the fixed feature package is incomplete and PIT is blocked; no MICRO/PASS claim.",
        "cost_bps": 14.5, "hold": "T+1 open → T+2 close", "changed": False,
    }
    atomic_json(OUT / "prereg.json", pre)
    atomic_json(OUT / "paired_metrics.json", {"status": "SKIPPED", "reason": pre["reason"], "baseline": None, "challenger": None})
    diag = {
        "status": "DIAGNOSTIC_ONLY", "feature_audit": feature_audit.to_dict(orient="records"),
        "bucket": bucket.to_dict(orient="records"),
        "limit_feature_available": False, "pit": "BLOCKED_PIT",
        "note": "New OHLCV features are descriptive only; no model comparison without exact limits/PIT.",
    }
    (OUT / "diagnostic.md").write_text("\n".join([
        "# DIAG-001 机制诊断", "",
        "状态：DIAGNOSTIC_ONLY；PIT BLOCKED；limit_touch_count_5 无精确历史涨停价，未用 9.9% 代理冒充。",
        "", "## 新特征审计", feature_audit.to_markdown(index=False), "",
        "## 第一页分组（T+1 open→T+2 close，14.5bps/边；仅诊断）", bucket.to_markdown(index=False), "",
        "分组不用于替换生产第一页规则；当前成分非 PIT。",
        "", "## MICRO-001", "SKIPPED：固定五特征包缺少精确涨停价数据，且 PIT 未解决；不做不完整的模型对照。",
    ]), encoding="utf-8")
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=8)
    ap.add_argument("--experiment-id", default="OVERNIGHT-FEAS-001")
    ap.add_argument("--research-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    t_all = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    RESEARCH.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "run_meta.json", vars(args))
    statuses = {}
    t = time.time(); stage("PRE-001", t_all)
    baseline = preflight(); statuses["PRE-001"] = "ENGINEERING_PASS"; stage_end("PRE-001", t, "ENGINEERING_PASS")
    t = time.time(); stage("DATA-001", t_all)
    da = data_access(baseline); statuses["DATA-001"] = da["status"]; stage_end("DATA-001", t, da["status"])
    (OUT / "pit_audit.md").write_text("# PIT 审计\n\n"
        "- PRICE: 历史日线可作为非 PIT 研究输入（来源 baostock/akshare 缓存）。\n"
        "- UNIVERSE: 当前电池成分快照，无历史成员；**不是 PIT**。\n"
        "- CLASSIFICATION: SW2021_PROXY，非券商 881262 历史分类。\n"
        "- FEATURE: OHLCV 特征只用 T 日及以前；但训练样本池仍受非 PIT 影响。\n"
        "- Tushare index_member_all/stock_basic/stk_limit/suspend_d：未安装/无凭证，无法核验历史成分与精确涨停价。\n"
        "- **总状态：BLOCKED_PIT**；不做正式历史策略认证。\n", encoding="utf-8")
    t = time.time(); stage("QA-001", t_all)
    qa_r = qa(); statuses["QA-001"] = qa_r["status"]; stage_end("QA-001", t, qa_r["status"], qa_r)
    t = time.time(); stage("EXEC-001", t_all)
    ex = execution(); statuses["EXEC-001"] = ex["status"]; stage_end("EXEC-001", t, ex["status"], ex)
    t = time.time(); stage("DIAG-001", t_all)
    dg = diagnostics(); statuses["DIAG-001"] = dg["status"]; stage_end("DIAG-001", t, dg["status"])
    statuses["MICRO-001"] = "SKIPPED_BLOCKED_PIT"
    after = guard(); atomic_json(OUT / "prod_guard_after.json", after)
    unchanged = before_equals = after.get("files") == (json.loads((OUT / "prod_guard_before.json").read_text(encoding="utf-8")).get("files"))
    statuses["PROD"] = "PROD_UNCHANGED" if unchanged else "PROD_CHANGED_ALERT"
    final_status = "BLOCKED_PIT" if da["status"] == "BLOCKED_PIT" else "DIAGNOSTIC_ONLY"
    # report
    report = [
        f"# OVERNIGHT-FEAS-001 可行性研究报告（{args.experiment_id}）", "",
        f"- 运行：{time.strftime('%Y-%m-%d %H:%M:%S')}；总耗时 {time.time()-t_all:.1f}s（上限 {args.max_hours}h，可提前结束）",
        "- **研究沙盒**；不产生实盘/影子信号，不修改生产 `latest.json`。", "",
        "## 结论", "",
        "| 阶段 | 状态 |", "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in statuses.items()] + [
        "", f"**最终状态：{final_status} + DIAGNOSTIC_ONLY；MICRO-001 SKIPPED。**", "",
        "## 关键事实", "",
        f"- QA：{qa_r['status']}；duplicates={qa_r['duplicates']}；leakage={qa_r['leakage_test']['status']}；log identity max={qa_r['log_identity']['max_abs_error']}；label identity max={qa_r['label_identity']['max_abs_error']}",
        f"- EXEC：{ex['status']}；Top1 {ex['top1_rows']} 行，可买 {ex['tradeable']}，买入受阻 {ex['blocked']}；独立重算不一致 {ex['mismatches']}；账户 {ex['account_trades']} 笔，跳过重叠 {ex['skipped_overlap_signals']}，最终权益 {ex['final_equity_x']}x；14.5bps 全信号理想均值 {ex['ideal_net_per_trade_145bps']}",
        f"- DIAG：{dg['status']}；新特征仅 {len(NEW_FEATURES)} 个可用；limit_touch_count_5 无精确数据；第一页分组诊断仅解释性。",
        f"- DATA：{da['status']}；Tushare 未安装/无凭证，历史成分与涨停价无法核验。",
        f"- MICRO：SKIPPED_BLOCKED_PIT；没有用不完整特征包强行训练，也没有宣称模型 PASS。",
        "", "## 下一步（数据前置）", "",
        "1. 获取合法 Tushare 权限（index_member_all/stock_basic/stk_limit/suspend_d），做 PIT 审计后再谈模型。",
        "2. 每日保存 881262 真实成分快照，而不是用 SW 代理。",
        "3. 只有数据 PIT 与账户级 EXEC 都审计通过后，才允许预注册 MICRO-001；旧 v4/A/NETRET/RELIAB/TARGET 保持 FAIL/STOP。",
        "", "## 产物", "",
        "preflight.json / data_access.json / pit_audit.md / qa_report.json / execution_report.md / diagnostic.md / prereg.json / paired_metrics.json / registry.csv / final_manifest.json / logs/",
        "", "**声明：研究输出，不构成投资建议；生产 v3 保持不变。**",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    row = pd.DataFrame([{"experiment_id": args.experiment_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "status": final_status, "qa": statuses["QA-001"], "exec": statuses["EXEC-001"],
                         "diag": statuses["DIAG-001"], "micro": statuses["MICRO-001"], "prod": statuses["PROD"]}])
    row.to_csv(OUT / "registry.csv", index=False, encoding="utf-8-sig")
    files = {str(p.relative_to(OUT)): sha256(p) for p in OUT.rglob("*") if p.is_file() and p.name != "final_manifest.json"}
    atomic_json(OUT / "final_manifest.json", {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "status": final_status,
                                              "prod_unchanged": unchanged, "files": files})
    print("OVERNIGHT_FEAS_DONE", json.dumps({"status": final_status, "qa": statuses["QA-001"], "exec": statuses["EXEC-001"], "micro": statuses["MICRO-001"], "prod": statuses["PROD"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
