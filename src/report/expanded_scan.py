"""扩池探索扫描（研究沙盒，不写入 latest.json / 不产生实盘信号）。

范围：用户 2026-10-26 截图里的强势 SW2 行业（按名称映射）+ 电池对照
数据：当前成分股，非 PIT；日线截至 2026-10-23（本地可见的最后完整交易日）
标签：T+1 open -> T+1 close（与“次日开盘买、收盘卖”执行口径一致），不是 close-to-close
模型：低维量价/行业相对特征 + 强正则 LightGBM + 最近 60 日 Platt 校准
输出：reports/expanded_scan/{report.md,latest.json,registry.csv}
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import PARQUET_DIR, RAW_DIR, ROOT  # noqa: E402

DATA = ROOT / "data" / "expanded"
OUT = ROOT / "reports" / "expanded_scan"
TARGETS = [
    ("801961", "油气开采Ⅱ", "石油石化"),
    ("801962", "油服工程", "石油石化"),
    ("801154", "医药商业", "医药生物"),
    ("801963", "炼化及贸易", "石油石化"),
    ("801128", "休闲食品", "食品饮料"),
    ("801737", "电池", "电力设备"),
]
START_DATA = "2022-12-01"
START_TRAIN = "2023-04-01"
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,isST"
FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    "vol_20", "vol_ratio", "turn_pctile", "amount_ratio",
    "rsi_14", "boll_pos", "dd_20", "close_pos", "gap_open",
    "ind_ret_5", "ind_ret_20", "ind_rel_5", "ind_rel_20",
    "xrank_ret5", "xrank_vol", "xrank_turn", "xrank_rel5",
]


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)


def _cons() -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "constituents.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import akshare as ak
    rows = []
    for code, name, parent in TARGETS:
        if code == "801737":
            c = pd.read_parquet(PARQUET_DIR / "cons_battery.parquet")[["code"]].copy()
            nm_path = RAW_DIR / "stock_names.parquet"
            if nm_path.exists():
                nm = pd.read_parquet(nm_path)
                c["name"] = c["code"].astype(str).map(nm.set_index("code")["name"])
            else:
                c["name"] = ""
        else:
            c = None
            for attempt in range(3):
                try:
                    c = ak.index_component_sw(symbol=code)
                    break
                except Exception as e:  # noqa: BLE001
                    log(f"cons fail {code} attempt {attempt+1}: {type(e).__name__} {str(e)[:80]}")
                    time.sleep(2 + attempt)
            if c is None or len(c) == 0:
                log(f"cons skipped {code} {name}")
                continue
            # 该接口列名可能随版本/编码变化：按已知中文列重命名，失败则按位置兜底
            c = c.rename(columns={"证券代码": "code", "证券简称": "name"})
            if "code" not in c.columns and len(c.columns) >= 3:
                c = c.rename(columns={c.columns[1]: "code", c.columns[2]: "name"})
            if "name" not in c.columns and len(c.columns) >= 3:
                c = c.copy()
                c["name"] = c.iloc[:, 2].astype(str)
            if "code" not in c.columns or "name" not in c.columns:
                log(f"cons unexpected columns {code}: {list(c.columns)}")
                continue
        c = c.copy()
        c["code"] = c["code"].astype(str).str.zfill(6)
        c["name"] = c.get("name", pd.Series("", index=c.index)).fillna("").astype(str)
        c["industry_code"] = code
        c["industry_name"] = name
        c["parent"] = parent
        rows.append(c[["code", "name", "industry_code", "industry_name", "parent"]])
        log(f"constituents {code} {name}: {len(c)}")
    if not rows:
        raise RuntimeError("no constituents fetched")
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)
    out.to_parquet(path, index=False)
    return out


def _fetch_daily(cons: pd.DataFrame) -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "raw_daily.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        old["date"] = pd.to_datetime(old["date"])
        have = set(old["code"].astype(str).unique())
        need = sorted(set(cons["code"]) - have)
        if not need:
            return old
    else:
        # 已有电池池日线可复用，避免把 93 只电池股票重抓一遍
        old = pd.DataFrame()
        bcode = set(cons[cons["industry_code"] == "801737"]["code"].astype(str))
        bpath = RAW_DIR / "daily_baostock.parquet"
        if bpath.exists() and bcode:
            b = pd.read_parquet(bpath)
            b["date"] = pd.to_datetime(b["date"])
            b = b[b["code"].astype(str).isin(bcode) & (b["date"] >= pd.Timestamp(START_DATA))].copy()
            old = b
        have = set(old["code"].astype(str).unique()) if len(old) else set()
        need = sorted(set(cons["code"]) - have)
    log(f"baostock fetch new stocks={len(need)}")
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)
    frames, failures = [], []
    for i, code in enumerate(need, 1):
        mkt = "sh" if code.startswith(("6", "9")) else "sz"
        try:
            rs = bs.query_history_k_data_plus(f"{mkt}.{code}", FIELDS, start_date=START_DATA, end_date="2026-10-23", frequency="d", adjustflag="2")
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df = pd.DataFrame(rows, columns=FIELDS.split(","))
                df["code"] = code
                frames.append(df)
        except Exception as e:  # noqa: BLE001
            failures.append({"code": code, "error": f"{type(e).__name__}: {str(e)[:80]}"})
        if i % 25 == 0:
            log(f"fetch {i}/{len(need)} fail={len(failures)}")
        time.sleep(0.15)
    bs.logout()
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FIELDS.split(","))
    if len(new):
        new["date"] = pd.to_datetime(new["date"])
    out = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(out):
        out["date"] = pd.to_datetime(out["date"])
        out = out.drop_duplicates(subset=["date", "code"], keep="last").sort_values(["code", "date"]).reset_index(drop=True)
    out.to_parquet(path, index=False)
    (DATA / "daily_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"raw rows={len(out)} codes={out['code'].nunique() if len(out) else 0} failures={len(failures)}")
    return out


def _industry_features(cons: pd.DataFrame) -> pd.DataFrame:
    sw2 = pd.read_parquet(ROOT / "data" / "industry" / "sw2_daily.parquet")
    sw2["date"] = pd.to_datetime(sw2["date"])
    sw2 = sw2[sw2["code"].astype(str).isin(set(cons["industry_code"].astype(str)))]
    sw2 = sw2.sort_values(["code", "date"])
    sw2["ret_5"] = sw2.groupby("code")["close"].pct_change(5)
    sw2["ret_20"] = sw2.groupby("code")["close"].pct_change(20)
    out = sw2[["date", "code", "ret_5", "ret_20"]].rename(columns={"code": "industry_code", "ret_5": "ind_ret_5", "ret_20": "ind_ret_20"})
    return out


def _build_panel(cons: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[raw["date"] >= pd.Timestamp(START_DATA)].copy()
    for c in ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "isST"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["pct_chg"] = raw["pctChg"]
    raw["date"] = pd.to_datetime(raw["date"])
    cal = pd.DatetimeIndex(sorted(raw["date"].unique()))
    frames = []
    for code, g in raw.groupby("code"):
        g = g.sort_values("date")
        x = pd.DataFrame({"date": cal}).merge(g.drop(columns=["code"]), on="date", how="left")
        x["code"] = code
        x["suspended"] = x["close"].isna().astype(int)
        for c in ("open", "high", "low", "close", "preclose"):
            x[c] = x[c].ffill(limit=5)
        x["volume"] = x["volume"].fillna(0)
        x["amount"] = x["amount"].fillna(0)
        x["turn"] = x["turn"].fillna(0)
        x["pct_chg"] = np.where(x["suspended"] == 1, 0.0, x["pct_chg"].fillna(0.0))
        x["isST"] = x["isST"].fillna(0).astype(int)
        frames.append(x)
    p = pd.concat(frames, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)
    m = cons[["code", "name", "industry_code", "industry_name", "parent"]].copy()
    m["code"] = m["code"].astype(str)
    p["code"] = p["code"].astype(str)
    p = p.merge(m, on="code", how="left")
    p = p.merge(_industry_features(cons), on=["date", "industry_code"], how="left")
    g = p.groupby("code", group_keys=False)
    for n in (1, 3, 5, 10, 20, 60):
        p[f"ret_{n}"] = g["close"].pct_change(n)
    p["vol_20"] = g["pct_chg"].transform(lambda s: s.rolling(20).std())
    p["vol_ratio"] = g["pct_chg"].transform(lambda s: s.rolling(5).std() / s.rolling(20).std())
    p["turn_pctile"] = g["turn"].transform(lambda s: s.rolling(20).rank(pct=True))
    p["amount_ratio"] = g["amount"].transform(lambda s: s.rolling(5).mean() / s.rolling(20).mean())
    delta = g["close"].diff()
    up = delta.clip(lower=0).groupby(p["code"]).transform(lambda s: s.rolling(14).mean())
    dn = (-delta.clip(upper=0)).groupby(p["code"]).transform(lambda s: s.rolling(14).mean())
    p["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    mid = g["close"].transform(lambda s: s.rolling(20).mean())
    std = g["close"].transform(lambda s: s.rolling(20).std())
    p["boll_pos"] = (p["close"] - mid) / (2 * std.replace(0, np.nan))
    p["dd_20"] = p["close"] / g["close"].transform(lambda s: s.rolling(20).max()) - 1
    rng = (p["high"] - p["low"]).replace(0, np.nan)
    p["close_pos"] = (p["close"] - p["low"]) / rng
    p["gap_open"] = p["open"] / p["preclose"].replace(0, np.nan) - 1
    p["ind_rel_5"] = p["ret_5"] - p["ind_ret_5"]
    p["ind_rel_20"] = p["ret_20"] - p["ind_ret_20"]
    p["xrank_ret5"] = p.groupby("date")["ret_5"].rank(pct=True)
    p["xrank_vol"] = p.groupby("date")["vol_20"].rank(pct=True)
    p["xrank_turn"] = p.groupby("date")["turn_pctile"].rank(pct=True)
    p["xrank_rel5"] = p.groupby("date")["ind_rel_5"].rank(pct=True)
    gg = p.groupby("code")
    p["next_open"] = gg["open"].shift(-1)
    p["next_close"] = gg["close"].shift(-1)
    p["next_suspended"] = gg["suspended"].shift(-1)
    p["y_oc"] = p["next_close"] / p["next_open"] - 1
    p.loc[(p["next_suspended"] == 1) | p["next_open"].le(0) | p["next_close"].le(0), "y_oc"] = np.nan
    p["y_up"] = (p["y_oc"] > 0).astype(float).where(p["y_oc"].notna())
    p["pct_chg"] = pd.to_numeric(p["pct_chg"], errors="coerce")
    return p


def _fit_scan(p: pd.DataFrame):
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    trad = p[(p["suspended"] == 0) & (p["isST"] == 0)].copy()
    dates = sorted(trad["date"].unique())
    t_last = dates[-1]
    labelled = trad[trad["date"] >= pd.Timestamp(START_TRAIN)].dropna(subset=["y_oc", "y_up"])
    val_dates = sorted(labelled["date"].unique())[-60:]
    tr = labelled[labelled["date"] < val_dates[0]]
    va = labelled[labelled["date"].isin(val_dates)]
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3,
        min_child_samples=100, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0, n_jobs=8, verbose=-1,
        random_state=20261026,
    )
    model.fit(tr[FEATURES], tr["y_up"].astype(int))
    raw_va = model.predict_proba(va[FEATURES])[:, 1]
    cal = LogisticRegression(max_iter=1000).fit(raw_va.reshape(-1, 1), va["y_up"].astype(int))
    p_va = cal.predict_proba(raw_va.reshape(-1, 1))[:, 1]
    metrics = {
        "train_rows": int(len(tr)), "valid_rows": int(len(va)),
        "train_end": str(pd.Timestamp(tr["date"].max()).date()),
        "valid_start": str(pd.Timestamp(val_dates[0]).date()),
        "valid_end": str(pd.Timestamp(va["date"].max()).date()),
        "auc_raw": round(float(roc_auc_score(va["y_up"], raw_va)), 4),
        "brier_raw": round(float(brier_score_loss(va["y_up"], raw_va)), 4),
        "brier_cal": round(float(brier_score_loss(va["y_up"], p_va)), 4),
    }
    # exp_ret regressor (same split, strong regularisation)
    reg = lgb.LGBMRegressor(
        objective="huber", alpha=0.02, n_estimators=250, learning_rate=0.03,
        num_leaves=7, max_depth=3, min_child_samples=100, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0, n_jobs=8, verbose=-1,
        random_state=20261026,
    )
    reg.fit(tr[FEATURES], tr["y_oc"].clip(-0.1, 0.1))
    day = trad[trad["date"] == t_last].copy()
    if day.empty:
        raise RuntimeError("latest day empty")
    day["p_raw"] = model.predict_proba(day[FEATURES])[:, 1]
    day["p_up"] = cal.predict_proba(day["p_raw"].to_numpy().reshape(-1, 1))[:, 1]
    day["exp_oc"] = reg.predict(day[FEATURES])
    day["is_limit"] = ((day["pct_chg"] >= 9.5) & ~day["code"].str.startswith(("30", "301", "68"))) | ((day["pct_chg"] >= 19.5) & day["code"].str.startswith(("30", "301", "68")))
    day = day.sort_values("p_up", ascending=False)
    top = day[["code", "name", "industry_name", "p_up", "p_raw", "exp_oc", "pct_chg", "ret_5", "turn_pctile", "amount_ratio", "is_limit"]].head(25)
    metrics["n_day"] = int(len(day))
    metrics["n_ge_55"] = int((day["p_up"] >= 0.55).sum())
    metrics["max_p"] = round(float(day["p_up"].max()), 4)
    metrics["top"] = top.replace({np.nan: None}).to_dict(orient="records")
    return t_last, metrics, day


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cons = _cons()
    raw = _fetch_daily(cons)
    panel = _build_panel(cons, raw)
    t_last, metrics, day = _fit_scan(panel)
    registry = OUT / "registry.csv"
    row = pd.DataFrame([{
        "experiment_id": "EXP-20261026-EXPAND01",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "universe": "user:screenshot_top10+801737",
        "label": "T+1_open_to_close",
        "model": "LightGBM_L2+Platt",
        "features": f"{len(FEATURES)}_lowdim",
        "data_asof": str(t_last.date()),
        "cost_bps": "",
        "threshold": 0.55,
        "non_pit": "true",
        "live_signal": "false",
        "status": "EXPLORATORY",
        "auc_raw": metrics["auc_raw"],
        "brier_cal": metrics["brier_cal"],
        "max_p": metrics["max_p"],
        "n_ge_55": metrics["n_ge_55"],
    }])
    if registry.exists():
        old = pd.read_csv(registry)
        pd.concat([old, row], ignore_index=True).to_csv(registry, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(registry, index=False, encoding="utf-8-sig")
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "T": str(t_last.date()), "universe": "用户截图强势行业(按SW2名称映射) + 电池对照", "label": "T+1 open-to-close", "not_pit": True, "live_signal": False, "metrics": metrics, "top": metrics["top"]}
    (OUT / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# 扩池探索扫描（研究沙盒，不是实盘）",
        "",
        f"- T={t_last.date()}（本地最后完整日线；你截图是 10-26 盘中，时间不一致）",
        f"- 宇宙：用户截图里的强势 SW2 行业名称 + 电池对照；当前成分，**非 PIT**；候选 {metrics['n_day']} 只",
        "- 标签：**T+1 开盘 → T+1 收盘**（与次日开盘买、收盘卖一致），不是 close-to-close",
        "- 模型：低维行业/量价特征 + 强正则 LightGBM + 最近60日 Platt 校准；不是实盘门槛",
        f"- 验证：训练 {metrics['train_rows']} 行（至 {metrics['train_end']}），验证 {metrics['valid_rows']} 行（{metrics['valid_start']}~{metrics['valid_end']}），原始 AUC {metrics['auc_raw']}，Brier {metrics['brier_raw']} → 校准后 {metrics['brier_cal']}",
        f"- 全池最高概率：**{metrics['max_p']:.2%}**；≥55% 数量：**{metrics['n_ge_55']}**",
        "",
        "## Top 15",
        "",
        "| 代码 | 名称 | 行业 | p_up | 预期 OC | 当日涨幅 | 5日 | 换手分位 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["top"][:15]:
        lines.append(f"| {r['code']} | {r['name']} | {r['industry_name']} | {r['p_up']:.2%} | {r['exp_oc']:+.2%} | {r['pct_chg']:+.2f}% | {r['ret_5']:+.1%} | {r['turn_pctile']:.2f} |")
    lines += [
        "",
        "## 必须保留的口径",
        "",
        "- 这是**探索性扩池扫描**，不是 PIT 历史回测，也不是已验证的个股层；不写入 latest.json、不替换电池主线。",
        "- 当前成分股存在幸存者偏差；行业名单来自你 10-26 的截图，而特征只到 10-23，不能用它推断 10-26 的实时名单。",
        "- 55% 是分类概率，不等于收益保证；跨行业概率校准仍不充分，实盘前必须做 PIT/分层校准与交易成本验证。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("DONE " + json.dumps({"max_p": metrics["max_p"], "n_ge_55": metrics["n_ge_55"], "auc": metrics["auc_raw"], "valid_rows": metrics["valid_rows"]}, ensure_ascii=True))
    print("EXPANDED_SCAN_DONE", json.dumps({"max_p": metrics["max_p"], "n_ge_55": metrics["n_ge_55"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
