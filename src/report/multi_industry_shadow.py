"""MULTI-IND-001（预注册）：12 个 SW2 行业的横截面 Top1 影子。

只研究，不写 latest.json、不改 55% 门槛。
数据：已有 6 行业 198 只 + 新增 6 行业（数字媒体/出版/影视院线/电机/白酒/农商行）；
当前成分、非 PIT；日线 2022-12-01 起。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.net_return_v31 import realizable  # noqa: E402
from src.eval.top1_audit import COST_SIDE, COST_STRESS, boot_by_date  # noqa: E402
from src.report.expanded_scan import FEATURES, START_DATA, _build_panel  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

DATA = ROOT / "data" / "multi"
OUT = ROOT / "reports" / "multi_industry_shadow"
RES = ROOT / "data" / "research" / "multi_industry_shadow"
NEW = [("801767", "数字媒体"), ("801769", "出版"), ("801766", "影视院线"),
       ("801731", "电机Ⅱ"), ("801125", "白酒Ⅱ"), ("801785", "农商行Ⅱ")]
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,isST"
LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3,
           min_child_samples=100, subsample=0.8, colsample_bytree=0.7,
           reg_alpha=1.0, reg_lambda=5.0, n_jobs=8, verbose=-1, random_state=20261026)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S") + " " + msg, flush=True)


def load_cons() -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "cons.parquet"
    if path.exists():
        return pd.read_parquet(path)
    base = pd.read_parquet(ROOT / "data" / "expanded" / "constituents.parquet")
    meta = pd.read_parquet(ROOT / "data" / "industry" / "sw2_meta.parquet")
    meta["code"] = meta["code"].astype(str)
    parent = meta.set_index("code")["parent"].to_dict()
    import akshare as ak
    rows = [base]
    for code, name in NEW:
        df = None
        for attempt in range(3):
            try:
                df = ak.index_component_sw(symbol=code)
                break
            except Exception as e:  # noqa: BLE001
                log(f"cons fail {code} {type(e).__name__}")
                time.sleep(1.5 + attempt)
        if df is None or len(df) == 0:
            log(f"cons skip {code} {name}")
            continue
        df = df.rename(columns={"证券代码": "code", "证券简称": "name"})
        if "code" not in df.columns:
            df = df.rename(columns={df.columns[1]: "code", df.columns[2]: "name"})
        df = df.copy()
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["name"] = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
        df["industry_code"] = code
        df["industry_name"] = name
        df["parent"] = parent.get(code, "")
        rows.append(df[["code", "name", "industry_code", "industry_name", "parent"]])
        log(f"cons {code} {name}: {len(df)}")
    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)
    out.to_parquet(path, index=False)
    return out


def fetch_raw(cons: pd.DataFrame) -> pd.DataFrame:
    path = DATA / "raw_daily.parquet"
    old = pd.read_parquet(ROOT / "data" / "expanded" / "raw_daily.parquet")
    if path.exists():
        old = pd.read_parquet(path)
    old["date"] = pd.to_datetime(old["date"])
    old["code"] = old["code"].astype(str)
    import baostock as bs
    lg = bs.login()
    assert lg.error_code == "0", lg.error_msg
    today = pd.Timestamp.today().normalize()
    frames, fail = [], []
    for i, code in enumerate(cons["code"].astype(str), 1):
        g = old[old["code"] == code]
        start = (g["date"].max() + pd.Timedelta(days=1)) if len(g) else pd.Timestamp(START_DATA)
        if start > today:
            continue
        mkt = "sh" if code.startswith(("6", "9")) else "sz"
        try:
            rs = bs.query_history_k_data_plus(f"{mkt}.{code}", FIELDS, start_date=start.strftime("%Y-%m-%d"),
                                              end_date=today.strftime("%Y-%m-%d"), frequency="d", adjustflag="2")
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df = pd.DataFrame(rows, columns=FIELDS.split(","))
                df["code"] = code
                frames.append(df)
        except Exception as e:  # noqa: BLE001
            fail.append(code)
            log(f"fetch fail {code} {type(e).__name__}")
        if i % 50 == 0:
            log(f"fetch {i}/{len(cons)} fail={len(fail)}")
        time.sleep(0.08)
    bs.logout()
    if frames:
        new = pd.concat(frames, ignore_index=True)
        new["date"] = pd.to_datetime(new["date"])
        out = pd.concat([old, new], ignore_index=True)
        out = out.drop_duplicates(subset=["date", "code"], keep="last").sort_values(["code", "date"]).reset_index(drop=True)
        for c in ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "isST"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out.to_parquet(path, index=False)
    else:
        out = old
    log(f"raw rows={len(out):,} codes={out['code'].nunique()} fail={len(fail)}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    prereg = json.loads((OUT / "prereg.json").read_text(encoding="utf-8"))
    assert prereg["status"] == "PRE-REGISTERED"
    # 同日重跑：影子已存在则跳过，避免重复 3 分钟回放
    try:
        latest_now = json.loads((ROOT / "reports" / "latest.json").read_text(encoding="utf-8"))
        t0 = str(latest_now.get("T", ""))
        if t0 and (RES / f"{t0}.json").exists():
            print(f"[multi] {t0} 多行业影子已存在，跳过", flush=True)
            return
    except Exception:  # noqa: BLE001
        pass
    cons = load_cons()
    raw = fetch_raw(cons)
    log(f"build features: {len(cons)} codes")
    panel = _build_panel(cons, raw)
    panel["date"] = pd.to_datetime(panel["date"])
    trad = panel[(panel["suspended"] == 0) & (panel["isST"] == 0)].copy()
    # 关键：标签必须是 T→T+1 的次日收益，不能用当日 ret_1（否则 100% 泄漏）
    trad["y_book_ret"] = trad.groupby("code")["close"].shift(-1) / trad["close"] - 1
    trad["y_book"] = (trad["y_book_ret"] > 0).astype(float).where(trad["y_book_ret"].notna())
    lab = trad.dropna(subset=["y_book"])
    dates_all = sorted(panel["date"].unique())
    px = panel.set_index(["code", "date"])[["open", "high", "low", "close", "preclose", "suspended"]]
    prod = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    prod = prod[prod["is_top1"]].set_index("date")
    # 与审计完全相同的 120 个日期，避免生产对照错一天
    test_dates = [pd.Timestamp(x) for x in sorted(prod.index)[-120:]]

    rows = []
    for start in range(0, len(test_dates), 20):
        block = test_dates[start:start + 20]
        tr = lab[lab["date"] <= block[0] - pd.Timedelta(days=2)]
        if len(tr) < 5000:
            continue
        model = LGBMClassifier(**LGB).fit(tr[FEATURES], tr["y_book"].astype(int))
        for d in block:
            day = trad[trad["date"] == d].copy()
            if day.empty:
                continue
            p = model.predict_proba(day[FEATURES])[:, 1]
            top = day.iloc[int(np.argmax(p))]
            rb = realizable(px, dates_all, str(top["code"]), d)
            rows.append({"date": d, "code": str(top["code"]), "name": str(top.get("name", "")),
                         "industry": str(top.get("industry_name", "")), "p_up": round(float(np.max(p)), 5),
                         "book": float(top["y_book_ret"]), "r2d": rb["r"], "status": rb["block"]})
        log(f"block {start // 20 + 1} done")
    df = pd.DataFrame(rows)
    df["net"] = (df["r2d"] - 2 * COST_SIDE).where(df["r2d"].notna(), 0.0)
    df["net25"] = (df["r2d"] - 2 * COST_STRESS).where(df["r2d"].notna(), 0.0)
    df.to_csv(OUT / "top1_days.csv", index=False, encoding="utf-8-sig")

    def stats(d: pd.DataFrame) -> dict:
        tr = d[d["r2d"].notna()]
        ud = sorted(d["date"].unique())
        half = set(ud[: len(ud) // 2])
        return {"days": len(d), "trades": len(tr),
                "opp": round(float(d["net"].mean()), 5),
                "per": round(float(tr["net"].mean()), 5) if len(tr) else None,
                "up": round(float((tr["net"] > 0).mean()), 4) if len(tr) else None,
                "hit": round(float((tr["book"] > 0).mean()), 4) if len(tr) else None,
                "first": round(float(d[d["date"].isin(half)]["net"].mean()), 5),
                "second": round(float(d[~d["date"].isin(half)]["net"].mean()), 5),
                "stress25": round(float(d["net25"].mean()), 5),
                "boot": boot_by_date(tr.assign(v=tr["net"]), "v") if len(tr) else None}

    multi = stats(df)
    pdays = []
    for d in test_dates:
        key = pd.Timestamp(d).strftime("%Y-%m-%d")
        if key in prod.index:
            r = prod.loc[key]
            pdays.append({"date": d, "net": (r["r_2d"] - 2 * COST_SIDE) if pd.notna(r["r_2d"]) else 0.0,
                          "r2d": r["r_2d"], "book": r["book"]})
    pdf = pd.DataFrame(pdays)
    prodst = {"days": len(pdf), "trades": int(pdf["r2d"].notna().sum()) if len(pdf) else 0,
              "opp": round(float(pdf["net"].mean()), 5) if len(pdf) else None,
              "per": round(float(pdf[pdf["r2d"].notna()]["net"].mean()), 5) if len(pdf) else None,
              "hit": round(float((pdf["book"] > 0).mean()), 4) if len(pdf) else None}

    checks = {
        "multi_opp > prod_opp": bool(multi["opp"] > (prodst["opp"] or -9)),
        "每笔>0 且 p<0.05": bool((multi["per"] or -1) > 0 and multi["boot"] and multi["boot"]["p_le_0"] < 0.05),
        "前后半段均>0": bool(multi["first"] > 0 and multi["second"] > 0),
        "25bps>0": bool(multi["stress25"] > 0),
    }
    passed = all(checks.values())
    lines = [
        "# MULTI-IND-001：12 行业横截面 Top1 影子（预注册）",
        "",
        f"- 运行：{pd.Timestamp.now():%Y-%m-%d %H:%M}｜{len(cons)} 只 / {cons['industry_name'].nunique()} 行业｜120 日 6×20 purged",
        "- 当前成分、非 PIT；label=次日收涨；Top1 按 p_up；T+1 open→T+2 close；14.5bps/边",
        "",
        "| 组合 | 交易天数 | 日均机会净收益 | 每笔净收益 | 每笔胜率 | 账面命中 | 前半段 | 后半段 | 25bps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| 多行业 Top1 | {multi['trades']} | {multi['opp']:+.3%} | {(multi['per'] or 0):+.3%} | "
        f"{(multi['up'] or 0):.1%} | {(multi['hit'] or 0):.1%} | {multi['first']:+.3%} | {multi['second']:+.3%} | {multi['stress25']:+.3%} |",
        f"| 生产电池 Top1 | {prodst['trades']} | {(prodst['opp'] or 0):+.3%} | {(prodst['per'] or 0):+.3%} | — | "
        f"{(prodst['hit'] or 0):.1%} | — | — | — |",
        "",
        f"- 多行业每笔 bootstrap：95%CI [{multi['boot']['ci95'][0]:+.3%},{multi['boot']['ci95'][1]:+.3%}]，p(≤0)={multi['boot']['p_le_0']:.3f}" if multi["boot"] else "- 无交易",
        "",
        "## 停止条件",
        "",
    ] + [f"- {'✅' if v else '❌'} {k}" for k, v in checks.items()] + [
        "", f"**判定：{'PASS（仅研究线索）' if passed else 'FAIL（保持电池主线）'}**", "",
        "## 局限", "",
        "- 当前成分、非 PIT；12 行业不是全市场；样本 120 日；",
        "- 不写 latest.json、不改 55%、不自动交易。",
        "", "## 免责声明", "", "研究实验输出，不构成投资建议。",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    res = {"exp_id": "MULTI-IND-001", "n_codes": len(cons), "n_industries": int(cons["industry_name"].nunique()),
           "multi": multi, "prod": prodst, "checks": checks, "passed": passed}
    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    # 当日影子快照（不写 latest；同日冻结首条）
    T = pd.Timestamp(dates_all[-1])
    tr = lab[lab["date"] <= T - pd.Timedelta(days=2)]
    model = LGBMClassifier(**LGB).fit(tr[FEATURES], tr["y_book"].astype(int))
    day = trad[trad["date"] == T].copy()
    p = model.predict_proba(day[FEATURES])[:, 1]
    top = day.iloc[int(np.argmax(p))]
    latest = json.loads((ROOT / "reports" / "latest.json").read_text(encoding="utf-8"))
    payload = {"snapshot_date": str(T.date()), "model": "MULTI-IND-001", "status": "SHADOW_ONLY",
               "top1": {"code": str(top["code"]), "name": str(top.get("name", "")),
                        "industry": str(top.get("industry_name", "")), "p_up": round(float(np.max(p)), 5)},
               "production": {"code": latest.get("short", {}).get("code"), "p_up": latest.get("short", {}).get("win_prob")}}
    target = RES / f"{T.date()}.json"
    if not target.exists():
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(f"MULTI_SHADOW_TOP1 {payload['top1']}")
    print("MULTI_IND_DONE")


if __name__ == "__main__":
    main()
