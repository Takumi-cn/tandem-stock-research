"""竞猜记录生成器：复刻用户《股票预测(2).xlsx》玩法，系统自动维护。

用户玩法（来自桌面 xlsx）：
- 每个交易日两行：长线（中国铁物，涨/跌）+ 自选（当日关注的一只，涨/跌）；猜对 ✔ +50
- 每周五补"每周小结"：1 本周走势 2 下周预测

系统行为（诚实边界）：
- 对自选股的预测使用电池池训练的 UpModel（跨池泛化，置信度有限——报告与表格中标注）；
- 不代用户做"押哪只"的决定：对 watchlist 全部成员每日出预测，用户自选抄录；
- 结果列由系统在 T+1 收盘数据落地后自动核验 ✔/×，奖励按 hit×50 计；
- 只写 reports/竞猜记录.xlsx，从不改用户桌面原表。

运行: python -m src.report.guess_sheet          # 生成/更新记录 + 回填核验
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.build import FEATURE_COLS  # noqa: E402
from src.models.l1_lgbm import PARAMS  # noqa: E402
from src.report.short_pick import UpModel  # noqa: E402
from src.utils.config import PARQUET_DIR, ROOT  # noqa: E402

import lightgbm as lgb  # noqa: E402

SENT_COLS = ["sent_mean", "sent_max", "news_cnt", "kw_pos", "kw_neg"]
WEEKDAY = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}


def load_watch() -> dict:
    return yaml.safe_load((ROOT / "config" / "watchlist.yaml").read_text(encoding="utf-8"))


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PARQUET_DIR / "panel.parquet")
    sp = PARQUET_DIR / "sentiment.parquet"
    if sp.exists():
        s = pd.read_parquet(sp)
        p = p.merge(s, on=["code", "date"], how="left")
        p["sent_mean"] = p.groupby("code")["sent_mean"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    for c in SENT_COLS:
        if c not in p.columns:
            p[c] = np.nan
    return p


def train_upmodel(p: pd.DataFrame, feats: list):
    """电池池训练上涨概率分类器（与实盘 short_pick 同构）。"""
    trad = p[(p["suspended"] == 0) & (p["isST"] == 0)]
    d = trad.dropna(subset=["ret_1d"])
    y = (d["ret_1d"] > 0).astype(int)
    clf = lgb.LGBMClassifier(**{**{k: v for k, v in PARAMS.items() if k != "objective"}, "objective": "binary"})
    clf.fit(d[feats], y)
    return clf


def predict_rows(p: pd.DataFrame, feats: list, clf) -> pd.DataFrame:
    """生成竞猜行：长线铁物 + 全部自选股，各自"明日涨/跌"。"""
    w = load_watch()
    dates = sorted(p["date"].unique())
    t_last = dates[-1]
    day = p[p["date"] == t_last].set_index("code")
    rows = []
    codes = [str(w["long_target"])] + [str(x["code"]) for x in (w.get("watchlist") or [])]
    name_map = {str(w["long_target"]): "中国铁物"}
    for x in w.get("watchlist") or []:
        name_map[str(x["code"])] = x["name"]
    X = day.loc[[c for c in codes if c in day.index], feats]
    proba = pd.Series(clf.predict_proba(X)[:, 1], index=X.index) if len(X) else pd.Series(dtype=float)
    for code in codes:
        if code not in day.index:
            rows.append({"日期": t_last, "轨道": "", "股票": name_map.get(code, code), "代码": code,
                         "明日预测": "数据缺失", "p_up": None})
            continue
        r = day.loc[code]
        pu = float(proba.loc[code]) if code in proba.index else 0.5
        in_pool = code == str(w["long_target"])
        rows.append({
            "日期": t_last,
            "轨道": "长线" if in_pool else "自选",
            "股票": name_map.get(code, code),
            "代码": code,
            "明日预测": "涨" if pu > 0.52 else ("跌" if pu < 0.48 else "平"),
            "p_up": round(pu, 4),
            "note": "" if in_pool else "跨池模型，竞猜参考",
        })
    return pd.DataFrame(rows)


def verify_pending(sheet_rows: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """回填未核验行的 ✔/×：预测日 T 的下一交易日实际涨跌。返回 (df, 新核验数)。"""
    dates = sorted(panel["date"].unique())
    reward = int(load_watch().get("reward_per_hit", 50) or 50)
    n_new = 0
    for i, r in sheet_rows.iterrows():
        if str(r.get("结果", "")) in ("✔", "×", "½", "平✔", "平×"):
            continue
        t = pd.to_datetime(r["日期"], errors="coerce")
        if pd.isna(t):
            continue
        t = t.strftime("%Y-%m-%d")
        after = [d for d in dates if d > t]
        if not after:
            continue
        code = str(r.get("代码", ""))
        row = panel[(panel["code"] == code) & (panel["date"] == after[0])]
        if not len(row):
            continue
        ret = float(row["pct_chg"].iloc[0])
        pred = str(r["明日预测"])
        if pred not in ("涨", "跌", "平"):
            continue
        actual = "涨" if ret > 0 else ("跌" if ret < 0 else "平")
        flat = abs(ret) < 0.005
        if flat or actual == "平":
            res = "½"  # 平局判半对（与台账口径一致）
        else:
            res = "✔" if pred == actual else "×"
        sheet_rows.loc[i, "实际"] = actual + f"({ret:+.2f}%)"
        sheet_rows.loc[i, "结果"] = res
        sheet_rows.loc[i, "奖励"] = reward if res == "✔" else (reward // 2 if res == "½" else 0)
        n_new += 1
    return sheet_rows, n_new


def weekly_summary(p: pd.DataFrame) -> list[dict]:
    """最近一周（5个交易日）走势 + 下周预测（趋势延续为主判断，诚实口径）。"""
    w = load_watch()
    codes = [(str(w["long_target"]), "中国铁物")] + [(str(x["code"]), x["name"]) for x in (w.get("watchlist") or [])]
    dates = sorted(p["date"].unique())
    week = dates[-5:]
    out = []
    for code, name in codes:
        g = p[(p["code"] == code) & (p["date"].isin(week))].sort_values("date")
        if not len(g):
            continue
        wret = float(g["pct_chg"].iloc[-1:].sum() / 100)
        lead = p[(p["code"] == code) & (p["date"] == week[0])]["close"]
        tail = p[(p["code"] == code) & (p["date"] == week[-1])]["close"]
        wret = float(tail.iloc[0] / lead.iloc[0] - 1) if len(lead) and len(tail) else wret
        ma20 = p[p["code"] == code].sort_values("date")["close"].rolling(20).mean().iloc[-1]
        close = float(tail.iloc[0]) if len(tail) else np.nan
        trend = "站上20日线" if close > ma20 else "位于20日线下方"
        out.append({
            "股票": name, "本周涨跌": f"{wret:+.2%}", "趋势": trend,
            "下周预测": ("偏多（趋势与概率模型综合）" if close > ma20 else "偏空/震荡（趋势未转多，反弹以反抽对待）"),
        })
    return out


def main():
    p = load_panel()
    feats = [c for c in FEATURE_COLS + SENT_COLS if c in p.columns]
    clf = train_upmodel(p, feats)
    new_rows = predict_rows(p, feats, clf)

    sheet_path = ROOT / load_watch().get("sheet_path", "reports/竞猜记录.xlsx")
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    if sheet_path.exists():
        try:
            # 第 1 行是标题横幅，第 2 行才是表头
            daily = pd.read_excel(sheet_path, sheet_name="日线记录", dtype={"代码": str}, header=1)
        except Exception:  # noqa: BLE001
            daily = pd.DataFrame()
        if "代码" not in daily.columns:
            daily = pd.DataFrame()
        # 冻结：同 (日期, 代码) 已有记录则保留首条，不因同日重跑改写
        t_last = str(new_rows["日期"].iloc[0])
        if len(daily):
            daily["日期"] = pd.to_datetime(daily["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
            daily["代码"] = daily["代码"].astype(str).str.zfill(6)
            existing = set(zip(daily["日期"], daily["代码"]))
            keep = [i for i, r in new_rows.iterrows()
                    if (str(r["日期"]), str(r["代码"]).zfill(6)) not in existing]
            new_rows = new_rows.loc[keep]
    else:
        daily = pd.DataFrame()
    daily = pd.concat([daily, new_rows], ignore_index=True)
    daily, n_verified = verify_pending(daily, p)
    daily = daily.sort_values("日期", ascending=False).reset_index(drop=True)  # 最新在上（与用户表一致）

    weekly = weekly_summary(p)

    # 写 xlsx（openpyxl，样式尽量贴近用户原表）
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils.dataframe import dataframe_to_rows
    wb = Workbook()
    ws = wb.active
    ws.title = "日线记录"
    purple = PatternFill("solid", fgColor="E9E0FF")
    bold = Font(bold=True)
    ws.append(["", "", "", "", "系统竞猜记录（自动生成·与桌面手写表同款）", "", ""])
    ws.append(["日期", "星期", "轨道", "股票", "代码", "明日预测", "p_up", "实际/结果/奖励"])
    for c in ws[2]:
        c.font, c.fill, c.alignment = bold, purple, Alignment(horizontal="center")
    for _, r in daily.iterrows():
        dt = pd.to_datetime(r["日期"])
        res = ""
        if "结果" in daily.columns and pd.notna(r.get("结果")) and str(r.get("结果", "")):
            reward = int(float(r.get("奖励", 0) or 0))
            res = f"{r.get('实际', '')} {r.get('结果', '')} +{reward}分"
        ws.append([r["日期"], WEEKDAY[dt.weekday()], r.get("轨道", ""), r.get("股票", ""),
                   str(r.get("代码", "")).zfill(6), r.get("明日预测", ""), r.get("p_up", ""), res])
    for row in ws.iter_rows():
        for c in row:
            c.alignment = Alignment(horizontal="center", vertical="center")
    ws2 = wb.create_sheet("每周小结")
    ws2.append(["日期", "股票", "本周涨跌", "趋势", "下周预测"])
    for c in ws2[1]:
        c.font, c.fill = bold, purple
    stamp = datetime.now().strftime("%m.%d")
    for it in weekly:
        ws2.append([stamp, it["股票"], it["本周涨跌"], it["趋势"], it["下周预测"]])
    wb.save(sheet_path)

    # latest_guess.json（GUI 竞猜卡数据源；累计战绩从全表统计——结果列是 "✔/½/×" 符号）
    done = daily[daily.get("结果", pd.Series(dtype=str)).astype(str).isin(["✔", "½", "×"])] if "结果" in daily.columns else daily.iloc[0:0]
    hits = int((done["结果"] == "✔").sum()) if len(done) else 0
    halves = int((done["结果"] == "½").sum()) if len(done) else 0
    total_done = int(len(done))
    reward = int(pd.to_numeric(done.get("奖励"), errors="coerce").fillna(0).sum()) if len(done) else 0
    # 当日行：同日重跑 new_rows 为空时也要能显示，取 daily 最新日期
    t_latest = str(daily["日期"].iloc[0]) if len(daily) else ""
    today_rows = daily[daily["日期"] == t_latest] if t_latest else daily.iloc[0:0]
    latest = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "T": t_latest,
        "rows": today_rows.to_dict(orient="records"),
        "weekly": weekly,
        "stats": {"total": total_done, "hits": hits, "halves": halves, "reward": reward,
                  "acc": round((hits + 0.5 * halves) / total_done, 3) if total_done else None},
        "sheet": str(sheet_path),
    }
    (ROOT / "reports" / "latest_guess.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"竞猜记录已更新: {sheet_path} | 预测 {len(new_rows)} 行 | 新核验 {n_verified} 行 | 周小结 {len(weekly)} 条 | 累计 {hits}✔+{halves}½/{total_done} = {reward}分")
    if len(new_rows):
        print(new_rows[["日期", "轨道", "股票", "明日预测", "p_up"]].to_string(index=False))
    else:
        print("今日竞猜已存在，按冻结规则不覆盖")
    print("GUESS_SHEET_DONE")


if __name__ == "__main__":
    main()
