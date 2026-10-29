"""PAGE-LGBM-FWD-001：页内 LightGBM 的每日前瞻影子（只记录，不交易）。

- 训练：全部历史页内样本（每日涨幅前 20），日期 <= T-2（purge 1 天）
- 预测：T 日第一页 20 只，按 p_up 取 Top1
- 输出：data/research/page_shadow/<T>.json + reports/page_shadow/history.csv
- 冻结：同日重跑不覆盖首条；不写 latest.json、不改 55% 门槛
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.net_return_v31 import realizable  # noqa: E402
from src.eval.top1_audit import COST_SIDE, FEATURE_COLS, SENT_COLS  # noqa: E402
from src.report.daily import load_panel  # noqa: E402
from src.utils.config import PARQUET_DIR, ROOT  # noqa: E402

OUT = ROOT / "data" / "research" / "page_shadow"
REP = ROOT / "reports" / "page_shadow"
LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
           subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=8, verbose=-1)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def evaluate(hist_path: Path) -> None:
    """回填已到期的影子/生产可实现 2 日净收益；不覆盖已有值。"""
    if not hist_path.exists():
        return
    hist = pd.read_csv(hist_path, dtype={"shadow_code": str, "prod_code": str})
    for c in ("shadow_r2d", "shadow_net", "prod_r2d", "prod_net"):
        if c not in hist.columns:
            hist[c] = np.nan
    panel = load_panel()
    panel["code"] = panel["code"].astype(str)
    dates = sorted(panel["date"].unique())
    px = panel.set_index(["code", "date"])[["open", "high", "low", "close", "preclose", "suspended"]]
    n = 0
    for i, r in hist.iterrows():
        if pd.notna(r.get("shadow_net")):
            continue
        t = str(r["T"])
        rb = realizable(px, dates, str(r["shadow_code"]), t)
        if pd.notna(rb["r"]):
            hist.loc[i, "shadow_r2d"] = round(float(rb["r"]), 5)
            hist.loc[i, "shadow_net"] = round(float(rb["r"]) - 2 * COST_SIDE, 5)
            n += 1
        pb = realizable(px, dates, str(r["prod_code"]), t)
        if pd.notna(pb["r"]):
            hist.loc[i, "prod_r2d"] = round(float(pb["r"]), 5)
            hist.loc[i, "prod_net"] = round(float(pb["r"]) - 2 * COST_SIDE, 5)
    if n:
        hist.to_csv(hist_path, index=False, encoding="utf-8-sig")
    print(f"[page_shadow] 新核验 {n} 天", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    panel["code"] = panel["code"].astype(str)
    cons = set(pd.read_parquet(PARQUET_DIR / "cons_battery.parquet")["code"].astype(str))
    trad = panel[(panel["suspended"] == 0) & (panel["isST"] == 0) & (panel["code"].isin(cons))].copy()
    dates = sorted(panel["date"].unique())
    T = dates[-1]
    pos = {d: i for i, d in enumerate(dates)}
    feats = [c for c in FEATURE_COLS + SENT_COLS if c in trad.columns]

    labelled = trad.dropna(subset=["ret_1d"])
    train_days = [d for d in sorted(labelled["date"].unique()) if pos[d] <= pos[T] - 2]
    pages = []
    for d in train_days:
        g = labelled[labelled["date"] == d].sort_values("pct_chg", ascending=False).head(20)
        if len(g) >= 5:
            pages.append(g)
    tr = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()
    if len(tr) < 400:
        print("[page_shadow] 训练样本不足，跳过", flush=True)
        return
    model = LGBMClassifier(**LGB).fit(tr[feats], (tr["ret_1d"] > 0).astype(int))

    day = trad[trad["date"] == T].sort_values("pct_chg", ascending=False).head(20).copy()
    if day.empty:
        print("[page_shadow] 当日第一页为空，跳过", flush=True)
        return
    p = model.predict_proba(day[feats])[:, 1]
    day["p_up"] = p
    top = day.iloc[int(np.argmax(p))]
    latest = json.loads((ROOT / "reports" / "latest.json").read_text(encoding="utf-8"))
    prod = latest.get("short", {}) or {}

    target = OUT / f"{T}.json"
    if target.exists():
        evaluate(REP / "history.csv")
        print(f"[page_shadow] {T} 已存在，冻结不覆盖", flush=True)
        return
    payload = {
        "snapshot_date": T,
        "model": "PAGE-LGBM-FWD-001",
        "status": "SHADOW_ONLY",
        "is_broker_pit": False,
        "trained_rows": int(len(tr)),
        "trained_days": int(len(pages)),
        "train_end": str(train_days[-1]) if train_days else "",
        "top1": {"code": str(top["code"]), "name": str(top.get("name", "")), "p_up": round(float(top["p_up"]), 5)},
        "production": {"code": str(prod.get("code", "")), "p_up": prod.get("win_prob")},
        "page": [
            {"code": str(r["code"]), "name": str(r.get("name", "")), "pct_chg": float(r["pct_chg"]),
             "p_up": round(float(r["p_up"]), 5)}
            for _, r in day.iterrows()
        ],
        "hashes": {
            "panel": _sha(PARQUET_DIR / "panel.parquet"),
            "cons": _sha(PARQUET_DIR / "cons_battery.parquet"),
            "latest": _sha(ROOT / "reports" / "latest.json"),
        },
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    hist = REP / "history.csv"
    row = pd.DataFrame([{
        "T": T, "shadow_code": str(top["code"]), "shadow_p_up": round(float(top["p_up"]), 5),
        "prod_code": str(prod.get("code", "")), "prod_p_up": prod.get("win_prob"),
        "train_rows": len(tr), "train_days": len(pages),
    }])
    if hist.exists():
        old = pd.read_csv(hist, dtype={"shadow_code": str, "prod_code": str})
        old = pd.concat([old, row], ignore_index=True).drop_duplicates(subset=["T"], keep="first")
        old.to_csv(hist, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(hist, index=False, encoding="utf-8-sig")
    evaluate(hist)
    print(f"[page_shadow] {T} 影子 Top1={top['code']} p={top['p_up']:.1%} | 生产={prod.get('code')} | 训练 {len(tr)} 行", flush=True)
    print("PAGE_SHADOW_DONE")


if __name__ == "__main__":
    main()
