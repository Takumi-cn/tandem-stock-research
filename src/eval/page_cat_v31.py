"""PAGE-CAT-001（预注册）：第一页 20 只上的 CatBoost vs LightGBM vs 生产 Top1。

同一 120 日、同一 6×20 purged 回放、同一第一页、同一可交易口径。
只研究，不写 latest.json、不改 55% 门槛。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.net_return_v31 import realizable  # noqa: E402
from src.eval.top1_audit import COST_SIDE, COST_STRESS, FEATURE_COLS, SENT_COLS, boot_by_date  # noqa: E402
from src.report.daily import load_panel  # noqa: E402
from src.report.short_pick import TRADE_P  # noqa: E402
from src.utils.config import PARQUET_DIR, ROOT  # noqa: E402

OUT = ROOT / "reports" / "page_cat_v31"
N_DAYS, BLOCK, PAGE = 120, 20, 20
CAT = dict(iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=3,
           loss_function="Logloss", random_seed=0, verbose=False,
           thread_count=8, allow_writing_files=False)
LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
           subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=8, verbose=-1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = json.loads((OUT / "prereg.json").read_text(encoding="utf-8"))
    assert prereg["status"] == "PRE-REGISTERED"

    panel = load_panel()
    cons = set(pd.read_parquet(PARQUET_DIR / "cons_battery.parquet")["code"].astype(str))
    panel["code"] = panel["code"].astype(str)
    trad = panel[(panel["suspended"] == 0) & (panel["isST"] == 0) & (panel["code"].isin(cons))].copy()
    dates_all = sorted(panel["date"].unique())
    pos = {d: i for i, d in enumerate(dates_all)}
    hist_days = sorted(trad.dropna(subset=["ret_1d"])["date"].unique())
    test_dates = hist_days[-N_DAYS:]
    feats = [c for c in FEATURE_COLS + SENT_COLS if c in trad.columns]
    px = panel.set_index(["code", "date"])[["open", "high", "low", "close", "preclose", "suspended"]]

    def page_of(day: str) -> pd.DataFrame:
        g = trad[(trad["date"] == day)].dropna(subset=["ret_1d"])
        return g.sort_values("pct_chg", ascending=False).head(PAGE).copy()

    # 生产 Top1（审计明细，同日期）
    prod = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    prod = prod[prod["is_top1"]].set_index("date")
    # 与审计使用完全相同的 120 个日期，避免窗口错一天
    test_dates = sorted(set(prod.index))[-N_DAYS:]

    rows = []
    for start in range(0, len(test_dates), BLOCK):
        block = test_dates[start:start + BLOCK]
        p0 = pos[block[0]]
        # 用全部历史页内样本（标签终点 purge 1 天），不只用最近 120 日
        train_days = [d for d in hist_days if pos[d] <= p0 - 2]
        tr = pd.concat([page_of(d) for d in train_days], ignore_index=True) if train_days else pd.DataFrame()
        if len(tr) < 400:
            continue
        y = (tr["ret_1d"] > 0).astype(int)
        models = {
            "cat": CatBoostClassifier(**CAT).fit(tr[feats], y),
            "lgbm": LGBMClassifier(**LGB).fit(tr[feats], y),
        }
        for d in block:
            pr = prod.loc[d] if d in prod.index else None
            if pr is not None:
                rows.append({"date": d, "model": "prod", "code": str(pr["code"]),
                             "p_up": float(pr["p_up"]), "book": float(pr["book"]),
                             "r2d": pr["r_2d"], "status": "OK" if pd.notna(pr["r_2d"]) else "BLOCKED"})
            pg = page_of(d)
            if pg.empty:
                continue
            for tag, m in models.items():
                p = m.predict_proba(pg[feats])[:, 1]
                top = pg.iloc[int(np.argmax(p))]
                rb = realizable(px, dates_all, str(top["code"]), d)
                rows.append({
                    "date": d, "model": tag, "code": str(top["code"]),
                    "p_up": round(float(np.max(p)), 5), "book": top["ret_1d"],
                    "r2d": rb["r"], "status": rb["block"],
                })
        print(f"  块 {start // BLOCK + 1}/{(len(test_dates) + BLOCK - 1) // BLOCK} 完成", flush=True)

    df = pd.DataFrame(rows)
    df["net"] = (df["r2d"] - 2 * COST_SIDE).where(df["r2d"].notna(), 0.0)
    df["net25"] = (df["r2d"] - 2 * COST_STRESS).where(df["r2d"].notna(), 0.0)
    df.to_csv(OUT / "top1_days.csv", index=False, encoding="utf-8-sig")

    res = {}
    for tag in ("cat", "lgbm", "prod"):
        d = df[df["model"] == tag]
        traded = d[d["r2d"].notna()]
        ud = sorted(d["date"].unique())
        half = set(ud[: len(ud) // 2])
        first, second = d[d["date"].isin(half)], d[~d["date"].isin(half)]
        gate = d[d["p_up"] >= TRADE_P]
        res[tag] = {
            "days": int(len(d)),
            "trade_days": int(len(traded)),
            "opp_mean": round(float(d["net"].mean()), 5),
            "per_trade": round(float(traded["net"].mean()), 5) if len(traded) else None,
            "per_up": round(float((traded["net"] > 0).mean()), 4) if len(traded) else None,
            "book_hit": round(float((traded["book"] > 0).mean()), 4) if len(traded) else None,
            "first_half": round(float(first["net"].mean()), 5),
            "second_half": round(float(second["net"].mean()), 5),
            "gate_days": int(len(gate)),
            "gate_net": round(float(gate["net"].mean()), 5) if len(gate) else None,
            "stress25": round(float(d["net25"].mean()), 5),
            "per_boot": boot_by_date(traded.assign(v=traded["net"]), "v") if len(traded) else None,
        }
    cat, lgbm, prod_r = res["cat"], res["lgbm"], res["prod"]
    checks = {
        "cat_opp > lgbm 且 > prod": bool(cat["opp_mean"] > lgbm["opp_mean"] and cat["opp_mean"] > prod_r["opp_mean"]),
        "每笔>0 且 p<0.05": bool((cat["per_trade"] or -1) > 0 and cat["per_boot"] and cat["per_boot"]["p_le_0"] < 0.05),
        "前后半段均>0": bool(cat["first_half"] > 0 and cat["second_half"] > 0),
        "25bps>0": bool(cat["stress25"] > 0),
    }
    passed = all(checks.values())
    lines = [
        "# PAGE-CAT-001：第一页 CatBoost vs LightGBM vs 生产（预注册）",
        "",
        f"- 运行：{pd.Timestamp.now():%Y-%m-%d %H:%M}｜{len(df['date'].unique())} 个测试日｜6×20 purged",
        "- 每日第一页 20 只；label=次日收涨；Top1 按 p_up；T+1 open→T+2 close；14.5bps/边",
        "",
        "| 模型 | 交易天数 | 日均机会净收益 | 每笔净收益 | 每笔胜率 | 账面命中 | 前半段 | 后半段 | 25bps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, name in (("cat", "CatBoost"), ("lgbm", "LightGBM(页内)"), ("prod", "生产 Top1")):
        r = res[tag]
        lines.append(
            f"| {name} | {r['trade_days']} | {r['opp_mean']:+.3%} | {(r['per_trade'] or 0):+.3%} | "
            f"{(r['per_up'] or 0):.1%} | {(r['book_hit'] or 0):.1%} | {r['first_half']:+.3%} | "
            f"{r['second_half']:+.3%} | {r['stress25']:+.3%} |"
        )
    lines += [
        "",
        f"- CatBoost 每笔 bootstrap：95%CI [{cat['per_boot']['ci95'][0]:+.3%},{cat['per_boot']['ci95'][1]:+.3%}]，p(≤0)={cat['per_boot']['p_le_0']:.3f}" if cat["per_boot"] else "- CatBoost 无交易",
        "",
        "## 停止条件",
        "",
    ] + [f"- {'✅' if v else '❌'} {k}" for k, v in checks.items()] + [
        "",
        f"**判定：{'PASS（仅研究线索，不自动上线）' if passed else 'FAIL（保持生产模型不变）'}**",
        "",
        "## 局限", "",
        "- 120 日、每块约 2,000 个页内样本；树模型对超参敏感，已固定不调；",
        "- 生产 Top1 训练集为全池，页内模型为页内样本，二者不是严格同分布对照；",
        "- 不写 latest.json、不替换 55% 门槛。",
        "", "## 免责声明", "", "研究实验输出，不构成投资建议。",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    out = {"exp_id": "PAGE-CAT-001", "res": res, "checks": checks, "passed": passed}
    (OUT / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    with open(ROOT / "reports" / "experiment_registry.csv", "a", encoding="utf-8") as f:
        f.write(f"PAGE-CAT-001,{pd.Timestamp.now():%Y-%m-%d},电池个股短线第一页,AUDIT-COMPLETE,"
                f"{'PASS' if passed else 'FAIL'};cat_opp={cat['opp_mean']};lgbm_opp={lgbm['opp_mean']};"
                f"prod_opp={prod_r['opp_mean']}\n")
    print(json.dumps({"cat": cat, "lgbm": lgbm, "prod": prod_r, "checks": checks, "passed": passed},
                     ensure_ascii=False, indent=1, default=float))
    print("PAGE_CAT_DONE")


if __name__ == "__main__":
    main()
