"""CONFORM-ADAPT-001: rolling-window adaptive conformal abstention.

Pre-registered, read-only research.  One fixed rule: use the most recent 60
Top1 days with realized book labels, alpha=0.10, and trade only when the
conformal set is the singleton {up}.  No alpha/warmup/window search.

Run:
  .venv\\Scripts\\python.exe -m src.eval.conform_adaptive_v31
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.top1_audit import COST_SIDE, COST_STRESS, boot_by_date  # noqa: E402
from src.report.short_pick import TRADE_P  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

OUT = ROOT / "reports" / "conform_adaptive_v31"
ALPHA = 0.10
WINDOW = 60
MIN_CAL = 30


def ensure_prereg() -> dict:
    p = OUT / "prereg.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    obj = {
        "exp_id": "CONFORM-ADAPT-001",
        "registered_at": "2026-10-28",
        "status": "PRE-REGISTERED",
        "hypothesis": "A rolling 60-day split-conformal calibration set adapts to regime changes and avoids the static CONFORM-001 over-abstention, while preserving coverage and improving daily opportunity net return versus B1.",
        "base_experiment": "CONFORM-001",
        "data": "TOP1-AUDIT-002 120 Top1 days; same book/r2d/buy_block/cost",
        "method": "score(y)=1-p_y; calibration = last 60 prior Top1 days with book; q = ceil((n+1)(1-alpha))/n empirical quantile; C={y:score<=q}; trade only C={up} and tradable",
        "alpha": ALPHA, "window": WINDOW, "min_calibration_days": MIN_CAL,
        "cost": "14.5bps/side; stress 25bps",
        "primary_metric": "daily opportunity net return (no trade=0) vs B1 p>=0.55",
        "stop_rules": [
            "adaptive daily opportunity net > B1",
            "per-trade net > 0 and bootstrap p(<=0) < 0.05",
            "both chronological halves per-trade net > 0",
            "25bps daily opportunity net > 0",
            "empirical coverage >= 0.85",
        ],
        "governance": "any failure = FAIL; no alpha/window/warmup search; no threshold change; no latest.json; no production",
        "production_policy": {"latest_json": "READ_ONLY", "threshold_55": "READ_ONLY", "stock_layer": "SHADOW_ONLY"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return obj


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = ensure_prereg()
    assert prereg["alpha"] == ALPHA and prereg["window"] == WINDOW
    t = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    t["buy_block"] = t["buy_block"].fillna("")
    top1 = t[t["is_top1"]].sort_values("date").reset_index(drop=True)
    rows = []
    for i, r in top1.iterrows():
        prior = top1.iloc[:i]
        prior = prior[prior["book"].notna()].tail(WINDOW)
        set1 = set0 = False
        q = np.nan
        if len(prior) >= MIN_CAL:
            scores = np.where(prior["book"] > 0, 1 - prior["p_up"], prior["p_up"])
            level = min(1.0, math.ceil((len(prior) + 1) * (1 - ALPHA)) / len(prior))
            q = float(np.quantile(scores, level, method="higher"))
            set1 = (1 - float(r["p_up"])) <= q
            set0 = float(r["p_up"]) <= q
        tradable = r["buy_block"] == "" and pd.notna(r["r_2d"])
        trade = bool(set1 and not set0 and tradable and len(prior) >= MIN_CAL)
        b1 = bool(float(r["p_up"]) >= TRADE_P and tradable)
        rows.append({
            "date": r["date"], "code": r["code"], "p_up": float(r["p_up"]),
            "book": r["book"], "r2d": r["r_2d"], "buy_block": r["buy_block"],
            "cal_n": len(prior), "q": q, "set_up": set1, "set_down": set0,
            "singleton_up": bool(set1 and not set0), "empty": bool(not set1 and not set0),
            "covered": bool((r["book"] > 0 and set1) or (r["book"] <= 0 and set0)) if pd.notna(r["book"]) and len(prior) >= MIN_CAL else None,
            "trade": trade, "b1": b1,
            "net": (r["r_2d"] - 2 * COST_SIDE) if trade and pd.notna(r["r_2d"]) else 0.0,
            "net25": (r["r_2d"] - 2 * COST_STRESS) if trade and pd.notna(r["r_2d"]) else 0.0,
            "b1_net": (r["r_2d"] - 2 * COST_SIDE) if b1 and pd.notna(r["r_2d"]) else 0.0,
            "b1_net25": (r["r_2d"] - 2 * COST_STRESS) if b1 and pd.notna(r["r_2d"]) else 0.0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "days.csv", index=False, encoding="utf-8-sig")
    ev = df[df["covered"].notna()]
    cov = float(ev["covered"].mean()) if len(ev) else float("nan")
    single = float(ev["singleton_up"].mean()) if len(ev) else float("nan")
    empty = float(ev["empty"].mean()) if len(ev) else float("nan")

    def block(flag: str, net: str) -> dict:
        mask = df[flag]
        r = df.loc[mask, net]
        return {"trade_days": int(len(r)), "opp_mean": float(df[net].mean()),
                "per_trade_mean": float(r.mean()) if len(r) else None,
                "per_trade_median": float(r.median()) if len(r) else None,
                "per_trade_up": float((r > 0).mean()) if len(r) else None,
                "boot": boot_by_date(df.assign(v=df[net]), "v"),
                "per_boot": boot_by_date(df[mask].assign(v=r), "v") if len(r) else None,
                "stress25": float(df[net.replace("net", "net25")].mean())}
    conf, b1 = block("trade", "net"), block("b1", "b1_net")
    df["diff"] = df["net"] - df["b1_net"]
    paired = boot_by_date(df, "diff")
    ud = sorted(df["date"].unique()); half = set(ud[: len(ud) // 2])
    halves = {}
    for tag, sub in (("first", df[df["date"].isin(half)]), ("second", df[~df["date"].isin(half)])):
        r = sub.loc[sub["trade"], "net"]; r1 = sub.loc[sub["b1"], "b1_net"]
        halves[tag] = {"days": len(sub), "conf_trades": len(r),
                       "conf_per_trade": float(r.mean()) if len(r) else None,
                       "conf_opp": float(sub["net"].mean()), "b1_opp": float(sub["b1_net"].mean())}
    checks = {
        "adaptive_opp > B1_opp": bool(conf["opp_mean"] > b1["opp_mean"]),
        "per_trade>0 and p<0.05": bool((conf["per_trade_mean"] or -1) > 0 and conf["per_boot"] and conf["per_boot"]["p_le_0"] < 0.05),
        "both_halves_per_trade>0": bool((halves["first"]["conf_per_trade"] or -1) > 0 and (halves["second"]["conf_per_trade"] or -1) > 0),
        "25bps>0": bool(conf["stress25"] > 0),
        "coverage>=0.85": bool(cov >= 0.85),
    }
    passed = all(checks.values())
    result = {"exp_id": "CONFORM-ADAPT-001", "alpha": ALPHA, "window": WINDOW,
              "coverage": cov, "singleton": single, "empty": empty,
              "adaptive": conf, "b1": b1, "paired": paired, "halves": halves,
              "checks": checks, "passed": passed}
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    lines = [
        "# CONFORM-ADAPT-001：滚动 60 日自适应保形弃权", "",
        f"- 运行：{pd.Timestamp.now():%Y-%m-%d %H:%M}｜120 天｜alpha=10%｜窗口 60｜最小校准 30",
        "- 只读 TOP1-AUDIT-002；不训练模型、不改 55%、不写 latest.json；T+1 open→T+2 close，14.5/25bps。",
        "", f"- 覆盖率 **{cov:.1%}**（目标 ≥85%）｜单例{{涨}} {single:.1%}｜空集 {empty:.1%}",
        "", "| 规则 | 交易天数 | 日均机会净收益 | 每笔净收益 | 每笔中位数 | 每笔胜率 | 25bps 日均 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| B1 生产（p≥55%） | {b1['trade_days']} | {b1['opp_mean']:+.3%} | {(b1['per_trade_mean'] or 0):+.3%} | {(b1['per_trade_median'] or 0):+.3%} | {(b1['per_trade_up'] or 0):.1%} | {b1['stress25']:+.3%} |",
        f"| 自适应保形 | {conf['trade_days']} | {conf['opp_mean']:+.3%} | {(conf['per_trade_mean'] or 0):+.3%} | {(conf['per_trade_median'] or 0):+.3%} | {(conf['per_trade_up'] or 0):.1%} | {conf['stress25']:+.3%} |",
        "", f"- 自适应 − B1 同日配对差：{paired['mean']:+.3%}，95%CI [{paired['ci95'][0]:+.3%},{paired['ci95'][1]:+.3%}]，p(≤0)={paired['p_le_0']:.3f}",
        f"- 自适应每笔 bootstrap：95%CI [{conf['per_boot']['ci95'][0]:+.3%},{conf['per_boot']['ci95'][1]:+.3%}]，p(≤0)={conf['per_boot']['p_le_0']:.3f}" if conf["per_boot"] else "- 自适应：无交易",
        "", "## 分半", "",
        "| 段 | 天数 | 自适应交易 | 自适应每笔 | 自适应日均机会 | B1 日均机会 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 前半 | {halves['first']['days']} | {halves['first']['conf_trades']} | {(halves['first']['conf_per_trade'] or 0):+.3%} | {halves['first']['conf_opp']:+.3%} | {halves['first']['b1_opp']:+.3%} |",
        f"| 后半 | {halves['second']['days']} | {halves['second']['conf_trades']} | {(halves['second']['conf_per_trade'] or 0):+.3%} | {halves['second']['conf_opp']:+.3%} | {halves['second']['b1_opp']:+.3%} |",
        "", "## 停止条件", "",
    ] + [f"- {'✅' if v else '❌'} {k}" for k, v in checks.items()] + [
        "", f"**判定：{'PASS（仅研究候选，不自动上线）' if passed else 'FAIL（保持 55% 门槛与空仓纪律）'}**",
        "", "## 局限", "",
        "- 120 天、60 日滚动窗口，时间序列上的 conformal 覆盖仍是近似；",
        "- 只改弃权规则，不改模型；alpha/窗口/最小校准均冻结，失败不调参；",
        "- 不写 latest.json、不改 55%；上线需新前瞻样本。",
        "", "## 免责声明", "", "研究实验输出，不构成投资建议。",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    reg = ROOT / "reports" / "experiment_registry.csv"
    with open(reg, "a", encoding="utf-8") as f:
        f.write(f"CONFORM-ADAPT-001,{pd.Timestamp.now():%Y-%m-%d},电池个股短线Top1,AUDIT-COMPLETE,"
                f"{'PASS' if passed else 'FAIL'};cov={cov:.3f};adaptive_opp={conf['opp_mean']};"
                f"b1_opp={b1['opp_mean']};trades={conf['trade_days']}\n")
    print(json.dumps({"coverage": cov, "adaptive": {k: conf[k] for k in ("trade_days", "opp_mean", "per_trade_mean", "stress25")},
                      "b1": {k: b1[k] for k in ("trade_days", "opp_mean", "per_trade_mean", "stress25")},
                      "checks": checks, "passed": passed}, ensure_ascii=False, indent=1, default=float))
    print("CONFORM_ADAPT_DONE")


if __name__ == "__main__":
    main()
