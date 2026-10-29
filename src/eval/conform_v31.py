"""CONFORM-001（预注册）：split conformal 预测集 + 单例出手。

只读 TOP1-AUDIT-002 的 Top1 明细，不训练模型、不改生产：
- score(y)=1-p_y（y∈{涨,跌}）
- 校准集=当日之前全部有 book 的 Top1，q 取 ceil((n+1)(1-alpha))/n 经验分位
- 仅当 C={涨} 且可买入时出手；否则空仓
"""
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

OUT = ROOT / "reports" / "conform_v31"
ALPHA = 0.10
MIN_CAL = 30


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (ctr - hw, ctr + hw)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = json.loads((OUT / "prereg.json").read_text(encoding="utf-8"))
    assert prereg["status"] == "PRE-REGISTERED" and prereg["alpha"] == ALPHA

    t = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    t["buy_block"] = t["buy_block"].fillna("")
    top1 = t[t["is_top1"]].sort_values("date").reset_index(drop=True)

    rows = []
    for i, r in top1.iterrows():
        prior = top1.iloc[:i]
        prior = prior[prior["book"].notna()]
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
            "q": q, "set_up": set1, "set_down": set0,
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
    both = float(((ev["set_up"]) & (ev["set_down"])).mean()) if len(ev) else float("nan")

    def block(traded: str, net: str, flag: str) -> dict:
        r = df.loc[df[flag], net]
        return {"trade_days": int(len(r)),
                "per_trade_mean": round(float(r.mean()), 5) if len(r) else None,
                "per_trade_median": round(float(r.median()), 5) if len(r) else None,
                "per_trade_up": round(float((r > 0).mean()), 4) if len(r) else None,
                "opp_mean": round(float(df[net].mean()), 5),
                "boot": boot_by_date(df.assign(v=df[net]), "v"),
                "per_boot": boot_by_date(df[df[flag]].assign(v=r), "v") if len(r) else None,
                "stress25": round(float(df[net.replace("net", "net25")].mean()), 5)}

    conf = block("trade", "net", "trade")
    b1 = block("b1", "b1_net", "b1")
    df["diff"] = df["net"] - df["b1_net"]
    paired = boot_by_date(df, "diff")

    ud = sorted(df["date"].unique())
    half = set(ud[: len(ud) // 2])
    halves = {}
    for tag, sub in (("first", df[df["date"].isin(half)]), ("second", df[~df["date"].isin(half)])):
        r = sub.loc[sub["trade"], "net"]
        r1 = sub.loc[sub["b1"], "b1_net"]
        halves[tag] = {"days": len(sub), "conf_trades": len(r),
                       "conf_per_trade": round(float(r.mean()), 5) if len(r) else None,
                       "conf_opp": round(float(sub["net"].mean()), 5),
                       "b1_opp": round(float(sub["b1_net"].mean()), 5)}

    checks = {
        "conf_opp > B1_opp": bool(conf["opp_mean"] > b1["opp_mean"]),
        "per_trade>0 且 p<0.05": bool((conf["per_trade_mean"] or -1) > 0 and conf["per_boot"] and conf["per_boot"]["p_le_0"] < 0.05),
        "前后半段均>0": bool((halves["first"]["conf_per_trade"] or -1) > 0 and (halves["second"]["conf_per_trade"] or -1) > 0),
        "25bps 压力>0": bool(conf["stress25"] > 0),
        "覆盖率>=0.85": bool(cov >= 0.85),
    }
    passed = all(checks.values())

    lines = [
        "# CONFORM-001：保形预测集 + 单例出手（预注册）",
        "",
        f"- 运行：{pd.Timestamp.now():%Y-%m-%d %H:%M}｜{len(df)} 天｜alpha={ALPHA:.0%}｜最小校准 {MIN_CAL} 天",
        "- 只读 TOP1-AUDIT-002 的 Top1；不训练模型、不改 55% 门槛、不写 latest.json",
        "- 口径：T+1 开盘买 → T+2 收盘卖；14.5bps/边（压力 25bps）",
        "",
        "## 预测集",
        "",
        f"- 经验覆盖率 **{cov:.1%}**（目标 ≥{1-ALPHA:.0%}）｜单例{{涨}} {single:.1%}｜空集 {empty:.1%}｜双集 {both:.1%}",
        "",
        "## 交易结果（每日机会净收益，无交易记0）",
        "",
        "| 规则 | 交易天数 | 日均机会净收益 | 每笔净收益 | 每笔中位数 | 每笔胜率 | 25bps 日均 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| B1 生产（p≥{TRADE_P:.0%}） | {b1['trade_days']} | {b1['opp_mean']:+.3%} | {(b1['per_trade_mean'] or 0):+.3%} | "
        f"{(b1['per_trade_median'] or 0):+.3%} | {(b1['per_trade_up'] or 0):.1%} | {b1['stress25']:+.3%} |",
        f"| 保形单例 | {conf['trade_days']} | {conf['opp_mean']:+.3%} | {(conf['per_trade_mean'] or 0):+.3%} | "
        f"{(conf['per_trade_median'] or 0):+.3%} | {(conf['per_trade_up'] or 0):.1%} | {conf['stress25']:+.3%} |",
        "",
        f"- 保形 − B1 同日配对差：{paired['mean']:+.3%}，95%CI [{paired['ci95'][0]:+.3%},{paired['ci95'][1]:+.3%}]，p(≤0)={paired['p_le_0']:.3f}",
        f"- 保形每笔 bootstrap：95%CI [{conf['per_boot']['ci95'][0]:+.3%},{conf['per_boot']['ci95'][1]:+.3%}]，p(≤0)={conf['per_boot']['p_le_0']:.3f}" if conf["per_boot"] else "- 保形每笔：无交易",
        "",
        "## 分半一致性",
        "",
        "| 段 | 天数 | 保形交易天数 | 保形每笔净收益 | 保形日均机会 | B1 日均机会 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 前半段 | {halves['first']['days']} | {halves['first']['conf_trades']} | {(halves['first']['conf_per_trade'] or 0):+.3%} | "
        f"{halves['first']['conf_opp']:+.3%} | {halves['first']['b1_opp']:+.3%} |",
        f"| 后半段 | {halves['second']['days']} | {halves['second']['conf_trades']} | {(halves['second']['conf_per_trade'] or 0):+.3%} | "
        f"{halves['second']['conf_opp']:+.3%} | {halves['second']['b1_opp']:+.3%} |",
        "",
        "## 停止条件",
        "",
    ] + [f"- {'✅' if v else '❌'} {k}" for k, v in checks.items()] + [
        "",
        f"**判定：{'PASS（仅表示值得进入下一阶段，不自动上线）' if passed else 'FAIL（保持 55% 门槛与空仓纪律）'}**",
        "",
        "## 局限",
        "",
        "- 120 天、80 个校准点，conformal 的覆盖率保证在时间序列上只是近似；",
        "- 只检验「弃权规则」，不改变模型本身；上线需新前瞻样本复核；",
        "- alpha 固定 0.10，未调参；若失败不调 alpha 救结果。",
        "", "## 免责声明", "", "研究实验输出，不构成投资建议。",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    res = {"exp_id": "CONFORM-001", "alpha": ALPHA, "coverage": round(cov, 4),
           "singleton": round(single, 4), "empty": round(empty, 4), "both": round(both, 4),
           "conf": conf, "b1": b1, "paired": paired, "halves": halves,
           "checks": checks, "passed": passed}
    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    with open(ROOT / "reports" / "experiment_registry.csv", "a", encoding="utf-8") as f:
        f.write(f"CONFORM-001,{pd.Timestamp.now():%Y-%m-%d},电池个股短线Top1,AUDIT-COMPLETE,"
                f"{'PASS' if passed else 'FAIL'};cov={cov:.3f};conf_opp={conf['opp_mean']};"
                f"b1_opp={b1['opp_mean']};trades={conf['trade_days']}\n")
    print(json.dumps({"coverage": cov, "singleton": single, "empty": empty,
                      "conf": {k: conf[k] for k in ("trade_days", "opp_mean", "per_trade_mean", "stress25")},
                      "b1": {k: b1[k] for k in ("trade_days", "opp_mean", "per_trade_mean", "stress25")},
                      "checks": checks, "passed": passed}, ensure_ascii=False, indent=1, default=float))
    print("CONFORM_DONE")


if __name__ == "__main__":
    main()
