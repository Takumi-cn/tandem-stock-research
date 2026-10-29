"""RELIAB-001（预注册）：基于历史门槛档可靠性的空仓门控。

预注册见 reports/reliability_v31/prereg.json：
- 信号：截至上一交易日的全部历史门槛日（p_up≥0.55、可买入）的实际上涨率（book）
- 规则：历史门槛日 ≥15 时，上涨率≥0.55 才允许出手；不足 15 天沿用 B1
- 当日仍需 p_up≥0.55 且可买入；评估口径与 TOP1-AUDIT-002 完全一致
- 失败即 FAIL，不写 latest.json、不替换 55% 门槛

只做一次预注册检验；不调窗口、不调阈值。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.top1_audit import COST_SIDE, COST_STRESS, boot_by_date  # noqa: E402
from src.report.short_pick import TRADE_P  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

OUT = ROOT / "reports" / "reliability_v31"
MIN_HIST = 15
REL_TH = 0.55


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = json.loads((OUT / "prereg.json").read_text(encoding="utf-8"))
    assert prereg["status"] == "PRE-REGISTERED"

    t = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    t["date"] = pd.to_datetime(t["date"])
    t["buy_block"] = t["buy_block"].fillna("")
    top1 = t[t["is_top1"]].sort_values("date").reset_index(drop=True)

    rows = []
    for i, r in top1.iterrows():
        prior = top1.iloc[:i]
        prior = prior[(prior["buy_block"] == "") & (prior["p_up"] >= TRADE_P)]
        if len(prior) >= MIN_HIST:
            rel = float((prior["book"] > 0).mean())
            allow = rel >= REL_TH
        else:
            rel, allow = np.nan, True
        want = bool(r["p_up"] >= TRADE_P and r["buy_block"] == "")
        rows.append({
            "date": r["date"], "code": r["code"], "p_up": r["p_up"], "book": r["book"],
            "r2d": r["r_2d"], "buy_block": r["buy_block"], "hist_gate_days": int(len(prior)),
            "hist_up": rel, "allow": allow, "b1_trade": want,
            "rule_trade": bool(want and allow),
            "b1_net": (r["r_2d"] - 2 * COST_SIDE) if want and pd.notna(r["r_2d"]) else 0.0,
            "rule_net": (r["r_2d"] - 2 * COST_SIDE) if (want and allow and pd.notna(r["r_2d"])) else 0.0,
            "b1_net25": (r["r_2d"] - 2 * COST_STRESS) if want and pd.notna(r["r_2d"]) else 0.0,
            "rule_net25": (r["r_2d"] - 2 * COST_STRESS) if (want and allow and pd.notna(r["r_2d"])) else 0.0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "days.csv", index=False, encoding="utf-8-sig")

    def block(col: str, traded: str, netcol: str) -> dict:
        r = df.loc[df[traded], netcol]
        return {
            "trade_days": int(r.notna().sum()),
            "per_trade_mean": round(float(r.mean()), 5) if len(r) else None,
            "per_trade_median": round(float(r.median()), 5) if len(r) else None,
            "per_trade_up": round(float((r > 0).mean()), 4) if len(r) else None,
        }

    b1 = {"opp_mean": round(float(df["b1_net"].mean()), 5), "opp_sum": round(float(df["b1_net"].sum()), 5),
          "opp_boot": boot_by_date(df.assign(v=df["b1_net"]), "v"), "stress25": round(float(df["b1_net25"].mean()), 5),
          **block("b1", "b1_trade", "b1_net")}
    rule = {"opp_mean": round(float(df["rule_net"].mean()), 5), "opp_sum": round(float(df["rule_net"].sum()), 5),
            "opp_boot": boot_by_date(df.assign(v=df["rule_net"]), "v"), "stress25": round(float(df["rule_net25"].mean()), 5),
            **block("rule", "rule_trade", "rule_net")}
    df["diff"] = df["rule_net"] - df["b1_net"]
    paired = boot_by_date(df, "diff")

    # 被门控挡掉的 B1 交易
    blocked = df[df["b1_trade"] & (~df["allow"]) & df["r2d"].notna()]
    avoided = {
        "n": int(len(blocked)),
        "mean_net": round(float(blocked["r2d"].mean()) - 2 * COST_SIDE, 5) if len(blocked) else None,
        "up": round(float((blocked["r2d"] > 0).mean()), 4) if len(blocked) else None,
    }

    ud = sorted(df["date"].unique())
    half = set(ud[: len(ud) // 2])
    halves = {}
    for tag, sub in (("first", df[df["date"].isin(half)]), ("second", df[~df["date"].isin(half)])):
        r = sub.loc[sub["rule_trade"], "rule_net"]
        r1 = sub.loc[sub["b1_trade"], "b1_net"]
        halves[tag] = {
            "days": int(len(sub)), "rule_trade_days": int(len(r)),
            "rule_per_trade": round(float(r.mean()), 5) if len(r) else None,
            "rule_opp": round(float(sub["rule_net"].mean()), 5),
            "b1_opp": round(float(sub["b1_net"].mean()), 5),
        }

    checks = {
        "rule_opp > B1_opp": bool(rule["opp_mean"] > b1["opp_mean"]),
        "per_trade>0 且 p<0.05": bool((rule["per_trade_mean"] or -1) > 0 and rule["opp_boot"]["p_le_0"] < 0.05),
        "前后半段均>0": bool((halves["first"]["rule_per_trade"] or -1) > 0 and (halves["second"]["rule_per_trade"] or -1) > 0),
        "25bps 压力>0": bool(rule["stress25"] > 0),
    }
    passed = all(checks.values())

    lines = [
        "# RELIAB-001：历史可靠性门控空仓（预注册）",
        "",
        f"- 运行：{pd.Timestamp.now():%Y-%m-%d %H:%M}｜{len(df)} 天｜历史门槛日 ≥{MIN_HIST} 才启用门控，阈值 {REL_TH:.0%}",
        "- 信号只用**上一交易日及更早**的门槛日实际结果，不含当日与未来；不调窗口/阈值",
        "- 口径：T+1 开盘买 → T+2 收盘卖；14.5bps/边（压力 25bps）",
        "",
        "## 结果（每日机会净收益，无交易记 0）",
        "",
        "| 规则 | 交易天数 | 日均机会净收益 | 每笔净收益 | 每笔中位数 | 每笔胜率 | 25bps 日均 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| B1 生产规则（p≥55%） | {b1['trade_days']} | {b1['opp_mean']:+.3%} | {(b1['per_trade_mean'] or 0):+.3%} | "
        f"{(b1['per_trade_median'] or 0):+.3%} | {(b1['per_trade_up'] or 0):.1%} | {b1['stress25']:+.3%} |",
        f"| 可靠性门控 | {rule['trade_days']} | {rule['opp_mean']:+.3%} | {(rule['per_trade_mean'] or 0):+.3%} | "
        f"{(rule['per_trade_median'] or 0):+.3%} | {(rule['per_trade_up'] or 0):.1%} | {rule['stress25']:+.3%} |",
        "",
        f"- 门控 − B1 同日配对差：{paired['mean']:+.3%}，95%CI [{paired['ci95'][0]:+.3%},{paired['ci95'][1]:+.3%}]，p(≤0)={paired['p_le_0']:.3f}",
        f"- 被门控挡掉的 B1 交易：{avoided['n']} 笔，平均净收益 {(avoided['mean_net'] or 0):+.3%}，胜率 {(avoided['up'] or 0):.1%}"
        f"（若为正，说明门控误杀）",
        f"- 门控日均机会净收益 bootstrap：95%CI [{rule['opp_boot']['ci95'][0]:+.3%},{rule['opp_boot']['ci95'][1]:+.3%}]，"
        f"p(≤0)={rule['opp_boot']['p_le_0']:.3f}",
        "",
        "## 分半一致性",
        "",
        "| 段 | 天数 | 门控交易天数 | 门控每笔净收益 | 门控日均机会 | B1 日均机会 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 前半段 | {halves['first']['days']} | {halves['first']['rule_trade_days']} | "
        f"{(halves['first']['rule_per_trade'] or 0):+.3%} | {halves['first']['rule_opp']:+.3%} | {halves['first']['b1_opp']:+.3%} |",
        f"| 后半段 | {halves['second']['days']} | {halves['second']['rule_trade_days']} | "
        f"{(halves['second']['rule_per_trade'] or 0):+.3%} | {halves['second']['rule_opp']:+.3%} | {halves['second']['b1_opp']:+.3%} |",
        "",
        "## 停止条件",
        "",
    ] + [f"- {'✅' if v else '❌'} {k}" for k, v in checks.items()] + [
        "",
        f"**判定：{'PASS（仅表示值得进入下一阶段，不自动上线）' if passed else 'FAIL（保持生产规则不变，不调参救结果）'}**",
        "",
        "## 局限",
        "",
        "- 复用 TOP1-AUDIT-002 的 Top1 与可实现收益；历史门槛日样本有限（尤其前 30 天）；",
        "- 可靠性信号是门槛档的 book 命中率，不是收益；命中率恢复不代表收益恢复；",
        "- 非 PIT 成分、单次 120 日回放；不写 latest.json、不替换 55% 门槛。",
        "", "## 免责声明", "", "研究实验输出，不构成投资建议。",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    res = {"exp_id": "RELIAB-001", "b1": b1, "rule": rule, "paired": paired,
           "avoided": avoided, "halves": halves, "gate": {"checks": checks, "passed": passed}}
    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    with open(ROOT / "reports" / "experiment_registry.csv", "a", encoding="utf-8") as f:
        f.write(f"RELIAB-001,{pd.Timestamp.now():%Y-%m-%d},电池个股短线,AUDIT-COMPLETE,"
                f"{'PASS' if passed else 'FAIL'};rule_opp={rule['opp_mean']};b1_opp={b1['opp_mean']};"
                f"rule_trades={rule['trade_days']}\n")
    print(json.dumps({"b1": {k: b1[k] for k in ("trade_days", "opp_mean", "per_trade_mean", "stress25")},
                      "rule": {k: rule[k] for k in ("trade_days", "opp_mean", "per_trade_mean", "stress25")},
                      "avoided": avoided, "checks": checks, "passed": passed}, ensure_ascii=False, indent=1, default=float))
    print("RELIAB_DONE")


if __name__ == "__main__":
    main()
