"""Reconcile three different return definitions.

- B1 daily opportunity mean: mean over all 120 signal days, no-trade days = 0.
- B1 per-trade mean: mean over the 79 B1 trade days only.
- Account final equity: event-level SLEEVE-1/SLEEVE-2 path, all Top1 vs B1-only.
These are not interchangeable.  Research only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.top1_audit import COST_SIDE, COST_STRESS  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

OUT = ROOT / "reports" / "metric_reconciliation"


def account_event(selected: pd.DataFrame, dates: list[pd.Timestamp], mode: int, cost_bps: float) -> dict:
    sleeves = 1 if mode == 1 else 2
    free = np.full(sleeves, -1, dtype=int)
    eq = np.ones(sleeves) / sleeves
    idx = {d: i for i, d in enumerate(dates)}
    trades = skipped = 0
    for r in selected.itertuples():
        si = idx.get(pd.Timestamp(r.date))
        if si is None or not np.isfinite(r.r_2d):
            continue
        entry, exit_ = si + 1, si + 2
        avail = np.where(free < entry)[0]
        if not len(avail):
            skipped += 1
            continue
        s = int(avail[0])
        eq[s] *= 1.0 + float(r.r_2d) - 2 * cost_bps / 10000.0
        free[s] = exit_
        trades += 1
    final = float(eq.sum())
    return {"final_equity": final, "ann": final ** (252 / max(1, len(selected))) - 1,
            "trades": trades, "skipped": skipped}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(ROOT / "data" / "parquet" / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    dates = sorted(panel["date"].unique())
    t = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    t["date"] = pd.to_datetime(t["date"])
    top = t[t["is_top1"]].sort_values("date").reset_index(drop=True)
    tradable = top["buy_block"].fillna("").eq("") & top["r_2d"].notna()
    b1 = tradable & (top["p_up"] >= 0.55)
    net = top["r_2d"] - 2 * COST_SIDE
    opp = float(net.where(b1, 0.0).mean())
    per = float(net[b1].mean())
    seq = float((1 + net[b1]).prod() - 1)
    b1s1 = account_event(top[b1].copy(), dates, 1, 14.5)
    b1s2 = account_event(top[b1].copy(), dates, 2, 14.5)
    alls1 = account_event(top[tradable].copy(), dates, 1, 14.5)
    alls2 = account_event(top[tradable].copy(), dates, 2, 14.5)
    # conformal adaptive account (15 trades) for comparison
    conf = pd.read_csv(ROOT / "reports" / "conform_adaptive_v31" / "days.csv")
    conf["date"] = pd.to_datetime(conf["date"])
    ct = conf[conf["trade"]].copy()
    ct["r_2d"] = ct["r2d"]
    cs2 = account_event(ct, dates, 2, 14.5)
    metrics = {
        "source": "TOP1-AUDIT-002; 120 signal days (2026-04-27..2026-10-23); 14.5bps/side",
        "b1": {
            "signal_days": int(len(top)),
            "tradable_days": int(tradable.sum()),
            "b1_trade_days": int(b1.sum()),
            "opportunity_mean": opp,
            "per_trade_mean": per,
            "per_trade_median": float(net[b1].median()),
            "per_trade_win": float((net[b1] > 0).mean()),
            "sequential_compounded": seq,
            "sleeve1_final": b1s1["final_equity"],
            "sleeve2_final": b1s2["final_equity"],
            "sleeve2_ann": b1s2["ann"],
            "sleeve2_trades": b1s2["trades"],
            "sleeve1_skipped": b1s1["skipped"],
            "naive_daily_mean_times_252": opp * 252,
        },
        "all_top1": {
            "sleeve1_final": alls1["final_equity"], "sleeve2_final": alls2["final_equity"],
            "sleeve2_ann": alls2["ann"], "sleeve2_trades": alls2["trades"],
            "sleeve1_skipped": alls1["skipped"],
        },
        "conformal_adaptive": {
            "trade_days": len(ct), "sleeve2_final": cs2["final_equity"],
            "sleeve2_ann": cs2["ann"], "sleeve2_trades": cs2["trades"],
        },
        "bridge": {
            "sum_per_trade_net": float(net[b1].sum()),
            "compounded_per_trade_net": seq,
            "b1_sleeve2_minus_one": b1s2["final_equity"] - 1,
            "explanation": "B1 opportunity mean is over 120 days (41 zero days); per-trade mean is over 79 trades; account equity compounds only when a sleeve is free and therefore cannot equal either average multiplied by days.",
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"metric": "B1_opportunity_mean", "scope": "120 days, no-trade=0", "value": opp},
        {"metric": "B1_per_trade_mean", "scope": "79 B1 trades", "value": per},
        {"metric": "B1_sequential_compounded", "scope": "79 trades, sequential full-capital counterfactual", "value": seq},
        {"metric": "B1_sleeve1_final", "scope": "79 B1 signals, 1 sleeve", "value": b1s1["final_equity"]},
        {"metric": "B1_sleeve2_final", "scope": "79 B1 signals, 2 sleeves", "value": b1s2["final_equity"]},
        {"metric": "all_top1_sleeve1_final", "scope": "118 tradeable Top1, 1 sleeve", "value": alls1["final_equity"]},
        {"metric": "all_top1_sleeve2_final", "scope": "118 tradeable Top1, 2 sleeves", "value": alls2["final_equity"]},
        {"metric": "adaptive_conformal_sleeve2_final", "scope": "15 conformal trades, 2 sleeves", "value": cs2["final_equity"]},
    ]).to_csv(OUT / "reconciliation.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 三套收益口径对账（B1 机会 / 单笔 / 账户权益）", "",
        "- 来源：TOP1-AUDIT-002；120 个信号日（2026-04-27~2026-10-23）；T+1 open→T+2 close；14.5bps/边。",
        "- 这不是新模型、不是新策略，也不改变生产；只核对过去报告里容易被混用的数字。",
        "", "## 结论先行", "",
        "| 口径 | 数值 | 不能怎么用 |", "|---|---:|---|",
        f"| B1 日均机会净收益 | **{opp:+.3%}**（120 天，41 天 0） | 不能乘 252 当年化，也不能直接和账户 0.9565x 比 |",
        f"| B1 单笔净收益 | **{per:+.3%}**（79 笔） | 不能当成连续账户收益；信号会重叠 |",
        f"| B1 双槽账户最终权益 | **{b1s2['final_equity']:.4f}x**（79 笔，0 跳过） | 正收益不等于统计显著；未做 matched-random/DSR |",
        f"| B1 单槽账户最终权益 | **{b1s1['final_equity']:.4f}x**（46 笔，33 跳过） | 单槽跳过重叠信号，不能代表选中质量 |",
        f"| 全部 Top1 双槽账户 | **{alls2['final_equity']:.4f}x**（118 笔） | 这是 V2 的 0.9565x，不是 B1-only |",
        "", f"- 若把 B1 的 {opp:.3%}/日机械乘 252，会得到 {opp*252:.1%} 的表面年化；它只是 120 个重叠机会的均值，不是账户可实现收益。",
        f"- B1 单笔顺序复利 {seq:+.2%} 也不是账户：真实双槽只有在空闲槽位时才能复利，最终为 {b1s2['final_equity']:.4f}x；单槽因为跳过 33 个重叠信号仅 {b1s1['final_equity']:.4f}x。",
        f"- V2 报告中的 **0.9565x** 是全部 118 个可交易 Top1 的双槽路径，**不是 B1 的账户结果**。B1-only 双槽是 **{b1s2['final_equity']:.4f}x**，两者样本和规则不同，不能混用。",
        "", "## B1 账户细节", "",
        f"- B1 交易 79 天；双槽 79 笔成交、0 跳过；单槽 46 笔成交、33 跳过。",
        f"- 双槽年化（仅 120 日样本）{b1s2['ann']:+.1%}；这是机械年化，不构成未来收益保证。",
        f"- 单笔中位 {net[b1].median():+.3%}、胜率 {(net[b1] > 0).mean():.1%}；均值由少数大赢家拉动。",
        "", "## 与 conformal 的区别", "",
        f"- 自适应 conformal 只有 {len(ct)} 个交易日；双槽账户 {cs2['final_equity']:.4f}x，账户收益 {metrics['conformal_adaptive']['sleeve2_final']-1:+.2%}；样本太小，不能和 B1 的 79 天直接比较。",
        "", "## 统计与治理", "",
        "- 这些数字没有 matched-random、DSR/PBO 或 250 日前瞻支持；账户为正不等于有 alpha。",
        "- 生产仍用 v3.1 的 55% 门槛；B1-only 1.1294x 只是历史诊断，不能据此加仓或改规则。",
        "", "## 免责声明", "", "研究实验输出，不构成投资建议。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
