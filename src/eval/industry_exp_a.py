"""Experiment A: v4.1 LightGBM Ranker + Top3/Top5 cost-aware hysteresis.

不重新训练。使用 reports/industry_v41/lgbm_scores.parquet（v4.1 OOS 分数）。
目标：Top3；已有持仓若新排名<=5保留，>5才替换；空位从新排名最高者填入。
基线：Random-3、turnover-matched random、score-permutation（同滞回规则）。
判定：换手下降>=25%；毛 selection alpha vs matched random>0；两段 selection alpha 均>0；
      经验 p 值（net Sharpe/net ann/selection alpha）<0.05（快速 2000 次筛选）。
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
from src.eval.industry_rotation import load_clean  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

OUT = ROOT / "reports" / "industry_exp_a"
SCORES = ROOT / "reports" / "industry_v41" / "lgbm_scores.parquet"
COST_BPS = 14.5
STRESS_BPS = 25.0
TOP_K = 3
BUFFER_RANK = 5
HOLD_DAYS = 5
N_MATCHED = 2000
N_PERM = 500
SEED = 20261026


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    if not len(r):
        return {}
    eq = (1 + r).cumprod()
    ann = float(eq.iloc[-1] ** (252 / (len(r) * HOLD_DAYS)) - 1)
    vol = float(r.std(ddof=1) * math.sqrt(252 / HOLD_DAYS)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252 / HOLD_DAYS)) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min())
    return {"ann": round(ann, 4), "vol": round(vol, 4), "sharpe": round(sharpe, 4), "mdd": round(mdd, 4), "win": round(float((r > 0).mean()), 4), "n": int(len(r))}


def load_scores():
    s = pd.read_parquet(SCORES)
    s["date"] = pd.to_datetime(s["date"])
    s["code"] = s["code"].astype(str)
    return s


def path_from_scores(score_map, dates, test_signals, universe):
    """Top3/Top5 hysteresis selection path. Returns list of holdings and per-signal change counts."""
    holdings: list[str] = []
    paths, changes = [], []
    for t in test_signals:
        sc = score_map.get(t)
        if sc is None:
            continue
        sc = sc.reindex(universe).dropna().sort_values(ascending=False)
        if len(sc) == 0:
            continue
        rank = {c: i + 1 for i, c in enumerate(sc.index)}
        if not holdings:
            target = list(sc.index[:TOP_K])
            changes.append(len(target))
        else:
            kept = [c for c in holdings if rank.get(c, 10**9) <= BUFFER_RANK]
            target = kept[:TOP_K]
            for c in sc.index:
                if len(target) >= TOP_K:
                    break
                if c not in target:
                    target.append(c)
            changes.append(len(set(target) - set(holdings)))
        paths.append((t, target))
        holdings = target
    return paths, changes


def path_returns(paths, changes, dates, c2, o2, side_bps):
    rets = []
    for (t, target), k in zip(paths, changes):
        idx = dates.get_loc(t)
        entry, exit_ = dates[idx + 1], dates[idx + HOLD_DAYS]
        gross = []
        for c in target:
            try:
                rr = float(c2.at[exit_, c] / o2.at[entry, c] - 1)
                if not np.isfinite(rr):
                    rr = 0.0
            except Exception:  # noqa: BLE001
                rr = 0.0
            gross.append(rr)
        turn = 0.5 if not rets else 0.5 * k / max(1, TOP_K)
        rets.append(float(np.mean(gross)) - turn * 2 * side_bps / 10000)
    return pd.Series(rets)


def matched_random_returns(changes, dates, test_signals, c2, o2, universe, rng, side_bps):
    holdings: list[str] = []
    rets = []
    for t, k in zip(test_signals, changes):
        idx = dates.get_loc(t)
        entry, exit_ = dates[idx + 1], dates[idx + HOLD_DAYS]
        if not holdings:
            kk = min(TOP_K, len(universe))
            target = list(rng.choice(universe, size=kk, replace=False))
        else:
            kk = min(int(k), len(universe))
            target = list(holdings)
            replaceable = [c for c in target]
            if kk > 0:
                drop = set(rng.choice(replaceable, size=min(kk, len(replaceable)), replace=False))
                target = [c for c in target if c not in drop]
                candidates = [c for c in universe if c not in target]
                add = list(rng.choice(candidates, size=min(kk, len(candidates)), replace=False))
                target.extend(add)
            while len(target) < TOP_K:
                candidates = [c for c in universe if c not in target]
                if not candidates:
                    break
                target.append(str(rng.choice(candidates)))
        gross = []
        for c in target:
            try:
                rr = float(c2.at[exit_, c] / o2.at[entry, c] - 1)
                if not np.isfinite(rr):
                    rr = 0.0
            except Exception:  # noqa: BLE001
                rr = 0.0
            gross.append(rr)
        turn = 0.5 if not holdings else 0.5 * len(set(target) - set(holdings)) / max(1, TOP_K)
        rets.append(float(np.mean(gross)) - turn * 2 * side_bps / 10000)
        holdings = target
    return pd.Series(rets)


def permuted_paths(score_map, dates, test_signals, universe, rng):
    out = {}
    for t in test_signals:
        if t in score_map:
            vals = np.array(score_map[t].reindex(universe).dropna().values, copy=True)
            rng.shuffle(vals)
            out[t] = pd.Series(vals, index=universe)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    dates, c1, c2, o2, a2, ret_cc, jump, recent_jump, meta1, meta2, quality = load_clean()
    test_signals = sorted(scores["date"].unique())
    test_signals = [pd.Timestamp(t) for t in test_signals]
    universe = sorted(scores["code"].unique())
    score_map = {t: scores[scores["date"] == t].set_index("code")["score_lgbm"].astype(float) for t in test_signals}
    # 原 v4.1 LGBM top3（无滞回）
    orig_paths = []
    holdings = []
    for t in test_signals:
        sc = score_map[t].reindex(universe).dropna().sort_values(ascending=False)
        target = list(sc.index[:TOP_K])
        orig_paths.append((t, target))
        holdings = target
    orig_changes = [TOP_K] + [len(set(target) - set(prev)) for (_, target), (_, prev) in zip(orig_paths[1:], orig_paths[:-1])]
    orig_net = path_returns(orig_paths, orig_changes, dates, c2, o2, COST_BPS)
    orig_gross = path_returns(orig_paths, orig_changes, dates, c2, o2, 0.0)
    # 滞回
    hys_paths, hys_changes = path_from_scores(score_map, dates, test_signals, universe)
    hys_net = path_returns(hys_paths, hys_changes, dates, c2, o2, COST_BPS)
    hys_gross = path_returns(hys_paths, hys_changes, dates, c2, o2, 0.0)
    hys_25 = path_returns(hys_paths, hys_changes, dates, c2, o2, STRESS_BPS)
    # random matched
    rng = np.random.default_rng(SEED)
    matched, matched_gross = [], []
    for _ in range(N_MATCHED):
        r = matched_random_returns(hys_changes, dates, test_signals, c2, o2, universe, rng, COST_BPS)
        matched.append(perf(r)); matched_gross.append(perf(matched_random_returns(hys_changes, dates, test_signals, c2, o2, universe, rng, 0.0)))
    # score permutation
    perm = []
    for _ in range(N_PERM):
        sm = permuted_paths(score_map, dates, test_signals, universe, rng)
        pp, cc = path_from_scores(sm, dates, test_signals, universe)
        perm.append(perf(path_returns(pp, cc, dates, c2, o2, COST_BPS)))
    # 分段
    mid = len(test_signals) // 2
    seg = {"first": perf(hys_net.iloc[:mid]), "second": perf(hys_net.iloc[mid:])}
    rand_seg = {"first": [], "second": []}
    for _ in range(N_MATCHED):
        r = matched_random_returns(hys_changes, dates, test_signals, c2, o2, universe, rng, COST_BPS)
        rand_seg["first"].append(perf(r.iloc[:mid])); rand_seg["second"].append(perf(r.iloc[mid:]))
    seg_rand = {k: {"ann_median": round(float(pd.DataFrame(v)["ann"].median()), 4), "sharpe_median": round(float(pd.DataFrame(v)["sharpe"].median()), 4)} for k, v in rand_seg.items()}
    # p 值（越大越差；strategy 的 percentile）
    def empirical_p(strategy_metric, samples, key):
        arr = np.array([x.get(key, np.nan) for x in samples if x.get(key) is not None], dtype=float)
        arr = arr[np.isfinite(arr)]
        return round(float((1 + np.sum(arr >= strategy_metric)) / (len(arr) + 1)), 4) if len(arr) else None
    hys_metrics = perf(hys_net)
    result = {
        "strategy": "LightGBM Ranker Top3/Top5 hysteresis",
        "orig": {"net": perf(orig_net), "gross": perf(orig_gross), "turnover": round(float(np.mean([0.5 if i == 0 else 0.5 * k / TOP_K for i, k in enumerate(orig_changes)])), 4)},
        "hys": {**hys_metrics, "gross": perf(hys_gross), "cost25": perf(hys_25), "turnover": round(float(np.mean([0.5 if i == 0 else 0.5 * k / TOP_K for i, k in enumerate(hys_changes)])), 4), "avg_changes": round(float(np.mean(hys_changes)), 3), "periods": len(hys_paths)},
        "matched_random": {"net_ann_median": round(float(pd.DataFrame(matched)["ann"].median()), 4), "net_sharpe_median": round(float(pd.DataFrame(matched)["sharpe"].median()), 4), "gross_ann_median": round(float(pd.DataFrame(matched_gross)["ann"].median()), 4)},
        "score_perm": {"net_ann_median": round(float(pd.DataFrame(perm)["ann"].median()), 4), "net_sharpe_median": round(float(pd.DataFrame(perm)["sharpe"].median()), 4)},
        "segments": {"hys": seg, "matched_random": seg_rand},
        "p_values": {
            "net_ann_vs_matched": empirical_p(hys_metrics.get("ann", 0), matched, "ann"),
            "net_sharpe_vs_matched": empirical_p(hys_metrics.get("sharpe", 0), matched, "sharpe"),
            "net_ann_vs_perm": empirical_p(hys_metrics.get("ann", 0), perm, "ann"),
            "gross_ann_vs_matched": empirical_p(perf(hys_gross).get("ann", 0), matched_gross, "ann"),
        },
    }
    turn_reduction = 1 - result["hys"]["turnover"] / result["orig"]["turnover"] if result["orig"]["turnover"] else 0
    # selection alpha = gross/net hys - matched random median (using annualized)
    result["selection_alpha_gross"] = round(result["hys"]["gross"].get("ann", 0) - result["matched_random"]["gross_ann_median"], 4)
    result["selection_alpha_net"] = round(result["hys"].get("ann", 0) - result["matched_random"]["net_ann_median"], 4)
    result["turnover_reduction"] = round(float(turn_reduction), 4)
    result["segments_alpha"] = {
        "first": round(seg["first"].get("ann", 0) - seg_rand["first"]["ann_median"], 4),
        "second": round(seg["second"].get("ann", 0) - seg_rand["second"]["ann_median"], 4),
    }
    gate = {
        "turnover_reduction_ge_25": bool(turn_reduction >= 0.25),
        "gross_alpha_gt_0": bool(result["selection_alpha_gross"] > 0),
        "segments_both_gt_0": bool(result["segments_alpha"]["first"] > 0 and result["segments_alpha"]["second"] > 0),
        "p_ann_lt_05": bool((result["p_values"]["net_ann_vs_matched"] or 1) < 0.05),
        "p_sharpe_lt_05": bool((result["p_values"]["net_sharpe_vs_matched"] or 1) < 0.05),
    }
    gate["pass"] = bool(all(gate.values()))
    result["gate"] = gate
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "data_asof": quality["as_of"], "test_signals": [str(x.date()) for x in test_signals], "universe": len(universe), "parameters": {"top_k": TOP_K, "buffer_rank": BUFFER_RANK, "cost_bps": COST_BPS, "stress_bps": STRESS_BPS, "matched_runs": N_MATCHED, "perm_runs": N_PERM}, "result": result}
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # trial registry (include A)
    reg = OUT / "registry.csv"
    row = pd.DataFrame([{"experiment_id": "IND-V42-EXPA-001", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "hypothesis": "Top3/Top5 cost-aware hysteresis on v4.1 LGBM scores", "data_asof": quality["as_of"], "status": "EXPLORATORY", "turnover_reduction": result["turnover_reduction"], "gross_alpha": result["selection_alpha_gross"], "net_alpha": result["selection_alpha_net"], "gate_pass": gate["pass"]}])
    if reg.exists():
        old = pd.read_csv(reg); old = old[old["experiment_id"] != "IND-V42-EXPA-001"]
        pd.concat([old, row], ignore_index=True).to_csv(reg, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(reg, index=False, encoding="utf-8-sig")
    lines = [
        "# Experiment A：LightGBM Top3/Top5 成本感知滞回", "",
        f"- 数据：{quality['as_of']}；复用 v4.1 OOS 分数；信号 {len(test_signals)} 个；{COST_BPS}bps/边；滞回带 Top5；Random matched {N_MATCHED} 次；permutation {N_PERM} 次。",
        f"- 换手：原 {result['orig']['turnover']:.2%} → 滞回 {result['hys']['turnover']:.2%}；下降 {result['turnover_reduction']:.1%}",
        f"- 策略净：{result['hys']['ann']:+.2%}（Sharpe {result['hys']['sharpe']:.2f}）；毛 {result['hys']['gross']['ann']:+.2%}；25bps {result['hys']['cost25']['ann']:+.2%}",
        f"- 匹配随机：净中位 {result['matched_random']['net_ann_median']:+.2%} / Sharpe {result['matched_random']['net_sharpe_median']:.2f}；毛中位 {result['matched_random']['gross_ann_median']:+.2%}",
        f"- Selection alpha：毛 {result['selection_alpha_gross']:+.2%}；净 {result['selection_alpha_net']:+.2%}",
        f"- 分段 alpha：{result['segments_alpha']}；随机分段 {result['segments']['matched_random']}",
        f"- 经验 p：{result['p_values']}", "",
        "## 停止条件", "",
        f"- 换手下降≥25%: {gate['turnover_reduction_ge_25']}",
        f"- 毛 selection alpha>0: {gate['gross_alpha_gt_0']}",
        f"- 两段 alpha 均>0: {gate['segments_both_gt_0']}",
        f"- p(net ann)<0.05: {gate['p_ann_lt_05']}；p(net Sharpe)<0.05: {gate['p_sharpe_lt_05']}",
        f"- **结论：{'PASS' if gate['pass'] else 'FAIL/STOP'}**", "",
        "## 口径", "",
        "- 不重新训练；使用 v4.1 已保存 OOS 分数。Top3；已有持仓排名≤5 保留；空位按新排名填；不持现金。",
        "- 基准含 turnover-matched random 与 score-permutation；Random-3 历史基线见 v4.1 报告；随机比较同成本/时序。",
        "- 非 PIT；仍未完成 walk-forward/PIT 审计；结果仅研究沙盒。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("EXP_A_DONE", json.dumps({"gate": gate, "result": result}, ensure_ascii=True))


if __name__ == "__main__":
    main()
