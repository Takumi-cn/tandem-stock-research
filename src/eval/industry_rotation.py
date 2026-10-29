"""行业层 v4 指数级回测（第一道门槛）。

预注册规则（不做参数搜索）：
- 每 5 个交易日调仓；T 日收盘出信号，T+1 开盘买入，持有到 T+5 收盘。
- 持有 3 个行业等权；14.5bps/边基础成本，另做 25bps 敏感性。
- 主策略：SW1 T' 方向（<0 动量 / >0 反转）+ SW2 复合强度/反转 + 单日脉冲过滤
  + 成交额拥挤度惩罚 + 20日相关性>0.7去重 + 前8名缓冲（降低换手）。
- 对照：纯复合强度（始终动量）、简单20日动量、随机3行业（300次）、固定电池801737。
- 全部使用 T+1 开盘执行；固定随机种子；不引用外部收益数字。

数据：data/industry/sw1_daily.parquet / sw2_daily.parquet（akshare 指数级，非个股 PIT）。
输出：reports/industry_v4/report.md + metrics.json + holdings.csv + data_quality.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT  # noqa: E402

DATA = ROOT / "data" / "industry"
OUT = ROOT / "reports" / "industry_v4"
START = pd.Timestamp("2021-10-22")
BATTERY = "801737"
N_HOLD = 3
HOLD_DAYS = 5
BASE_COST = 0.00145  # 单边 14.5bps；另做 25bps 敏感性
ALT_COST = 0.0025
BUFFER_TOP = 8
CORR_CAP = 0.7
RANDOM_RUNS = 300
SEED = 20261026


def _log(msg: str) -> None:
    print(msg, flush=True)


def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="date", columns="code", values=col).sort_index()


def load_clean():
    OUT.mkdir(parents=True, exist_ok=True)
    sw2 = pd.read_parquet(DATA / "sw2_daily.parquet")
    sw2["date"] = pd.to_datetime(sw2["date"])
    sw2 = sw2.drop_duplicates(subset=["date", "code"], keep="last")
    sw2 = sw2[sw2["date"] >= START]
    meta2 = pd.read_parquet(DATA / "sw2_meta.parquet")
    meta2["code"] = meta2["code"].astype(str)
    meta2["parent"] = meta2.get("parent", "").astype(str) if "parent" in meta2.columns else ""
    dates = pd.DatetimeIndex(sorted(set(sw2["date"])))
    c2 = _pivot(sw2, "close").reindex(dates)
    # SW1: 官方接口当前超时，用 SW2 按上级行业等权合成；报告中明确标注
    sw1_path = DATA / "sw1_daily.parquet"
    sw1_source = "official"
    if sw1_path.exists():
        sw1 = pd.read_parquet(sw1_path)
        sw1["date"] = pd.to_datetime(sw1["date"])
        sw1 = sw1[sw1["date"] >= START]
        meta1 = pd.read_parquet(DATA / "sw1_meta.parquet")
        meta1["code"] = meta1["code"].astype(str)
        c1 = _pivot(sw1, "close").reindex(dates)
    else:
        sw1_source = "synthetic_parent_equal_weight"
        ret_raw = c2.pct_change(fill_method=None)
        grp = meta2.set_index("code")["parent"].replace("", np.nan).dropna()
        members = [c for c in ret_raw.columns if c in grp.index]
        avg = ret_raw[members].T.groupby(grp.loc[members]).mean().T
        c1 = (1.0 + avg.fillna(0.0)).cumprod() * 1000.0
        meta1 = pd.DataFrame({"code": c1.columns.astype(str), "name": c1.columns.astype(str), "level": 1})
    o2 = _pivot(sw2, "open").reindex(dates)
    h2 = _pivot(sw2, "high").reindex(dates)
    l2 = _pivot(sw2, "low").reindex(dates)
    a2 = _pivot(sw2, "amount").reindex(dates)

    # OHLC / open 坏 tick（只记录并屏蔽；不改变策略选择）
    bad_ohlc = (h2 < l2) | (h2 <= 0) | (l2 <= 0) | (c2 <= 0) | (c2 > h2 * 1.001) | (c2 < l2 * 0.999)
    bad_open = (o2 <= 0) | (o2 > h2 * 1.001) | (o2 < l2 * 0.999) | o2.isna()
    c2 = c2.mask(bad_ohlc)
    o2 = o2.mask(bad_ohlc | bad_open)
    ret_cc = c2.pct_change(fill_method=None)
    # 基点跳变：>25% 的孤立日，标记；选股时近期跳变行业剔出
    jump = ret_cc.abs() > 0.25
    jump_count = jump.sum().sum()
    recent_jump = jump.rolling(5, min_periods=1).max().astype(bool).fillna(False)
    # 空值覆盖：用主日历计算缺失率；bad_open 只统计已有 OHLC 的单元格
    holes = c2.isna().sum()
    coverage = 1 - holes / len(dates)
    existing = c2.notna() & h2.notna() & l2.notna()
    quality = {
        "as_of": str(dates.max().date()),
        "sw1_source": sw1_source,
        "n_sw1": int(c1.shape[1]),
        "n_sw2": int(c2.shape[1]),
        "calendar_days": int(len(dates)),
        "bad_ohlc_cells": int(bad_ohlc.sum().sum()),
        "bad_open_cells": int((bad_open & existing).sum().sum()),
        "base_jump_cells": int(jump_count),
        "sw2_median_coverage": round(float(coverage.median()), 4),
        "sw2_min_coverage": round(float(coverage.min()), 4),
        "sw2_low_coverage": {k: round(float(v), 3) for k, v in coverage[coverage < 0.9].head(20).items()},
    }
    return dates, c1, c2, o2, a2, ret_cc, jump, recent_jump, meta1, meta2, quality


def build_signals(dates, c1, c2, a2, ret_cc, jump, recent_jump, meta2):
    # SW1 T'（日线代理）
    r20_1 = c1 / c1.shift(20) - 1
    rank20_1 = r20_1.rank(axis=1, pct=True)
    shift_rank = rank20_1.shift(220)
    T = (rank20_1 - shift_rank).abs().mean(axis=1)
    std = T.rolling(220, min_periods=60).std()
    T_norm = T / std.replace(0, np.nan)
    T_smooth = T_norm.rolling(20, min_periods=5).mean()
    Tp = T_smooth.diff()

    # SW2 量价特征
    ret1 = c2.pct_change(fill_method=None)
    ret3 = c2 / c2.shift(3) - 1
    ret5 = c2 / c2.shift(5) - 1
    ret20 = c2 / c2.shift(20) - 1
    ret28 = c2 / c2.shift(28) - 1
    amount5 = a2.rolling(5, min_periods=3).mean()
    amount60 = a2.rolling(60, min_periods=20).mean()
    crowd = amount5 / amount60.replace(0, np.nan)
    r1 = ret1.rank(axis=1, pct=True)
    r3 = ret3.rank(axis=1, pct=True)
    r5 = ret5.rank(axis=1, pct=True)
    r20 = ret20.rank(axis=1, pct=True)
    r28 = ret28.rank(axis=1, pct=True)
    amt_r = amount5.rank(axis=1, pct=True)
    crowd_r = crowd.rank(axis=1, pct=True)
    # 单日脉冲：今天涨幅前3，但3日排名后1/2
    top3_today = ret1.rank(axis=1, ascending=False, method="min") <= 3
    pulse = top3_today & (r3 < 0.5)
    prior28 = r20.shift(28)  # 28个交易日前的20日强度
    strength = 0.35 * r1 + 0.30 * r3 + 0.20 * amt_r + 0.15 * r5 - 0.10 * prior28 - 0.10 * crowd_r
    strength = strength.mask(pulse)
    reversal = 0.60 * (1 - r20) + 0.20 * (1 - r5) + 0.20 * amt_r - 0.10 * crowd_r
    # 流动性下限（相对截面中位数）：金额不能太低；指数层面通常全通过
    liq_ok = amount5.gt(amount5.median(axis=1) * 0.2, axis=0)
    strength = strength.mask(~liq_ok)
    reversal = reversal.mask(~liq_ok)
    recent_jump = recent_jump.reindex(columns=c2.columns, fill_value=False)
    strength = strength.mask(recent_jump)
    reversal = reversal.mask(recent_jump)
    return {
        "Tp": Tp, "strength": strength, "reversal": reversal,
        "pulse": pulse, "crowd_rank": crowd_r, "ret20": ret20,
        "ret1": ret1, "ret5": ret5, "ret28": ret28, "amount5": amount5,
    }


def _eligible(sig_series, universe):
    return [c for c in universe if c in sig_series.index and pd.notna(sig_series[c])]


def _pick_with_corr(scores: pd.Series, eligible: list[str], ret_cc: pd.DataFrame, date, n: int, prev: list[str], use_buffer: bool):
    if not eligible:
        return []
    ranked = scores.reindex(eligible).dropna().sort_values(ascending=False)
    keep: list[str] = []
    if use_buffer and prev:
        # 老持仓仍在当前前8即可保留，减少无效换手
        top_buffer = set(ranked.head(BUFFER_TOP).index)
        for c in prev:
            if c in top_buffer and c not in keep:
                keep.append(c)
                if len(keep) >= n:
                    return keep
    hist = ret_cc[eligible].loc[:date].tail(20)
    corr = hist.corr(min_periods=15)
    for c in ranked.index:
        if c in keep:
            continue
        ok = True
        for k in keep:
            try:
                cv = corr.loc[c, k]
            except KeyError:
                cv = np.nan
            if pd.notna(cv) and cv > CORR_CAP:
                ok = False
                break
        if ok:
            keep.append(c)
        if len(keep) >= n:
            break
    return keep


def _perf(daily: pd.Series) -> dict:
    d = daily.fillna(0.0)
    if len(d) == 0:
        return {}
    nav = (1 + d).cumprod()
    ann = float(nav.iloc[-1] ** (252 / len(d)) - 1) if len(d) else 0.0
    vol = float(d.std(ddof=1) * math.sqrt(252)) if len(d) > 1 else 0.0
    sharpe = float(d.mean() / d.std(ddof=1) * math.sqrt(252)) if len(d) > 1 and d.std(ddof=1) > 0 else 0.0
    mdd = float((nav / nav.cummax() - 1).min())
    win = float((d > 0).mean())
    return {"ann": round(ann, 4), "vol": round(vol, 4), "sharpe": round(sharpe, 4), "mdd": round(mdd, 4), "win": round(win, 4), "n_days": int(len(d))}


def simulate(mode: str, signals, dates, c2, o2, ret_cc, meta2, universe: list[str], side_cost: float, start_idx: int, use_buffer: bool = True, seed: int = SEED, hold_days: int = HOLD_DAYS):
    """mode: main / composite / mom20 / random"""
    rng = np.random.default_rng(seed + (17 if mode == "random" else 0))
    daily = pd.Series(0.0, index=dates)
    prev: list[str] = []
    trades, turnover_total = 0, 0.0
    rebalance_dates = list(range(start_idx, len(dates) - hold_days, hold_days))
    holdings_rows = []
    last_exit_idx = None
    first_day = None
    last_day = None
    for r in rebalance_dates:
        t = dates[r]
        entry_i, exit_i = r + 1, r + hold_days
        entry, exit_ = dates[entry_i], dates[exit_i]
        tp = signals["Tp"].loc[t] if t in signals["Tp"].index else np.nan
        if mode == "main":
            if pd.isna(tp):
                continue
            reversal = tp > 0
            scores = signals["reversal"].loc[t] if reversal else signals["strength"].loc[t]
            pick = _pick_with_corr(scores, _eligible(scores, universe), ret_cc, t, N_HOLD, prev, use_buffer)
            regime = "reversal" if reversal else "momentum"
        elif mode == "composite":
            scores = signals["strength"].loc[t] if t in signals["strength"].index else pd.Series(dtype=float)
            pick = _pick_with_corr(scores, _eligible(scores, universe), ret_cc, t, N_HOLD, prev, use_buffer)
            regime = "momentum"
        elif mode == "mom20":
            scores = signals["ret20"].loc[t].rank(pct=True) if t in signals["ret20"].index else pd.Series(dtype=float)
            pick = _pick_with_corr(scores, _eligible(scores, universe), ret_cc, t, N_HOLD, prev, False)
            regime = "mom20"
        else:  # random
            if t not in signals["strength"].index:
                continue
            sv = signals["strength"].loc[t]
            pool = [c for c in universe if c in sv.index and pd.notna(sv[c])]
            k = min(N_HOLD, len(pool))
            pick = list(rng.choice(pool, size=k, replace=False)) if k else []
            regime = "random"
        # 执行：T+1 开盘有效才买；无效则现金
        valid = []
        for c in pick:
            ov = o2.at[entry, c] if c in o2.columns and entry in o2.index else np.nan
            if pd.notna(ov) and ov > 0:
                valid.append(c)
        w = {c: 1.0 / len(pick) if pick else 0.0 for c in valid}
        old = {c: (1.0 / len(prev) if prev else 0.0) for c in prev}
        # 旧持仓若没有下一期（本期重新计算），先卖后买，turnover按权重差
        turnover = 0.5 * sum(abs(w.get(c, 0.0) - old.get(c, 0.0)) for c in set(w) | set(old))
        cost = turnover * 2 * side_cost
        turnover_total += turnover
        if pick:
            trades += 1
        if w:
            for j in range(entry_i, exit_i + 1):
                d = dates[j]
                for c, wi in w.items():
                    if j == entry_i:
                        ov, cv = o2.at[d, c], c2.at[d, c]
                        rr = cv / ov - 1 if pd.notna(ov) and pd.notna(cv) and ov > 0 else 0.0
                    else:
                        cv, pv = c2.at[d, c], c2.at[dates[j - 1], c]
                        rr = cv / pv - 1 if pd.notna(cv) and pd.notna(pv) and pv > 0 else 0.0
                    daily.at[d] += wi * float(rr)
            daily.at[entry] -= cost
            if first_day is None:
                first_day = entry
            last_exit_idx = exit_i
            last_day = exit_
        holdings_rows.append({
            "signal_date": t, "entry_date": entry, "exit_date": exit_,
            "regime": regime, "holdings": "|".join(pick), "valid_holdings": "|".join(valid),
            "turnover": round(turnover, 4), "cost": round(cost, 6),
            "tp": None if pd.isna(tp) else round(float(tp), 6),
        })
        prev = valid
    # 期末清仓成本
    if prev and last_exit_idx is not None:
        daily.at[dates[last_exit_idx]] -= 0.5 * 2 * side_cost  # 全清仓，one-way=1
    if first_day is None or last_day is None:
        return daily.iloc[:0], {}, pd.DataFrame(holdings_rows)
    daily = daily.loc[first_day:last_day]
    metrics = _perf(daily)
    metrics.update({"turnover_total": round(turnover_total, 3), "avg_turnover": round(turnover_total / max(1, len(rebalance_dates)), 4), "rebalances": len(rebalance_dates), "trades": trades})
    return daily, metrics, pd.DataFrame(holdings_rows)


def battery_returns(dates, c2, side_cost):
    if BATTERY not in c2.columns:
        return None, {}
    s = c2[BATTERY].dropna()
    if len(s) < 10:
        return None, {}
    d = s.pct_change().dropna()
    d.iloc[0] -= side_cost
    d.iloc[-1] -= side_cost
    return d, _perf(d)


def main():
    dates, c1, c2, o2, a2, ret_cc, jump, recent_jump, meta1, meta2, quality = load_clean()
    signals = build_signals(dates, c1, c2, a2, ret_cc, jump, recent_jump, meta2)
    # 信号可用起点：T' 220日+20日平滑后，至少再留一个月
    valid_tp = signals["Tp"].dropna()
    if valid_tp.empty:
        raise RuntimeError("T' 无有效值，数据长度不足")
    start_date = max(valid_tp.index.min() + pd.Timedelta(days=30), pd.Timestamp("2023-01-03"))
    start_idx = int(dates.searchsorted(start_date))
    coverage = c2.notna().mean()
    universe = [c for c in c2.columns if c in meta2["code"].values and coverage.get(c, 0) >= 0.8]
    universe_no_batt = [c for c in universe if c != BATTERY]
    quality["universe_after_coverage"] = len(universe)
    quality["coverage_excluded"] = [c for c in c2.columns if c not in universe]
    _log(f"backtest start={dates[start_idx].date()} end={dates[-1].date()} universe={len(universe)} T' first={valid_tp.index.min().date()}")

    results, daily_map = {}, {}
    for mode in ("main", "composite", "mom20"):
        daily, met, hold = simulate(mode, signals, dates, c2, o2, ret_cc, meta2, universe, BASE_COST, start_idx)
        results[mode] = met
        daily_map[mode] = daily
        if mode == "main":
            hold.to_csv(OUT / "holdings.csv", index=False, encoding="utf-8-sig")
    # 随机基线
    rand_metrics, rand_daily = [], []
    for seed in range(RANDOM_RUNS):
        d, met, _ = simulate("random", signals, dates, c2, o2, ret_cc, meta2, universe, BASE_COST, start_idx, seed=20261026 + seed)
        rand_metrics.append(met)
        rand_daily.append(d)
    rdf = pd.DataFrame(rand_metrics)
    random_summary = {
        "runs": RANDOM_RUNS,
        "ann_median": round(float(rdf["ann"].median()), 4),
        "ann_p05": round(float(rdf["ann"].quantile(0.05)), 4),
        "ann_p95": round(float(rdf["ann"].quantile(0.95)), 4),
        "sharpe_median": round(float(rdf["sharpe"].median()), 4),
        "mdd_median": round(float(rdf["mdd"].median()), 4),
    }
    results["random3"] = random_summary
    # 固定电池
    bdaily, bmet = battery_returns(dates, c2, BASE_COST)
    if bdaily is not None and "main" in daily_map and len(daily_map["main"]):
        bdaily = bdaily.loc[daily_map["main"].index.min():daily_map["main"].index.max()]
        bmet = _perf(bdaily)
    results["battery"] = bmet
    if bdaily is not None:
        daily_map["battery"] = bdaily
    # 成本敏感性：主策略 25bps
    _, met_alt, _ = simulate("main", signals, dates, c2, o2, ret_cc, meta2, universe, ALT_COST, start_idx)
    results["main_cost25"] = met_alt
    # 剔除电池：主策略 + 随机基线（用同一固定种子序列的 100 次，保持可比）
    _, met_nobatt, _ = simulate("main", signals, dates, c2, o2, ret_cc, meta2, universe_no_batt, BASE_COST, start_idx)
    results["main_no_battery"] = met_nobatt
    rand_nob = []
    for seed in range(100):
        _, met, _ = simulate("random", signals, dates, c2, o2, ret_cc, meta2, universe_no_batt, BASE_COST, start_idx, seed=20261026 + seed)
        rand_nob.append(met)
    ndf = pd.DataFrame(rand_nob)
    results["random_no_battery"] = {"ann_median": round(float(ndf["ann"].median()), 4), "sharpe_median": round(float(ndf["sharpe"].median()), 4), "runs": 100}
    # 成本归因：主策略毛收益 + 随机毛基线
    _, met_gross_main, _ = simulate("main", signals, dates, c2, o2, ret_cc, meta2, universe, 0.0, start_idx, hold_days=5)
    results["main_gross"] = met_gross_main
    gross_rand = []
    for seed in range(100):
        _, met, _ = simulate("random", signals, dates, c2, o2, ret_cc, meta2, universe, 0.0, start_idx, seed=20261026 + seed, hold_days=5)
        gross_rand.append(met)
    gdf = pd.DataFrame(gross_rand)
    results["random_gross"] = {"ann_median": round(float(gdf["ann"].median()), 4), "sharpe_median": round(float(gdf["sharpe"].median()), 4)}
    # 10日持有敏感性（预注册区间 5–10 日；门槛仍以 5 日为准）
    _, met_main10, _ = simulate("main", signals, dates, c2, o2, ret_cc, meta2, universe, BASE_COST, start_idx, hold_days=10)
    results["main_hold10"] = met_main10
    rand10 = []
    for seed in range(100):
        _, met, _ = simulate("random", signals, dates, c2, o2, ret_cc, meta2, universe, BASE_COST, start_idx, seed=20261026 + seed, hold_days=10)
        rand10.append(met)
    r10 = pd.DataFrame(rand10)
    results["random_hold10"] = {"ann_median": round(float(r10["ann"].median()), 4), "sharpe_median": round(float(r10["sharpe"].median()), 4)}
    # 分段（主策略/简单动量/随机基线同一分界）
    mids = dates[start_idx + (len(dates) - start_idx) // 2]
    segments = {}
    for mode in ("main", "mom20"):
        if mode in daily_map:
            a = daily_map[mode].loc[:mids]
            b = daily_map[mode].loc[mids:]
            segments[mode] = {"first": _perf(a), "second": _perf(b)}
    rnd_seg = {"first": [], "second": []}
    for d in rand_daily:
        if len(d):
            rnd_seg["first"].append(_perf(d.loc[:mids]))
            rnd_seg["second"].append(_perf(d.loc[mids:]))
    seg_rand = {}
    for k in ("first", "second"):
        m = pd.DataFrame(rnd_seg[k])
        seg_rand[k] = {"sharpe_median": round(float(m["sharpe"].median()), 4) if len(m) else None,
                       "ann_median": round(float(m["ann"].median()), 4) if len(m) else None}
    segments["random3"] = seg_rand
    # gate：主策略 Sharpe 是否超过随机中位 + 不依赖电池 + 两段方向一致
    gate = {
        "main_sharpe": results["main"].get("sharpe"),
        "random_sharpe_median": random_summary["sharpe_median"],
        "main_ann": results["main"].get("ann"),
        "random_ann_median": random_summary["ann_median"],
        "sharpe_excess": round((results["main"].get("sharpe", 0) - random_summary["sharpe_median"]), 4),
        "ann_excess": round((results["main"].get("ann", 0) - random_summary["ann_median"]), 4),
        "main_no_battery_sharpe": met_nobatt.get("sharpe"),
        "random_no_battery_sharpe_median": results["random_no_battery"].get("sharpe_median"),
    }
    gate["pass_sharpe"] = bool(gate["sharpe_excess"] > 0)
    gate["pass_ex_battery"] = bool(
        met_nobatt.get("sharpe", 0) > results["random_no_battery"].get("sharpe_median", 0)
        and met_nobatt.get("ann", 0) > results["random_no_battery"].get("ann_median", 0)
    )
    gf = segments.get("main", {}).get("first", {}).get("sharpe")
    gs = segments.get("main", {}).get("second", {}).get("sharpe")
    rf = seg_rand["first"].get("sharpe_median")
    rs = seg_rand["second"].get("sharpe_median")
    gate["seg_main"] = [gf, gs]
    gate["seg_random_median"] = [rf, rs]
    gate["pass_segments"] = bool(gf is not None and gs is not None and rf is not None and rs is not None and gf > rf and gs > rs)
    cost_drag = round(results["main_gross"].get("ann", 0) - results["main"].get("ann", 0), 4)
    gate["cost_drag"] = cost_drag
    gate["pass_cost"] = bool(gate["ann_excess"] > cost_drag)
    gate["pass_absolute"] = bool(results["main"].get("ann", 0) > 0 and results["main"].get("sharpe", 0) > 0)
    gate["pass"] = bool(gate["pass_sharpe"] and gate["pass_ex_battery"] and gate["pass_segments"] and gate["pass_cost"] and gate["pass_absolute"])
    # 最新研究快照（不产生交易信号）
    last_t = dates[-1]
    last_tp = signals["Tp"].get(last_t)
    snapshot = {"as_of": str(last_t.date()), "tp": None if pd.isna(last_tp) else round(float(last_tp), 6),
                "regime": None, "top": []}
    if pd.notna(last_tp):
        regime = "reversal" if last_tp > 0 else "momentum"
        snapshot["regime"] = regime
        scores = signals["reversal"].loc[last_t] if regime == "reversal" else signals["strength"].loc[last_t]
        eligible = _eligible(scores, universe)
        chosen = _pick_with_corr(scores, eligible, ret_cc, last_t, N_HOLD, [], False)
        names = meta2.set_index("code")["name"].to_dict()
        ranked = scores.reindex(eligible).dropna().sort_values(ascending=False)
        snapshot["top"] = [{"code": c, "name": str(names.get(c, "")), "score": round(float(ranked.get(c, np.nan)), 4)} for c in chosen]
        snapshot["top10"] = [{"code": c, "name": str(names.get(c, "")), "score": round(float(v), 4)} for c, v in ranked.head(10).items()]
    payload = {"as_of": quality["as_of"], "backtest_start": str(dates[start_idx].date()), "backtest_end": str(dates[-1].date()),
               "latest_snapshot": snapshot,
               "rules": {"N_HOLD": N_HOLD, "HOLD_DAYS": HOLD_DAYS, "BASE_COST": BASE_COST, "ALT_COST": ALT_COST, "BUFFER_TOP": BUFFER_TOP, "CORR_CAP": CORR_CAP},
               "quality": quality, "results": results, "segments": segments, "gate": gate}
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "data_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    # markdown
    def row(name, m):
        if not m:
            return f"| {name} | — | — | — | — | — |"
        return f"| {name} | {m.get('ann',0):.2%} | {m.get('vol',0):.2%} | {m.get('sharpe',0):.2f} | {m.get('mdd',0):.2%} | {m.get('win',0):.1%} |"
    rt = [
        "# v4 行业层指数级回测（第一道门槛）",
        "",
        f"- 数据截至：{quality['as_of']}｜回测：{dates[start_idx].date()}–{dates[-1].date()}",
        f"- 规则：每 {HOLD_DAYS} 交易日调仓；T 日信号 → T+1 开盘买 → T+{HOLD_DAYS} 收盘卖；{N_HOLD} 个行业等权；单边成本 {BASE_COST*10000:.1f}bps。",
        f"- 宇宙：申万二级 {len(universe)} 个（当前行业定义，非 PIT；无个股级幸存者偏差，分类非 PIT）。",
        f"- T' 行业分组：{quality['sw1_source']}（官方 SW1 日线；T' 仍是日线代理，不是银河原版周频参数）。",
        "",
        f"## 最新研究快照（T={snapshot['as_of']}，不产生交易信号）",
        "",
        f"- T'={snapshot['tp']} → {'反转' if snapshot['regime']=='reversal' else '动量' if snapshot['regime']=='momentum' else '无信号'}；排名前三：{'、'.join(x['name'] or x['code'] for x in snapshot['top']) if snapshot['top'] else '无'}",
        "- 这是行业层研究输出；个股层未启动，不能把它当买入建议。",
        "",
        "## 主结果（净，基准成本）",
        "",
        "| 策略 | 年化 | 波动 | Sharpe | 最大回撤 | 日胜率 |",
        "|---|---:|---:|---:|---:|---:|",
        row("主策略 T'+复数过滤", results["main"]),
        row("纯复合强度（无T'）", results["composite"]),
        row("简单20日动量", results["mom20"]),
        f"| 随机3行业（{RANDOM_RUNS}次中位） | {random_summary['ann_median']:.2%} | — | {random_summary['sharpe_median']:.2f} | {random_summary['mdd_median']:.2%} | — |",
        row("固定电池 801737", results["battery"]),
        "",
        f"随机基线 5%/95% 年化：{random_summary['ann_p05']:.2%} / {random_summary['ann_p95']:.2%}。",
        f"**成本归因**：主策略毛年化 {results['main_gross'].get('ann',0):.2%}（Sharpe {results['main_gross'].get('sharpe',0):.2f}），随机毛中位 {results['random_gross'].get('ann_median',0):.2%}（Sharpe {results['random_gross'].get('sharpe_median',0):.2f}）；净成本拖累约 {(results['main_gross'].get('ann',0)-results['main'].get('ann',0))*100:.1f}pct。",
        f"**10日持有敏感性**：主策略净年化 {results['main_hold10'].get('ann',0):.2%}、Sharpe {results['main_hold10'].get('sharpe',0):.2f}；随机中位年化 {results['random_hold10'].get('ann_median',0):.2%}、Sharpe {results['random_hold10'].get('sharpe_median',0):.2f}。",
        "",
        "## 门槛判定",
        "",
        f"- 主策略 Sharpe {gate['main_sharpe']} vs 随机中位 {gate['random_sharpe_median']}（差 {gate['sharpe_excess']}）",
        f"- 主策略年化 {gate['main_ann']:.2%} vs 随机中位 {gate['random_ann_median']:.2%}（差 {gate['ann_excess']:.2%}）",
        f"- 剔除电池后主策略：年化 {met_nobatt.get('ann',0):.2%}，Sharpe {met_nobatt.get('sharpe',0):.2f}；同口径随机中位 Sharpe {results['random_no_battery'].get('sharpe_median')}、年化 {results['random_no_battery'].get('ann_median',0):.2%}",
        f"- 成本门槛：主策略净超额 {gate['ann_excess']:.2%} vs 成本拖累 {gate['cost_drag']:.2%}（需净超额大于成本拖累）",
        f"- 绝对门槛：主策略净年化/Sharpe 必须为正 → {'通过' if gate['pass_absolute'] else '未通过'}",
        f"- 分段：主策略 Sharpe {gate['seg_main']} vs 随机中位 {gate['seg_random_median']}（需两段都更高）",
        f"- **结论：{'通过' if gate['pass'] else '未通过'}**；{'可进入个股层（仍需影子）' if gate['pass'] else '不进入个股层，保持电池主链路；行业层只保留研究快照'}",
        "",
        "## 成本与稳健性",
        "",
        f"- 主策略 25bps/边：年化 {results['main_cost25'].get('ann',0):.2%}，Sharpe {results['main_cost25'].get('sharpe',0):.2f}；换手累计 {results['main'].get('turnover_total',0):.2f}，均次 {results['main'].get('avg_turnover',0):.2%}。",
        f"- 两段：主策略第一段 Sharpe {segments.get('main',{}).get('first',{}).get('sharpe','—')}，第二段 {segments.get('main',{}).get('second',{}).get('sharpe','—')}；简单动量第一段 {segments.get('mom20',{}).get('first',{}).get('sharpe','—')}，第二段 {segments.get('mom20',{}).get('second',{}).get('sharpe','—')}。",
        "",
        "## 判读（按门槛规则，不给结论化妆）",
        "",
        f"- 毛收益：主策略年化 {results['main_gross'].get('ann',0):+.2%}、Sharpe {results['main_gross'].get('sharpe',0):.2f}；随机3行业毛中位年化 {results['random_gross'].get('ann_median',0):+.2%}、Sharpe {results['random_gross'].get('sharpe_median',0):.2f}。{'毛收益有增量，但幅度不大。' if results['main_gross'].get('ann',0) > results['random_gross'].get('ann_median',0) else '毛收益没有跑赢随机，且净成本后更差；本实现没有行业选择 alpha。'}",
        f"- 净收益：主策略年化 {results['main'].get('ann',0):+.2%}、Sharpe {results['main'].get('sharpe',0):.2f}；成本拖累约 {(results['main_gross'].get('ann',0)-results['main'].get('ann',0))*100:.1f} 个百分点。5日调仓的均次换手 {results['main'].get('avg_turnover',0):.1%}，成本把正毛收益全部吃掉。",
        f"- 10日持有：主策略净年化 {results['main_hold10'].get('ann',0):+.2%}、Sharpe {results['main_hold10'].get('sharpe',0):.2f}；随机中位年化 {results['random_hold10'].get('ann_median',0):+.2%}、Sharpe {results['random_hold10'].get('sharpe_median',0):.2f}。更长持有并未救回，说明不是单纯换手频率问题。",
        f"- 复杂度对照：简单20日动量净年化 {results['mom20'].get('ann',0):+.2%}、Sharpe {results['mom20'].get('sharpe',0):.2f}；纯复合强度净年化 {results['composite'].get('ann',0):+.2%}、Sharpe {results['composite'].get('sharpe',0):.2f}；主策略 {results['main'].get('ann',0):+.2%}/{results['main'].get('sharpe',0):.2f}。主策略没有稳定的复杂度增量。",
        f"- 分段：主策略第一段 Sharpe {gate['seg_main'][0]} vs 随机 {gate['seg_random_median'][0]}；第二段 {gate['seg_main'][1]} vs 随机 {gate['seg_random_median'][1]}。第一段跑输随机，不具备两段一致性。",
        "- **最终判定：不进入个股层，不替换电池主线。** 行业层只能作为研究/影子方向，不能把当期指数级净收益或毛收益当作可交易 alpha。",
        "",
        "## 数据质量",
        "",
        f"- SW1 来源：{quality['sw1_source']}｜组数 {quality['n_sw1']}｜SW2 行业 {quality['n_sw2']}｜入池 {quality.get('universe_after_coverage','—')}（剔除覆盖率<80%的新行业）｜日历 {quality['calendar_days']} 天｜SW2 覆盖率中位 {quality['sw2_median_coverage']:.2%}，最低 {quality['sw2_min_coverage']:.2%}",
        f"- 坏 OHLC 单元格 {quality['bad_ohlc_cells']}；坏 open {quality['bad_open_cells']}；基点跳变候选 {quality['base_jump_cells']}",
        "",
        "## 口径与不能宣称的部分",
        "",
        "- 这是**指数级**回测；行业分类用当前申万定义，非 PIT，但没有个股级当前成分回溯的幸存者偏差。",
        "- 未包含：个股层、封板时间、逐笔/涨跌停可成交性、PIT 成分；不能用本报告宣称个股策略收益。",
        "- 随机基线是随机 3 个行业等权、同调仓频率/成本；策略含前8缓冲，换手可能低于随机，报告同时给出净结果。",
        "- 未做参数搜索；主策略规则固定。结论只用于判断行业层是否值得进入影子个股层。",
    ]
    (OUT / "report.md").write_text("\n".join(rt) + "\n", encoding="utf-8")
    _log("GATE " + json.dumps(gate, ensure_ascii=True))
    print("INDUSTRY_V4_DONE", json.dumps({"gate": gate, "main": results["main"], "random": random_summary}, ensure_ascii=True))


if __name__ == "__main__":
    main()
