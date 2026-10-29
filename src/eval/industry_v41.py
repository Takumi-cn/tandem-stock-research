"""v4.1-industry 预注册挑战者（指数级，单次 OOS 试验）。

这是 v4 之外的独立试验，不修改 v4 报告，不写 latest.json，不进入个股层。
规则：
- 官方 SW1（31）+ SW2（131），入池覆盖>=80% 的行业；T' 用官方 SW1 日线代理。
- 标签：T+1 开盘 -> T+5 收盘（与执行完全一致），训练按 label_end < 测试起点 purge。
- 特征：约 25 个低维收益/波动/父行业相对/市场相对/横截面排名/市场状态特征。
- 模型：Bayesian Ridge、RandomForest(depth4)、LightGBM rank_xendcg。
- 组合：Top3 等权，5 日调仓，15bps/边，OOS 随机 3 行业 500 次、简单20日动量、固定电池。
- 单次 train/test OOS（训练 2023-04~2024-12，测试 2025-01~2026-10），不是完整 walk-forward；
  PIT 成分、CatBoost、5000 次随机、DSR/PBO 未完成，报告中逐项标注。
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
from src.eval.industry_rotation import _perf, load_clean  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

OUT = ROOT / "reports" / "industry_v41"
TEST_START = pd.Timestamp("2025-01-02")
TRAIN_START = pd.Timestamp("2023-04-03")
COST_BPS = 15.0
ALT_COST_BPS = 25.0
TOP_K = 3
HOLD_DAYS = 5
RANDOM_RUNS = 500
RANDOM_SEED = 20261026
BATTERY = "801737"


def _metrics_from_tranches(ret: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = ret.dropna()
    if len(r) == 0:
        return {}
    eq = (1 + r).cumprod()
    ann = float(eq.iloc[-1] ** (252 / (len(r) * HOLD_DAYS)) - 1)
    vol = float(r.std(ddof=1) * math.sqrt(252 / HOLD_DAYS)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252 / HOLD_DAYS)) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min())
    return {"ann": round(ann, 4), "vol": round(vol, 4), "sharpe": round(sharpe, 4),
            "mdd": round(mdd, 4), "win": round(float((r > 0).mean()), 4), "n_periods": int(len(r)),
            "turnover": round(float(turnover.mean()), 4) if turnover is not None and len(turnover) else None}


def _portfolio(score_map: dict, dates, c2, o2, universe: list[str], side_cost: float, start_idx: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """score_map[signal_date] = Series(code->score)；5日持有，T+1 open -> T+5 close。"""
    rets, turns, periods = [], [], []
    prev: set[str] = set()
    signal_dates = []
    for i in range(start_idx, len(dates) - HOLD_DAYS, HOLD_DAYS):
        t = dates[i]
        if t not in score_map:
            continue
        s = score_map[t].reindex(universe).dropna()
        if len(s) == 0:
            continue
        pick = list(s.sort_values(ascending=False).head(TOP_K).index)
        entry, exit_ = dates[i + 1], dates[i + HOLD_DAYS]
        gross = []
        for c in pick:
            try:
                rr = float(c2.at[exit_, c] / o2.at[entry, c] - 1)
                if not np.isfinite(rr):
                    rr = 0.0
            except Exception:  # noqa: BLE001
                rr = 0.0
            gross.append(rr)
        turn = 0.5 * len(set(pick) ^ prev) / max(1, TOP_K) if prev else 0.5
        # 权重变化的近似 one-way turnover；初始买入 0.5，随后按换名比例
        if prev:
            turn = 0.5 * len(set(pick) - prev) / max(1, len(pick))
        cost = turn * 2 * side_cost
        rets.append(float(np.mean(gross)) - cost)
        turns.append(turn)
        periods.append(t)
        prev = set(pick)
        signal_dates.append(t)
    return pd.Series(rets, index=pd.DatetimeIndex(signal_dates)), pd.Series(turns, index=pd.DatetimeIndex(signal_dates)), pd.Series(periods, index=pd.DatetimeIndex(signal_dates))


def _model_predictions(data: pd.DataFrame, feature_cols: list[str], test_signals: list[pd.Timestamp], universe: list[str]):
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import BayesianRidge

    train = data[(data["date"] >= TRAIN_START) & (data["label_end"] < TEST_START) & data["y_oc5"].notna()].copy()
    train = train[train["date"] < TEST_START].sort_values(["date", "code"])
    if len(train) < 5000:
        raise RuntimeError(f"training rows too few: {len(train)}")
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(train[feature_cols])
    ytr = train["y_oc5"].astype(float).clip(-0.2, 0.2).values
    # 1) Bayesian Ridge
    br = BayesianRidge()
    br.fit(Xtr, ytr)
    # 2) Random Forest，固定小深度
    rf = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=40,
                               max_features=0.7, n_jobs=8, random_state=RANDOM_SEED)
    rf.fit(Xtr, ytr)
    # 3) LightGBM rank_xendcg：按日 group，标签为截面 rank（整数）
    tr = train.copy()
    tr["rank_int"] = (tr.groupby("date")["y_oc5"].rank(method="first", pct=True) * 30).round().astype(int)
    groups = tr.groupby("date", sort=True).size().astype(int).tolist()
    lr = lgb.LGBMRanker(
        objective="rank_xendcg", n_estimators=250, learning_rate=0.03,
        num_leaves=15, max_depth=4, min_child_samples=40, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0, n_jobs=8, verbose=-1,
        random_state=RANDOM_SEED,
    )
    lr.fit(imp.transform(tr[feature_cols]), tr["rank_int"].values, group=groups)
    pred = {"ridge": {}, "rf": {}, "lgbm": {}}
    for t in test_signals:
        row = data[data["date"] == t].copy()
        if row.empty:
            continue
        row = row[row["code"].isin(universe)]
        if row.empty:
            continue
        X = pd.DataFrame(imp.transform(row[feature_cols]), columns=feature_cols, index=row.index)
        pred["ridge"][t] = pd.Series(br.predict(X), index=row["code"])
        pred["rf"][t] = pd.Series(rf.predict(X), index=row["code"])
        pred["lgbm"][t] = pd.Series(lr.predict(X), index=row["code"])
    # ensemble = 每个模型截面 rank 的均值
    ens = {}
    for t in test_signals:
        if all(t in pred[m] for m in pred):
            frames = [pred[m][t].rank(pct=True) for m in pred]
            ens[t] = pd.concat(frames, axis=1).mean(axis=1)
    return {"individual": pred, "ensemble": ens, "train_rows": int(len(tr)),
            "train_start": str(train["date"].min().date()), "train_end": str(train["date"].max().date())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dates, c1, c2, o2, a2, ret_cc, jump, recent_jump, meta1, meta2, quality = load_clean()
    universe = [c for c in c2.columns if c in meta2["code"].astype(str).values and c2[c].notna().mean() >= 0.8]
    universe_no_batt = [c for c in universe if c != BATTERY]
    # 官方 SW1 父行业映射
    m1 = meta1.copy()
    m1["code"] = m1["code"].astype(str)
    m1["name"] = m1["name"].astype(str)
    parent_code = m1.set_index("name")["code"].to_dict()
    parent = dict(zip(meta2["code"].astype(str), meta2["parent"].astype(str)))
    # 基础收益
    ret1 = c2.pct_change(fill_method=None)
    ret5 = c2 / c2.shift(5) - 1
    ret20 = c2 / c2.shift(20) - 1
    ret60 = c2 / c2.shift(60) - 1
    ret3 = c2 / c2.shift(3) - 1
    ret10 = c2 / c2.shift(10) - 1
    vol5 = ret1.rolling(5).std()
    vol20 = ret1.rolling(20).std()
    dd20 = c2 / c2.rolling(20).max() - 1
    gap = o2 / c2.shift(1) - 1
    amount_ratio = a2.rolling(5).mean() / a2.rolling(20).mean()
    # 官方 SW1 父行业与市场
    mret = ret1.mean(axis=1)
    mret5 = c1.pct_change(5)
    mret20 = c1.pct_change(20)
    parent_rel5 = pd.DataFrame(index=c2.index, columns=c2.columns, dtype=float)
    parent_rel20 = pd.DataFrame(index=c2.index, columns=c2.columns, dtype=float)
    for c in universe:
        p = parent.get(c, "")
        pc = parent_code.get(p)
        if pc in c1.columns:
            pr5 = c1[pc].pct_change(5)
            pr20 = c1[pc].pct_change(20)
        else:
            pr5 = mret.rolling(5).sum()
            pr20 = mret.rolling(20).sum()
        parent_rel5[c] = ret5[c] - pr5
        parent_rel20[c] = ret20[c] - pr20
    market_ret5 = mret.rolling(5).sum()
    market_ret20 = mret.rolling(20).sum()
    market_vol20 = mret.rolling(20).std()
    market_disp = ret20.std(axis=1)
    market_up = (ret1 > 0).mean(axis=1)
    # 官方 SW1 T' 日线代理
    r20_1 = c1 / c1.shift(20) - 1
    rank20_1 = r20_1.rank(axis=1, pct=True)
    T = (rank20_1 - rank20_1.shift(220)).abs().mean(axis=1)
    tz = T / T.rolling(220, min_periods=60).std()
    tp = tz.rolling(20, min_periods=5).mean().diff()
    feature_wide = {
        "ret_1": ret1, "ret_3": ret3, "ret_5": ret5, "ret_10": ret10, "ret_20": ret20, "ret_60": ret60,
        "vol_5": vol5, "vol_20": vol20, "dd_20": dd20,
        "gap": gap, "amount_ratio": amount_ratio,
        "parent_rel_5": parent_rel5, "parent_rel_20": parent_rel20,
        "market_rel_5": ret5.sub(market_ret5, axis=0),
        "market_rel_20": ret20.sub(market_ret20, axis=0),
        "rank_ret5": ret5.rank(axis=1, pct=True),
        "rank_ret20": ret20.rank(axis=1, pct=True),
        "rank_vol": vol20.rank(axis=1, pct=True),
        "rank_rel5": parent_rel5.rank(axis=1, pct=True),
        "rank_change": ret20.rank(axis=1, pct=True).shift(1),
    }
    feature_wide["rank_change"] = feature_wide["rank_ret20"] - feature_wide["rank_change"]
    for c in feature_wide:
        s = feature_wide[c]
        if isinstance(s, pd.DataFrame):
            pass
    # 市场状态广播
    for name, ser in (("mkt_ret20", market_ret20), ("mkt_vol20", market_vol20),
                      ("mkt_disp", market_disp), ("mkt_up", market_up), ("tp", tp)):
        feature_wide[name] = pd.DataFrame(np.repeat(ser.values[:, None], len(universe), axis=1), index=c2.index, columns=universe)
    feature_cols = ["ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
                    "vol_5", "vol_20", "dd_20", "gap", "amount_ratio",
                    "parent_rel_5", "parent_rel_20", "market_rel_5", "market_rel_20",
                    "rank_ret5", "rank_ret20", "rank_vol", "rank_rel5", "rank_change",
                    "mkt_ret20", "mkt_vol20", "mkt_disp", "mkt_up", "tp"]
    # 连续 5 日精确标签：T+1 open -> T+5 close
    y_oc5 = c2.shift(-5) / o2.shift(-1) - 1
    long = None
    for k, v in feature_wide.items():
        s = v.stack().rename(k).reset_index()
        s.columns = ["date", "code", k]
        long = s if long is None else long.merge(s, on=["date", "code"], how="outer")
    lab = y_oc5.stack().rename("y_oc5").reset_index()
    lab.columns = ["date", "code", "y_oc5"]
    long = long.merge(lab, on=["date", "code"], how="outer")
    long = long[long["code"].isin(universe)].copy()
    date_idx = {d: i for i, d in enumerate(dates)}
    long["label_end"] = long["date"].map(lambda d: dates[date_idx[d] + 5] if d in date_idx and date_idx[d] + 5 < len(dates) else pd.NaT)
    test_start_idx = int(dates.searchsorted(TEST_START))
    last_exit_idx = len(dates) - HOLD_DAYS - 1
    test_signals = [dates[i] for i in range(test_start_idx, last_exit_idx + 1) if (i - test_start_idx) % HOLD_DAYS == 0]
    if not test_signals:
        raise RuntimeError("no test signals")
    bundle = _model_predictions(long, feature_cols, test_signals, universe)
    pred = bundle["individual"]
    ens = bundle["ensemble"]
    # 缓存 OOS 分数与面板，供实验 A/B/C 复用（不重新训练，不写实盘）
    score_rows = []
    for dt, s in pred["lgbm"].items():
        for code, val in s.items():
            score_rows.append({"date": dt, "code": str(code), "score_lgbm": float(val)})
    pd.DataFrame(score_rows).to_parquet(OUT / "lgbm_scores.parquet", index=False)
    long.to_parquet(OUT / "panel_v41.parquet", index=False)
    # 回测：模型、集成、简单动量、电池、随机
    def score_series_from_map(mp):
        return {t: s for t, s in mp.items()}
    results, daily = {}, {}
    for name, mp in (("ridge", pred["ridge"]), ("rf", pred["rf"]), ("lgbm", pred["lgbm"]), ("ensemble", ens)):
        r, turns, periods = _portfolio(mp, dates, c2, o2, universe, COST_BPS / 10000, test_start_idx)
        results[name] = _metrics_from_tranches(r, turns)
        daily[name] = r
    # 简单20日动量
    mom_scores = {}
    for t in test_signals:
        if t in ret20.index:
            mom_scores[t] = ret20.loc[t].reindex(universe).dropna()
    r_mom, turn_mom, _ = _portfolio(mom_scores, dates, c2, o2, universe, COST_BPS / 10000, test_start_idx)
    results["mom20"] = _metrics_from_tranches(r_mom, turn_mom)
    # 25bps 应力
    r_ens_alt, _, _ = _portfolio(ens, dates, c2, o2, universe, ALT_COST_BPS / 10000, test_start_idx)
    results["ensemble_cost25"] = _metrics_from_tranches(r_ens_alt)
    # 电池固定（同一信号节奏滚动持有）
    batt_scores = {t: pd.Series({BATTERY: 1.0}) for t in test_signals if BATTERY in c2.columns}
    r_batt, _, _ = _portfolio(batt_scores, dates, c2, o2, [BATTERY], COST_BPS / 10000, test_start_idx)
    results["battery"] = _metrics_from_tranches(r_batt)
    # 随机 3 行业 500 次，同信号/成本
    rng = np.random.default_rng(RANDOM_SEED)
    rand_metrics, rand_returns = [], []
    for _ in range(RANDOM_RUNS):
        mp = {}
        for t in test_signals:
            choices = rng.choice(universe, size=min(TOP_K, len(universe)), replace=False)
            mp[t] = pd.Series(1.0, index=choices)
        r, turns, _ = _portfolio(mp, dates, c2, o2, universe, COST_BPS / 10000, test_start_idx)
        rand_metrics.append(_metrics_from_tranches(r, turns))
        rand_returns.append(r)
    rdf = pd.DataFrame(rand_metrics)
    results["random3"] = {"runs": RANDOM_RUNS,
                          "ann_median": round(float(rdf["ann"].median()), 4),
                          "sharpe_median": round(float(rdf["sharpe"].median()), 4),
                          "ann_p05": round(float(rdf["ann"].quantile(.05)), 4),
                          "ann_p95": round(float(rdf["ann"].quantile(.95)), 4)}
    # 剔除电池的集成
    ens_nb = {}
    for t, s in ens.items():
        ens_nb[t] = s.reindex(universe_no_batt).dropna()
    r_nb, _, _ = _portfolio(ens_nb, dates, c2, o2, universe_no_batt, COST_BPS / 10000, test_start_idx)
    results["ensemble_no_battery"] = _metrics_from_tranches(r_nb)
    rand_nb = []
    for _ in range(200):
        mp = {}
        for t in test_signals:
            choices = rng.choice(universe_no_batt, size=min(TOP_K, len(universe_no_batt)), replace=False)
            mp[t] = pd.Series(1.0, index=choices)
        r, _, _ = _portfolio(mp, dates, c2, o2, universe_no_batt, COST_BPS / 10000, test_start_idx)
        rand_nb.append(_metrics_from_tranches(r))
    rnb = pd.DataFrame(rand_nb)
    results["random3_no_battery"] = {"ann_median": round(float(rnb["ann"].median()), 4), "sharpe_median": round(float(rnb["sharpe"].median()), 4)}
    # 主模型：LightGBM Ranker 的完整稳健性检查
    lgb_scores = pred["lgbm"]
    r_lgb, turn_lgb, _ = _portfolio(lgb_scores, dates, c2, o2, universe, COST_BPS / 10000, test_start_idx)
    lgb_nb_scores = {t: s.reindex(universe_no_batt).dropna() for t, s in lgb_scores.items()}
    r_lgb_nb, _, _ = _portfolio(lgb_nb_scores, dates, c2, o2, universe_no_batt, COST_BPS / 10000, test_start_idx)
    results["lgbm_no_battery"] = _metrics_from_tranches(r_lgb_nb)
    r_lgb25, _, _ = _portfolio(lgb_scores, dates, c2, o2, universe, ALT_COST_BPS / 10000, test_start_idx)
    results["lgbm_cost25"] = _metrics_from_tranches(r_lgb25)
    r_lgb_gross, _, _ = _portfolio(lgb_scores, dates, c2, o2, universe, 0.0, test_start_idx)
    results["lgbm_gross"] = _metrics_from_tranches(r_lgb_gross)
    # 随机 25bps / 毛收益基线，用于同口径比较
    rand25, randg = [], []
    for i in range(200):
        mp = {}
        for t in test_signals:
            choices = rng.choice(universe, size=min(TOP_K, len(universe)), replace=False)
            mp[t] = pd.Series(1.0, index=choices)
        r25, _, _ = _portfolio(mp, dates, c2, o2, universe, ALT_COST_BPS / 10000, test_start_idx)
        rg, _, _ = _portfolio(mp, dates, c2, o2, universe, 0.0, test_start_idx)
        rand25.append(_metrics_from_tranches(r25)); randg.append(_metrics_from_tranches(rg))
    results["random3_cost25"] = {"ann_median": round(float(pd.DataFrame(rand25)["ann"].median()), 4), "sharpe_median": round(float(pd.DataFrame(rand25)["sharpe"].median()), 4)}
    results["random3_gross"] = {"ann_median": round(float(pd.DataFrame(randg)["ann"].median()), 4), "sharpe_median": round(float(pd.DataFrame(randg)["sharpe"].median()), 4)}
    # 分段：LGBM 与随机都按相同信号位置切
    mid = len(test_signals) // 2
    segments = {}
    for name, sr in (("ensemble", daily["ensemble"]), ("lgbm", r_lgb), ("mom20", r_mom)):
        a, b = sr.iloc[:mid], sr.iloc[mid:]
        segments[name] = {"first": _metrics_from_tranches(a), "second": _metrics_from_tranches(b)}
    rseg = {"first": [], "second": []}
    for r in rand_returns:
        if len(r) >= mid:
            rseg["first"].append(_metrics_from_tranches(r.iloc[:mid]))
            rseg["second"].append(_metrics_from_tranches(r.iloc[mid:]))
    segments["random3"] = {k: {"sharpe_median": round(float(pd.DataFrame(v)["sharpe"].median()), 4), "ann_median": round(float(pd.DataFrame(v)["ann"].median()), 4)} for k, v in rseg.items()}
    # 预先指定的主模型是 LightGBM Ranker；Ridge/RF 是 challenger，等权集成不是主门槛。
    gate = {
        "primary": "lgbm",
        "lgbm_ann": results["lgbm"].get("ann"), "random_ann_median": results["random3"]["ann_median"],
        "lgbm_sharpe": results["lgbm"].get("sharpe"), "random_sharpe_median": results["random3"]["sharpe_median"],
        "lgbm_gross_ann": results["lgbm_gross"].get("ann"), "random_gross_ann_median": results["random3_gross"]["ann_median"],
        "lgbm_nb_ann": results["lgbm_no_battery"].get("ann"), "random_nb_ann_median": results["random3_no_battery"]["ann_median"],
        "lgbm_nb_sharpe": results["lgbm_no_battery"].get("sharpe"), "random_nb_sharpe_median": results["random3_no_battery"]["sharpe_median"],
        "lgbm_25_sharpe": results["lgbm_cost25"].get("sharpe"), "random_25_sharpe_median": results["random3_cost25"]["sharpe_median"],
        "seg_lgbm": [segments["lgbm"]["first"].get("sharpe"), segments["lgbm"]["second"].get("sharpe")],
        "seg_random": [segments["random3"]["first"].get("sharpe_median"), segments["random3"]["second"].get("sharpe_median")],
        "ensemble_sharpe_artifact": results["ensemble"].get("sharpe"),
    }
    gate["pass_sharpe"] = bool(gate["lgbm_sharpe"] > gate["random_sharpe_median"])
    gate["pass_abs"] = bool(gate["lgbm_ann"] > 0 and gate["lgbm_sharpe"] > 0)
    gate["pass_ex_battery"] = bool(gate["lgbm_nb_ann"] > gate["random_nb_ann_median"] and gate["lgbm_nb_sharpe"] > gate["random_nb_sharpe_median"])
    gate["pass_segments"] = bool(gate["seg_lgbm"][0] > gate["seg_random"][0] and gate["seg_lgbm"][1] > gate["seg_random"][1])
    gate["pass_cost25"] = bool(gate["lgbm_25_sharpe"] > gate["random_25_sharpe_median"])
    gate["pass_vs_mom"] = bool(gate["lgbm_sharpe"] > results["mom20"].get("sharpe", -9))
    gate["pass"] = bool(gate["pass_sharpe"] and gate["pass_abs"] and gate["pass_ex_battery"] and gate["pass_segments"] and gate["pass_cost25"] and gate["pass_vs_mom"])
    registry = OUT / "registry.csv"
    row = pd.DataFrame([{
        "experiment_id": "IND-V41-EXP-002", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hypothesis": "PRIMARY LightGBM rank_xendcg, exact OC5 labels, low-dim features; Ridge/RF challengers; equal ensemble is an artifact",
        "train": f"{bundle['train_start']}~{bundle['train_end']}", "test": f"{test_signals[0].date()}~{test_signals[-1].date()}",
        "universe": len(universe), "label": "T+1_open_to_T+5_close", "cost_bps": COST_BPS,
        "random_runs": RANDOM_RUNS, "pit": "false", "walk_forward": "false", "status": "EXPLORATORY",
        "lgbm_ann": results["lgbm"].get("ann"), "lgbm_sharpe": results["lgbm"].get("sharpe"),
        "random_ann_median": results["random3"]["ann_median"], "random_sharpe_median": results["random3"]["sharpe_median"],
        "lgbm_no_battery_sharpe": results["lgbm_no_battery"].get("sharpe"),
        "lgbm_cost25_sharpe": results["lgbm_cost25"].get("sharpe"),
        "gate_pass": gate["pass"],
    }])
    registry.parent.mkdir(parents=True, exist_ok=True)
    if registry.exists():
        old = pd.read_csv(registry)
        old = old[old["experiment_id"] != "IND-V41-EXP-002"]
        pd.concat([old, row], ignore_index=True).to_csv(registry, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(registry, index=False, encoding="utf-8-sig")
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "data_asof": quality["as_of"],
               "train_rows": bundle["train_rows"], "train_start": bundle["train_start"], "train_end": bundle["train_end"],
               "test_signals": [str(x.date()) for x in test_signals],
               "features": feature_cols, "results": results, "segments": segments, "gate": gate,
               "caveats": ["single fixed OOS split, not walk-forward", "non-PIT classification/constituents", "no PIT breadth", "CatBoost/5000 random/DSR not implemented"]}
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# v4.1-industry 挑战者（单次 OOS，指数级）", "",
        f"- 数据：{quality['as_of']}；官方 SW1 31 / SW2 131；入池 {len(universe)}；标签 T+1 open→T+5 close。",
        f"- 训练 {bundle['train_start']}~{bundle['train_end']}；测试 {test_signals[0].date()}~{test_signals[-1].date()}；Top3 等权；15bps/边；随机 {RANDOM_RUNS} 次。",
        "- 组合：T 日信号 → T+1 开盘买入 → T+5 收盘卖出；训练按 label_end < 测试起点 purge。",
        "",
        "## 结果（净，基本成本）", "",
        "| 组合 | 年化 | Sharpe | 最大回撤 | 胜率 |", "|---|---:|---:|---:|---:|",
    ]
    for name in ("lgbm", "ridge", "rf", "ensemble", "mom20", "battery"):
        m = results[name]
        lines.append(f"| {name} | {m.get('ann',0):+.2%} | {m.get('sharpe',0):.2f} | {m.get('mdd',0):.2%} | {m.get('win',0):.1%} |")
    lines += [f"| random3 median | {results['random3']['ann_median']:+.2%} | {results['random3']['sharpe_median']:.2f} | — | — |",
              f"| LGBM 剔除电池 | {results['lgbm_no_battery'].get('ann',0):+.2%} | {results['lgbm_no_battery'].get('sharpe',0):.2f} | — | — |",
              f"| LGBM 25bps | {results['lgbm_cost25'].get('ann',0):+.2%} | {results['lgbm_cost25'].get('sharpe',0):.2f} | — | — |",
              f"| LGBM 毛收益 | {results['lgbm_gross'].get('ann',0):+.2%} | {results['lgbm_gross'].get('sharpe',0):.2f} | — | — |",
              f"| 随机毛中位 | {results['random3_gross']['ann_median']:+.2%} | {results['random3_gross']['sharpe_median']:.2f} | — | — |",
              "", "## Gate（主模型 = LightGBM Ranker）", "",
              f"- 主模型 Sharpe {gate['lgbm_sharpe']} vs 随机中位 {gate['random_sharpe_median']}；净年化 {gate['lgbm_ann']:+.2%} vs {gate['random_ann_median']:+.2%}",
              f"- 绝对为正: {gate['pass_abs']}；剔除电池：年化 {gate['lgbm_nb_ann']:+.2%} vs 随机 {gate['random_nb_ann_median']:+.2%}，Sharpe {gate['lgbm_nb_sharpe']} vs {gate['random_nb_sharpe_median']}",
              f"- 25bps/边：Sharpe {gate['lgbm_25_sharpe']} vs 随机 25bps {gate['random_25_sharpe_median']}",
              f"- 分段 Sharpe {gate['seg_lgbm']} vs 随机分段 {gate['seg_random']}",
              f"- 对简单动量 Sharpe: {gate['pass_vs_mom']}；等权集成（非主模型）Sharpe {gate['ensemble_sharpe_artifact']}",
              f"- **结论：{'通过' if gate['pass'] else '未通过'}**；未通过则不进入个股层、不替换电池主线；通过也只进入后续验证，不自动上线。",
              "",
              "## 未完成/不能宣称", "",
              "- 单次 OOS，不是 walk-forward；PIT 成分/宽度未接；CatBoost、5000 次随机、DSR/PBO 未实现。",
              "- 指数级结果；不产生个股买入信号；不写 latest.json。",
              "- 随机/动量/模型使用同一日历与成本，但模型不是完整机构级执行/容量模型。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("V41_DONE", json.dumps({"gate": gate, "lgbm": results["lgbm"], "random": results["random3"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
