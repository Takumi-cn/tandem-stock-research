"""完整预测管线（GUI 入口，目标 10–20 分钟，进度事件 → reports/runs/progress.jsonl）。

成员与用途：
- L1(+Optuna轻调参10 trials，最优参数持久化 config/l1_best.json —— 自学习闭环之一)
- NonStat ×3 种子 / TGNS-lite ×2 / ITrans-lite ×2（种子集成降方差）
- Kronos-Base 零样本（vendors/Kronos + 本地权重；分词器缺失时优雅跳过）
- MR20 / REV5（L0 族）
- 短线上涨概率：LightGBM 二分类 + Huber 预期收益，第一页只按上涨概率选股
集成：状态路由权重（今日桶）+ 家族份额（NonStat α、TGNS/ITrans/Kronos 探索 ε）
输出：latest.json（GUI 数据源，含"为什么"小结）+ reports/<日期>.md + 台账 + 工作日志 RUN 行
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.update import step_data  # noqa: E402
from src.features.build import FEATURE_COLS  # noqa: E402
from src.models.deep_common import DEVICE, DayDataset  # noqa: E402
from src.models.itransformer_lite import ITransformerLite  # noqa: E402
from src.models.l1_lgbm import PARAMS  # noqa: E402
from src.models.nonstat import NonStatNet  # noqa: E402
from src.models.tgns_lite import TgnsLite  # noqa: E402
from src.report.daily import load_names, load_panel, trend_state  # noqa: E402
from src.report.ledger import fill_reviews, stale_note, upsert  # noqa: E402
from src.report.runlog import append_run  # noqa: E402
from src.report.short_pick import TRADE_P, UpModel, confidence_of, oos_backtest, rank_first_page, risk_levels  # noqa: E402
from src.utils.config import PARQUET_DIR, ROOT, targets  # noqa: E402

import lightgbm as lgb  # noqa: E402
import torch  # noqa: E402

SENT_COLS = ["sent_mean", "sent_max", "news_cnt", "kw_pos", "kw_neg"]
PROG = ROOT / "reports" / "runs" / "progress.jsonl"
def _hub_snapshot(repo: str) -> Path | None:
    dirs = sorted((Path.home() / ".cache/huggingface/hub").glob(f"models--{repo.replace('/', '--')}/snapshots/*"))
    return dirs[0] if dirs else None


KRONOS_MODEL = _hub_snapshot("NeoQuasar/Kronos-Base")
BUCKET_DESC = {0: "低波动×趋势下", 1: "低波动×趋势上", 2: "高波动×趋势下", 3: "高波动×趋势上"}


def prog(pct: float, stage: str, msg: str = ""):
    PROG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"pct": round(pct, 2), "stage": stage, "msg": msg, "ts": time.strftime("%H:%M:%S")}, ensure_ascii=False) + "\n")
    print(f"[{pct:5.1f}%] {stage} {msg}", flush=True)


def tune_l1(tr_rows, feats):
    """Optuna 轻调参（最近60日为验证块，无泄漏）；best 参数写入 config/l1_best.json。"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dates = sorted(tr_rows["date"].unique())
    val_dates = set(dates[-60:])
    tr, va = tr_rows[~tr_rows["date"].isin(val_dates)], tr_rows[tr_rows["date"].isin(val_dates)]

    def obj(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 450, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            min_child_samples=trial.suggest_int("min_child_samples", 30, 120, step=10),
            subsample=trial.suggest_float("subsample", 0.7, 0.95),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 0.9),
        )
        m = lgb.LGBMRegressor(**{**PARAMS, **p})
        m.fit(tr[feats], tr["rank_1d"])
        d = va[["date", "code", "ret_1d"]].copy()
        d["score"] = m.predict(va[feats])
        from scipy.stats import spearmanr
        ics = d.groupby("date").apply(lambda g: spearmanr(g["score"], g["ret_1d"]).statistic, include_groups=False)
        return float(np.nanmean(ics))

    study = optuna.create_study(direction="maximize")
    for i in range(10):
        study.optimize(obj, n_trials=1)
        prog(12 + 18 * (i + 1) / 10, "L1 调参", f"trial {i+1}/10 best={study.best_value:.4f}")
    best = {**PARAMS, **study.best_params}
    (ROOT / "config" / "l1_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


def final_deep_family(model_fn, ds, dates, seeds, tag, p0, p1):
    """多种子终训 + 预测最新交易日，返回 (均值分数Series, 明细dict)。"""
    t_last = dates[-1]
    tr_dates = [d for d in dates[-755:-5] if ds.day(d) is not None][-300:]
    scores = []
    for si, seed in enumerate(seeds):
        torch.manual_seed(seed)
        model = model_fn().to(DEVICE)
        opt_ = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = torch.nn.MSELoss()
        model.train()
        for ep in range(10):
            rng = np.random.default_rng(100 * seed + ep)
            for d in rng.permutation(tr_dates):
                _, X, s, y = ds.day(d)
                Xt, st, yt = torch.from_numpy(X).to(DEVICE), torch.from_numpy(s).to(DEVICE), torch.from_numpy(y).to(DEVICE)
                opt_.zero_grad()
                lossf(model(Xt, st).squeeze(-1), yt).backward()
                opt_.step()
        model.eval()
        pack = ds.day(t_last, need_y=False)
        codes, X, s, _ = pack
        with torch.no_grad():
            sc = model(torch.from_numpy(X).to(DEVICE), torch.from_numpy(s).to(DEVICE)).squeeze(-1).cpu().numpy()
        scores.append(pd.Series(sc, index=codes))
        prog(p0 + (p1 - p0) * (si + 1) / len(seeds), "深度模型", f"{tag} seed{seed} 完成")
        del model
        torch.cuda.empty_cache()
    fam = sum(scores) / len(scores)
    return fam, {f"{tag}#{sd}": sv for sd, sv in zip(seeds, scores)}


def kronos_member(p, t_last):
    """Kronos-Base 零样本成员：逐股预测未来5日OHLC，取首日收益做截面分。失败返回 None。"""
    try:
        tok_dir = _hub_snapshot("NeoQuasar/Kronos-Tokenizer-Base")
        if tok_dir is None or KRONOS_MODEL is None:
            return None
        sys.path.insert(0, str(ROOT / "vendors" / "Kronos"))
        from model.kronos import Kronos, KronosTokenizer, KronosPredictor  # noqa
        # vendored Kronos 的 from_pretrained 不接受 device_map；由 KronosPredictor 统一搬到 GPU
        model = Kronos.from_pretrained(str(KRONOS_MODEL))
        tokenizer = KronosTokenizer.from_pretrained(str(tok_dir))
        pred = KronosPredictor(model, tokenizer, device="cuda:0")
        out = {}
        codes = p["code"].unique()
        for i, c in enumerate(codes):
            g = p[(p["code"] == c) & (p["date"] <= t_last)].sort_values("date").tail(512)
            g = g.dropna(subset=["open", "high", "low", "close"])
            if len(g) < 60:
                continue
            df = pd.DataFrame({k: g[k].astype(float).values for k in ("open", "high", "low", "close", "volume")})
            # KronosPredictor.calc_time_stamps 需要 .dt 访问器 → 必须传 Series，不能传 numpy 数组
            x_ts = pd.Series(pd.to_datetime(g["date"]).values)
            y_ts = pd.Series(pd.date_range(x_ts.iloc[-1] + pd.Timedelta(days=1), periods=5, freq="D"))
            pdf = pred.predict(df, x_ts, y_ts, pred_len=5, sample_count=1, verbose=False)
            out[c] = float(pdf["close"].iloc[0] / df["close"].iloc[-1] - 1)
            if (i + 1) % 20 == 0:
                prog(62 + 10 * (i + 1) / len(codes), "Kronos 零样本", f"{i+1}/{len(codes)}")
        del model, tokenizer, pred
        torch.cuda.empty_cache()
        return pd.Series(out)
    except Exception as e:  # noqa: BLE001
        prog(72, "Kronos 零样本", f"跳过（{type(e).__name__}: {str(e)[:60]}）")
        return None


def main():
    t0 = time.time()
    PROG.parent.mkdir(parents=True, exist_ok=True)
    PROG.write_text("", encoding="utf-8")
    prog(0.5, "启动", "完整管线开始")
    T = targets()
    step_data(prog)

    p = load_panel()
    dates = sorted(p["date"].unique())
    t_last = dates[-1]
    feats = FEATURE_COLS + SENT_COLS
    L = str(T["long_target"])
    # 短线圈只含电池成分：000927 与竞猜自选股（watchlist）都在面板里，必须剔除
    cons = set(pd.read_parquet(PARQUET_DIR / "cons_battery.parquet")["code"].astype(str))
    trad_all = p[(p["suspended"] == 0) & (p["isST"] == 0)]
    trad = trad_all[trad_all["code"].isin(cons)]
    tr_rows = trad[trad["date"] < t_last].dropna(subset=["rank_1d"])

    best_params = tune_l1(tr_rows, feats)
    prog(30, "L1 终训", "全历史拟合")
    l1 = lgb.LGBMRegressor(**best_params)
    l1.fit(tr_rows[feats], tr_rows["rank_1d"])
    imp = pd.Series(l1.feature_importances_, index=feats).sort_values(ascending=False).head(8)

    ds = DayDataset(p)
    fam_ns, det_ns = final_deep_family(NonStatNet, ds, dates, [7, 42, 2026], "NonStat", 31, 46)
    fam_tg, det_tg = final_deep_family(TgnsLite, ds, dates, [11, 23], "TGNS", 46, 55)
    fam_it, det_it = final_deep_family(ITransformerLite, ds, dates, [5, 9], "ITrans", 55, 62)

    prog(62, "Kronos 零样本", "加载模型…")
    k_score = kronos_member(p, t_last)

    prog(74, "路由集成", "计算状态桶权重…")
    day_all = trad_all[trad_all["date"] == t_last].copy().set_index("code")
    day_all["s_l1"] = l1.predict(day_all[feats])
    day_all["s_ns"] = fam_ns.reindex(day_all.index)
    day_all["s_tg"] = fam_tg.reindex(day_all.index)
    day_all["s_it"] = fam_it.reindex(day_all.index)
    if k_score is not None:
        day_all["s_kr"] = k_score.reindex(day_all.index)
    day_all["s_mr"] = 1 - day_all["ret_20"].rank(pct=True)
    day_all["s_rv"] = 1 - day_all["ret_5"].rank(pct=True)

    from src.models.regime_gate import state_buckets
    gw_path = ROOT / "reports" / "regmoe_weights.csv"
    alpha, e_tg, e_it, e_kr = 0.25, 0.05, 0.03, 0.07
    W = {}
    try:
        gw = pd.read_csv(gw_path)
        bucket = int(state_buckets(p).iloc[-1])
        recent = gw.merge(p[["date"]].drop_duplicates(), on="date")
        recent["bucket"] = recent["date"].map(state_buckets(p))
        cand = recent[recent["bucket"] == bucket].sort_values("date").tail(120)
        wb = cand.groupby("model")["weight"].mean()
        rest = 1 - alpha - e_tg - e_it - (e_kr if k_score is not None else 0)
        W = {"L1": float(wb.get("L1-LGBM", 0.5)) * rest, "MR20": float(wb.get("L0-MR20", 0.3)) * rest, "REV5": float(wb.get("L0-REV5", 0.2)) * rest,
             "NonStat": alpha, "TGNS": e_tg, "ITrans": e_it}
        if k_score is not None:
            W["Kronos"] = e_kr
        s = sum(W.values())
        W = {k: round(v / s, 4) for k, v in W.items()}
        bucket_desc = BUCKET_DESC.get(bucket, str(bucket))
    except Exception as e:  # noqa: BLE001
        bucket_desc = "未知"
        W = {"L1": 0.35, "MR20": 0.22, "REV5": 0.12, "NonStat": 0.25, "TGNS": 0.04, "ITrans": 0.02}
        if k_score is not None:
            W["Kronos"] = round(1 - sum(W.values()), 4)

    day_all["ens"] = (
        day_all["s_l1"].rank(pct=True) * W.get("L1", 0)
        + day_all["s_mr"] * W.get("MR20", 0) + day_all["s_rv"] * W.get("REV5", 0)
        + day_all["s_ns"].rank(pct=True).fillna(0.5) * W.get("NonStat", 0)
        + day_all["s_tg"].rank(pct=True).fillna(0.5) * W.get("TGNS", 0)
        + day_all["s_it"].rank(pct=True).fillna(0.5) * W.get("ITrans", 0)
        + (day_all["s_kr"].rank(pct=True).fillna(0.5) * W.get("Kronos", 0) if "s_kr" in day_all else 0)
    )
    aux_score = float(day_all.loc[L, "ens"]) if L in day_all.index else 0.5
    day = day_all[day_all.index.isin(cons)].reset_index()

    prog(82, "报告生成", "训练次日上涨概率模型…")
    n = int(T.get("page_size", 20))
    upm = UpModel(feats).fit(tr_rows)
    first_page = rank_first_page(day, upm, feats, n)
    pick = first_page.iloc[0]
    p_up = float(pick["p_up"])
    exp_ret = float(pick["exp_ret"])
    close_pick = float(pick["close"])
    confidence, action = confidence_of(p_up)
    direction_a = "看涨" if p_up >= 0.55 else ("中性" if p_up >= 0.45 else "看跌")
    levels = risk_levels(close_pick, exp_ret, float(pick.get("atr_norm", 0.02)))
    fp_avg_up = float(first_page["p_up"].mean())
    rank_all = day.sort_values("ens", ascending=False).reset_index(drop=True)
    pick_rank = int((rank_all["code"] == pick["code"]).argmax()) + 1
    fp2 = first_page
    gap = float(fp2["p_up"].iloc[0] - fp2["p_up"].iloc[1]) if len(fp2) > 1 else 0.0
    names = load_names()
    pick_name = names.get(str(pick["code"]), "")
    prog(85, "报告生成", "上涨概率样本外回测…")
    bt = oos_backtest(trad, feats, pool_n=n, n_days=60)
    bt_txt = (
        f"近{bt.get('days', 0)}日样本外：首选上涨命中 {bt.get('up_rate', 0):.1%}、平均次日收益 {bt.get('avg_ret', 0):+.2%}"
        + (f"；概率≥{TRADE_P:.0%}时出手 {bt.get('gate_days', 0)} 天，命中 {bt.get('gate_up_rate') or 0:.1%}" if bt.get("gate_days") else "")
        if bt else "样本外回测数据不足"
    )

    # ---- 投票与小结（标的A） ----
    vote_cols = ["s_l1", "s_ns", "s_tg", "s_it", "s_mr", "s_rv"] + (["s_kr"] if "s_kr" in pick.index else [])
    votes_up = sum(1 for c in vote_cols if pick[c] > 0.55)
    top_w = max(W, key=W.get)
    news_items = []
    npath = ROOT / "data/raw/news.parquet"
    if npath.exists():
        nw = pd.read_parquet(npath)
        nw_pick = nw[nw["code"] == pick["code"]].sort_values("ts").tail(3)
        news_items = [{"d": f"{pd.to_datetime(r['ts']):%m-%d}", "t": str(r["title"])[:42]} for _, r in nw_pick.iterrows()]
    reasons_a = [
        f"短线只做上涨：第一页 {n} 只按次日上涨概率排序，选中 {pick['code']}{(' ' + pick_name) if pick_name else ''}（概率 {p_up:.1%}，第一页均值 {fp_avg_up:.1%}，领先第二名 {gap:.1%}）",
        f"预期与风控：次日预期涨幅 {exp_ret:+.2%}，收盘 {close_pick:.2f}；止盈参考 {levels['target']:.2f}、止损参考 {levels['stop']:.2f}（ATR {levels['atr_pct']:.1f}%），结论「{confidence}·{action}」",
        f"样本外验证：{bt_txt}",
        f"市场状态：{bucket_desc} → 路由主要权重给 {top_w}（{W[top_w]:.0%}），该类市况下其历史滚动 IC 最佳",
        f"盘面证据：当日涨幅 {pick['pct_chg']:+.1f}%、5日 {pick['ret_5']:+.1%}、换手分位 {pick['turn_pctile']:.2f}" + (f"、近3日情绪 {pick['sent_mean']:+.2f}" if pd.notna(pick.get("sent_mean")) else "") + f"；{votes_up}/{len(vote_cols)} 路成员偏多",
        f"交易纪律：上涨概率未达 {TRADE_P:.0%} 不出手——只做有上涨把握的机会，不为交易而交易",
    ]
    if p_up < TRADE_P:
        reasons_a.append(f"风险提示：当前首选概率 {p_up:.1%} 低于出手门槛，建议空仓等待，而不是勉强买入。")

    # ---- 标的B ----
    g = p[p["code"] == L].sort_values("date")
    close, ma60, ma200 = float(g["close"].iloc[-1]), float(g["close"].rolling(60).mean().iloc[-1]), float(g["close"].rolling(200).mean().iloc[-1])
    hi252, lo252 = float(g["close"].rolling(252).max().iloc[-1]), float(g["close"].rolling(252).min().iloc[-1])
    state = trend_state(g)
    aux_label = "偏多" if aux_score > 0.6 else ("偏空" if aux_score < 0.4 else "中性")
    triggers = []
    if close < ma200 * 0.99:
        triggers.append("收盘跌破200日线（趋势警戒）")
    if close / lo252 > 1.5:
        triggers.append("距52周低点反弹超50%")
    advice_long = {"上升": "长线持有 · 趋势向上，维持", "震荡": "长线持有 · 趋势震荡，耐心持有"}.get(state, "长线持有 · 趋势向下，暂缓加仓")
    reasons_b = [
        f"长线定位：中国铁物按长期持有管理；本轨预测允许涨/跌/中性，明日方向不改变长期框架",
        f"趋势结构：收盘 {close:.2f} 相对 MA60({ma60:.2f}){'上方' if close > ma60 else '下方'}、MA200({ma200:.2f}){'上方' if close > ma200 else '下方'} → 判定【{state}】",
        f"价格位置：距52周高点 {close/hi252-1:+.1%}、距52周低点 {close/lo252-1:+.1%} → " + ("处于深度回调区，长线可跟踪分批信号" if close/hi252-1 < -0.2 else "位置中性与偏高之间"),
        f"均线含义：{'多头排列未破坏' if state == '上升' else ('空头排列，趋势向下，右侧信号未出现' if state == '下降' else '多空交织，趋势震荡')}；结论：{advice_long}",
        f"明日方向（辅助，低决策权重）：{aux_label}（集成分 {aux_score:.2f}）——可为涨/跌/中性，长线决策不应依赖单日涨跌",
    ] + (["触发提醒：" + "；".join(triggers)] if triggers else [])

    # ---- 台账 ----
    ledger = ROOT / "reports/predictions_ledger.csv"
    gen_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    new_rows = pd.DataFrame([
        {"gen_date": gen_date, "T": t_last, "pred_for": "next_trading_day", "track": "short", "code": pick["code"], "score": round(float(p_up), 4), "pool_rank": pick_rank, "direction": direction_a, "actual": "", "hit": ""},
        {"gen_date": gen_date, "T": t_last, "pred_for": "next_trading_day", "track": "long_aux", "code": L, "score": round(aux_score, 4), "pool_rank": "", "direction": aux_label, "actual": "", "hit": ""},
    ])
    review_text, hits, total = "首日运行，无历史可复盘", 0, 0
    if ledger.exists():
        old = pd.read_csv(ledger, dtype={"code": str})
        old, review_text, hits, total = fill_reviews(old, p, dates, t_last)
        note = stale_note(t_last)
        if note:
            print(f"[ledger] 跳过写入：{note}", flush=True)
        else:
            old, skipped = upsert(old, new_rows)
            if skipped:
                print(f"[ledger] 已有 {skipped} 条同 (T,track) 记录，按冻结规则不覆盖", flush=True)
        old.to_csv(ledger, index=False, encoding="utf-8-sig")

    gp = p[p["code"] == pick["code"]].sort_values("date").tail(30)
    short_series = [{"d": pd.Timestamp(d).strftime("%m-%d"), "c": round(float(c), 2)} for d, c in zip(gp["date"], gp["close"])]
    gl = g.tail(120)
    ma60_line = g["close"].rolling(60).mean().tail(120)
    long_series = [
        {"d": pd.Timestamp(d).strftime("%m-%d"), "c": round(float(c), 2), "ma60": None if pd.isna(m) else round(float(m), 2)}
        for d, c, m in zip(gl["date"], gl["close"], ma60_line)
    ]

    # ---- latest.json（GUI） ----
    latest = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "T": t_last, "runtime_s": int(time.time() - t0), "bucket": bucket_desc, "weights": W,
        "short": {
            "code": pick["code"], "name": pick_name, "rank": pick_rank, "pool": len(day),
            "page_size": n, "p_up_rank": 1, "p_up_avg": round(fp_avg_up, 4), "win_prob": round(p_up, 4),
            "exp_ret": round(exp_ret, 4), "confidence": confidence, "action": action,
            "direction": direction_a, "close": round(close_pick, 2), "risk": levels, "backtest": bt,
            "warning": "" if p_up >= TRADE_P else f"首选上涨概率 {p_up:.1%} 未达 {TRADE_P:.0%} 出手门槛：按纪律应空仓等待，不勉强交易。",
            "ens": round(float(pick["ens"]), 4), "votes_up": votes_up, "votes_total": len(vote_cols),
            "evidence": {"pct_chg": round(float(pick["pct_chg"]), 2), "ret_5": round(float(pick["ret_5"]), 4), "turn": round(float(pick["turn_pctile"]), 2),
                         "sent": None if pd.isna(pick.get("sent_mean")) else round(float(pick["sent_mean"]), 2)},
            "members": {
                k: round(float(pick[f"s_{k2}"]), 3)
                for k, k2 in [("L1", "l1"), ("NonStat", "ns"), ("TGNS", "tg"), ("ITrans", "it"), ("Kronos", "kr"), ("MR20", "mr"), ("REV5", "rv")]
                if f"s_{k2}" in pick.index and pd.notna(pick[f"s_{k2}"])
            },
            "reasons": reasons_a, "news": news_items, "series": short_series,
            "first_page": [
                {"code": r["code"], "name": names.get(str(r["code"]), ""), "pct": round(float(r["pct_chg"]), 2),
                 "p_up": round(float(r["p_up"]), 4), "exp_ret": round(float(r["exp_ret"]), 4), "ens": round(float(r["ens"]), 3)}
                for _, r in fp2.head(10).iterrows()
            ],
            "feat_imp": [{"f": k, "v": int(v)} for k, v in imp.items()],
        },
        "long": {
            "code": L, "name": "中国铁物", "close": round(close, 2), "ma60": round(ma60, 2), "ma200": round(ma200, 2),
            "hi252": round(close / hi252 - 1, 3), "lo252": round(close / lo252 - 1, 3),
            "state": state, "advice": advice_long,
            "pred_note": "长线持有：允许上涨/下跌/中性，明日方向仅作辅助，不用单日涨跌推翻长线框架",
            "aux": aux_label, "aux_score": round(aux_score, 3), "triggers": triggers, "series": long_series,
            "reasons": reasons_b,
        },
        "review": {"text": review_text, "hits": hits, "total": total, "label": "短线次日上涨"},
    }
    (ROOT / "reports" / "latest.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Forward evidence + monitor (append-only, never allowed to break the report)
    try:
        from src.data.forward_evidence import post_run

        post_run()
    except Exception as e:  # noqa: BLE001
        print(f"[run_full] forward hook failed: {type(e).__name__}: {e}", flush=True)

    # ---- md 报告 ----
    rep = ROOT / "reports" / f"{gen_date}.md"
    rep.write_text("\n".join([
        f"# 每日双轨预测（完整管线）｜生成 {gen_date}｜T={t_last}",
        "", "> 量化研究输出，不构成投资建议。",
        "", f"## 标的A 短线：{pick['code']}{(' ' + pick_name) if pick_name else ''}｜次日上涨概率 {p_up:.1%}｜{confidence}·{action}",
        "", f"- 预期次日涨幅 {exp_ret:+.2%}｜收盘 {close_pick:.2f}｜风控参考 关注价 {levels['entry']:.2f} / 止盈 {levels['target']:.2f} / 止损 {levels['stop']:.2f}（ATR {levels['atr_pct']:.1f}%）",
        f"- 短线纪律：第一页中只选上涨概率最高者；概率不足 {TRADE_P:.0%} 不出手（涨了才赚钱）",
        *[f"- {r}" for r in reasons_a], "", f"## 标的B 长线：中国铁物（{state}）· 长期持有", "",
        "- 长线预测允许涨/跌/中性，明日方向仅作辅助，不改变长期持有框架。",
        *[f"- {r}" for r in reasons_b],
        "", f"## 复盘：{review_text}", "",
        "## 研究状态（截至 2026-10-26）",
        "- 120 日审计：≥55% 门槛日实际上涨 49.4%（Wilson 38.8–60.1%），门槛未被证实；本报告仅为研究记录，不构成加仓依据。",
        "- NETRET-001 / RELIAB-001 / TARGET-001 均未通过，详见 reports/top1_audit/report.md 与各实验目录。",
        "", f"---\n*耗时 {time.time()-t0:.0f}s｜桶 {bucket_desc}｜权重 {W}*",
    ]), encoding="utf-8")

    with open(ROOT / "工作日志.md", "a", encoding="utf-8") as f:
        f.write(f"\n### RUN-FULL {gen_date} {pd.Timestamp.now():%H:%M} ｜ 完整管线 ｜ T={t_last} ｜ 耗时 {int(time.time()-t0)}s\n- 标的A(短线·上涨概率): {pick['code']} p_up={p_up:.1%} exp={exp_ret:+.2%} {confidence}/{action} ｜ 标的B(长线): {state}/{aux_label}\n- 回测: {bt_txt}\n- Kronos: {'启用' if k_score is not None else '跳过'} ｜ 权重: {W}\n- 复盘: {review_text}\n")
    append_run("gui_run.log", "run_full", {
        "T": t_last, "耗时": f"{int(time.time() - t0)}s", "状态桶": bucket_desc,
        "短线": f"{pick['code']} {pick_name} p_up={p_up:.1%} exp={exp_ret:+.2%} {confidence}/{action}",
        "长线": f"{state}/{aux_label}",
        "Kronos": "启用" if k_score is not None else "跳过",
        "权重": W, "报告": f"reports/{gen_date}.md", "数据时点": t_last,
    })
    try:
        from src.report.guess_sheet import main as _guess_main

        _guess_main()
    except Exception as e:  # noqa: BLE001
        print(f"[guess] 竞猜记录更新失败（不影响主报告）: {type(e).__name__}: {e}", flush=True)
    try:
        from src.report.page_shadow import main as _shadow_main

        _shadow_main()
    except Exception as e:  # noqa: BLE001
        print(f"[shadow] 页内影子更新失败（不影响主报告）: {type(e).__name__}: {e}", flush=True)
    try:
        from src.report.industry_radar import main as _radar_main

        _radar_main()
    except Exception as e:  # noqa: BLE001
        print(f"[radar] 行业雷达更新失败（不影响主报告）: {type(e).__name__}: {e}", flush=True)
    try:
        from src.report.multi_industry_shadow import main as _multi_main

        _multi_main()
    except Exception as e:  # noqa: BLE001
        print(f"[multi] 多行业影子更新失败（不影响主报告）: {type(e).__name__}: {e}", flush=True)
    prog(100, "完成", f"耗时 {int(time.time()-t0)}s")
    print(f"RUN_FULL_DONE {int(time.time()-t0)}s")


if __name__ == "__main__":
    main()
