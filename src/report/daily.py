"""P5 核心：每日双轨报告生成器（docs/06 双轨方案落地）。

运行: python -m src.report.daily          # T=面板最新交易日，预测下一交易日
链路: 面板(+情绪) → L1终训 → NonStat终训(GPU~2min) → MR20/REV5 → 固化权重集成
     → 短线上涨概率选股(第一页Top1) → 长线轨(000927仪表盘，允许涨/跌/中性)
     → 台账追加+昨日复盘 → reports/报告 + latest.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.build import FEATURE_COLS  # noqa: E402
from src.models.deep_common import DEVICE, DayDataset  # noqa: E402
from src.models.l1_lgbm import PARAMS  # noqa: E402
from src.models.nonstat import NonStatNet  # noqa: E402
from src.report.ledger import fill_reviews, stale_note, upsert  # noqa: E402
from src.report.runlog import append_run  # noqa: E402
from src.report.short_pick import TRADE_P, UpModel, confidence_of, oos_backtest, rank_first_page, risk_levels  # noqa: E402
from src.utils.config import PARQUET_DIR, ROOT, targets  # noqa: E402

import lightgbm as lgb  # noqa: E402

SENT_COLS = ["sent_mean", "sent_max", "news_cnt", "kw_pos", "kw_neg"]
WEIGHTS_FALLBACK = {"L1": 0.45, "NonStat": 0.25, "MR20": 0.18, "REV5": 0.12}


def load_names() -> dict:
    f = ROOT / "data" / "raw" / "stock_names.parquet"
    if not f.exists():
        return {}
    try:
        n = pd.read_parquet(f)
        return {str(r["code"]): str(r["name"]) for _, r in n.iterrows()}
    except Exception:  # noqa: BLE001
        return {}


def load_panel():
    p = pd.read_parquet(PARQUET_DIR / "panel.parquet")
    spath = PARQUET_DIR / "sentiment.parquet"
    if spath.exists():
        sent = pd.read_parquet(spath)
        p = p.merge(sent, on=["code", "date"], how="left")
        p["sent_mean"] = p.groupby("code")["sent_mean"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    for c in SENT_COLS:
        if c not in p.columns:
            p[c] = np.nan
    return p


def nonstat_final(ds: DayDataset, dates, epochs=10, train_take=300):
    """NonStat 终训（最近750交易日截去末5日）+ 仅预测最新交易日。"""
    tr_dates = [d for d in dates[-755:-5] if ds.day(d) is not None][-train_take:]
    torch.manual_seed(0)
    model = NonStatNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.MSELoss()
    model.train()
    for ep in range(epochs):
        rng = np.random.default_rng(ep)
        for d in rng.permutation(tr_dates):
            _, X, s, y = ds.day(d)
            Xt, st, yt = torch.from_numpy(X).to(DEVICE), torch.from_numpy(s).to(DEVICE), torch.from_numpy(y).to(DEVICE)
            opt.zero_grad()
            lossf(model(Xt, st).squeeze(-1), yt).backward()
            opt.step()
    model.eval()
    pack = ds.day(dates[-1], need_y=False)
    if pack is None:
        return pd.Series(dtype=float), model
    codes, X, s, _ = pack
    with torch.no_grad():
        sc = model(torch.from_numpy(X).to(DEVICE), torch.from_numpy(s).to(DEVICE)).squeeze(-1).cpu().numpy()
    return pd.Series(sc, index=codes), model


def trend_state(g: pd.DataFrame) -> str:
    close, ma60, ma200 = g["close"].iloc[-1], g["close"].rolling(60).mean().iloc[-1], g["close"].rolling(200).mean().iloc[-1]
    if close > ma60 > ma200:
        return "上升"
    if close < ma60 < ma200:
        return "下降"
    return "震荡"


def main():
    t0 = time.time()
    T = targets()
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
    l1 = lgb.LGBMRegressor(**PARAMS)
    l1.fit(tr_rows[feats], tr_rows["rank_1d"])
    day_all = trad_all[trad_all["date"] == t_last].copy().set_index("code")

    print(f"[daily] NonStat 终训（GPU）… T={t_last}", flush=True)
    ds = DayDataset(p)
    ns_day, _ = nonstat_final(ds, dates)

    day_all["s_l1"] = l1.predict(day_all[feats])
    day_all["s_ns"] = ns_day.reindex(day_all.index)
    day_all["s_mr"] = 1 - day_all["ret_20"].rank(pct=True)
    day_all["s_rv"] = 1 - day_all["ret_5"].rank(pct=True)
    wf = ROOT / "reports" / "ens_fixed_weights.json"
    W = json.loads(wf.read_text(encoding="utf-8")) if wf.exists() else WEIGHTS_FALLBACK
    # 部署版路由权重（P5 结论：朴素IC加权0.0495不敌状态路由0.0757）：
    # 读取回测产出的 regmoe_weights.csv，取"今日状态桶"的近期均值作为 L1/MR20/REV5 权重，
    # NonStat 以固定份额 α 并入（未与路由联合验证，α 取保守 0.2）。
    gw_path = ROOT / "reports" / "regmoe_weights.csv"
    try:
        from src.models.regime_gate import state_buckets
        gw = pd.read_csv(gw_path)
        bucket_today = int(state_buckets(p).iloc[-1])
        recent = gw.merge(p[["date"]].drop_duplicates(), on="date")
        bmap = state_buckets(p)
        recent["bucket"] = recent["date"].map(bmap)
        cand = recent[(recent["bucket"] == bucket_today)].sort_values("date").tail(120)
        w_by_model = cand.groupby("model")["weight"].mean()
        alpha = 0.2
        W = {
            "L1": float(w_by_model.get("L1-LGBM", 0.5)) * (1 - alpha),
            "MR20": float(w_by_model.get("L0-MR20", 0.3)) * (1 - alpha),
            "REV5": float(w_by_model.get("L0-REV5", 0.2)) * (1 - alpha),
            "NonStat": alpha,
        }
        s = sum(W.values())
        W = {k: round(v / s, 4) for k, v in W.items()}
        print(f"[daily] 路由权重（bucket={bucket_today}）: {W}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[daily] 路由权重不可用，回退固化权重: {e}", flush=True)
    day_all["ens"] = (
        day_all["s_l1"].rank(pct=True) * W.get("L1", 0.45)
        + day_all["s_ns"].rank(pct=True).fillna(0.5) * W.get("NonStat", 0.25)
        + day_all["s_mr"] * W.get("MR20", 0.18)
        + day_all["s_rv"] * W.get("REV5", 0.12)
    )
    aux_score = float(day_all.loc[L, "ens"]) if L in day_all.index else 0.5
    day = day_all[day_all.index.isin(cons)].reset_index()

    n = int(T.get("page_size", 20))
    print("[daily] 训练次日上涨概率模型…", flush=True)
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
    print("[daily] 上涨概率样本外回测（60日）…", flush=True)
    bt = oos_backtest(trad, feats, pool_n=n, n_days=60)
    rank_all = day.sort_values("ens", ascending=False).reset_index(drop=True)
    pick_rank = int((rank_all["code"] == pick["code"]).argmax()) + 1
    names = load_names()
    pick_name = names.get(str(pick["code"]), "")

    imp = pd.Series(l1.feature_importances_, index=feats).sort_values(ascending=False).head(8)
    news_txt = "- 近期新闻：无"
    npath = ROOT / "data" / "raw" / "news.parquet"
    if npath.exists():
        nw = pd.read_parquet(npath)
        nw_pick = nw[nw["code"] == pick["code"]].sort_values("ts").tail(3)
        if len(nw_pick):
            news_txt = "- 近期新闻：\n" + "\n".join(f"  - {pd.to_datetime(r['ts']):%m-%d} {str(r['title'])[:40]}" for _, r in nw_pick.iterrows())

    g = p[p["code"] == L].sort_values("date")
    close, ma60, ma200 = g["close"].iloc[-1], g["close"].rolling(60).mean().iloc[-1], g["close"].rolling(200).mean().iloc[-1]
    hi252, lo252 = g["close"].rolling(252).max().iloc[-1], g["close"].rolling(252).min().iloc[-1]
    state = trend_state(g)
    aux_label = "偏多" if aux_score > 0.6 else ("偏空" if aux_score < 0.4 else "中性")
    triggers = []
    if close < ma200 * 0.99:
        triggers.append("收盘跌破200日线1%（趋势警戒）")
    if close / lo252 > 1.5:
        triggers.append("距52周低点反弹超50%（关注持续性）")
    if not triggers:
        triggers.append("无触发")

    # ---- 台账 + 昨日复盘 ----
    ledger = ROOT / "reports" / "predictions_ledger.csv"
    gen_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    new_rows = pd.DataFrame([{
        "gen_date": gen_date, "T": t_last, "pred_for": "next_trading_day", "track": "short",
        "code": pick["code"], "score": round(float(p_up), 4), "pool_rank": pick_rank,
        "direction": direction_a, "actual": "", "hit": "",
    }, {
        "gen_date": gen_date, "T": t_last, "pred_for": "next_trading_day", "track": "long_aux",
        "code": L, "score": round(aux_score, 4), "pool_rank": "",
        "direction": aux_label, "actual": "", "hit": "",
    }])
    review = "（首日运行，无历史预测可复盘）"
    review_hits = review_total = 0
    if ledger.exists():
        old = pd.read_csv(ledger, dtype={"code": str})
        old, review, review_hits, review_total = fill_reviews(old, p, dates, t_last)
        note = stale_note(t_last)
        if note:
            print(f"[ledger] 跳过写入：{note}", flush=True)
        else:
            old, skipped = upsert(old, new_rows)
            if skipped:
                print(f"[ledger] 已有 {skipped} 条同 (T,track) 记录，按冻结规则不覆盖", flush=True)
        old.to_csv(ledger, index=False, encoding="utf-8-sig")
    else:
        new_rows.to_csv(ledger, index=False, encoding="utf-8-sig")

    fp_tbl = first_page.head(10).copy()
    fp_tbl["p_up"] = (fp_tbl["p_up"] * 100).round(1)
    fp_tbl["exp_ret"] = (fp_tbl["exp_ret"] * 100).round(2)
    fp_tbl = fp_tbl[["code", "pct_chg", "p_up", "exp_ret", "ens"]].round({"pct_chg": 2, "ens": 3})
    bt_txt = (
        f"近{bt.get('days', 0)}日样本外：首选上涨命中 {bt.get('up_rate', 0):.1%}，平均次日收益 {bt.get('avg_ret', 0):+.2%}"
        if bt else "样本外回测数据不足"
    )
    sent_disp = f"{pick['sent_mean']:.2f}" if pd.notna(pick.get("sent_mean")) else "无数据"
    rep = ROOT / "reports" / f"{gen_date}.md"
    rep.write_text("\n".join([
        f"# 每日双轨预测报告（生成 {gen_date}｜基于 T={t_last} 收盘｜预测下一交易日）",
        "",
        "> 免责声明：量化研究输出，**不构成投资建议**；模型概率为历史统计估计，不代表未来收益。",
        "",
        "## 标的A · 短线轨（电池 881262 · 第一页上涨概率 Top1）",
        "",
        f"- **首选：{pick['code']}{(' ' + pick_name) if pick_name else ''}**（次日上涨概率 {p_up:.1%}｜{confidence}·{action}｜第一页概率第 1）",
        f"- 预期次日涨幅 {exp_ret:+.2%}｜收盘 {close_pick:.2f}｜风控参考 关注价 {levels['entry']:.2f} / 止盈 {levels['target']:.2f} / 止损 {levels['stop']:.2f}（ATR {levels['atr_pct']:.1f}%）",
        f"- 短线纪律：只选第一页中次日上涨概率最高的标的，涨了才能赚钱；概率未过半时不强行交易",
        f"- **{bt_txt}**",
        f"- 当日涨幅 {pick['pct_chg']:.2f}%｜5日 {pick['ret_5']:+.1%}｜换手分位 {pick['turn_pctile']:.2f}｜情绪3日 {sent_disp}",
        f"- 成员分：L1={pick['s_l1']:.2f}｜NonStat={pick['s_ns']:.2f}｜MR20={pick['s_mr']:.2f}｜REV5={pick['s_rv']:.2f}｜权重 {W}",
        "- L1 特征重要性 Top8: " + ", ".join(f"{k}({int(v)})" for k, v in imp.items()),
        news_txt,
        "",
        "### 第一页前10（按次日上涨概率）",
        "",
        fp_tbl.to_markdown(index=False),
        "",
        "## 标的B · 长线轨（中国铁物 000927 · 长期持有）",
        "",
        "> 长线预测允许上涨、下跌或中性；明日方向只是辅助信号，不改变长期持有框架。",
        "",
        f"- 收盘 {close:.2f}｜MA60 {ma60:.2f}（{'上' if close > ma60 else '下'}）｜MA200 {ma200:.2f}（{'上' if close > ma200 else '下'}）",
        f"- 距52周高/低：{close / hi252 - 1:+.1%} / {close / lo252 - 1:+.1%}",
        f"- **趋势状态：{state}**｜建议：{'长线持有，趋势向上' if state == '上升' else ('长线持有，耐心等待震荡结束' if state == '震荡' else '长线持有，暂缓加仓（趋势向下）')}",
        f"- 明日方向（可为涨/跌/中性，辅助信号·低决策权重）：**{aux_label}**（集成分 {aux_score:.2f}）",
        f"- 触发提醒：{'；'.join(triggers)}",
        "",
        "## 复盘",
        "",
        f"- {review}",
        "",
        "## 研究状态（截至 2026-10-26）",
        "- 120 日审计：≥55% 门槛日实际上涨 49.4%（Wilson 38.8–60.1%），门槛未被证实；本报告仅为研究记录，不构成加仓依据。",
        "- NETRET-001 / RELIAB-001 / TARGET-001 均未通过，详见 reports/top1_audit/report.md 与各实验目录。",
        "",
        f"---\n*耗时 {time.time() - t0:.0f}s｜配置 D-23｜权重 reports/ens_fixed_weights.json*",
    ]), encoding="utf-8")

    # ---- GUI 数据（短线按上涨概率选股；长线允许涨/跌/中性） ----
    gp = p[p["code"] == pick["code"]].sort_values("date").tail(30)
    short_series = [{"d": pd.Timestamp(d).strftime("%m-%d"), "c": round(float(c), 2)} for d, c in zip(gp["date"], gp["close"])]
    gl = g.tail(120)
    ma60_line = g["close"].rolling(60).mean().tail(120)
    long_series = [
        {"d": pd.Timestamp(d).strftime("%m-%d"), "c": round(float(c), 2), "ma60": None if pd.isna(m) else round(float(m), 2)}
        for d, c, m in zip(gl["date"], gl["close"], ma60_line)
    ]
    votes_up = sum(1 for c in ("s_l1", "s_ns", "s_mr", "s_rv") if pick[c] > 0.55)
    warning = "" if p_up >= TRADE_P else f"首选上涨概率 {p_up:.1%} 未达 {TRADE_P:.0%} 出手门槛：按纪律应空仓等待，不勉强交易。"
    reasons_a = [
        f"短线只做上涨：在第一页 {n} 只候选里按次日上涨概率排序，选中 {pick['code']}{(' ' + pick_name) if pick_name else ''}（概率 {p_up:.1%}，第一页均值 {fp_avg_up:.1%}，概率第 1）",
        f"双模型输出：LightGBM 分类器给出上涨概率，Huber 回归给出预期次日涨幅 {exp_ret:+.2%}；上涨概率是首要排序依据",
        f"交易计划：收盘 {close_pick:.2f} 附近关注，止盈参考 {levels['target']:.2f}，止损参考 {levels['stop']:.2f}（ATR {levels['atr_pct']:.1f}%）；当前结论「{confidence}·{action}」",
        f"样本外验证：{bt_txt}",
        f"盘面证据：当日涨幅 {pick['pct_chg']:+.1f}%，5日 {pick['ret_5']:+.1%}，换手分位 {pick['turn_pctile']:.2f}，近3日情绪 {sent_disp}",
        f"模型共振：L1 {pick['s_l1']:.2f}、NonStat {pick['s_ns']:.2f}、MR20 {pick['s_mr']:.2f}、REV5 {pick['s_rv']:.2f}；{votes_up}/4 路偏多（上涨概率独立于集成分排序）",
    ]
    if warning:
        reasons_a.append("风险提示：" + warning)
    advice_long = {"上升": "长线持有 · 趋势向上，维持", "震荡": "长线持有 · 趋势震荡，耐心持有"}.get(state, "长线持有 · 趋势向下，暂缓加仓")
    latest = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "T": t_last, "runtime_s": int(time.time() - t0), "bucket": "轻量例行 · 状态路由", "weights": W,
        "short": {
            "code": pick["code"], "name": pick_name, "rank": pick_rank, "pool": len(day),
            "page_size": n, "p_up_rank": 1, "p_up_avg": round(fp_avg_up, 4), "win_prob": round(p_up, 4),
            "exp_ret": round(exp_ret, 4), "confidence": confidence, "action": action,
            "direction": direction_a, "close": round(close_pick, 2), "risk": levels, "backtest": bt,
            "warning": warning, "ens": round(float(pick["ens"]), 4),
            "votes_up": votes_up, "votes_total": 4,
            "evidence": {"pct_chg": round(float(pick["pct_chg"]), 2), "ret_5": round(float(pick["ret_5"]), 4),
                         "turn": round(float(pick["turn_pctile"]), 2),
                         "sent": None if pd.isna(pick.get("sent_mean")) else round(float(pick["sent_mean"]), 2)},
            "members": {"L1": round(float(pick["s_l1"]), 3), "NonStat": round(float(pick["s_ns"]), 3),
                        "MR20": round(float(pick["s_mr"]), 3), "REV5": round(float(pick["s_rv"]), 3)},
            "reasons": reasons_a,
            "news": [], "series": short_series,
            "first_page": [
                {"code": r["code"], "name": names.get(str(r["code"]), ""), "pct": round(float(r["pct_chg"]), 2),
                 "p_up": round(float(r["p_up"]), 4), "exp_ret": round(float(r["exp_ret"]), 4), "ens": round(float(r["ens"]), 3)}
                for _, r in first_page.head(10).iterrows()
            ],
            "feat_imp": [{"f": k, "v": int(v)} for k, v in imp.items()],
        },
        "long": {
            "code": L, "name": "中国铁物", "close": round(close, 2), "ma60": round(ma60, 2), "ma200": round(ma200, 2),
            "hi252": round(close / hi252 - 1, 3), "lo252": round(close / lo252 - 1, 3),
            "state": state, "advice": advice_long,
            "pred_note": "长线持有：允许上涨/下跌/中性，明日方向仅作辅助，不用单日涨跌推翻长线框架",
            "aux": aux_label, "aux_score": round(aux_score, 3), "triggers": triggers, "series": long_series,
            "reasons": [
                f"长线定位：中国铁物按长期持有管理；本轨预测允许涨/跌/中性，明日方向不改变持有框架",
                f"趋势结构：收盘 {close:.2f} 相对 MA60({ma60:.2f}){'上方' if close > ma60 else '下方'}、MA200({ma200:.2f}){'上方' if close > ma200 else '下方'} → 【{state}】",
                f"价格位置：距52周高点 {close/hi252-1:+.1%}、距52周低点 {close/lo252-1:+.1%}；{'深度回调区，长线可跟踪分批信号' if close/hi252-1 < -0.2 else '位置中性与偏高之间'}",
                f"均线含义：{'多头排列未破坏' if state == '上升' else ('空头排列，趋势向下，等待右侧信号' if state == '下降' else '多空交织，趋势震荡')}",
                f"明日方向（辅助，低决策权重）：{aux_label}（集成分 {aux_score:.2f}）——可为涨/跌/中性",
            ] + (["触发提醒：" + "；".join(triggers)] if triggers else []),
        },
        "review": {"text": review, "hits": review_hits, "total": review_total, "label": "短线次日上涨"},
    }
    (ROOT / "reports" / "latest.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Forward evidence + monitor (append-only, never allowed to break the report)
    try:
        from src.data.forward_evidence import post_run

        post_run()
    except Exception as e:  # noqa: BLE001
        print(f"[daily] forward hook failed: {type(e).__name__}: {e}", flush=True)

    with open(ROOT / "工作日志.md", "a", encoding="utf-8") as f:
        f.write(f"\n### RUN {gen_date} {pd.Timestamp.now():%H:%M} ｜ 日常报告 ｜ T={t_last} ｜ 耗时 {time.time() - t0:.0f}s\n- 标的A(短线·上涨概率): {pick['code']} p_up={p_up:.1%} exp={exp_ret:+.2%} {confidence}/{action} ｜ 标的B(长线): {state}/{aux_label}\n- 回测: {bt_txt}\n- 复盘: {review}\n- 报告: reports/{gen_date}.md\n")
    append_run("last_run.log", "run_daily", {
        "T": t_last, "耗时": f"{time.time() - t0:.0f}s",
        "短线": f"{pick['code']} {pick_name} p_up={p_up:.1%} {confidence}/{action}",
        "长线": f"{state}/{aux_label}", "复盘": review,
        "报告": f"reports/{gen_date}.md",
        "数据提醒": stale_note(t_last) or "无",
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
    print(f"[daily] 报告: {rep} | Top1={pick['code']} p_up={p_up:.3f} en={confidence} | B:{state}/{aux_label} | {time.time() - t0:.0f}s\nDAILY_DONE")


if __name__ == "__main__":
    main()
