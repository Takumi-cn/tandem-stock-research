"""P0 可交易性审计 + P1 Top1 选择后校准（docx 评审意见落地，v3 2026-10-26）。

一次重放（与实盘 short_pick 同口径：第一页20只、p_up 降序取 Top1、20日块滚动重训），
双产出：

P0 可交易收益审计（A股 T+1 制度）：
  - 账面 book = close_{T+1}/close_T - 1（生产标签；T日收盘价买入，实际不可实现）
  - 隔夜 gap = open_{T+1}/close_T - 1（信号产生后无法再买到的部分）
  - 盘中 oc  = close_{T+1}/open_{T+1} - 1（T+1 当日买入当日卖出，T+1 制度不允许）
  - 可实现 r_2d = close_{T+2}/open_{T+1} - 1（T+1 开盘买 → T+2 收盘卖，最短合规）
  - 可实现 r_5d = close_{T+6}/open_{T+1} - 1（一周持有）
  - 买入障碍：T+1 停牌 / 一字涨停开盘 → 不可买入；卖出障碍：卖出日跌停 → 顺延（≤3日）
  - 成本：14.5bps/边（基准）与 25bps/边（压力）

P1 Top1 选择后校准：
  - 全池十分位可靠性 vs Top1 分档校准
  - 门槛日（p_up≥0.55）Wilson / 二项检验 / 按日 bootstrap
  - 选择偏差：同概率档 Top1 实际胜率 vs 同档非 Top1 全池
  - 前后半段一致性（防止把单段行情当成优势）
  - Brier（全池 / Top1）

局限（诚实标注）：当前成分为最新快照、非 PIT；样本为近 120 个有标签交易日（6块×20日）；
不修改 latest.json、不改变实盘行为。

用法：
    python -m src.eval.top1_audit                 # 重放 + 报告
    python -m src.eval.top1_audit --report-only   # 复用 trades.csv/days.csv 重算报告
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.build import FEATURE_COLS  # noqa: E402
from src.report.daily import load_panel  # noqa: E402
from src.report.short_pick import TRADE_P, UpModel  # noqa: E402
from src.utils.config import PARQUET_DIR, ROOT, targets  # noqa: E402

OUT = ROOT / "reports" / "top1_audit"
N_DAYS, BLOCK, POOL_N = 120, 20, 20
COST_SIDE = 0.00145  # 14.5bps/边（与 v4/v4.1 研究口径一致）
COST_STRESS = 0.0025  # 25bps/边 压力
SENT_COLS = ["sent_mean", "sent_max", "news_cnt", "kw_pos", "kw_neg"]


def limit_of(code: str) -> float:
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("83", "87", "88", "43", "92")):
        return 0.30
    return 0.10


def is_limit_up_open(o, h, l, c, preclose, code) -> bool:
    """一字涨停开盘（无法买入）。"""
    if pd.isna(o) or pd.isna(preclose) or preclose <= 0:
        return False
    if not (abs(o - h) < 1e-6 and abs(o - l) < 1e-6 and abs(o - c) < 1e-6):
        return False
    return o / preclose - 1 >= limit_of(code) * 0.995


def is_limit_down_close(c, preclose, code) -> bool:
    if pd.isna(c) or pd.isna(preclose) or preclose <= 0:
        return False
    return c / preclose - 1 <= -limit_of(code) * 0.995


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (ctr - hw, ctr + hw)


def boot_by_date(df: pd.DataFrame, col: str, n_boot: int = 5000, seed: int = 42) -> dict:
    """按交易日重采样（同一天内的多只票不再独立），返回均值 95% CI 与单侧 p(均值<=0)。"""
    d = df.dropna(subset=[col])
    if len(d) == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    days = d["date"].unique()
    grouped = {day: g[col].to_numpy() for day, g in d.groupby("date")}
    means = []
    for _ in range(n_boot):
        pick = rng.choice(days, len(days), replace=True)
        vals = np.concatenate([grouped[x] for x in pick])
        means.append(float(vals.mean()))
    means = np.asarray(means)
    return {
        "n": int(len(d)), "n_days": int(len(days)),
        "mean": round(float(d[col].mean()), 5),
        "ci95": [round(float(np.percentile(means, 2.5)), 5), round(float(np.percentile(means, 97.5)), 5)],
        "p_le_0": round(float((means <= 0).mean()), 4),
    }


def replay(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """与实盘同口径重放，返回 (第一页逐票明细, 每日第一页汇总)。"""
    long_code = str(targets()["long_target"])
    cons = set(pd.read_parquet(PARQUET_DIR / "cons_battery.parquet")["code"].astype(str))
    trad = panel[(panel["suspended"] == 0) & (panel["isST"] == 0)
                 & (panel["code"] != long_code) & (panel["code"].isin(cons))].copy()
    labelled = trad.dropna(subset=["ret_1d", "rank_1d"])
    dates_lab = sorted(labelled["date"].unique())
    dates_all = sorted(panel["date"].unique())
    test_dates = dates_lab[-N_DAYS:]
    px = panel.set_index(["code", "date"])[
        ["open", "high", "low", "close", "preclose", "suspended", "pct_chg"]
    ]
    rows, day_rows = [], []
    for start in range(0, len(test_dates), BLOCK):
        block = test_dates[start:start + BLOCK]
        tr = labelled[labelled["date"] < block[0]]
        if len(tr) < 4000:
            continue
        feats = [c for c in FEATURE_COLS + SENT_COLS if c in tr.columns]
        model = UpModel(feats=feats).fit(tr)
        bset = set(block)
        for d, g in trad[trad["date"].isin(bset)].groupby("date"):
            g = g.dropna(subset=["ret_1d"])
            if g.empty:
                continue
            fp = g.sort_values("pct_chg", ascending=False).head(POOL_N).copy()
            sc = model.predict(fp[feats])
            fp["p_up"] = sc["p_up"].values
            fp["exp_ret"] = sc["exp_ret"].values
            fp = fp.sort_values(["p_up", "exp_ret"], ascending=False).reset_index(drop=True)
            seq = [x for x in dates_all if x > d]
            for rank_i, r in fp.iterrows():
                rank_i += 1
                code = str(r["code"])
                row = {
                    "date": d, "code": code, "rank_in_fp": rank_i, "is_top1": rank_i == 1,
                    "p_up": round(float(r["p_up"]), 4), "exp_ret": round(float(r["exp_ret"]), 5),
                    "pct_chg": round(float(r["pct_chg"]), 3), "close_T": round(float(r["close"]), 4),
                    "book": round(float(r["ret_1d"]), 5),
                    "gap": np.nan, "oc": np.nan, "r_2d": np.nan, "r_5d": np.nan,
                    "buy_block": "", "sell_delay": 0,
                }
                o1 = px.loc[(code, seq[0])] if len(seq) and (code, seq[0]) in px.index else None
                if o1 is None or bool(o1["suspended"]):
                    row["buy_block"] = "susp_t1"
                elif is_limit_up_open(o1["open"], o1["high"], o1["low"], o1["close"], o1["preclose"], code):
                    row["buy_block"] = "limit_up_open"
                else:
                    entry = float(o1["open"])
                    row["entry_open"] = round(entry, 4)
                    row["gap"] = round(entry / float(r["close"]) - 1, 5)
                    row["oc"] = round(float(o1["close"]) / entry - 1, 5)
                    for tag, k in (("r_2d", 1), ("r_5d", 5)):
                        sell_i, delay, val = k, 0, np.nan
                        while delay < 3 and len(seq) > sell_i:
                            cc = px.loc[(code, seq[sell_i])] if (code, seq[sell_i]) in px.index else None
                            if cc is None or bool(cc["suspended"]) or is_limit_down_close(cc["close"], cc["preclose"], code):
                                sell_i += 1
                                delay += 1
                                continue
                            val = float(cc["close"]) / entry - 1
                            row["sell_delay"] = max(row["sell_delay"], delay)
                            break
                        row[tag] = np.nan if pd.isna(val) else round(val, 5)
                rows.append(row)
            blk = [x for x in rows if x["date"] == d]
            valid = [x for x in blk if x["buy_block"] == ""]
            fp_r2d = [x["r_2d"] for x in valid if pd.notna(x.get("r_2d", np.nan))]
            t1 = next((x for x in blk if x["is_top1"]), {})
            day_rows.append({
                "date": d,
                "fp_book": round(float(np.mean([x["book"] for x in blk])), 5),
                "fp_gap": round(float(np.mean([x["gap"] for x in valid])), 5) if valid else np.nan,
                "fp_oc": round(float(np.mean([x["oc"] for x in valid])), 5) if valid else np.nan,
                "fp_r2d": round(float(np.mean(fp_r2d)), 5) if fp_r2d else np.nan,
                "top1_code": t1.get("code", ""),
                "top1_block": t1.get("buy_block", ""),
                "top1_r2d": t1.get("r_2d", np.nan),
                "top1_book": t1.get("book", np.nan),
                "top1_p": t1.get("p_up", np.nan),
            })
        print(f"  块 {start // BLOCK + 1}/{(len(test_dates) + BLOCK - 1) // BLOCK} 完成（累计 {len(rows)} 行）", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(day_rows)


def _stats(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": round(float(s.mean()), 5),
        "median": round(float(s.median()), 5),
        "log_mean": round(float(np.log1p(s).mean()), 6),
        "up_rate": round(float((s > 0).mean()), 4),
        "std": round(float(s.std()), 5),
    }


def analyze(t: pd.DataFrame, daysum: pd.DataFrame) -> dict:
    t = t.copy()
    t["buy_block"] = t["buy_block"].fillna("")
    top1 = t[t["is_top1"]].copy()
    top1["y_book"] = (top1["book"] > 0).astype(int)
    ok = top1[top1["buy_block"] == ""].copy()
    blocked = top1[top1["buy_block"] != ""]
    ud = sorted(top1["date"].unique())
    half = set(ud[: len(ud) // 2])

    # ---------- P0 ----------
    p0 = {
        "top1_days": int(len(top1)),
        "blocked_days": int(len(blocked)),
        "blocked_rate": round(len(blocked) / len(top1), 4) if len(top1) else None,
        "blocked_breakdown": blocked["buy_block"].value_counts().to_dict(),
        "book_all": _stats(top1["book"]),
        "book_tradeable": _stats(ok["book"]),
        "gap": _stats(ok["gap"]),
        "oc": _stats(ok["oc"]),
        "r2d": _stats(ok["r_2d"]),
        "r5d": _stats(ok["r_5d"]),
        "first_page": {
            "book": _stats(daysum["fp_book"]),
            "gap": _stats(daysum["fp_gap"]),
            "oc": _stats(daysum["fp_oc"]),
            "r2d": _stats(daysum["fp_r2d"]),
        },
    }
    if len(ok):
        p0["r2d"]["net_145"] = round(float(ok["r_2d"].mean()) - 2 * COST_SIDE, 5)
        p0["r2d"]["net_250"] = round(float(ok["r_2d"].mean()) - 2 * COST_STRESS, 5)
        p0["r5d"]["net_145"] = round(float(ok["r_5d"].mean()) - 2 * COST_SIDE, 5)
        p0["book_win_but_2d_lose"] = round(float(((ok["book"] > 0) & (ok["r_2d"] < 0)).mean()), 4)
        p0["book2d_corr"] = round(float(ok[["book", "r_2d"]].corr().iloc[0, 1]), 4)
        p0["r2d_net_boot"] = boot_by_date(ok.assign(net=ok["r_2d"] - 2 * COST_SIDE), "net")
        # 选择价值：同一天 Top1 与第一页等权（可交易口径）之差
        sel = daysum.dropna(subset=["top1_r2d", "fp_r2d"]).copy()
        sel["diff"] = sel["top1_r2d"] - sel["fp_r2d"]
        p0["selection_value"] = {"mean_diff": round(float(sel["diff"].mean()), 5),
                                 "boot": boot_by_date(sel, "diff")}
        # 隔夜/盘中按对数分解（可加性）
        lb, lg, lo = (float(np.log1p(ok[c]).mean()) for c in ("book", "gap", "oc"))
        p0["log_decomp"] = {"book": round(lb, 6), "gap": round(lg, 6), "oc": round(lo, 6),
                            "gap_share": round(lg / lb, 3) if lb else None,
                            "oc_share": round(lo / lb, 3) if lb else None}

    def half_block(sub: pd.DataFrame) -> dict:
        s = sub.copy()
        s["buy_block"] = s["buy_block"].fillna("")
        s_ok = s[s["buy_block"] == ""]
        g = s[s["p_up"] >= TRADE_P]
        g_ok = g[g["buy_block"] == ""]
        return {
            "days": int(len(s)),
            "book_up": round(float((s["book"] > 0).mean()), 4),
            "book_mean": round(float(s["book"].mean()), 5),
            "book_median": round(float(s["book"].median()), 5),
            "r2d_mean": round(float(s_ok["r_2d"].mean()), 5) if len(s_ok) else None,
            "r2d_net": round(float(s_ok["r_2d"].mean()) - 2 * COST_SIDE, 5) if len(s_ok) else None,
            "r2d_up": round(float((s_ok["r_2d"] > 0).mean()), 4) if len(s_ok) else None,
            "gate_days": int(len(g)),
            "gate_up": round(float((g["book"] > 0).mean()), 4) if len(g) else None,
            "gate_r2d_net": round(float(g_ok["r_2d"].mean()) - 2 * COST_SIDE, 5) if len(g_ok) else None,
            "gate_p_mean": round(float(g["p_up"].mean()), 4) if len(g) else None,
        }

    p0["halves"] = {"first": half_block(top1[top1["date"].isin(half)]),
                    "second": half_block(top1[~top1["date"].isin(half)])}

    # ---------- P1 ----------
    cal_pool, cal_top, bias = [], [], []
    tb = t.dropna(subset=["p_up"]).copy()
    tb["bucket"] = pd.qcut(tb["p_up"], 10, duplicates="drop")
    for b, g in tb.groupby("bucket", observed=True):
        cal_pool.append({"bucket": f"{b.left:.3f}~{b.right:.3f}", "n": int(len(g)),
                         "p_mean": round(float(g["p_up"].mean()), 4),
                         "actual_book": round(float((g["book"] > 0).mean()), 4),
                         "actual_2d": round(float((g["r_2d"] > 0).mean()), 4) if g["r_2d"].notna().any() else None})
    bins = [(0.00, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)]
    for lo, hi in bins:
        g = top1[(top1["p_up"] >= lo) & (top1["p_up"] < hi)]
        if len(g) == 0:
            continue
        k = int(g["y_book"].sum())
        lo95, hi95 = wilson(k, len(g))
        gok = g[g["buy_block"] == ""]
        cal_top.append({
            "range": f"[{lo:.2f},{hi:.2f})", "n": int(len(g)), "p_mean": round(float(g["p_up"].mean()), 4),
            "actual_book": round(k / len(g), 4), "wilson_lo": round(lo95, 4), "wilson_hi": round(hi95, 4),
            "n_tradeable": int(len(gok)),
            "actual_2d": round(float((gok["r_2d"] > 0).mean()), 4) if len(gok) else None,
            "r2d_net": round(float(gok["r_2d"].mean()) - 2 * COST_SIDE, 5) if len(gok) else None,
        })
        gp = t[(t["p_up"] >= lo) & (t["p_up"] < hi) & (~t["is_top1"]) & (t["buy_block"] == "")]
        if len(gok) >= 3 and len(gp) >= 20:
            bias.append({"range": f"[{lo:.2f},{hi:.2f})",
                         "top1_actual_book": round(float((gok["book"] > 0).mean()), 4),
                         "pool_actual_book": round(float((gp["book"] > 0).mean()), 4),
                         "top1_r2d_net": round(float(gok["r_2d"].mean()) - 2 * COST_SIDE, 5),
                         "pool_r2d_net": round(float(gp["r_2d"].mean()) - 2 * COST_SIDE, 5),
                         "n_top1": int(len(gok)), "n_pool": int(len(gp))})
    gate = top1[top1["p_up"] >= TRADE_P]
    gate_ok = gate[gate["buy_block"] == ""]
    n_gate, k_gate = int(len(gate)), int(gate["y_book"].sum())
    lo95, hi95 = wilson(k_gate, n_gate)
    binom_p50 = float(stats.binomtest(k_gate, n_gate, 0.5, alternative="greater").pvalue) if n_gate else float("nan")
    binom_p55 = float(stats.binomtest(k_gate, n_gate, 0.55, alternative="greater").pvalue) if n_gate else float("nan")
    brier_pool = float(np.mean((tb["p_up"] - (tb["book"] > 0)) ** 2))
    brier_top1 = float(np.mean((top1["p_up"] - top1["y_book"]) ** 2))
    base_pool = float((tb["book"] > 0).mean())
    base_top1 = float(top1["y_book"].mean())
    p1 = {
        "n_top1": int(len(top1)), "n_gate": n_gate,
        "gate_p_mean": round(float(gate["p_up"].mean()), 4) if n_gate else None,
        "gate_actual_book": round(k_gate / n_gate, 4) if n_gate else None,
        "gate_wilson95": [round(lo95, 4), round(hi95, 4)],
        "gate_binom_p_50": round(binom_p50, 4) if n_gate else None,
        "gate_binom_p_55": round(binom_p55, 4) if n_gate else None,
        "gate_r2d_net": round(float(gate_ok["r_2d"].mean()) - 2 * COST_SIDE, 5) if len(gate_ok) else None,
        "gate_r2d_n": int(len(gate_ok)),
        "overconfidence_pp": round((float(gate["p_up"].mean()) - k_gate / n_gate) * 100, 1) if n_gate else None,
        "brier_pool": round(brier_pool, 4), "brier_pool_base": round(base_pool, 4),
        "brier_pool_naive": round(base_pool * (1 - base_pool), 4),
        "brier_top1": round(brier_top1, 4), "brier_top1_base": round(base_top1, 4),
        "brier_top1_naive": round(base_top1 * (1 - base_top1), 4),
        "cal_pool": cal_pool, "cal_top1": cal_top, "selection_bias": bias,
    }
    if len(gate_ok):
        p1["gate_r2d_boot"] = boot_by_date(gate_ok.assign(net=gate_ok["r_2d"] - 2 * COST_SIDE), "net")
    return {"p0": p0, "p1": p1}


def render(res: dict) -> str:
    p0, p1 = res["p0"], res["p1"]
    b, g, o, r2, r5 = p0["book_tradeable"], p0["gap"], p0["oc"], p0["r2d"], p0["r5d"]
    fp = p0["first_page"]
    h1, h2 = p0["halves"]["first"], p0["halves"]["second"]

    def ci_line(boot, label):
        if not boot or not boot.get("ci95"):
            return f"- {label}：样本不足"
        return (f"- {label}：均值 {boot['mean']:+.3%}，95%CI [{boot['ci95'][0]:+.3%},{boot['ci95'][1]:+.3%}]，"
                f"p(≤0)={boot['p_le_0']:.3f}（按交易日 bootstrap，{boot.get('n_days', 0)} 天）")

    verdict = []
    if p1["n_gate"] >= 20:
        if p1["gate_wilson95"][1] < 0.55:
            verdict.append(f"门槛日实际上涨率 {p1['gate_actual_book']:.1%}，Wilson95 "
                           f"[{p1['gate_wilson95'][0]:.1%},{p1['gate_wilson95'][1]:.1%}] → **上界<55%，55% 门槛不成立（过自信）**")
        elif p1["gate_wilson95"][0] >= 0.55:
            verdict.append(f"门槛日实际上涨率 {p1['gate_actual_book']:.1%}，Wilson95 "
                           f"[{p1['gate_wilson95'][0]:.1%},{p1['gate_wilson95'][1]:.1%}] → **下界≥55%，门槛可信**")
        else:
            verdict.append(f"门槛日实际上涨率 {p1['gate_actual_book']:.1%}（点估计低于 55%），Wilson95 "
                           f"[{p1['gate_wilson95'][0]:.1%},{p1['gate_wilson95'][1]:.1%}] 覆盖 55% → **55% 门槛未被证实，也未证伪**")
        verdict.append(f"- 二项检验：H0(≤50%) p={p1['gate_binom_p_50']:.3f}"
                       f"（{'拒绝，优于硬币' if p1['gate_binom_p_50'] < 0.05 else '不能拒绝，连优于硬币都未证明'}）；"
                       f"H0(≤55%) p={p1['gate_binom_p_55']:.3f}")
        verdict.append(f"- 选择后过自信幅度：模型给 {p1['gate_p_mean']:.1%}，实际 {p1['gate_actual_book']:.1%}，"
                       f"高估 {p1['overconfidence_pp']:.1f} 个百分点")
        if p1.get("gate_r2d_boot"):
            gb = p1["gate_r2d_boot"]
            verdict.append(f"- 门槛日可实现净收益：均值 {gb['mean']:+.3%}，95%CI "
                           f"[{gb['ci95'][0]:+.3%},{gb['ci95'][1]:+.3%}]，p(≤0)={gb['p_le_0']:.3f}（{gb['n_days']} 天）")

    sel = pd.DataFrame(p1["selection_bias"])
    lines = [
        "# P0 可交易性审计 + P1 Top1 选择后校准",
        "",
        f"- 运行：{pd.Timestamp.now():%Y-%m-%d %H:%M}｜重放 {N_DAYS} 个有标签交易日（6块×20日滚动重训）",
        f"- 口径：第一页 {POOL_N} 只（**已剔除长线标的 000927**），p_up 降序 Top1；成本 {COST_SIDE*1e4:.1f}bps/边，压力 {COST_STRESS*1e4:.0f}bps/边",
        "- 局限：当前成分为最新快照非 PIT；不修改 latest.json、不改变实盘行为",
        "",
        "## P0：账面 vs 可实现（A股 T+1 制度）",
        "",
        f"- Top1 {p0['top1_days']} 日，其中 **{p0['blocked_days']} 日买入受阻**（{p0['blocked_breakdown'] or '无'}）",
        "",
        "| 口径 | n | 均值 | 中位数 | 胜率 | 说明 |",
        "|---|---:|---:|---:|---:|---|",
        f"| 账面 book（close_T→close_T+1） | {p0['book_all']['n']} | {p0['book_all']['mean']:+.3%} | {p0['book_all']['median']:+.3%} | {p0['book_all']['up_rate']:.1%} | 生产标签，**T 日收盘买入不可实现** |",
        f"| 隔夜 gap（close_T→open_T+1） | {g['n']} | {g['mean']:+.3%} | {g['median']:+.3%} | {g['up_rate']:.1%} | 信号产生后无法买到 |",
        f"| 盘中 oc（open→close，T+1） | {o['n']} | {o['mean']:+.3%} | {o['median']:+.3%} | {o['up_rate']:.1%} | 当日买当日卖，T+1 不允许 |",
        f"| **可实现 r_2d（T+1 开盘买→T+2 收盘卖）** | {r2['n']} | **{r2['mean']:+.3%}** | {r2['median']:+.3%} | {r2['up_rate']:.1%} | 最短合规持有 |",
        f"| 可实现 r_5d（T+6 收盘卖） | {r5['n']} | {r5['mean']:+.3%} | {r5['median']:+.3%} | {r5['up_rate']:.1%} | 一周持有 |",
        "",
        f"- 成本后：2 日 **{r2['net_145']:+.3%}**（14.5bps/边）→ 25bps 压力 **{r2['net_250']:+.3%}**；一周 {r5['net_145']:+.3%}",
        f"- 对数分解（可加）：账面几何均值 {p0['log_decomp']['book']:+.4%} = 隔夜 {p0['log_decomp']['gap']:+.4%} + 盘中 {p0['log_decomp']['oc']:+.4%}"
        f"（隔夜为正但信号产生后买不到；盘中为负且当日卖不出）",
        f"- 账面赢但可交易亏的比例 {p0['book_win_but_2d_lose']:.0%}；两口径相关系数 {p0['book2d_corr']:.3f}",
        f"- 选择价值（同日 Top1 − 第一页等权，可实现口径）：{p0['selection_value']['mean_diff']:+.3%}，"
        f"95%CI [{p0['selection_value']['boot']['ci95'][0]:+.3%},{p0['selection_value']['boot']['ci95'][1]:+.3%}]",
        ci_line(p0["r2d_net_boot"], "可实现净收益（Top1，扣 14.5bps/边）"),
        "",
        "### 前后半段一致性",
        "",
        "| 段 | 天数 | 账面胜率 | 账面中位数 | 可实现净收益 | 门槛日 | 门槛日胜率 | 门槛日净收益 | 门槛日均概率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| 前半段 | {h1['days']} | {h1['book_up']:.1%} | {h1['book_median']:+.3%} | {h1['r2d_net']:+.3%} | {h1['gate_days']} | {h1['gate_up']:.1%} | {h1['gate_r2d_net']:+.3%} | {h1['gate_p_mean']:.1%} |",
        f"| 后半段 | {h2['days']} | {h2['book_up']:.1%} | {h2['book_median']:+.3%} | {h2['r2d_net']:+.3%} | {h2['gate_days']} | {h2['gate_up']:.1%} | {h2['gate_r2d_net']:+.3%} | {h2['gate_p_mean']:.1%} |",
        "",
        "## P1：Top1 选择后校准（55% 门槛是否可信）",
        "",
    ] + [f"- {v.lstrip('- ')}" for v in verdict] + [
        f"- Brier：全池 {p1['brier_pool']} vs 常数基准 {p1['brier_pool_naive']}（基准率 {p1['brier_pool_base']:.1%}）｜"
        f"Top1 {p1['brier_top1']} vs 常数基准 {p1['brier_top1_naive']}（基准率 {p1['brier_top1_base']:.1%}）"
        f"→ {'Top1 校准劣于「永远猜基准率」，模型在选中的样本上过自信' if p1['brier_top1'] > p1['brier_top1_naive'] else 'Top1 校准不明显劣于基准率'}",
        "",
        "### Top1 分档校准", "",
        pd.DataFrame(p1["cal_top1"]).to_markdown(index=False) if p1["cal_top1"] else "（无样本）",
        "", "### 全池十分位可靠性（对照）", "",
        pd.DataFrame(p1["cal_pool"]).to_markdown(index=False) if p1["cal_pool"] else "（无样本）",
        "", "### 选择偏差（同概率档：Top1 vs 非 Top1）", "",
        sel.to_markdown(index=False) if len(sel) else "（样本不足）",
        "", "## 结论与治理", "",
        f"- **P0**：生产标签（close→close）均值 {p0['book_all']['mean']:+.3%}，但中位数 {p0['book_all']['median']:+.3%}、"
        f"胜率 {p0['book_all']['up_rate']:.1%}、几何均值 {p0['book_tradeable']['log_mean']:+.3%} → **不是稳定方向优势，正均值由少数尾部大涨幅贡献**。"
        f"可实现 2 日扣成本 {r2['net_145']:+.3%}，CI 覆盖 0；25bps 压力为负；一周持有 {r5['net_145']:+.3%}。"
        f"隔夜段（买不到）与盘中段（当日卖不出）都无法单独变现。",
        f"- **P1**：门槛日（p≥55%）{p1['n_gate']} 天、实际上涨 {p1['gate_actual_book']:.1%}，"
        f"Wilson95 [{p1['gate_wilson95'][0]:.1%},{p1['gate_wilson95'][1]:.1%}]；"
        f"**55% 门槛在 120 日样本上没有获得支持**；生产日志里的 61.8%/69.7% 属于后半段窗口"
        f"（前半段门槛日胜率仅 {h1['gate_up']:.1%}），不能当作稳定 alpha。",
        "- **治理**：不下调 55%、不放大仓位、不写 latest.json；生产维持「不出手」纪律。"
        "继续研究只能以**净收益/下行**为目标重做预注册候选（P2），不得用本次结果宣称成功。",
        "- 试验登记：TOP1-AUDIT-002（P0/P1），状态 AUDIT-COMPLETE。",
        "", "## 同日后续实验与生产状态（截至 2026-10-26，均为研究结论）", "",
        "| 项目 | 结论 | 关键数字 |",
        "|---|---|---|",
        "| NETRET-001 成本感知净期望 | FAIL | 日均机会 +0.127% vs 生产 +0.250%；每笔 p=0.375；25bps -0.006% |",
        "| RELIAB-001 可靠性门控 | FAIL | 仅 15 天交易；日均 +0.195% vs +0.250%；误杀 64 笔正收益 |",
        "| TARGET-001 目标对比 | 无增量 | C−A 日IC -0.0247（p=0.39）；现用 rank_1d 最优 |",
        "| 生产修复 | 已生效 | 第一页剔除 000927；台账按 (T,track) 冻结；每日任务 18:45+增量；Kronos 0.07 生效 |",
        "| 数据治理 | 已生效 | data/manifest.json 记录指纹/时点；非 PIT 声明 |",
        "",
        "- 相关报告：[NETRET-001](../net_return_v31/report.md)、[RELIAB-001](../reliability_v31/report.md)、[TARGET-001](../target_v31/report.md)。",
        "- 以上均未改变生产规则：55% 门槛、空仓纪律与 latest.json 输出保持不变。",
        "", "## 免责声明", "", "研究审计输出，不构成投资建议。",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--report-only" in argv:
        t = pd.read_csv(OUT / "trades.csv", dtype={"code": str})
        daysum = pd.read_csv(OUT / "days.csv")
        t["date"] = pd.to_datetime(t["date"])
        daysum["date"] = pd.to_datetime(daysum["date"])
        print(f"复用重放结果：{len(t)} 行 / {daysum['date'].nunique()} 天", flush=True)
    else:
        panel = load_panel()
        print(f"面板 {len(panel):,} 行 × {panel['code'].nunique()} 只 | 最新 {panel['date'].max()}", flush=True)
        print(f"重放 {N_DAYS} 日 OOS（6块×20日，第一页 Top1，T+1 可交易口径）…", flush=True)
        t, daysum = replay(panel)
        t.to_csv(OUT / "trades.csv", index=False, encoding="utf-8-sig")
        daysum.to_csv(OUT / "days.csv", index=False, encoding="utf-8-sig")
    res = analyze(t, daysum)
    (OUT / "report.md").write_text(render(res), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(res["p1"]["cal_pool"]).to_csv(OUT / "calib_pool.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(res["p1"]["cal_top1"]).to_csv(OUT / "calib_top1.csv", index=False, encoding="utf-8-sig")
    if "--report-only" not in argv:
        reg = ROOT / "reports" / "experiment_registry.csv"
        with open(reg, "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("exp_id,date,scope,status,conclusion\n")
            f.write(f"TOP1-AUDIT-002,{pd.Timestamp.now():%Y-%m-%d},电池个股短线Top1,AUDIT-COMPLETE,"
                    f"T+1可实现口径+选择后校准；gate_up={res['p1']['gate_actual_book']}；"
                    f"gate_net={res['p1']['gate_r2d_net']}\n")
    print(json.dumps({"p0": {k: res["p0"][k] for k in ("book_all", "book_tradeable", "gap", "oc", "r2d", "r5d", "log_decomp", "halves")},
                      "p1": {k: res["p1"][k] for k in ("n_gate", "gate_actual_book", "gate_wilson95", "gate_binom_p_50", "gate_r2d_net", "brier_top1")}},
                     ensure_ascii=False, indent=1, default=float))
    print("AUDIT_DONE")


if __name__ == "__main__":
    main(sys.argv[1:])
