"""P3 评估：FactorMoE式路由 + TGNS式/非平稳/iTransformer式深度模型 → 门槛检验 + P3 报告。

运行: python -m src.report.eval_p3 [--skip-deep]
产出: reports/eval_P3.md + eval_P3_summary.csv + corr 矩阵
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.walkforward import evaluate, make_windows  # noqa: E402
from src.features.build import FEATURE_COLS  # noqa: E402
from src.models import baselines  # noqa: E402
from src.models.deep_common import DayDataset, train_predict  # noqa: E402
from src.models.itransformer_lite import ITransformerLite  # noqa: E402
from src.models.l1_lgbm import fit_predict  # noqa: E402
from src.models.nonstat import NonStatNet  # noqa: E402
from src.models.regime_gate import MODEL_KEYS, daily_ic, route, state_buckets  # noqa: E402
from src.models.tgns_lite import TgnsLite  # noqa: E402
from src.utils.config import PARQUET_DIR, ROOT  # noqa: E402


def tradable(p):
    q = p[(p["suspended"] == 0) & (p["isST"] == 0) & p["ret_1d"].notna()]
    one_price = (q["open"] == q["high"]) & (q["high"] == q["low"])
    return q[~one_price]


def l1_scores(p, dates):
    rows = []
    for wi, w in enumerate(make_windows(dates, 750, 5, 20)):
        tr, te = p[p["date"].isin(w["train"])], p[p["date"].isin(w["test"])]
        pred, _ = fit_predict(tr, te)
        d = te[["date", "code"]].copy()
        d["score"] = pred
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-deep", action="store_true")
    ap.add_argument("--use-deep-cache", action="store_true", help="复用 reports/cache/ 的深度预测，跳过重训")
    args = ap.parse_args()
    t0 = time.time()
    p = tradable(pd.read_parquet(PARQUET_DIR / "panel.parquet"))
    dates = sorted(p["date"].unique())
    truth = p[["date", "code", "ret_1d"]]
    print(f"面板 {len(p):,} 行 | 交易日 {len(dates)}", flush=True)

    # --- L1（与 P2 同口径） ---
    print("[1/4] L1 walk-forward(20日步) …", flush=True)
    s_l1 = l1_scores(p, dates)
    s_l1["model"] = "L1-LGBM"

    # --- L0 族逐日分数 ---
    full = pd.read_parquet(PARQUET_DIR / "panel.parquet")
    l0 = {
        "L0-MR20": baselines.score_mr20(full),
        "L0-REV5": 1 - full.groupby("date")["ret_5"].rank(pct=True),  # 反转动量（P2 发现）
    }
    s_l0 = pd.concat([
        pd.DataFrame({"date": full["date"], "code": full["code"], "score": v, "model": k})
        for k, v in l0.items()
    ], ignore_index=True)

    # --- FactorMoE 式路由 ---
    print("[2/4] FactorMoE 式状态路由 …", flush=True)
    scores_all = {"L1-LGBM": s_l1[["date", "code", "score"]], "L0-MR20": s_l0[s_l0["model"] == "L0-MR20"][["date", "code", "score"]], "L0-REV5": s_l0[s_l0["model"] == "L0-REV5"][["date", "code", "score"]]}
    ic_daily = daily_ic(scores_all, truth)
    buckets = state_buckets(full)
    weights = route(ic_daily, buckets)
    # 组合门控分数（逐日按权重合并 rank 分）
    gated_rows = []
    for d, g in weights.groupby("date"):
        parts, ws = [], []
        for name, w in zip(g["model"], g["weight"]):
            df = scores_all[name]
            dd = df[df["date"] == d]
            if len(dd):
                parts.append(dd.set_index("code")["score"].rank(pct=True))
                ws.append(w)
        if not parts:
            continue
        wsum = sum(ws)
        comb = sum((pr * w / wsum for pr, w in zip(parts, ws)))
        gated_rows.append(pd.DataFrame({"date": d, "code": comb.index, "score": comb.values}))
    s_gate = pd.concat(gated_rows, ignore_index=True)
    s_gate["model"] = "REGMOE-gate"

    # --- 深度模型（预测结果落盘缓存，崩溃/复评不再重训） ---
    deep_preds = {}
    if not args.skip_deep:
        print("[3/4] 深度模型 walk-forward(60日步, GPU) …", flush=True)
        ds = DayDataset(full)
        cache_dir = ROOT / "reports" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name, fn in {"TGNS-lite": TgnsLite, "NonStat": NonStatNet, "ITrans-lite": ITransformerLite}.items():
            cf = cache_dir / f"deep_{name}.parquet"
            if args.use_deep_cache and cf.exists():
                df = pd.read_parquet(cf)
                print(f"  {name}: 载入缓存 {len(df):,} 条", flush=True)
            else:
                t1 = time.time()
                df = train_predict(fn, ds, dates, test_days=60, train_take=300, epochs=10, tag=name)
                df["model"] = name
                df.to_parquet(cf)
                print(f"  {name}: {len(df):,} 条预测(已缓存) | {time.time() - t0:.0f}s", flush=True)
            deep_preds[name] = df

    # 路由口径对齐：REGMOE-gate 限定在 L1 覆盖的同一批日期上重评（可比性）
    common_dates = set(s_l1["date"])
    s_gate_c = s_gate[s_gate["date"].isin(common_dates)].copy()
    s_gate_c["model"] = "REGMOE-gate*"

    # --- 汇总与门槛 ---
    print("[4/4] 汇总评估 …", flush=True)
    preds = {"L1-LGBM": s_l1, "REGMOE-gate": s_gate, **{k: v for k, v in deep_preds.items()}}
    l0d = pd.concat([
        pd.DataFrame({"date": full["date"], "code": full["code"], "score": l0["L0-MR20"], "model": "L0-MR20"}),
        pd.DataFrame({"date": full["date"], "code": full["code"], "score": l0["L0-REV5"], "model": "L0-REV5"}),
    ], ignore_index=True)
    allp = pd.concat([preds[m][["date", "code", "score", "model"]] for m in preds] + [l0d, s_gate_c[["date", "code", "score", "model"]]], ignore_index=True)
    summary = []
    for name, g in allp.groupby("model"):
        m = evaluate(g.merge(truth, on=["date", "code"]))
        m["model"] = name
        summary.append(m)
    sdf = pd.DataFrame(summary)[["model", "n_days", "rank_ic", "icir", "dir_acc", "top3_hit"]].sort_values("rank_ic", ascending=False)

    # 相关性矩阵（与 L1 的预测相关性，门槛<0.9）——剔除NaN（次新股早期特征缺失所致）
    base = s_l1.merge(truth, on=["date", "code"]).rename(columns={"score": "s_l1"})
    corrs = {}
    for name in list(preds.keys())[1:] + ["L0-MR20", "L0-REV5"]:
        df = preds.get(name)
        if df is None:
            df = l0d[l0d["model"] == name]
        mg = base.merge(df[["date", "code", "score"]], on=["date", "code"]).dropna(subset=["s_l1", "score"])
        corrs[name] = spearmanr(mg["s_l1"], mg["score"]).statistic

    # 非平稳诊断（过平滑比/滞后阶数）：以 NonStat vs L0-MR20 输出对比（若有）
    diag = ""
    if "NonStat" in deep_preds:
        d = deep_preds["NonStat"].merge(truth, on=["date", "code"])
        g = d.groupby("date").agg(ps=("score", lambda s: s.std()), pr=("ret_1d", lambda s: s.std()))
        smooth_ratio = float((g["ps"] / g["pr"]).median())
        lag = float(np.argmax([d.groupby("date").apply(lambda x, k=kk: x["score"].corr(x["ret_1d"].shift(-k)), include_groups=False).mean() for kk in range(0, 4)]))
        diag = f"- NonStat 过平滑比(score_std/ret_std 中位)={smooth_ratio:.2f}；最优提前阶数={lag}（0=无滞后）"

    rep = ROOT / "reports" / "eval_P3.md"
    lines = [
        "# P3 评估报告：前沿模型与状态路由",
        f"",
        f"- 运行: {pd.Timestamp.now():%Y-%m-%d %H:%M} | 耗时 {time.time() - t0:.0f}s | L1口径: 750+5/20；深度模型: 750+5/60（月度再训练口径）",
        "",
        "## 汇总（样本外，按 RankIC 排序）",
        "",
        sdf.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 入集成门槛检验（与 L1 预测相关性 <0.9 且单独不弱于 L0 最强基线）",
        "",
    ]
    l0_best = sdf[sdf["model"].str.startswith("L0")]["rank_ic"].max()
    for name, c in corrs.items():
        ic = sdf.set_index("model").loc[name, "rank_ic"] if name in sdf["model"].values else np.nan
        ok = (abs(c) < 0.9) and (ic >= l0_best)
        lines.append(f"- {name}: corr_with_L1={c:.3f}, RankIC={ic:.4f} → {'✅ 入集成' if ok else '❌ 未过门槛'}")
    lines += ["", diag, "", "## 说明", "",
              "- REGMOE-gate 为 FactorMoE 式状态路由（波动分位×趋势 四桶条件化滚动IC门控，权重每日可读出）",
              "- 深度模型再训练频率为月度（60日窗口），与日常推理分离（D-12）",
              "- 论文宣称增益（TGNS +12.3–21.9% 等）为待验证假设，以本表实测为准", "",
              "## 免责声明", "", "历史滚动验证，不构成投资建议。"]
    rep.write_text("\n".join(lines), encoding="utf-8")
    sdf.to_csv(ROOT / "reports" / "eval_P3_summary.csv", index=False)
    weights.to_csv(ROOT / "reports" / "regmoe_weights.csv", index=False)
    print(sdf.to_string(index=False))
    print("\n与L1相关性:", {k: round(v, 3) for k, v in corrs.items()})
    print(f"报告: {rep} | 总耗时 {time.time() - t0:.0f}s\nEVAL_P3_DONE")


if __name__ == "__main__":
    main()
