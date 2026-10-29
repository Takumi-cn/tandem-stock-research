"""P4: 情绪特征 A/B 评估（诚实口径：新闻历史仅覆盖近期，重叠窗口小样本，结论为"方向性参考"）。

A = L1（28特征，与 P2/P3 同构）  B = L1 + 5个情绪特征
单一时点切分（train: 重叠期前全部，embargo 5日；test: 重叠期），对比重叠窗口指标。
另附情绪原始预测力（IC）探针。
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.walkforward import evaluate  # noqa: E402
from src.features.build import FEATURE_COLS  # noqa: E402
from src.models.l1_lgbm import PARAMS, fit_predict  # noqa: E402
from src.utils.config import PARQUET_DIR, ROOT  # noqa: E402

import lightgbm as lgb  # noqa: E402

SENT_COLS = ["sent_mean", "sent_max", "news_cnt", "kw_pos", "kw_neg"]


def tradable(p):
    q = p[(p["suspended"] == 0) & (p["isST"] == 0) & p["ret_1d"].notna()]
    one_price = (q["open"] == q["high"]) & (q["high"] == q["low"])
    return q[~one_price]


def main():
    t0 = time.time()
    panel = pd.read_parquet(PARQUET_DIR / "panel.parquet")
    sent = pd.read_parquet(PARQUET_DIR / "sentiment.parquet")
    p = panel.merge(sent, on=["code", "date"], how="left")
    p["sent_mean"] = p.groupby("code")["sent_mean"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    dates = sorted(p["date"].unique())
    span = sorted(sent["date"].unique())
    overlap = [d for d in dates if d >= span[0]]
    cut = [d for d in dates if d < span[0]][-6]  # embargo 5日
    print(f"情绪覆盖 {span[0]}~{span[-1]}（{len(span)}个交易日）| A/B 测试窗 {len(overlap)} 日", flush=True)

    # 情绪原始预测力探针
    probe = p[p["date"].isin(overlap)].dropna(subset=["sent_mean"])
    if len(probe):
        ics = probe.groupby("date").apply(lambda g: spearmanr(g["sent_mean"], g["ret_1d"]).statistic, include_groups=False) if len(probe) else pd.Series()
        probe_ic = float(np.nanmean(ics))
        probe_days = int(ics.notna().sum())
    else:
        probe_ic, probe_days = float("nan"), 0

    pt = tradable(p)
    tr, te = pt[pt["date"] <= cut], pt[pt["date"].isin(overlap)]
    truth = te[["date", "code", "ret_1d"]]
    res = {}
    for tag, cols in [("A(无情绪)", FEATURE_COLS), ("B(加情绪)", FEATURE_COLS + SENT_COLS)]:
        feats = [c for c in cols if c in tr.columns]
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(tr[feats], tr["rank_1d"])
        d = truth.copy()
        d["score"] = m.predict(te[feats])
        res[tag] = evaluate(d)
        res[tag]["model"] = tag
        if tag == "B(加情绪)":
            imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    sdf = pd.DataFrame(res.values())[["model", "n_days", "rank_ic", "icir", "dir_acc", "top3_hit"]]

    rep = ROOT / "reports" / "eval_P4.md"
    lines = [
        "# P4 情绪特征 A/B 评估",
        "",
        f"- 运行: {pd.Timestamp.now():%Y-%m-%d %H:%M} | 情绪覆盖 {span[0]}~{span[-1]}（{len(span)} 交易日，news 历史浅）",
        f"- 单切分（train≤{cut}，embargo5日）× 测试=重叠期；**小样本方向性参考，统计效力有限，随每日运行积累后复评**",
        "",
        sdf.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"- 情绪原始预测力探针: sent_mean3 对 T+1 的日均 RankIC = {probe_ic:.4f}（{probe_days} 日）",
        f"- B 相对 A 的 RankIC 变化: {res['B(加情绪)']['rank_ic'] - res['A(无情绪)']['rank_ic']:+.4f}",
        "",
        "## 说明",
        "- 新闻来源为 akshare 个股新闻页（近期~100条/只），历史浅是接口固有约束；情绪特征的长周期验证依赖日常运行积累的台账",
        "- FinBERT+关键词双通道设计依据见工作日志 2026-10-26 组件探针（模型对事实型表述漏判→词典对冲）",
        "",
        "## 免责声明", "", "不构成投资建议。",
    ]
    rep.write_text("\n".join(lines), encoding="utf-8")
    sdf.to_csv(ROOT / "reports" / "eval_P4_summary.csv", index=False)
    print(sdf.to_string(index=False))
    print(f"情绪探针IC={probe_ic:.4f}({probe_days}日) | 报告: {rep} | {time.time() - t0:.0f}s\nEVAL_P4_DONE")


if __name__ == "__main__":
    main()
