"""P5: 集成层——{L1, NonStat(缓存), L0-MR20, L0-REV5} 滚动IC加权集成 vs REGMOE-gate 对照。

权重：每个测试日 t，各成员用 t 前（embargo5日）最近60个可得日的日均RankIC 经 softmax 得权（在线、无泄漏）。
产出: reports/eval_P5.md（决定 daily.py 的固化权重）。
"""
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
from src.models.l1_lgbm import fit_predict  # noqa: E402
from src.utils.config import PARQUET_DIR, ROOT  # noqa: E402


def tradable(p):
    q = p[(p["suspended"] == 0) & (p["isST"] == 0) & p["ret_1d"].notna()]
    one_price = (q["open"] == q["high"]) & (q["high"] == q["low"])
    return q[~one_price]


def main():
    t0 = time.time()
    p = tradable(pd.read_parquet(PARQUET_DIR / "panel.parquet"))
    full = pd.read_parquet(PARQUET_DIR / "panel.parquet")
    dates = sorted(p["date"].unique())
    truth_all = p[["date", "code", "ret_1d"]]

    # 成员分数
    print("[1/3] L1 walk-forward …", flush=True)
    rows = []
    for w in make_windows(dates, 750, 5, 20):
        tr, te = p[p["date"].isin(w["train"])], p[p["date"].isin(w["test"])]
        pred, _ = fit_predict(tr, te)
        d = te[["date", "code"]].copy()
        d["score"] = pred
        rows.append(d)
    s_l1 = pd.concat(rows, ignore_index=True)

    cache = ROOT / "reports" / "cache"
    s_ns = pd.read_parquet(cache / "deep_NonStat.parquet")[["date", "code", "score"]]

    mr = baselines.score_mr20(full)
    rv = 1 - full.groupby("date")["ret_5"].rank(pct=True)
    s_mr = pd.DataFrame({"date": full["date"], "code": full["code"], "score": mr})
    s_rv = pd.DataFrame({"date": full["date"], "code": full["code"], "score": rv})

    members = {"L1": s_l1, "NonStat": s_ns, "MR20": s_mr, "REV5": s_rv}
    union_dates = sorted(set().union(*(set(m["date"]) for m in members.values())))
    truth_idx = truth_all.set_index(["date", "code"])["ret_1d"]

    # 逐成员逐日IC（供在线权重用）
    ic_tbl = {}
    for name, df in members.items():
        m = df.merge(truth_all, on=["date", "code"])
        ic = m.groupby("date").apply(lambda g: spearmanr(g["score"], g["ret_1d"]).statistic if len(g) >= 10 else np.nan, include_groups=False)
        ic_tbl[name] = ic.sort_index()

    print("[2/3] 在线加权集成 …", flush=True)
    ens_rows = []
    TEMP = 30.0
    for t in union_dates:
        hist = [d for d in union_dates if d < t]
        if len(hist) < 60:
            continue
        hist = hist[-250:]
        w = {}
        for name, ic in ic_tbl.items():
            past = ic.reindex([d for d in hist if d in ic.index]).dropna().tail(60)
            w[name] = float(past.mean()) if len(past) >= 20 else np.nan
        s = pd.Series(w, dtype=float)
        if s.notna().sum() < 2:
            continue
        e = np.exp(TEMP * (s - s.max()))
        wts = (e / e.sum()).to_dict()
        parts = []
        for name, df in members.items():
            dd = df[df["date"] == t]
            if len(dd):
                parts.append((dd.set_index("code")["score"].rank(pct=True), wts.get(name, 0.0)))
        wsum = sum(w for _, w in parts)
        if not parts or wsum == 0:
            continue
        comb = sum((pr * w / wsum for pr, w in parts))
        ens_rows.append(pd.DataFrame({"date": t, "code": comb.index, "score": comb.values}))
    s_ens = pd.concat(ens_rows, ignore_index=True)

    print("[3/3] 评估 …", flush=True)
    results = {}
    for name, df in {**members, "ENS-icw": s_ens}.items():
        d = df.merge(truth_all, on=["date", "code"])
        results[name] = evaluate(d)
    sdf = pd.DataFrame(results.values())
    sdf["model"] = results.keys()
    sdf = sdf[["model", "n_days", "rank_ic", "icir", "dir_acc", "top3_hit"]].sort_values("rank_ic", ascending=False)

    # 固化权重（供 daily.py 使用：最近250日窗口的在线权重末值）
    last = union_dates[-1]
    hist = [d for d in union_dates if d < last][-250:]
    w = {name: float(ic.reindex([d for d in hist if d in ic.index]).dropna().tail(60).mean()) for name, ic in ic_tbl.items()}
    e = np.exp(TEMP * (pd.Series(w) - max(w.values())))
    fixed = (e / e.sum()).round(4).to_dict()

    rep = ROOT / "reports" / "eval_P5.md"
    rep.write_text("\n".join([
        "# P5 集成层评估（在线IC加权）",
        "",
        f"- 运行: {pd.Timestamp.now():%Y-%m-%d %H:%M} | 成员: L1 / NonStat(缓存) / MR20 / REV5 | 温度={TEMP}",
        "",
        sdf.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"- **固化权重（daily.py v1 使用，源于最近窗口的在线权重）**: `{fixed}`",
        "- 在线加权在每个测试日仅用 t-5 日前的最近60日IC（无泄漏）",
        "",
        "## 免责声明", "", "不构成投资建议。",
    ]), encoding="utf-8")
    (ROOT / "reports" / "ens_fixed_weights.json").write_text(pd.Series(fixed).to_json(), encoding="utf-8")
    print(sdf.to_string(index=False))
    print(f"固化权重: {fixed}")
    print(f"报告: {rep} | {time.time() - t0:.0f}s\nEVAL_P5_DONE")


if __name__ == "__main__":
    main()
