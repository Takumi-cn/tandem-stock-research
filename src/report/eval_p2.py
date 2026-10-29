"""P2 首份诚实评估报告（docs/01 §4.2 + docs/07 §3）。

运行: python -m src.report.eval_p2
产出: reports/eval_P2.md + reports/eval_P2_summary.csv
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.walkforward import evaluate, make_windows  # noqa: E402
from src.features.build import FEATURE_COLS  # noqa: E402
from src.models import baselines  # noqa: E402
from src.models.l1_lgbm import fit_predict  # noqa: E402
from src.utils.config import PARQUET_DIR, ROOT  # noqa: E402


def tradable(p: pd.DataFrame) -> pd.DataFrame:
    """可交易样本：非停牌、非ST、非一字板（open==high==low==close 视为一字，近似）、有标签。"""
    q = p[(p["suspended"] == 0) & (p["isST"] == 0) & p["ret_1d"].notna()]
    one_price = (q["open"] == q["high"]) & (q["high"] == q["low"])
    return q[~one_price]


def main():
    t0 = time.time()
    p = pd.read_parquet(PARQUET_DIR / "panel.parquet")
    p = tradable(p)
    dates = sorted(p["date"].unique())
    windows = make_windows(dates)
    print(f"面板 {len(p):,} 行 | 交易日 {len(dates)} | walk-forward 窗口 {len(windows)} 个（750+5/20）", flush=True)

    rows = []
    for wi, w in enumerate(windows):
        tr = p[p["date"].isin(w["train"])]
        te = p[p["date"].isin(w["test"])]
        # L1
        pred_l1, feats = fit_predict(tr, te, weighted=False)
        # L1-W（WRL 加权）
        pred_l1w, _ = fit_predict(tr, te, weighted=True)
        # L0 基线分数
        preds = {"L1-LGBM": pred_l1, "L1-LGBM-W": pred_l1w}
        for name, fn in baselines.BASELINES.items():
            preds[name] = fn(pd.concat([tr, te])).loc[te.index].values
        base = te[["date", "code", "ret_1d"]].reset_index(drop=True)
        for name, sc in preds.items():
            d = base.copy()
            d["score"] = sc
            d["window"] = wi
            d["model"] = name
            rows.append(d)
        if (wi + 1) % 3 == 0 or wi == len(windows) - 1:
            print(f"  窗口 {wi + 1}/{len(windows)} 完成 | 特征 {len(feats)} 个 | 累计 {time.time() - t0:.0f}s", flush=True)

    allp = pd.concat(rows, ignore_index=True)
    summary = []
    for name, g in allp.groupby("model"):
        m = evaluate(g)
        m["model"] = name
        summary.append(m)
    sdf = pd.DataFrame(summary)[["model", "n_days", "rank_ic", "icir", "dir_acc", "top3_hit"]].sort_values("rank_ic", ascending=False)

    # 逐窗口的 L1 稳定性（衡量漂移鲁棒性）
    l1w = allp[allp["model"] == "L1-LGBM"].groupby("window").apply(lambda g: evaluate(g)[ "rank_ic"], include_groups=False)
    neg_ratio = (l1w < 0).mean()

    # 报告
    rep = ROOT / "reports" / "eval_P2.md"
    rep.parent.mkdir(exist_ok=True)
    lines = [
        "# P2 首份诚实评估报告",
        f"",
        f"- 运行时间: {pd.Timestamp.now():%Y-%m-%d %H:%M} | 窗口数: {len(windows)}（750训练+5禁运/20测试） | 面板: 电池行业98只×5年",
        f"- 预测对象: T+1 截面涨幅排名（rank_1d） | 可交易样本口径: 非停牌/非ST/非一字板",
        f"- 特征: {len(FEATURE_COLS)} 个（九族中的量价七族；财务/资金流/情绪属 P4）",
        "",
        "## 汇总指标",
        "",
        sdf.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"- L1 逐窗口 RankIC 为负的窗口占比: **{neg_ratio:.0%}**（漂移鲁棒性粗测）",
        "",
        "## 判读（对照 docs/01 §4.2 合格线）",
        "",
        f"- L1 RankIC {sdf.set_index('model').loc['L1-LGBM', 'rank_ic']:.4f}（合格线>0.03，优秀>0.08）",
        f"- L1 方向准确率 {sdf.set_index('model').loc['L1-LGBM', 'dir_acc']:.1%}（合格>50%，优秀>55%）",
        f"- L1 是否强于全部 L0 基线: {'是' if sdf.iloc[0]['model'] == 'L1-LGBM' else '否——先修特征不加模型（总规划铁律）'}",
        "",
        "## 免责声明",
        "",
        "本报告为历史滚动验证，不构成投资建议；样本外表现可能随市场结构变化衰减。",
    ]
    rep.write_text("\n".join(lines), encoding="utf-8")
    sdf.to_csv(ROOT / "reports" / "eval_P2_summary.csv", index=False)
    print(sdf.to_string(index=False))
    print(f"\n报告: {rep} | 总耗时 {time.time() - t0:.0f}s")
    print("EVAL_P2_DONE")


if __name__ == "__main__":
    main()
