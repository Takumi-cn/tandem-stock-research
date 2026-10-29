"""TARGET-001（预注册，研究沙盒）：三种预测目标对经济结果的横截面排序对比。

A_rank：target=rank_1d（当前 L1 口径）
B_ret ：target=ret_1d（次日原始收益）
C_exec：target=trade_ret（T+1 open → T+2 close，可交易2日收益）

同一训练样本、同一测试日、同一第一页；评价标准是**经济结果 trade_ret** 的日 Spearman IC。
仅报告，不进入实盘；预注册见 reports/target_v31/prereg.json。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.net_return_v31 import build_labels, realizable  # noqa: E402
from src.eval.top1_audit import COST_SIDE, COST_STRESS, FEATURE_COLS, SENT_COLS  # noqa: E402
from src.report.daily import load_panel  # noqa: E402
from src.report.short_pick import REG_PARAMS  # noqa: E402
from src.utils.config import ROOT, targets  # noqa: E402

import lightgbm as lgb  # noqa: E402

OUT = ROOT / "reports" / "target_v31"
N_DAYS, BLOCK, POOL_N = 120, 20, 20
MODELS = {"A_rank": "rank_1d", "B_ret": "ret_1d", "C_exec": "trade_ret"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = json.loads((OUT / "prereg.json").read_text(encoding="utf-8"))
    assert prereg["status"] == "PRE-REGISTERED"

    panel = load_panel()
    L = str(targets()["long_target"])
    p = build_labels(panel)
    dates_all = sorted(p["date"].unique())
    pos = {d: i for i, d in enumerate(dates_all)}
    trad_all = p[(p["suspended"] == 0) & (p["isST"] == 0)]
    trad = trad_all[trad_all["code"] != L]
    labelled = trad.dropna(subset=["ret_1d", "rank_1d", "trade_ret"]).copy()
    labelled["_pos"] = labelled["date"].map(pos)
    test_dates = sorted(trad.dropna(subset=["ret_1d"])["date"].unique())[-N_DAYS:]
    px = panel.set_index(["code", "date"])[["open", "high", "low", "close", "preclose", "suspended"]]
    print(f"目标对比：{len(test_dates)} 测试日｜训练样本 {len(labelled):,}", flush=True)

    rows = []
    for start in range(0, len(test_dates), BLOCK):
        block = test_dates[start:start + BLOCK]
        tr = labelled[labelled["_pos"] <= pos[block[0]] - 3]
        if len(tr) < 4000:
            continue
        feats = [c for c in FEATURE_COLS + SENT_COLS if c in tr.columns]
        models = {}
        for tag, target in MODELS.items():
            y = tr[target].clip(-0.25, 0.25) if target == "trade_ret" else tr[target]
            models[tag] = lgb.LGBMRegressor(**REG_PARAMS).fit(tr[feats], y)
        for d in block:
            g = trad[trad["date"] == d].dropna(subset=["ret_1d"])
            if g.empty:
                continue
            fp = g.sort_values("pct_chg", ascending=False).head(POOL_N).copy()
            for tag, m in models.items():
                fp[f"s_{tag}"] = m.predict(fp[feats])
            sub = fp.dropna(subset=["trade_ret"])
            row = {"date": d, "n": int(len(sub))}
            for tag in MODELS:
                row[f"ic_{tag}"] = (float(st.spearmanr(sub[f"s_{tag}"], sub["trade_ret"]).statistic)
                                    if len(sub) >= 5 else np.nan)
                top = fp.sort_values(f"s_{tag}", ascending=False).iloc[0]
                rb = realizable(px, dates_all, str(top["code"]), d)
                row[f"top_{tag}"] = str(top["code"])
                row[f"r_{tag}"] = rb["r"]
                row[f"net_{tag}"] = (rb["r"] - 2 * COST_SIDE) if pd.notna(rb["r"]) else np.nan
                row[f"net25_{tag}"] = (rb["r"] - 2 * COST_STRESS) if pd.notna(rb["r"]) else np.nan
            rows.append(row)
        print(f"  块 {start // BLOCK + 1}/{(len(test_dates) + BLOCK - 1) // BLOCK} 完成", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "days.csv", index=False, encoding="utf-8-sig")

    res = {"exp_id": "TARGET-001", "n_days": int(len(df)), "models": {}}
    for tag in MODELS:
        ic = df[f"ic_{tag}"]
        net = df[f"net_{tag}"].dropna()
        net25 = df[f"net25_{tag}"].dropna()
        ud = sorted(df["date"].unique())
        half = set(ud[: len(ud) // 2])
        first, second = df[df["date"].isin(half)], df[~df["date"].isin(half)]
        res["models"][tag] = {
            "target": MODELS[tag],
            "ic_mean": round(float(ic.mean()), 4),
            "ic_std": round(float(ic.std()), 4),
            "icir": round(float(ic.mean() / ic.std()), 4) if ic.std() else None,
            "ic_pos_share": round(float((ic > 0).mean()), 4),
            "ic_first_half": round(float(first[f"ic_{tag}"].mean()), 4),
            "ic_second_half": round(float(second[f"ic_{tag}"].mean()), 4),
            "top1_trade_days": int(len(net)),
            "top1_per_trade": round(float(net.mean()), 5) if len(net) else None,
            "top1_up": round(float((net > 0).mean()), 4) if len(net) else None,
            "top1_opp": round(float(df[f"net_{tag}"].fillna(0).mean()), 5),
            "top1_stress25": round(float(net25.mean()), 5) if len(net25) else None,
        }
    # 配对检验：C vs A / C vs B（日 IC）
    for other in ("A_rank", "B_ret"):
        d = (df["ic_C_exec"] - df[f"ic_{other}"]).dropna()
        res[f"C_vs_{other}"] = {
            "mean_diff": round(float(d.mean()), 4),
            "t_p": round(float(st.ttest_1samp(d, 0).pvalue), 4) if len(d) > 2 else None,
            "wins": round(float((d > 0).mean()), 4),
        }
    best = max(res["models"], key=lambda k: res["models"][k]["ic_mean"])
    res["best_ic"] = best
    c = res["models"]["C_exec"]
    a = res["models"]["A_rank"]
    lead = bool(c["ic_mean"] > a["ic_mean"] and res["C_vs_A_rank"]["t_p"] is not None
                and res["C_vs_A_rank"]["t_p"] < 0.05
                and c["ic_first_half"] > 0 and c["ic_second_half"] > 0)
    res["lead_for_next_prereg"] = lead

    lines = [
        "# TARGET-001：三种预测目标的横截面排序对比（研究沙盒）",
        "",
        f"- 运行：{pd.Timestamp.now():%Y-%m-%d %H:%M}｜{len(df)} 个测试日｜6块×20日滚动重训（严格 purge）",
        "- 评价：第一页内 预测值 vs **可交易 2 日收益 trade_ret** 的日 Spearman IC",
        "- 仅研究：不写 latest.json、不改生产、不调参",
        "",
        "## 主结果",
        "",
        "| 目标 | 日IC均值 | IC标准差 | ICIR | IC>0占比 | 前半段IC | 后半段IC | Top1每笔净收益 | 25bps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, name in (("A_rank", "A 截面排名 rank_1d（现 L1）"), ("B_ret", "B 次日原始收益 ret_1d"), ("C_exec", "C 可交易2日收益 trade_ret")):
        s = res["models"][tag]
        lines.append(
            f"| {name} | {s['ic_mean']:+.4f} | {s['ic_std']:.4f} | {s['icir']:+.3f} | {s['ic_pos_share']:.1%} | "
            f"{s['ic_first_half']:+.4f} | {s['ic_second_half']:+.4f} | {(s['top1_per_trade'] or 0):+.3%} | {(s['top1_stress25'] or 0):+.3%} |"
        )
    lines += [
        "",
        f"- 配对检验（日 IC）：C−A 均值 {res['C_vs_A_rank']['mean_diff']:+.4f}，p={res['C_vs_A_rank']['t_p']}，C 胜率 {res['C_vs_A_rank']['wins']:.1%}；"
        f"C−B 均值 {res['C_vs_B_ret']['mean_diff']:+.4f}，p={res['C_vs_B_ret']['t_p']}，C 胜率 {res['C_vs_B_ret']['wins']:.1%}",
        f"- IC 最优目标：**{best}**；C 是否达到「下一轮候选线索」标准：**{'是' if lead else '否'}**"
        f"（标准：C 的 IC 显著优于 A、且前后半段同为正）",
        "",
        "## 判读与治理",
        "",
        "- 这项对比回答的是「目标函数是否改变经济排序」，不是「能否盈利」；Top1 净收益只作辅助，样本小、尾部主导。",
        "- 不因单一沙盒结果改生产目标；若 C 有线索，只能作为**新的预注册**（PIT/walk-forward + 独立样本）继续。",
        "- 失败或无差异同样记录，不调特征/树参数救结果。",
        "", "## 免责声明", "", "研究实验输出，不构成投资建议。",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    with open(ROOT / "reports" / "experiment_registry.csv", "a", encoding="utf-8") as f:
        f.write(f"TARGET-001,{pd.Timestamp.now():%Y-%m-%d},电池个股短线-研究沙盒,EXPLORATORY,"
                f"best_ic={best};C_vs_A_p={res['C_vs_A_rank']['t_p']};lead={lead}\n")
    print(json.dumps({k: res["models"][k] for k in MODELS}, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("C_vs_A_rank", "C_vs_B_ret", "best_ic", "lead_for_next_prereg")}, ensure_ascii=False, indent=1, default=float))
    print("TARGET_DONE")


if __name__ == "__main__":
    main()
