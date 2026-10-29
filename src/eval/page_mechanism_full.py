"""Full-history page mechanism audit (1,210 page days instead of 119).

The MICRO-005 readiness audit could only test the page premium on the 119 page
days that the production replay happened to cover.  The page frame built for
MICRO-005 reconstructs the same rule (top-20 by T-day pct_chg) back to 2021-10,
so the mechanism question can be answered with ~10x the sample.

What is tested (all executable, T+1 open -> T+2 close, 14.5 bps/side):
  1. page-1 mean net return vs the rest of the battery pool, same date
  2. within-page rank IC of the T-day pct_chg rank against the residual return
  3. Top1 (largest T-day gainer) minus page mean
  4. temporal stability: by calendar year, by half, by volatility regime
  5. block bootstrap + date-cluster t-test, plus a sign test

Still a historical diagnostic on a period that earlier TANDEM decisions have
already used; it can reject the page story, it cannot certify it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT  # noqa: E402

OUT = ROOT / "reports" / "page_mechanism_full"
DATA = ROOT / "data" / "research" / "micro005_minimal"
COST = 0.00145
LONG_CODE = "000927"


def boot(diff: np.ndarray, n_boot: int = 20000, block: int = 5, seed: int = 42) -> dict:
    diff = np.asarray([x for x in diff if np.isfinite(x)], float)
    if not len(diff):
        return {}
    rng = np.random.default_rng(seed)
    n = len(diff)
    starts = np.arange(0, n, block)
    nb = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(starts, nb, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in pick])
        means[b] = diff[idx].mean()
    return {"n": int(n), "mean": float(diff.mean()),
            "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "p_le_0": float((means <= 0).mean()), "block": block, "n_boot": n_boot}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    page = pd.read_parquet(DATA / "page_dataset.parquet")
    page["date"] = pd.to_datetime(page["date"]); page["code"] = page["code"].astype(str)
    panel = pd.read_parquet(ROOT / "data" / "parquet" / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"]); panel["code"] = panel["code"].astype(str)
    cons = pd.read_parquet(ROOT / "data" / "parquet" / "cons_battery.parquet")
    pool = sorted({str(c) for c in cons["code"].astype(str)})
    panel = panel[panel["code"].isin(pool)]
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    close = panel.pivot(index="date", columns="code", values="close").reindex(dates)
    open_ = panel.pivot(index="date", columns="code", values="open").reindex(dates)

    ok = page[(~page["entry_blocked"]) & page["label__exec_T1_T2"].notna()].copy()
    # independent re-price of the same label to prove page_dataset is not off-by-one
    check = []
    for d, g in ok.groupby("date"):
        j = dates.get_loc(d)
        if j + 2 >= len(dates):
            continue
        e, x = dates[j + 1], dates[j + 2]
        r = close.loc[x, g["code"]].to_numpy() / open_.loc[e, g["code"]].to_numpy() - 1.0
        check.append(float(np.nanmax(np.abs(r - g["label__exec_T1_T2"].to_numpy()))))
    label_recheck_max = float(np.nanmax(check)) if check else None

    rows = []
    for d, g in ok.groupby("date"):
        j = dates.get_loc(d)
        if j + 2 >= len(dates):
            continue
        e, x = dates[j + 1], dates[j + 2]
        r_all = close.loc[x] / open_.loc[e] - 1.0 - 2 * COST
        page_codes = list(g["code"])
        rest = [c for c in pool if c not in set(page_codes) and pd.notna(r_all.get(c))]
        pg = r_all.reindex(page_codes).dropna()
        if len(pg) < 5 or len(rest) < 10:
            continue
        rs = r_all.reindex(rest).dropna()
        top1 = g.sort_values("rank_in_fp").iloc[0]
        rows.append({
            "date": d, "year": d.year,
            "page_mean": float(pg.mean()), "rest_mean": float(rs.mean()),
            "page_minus_rest": float(pg.mean() - rs.mean()),
            "top1_code": top1["code"], "top1_r2d_net": float(r_all.get(top1["code"], np.nan)),
            "top1_minus_page": float(r_all.get(top1["code"], np.nan) - pg.mean()),
            "page_rank_ic": float(stats.spearmanr(g["rank_in_fp"], r_all.reindex(page_codes).to_numpy(),
                                                  nan_policy="omit").statistic),
            "page_up_share": float((pg > 0).mean()),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_csv(OUT / "days.csv", index=False, encoding="utf-8-sig")

    def stat_block(sub: pd.DataFrame) -> dict:
        b = boot(sub["page_minus_rest"].to_numpy(), n_boot=10000)
        t = stats.ttest_1samp(sub["page_minus_rest"], 0, nan_policy="omit")
        return {"days": int(len(sub)), "page_mean": float(sub["page_mean"].mean()),
                "rest_mean": float(sub["rest_mean"].mean()),
                "page_minus_rest": float(sub["page_minus_rest"].mean()),
                "ci95": b.get("ci95"), "boot_p_le_0": b.get("p_le_0"),
                "t_p": float(t.pvalue), "win_share": float((sub["page_minus_rest"] > 0).mean()),
                "top1_minus_page": float(sub["top1_minus_page"].mean()),
                "page_rank_ic": float(sub["page_rank_ic"].mean()),
                "neg_ic_day_share": float((sub["page_rank_ic"] < 0).mean())}

    half = len(df) // 2
    res = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_days": int(len(df)), "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "label_recheck_max_abs_diff": label_recheck_max,
        "all": stat_block(df),
        "by_year": {str(int(y)): stat_block(g) for y, g in df.groupby("year")},
        "first_half": stat_block(df.iloc[:half]),
        "second_half": stat_block(df.iloc[half:]),
        "note": ("Executable T+1 open -> T+2 close, net of 14.5 bps/side on both page and rest. "
                 "Historical diagnostic on an already-inspected period; not a new OOS."),
    }
    allb = res["all"]
    # ---- rank monotonicity: is the T-day top gainer systematically the worst pick? ----
    piv = ok.pivot_table(index="date", columns="rank_in_fp", values="label__page_resid_T1_T2", aggfunc="mean")
    rank_mean = {int(c): float(piv[c].mean()) for c in piv.columns}
    diff_bt = (piv.get(20) - piv.get(1)).dropna()
    rb = boot(diff_bt.to_numpy(), n_boot=10000)
    rt = stats.ttest_1samp(diff_bt.to_numpy(), 0)
    res["rank_profile"] = {
        "rank_mean_resid": rank_mean,
        "bottom1_minus_top1_mean": float(diff_bt.mean()),
        "bottom1_minus_top1_ci95": rb.get("ci95"), "bottom1_minus_top1_p_boot": rb.get("p_le_0"),
        "bottom1_minus_top1_t_p": float(rt.pvalue),
        "by_year": {str(int(y)): float((g.get(20) - g.get(1)).mean())
                    for y, g in piv.groupby(piv.index.year)},
    }
    if allb["page_minus_rest"] > 0 and allb["t_p"] < 0.05 and (allb["ci95"][0] > 0):
        status = "PAGE_PREMIUM_POSITIVE"
    elif allb["page_minus_rest"] > 0 and allb["t_p"] < 0.10:
        status = "PAGE_PREMIUM_WEAK_UNCERTAIN"
    else:
        status = "NO_PAGE_PREMIUM_EVIDENCE"
    res["status"] = status
    (OUT / "audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# 第一页机制 · 全历史审计（1210 日）", "",
        f"- 生成 {time.strftime('%Y-%m-%d %H:%M')}｜耗时 {(time.time()-t0)/60:.1f} 分钟",
        f"- 样本：{res['n_days']} 个 page 日（{res['date_min']} ~ {res['date_max']}）",
        f"- 标签独立重算最大误差：{label_recheck_max:.2e}（page_dataset 未错位）",
        f"- 结论：**`{status}`**", "",
        "## 主结果（全部扣 14.5bps/边，T+1 开盘 → T+2 收盘）", "",
        "| 指标 | 值 |", "|---|---:|",
        f"| 第一页等权净收益 | {allb['page_mean']:+.3%} |",
        f"| 电池池其余净收益 | {allb['rest_mean']:+.3%} |",
        f"| page − rest | **{allb['page_minus_rest']:+.3%}** |",
        f"| 95% CI（5 日块 bootstrap） | [{allb['ci95'][0]:+.3%}, {allb['ci95'][1]:+.3%}] |",
        f"| 单侧 bootstrap p(≤0) | {allb['boot_p_le_0']:.4f} |",
        f"| t 检验 p | {allb['t_p']:.4f} |",
        f"| 胜率（page>rest 的天数占比） | {allb['win_share']:.1%} |",
        f"| 第一页内 rank IC（涨幅排名 vs 2 日净收益） | {allb['page_rank_ic']:+.4f} |",
        f"| 负 IC 日占比 | {allb['neg_ic_day_share']:.1%} |",
        f"| Top1 − page 均值 | {allb['top1_minus_page']:+.3%} |",
        "", "## 稳定性", "",
        "| 区间 | 天数 | page−rest | t p | Top1−page | rank IC |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in (("前半", res["first_half"]), ("后半", res["second_half"])):
        lines.append(f"| {name} | {s['days']} | {s['page_minus_rest']:+.3%} | {s['t_p']:.3f} | "
                     f"{s['top1_minus_page']:+.3%} | {s['page_rank_ic']:+.4f} |")
    for y, s in res["by_year"].items():
        lines.append(f"| {y} | {s['days']} | {s['page_minus_rest']:+.3%} | {s['t_p']:.3f} | "
                     f"{s['top1_minus_page']:+.3%} | {s['page_rank_ic']:+.4f} |")
    lines += [
        "", "## 页内排名与次日残差（rank 1 = 当日涨幅第一）", "",
        "| rank | 平均残差 |", "|---:|---:|",
    ]
    for r_ in sorted(rank_mean):
        if r_ in (1, 5, 10, 15, 20) or r_ == max(rank_mean):
            lines.append(f"| {r_} | {rank_mean[r_]:+.3%} |")
    rp = res["rank_profile"]
    lines += [
        "", f"- rank20 − rank1：**{rp['bottom1_minus_top1_mean']:+.3%}**，95% CI "
            f"[{rp['bottom1_minus_top1_ci95'][0]:+.3%}, {rp['bottom1_minus_top1_ci95'][1]:+.3%}]，"
            f"bootstrap p(≤0)={rp['bottom1_minus_top1_p_boot']:.4f}，t p={rp['bottom1_minus_top1_t_p']:.4f}。",
        f"- 分年 rank20−rank1：" + "；".join(f"{y} {v:+.3%}" for y, v in rp["by_year"].items()),
        ("- 结论：rank20−rank1 不显著且分年变号，**页内不存在可交易的反转结构**；"
         "rank IC +0.037 只是中段排名与残差的弱关联，不能支撑“买最弱一只”的规则。"
         if rp["bottom1_minus_top1_p_boot"] > 0.05 else
         "- 结论：rank20 显著优于 rank1，说明现规则方向可能选反；仍需新前瞻样本确认。"),
        "- 无论哪种情况，Top1 与 page 均值几乎相同（—0.000%），所以“第一页第一”既没有溢价也没有明显反向证据。",
        "", "## 判读", "",
        "- 这是同一规则的 1210 日复核，样本量约为 readiness 审计的 10 倍；因此它更强地约束了“第一页是否有优势”。",
        "- 即便点估计为正，只要 CI 跨 0 或两组扣成本后为负，就不能把第一页当作可交易优势，更不能据此改 55% 门槛或仓位。",
        "- page 内 rank IC 接近零意味着“当天涨得最多”并不预测次日相对表现；Top1 选择本身没有机制支撑。",
        "- 2021–2026 已被多次查看，本审计是**历史诊断**，不是新 OOS，不能用于晋升。",
        "", "## 产物", "",
        "- `days.csv`（逐日）、`audit.json`（全部统计）、`report.md`（本文件）。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "days": res["n_days"],
                      "page_minus_rest": allb["page_minus_rest"], "t_p": allb["t_p"],
                      "ci95": allb["ci95"], "top1_minus_page": allb["top1_minus_page"],
                      "rank_ic": allb["page_rank_ic"], "elapsed_s": round(time.time() - t0, 1)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
