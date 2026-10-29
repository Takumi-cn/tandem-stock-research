"""NETRET-001（预注册）：成本感知的「可交易净期望」候选 vs 现有 p_up Top1。

预注册见 reports/net_return_v31/prereg.json（运行前写定，不调阈值/不调树参数）。

口径：
- 宇宙：电池第一页 20 只（剔除 000927）；测试日与 TOP1-AUDIT-002 完全一致（最近120个有标签交易日）
- 标签：trade_ret = close_{T+2}/open_{T+1} - 1（T+1 开盘买、T+2 收盘卖，最短合规）
- 训练：每 20 日块滚动重训；严格 purge（label_end=pos(T)+2 < block 起点）
- 候选：Huber 回归 pred_ret；score = pred_ret - 2*14.5bps；score>0 才出手，第一页取 max score
- 基线：B0 = p_up Top1 每日必做；B1 = p_up Top1 且 p_up>=0.55（生产规则）
- 主指标：每日机会净收益（无交易=0）；辅以每笔净收益、分半、25bps 压力、按日 bootstrap

失败即 FAIL，保持生产规则不变；不写 latest.json。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.top1_audit import (  # noqa: E402
    COST_SIDE, COST_STRESS, FEATURE_COLS, SENT_COLS,
    boot_by_date, is_limit_down_close, is_limit_up_open, limit_of,
)
from src.report.daily import load_panel  # noqa: E402
from src.report.short_pick import CLF_PARAMS, REG_PARAMS, TRADE_P, UpModel  # noqa: E402
from src.utils.config import ROOT, targets  # noqa: E402

import lightgbm as lgb  # noqa: E402

OUT = ROOT / "reports" / "net_return_v31"
N_DAYS, BLOCK, POOL_N = 120, 20, 20


def build_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """向量化可交易标签（不做卖出顺延；评估阶段用精确路径）。"""
    p = panel.sort_values(["code", "date"]).reset_index(drop=True)
    g = p.groupby("code", sort=False)
    o1, h1, l1, c1, pc1 = (g[c].shift(-1) for c in ("open", "high", "low", "close", "preclose"))
    c2 = g["close"].shift(-2)
    s1, s2 = g["suspended"].shift(-1), g["suspended"].shift(-2)
    lim = p["code"].map(limit_of).astype(float)
    oneword = (o1 == h1) & (h1 == l1) & (l1 == c1)
    limit_up = oneword & ((o1 / pc1 - 1) >= lim * 0.995)
    ok = (s1 == 0) & (s2 == 0) & (~limit_up.fillna(False))
    p["trade_ret"] = np.where(ok, c2 / o1 - 1, np.nan)
    return p


def realizable(px, dates_all, code, d):
    """精确的可交易路径：T+1 开盘买，T+2 收盘卖；跌停/停牌顺延 ≤3 日。"""
    seq = [x for x in dates_all if x > d]
    if len(seq) < 2:
        return {"block": "no_next", "r": np.nan}
    o1 = px.loc[(code, seq[0])] if (code, seq[0]) in px.index else None
    if o1 is None or bool(o1["suspended"]):
        return {"block": "susp_t1", "r": np.nan}
    if is_limit_up_open(o1["open"], o1["high"], o1["low"], o1["close"], o1["preclose"], code):
        return {"block": "limit_up_open", "r": np.nan}
    entry = float(o1["open"])
    for k in range(1, 4):
        if len(seq) <= k:
            break
        cc = px.loc[(code, seq[k])] if (code, seq[k]) in px.index else None
        if cc is None or bool(cc["suspended"]) or is_limit_down_close(cc["close"], cc["preclose"], code):
            continue
        return {"block": "", "entry": entry, "exit": float(cc["close"]), "r": float(cc["close"]) / entry - 1}
    return {"block": "no_exit", "r": np.nan}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = json.loads((OUT / "prereg.json").read_text(encoding="utf-8"))
    assert prereg["status"] == "PRE-REGISTERED", "预注册文件状态异常"

    panel = load_panel()
    L = str(targets()["long_target"])
    p = build_labels(panel)
    dates_all = sorted(p["date"].unique())
    pos = {d: i for i, d in enumerate(dates_all)}
    trad_all = p[(p["suspended"] == 0) & (p["isST"] == 0)]
    trad = trad_all[trad_all["code"] != L]
    px = panel.set_index(["code", "date"])[["open", "high", "low", "close", "preclose", "suspended"]]

    labelled = trad.dropna(subset=["trade_ret"]).copy()
    labelled["_pos"] = labelled["date"].map(pos)
    test_dates = sorted(trad.dropna(subset=["ret_1d"])["date"].unique())[-N_DAYS:]
    print(f"候选实验：{len(test_dates)} 测试日 | 训练样本 {len(labelled):,}", flush=True)

    rows = []
    for start in range(0, len(test_dates), BLOCK):
        block = test_dates[start:start + BLOCK]
        p0 = pos[block[0]]
        # 基线 = 生产规则原样复刻（训练样本 ret_1d，date < block 起点；与 TOP1-AUDIT-002 完全一致）
        tr_b = trad.dropna(subset=["ret_1d", "rank_1d"])
        tr_b = tr_b[tr_b["date"].map(pos) < p0]
        # 候选按可交易标签（T+1 open → T+2 close），严格 purge 两天
        tr = labelled[labelled["_pos"] <= p0 - 3]
        if len(tr) < 4000 or len(tr_b) < 4000:
            continue
        feats = [c for c in FEATURE_COLS + SENT_COLS if c in tr.columns]
        base = UpModel(feats=feats).fit(tr_b)
        y = tr["trade_ret"].clip(-0.25, 0.25)
        reg = lgb.LGBMRegressor(**REG_PARAMS).fit(tr[feats], y)
        clf = lgb.LGBMClassifier(**CLF_PARAMS).fit(tr[feats], (tr["trade_ret"] > 0).astype(int))
        for d in block:
            g = trad[trad["date"] == d].dropna(subset=["ret_1d"])
            if g.empty:
                continue
            fp = g.sort_values("pct_chg", ascending=False).head(POOL_N).copy()
            X = fp[feats]
            b = base.predict(X)
            fp["p_up"] = b["p_up"].values
            fp["exp_ret"] = b["exp_ret"].values
            fp["pred_ret"] = reg.predict(X)
            fp["p_net_up"] = clf.predict_proba(X)[:, 1]
            fp["score"] = fp["pred_ret"] - 2 * COST_SIDE
            b0 = fp.sort_values(["p_up", "exp_ret"], ascending=False).iloc[0]
            c = fp.sort_values(["score", "p_net_up"], ascending=False).iloc[0]
            rb = realizable(px, dates_all, str(b0["code"]), d)
            rc = realizable(px, dates_all, str(c["code"]), d)
            rows.append({
                "date": d,
                "b0_code": str(b0["code"]), "b0_p": round(float(b0["p_up"]), 4), "b0_r": rb["r"], "b0_block": rb["block"],
                "b1_trade": bool(b0["p_up"] >= TRADE_P),
                "cand_code": str(c["code"]), "cand_p": round(float(c["p_up"]), 4),
                "cand_pred": round(float(c["pred_ret"]), 5), "cand_score": round(float(c["score"]), 5),
                "cand_pnet": round(float(c["p_net_up"]), 4), "cand_r": rc["r"], "cand_block": rc["block"],
                "cand_trade": bool(c["score"] > 0),
            })
        print(f"  块 {start // BLOCK + 1}/{(len(test_dates) + BLOCK - 1) // BLOCK} 完成", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trades.csv", index=False, encoding="utf-8-sig")

    def net(col: str, cost: float) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce") - 2 * cost

    out = {"exp_id": prereg["exp_id"], "n_days": int(len(df))}
    for tag, traded, col in (
        ("B0", pd.Series(True, index=df.index), "b0_r"),
        ("B1", df["b1_trade"], "b0_r"),
        ("CAND", df["cand_trade"], "cand_r"),
    ):
        r = net(col, COST_SIDE)
        opp = pd.Series(np.where(traded & r.notna(), r, 0.0), index=df.index)
        per = r[traded & r.notna()]
        out[tag] = {
            "trade_days": int((traded & r.notna()).sum()),
            "abstain_days": int(len(df) - (traded & r.notna()).sum()),
            "opp_mean": round(float(opp.mean()), 5),
            "opp_sum": round(float(opp.sum()), 5),
            "per_trade_mean": round(float(per.mean()), 5) if len(per) else None,
            "per_trade_median": round(float(per.median()), 5) if len(per) else None,
            "per_trade_up": round(float((per > 0).mean()), 4) if len(per) else None,
            "per_trade_boot": boot_by_date(df[traded & r.notna()].assign(net=per), "net"),
            "opp_boot": boot_by_date(df.assign(v=opp), "v"),
            "stress25_opp_mean": round(float(np.where(traded & net(col, COST_STRESS).notna(), net(col, COST_STRESS), 0.0).mean()), 5),
        }
    # 配对差：候选 - B0（同日）
    r_b0 = net("b0_r", COST_SIDE).fillna(0.0)
    r_c = net("cand_r", COST_SIDE)
    opp_c = pd.Series(np.where(df["cand_trade"] & r_c.notna(), r_c, 0.0), index=df.index)
    df["diff_b0"] = opp_c - r_b0
    out["paired_vs_B0"] = boot_by_date(df, "diff_b0")
    # 分半（按测试日）
    ud = sorted(df["date"].unique())
    half = set(ud[: len(ud) // 2])
    halves = {}
    for tag, sub in (("first", df[df["date"].isin(half)]), ("second", df[~df["date"].isin(half)])):
        rc = net("cand_r", COST_SIDE)[sub.index]
        per = rc[sub["cand_trade"] & rc.notna()]
        rb = net("b0_r", COST_SIDE)[sub.index]
        halves[tag] = {
            "days": int(len(sub)),
            "cand_trade_days": int((sub["cand_trade"] & rc.notna()).sum()),
            "cand_per_trade": round(float(per.mean()), 5) if len(per) else None,
            "cand_opp_mean": round(float(pd.Series(np.where(sub["cand_trade"] & rc.notna(), rc, 0.0)).mean()), 5),
            "b0_opp_mean": round(float(pd.Series(np.where(rb.notna(), rb, 0.0)).mean()), 5),
        }
    out["halves"] = halves

    # 停止条件
    c = out["CAND"]
    checks = {
        "cand_opp > B0_opp": bool(c["opp_mean"] > out["B0"]["opp_mean"]),
        "cand_opp > B1_opp": bool(c["opp_mean"] > out["B1"]["opp_mean"]),
        "per_trade>0 且 p<0.05": bool((c["per_trade_mean"] or -1) > 0 and c["per_trade_boot"]["p_le_0"] < 0.05),
        "前后半段均>0": bool((halves["first"]["cand_per_trade"] or -1) > 0 and (halves["second"]["cand_per_trade"] or -1) > 0),
        "25bps 压力>0": bool(c["stress25_opp_mean"] > 0),
    }
    passed = all(checks.values())
    out["gate"] = {"checks": checks, "passed": passed}

    lines = [
        "# NETRET-001：成本感知净期望候选（预注册）",
        "",
        f"- 运行：{pd.Timestamp.now():%Y-%m-%d %H:%M}｜{len(df)} 个测试日｜6块×20日滚动重训（严格 purge）",
        f"- 口径：T+1 开盘买 → T+2 收盘卖；成本 {COST_SIDE*1e4:.1f}bps/边（压力 {COST_STRESS*1e4:.0f}bps）",
        f"- 候选规则：score = Huber 预测 trade_ret − 往返成本；score>0 才出手；第一页取 max score",
        "- 预注册：`prereg.json`（运行前写定）；不调阈值、不替换生产规则",
        "",
        "## 主结果（每日机会净收益，无交易记 0）",
        "",
        "| 规则 | 交易天数 | 空仓天数 | 日均机会净收益 | 每笔净收益 | 每笔中位数 | 每笔胜率 | 25bps 日均 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, name in (("B0", "B0 p_up Top1（必做）"), ("B1", "B1 生产规则（p≥55%）"), ("CAND", "候选（净期望>0）")):
        s = out[tag]
        lines.append(
            f"| {name} | {s['trade_days']} | {s['abstain_days']} | {s['opp_mean']:+.3%} | "
            f"{(s['per_trade_mean'] or 0):+.3%} | {(s['per_trade_median'] or 0):+.3%} | "
            f"{(s['per_trade_up'] or 0):.1%} | {s['stress25_opp_mean']:+.3%} |"
        )
    lines += [
        "",
        f"- 候选 − B0 同日配对差：{out['paired_vs_B0']['mean']:+.3%}，95%CI "
        f"[{out['paired_vs_B0']['ci95'][0]:+.3%},{out['paired_vs_B0']['ci95'][1]:+.3%}]，p(≤0)={out['paired_vs_B0']['p_le_0']:.3f}",
        f"- 候选日均机会净收益 bootstrap：95%CI [{c['opp_boot']['ci95'][0]:+.3%},{c['opp_boot']['ci95'][1]:+.3%}]，"
        f"p(≤0)={c['opp_boot']['p_le_0']:.3f}",
        f"- 候选每笔净收益 bootstrap：95%CI [{c['per_trade_boot']['ci95'][0]:+.3%},{c['per_trade_boot']['ci95'][1]:+.3%}]，"
        f"p(≤0)={c['per_trade_boot']['p_le_0']:.3f}（{c['per_trade_boot']['n_days']} 个交易日）",
        "",
        "## 分半一致性",
        "",
        "| 段 | 天数 | 候选交易天数 | 候选每笔净收益 | 候选日均机会 | B0 日均机会 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 前半段 | {halves['first']['days']} | {halves['first']['cand_trade_days']} | "
        f"{(halves['first']['cand_per_trade'] or 0):+.3%} | {halves['first']['cand_opp_mean']:+.3%} | {halves['first']['b0_opp_mean']:+.3%} |",
        f"| 后半段 | {halves['second']['days']} | {halves['second']['cand_trade_days']} | "
        f"{(halves['second']['cand_per_trade'] or 0):+.3%} | {halves['second']['cand_opp_mean']:+.3%} | {halves['second']['b0_opp_mean']:+.3%} |",
        "",
        "## 停止条件",
        "",
    ] + [f"- {'✅' if v else '❌'} {k}" for k, v in checks.items()] + [
        "",
        f"**判定：{'PASS（仅表示值得进入下一阶段验证，不自动上线）' if passed else 'FAIL（保持生产规则不变，不调参救结果）'}**",
        "",
        "## 局限",
        "",
        "- 非 PIT 成分；单次 120 日回放；2 日持有叠加未做组合容量/资金曲线建模；",
        "- 候选只做了一次预注册，未做 walk-forward/DSR/PBO；通过也不等于可实盘；",
        "- 不写 latest.json、不替换 55% 门槛、不改变桌面端输出。",
        "", "## 免责声明", "", "研究实验输出，不构成投资建议。",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    reg = ROOT / "reports" / "experiment_registry.csv"
    with open(reg, "a", encoding="utf-8") as f:
        if f.tell() == 0:
            f.write("exp_id,date,scope,status,conclusion\n")
        f.write(f"NETRET-001,{pd.Timestamp.now():%Y-%m-%d},电池个股短线,AUDIT-COMPLETE,"
                f"{'PASS' if passed else 'FAIL'};cand_opp={c['opp_mean']};b0_opp={out['B0']['opp_mean']};"
                f"b1_opp={out['B1']['opp_mean']}\n")
    print(json.dumps({k: {kk: vv for kk, vv in out[k].items() if kk != 'opp_boot'} if isinstance(out[k], dict) else out[k]
                      for k in ("B0", "B1", "CAND", "gate")}, ensure_ascii=False, indent=1, default=float))
    print("NETRET_DONE")


if __name__ == "__main__":
    main()
