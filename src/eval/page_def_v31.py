"""PAGE-DEF-002：第一页定义敏感性审计（只研究，不改生产）。

主假设：成交额前20 vs 涨幅前20；次假设：换手前20、涨幅/波动前20。
同一电池池、同一 T+1 open→T+2 close、14.5bps。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.forward_evidence import limit_of  # noqa: E402
from src.utils.config import PARQUET_DIR, ROOT  # noqa: E402

OUT = ROOT / "reports" / "page_def_v31"
COST = 0.00145
N = 20


def boot(x: np.ndarray, block: int = 5, runs: int = 2000, seed: int = 7):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return {"mean": float(np.mean(x)), "ci95": [None, None], "p_le_0": None, "n": len(x)}
    rng = np.random.default_rng(seed)
    nb = max(1, len(x) // block)
    means = []
    for _ in range(runs):
        starts = rng.integers(0, max(1, len(x) - block + 1), nb)
        means.append(np.concatenate([x[s:s + block] for s in starts]).mean())
    means = np.array(means)
    return {"mean": round(float(x.mean()), 5), "ci95": [round(float(np.quantile(means, 0.025)), 5),
                                                         round(float(np.quantile(means, 0.975)), 5)],
            "p_le_0": round(float((means <= 0).mean()), 4), "n": int(len(x))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = json.loads((OUT / "prereg.json").read_text(encoding="utf-8"))
    assert prereg["status"] == "PRE-REGISTERED"
    panel = pd.read_parquet(PARQUET_DIR / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype(str)
    cons = set(pd.read_parquet(PARQUET_DIR / "cons_battery.parquet")["code"].astype(str))
    d = panel[panel["code"].isin(cons)].copy()
    dates = sorted(d["date"].unique())
    cols = ["open", "high", "low", "close", "preclose", "suspended", "isST", "pct_chg", "amount", "turn", "vol_20"]
    p = {c: d.pivot(index="date", columns="code", values=c).reindex(dates).astype(float) for c in cols}
    ok = (p["suspended"] == 0) & (p["isST"] == 0)
    limits = pd.Series({c: limit_of(c) for c in p["close"].columns})
    entry = p["open"].shift(-1)
    exit_ = p["close"].shift(-2)
    susp1 = p["suspended"].shift(-1)
    up = (entry == p["high"].shift(-1)) & (entry == p["low"].shift(-1)) & (entry == p["close"].shift(-1))
    up = up & ((entry / p["preclose"].shift(-1) - 1) >= (limits * 0.995))
    r = exit_ / entry - 1 - 2 * COST
    r = r.where(ok & (susp1 == 0) & ~up)

    defs = {
        "current_pct": lambda g: g.sort_values("pct_chg", ascending=False),
        "amount": lambda g: g.sort_values("amount", ascending=False),
        "turn": lambda g: g.sort_values("turn", ascending=False),
        "lowvol": lambda g: g.assign(sc=g["pct_chg"] / g["vol_20"].replace(0, np.nan)).sort_values("sc", ascending=False),
    }
    rows = []
    for dt in dates:
        codes = list(p["close"].columns[ok.loc[dt]])
        if len(codes) < N + 5:
            continue
        g = pd.DataFrame({"code": codes}).set_index("code").join(
            pd.DataFrame({c: p[c].loc[dt] for c in ("pct_chg", "amount", "turn", "vol_20")}))
        for name, fn in defs.items():
            page = list(fn(g).head(N).index)
            rest = [c for c in codes if c not in set(page)]
            rp = r.loc[dt, page].dropna()
            rr = r.loc[dt, rest].dropna()
            if len(rp) < 5 or len(rr) < 5:
                continue
            rows.append({"date": dt, "def": name, "page_net": float(rp.mean()),
                         "rest_net": float(rr.mean()), "diff": float(rp.mean() - rr.mean()),
                         "top1_net": float(r.loc[dt, page[0]]) if pd.notna(r.loc[dt, page[0]]) else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "days.csv", index=False, encoding="utf-8-sig")
    res = {}
    for name in defs:
        sub = df[df["def"] == name]
        res[name] = {"days": len(sub), "page": round(float(sub["page_net"].mean()), 5),
                     "rest": round(float(sub["rest_net"].mean()), 5),
                     "diff": round(float(sub["diff"].mean()), 5),
                     "top1": round(float(sub["top1_net"].mean()), 5),
                     "hit": round(float((sub["top1_net"] > 0).mean()), 4),
                     "boot": boot(sub["diff"].to_numpy())}
    cur = df[df["def"] == "current_pct"].set_index("date")["page_net"]
    amt = df[df["def"] == "amount"].set_index("date")["page_net"]
    paired = (amt - cur).dropna()
    prim = boot(paired.to_numpy())
    passed = bool(prim["mean"] > 0 and prim["p_le_0"] is not None and prim["p_le_0"] < 0.05)
    lines = [
        "# PAGE-DEF-002：第一页定义敏感性（预注册）",
        "",
        f"- {pd.Timestamp.now():%Y-%m-%d %H:%M}｜电池 {p['close'].shape[1]} 只｜{len(dates)} 日｜T+1 open→T+2 close｜14.5bps",
        "- 主假设：成交额前20 vs 涨幅前20；次假设只探索",
        "",
        "| 定义 | 天数 | page 净收益 | rest 净收益 | page−rest | Top1 净收益 | Top1 胜率 | diff bootstrap p(≤0) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("current_pct", "涨幅前20（现）"), ("amount", "成交额前20"), ("turn", "换手前20"), ("lowvol", "涨幅/波动前20")):
        x = res[name]
        lines.append(f"| {label} | {x['days']} | {x['page']:+.3%} | {x['rest']:+.3%} | {x['diff']:+.3%} | "
                     f"{x['top1']:+.3%} | {x['hit']:.1%} | {x['boot']['p_le_0']} |")
    lines += [
        "",
        f"- 主假设 amount−current 配对：均值 **{prim['mean']:+.3%}**，95%CI "
        f"[{prim['ci95'][0]:+.3%},{prim['ci95'][1]:+.3%}]，p(≤0)={prim['p_le_0']}（{prim['n']} 天）",
        f"- 判定：**{'PASS（仅进入 60 日前瞻影子，不上线）' if passed else 'NO_PAGE_DEFINITION_EDGE'}**",
        "",
        "## 停止/治理",
        "",
        "- 不调 N、不调行业、不改 55%；只写本目录，不写 latest.json。",
        "- 次假设（换手/低波）只探索，不据此改生产。",
        "", "## 免责声明", "", "研究审计输出，不构成投资建议。",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    metrics = {"exp_id": "PAGE-DEF-002", "res": res, "primary": prim, "passed": passed}
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    with open(ROOT / "reports" / "experiment_registry.csv", "a", encoding="utf-8") as f:
        f.write(f"PAGE-DEF-002,{pd.Timestamp.now():%Y-%m-%d},第一页定义敏感性,AUDIT-COMPLETE,"
                f"{'PASS' if passed else 'NO_PAGE_DEFINITION_EDGE'};amount_minus_cur={prim['mean']};p={prim['p_le_0']}\n")
    print(json.dumps({"amount": res["amount"], "current": res["current_pct"], "primary": prim, "passed": passed},
                     ensure_ascii=False, indent=1, default=float))
    print("PAGE_DEF_DONE")


if __name__ == "__main__":
    main()
