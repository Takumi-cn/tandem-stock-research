"""Assess whether broadening the universe / portfolio construction is worth it.

No model training, no production writes.  Uses the frozen 19-fold B OOS
predictions and the page dataset labels to answer three questions:

1. Within the current 20-name battery page, does top-K equal-weight improve
   account-level outcomes versus top-1?
2. How much independent breadth does the page actually contain?
3. What statistical power and cost hurdle would a genuinely broad portfolio
   face, and what data is missing to test it?

Outputs: reports/pool_portfolio_decision/*.
Run:
  .venv\\Scripts\\python.exe -m src.eval.pool_portfolio_assessment
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metric_reconciliation import account_event  # noqa: E402
from src.eval.micro005_minimal import boot_p  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

OUT = ROOT / "reports" / "pool_portfolio_decision"
B_PATH = ROOT / "data" / "research" / "micro005_encoder_002" / "oos_pred_B_21f.parquet"
PAGE_PATH = ROOT / "data" / "research" / "micro005_minimal" / "page_dataset.parquet"
PANEL_PATH = ROOT / "data" / "parquet" / "panel.parquet"
PIT_STATUS = ROOT / "data" / "pit" / "STATUS.json"
COST_SIDE = 0.00145
KS = [1, 3, 5, 10, 20]


def load_b() -> pd.DataFrame:
    b = pd.read_parquet(B_PATH)
    b["date"] = pd.to_datetime(b["date"])
    b["code"] = b["code"].astype(str)
    if b.duplicated(["date", "code"]).any():
        raise SystemExit("duplicate (date, code) in B OOS")
    return b.sort_values(["date", "code"]).reset_index(drop=True)


def topk_daily(b: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    prev: set[str] = set()
    for d, g in b.groupby("date"):
        g = g.sort_values("pred", ascending=False)
        sel = g.head(k)
        codes = set(sel["code"])
        rows.append({
            "date": d, "k": k, "n": len(sel),
            "gross": float(sel["label__exec_T1_T2"].mean()),
            "page_gross": float(g["label__exec_T1_T2"].mean()),
            "turnover": (1 - len(codes & prev) / max(1, len(codes))) if prev else np.nan,
        })
        prev = codes
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize(d: pd.DataFrame, dates: list[pd.Timestamp]) -> dict:
    net = d["gross"] - 2 * COST_SIDE
    net25 = d["gross"] - 2 * 0.0025
    net40 = d["gross"] - 2 * 0.0040
    nav = (1 + net).cumprod()
    dd = float((nav / nav.cummax() - 1).min())
    acct = {}
    for cb in (14.5, 25.0, 40.0):
        sel = d[["date", "gross"]].copy()
        sel["r_2d"] = sel["gross"]
        acct[cb] = account_event(sel, dates, 2, cb)["final_equity"]
    excess = d["gross"] - d["page_gross"]
    return {
        "days": int(len(d)),
        "mean_gross": float(d["gross"].mean()),
        "mean_net14": float(net.mean()),
        "mean_net25": float(net25.mean()),
        "mean_net40": float(net40.mean()),
        "std_net14": float(net.std()),
        "sharpe_net14": float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else None,
        "geo_net14_ann": float((1 + net).prod() ** (252 / len(d)) - 1),
        "cohort_nav_maxdd": dd,
        "hit_net14": float((net > 0).mean()),
        "avg_turnover": float(d["turnover"].dropna().mean()),
        "mean_excess_vs_page": float(excess.mean()),
        "p_net14": float(boot_p(net.to_numpy())["p_one_sided"]),
        "p_excess": float(boot_p(excess.to_numpy())["p_one_sided"]),
        "account14": float(acct[14.5]),
        "account25": float(acct[25.0]),
        "account40": float(acct[40.0]),
    }


def breadth() -> dict:
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    piv = panel.pivot(index="date", columns="code", values="pct_chg")
    corr = piv.corr(min_periods=200)
    iu = np.triu_indices_from(corr.values, 1)
    rho = float(np.nanmean(corr.values[iu]))
    page = pd.read_parquet(PAGE_PATH)
    page["code"] = page["code"].astype(str)
    codes = [c for c in sorted(page["code"].unique()) if c in piv.columns]
    sub = corr.loc[codes, codes]
    iu2 = np.triu_indices_from(sub.values, 1)
    rho_page = float(np.nanmean(sub.values[iu2]))
    n_page = int(len(codes))
    return {
        "panel_codes": int(piv.shape[1]), "panel_dates": int(piv.shape[0]),
        "avg_pairwise_corr_pct_chg": rho,
        "n_eff_panel": float(piv.shape[1] / (1 + (piv.shape[1] - 1) * rho)),
        "page_codes": n_page,
        "avg_pairwise_corr_page": rho_page,
        "n_eff_page20": float(20 / (1 + 19 * rho_page)),
        "n_eff_page10": float(10 / (1 + 9 * rho_page)),
        "n_eff_page5": float(5 / (1 + 4 * rho_page)),
    }


def power_table() -> pd.DataFrame:
    """MDE for the mean daily net return, 80% power / one-sided 5%."""
    rows = []
    for k in KS:
        d = pd.read_csv(OUT / "daily_topk.csv")
        s = d[d["k"] == k]
        net = s["gross"] - 2 * COST_SIDE
        n = len(net)
        n_eff = n / 2  # T+1 open -> T+2 close overlaps two trading days
        se = float(net.std() / np.sqrt(n_eff))
        rows.append({"k": k, "n_days": n, "n_eff": n_eff, "std_daily": float(net.std()),
                     "se_mean": se, "mde_mean_80pct": 2.485 * se})
    for label, sigma, k, rho in [("broad_20pct_vol", 0.02, 30, 0.30),
                                 ("broad_30pct_vol", 0.03, 30, 0.30),
                                 ("broad_20pct_vol_k50", 0.02, 50, 0.30)]:
        sigma_p = sigma * np.sqrt((1 + (k - 1) * rho) / k)
        for n_days in (742, 250):
            n_eff = n_days / 2
            se = sigma_p / np.sqrt(n_eff)
            rows.append({"k": f"{label}_k{k}_n{n_days}", "n_days": n_days, "n_eff": n_eff,
                         "std_daily": float(sigma_p), "se_mean": float(se),
                         "mde_mean_80pct": float(2.485 * se)})
    return pd.DataFrame(rows)


def cost_drag() -> pd.DataFrame:
    rows = []
    for turn in (0.10, 0.20, 0.30, 0.50):
        for cb in (14.5, 25.0, 40.0):
            rows.append({"weekly_turnover": turn, "cost_bps_side": cb,
                         "annual_cost_pct": turn * 2 * cb / 10000 * 52 * 100})
    return pd.DataFrame(rows)


def risk_filter(b: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Exploratory: top1 account with causal battery-equal-weight risk filters.

    The panel's idx_ret is all zero, so the market proxy is the equal-weight
    return of the 101-stock panel.  Filters use T-close information only.
    """
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    ew = panel.groupby("date")["pct_chg"].mean().sort_index()
    level = (1 + ew.fillna(0)).cumprod()
    vol20 = ew.rolling(20).std()
    dd20 = level / level.rolling(20).max() - 1
    ma20 = level.rolling(20).mean()
    breadth = panel.assign(pos=panel["pct_chg"] > 0).groupby("date")["pos"].mean()
    d = topk_daily(b, 1).set_index("date")
    mid = d.index[len(d) // 2]
    filters = {
        "baseline": pd.Series(False, index=d.index),
        "breadth_lt_0.4": (breadth < 0.40).reindex(d.index).fillna(False),
        "below_ma20": (level < ma20).reindex(d.index).fillna(False),
        "dd20_lt_3pct": (dd20 < -0.03).reindex(d.index).fillna(False),
        "vol_top20_exp": (vol20 > vol20.expanding(120).quantile(0.80)).reindex(d.index).fillna(False),
    }
    rows = []
    for name, skip in filters.items():
        take = ~skip
        sel = d[take][["gross"]].copy().reset_index()
        sel["r_2d"] = sel["gross"]
        a = account_event(sel, dates, 2, 14.5)
        net = sel["gross"] - 2 * COST_SIDE
        h1 = sel[sel["date"] < mid]
        h2 = sel[sel["date"] >= mid]
        n1 = h1["gross"] - 2 * COST_SIDE
        n2 = h2["gross"] - 2 * COST_SIDE
        rows.append({
            "filter": name,
            "trigger_days": int(skip.sum()),
            "trades": a["trades"],
            "final_equity": a["final_equity"],
            "mean_net_taken": float(net.mean()) if len(net) else float("nan"),
            "h1_trades": int(len(h1)), "h1_compound": float((1 + n1).prod()) if len(n1) else float("nan"),
            "h2_trades": int(len(h2)), "h2_compound": float((1 + n2).prod()) if len(n2) else float("nan"),
        })
    return pd.DataFrame(rows)


def data_inventory() -> dict:
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    page = pd.read_parquet(PAGE_PATH)
    page["date"] = pd.to_datetime(page["date"])
    files = glob.glob(str(ROOT / "data" / "raw" / "5m" / "*.parquet"))
    bars = 0
    for f in files:
        bars += len(pd.read_parquet(f, columns=["date"]))
    pit = json.loads(PIT_STATUS.read_text(encoding="utf-8")) if PIT_STATUS.exists() else {}
    return {
        "daily_panel": {"rows": int(len(panel)), "codes": int(panel["code"].nunique()),
                        "dates": int(panel["date"].nunique()),
                        "date_min": str(panel["date"].min().date()),
                        "date_max": str(panel["date"].max().date())},
        "page": {"rows": int(len(page)), "codes": int(page["code"].nunique()),
                 "dates": int(page["date"].nunique())},
        "five_min": {"codes": len(files), "bars": int(bars),
                     "approx_gb": round(sum(Path(f).stat().st_size for f in files) / 1e9, 3)},
        "pit_status": pit,
        "broad_universe_required": {
            "codes_needed_for_300_plus": 300,
            "current_daily_codes": int(panel["code"].nunique()),
            "current_5m_codes": len(files),
            "5m_storage_at_300_codes_gb_est": round(0.092 * 300 / max(1, len(files)), 2),
            "missing": ["PIT industry membership", "exact limit/suspension",
                        "delisted names / survivorship-free panel", "broad 5m history"],
        },
    }


def production_overlay(dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Exploratory breadth/55% overlay on the 120-day production Top1 audit."""
    t = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    t["date"] = pd.to_datetime(t["date"])
    top = t[t["is_top1"]].copy()
    top = top[top["buy_block"].fillna("").eq("") & top["r_2d"].notna()].sort_values("date")
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    breadth = panel.assign(pos=panel["pct_chg"] > 0).groupby("date")["pos"].mean()
    top["breadth"] = breadth.reindex(top["date"]).to_numpy()
    top["b1"] = top["p_up"] >= 0.55
    top["bf"] = top["breadth"] >= 0.40
    rows = []
    for name, mask in [("all_top1", pd.Series(True, index=top.index)),
                       ("b1_55", top["b1"]),
                       ("breadth_ge_0.4", top["bf"]),
                       ("b1_and_breadth", top["b1"] & top["bf"]),
                       ("b1_or_breadth", top["b1"] | top["bf"])]:
        sel = top[mask][["date", "r_2d"]].copy()
        row = {"overlay": name, "trades": int(len(sel))}
        for cb in (14.5, 25.0, 40.0):
            row[f"equity_{cb:g}bps"] = account_event(sel, dates, 2, cb)["final_equity"]
        row["mean_net_14"] = float((sel["r_2d"] - 2 * COST_SIDE).mean()) if len(sel) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def breadth_sensitivity(b: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    breadth = panel.assign(pos=panel["pct_chg"] > 0).groupby("date")["pos"].mean()
    d = topk_daily(b, 1).set_index("date")
    t = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    t["date"] = pd.to_datetime(t["date"])
    top = t[t["is_top1"]].copy()
    top = top[top["buy_block"].fillna("").eq("") & top["r_2d"].notna()].sort_values("date")
    rows = []
    for th in (0.30, 0.35, 0.40, 0.45, 0.50):
        sb = d[(breadth >= th).reindex(d.index).fillna(False)][["gross"]].copy().reset_index()
        sb["r_2d"] = sb["gross"]
        st = top[(breadth >= th).reindex(top["date"]).fillna(False).to_numpy()][["date", "r_2d"]].copy()
        rows.append({
            "threshold": th,
            "b_trades": len(sb), "b_equity14": account_event(sb, dates, 2, 14.5)["final_equity"],
            "b_equity25": account_event(sb, dates, 2, 25)["final_equity"],
            "b_mean_net14": float((sb["gross"] - 2 * COST_SIDE).mean()) if len(sb) else float("nan"),
            "prod_trades": len(st), "prod_equity14": account_event(st, dates, 2, 14.5)["final_equity"],
            "prod_equity25": account_event(st, dates, 2, 25)["final_equity"],
            "prod_mean_net14": float((st["r_2d"] - 2 * COST_SIDE).mean()) if len(st) else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    b = load_b()
    dates = sorted(b["date"].unique())
    daily = []
    summary = []
    for k in KS:
        d = topk_daily(b, k)
        daily.append(d)
        summary.append({"k": k, **summarize(d, dates)})
    daily_df = pd.concat(daily, ignore_index=True)
    daily_df.to_csv(OUT / "daily_topk.csv", index=False, encoding="utf-8-sig")
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "topk_summary.csv", index=False, encoding="utf-8-sig")
    bd = breadth()
    (OUT / "breadth.json").write_text(json.dumps(bd, ensure_ascii=False, indent=2), encoding="utf-8")
    pw = power_table()
    pw.to_csv(OUT / "power.csv", index=False, encoding="utf-8-sig")
    cd = cost_drag()
    cd.to_csv(OUT / "cost_turnover.csv", index=False, encoding="utf-8-sig")
    rf = risk_filter(b, dates)
    rf.to_csv(OUT / "risk_filter.csv", index=False, encoding="utf-8-sig")
    prod = production_overlay(dates)
    prod.to_csv(OUT / "production_overlay.csv", index=False, encoding="utf-8-sig")
    sens = breadth_sensitivity(b, dates)
    sens.to_csv(OUT / "breadth_sensitivity.csv", index=False, encoding="utf-8-sig")
    inv = data_inventory()
    (OUT / "data_inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=2, default=str),
                                             encoding="utf-8")
    meta = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows_b": int(len(b)),
            "days_b": int(len(dates)), "cost_side_bps": COST_SIDE * 1e4,
            "elapsed_sec": round(time.time() - t0, 1), "training": False,
            "production_write": False}
    try:
        import psutil
        meta["rss_mb"] = round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        pass
    (OUT / "assessment.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "breadth": bd, "risk_filter": rf.to_dict("records"),
                      "inventory": inv}, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
