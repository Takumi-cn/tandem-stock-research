"""CALIB-002：滚动 Platt/Isotonic 校准（冻结模型，只加校准层）。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.top1_audit import boot_by_date  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

OUT = ROOT / "reports" / "calib_v31"
COST = 0.00145
BLOCK = 20


def fit_platt(p, y):
    m = LogisticRegression(max_iter=1000)
    m.fit(np.asarray(p).reshape(-1, 1), np.asarray(y).astype(int))
    return lambda z: m.predict_proba(np.asarray(z).reshape(-1, 1))[:, 1]


def fit_iso(p, y):
    m = IsotonicRegression(out_of_bounds="clip")
    m.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
    return lambda z: m.predict(np.asarray(z, dtype=float))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = json.loads((OUT / "prereg.json").read_text(encoding="utf-8"))
    assert prereg["status"] == "PRE-REGISTERED"
    t = pd.read_csv(ROOT / "reports" / "top1_audit" / "trades.csv", dtype={"code": str})
    t["date"] = pd.to_datetime(t["date"])
    t["buy_block"] = t["buy_block"].fillna("")
    dates = sorted(t["date"].unique())
    test_dates = dates[-120:]

    rows, brier_parts = [], []
    for start in range(0, len(test_dates), BLOCK):
        block = test_dates[start:start + BLOCK]
        tr = t[t["date"] < block[0]].dropna(subset=["p_up", "book"])
        if len(tr) < 200:
            continue
        platt = fit_platt(tr["p_up"].to_numpy(), (tr["book"] > 0).to_numpy())
        iso = fit_iso(tr["p_up"].to_numpy(), (tr["book"] > 0).to_numpy())
        te = t[t["date"].isin(block)].copy()
        te["p_platt"] = platt(te["p_up"].to_numpy())
        te["p_iso"] = iso(te["p_up"].to_numpy())
        brier_parts.append(te[["book", "p_up", "p_platt", "p_iso"]].copy())
        for d, g in te.groupby("date"):
            base = g[g["is_top1"]]
            if base.empty:
                continue
            b = base.iloc[0]
            best_pl = g.loc[g["p_platt"].idxmax()]
            best_iso = g.loc[g["p_iso"].idxmax()]
            rows.append({
                "date": d,
                "raw_code": str(b["code"]), "raw_p": float(b["p_up"]),
                "raw_r2d": b["r_2d"], "raw_net": (b["r_2d"] - 2 * COST) if pd.notna(b["r_2d"]) else 0.0,
                "pl_code": str(best_pl["code"]), "pl_p": float(best_pl["p_platt"]),
                "pl_r2d": best_pl["r_2d"], "pl_net": (best_pl["r_2d"] - 2 * COST) if pd.notna(best_pl["r_2d"]) else 0.0,
                "iso_code": str(best_iso["code"]), "iso_p": float(best_iso["p_iso"]),
                "iso_r2d": best_iso["r_2d"], "iso_net": (best_iso["r_2d"] - 2 * COST) if pd.notna(best_iso["r_2d"]) else 0.0,
                "raw_book": b["book"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "days.csv", index=False, encoding="utf-8-sig")

    # 全 page Brier（只用测试块，按块前校准；不拿 Top1 近似）
    bp = pd.concat(brier_parts, ignore_index=True)
    yb = (bp["book"] > 0).astype(float)
    brier = {
        "raw": round(float(np.mean((bp["p_up"] - yb) ** 2)), 4),
        "platt": round(float(np.mean((bp["p_platt"] - yb) ** 2)), 4),
        "iso": round(float(np.mean((bp["p_iso"] - yb) ** 2)), 4),
    }

    def block(tag):
        net = df[f"{tag}_net"]
        r = net[df[f"{tag}_r2d"].notna()]
        ud = sorted(df["date"].unique())
        half = set(ud[: len(ud) // 2])
        pcol = "raw_p" if tag == "raw" else f"{tag}_p"
        gate = df[pcol] >= 0.55
        gnet = net.where(gate, 0.0)
        gr = gnet[gate & df[f"{tag}_r2d"].notna()]
        return {"trades": int(len(r)), "opp": round(float(net.mean()), 5),
                "per": round(float(r.mean()), 5) if len(r) else None,
                "up": round(float((r > 0).mean()), 4) if len(r) else None,
                "first": round(float(net[df["date"].isin(half)].mean()), 5),
                "second": round(float(net[~df["date"].isin(half)].mean()), 5),
                "gate_trades": int(len(gr)), "gate_opp": round(float(gnet.mean()), 5),
                "gate_per": round(float(gr.mean()), 5) if len(gr) else None,
                "gate_first": round(float(gnet[df["date"].isin(half)].mean()), 5),
                "gate_second": round(float(gnet[~df["date"].isin(half)].mean()), 5),
                "boot": boot_by_date(df.assign(v=net), "v")}

    res = {"raw": block("raw"), "platt": block("pl"), "iso": block("iso"), "brier": brier}
    g_raw = df["raw_net"].where(df["raw_p"] >= 0.55, 0.0)
    g_pl = df["pl_net"].where(df["pl_p"] >= 0.55, 0.0)
    res["paired"] = boot_by_date(df.assign(v=g_pl - g_raw), "v")
    checks = {
        "Brier 改善": bool(brier["platt"] < brier["raw"]),
        "Platt gate 机会净收益 >= raw gate": bool(res["platt"]["gate_opp"] >= res["raw"]["gate_opp"]),
        "Platt gate 每笔>0": bool((res["platt"]["gate_per"] or -1) > 0),
        "Platt gate 前后半段同号": bool((res["platt"]["gate_first"] > 0) == (res["platt"]["gate_second"] > 0)),
        "Platt gate 交易 >=20 天": bool(res["platt"]["gate_trades"] >= 20),
    }
    passed = all(checks.values())
    lines = [
        "# CALIB-002：滚动 Platt/Isotonic 概率校准（预注册）",
        "",
        f"- {pd.Timestamp.now():%Y-%m-%d %H:%M}｜{len(df)} 个测试日｜块前拟合，测试块不参与",
        "- 冻结生产模型；只加校准层；T+1 open→T+2 close，14.5bps",
        "",
        "| 规则 | Top1 天数 | Top1 日均 | 55%档天数 | 55%档日均 | 55%档每笔 | 前半段 gate | 后半段 gate | Brier(全page) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| 原始 p | {res['raw']['trades']} | {res['raw']['opp']:+.3%} | {res['raw']['gate_trades']} | {res['raw']['gate_opp']:+.3%} | {(res['raw']['gate_per'] or 0):+.3%} | {res['raw']['gate_first']:+.3%} | {res['raw']['gate_second']:+.3%} | {brier['raw']} |",
        f"| Platt | {res['platt']['trades']} | {res['platt']['opp']:+.3%} | {res['platt']['gate_trades']} | {res['platt']['gate_opp']:+.3%} | {(res['platt']['gate_per'] or 0):+.3%} | {res['platt']['gate_first']:+.3%} | {res['platt']['gate_second']:+.3%} | {brier['platt']} |",
        f"| Isotonic | {res['iso']['trades']} | {res['iso']['opp']:+.3%} | {res['iso']['gate_trades']} | {res['iso']['gate_opp']:+.3%} | {(res['iso']['gate_per'] or 0):+.3%} | {res['iso']['gate_first']:+.3%} | {res['iso']['gate_second']:+.3%} | {brier['iso']} |",
        "",
        f"- Platt−raw（仅在 55% 档）配对：{res['paired']['mean']:+.3%}，95%CI [{res['paired']['ci95'][0]:+.3%},{res['paired']['ci95'][1]:+.3%}]，p(≤0)={res['paired']['p_le_0']}",
        "",
        "## 停止条件",
    ] + [f"- {'✅' if v else '❌'} {k}" for k, v in checks.items()] + [
        "", f"**判定：{'PASS（只进前瞻影子）' if passed else 'CALIBRATION_FAIL'}**", "",
        "## 局限", "",
        "- 120 日、每块前约 1–2 千行；校准器本身也会过拟合；",
        "- Brier 用 Top1 的 p 近似全 page；报告不据此改门槛；",
        "- 若校准后 p 变小导致交易减少，这是预期，不调阈值救结果。",
        "", "## 免责声明", "", "研究实验输出，不构成投资建议。",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    out = {"exp_id": "CALIB-002", "res": res, "checks": checks, "passed": passed}
    (OUT / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    with open(ROOT / "reports" / "experiment_registry.csv", "a", encoding="utf-8") as f:
        f.write(f"CALIB-002,{pd.Timestamp.now():%Y-%m-%d},Top1概率校准,SHADOW-COMPLETE,"
                f"{'PASS' if passed else 'CALIBRATION_FAIL'};brier_raw={brier['raw']};brier_platt={brier['platt']};"
                f"raw_opp={res['raw']['opp']};pl_opp={res['platt']['opp']}\n")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print("CALIB_DONE")


if __name__ == "__main__":
    main()
