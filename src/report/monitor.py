"""Forward-only monitoring: real snapshots + realized outcomes + frozen ledger.

Rules:
- Only snapshots captured on the snapshot date count as true forward evidence.
  Snapshots created later are reported separately as `late_capture` and are
  excluded from the primary statistics.
- No backtest numbers here; this file is intentionally boring until enough real
  days accumulate (target >=250 trading days before any re-evaluation).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.forward_evidence import backfill
from src.report.short_pick import TRADE_P
from src.utils.config import ROOT

OUT = ROOT / "reports" / "monitor"
EV = ROOT / "data" / "research" / "forward_evidence"
LEDGER = ROOT / "reports" / "predictions_ledger.csv"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (ctr - hw, ctr + hw)


def load_snapshots() -> pd.DataFrame:
    rows = []
    if not EV.exists():
        return pd.DataFrame(rows)
    for d in sorted(p for p in EV.iterdir() if p.is_dir()):
        man_path = d / "manifest.json"
        if not man_path.exists():
            continue
        man = json.loads(man_path.read_text(encoding="utf-8"))
        prod = man.get("production", {}) or {}
        snap_date = str(man.get("snapshot_date"))
        captured = str(man.get("captured_at", ""))
        row = {
            "snapshot_date": snap_date,
            "captured_at": captured,
            "late_capture": bool(captured[:10] and captured[:10] > snap_date),
            "n_page": man.get("n_page"),
            "pick": prod.get("code"), "p_up": prod.get("win_prob"),
            "action": prod.get("action"), "status": "PENDING_OUTCOME",
            "book_1d": np.nan, "r2d": np.nan, "r2d_net": np.nan,
            "page_mean_r2d_net": np.nan, "pick_minus_page": np.nan,
        }
        out_path = d / "outcome.json"
        if out_path.exists():
            o = json.loads(out_path.read_text(encoding="utf-8"))
            pick = o.get("pick") or {}
            row.update({
                "status": "COMPUTED",
                "book_1d": pick.get("book_1d"),
                "r2d": pick.get("r2d"),
                "r2d_net": pick.get("r2d_net"),
                "page_mean_r2d_net": o.get("page_mean_r2d_net"),
                "pick_minus_page": o.get("pick_minus_page"),
                "outcome_computed_at": o.get("computed_at"),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _stats(sub: pd.DataFrame) -> dict:
    d = sub.dropna(subset=["r2d_net"])
    if len(d) == 0:
        return {"n": 0}
    book = d["book_1d"].dropna()
    k = int((book > 0).sum()) if len(book) else 0
    lo, hi = wilson(k, len(book))
    gate = d[d["p_up"].fillna(0) >= TRADE_P] if "p_up" in d else d.iloc[0:0]
    brier = None
    both = d.dropna(subset=["p_up", "book_1d"])
    if len(both):
        brier = float(np.mean((both["p_up"] - (both["book_1d"] > 0).astype(float)) ** 2))
    return {
        "n": int(len(d)),
        "book_hit": round(k / len(book), 4) if len(book) else None,
        "book_wilson95": [round(lo, 4), round(hi, 4)] if len(book) else None,
        "net_mean": round(float(d["r2d_net"].mean()), 5),
        "net_up": round(float((d["r2d_net"] > 0).mean()), 4),
        "page_mean": round(float(d["page_mean_r2d_net"].mean()), 5),
        "pick_minus_page": round(float(d["pick_minus_page"].mean()), 5),
        "gate_n": int(len(gate)),
        "gate_net_mean": round(float(gate["r2d_net"].mean()), 5) if len(gate) else None,
        "brier": round(brier, 4) if brier is not None else None,
    }


def ledger_summary() -> dict:
    if not LEDGER.exists():
        return {"rows": 0}
    d = pd.read_csv(LEDGER, dtype={"code": str})
    done = d[pd.to_numeric(d["hit"], errors="coerce").notna()]
    short = done[done["track"] == "short"]
    return {
        "rows": int(len(d)), "completed": int(len(done)),
        "short_completed": int(len(short)),
        "short_hits": int((pd.to_numeric(short["hit"], errors="coerce") >= 0.5).sum()) if len(short) else 0,
        "dates": int(d["T"].nunique()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bf = backfill()
    df = load_snapshots()
    if len(df):
        df.to_csv(OUT / "history.csv", index=False, encoding="utf-8-sig")
    true_f = df[~df["late_capture"]] if len(df) else df
    late_f = df[df["late_capture"]] if len(df) else df
    st = _stats(true_f)
    st_late = _stats(late_f)
    led = ledger_summary()

    lines = [
        "# 前瞻监控（Forward-only）",
        "",
        f"- 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}｜数据截至 panel 最新日",
        f"- 快照：{len(df)} 份（真前瞻 {len(true_f)} / 迟到 {len(late_f)}）；已实现 outcome：{int((df['status'] == 'COMPUTED').sum()) if len(df) else 0}",
        "- 迟到快照 = `captured_at` 日期晚于 `snapshot_date`，不计入主统计（不是真正的 T 日盘后证据）。",
        "",
        "## 真前瞻统计（T 日收盘后捕获）",
        "",
    ]
    if st["n"] == 0:
        lines.append("- 尚无已实现的可交易 outcome；等待 T+1/T+2 数据。样本 <20 时不做任何优势结论。")
    else:
        lines += [
            f"- 已实现 {st['n']} 天：账面胜率 {st['book_hit']:.1%}（Wilson95 {st['book_wilson95'][0]:.1%}–{st['book_wilson95'][1]:.1%}）",
            f"- 可实现 2 日净收益均值 {st['net_mean']:+.3%}，胜率 {st['net_up']:.1%}；第一页等权 {st['page_mean']:+.3%}；Top1−page {st['pick_minus_page']:+.3%}",
            f"- 55% 门槛档：{st['gate_n']} 天，净收益均值 {st['gate_net_mean']:+.3%}" if st["gate_n"] else "- 55% 门槛档：0 天",
            f"- Brier（pick p_up vs 账面上涨）：{st['brier']}" if st["brier"] is not None else "- Brier：样本不足",
            "- 样本仍小：这些数字只用于积累，不用于改门槛/仓位。",
        ]
    if st_late["n"]:
        lines += ["", "## 迟到快照（不计入主统计）", "",
                  f"- {st_late['n']} 天：账面胜率 {st_late['book_hit']:.1%}、可实现净收益 {st_late['net_mean']:+.3%}"]

    lines += [
        "",
        "## 快照明细",
        "",
        "| 日期 | 捕获时间 | 迟到 | 首选 | p_up | 动作 | 状态 | 账面 | 2日净 | 第一页净 | Top1−page |",
        "|---|---|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows() if len(df) else []:
        def f(v, pct=False):
            if pd.isna(v):
                return "—"
            return f"{v:+.2%}" if pct else f"{v:.4f}"
        lines.append(
            f"| {r['snapshot_date']} | {r['captured_at']} | {'是' if r['late_capture'] else '否'} | {r['pick']} | "
            f"{'—' if pd.isna(r['p_up']) else f'{r['p_up']:.1%}'} | {r['action']} | {r['status']} | "
            f"{f(r['book_1d'], True)} | {f(r['r2d_net'], True)} | {f(r['page_mean_r2d_net'], True)} | {f(r['pick_minus_page'], True)} |"
        )
    if len(df) == 0:
        lines.append("| — | — | — | — | — | — | 无快照 | — | — | — | — |")

    lines += [
        "",
        "## 台账（冻结写入）",
        "",
        f"- 总记录 {led.get('rows', 0)} 条，覆盖 {led.get('dates', 0)} 个交易日；已复盘 {led.get('completed', 0)} 条。",
        f"- 短线已复盘 {led.get('short_completed', 0)} 条，命中 {led.get('short_hits', 0)} 条。",
        "- 台账只保留 (T, track) 首条预测；同日重跑不覆盖。",
        "",
        "## 判读与治理",
        "",
        "- 本监控只读真实前瞻证据与冻结台账，不做回测、不合成历史。",
        "- 目标：累计 ≥250 个交易日后再评估是否稳定 >53%；在此之前不改 55% 门槛、不加仓、不替换电池主线。",
        "- 若 `PENDING_OUTCOME` 持续超过 3 个交易日，检查 panel 是否更新或个股是否停牌。",
        "", "## 免责声明", "", "研究监控输出，不构成投资建议。",
    ]
    (OUT / "latest.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {"snapshots": len(df), "true_forward": len(true_f), "late": len(late_f),
               "computed": int((df["status"] == "COMPUTED").sum()) if len(df) else 0,
               "true_forward_stats": st, "ledger": led, "backfill": bf}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"snapshots": summary["snapshots"], "computed": summary["computed"],
                      "stats": st, "ledger": led}, ensure_ascii=False, indent=1, default=float))
    print("MONITOR_DONE")


if __name__ == "__main__":
    main()
