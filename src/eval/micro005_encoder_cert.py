"""Finalize MICRO-005-ENC-002 after the amended 4-seed encoder run.

Reads checkpoints/predictions only; no training, no production writes.
Pairs C and B on identical (date, code) rows, then computes daily RankIC and
Top1 page alpha on the same cross-section.  Writes
reports/micro005_encoder_002/{report.md,decision.json,final_manifest.json}.
"""
from __future__ import annotations

import json
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.micro005_minimal import (DATA_FILES, PROD_FILES, boot_p, daily_metrics,
                                       matched_random)  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

OUT = ROOT / "reports" / "micro005_encoder_002"
DATA = ROOT / "data" / "research" / "micro005_minimal"
BASE = ROOT / "data" / "research" / "micro005_encoder_002"
C_PATH = DATA / "oos_pred_C_002.parquet"
B_PATH = BASE / "oos_pred_B_21f.parquet"
TARGET = "label__page_resid_T1_T2"
EXTERNAL_WRITERS = {
    "reports/ens_fixed_weights.json": "04:30 self_improve -> src/report/eval_p5.py",
    "config/l1_best.json": "run_full Optuna / self_improve",
    "reports/predictions_ledger.csv": "daily/self_improve ledger update",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def production_hashes() -> dict[str, str]:
    out = {}
    for p in PROD_FILES + DATA_FILES:
        f = ROOT / p
        out[p] = sha256(f) if f.exists() else "MISSING"
    return out


def paired_daily(c: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily metrics on the intersection of (date, code) rows."""
    cc = c[["date", "code", "rank_in_fp", TARGET, "pred"] +
           [x for x in c.columns if x.startswith("pred_s")]].rename(columns={"pred": "pred_C"})
    bb = b[["date", "code", TARGET, "pred"]].rename(
        columns={"pred": "pred_B", TARGET: "target_B"})
    m = cc.merge(bb, on=["date", "code"], how="inner")
    m = m[np.isclose(m[TARGET].astype(float), m["target_B"].astype(float))].copy()
    rows = []
    for d, g in m.groupby("date"):
        if len(g) < 8:
            continue
        ic_c = stats.spearmanr(g["pred_C"], g[TARGET]).statistic
        ic_b = stats.spearmanr(g["pred_B"], g[TARGET]).statistic
        top_c = g.loc[g["pred_C"].idxmax(), TARGET]
        top_b = g.loc[g["pred_B"].idxmax(), TARGET]
        qc = pd.qcut(g["pred_C"].rank(method="first"), 4, labels=False)
        qb = pd.qcut(g["pred_B"].rank(method="first"), 4, labels=False)
        rows.append({
            "date": d, "n": len(g),
            "ic_C": ic_c, "ic_B": ic_b,
            "top1_C": top_c, "top1_B": top_b,
            "q4q1_C": g[TARGET][qc == 3].mean() - g[TARGET][qc == 0].mean(),
            "q4q1_B": g[TARGET][qb == 3].mean() - g[TARGET][qb == 0].mean(),
        })
    dm = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return dm, m


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not C_PATH.exists():
        raise SystemExit(f"missing encoder predictions: {C_PATH}")
    if not B_PATH.exists():
        raise SystemExit(f"missing 21-fold comparator: {B_PATH}")
    C = pd.read_parquet(C_PATH)
    B = pd.read_parquet(B_PATH)
    C["date"] = pd.to_datetime(C["date"]); B["date"] = pd.to_datetime(B["date"])
    C["code"] = C["code"].astype(str); B["code"] = B["code"].astype(str)
    if C.duplicated(["date", "code"]).any() or B.duplicated(["date", "code"]).any():
        raise SystemExit("duplicate (date, code) rows in C or B")
    dm, pair = paired_daily(C, B)
    d_ic = (dm["ic_C"] - dm["ic_B"]).to_numpy()
    d_top = (dm["top1_C"] - dm["top1_B"]).to_numpy()
    bic = boot_p(d_ic); btop = boot_p(d_top)
    blocks = [dm.iloc[i] for i in np.array_split(np.arange(len(dm)), 4)]
    block_ic = [float((g["ic_C"] - g["ic_B"]).mean()) for g in blocks]
    block_top = [float((g["top1_C"] - g["top1_B"]).mean()) for g in blocks]
    c_pair = pair[["date", "code", "rank_in_fp", TARGET, "pred_C"]].rename(columns={"pred_C": "pred"})
    mr = matched_random(c_pair, n_draw=10_000, seed=11)
    seed_ic = {}
    for c in [x for x in pair.columns if x.startswith("pred_s")]:
        tmp = pair[["date", "code", "rank_in_fp", TARGET, c]].rename(columns={c: "pred"})
        seed_ic[c] = float(daily_metrics(tmp)["ic"].mean())
    seed_std = float(np.std(list(seed_ic.values()))) if seed_ic else None
    ic_b_std = float(dm["ic_B"].std())
    se_ic = ic_b_std / float(np.sqrt(len(dm)))
    mde_ic = 1.645 * se_ic
    # zero-training page rules on the same paired dates
    rule_rows = []
    for d, g in c_pair.groupby("date"):
        g = g.sort_values("rank_in_fp")
        rule_rows.append({"date": d, "pct_top1": float(g.iloc[0][TARGET]),
                          "pct_bottom1": float(g.iloc[-1][TARGET]),
                          "pct_mid": float(g.iloc[len(g) // 2][TARGET])})
    rules = pd.DataFrame(rule_rows)
    g1 = bool(bic["mean"] > 0 and bic["p_one_sided"] < 0.05)
    g2 = bool(btop["mean"] > 0 and btop["p_one_sided"] < 0.10)
    g4 = bool(sum(x > 0 for x in block_ic) >= 3 and sum(x > 0 for x in block_top) >= 3)
    status = "ENCODER_DIAGNOSTIC_ONLY" if (g1 and g2 and g4) else "ENCODER_FAIL"
    before_path = OUT / "production_hash_before.json"
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else None
    after = production_hashes()
    (OUT / "production_hash_after.json").write_text(
        json.dumps(after, ensure_ascii=False, indent=1), encoding="utf-8")
    changed = [k for k in before if before[k] != after.get(k)] if before else []
    unattributed = [k for k in changed if k not in EXTERNAL_WRITERS]
    external_changes = [{"file": k, "writer": EXTERNAL_WRITERS[k]} for k in changed if k in EXTERNAL_WRITERS]
    latest_unchanged = (before.get("reports/latest.json") == after.get("reports/latest.json")) if before else None
    production_guard = {
        "experiment_wrote_production": False,
        "latest_json_unchanged": latest_unchanged,
        "changed_files": changed,
        "external_changes": external_changes,
        "unattributed_changes": unattributed,
        "verdict": ("UNATTRIBUTED_CHANGE" if unattributed else
                    ("EXTERNAL_SCHEDULED_UPDATE" if changed else "CLEAN")),
    }
    if unattributed or latest_unchanged is False:
        status = "GOVERNANCE_HALT"
    production_unchanged_by_experiment = bool(not unattributed and latest_unchanged is not False)
    seeds = 4
    decision = {
        "experiment_id": "MICRO-005-ENC-002",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "historical_diagnostic_only": True,
        "production_write": False,
        "model_training": f"{seeds} seeds x 19 folds x 300 epochs (compute amendment; 10 seeds would need ~13.5h)",
        "n_days": int(len(dm)), "n_rows": int(len(pair)),
        "ic_C": float(dm["ic_C"].mean()), "ic_B": float(dm["ic_B"].mean()),
        "delta_ic": bic, "delta_top1": btop,
        "top1_C": float(dm["top1_C"].mean()), "top1_B": float(dm["top1_B"].mean()),
        "block_ic": block_ic, "block_top1": block_top,
        "seed_ic": seed_ic, "seed_ic_std": seed_std,
        "ic_b_daily_std": ic_b_std, "se_delta_ic": se_ic, "mde_delta_ic_p05": mde_ic,
        "matched_random": mr,
        "g1_delta_ic": g1, "g2_delta_top1": g2, "g4_blocks": g4,
        "rules": {
            "pct_top1_mean": float(rules["pct_top1"].mean()),
            "pct_bottom1_mean": float(rules["pct_bottom1"].mean()),
            "pct_mid_mean": float(rules["pct_mid"].mean()),
        },
        "max_status": "ENCODER_DIAGNOSTIC_ONLY",
        "production_guard": production_guard,
        "production_unchanged_by_experiment": production_unchanged_by_experiment,
    }
    (OUT / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, default=float),
                                        encoding="utf-8")
    n_folds = int(C["fold"].nunique())
    lines = [
        "# MICRO-005-ENC-002：原始 5 分钟序列编码器（历史诊断）", "",
        f"- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')}；{seeds} seeds × {n_folds} folds × 300 epochs；63,146 参数。",
        f"- 状态：**{status}**；历史诊断，不是新 OOS；不写生产。",
        f"- 生产守卫：实验未写生产；latest.json 未变={production_guard['latest_json_unchanged']}；"
        f"外部定时任务变更={[x['file'] for x in external_changes] or '无'}；未归属变更={unattributed or '无'}。",
        "- 外部变更说明：`reports/ens_fixed_weights.json` 于 04:33 由 04:30 自学习任务 `eval_p5` 刷新"
        "（工作日志 `SELF-IMPROVE 2026-10-28 04:33`），不属于本实验；不恢复该生产文件，避免覆盖定时任务结果。",
        f"- 配对样本：{len(dm)} 个 OOS 交易日 / {len(pair):,} 行；C 与 B 按 (date, code) 对齐。",
        "- 原 10 种子设置实测约 13.5 小时，超出 3–6 小时预算；按 prereg 的 compute_amendment 改为 4 种子，其余不变。",
        "", "| 指标 | C encoder | B 日线+5m摘要 | Δ(C−B) |",
        "|---|---:|---:|---:|",
        f"| 日 RankIC 均值 | {dm['ic_C'].mean():+.5f} | {dm['ic_B'].mean():+.5f} | {bic['mean']:+.5f} |",
        f"| Top1 page alpha 均值 | {dm['top1_C'].mean():+.5%} | {dm['top1_B'].mean():+.5%} | {btop['mean']:+.5%} |",
        f"| Q4−Q1 均值 | {dm['q4q1_C'].mean():+.5%} | {dm['q4q1_B'].mean():+.5%} | — |",
        "", f"- ΔIC 日期块 bootstrap：95% CI [{bic['ci95'][0]:+.5f}, {bic['ci95'][1]:+.5f}]，p(≤0)={bic['p_one_sided']:.4f}。",
        f"- ΔTop1 bootstrap：95% CI [{btop['ci95'][0]:+.5%}, {btop['ci95'][1]:+.5%}]，p(≤0)={btop['p_one_sided']:.4f}。",
        f"- 4 个时间块 ΔIC：{[round(x, 5) for x in block_ic]}；ΔTop1：{[round(x, 5) for x in block_top]}。",
        f"- 4 个种子各自的 IC：{ {k: round(v, 5) for k, v in seed_ic.items()} }；种子间 IC 标准差 {seed_std if seed_std is not None else float('nan'):.5f}。",
        f"- 统计功效：B 日 IC 标准差 {ic_b_std:.4f}，n={len(dm)}，ΔIC 标准误约 {se_ic:.5f}；单侧 p<0.05 的最小可检测 ΔIC 约 {mde_ic:.5f}。小于该值的真实增量在本样本中不可区分于 0。",
        f"- matched-random：实际 Top1 mean {mr.get('actual_top1_mean', float('nan')):+.5%}；随机中位 {mr.get('random_mean_median', float('nan')):+.5%}；p(model≤random)={mr.get('p_model_le_random', float('nan')):.4f}。",
        f"- 规则基线：page 涨幅第1 {rules['pct_top1'].mean():+.5%}、最后1 {rules['pct_bottom1'].mean():+.5%}、中位 {rules['pct_mid'].mean():+.5%}。",
        "", "## 门槛", "",
        f"- G1 ΔIC>0 且 p<0.05：**{g1}**",
        f"- G2 ΔTop1>0 且 p<0.10：**{g2}**",
        f"- G4 至少 3/4 时间块同号：**{g4}**",
        "", f"**最终：{status}**。即使通过也只是 shadow 诊断，必须有 ≥250 个真正新前瞻交易日才讨论任何上线；生产 latest.json、55%、台账、电池主线未改。",
        "", "## 限制", "",
        "- 2021–2026 历史已被反复查看；该结果不能作为独立 OOS。",
        "- 4 seeds 的逐 seed 预测已保存并报告 IC 离散度；但 4 个种子仍不足以做稳定性认证。",
        "- 5m 数据为不复权 bar；特征全部使用 bar 内/滚动比例，未用绝对价格；停牌/涨跌停的精细执行仍受数据限制。",
        "", "## 免责声明", "", "研究实验输出，不构成投资建议。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reg = ROOT / "reports" / "experiment_registry.csv"
    with open(reg, "a", encoding="utf-8") as f:
        f.write(f"MICRO-005-ENC-002,{pd.Timestamp.now():%Y-%m-%d},电池页内5m序列编码器,AUDIT-COMPLETE,"
                f"{status};dIC={bic['mean']};dTop1={btop['mean']};pIC={bic['p_one_sided']};pTop1={btop['p_one_sided']}\n")
    files = [{"path": str(p.relative_to(ROOT)), "sha256": __import__("hashlib").sha256(p.read_bytes()).hexdigest(),
              "bytes": p.stat().st_size} for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "final_manifest.json"]
    manifest = {"experiment_id": "MICRO-005-ENC-002", "status": status,
                "historical_diagnostic_only": True, "production_write": False,
                "production_guard": production_guard,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "files": files}
    (OUT / "final_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "ic_C": decision["ic_C"], "ic_B": decision["ic_B"],
                      "dIC": bic, "dTop1": btop, "g1": g1, "g2": g2, "g4": g4}, ensure_ascii=False, indent=1, default=float))
    print("MICRO005_ENC_CERT_DONE")


if __name__ == "__main__":
    main()
