"""Feature contract v2 tests: namespace isolation + metadata completeness.

Run: .venv\\Scripts\\python.exe tests\\test_contract_v2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.feature_contract_v2 import (  # noqa: E402
    DAILY_ARM, FULL_ARM, INTRADAY_ARM, LABELS, FORBIDDEN, TARGET,
    metadata, select_X, validate_frame, write_contract,
)
from src.utils.config import ROOT  # noqa: E402


def frame(n=200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=n // 20).repeat(20),
                       "code": [f"{i:06d}" for i in range(20)] * (n // 20)})
    for c in set(FULL_ARM):
        df[c] = rng.normal(0, 1, len(df))
    df[TARGET] = rng.normal(0, .01, len(df))
    df["label__book_T_T1"] = rng.normal(0, .01, len(df))
    df["label__exec_T1_T2"] = rng.normal(0, .01, len(df))
    df["ret_1d"] = rng.normal(0, .01, len(df))  # legacy leak column
    return df


def main() -> None:
    df = frame()
    failures = []

    # 1. select_X returns exactly the arm, in order, never a label
    for arm, cols in (("daily", DAILY_ARM), ("daily_intraday", FULL_ARM), ("intraday", INTRADAY_ARM)):
        X = select_X(df, arm)
        if list(X.columns) != cols:
            failures.append(f"{arm}: column order/content mismatch")
        if any(c.startswith("label__") for c in X.columns):
            failures.append(f"{arm}: label column leaked into X")
        if any(c in FORBIDDEN for c in X.columns):
            failures.append(f"{arm}: forbidden legacy column in X")

    # 2. an arm containing a label must be rejected, not silently accepted
    bad = FULL_ARM + ["label__page_resid_T1_T2"]
    res = validate_frame(df, bad)
    if res["ok"]:
        failures.append("validate_frame accepted an arm containing a label")

    # 3. namespace invariants
    if not all(c.startswith("feat__") for c in FULL_ARM):
        failures.append("non feat__ column in FULL_ARM")
    overlap = set(FULL_ARM) & (set(LABELS) | set(FORBIDDEN))
    if overlap:
        failures.append(f"arms overlap labels/forbidden: {overlap}")

    # 4. metadata covers every feature with the required keys
    meta = metadata()
    need = {"available_at", "source", "lookback_start", "lookback_end", "version_hash"}
    missing = [c for c in FULL_ARM if c not in meta["features"] or not need <= set(meta["features"][c])]
    if missing:
        failures.append(f"metadata incomplete for {len(missing)} features: {missing[:3]}")
    if meta["target"] != TARGET:
        failures.append("metadata target mismatch")

    # 5. written contract is valid json and matches the module
    p = write_contract()
    written = json.loads(p.read_text(encoding="utf-8"))
    if written["version"] != meta["version"] or len(written["features"]) != len(FULL_ARM):
        failures.append("written contract diverges from module")

    # 6. the real page dataset (if present) exposes every arm column after rename
    pdp = ROOT / "data" / "research" / "micro005_minimal" / "page_dataset.parquet"
    if pdp.exists():
        page = pd.read_parquet(pdp)
        rename = {c: f"feat__{c}" for c in
                  ["ret_5", "ret_10", "ret_20", "ret_60", "exret_5", "exret_20", "vol_20", "vol_ratio",
                   "atr_norm", "vol_spike", "turn_pctile", "pv_corr_10", "rsi_14", "macd_hist_norm",
                   "boll_pos", "dd_20", "new_high_20", "consec_up", "consec_dn", "upper_shadow",
                   "lower_shadow", "body_ratio", "gap_open", "close_pos", "xrank_ret5", "xrank_vol",
                   "xrank_turn", "xrank_macd", "pct_chg"]}
        renamed = page.rename(columns=rename)
        missing_daily = [c for c in DAILY_ARM if c not in renamed.columns]
        if missing_daily:
            failures.append(f"page_dataset lacks daily arm columns: {missing_daily[:3]}")
        if "label__page_resid_T1_T2" in renamed.columns:
            ok = renamed["label__page_resid_T1_T2"].dropna().groupby(renamed["date"]).mean().abs().max()
            if ok is not None and ok > 1e-9:
                failures.append(f"page residual is not cross-sectionally zero-mean: max|mean|={ok:.2e}")

    print(f"daily_arm={len(DAILY_ARM)} intraday_arm={len(INTRADAY_ARM)} full_arm={len(FULL_ARM)}")
    if failures:
        print("CONTRACT_V2_FAIL")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print("CONTRACT_V2_PASS")


if __name__ == "__main__":
    main()
