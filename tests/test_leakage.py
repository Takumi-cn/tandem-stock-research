"""无泄漏审计（docs/03 §3.2 强制）：截断重算 vs 全量取行 一致性校验。

原理：若某特征在 T 日用到 t>T 的数据，则用"截至T的截断数据"重算该行必然不同。
抽样：3只股票 × 各2个时点，校验全部 28 个特征。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.build import FEATURE_COLS, FEATURE_PIPELINE, _load_panel  # noqa: E402
from src.utils.config import PARQUET_DIR  # noqa: E402


def main():
    full = pd.read_parquet(PARQUET_DIR / "panel.parquet")
    raw = _load_panel()
    rng = np.random.default_rng(7)
    codes = rng.choice(full["code"].unique(), size=3, replace=False)
    dates = sorted(full["date"].unique())
    checks = []
    for code in codes:
        for cut in (dates[400], dates[900]):
            # 截断数据（date<=cut）重跑全流水线
            trunc_raw = raw[raw["date"] <= cut].copy()
            t = trunc_raw
            for fn in FEATURE_PIPELINE:
                t = fn(t)
            for col in FEATURE_COLS:
                if col not in t.columns:
                    continue
                a = t.loc[t["code"] == code, ["date", col]].set_index("date")[col]
                b = full.loc[(full["code"] == code) & (full["date"] == cut), col]
                if len(b) == 0 or cut not in a.index:
                    continue
                va, vb = a.loc[cut], b.iloc[0]
                ok = (pd.isna(va) and pd.isna(vb)) or (np.isclose(va, vb, equal_nan=True, atol=1e-10))
                checks.append((code, cut, col, ok))
    n_bad = sum(1 for c in checks if not c[3])
    print(f"泄漏审计: {len(checks)} 项校验（{len(codes)}股×2时点×{len(FEATURE_COLS)}特征），不一致 {n_bad} 项")
    if n_bad:
        for c in checks:
            if not c[3]:
                print("  ✗", c)
        sys.exit(1)
    print("LEAKAGE_AUDIT_OK")


if __name__ == "__main__":
    main()
