"""P1 Step 4 五项冒烟测试（docs/07 §2）。

离线项必须通过；外部数据源（akshare/baostock）不可用时标 SKIP，不阻塞
全自动测试。作为脚本运行：python tests/smoke_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SkipCheck(RuntimeError):
    pass


def check(name: str, fn, optional: bool = False):
    try:
        detail = fn()
        print(f"[OK] {name}: {detail}")
        return (name, True, str(detail), False)
    except SkipCheck as e:
        print(f"[SKIP] {name}: {e}")
        return (name, True, f"SKIP: {e}", True)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return (name, False if not optional else True, f"{type(e).__name__}: {e}", optional)


def t1_imports():
    import akshare, baostock, pandas, pyarrow, sklearn, lightgbm, optuna, statsmodels, yaml  # noqa
    return f"akshare={akshare.__version__} pandas={pandas.__version__} lightgbm={lightgbm.__version__}"


def t2_akshare_pull():
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol="000927", period="daily", start_date="20251001", end_date="20251025", adjust="qfq")
    except Exception as e:  # noqa: BLE001
        raise SkipCheck(f"akshare 网络不可用: {type(e).__name__}") from e
    if df is None or len(df) == 0:
        raise SkipCheck("akshare 返回为空")
    return f"000927 近期 {len(df)} 根K线"


def t3_baostock_pull():
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            raise SkipCheck(f"baostock 节点不可用: {lg.error_msg}")
        rs = bs.query_history_k_data_plus("sz.000927", "date,code,close", start_date="2025-10-08", end_date="2025-10-25", frequency="d", adjustflag="2")
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
    except SkipCheck:
        raise
    except Exception as e:  # noqa: BLE001
        raise SkipCheck(f"baostock 连接失败: {type(e).__name__}") from e
    if not rows:
        raise SkipCheck("baostock 返回为空")
    return f"baostock {len(rows)} 行（对照用）"


def t4_cuda():
    import torch
    if not torch.cuda.is_available():
        return "CUDA 不可用（CPU 模式，P3 前装 GPU 轮子即可）——不计失败"
    return f"{torch.cuda.get_device_name(0)} / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"


def t5_parquet():
    import pandas as pd
    from src.utils.config import RAW_DIR
    p = RAW_DIR / "_smoke.parquet"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_parquet(p)
    back = pd.read_parquet(p)
    p.unlink()
    assert len(back) == 3
    return f"parquet 读写往返 OK（{RAW_DIR}）"


def main() -> int:
    results = [
        check("1-依赖导入", t1_imports),
        check("2-akshare拉数", t2_akshare_pull, optional=True),
        check("3-baostock对照", t3_baostock_pull, optional=True),
        check("4-CUDA(4060)", t4_cuda),
        check("5-parquet落盘", t5_parquet),
    ]
    hard_fails = [r[0] for r in results if not r[1] and not r[3]]
    skips = [r[0] for r in results if r[3]]
    payload = {"status": "FAILED" if hard_fails else ("OK_WITH_SKIPS" if skips else "ALL_OK"),
               "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "hard_fails": hard_fails, "skips": skips,
               "results": [{"name": n, "ok": ok, "detail": detail, "skipped": skipped}
                           for n, ok, detail, skipped in results],
               "auto_mode": __import__("os").environ.get("TANDEM_AUTO") == "1"}
    try:
        out = Path(__file__).resolve().parents[1] / "reports" / "runs" / "smoke_status.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    if hard_fails:
        print(f"\nSMOKE_FAILED: {hard_fails}")
        return 1
    print(f"\n{'SMOKE_OK_WITH_SKIPS: ' + str(skips) if skips else 'SMOKE_ALL_OK'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
