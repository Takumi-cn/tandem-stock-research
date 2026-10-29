"""凌晨自学习任务（04:30 静默运行，4-8AM 为不限占用窗口；白天零后台）。

内容：
1. 台账复盘补全（fill actual/hit）
2. 刷新回测权重：eval_p3 --use-deep-cache（regmoe_weights.csv）+ eval_p5（ens_fixed_weights.json）
3. Kronos 分词器补装重试（hf 直链 curl，多镜像）
4. 工作日志留痕
不做的：不做 Optuna（在 run_full 内做并持久化 l1_best.json，本任务不重复消耗）。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import PARQUET_DIR, ROOT  # noqa: E402


def review_ledger():
    ledger = ROOT / "reports" / "predictions_ledger.csv"
    if not ledger.exists():
        return "无台账"
    p = pd.read_parquet(PARQUET_DIR / "panel.parquet")
    dates = sorted(p["date"].unique())
    old = pd.read_csv(ledger, dtype={"code": str})
    mask = (old["actual"].isna() | (old["actual"] == "")) & (old["T"] < dates[-1])
    n = 0
    for i, r in old[mask].iterrows():
        after = [d for d in dates if d > r["T"]]
        if not after:
            continue
        row = p[(p["code"] == r["code"]) & (p["date"] == after[0])]
        if len(row):
            ret = float(row["pct_chg"].iloc[0]) / 100
            old.loc[i, "actual"] = round(ret, 4)
            if r["direction"] in ("中性",) and abs(ret) < 0.005:
                old.loc[i, "hit"] = 0.5
            else:
                old.loc[i, "hit"] = 1 if (ret > 0 and r["direction"] in ("看涨", "偏多")) or (ret < 0 and r["direction"] in ("看跌", "偏空")) else 0
            n += 1
    old.to_csv(ledger, index=False, encoding="utf-8-sig")
    done = old[pd.to_numeric(old["hit"], errors="coerce").notna()]
    acc = f"{(pd.to_numeric(done['hit']) >= 0.5).mean():.0%}" if len(done) else "—"
    return f"补复盘 {n} 条；累计命中 {acc}（{len(done)} 条）"


def refresh_weights():
    outs = []
    for cmd in (["-m", "src.report.eval_p3", "--use-deep-cache"], ["-m", "src.report.eval_p5"]):
        r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=str(ROOT))
        outs.append(f"{cmd[-1]}:{'OK' if r.returncode == 0 else 'FAIL'}")
        if r.returncode != 0:
            print(r.stdout[-400:], r.stderr[-400:], flush=True)
    return " ".join(outs)


def retry_kronos_tokenizer():
    import os
    hub = Path.home() / ".cache/huggingface/hub"
    if list(hub.glob("models--NeoQuasar--Kronos-Tokenizer*/snapshots/*/*")):
        return "已存在"
    try:
        os.environ.update(HF_ENDPOINT="https://hf-mirror.com", HF_HUB_DISABLE_XET="1")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-Base")
        return "补装成功"
    except Exception as e:  # noqa: BLE001
        return f"仍失败({type(e).__name__})，明晨再试"


def main():
    t0 = time.time()
    r1 = review_ledger()
    r2 = refresh_weights()
    r3 = retry_kronos_tokenizer()
    r4 = ""
    try:  # 竞猜核验回填（T+1 数据落地后自动 ✔/× + 奖励；同日冻结，重复运行安全）
        from src.report.guess_sheet import main as _guess_main

        _guess_main()
        r4 = "竞猜记录核验完成"
    except Exception as e:  # noqa: BLE001
        r4 = f"竞猜核验跳过({type(e).__name__}: {e})"
    with open(ROOT / "工作日志.md", "a", encoding="utf-8") as f:
        f.write(f"\n### SELF-IMPROVE {pd.Timestamp.now():%Y-%m-%d %H:%M} ｜ 凌晨自学习 ｜ 耗时 {int(time.time()-t0)}s\n- {r1}\n- 权重刷新: {r2}\n- Kronos分词器: {r3}\n- {r4}\n")
    print(f"SELF_IMPROVE_DONE {int(time.time()-t0)}s | {r1} | {r2} | {r3} | {r4}")


if __name__ == "__main__":
    main()
