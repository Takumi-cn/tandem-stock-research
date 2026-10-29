"""全行业收盘雷达：131 个 SW2 行业的 1/5/20 日强度与排名。

定位：信息层，不是交易信号；用 T 日收盘数据，不追盘中截图。
输出：reports/industry_radar/{report.md,latest.json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT  # noqa: E402

DATA = ROOT / "data" / "industry"
OUT = ROOT / "reports" / "industry_radar"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(DATA / "sw2_daily.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["code", "date"])
    d["r1"] = d.groupby("code")["close"].pct_change(1)
    d["r5"] = d.groupby("code")["close"].pct_change(5)
    d["r20"] = d.groupby("code")["close"].pct_change(20)
    last_date = d["date"].max()
    last = d[d["date"] == last_date].copy()
    # 强度：5 日为主、1 日为辅、20 日为趋势确认
    last["score"] = 0.55 * last["r5"].fillna(0) + 0.25 * last["r1"].fillna(0) + 0.20 * last["r20"].fillna(0)
    last["rank"] = last["score"].rank(ascending=False, method="min").astype(int)
    last = last.sort_values("score", ascending=False)
    cols = ["rank", "code", "name", "parent", "r1", "r5", "r20", "score", "amount"]
    top = last[cols].head(12)
    bottom = last[cols].tail(5).iloc[::-1]

    def fmt(df: pd.DataFrame) -> list[str]:
        lines = ["| # | 行业 | 一级 | 1日 | 5日 | 20日 | 强度 |", "|---:|---|---|---:|---:|---:|---:|"]
        for _, r in df.iterrows():
            lines.append(f"| {int(r['rank'])} | {r['name']} | {r['parent']} | {r['r1']:+.2%} | "
                         f"{r['r5']:+.2%} | {r['r20']:+.2%} | {r['score']:+.2%} |")
        return lines

    lines = [
        "# 全行业收盘雷达",
        "",
        f"- 数据：SW2 131 个行业，T = **{last_date.date()}** 收盘（不是盘中截图）",
        "- 强度 = 0.55×5日 + 0.25×1日 + 0.20×20日；仅信息展示，不构成交易信号",
        "",
        "## 今日强势（Top 12）",
        "",
    ] + fmt(top) + [
        "",
        "## 今日弱势（Bottom 5）",
        "",
    ] + fmt(bottom) + [
        "",
        "## 说明",
        "",
        "- 你截图里的普钢/金属新材料/电池等是**盘中**涨幅；本雷达用收盘数据，两者可能方向不同（10-27 收盘电池 -2.80%、普钢 -1.63%）。",
        "- 行业轮动 v4/v4.1/Experiment A 均已 FAIL，本雷达不改变 55% 门槛与电池主线。",
        "- 多行业个股影子见 `reports/multi_industry_shadow/`；只记录，不写 latest.json。",
        "",
        "## 免责声明",
        "",
        "研究信息展示，不构成投资建议。",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "as_of": str(last_date.date()),
        "n_industries": int(len(last)),
        "top": top.replace({np.nan: None}).to_dict(orient="records"),
        "bottom": bottom.replace({np.nan: None}).to_dict(orient="records"),
        "note": "close-based radar; not a trading signal",
    }
    (OUT / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"行业雷达 {payload['as_of']}｜强势: " + " / ".join(top["name"].head(5).astype(str)))
    print("INDUSTRY_RADAR_DONE")


if __name__ == "__main__":
    main()
