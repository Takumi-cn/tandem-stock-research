"""P4: 情绪特征——FinBERT 模型分 + 关键词词典分 双通道（组件评估结论：FinBERT 对事实型表述有漏判）。

输入: data/raw/news.parquet（akshare stock_news_em: 标题/内容/发布时间）
输出: data/parquet/sentiment.parquet —— (code, date, sent_score, kw_pos, kw_neg, news_cnt)
      date 为新闻所属交易日（收盘前→当日；收盘后/非交易日→次一交易日）
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import PARQUET_DIR, RAW_DIR  # noqa: E402

FINBERT_PATH = r"C:\Users\81019\.cache\huggingface\hub\models--yiyanghkust--finbert-tone-chinese\snapshots\e91b1a3af10e1e8c9c03429d3cd7d5e9a1c8000d"

KW_POS = ["预增", "预盈", "扭亏", "中标", "签约", "订单", "回购", "增持", "新高", "突破", "满产", "涨价", "扩产", "获", "合作", "增长"]
KW_NEG = ["预亏", "巨亏", "亏损", "减持", "质押", "警示", "立案", "调查", "处罚", "退市", "暴跌", "下滑", "终止", "违约", "诉讼", "解禁"]


def to_trading_date(ts: pd.Series, cal: np.ndarray) -> pd.Series:
    """新闻时间戳 → 信息可得的首个交易日收盘（<=15:00 当日，否则次交易日）。"""
    cal_set = set(cal)
    cal_arr = np.array(cal)

    def map_one(t):
        d = pd.Timestamp(t)
        cut = d.replace(hour=15, minute=0, second=0, microsecond=0)
        if d <= cut and d.strftime("%Y-%m-%d") in cal_set:
            return d.strftime("%Y-%m-%d")
        nxt = cal_arr[cal_arr > d.strftime("%Y-%m-%d")]
        return nxt[0] if len(nxt) else None

    return ts.map(map_one)


def main():
    news = pd.read_parquet(RAW_DIR / "news.parquet")
    news = news.rename(columns={"ts": "ts", "title": "title", "content": "content"})
    news["ts"] = pd.to_datetime(news["ts"], errors="coerce")
    news = news.dropna(subset=["ts"])
    text = (news["title"].fillna("") + "。 " + news["content"].fillna("")).str.slice(0, 160)

    # --- 通道1: FinBERT（GPU 批量） ---
    import torch
    import torch.nn.functional as F
    from transformers import BertForSequenceClassification, BertTokenizer
    tok = BertTokenizer.from_pretrained(FINBERT_PATH)
    model = BertForSequenceClassification.from_pretrained(FINBERT_PATH).eval().cuda()
    scores = []
    B = 64
    with torch.no_grad():
        for i in range(0, len(text), B):
            enc = tok(text.iloc[i:i + B].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            p = F.softmax(model(**enc).logits, dim=-1).cpu().numpy()  # [Neutral, Positive, Negative]
            scores.append(p[:, 1] - p[:, 2])  # 情绪分 ∈ [-1,1]
            if (i // B) % 20 == 0:
                print(f"  FinBERT {i}/{len(text)}", flush=True)
    news["sent"] = np.concatenate(scores)
    del model
    torch.cuda.empty_cache()

    # --- 通道2: 关键词词典 ---
    pos_hits = text.str.count("|".join(map(re.escape, KW_POS)))
    neg_hits = text.str.count("|".join(map(re.escape, KW_NEG)))
    news["kw_pos"] = pos_hits
    news["kw_neg"] = neg_hits

    # --- 归到交易日并聚合 ---
    panel = pd.read_parquet(PARQUET_DIR / "panel.parquet")
    cal = np.sort(panel["date"].unique())
    news["date"] = to_trading_date(news["ts"], cal)
    news = news.dropna(subset=["date"])
    agg = news.groupby(["code", "date"]).agg(
        sent_mean=("sent", "mean"),
        sent_max=("sent", "max"),
        news_cnt=("sent", "size"),
        kw_pos=("kw_pos", "sum"),
        kw_neg=("kw_neg", "sum"),
    ).reset_index()
    agg.to_parquet(PARQUET_DIR / "sentiment.parquet")
    span = sorted(agg["date"].unique())
    print(f"情绪特征落盘: {len(agg):,} 行 (code×date) | 日期跨度 {span[0]} ~ {span[-1]}（{len(span)} 个交易日）", flush=True)


if __name__ == "__main__":
    main()
