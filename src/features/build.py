"""特征与标签构建（P2）。

输入：data/raw/daily_baostock.parquet（98只×5年）+ data/raw/index_sw_battery.parquet
输出：data/parquet/panel.parquet —— 面板（特征 + 标签 + 元数据）

停牌语义（P1 质检结论）：baostock 停牌日整行缺失。
处理：按全市场日历重排 → close/high/low ffill(限5日) → volume=0、suspended=1；
滚动特征在填充后序列上计算保持连续性；训练/评估时剔除 suspended 行（不可交易）。
所有 rolling/shift 只用历史窗口（date<=T），无未来函数。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.base import feature_info  # noqa: E402
from src.utils.config import PARQUET_DIR, RAW_DIR  # noqa: E402

FFILL_LIMIT = 5  # docs/03 §3.2：停牌填充最多5日


def _load_panel() -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / "daily_baostock.parquet")
    df = df.rename(columns={"pctChg": "pct_chg"})
    for c in ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pct_chg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["isST"] = pd.to_numeric(df["isST"], errors="coerce").fillna(0).astype(int)
    # 全市场日历
    cal = np.sort(df["date"].unique())
    cal_df = pd.DataFrame({"date": cal})
    # 行业指数对齐（申万电池 801737）
    idx = pd.read_parquet(RAW_DIR / "index_sw_battery.parquet")
    idx = idx.rename(columns={"日期": "date", "收盘": "idx_close"})
    idx["idx_close"] = pd.to_numeric(idx["idx_close"], errors="coerce")
    idx["idx_ret"] = idx["idx_close"].pct_change()
    cal_df = cal_df.merge(idx[["date", "idx_close", "idx_ret"]], on="date", how="left")
    cal_df["idx_ret"] = cal_df["idx_ret"].fillna(0.0)
    # 每股重排到全日历 → 停牌填充
    out = []
    for code, g in df.groupby("code"):
        g = cal_df.merge(g.drop(columns=["code"]), on="date", how="left")
        g["code"] = code
        g["suspended"] = g["close"].isna().astype(int)
        g[["open", "high", "low", "close", "preclose"]] = g[["open", "high", "low", "close", "preclose"]].ffill(limit=FFILL_LIMIT)
        g["volume"] = g["volume"].fillna(0)
        g["amount"] = g["amount"].fillna(0)
        g["turn"] = g["turn"].fillna(0)
        g["pct_chg"] = np.where(g["suspended"] == 1, 0.0, g["pct_chg"].fillna(0.0))
        g["isST"] = g["isST"].fillna(0).astype(int)
        out.append(g)
    p = pd.concat(out, ignore_index=True)
    p = p.sort_values(["code", "date"]).reset_index(drop=True)
    return p


# ---------------- 特征族（全部 available_at=T_close） ----------------

@feature_info("T_close")
def add_momentum(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("code")["close"]
    for n in (5, 10, 20, 60):
        p[f"ret_{n}"] = g.pct_change(n)
    p["exret_5"] = p["ret_5"] - p["idx_ret"]  # 超额（当日）
    p["exret_20"] = p["ret_20"] - p.groupby("code")["idx_ret"].transform(lambda s: s.rolling(20).sum())
    return p


@feature_info("T_close")
def add_volatility(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("code")["pct_chg"]
    p["vol_20"] = g.transform(lambda s: s.rolling(20).std())
    p["vol_ratio"] = p.groupby("code")["pct_chg"].transform(lambda s: s.rolling(5).std()) / p["vol_20"]
    # ATR_14 归一
    tr = pd.concat([
        p["high"] - p["low"],
        (p["high"] - p.groupby("code")["close"].shift()).abs(),
        (p["low"] - p.groupby("code")["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    p["atr_norm"] = tr.groupby(p["code"]).transform(lambda s: s.rolling(14).mean()) / p["close"]
    return p


@feature_info("T_close")
def add_volume_price(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("code")
    p["vol_spike"] = g["volume"].transform(lambda s: s.rolling(5).mean()) / g["volume"].transform(lambda s: s.rolling(20).mean())
    p["turn_pctile"] = g["turn"].transform(lambda s: s.rolling(20).rank(pct=True))
    # 量价相关性（10日）：收益 r 与带符号对数量 v 的滚动相关。
    # 用代数式 corr=(E[rv]-E[r]E[v])/(σrσv) 全向量化实现——transform 原生保持索引对齐，
    # 避免 groupby.apply + reset_index 在 pandas3 下的索引错位（P2 泄漏审计抓到的 bug）。
    r = p["pct_chg"]
    v = np.sign(r) * np.log1p(p["volume"])
    gc = p["code"]
    mr = r.groupby(gc).transform(lambda s: s.rolling(10).mean())
    mv = v.groupby(gc).transform(lambda s: s.rolling(10).mean())
    mrv = (r * v).groupby(gc).transform(lambda s: s.rolling(10).mean())
    sr = r.groupby(gc).transform(lambda s: s.rolling(10).std())
    sv = v.groupby(gc).transform(lambda s: s.rolling(10).std())
    p["pv_corr_10"] = (mrv - mr * mv) / (sr * sv).replace(0, np.nan)
    return p


@feature_info("T_close")
def add_technical(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("code")
    # RSI_14
    delta = g["close"].diff()
    up = delta.clip(lower=0).groupby(p["code"]).transform(lambda s: s.rolling(14).mean())
    dn = (-delta.clip(upper=0)).groupby(p["code"]).transform(lambda s: s.rolling(14).mean())
    p["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    # MACD 柱归一
    ema12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    macd = ema12 - ema26
    signal = macd.groupby(p["code"]).transform(lambda s: s.ewm(span=9, adjust=False).mean())
    p["macd_hist_norm"] = (macd - signal) / p["close"]
    # BOLL 位置
    mid = g["close"].transform(lambda s: s.rolling(20).mean())
    std = g["close"].transform(lambda s: s.rolling(20).std())
    p["boll_pos"] = (p["close"] - mid) / (2 * std.replace(0, np.nan))
    # 距20日高点回撤
    p["dd_20"] = p["close"] / g["close"].transform(lambda s: s.rolling(20).max()) - 1
    # 20日新高标记
    p["new_high_20"] = (p["close"] >= g["close"].transform(lambda s: s.rolling(20).max()) - 1e-9).astype(int)
    return p


@feature_info("T_close")
def add_kline(p: pd.DataFrame) -> pd.DataFrame:
    up = (p["pct_chg"] > 0).astype(int)
    dn = (p["pct_chg"] < 0).astype(int)

    def _consec(flag: pd.Series, codes: pd.Series) -> pd.Series:
        """连续计数：以(code, 新段序号)分组累加。"""
        seg = (flag != flag.groupby(codes).shift()).groupby(codes).cumsum()
        return flag.groupby([codes, seg]).cumsum() * flag

    p["consec_up"] = _consec(up, p["code"])
    p["consec_dn"] = _consec(dn, p["code"])
    rng = (p["high"] - p["low"]).replace(0, np.nan)
    p["upper_shadow"] = (p["high"] - p[["open", "close"]].max(axis=1)) / rng
    p["lower_shadow"] = (p[["open", "close"]].min(axis=1) - p["low"]) / rng
    p["body_ratio"] = (p["close"] - p["open"]).abs() / rng
    p["gap_open"] = p["open"] / p.groupby("code")["preclose"].shift() - 1
    p["close_pos"] = (p["close"] - p["low"]) / rng
    return p


@feature_info("T_close")
def add_cross_section(p: pd.DataFrame) -> pd.DataFrame:
    """截面排名（当日全池百分位）。"""
    for col, name in (("ret_5", "xrank_ret5"), ("vol_20", "xrank_vol"), ("turn_pctile", "xrank_turn"), ("macd_hist_norm", "xrank_macd")):
        if col in p.columns:
            p[name] = p.groupby("date")[col].rank(pct=True)
    return p


FEATURE_PIPELINE = [add_momentum, add_volatility, add_volume_price, add_technical, add_kline, add_cross_section]

FEATURE_COLS = [
    "ret_5", "ret_10", "ret_20", "ret_60", "exret_5", "exret_20",
    "vol_20", "vol_ratio", "atr_norm",
    "vol_spike", "turn_pctile", "pv_corr_10",
    "rsi_14", "macd_hist_norm", "boll_pos", "dd_20", "new_high_20",
    "consec_up", "consec_dn", "upper_shadow", "lower_shadow", "body_ratio", "gap_open", "close_pos",
    "xrank_ret5", "xrank_vol", "xrank_turn", "xrank_macd",
]


# ---------------- 标签 ----------------

def add_labels(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("code")
    p["ret_1d"] = g["close"].shift(-1) / p["close"] - 1          # T+1 收益（前复权）
    p["rank_1d"] = p.groupby("date")["ret_1d"].rank(pct=True)     # 截面百分位（主标签）
    p["cls_3"] = np.select([p["ret_1d"] > 0.005, p["ret_1d"] < -0.005], [2, 0], default=1)  # 2涨1平0跌
    return p


def build() -> pd.DataFrame:
    p = _load_panel()
    for fn in FEATURE_PIPELINE:
        p = fn(p)
    p = add_labels(p)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    p.to_parquet(PARQUET_DIR / "panel.parquet")
    n_feat = len([c for c in FEATURE_COLS if c in p.columns])
    print(f"面板构建完成: {len(p):,} 行 × {p['code'].nunique()} 只 | 特征 {n_feat}/{len(FEATURE_COLS)} | 停牌行占比 {p['suspended'].mean():.1%}")
    return p


if __name__ == "__main__":
    build()
