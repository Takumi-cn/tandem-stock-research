"""Purged Walk-Forward 验证框架（docs/01 §4.1）：训练750日 + embargo 5日 + 测试20日，步长20日。"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def make_windows(dates: list, train_days: int = 750, embargo: int = 5, test_days: int = 20):
    n = len(dates)
    out = []
    start = train_days + embargo
    while start + test_days <= n:
        out.append(dict(
            train=dates[start - embargo - train_days: start - embargo],
            test=dates[start: start + test_days],
        ))
        start += test_days
    return out


def evaluate(pred_df: pd.DataFrame) -> dict:
    """pred_df: columns=[date, code, score, ret_1d]（已剔除停牌/ST/缺标签）。返回指标字典。"""
    ics, accs, top3 = [], [], []
    for d, g in pred_df.groupby("date"):
        if len(g) < 10:
            continue
        ic = spearmanr(g["score"], g["ret_1d"]).statistic
        if not np.isnan(ic):
            ics.append(ic)
        # 方向：score>截面中位 → 看涨
        pred_up = g["score"] > g["score"].median()
        actual_up = g["ret_1d"] > 0
        accs.append((pred_up == actual_up).mean())
        # Top-3 命中：预测前3均值 > 池均值
        top3.append(g.nlargest(3, "score")["ret_1d"].mean() > g["ret_1d"].mean())
    ics, accs, top3 = np.array(ics), np.array(accs), np.array(top3, dtype=float)
    return dict(
        n_days=len(ics),
        rank_ic=float(np.mean(ics)),
        icir=float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0.0,
        dir_acc=float(np.mean(accs)),
        top3_hit=float(np.mean(top3)),
    )
