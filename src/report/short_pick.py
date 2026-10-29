"""短线选股：只选"次日最可能上涨"的第一页候选。

两路模型：
- 上涨概率分类器（LightGBM 二分类，标签 ret_1d > 0）
- 次日涨幅回归器（Huber，对极端涨跌稳健）
选择规则：第一页 20 只中按 上涨概率 降序取第一，概率相同再看预期涨幅。
回测口径：最近 N 个有标签交易日，按 20 日块滚动重训，全样本外；命中 = 次日实际上涨。
"""
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

CLF_PARAMS = dict(
    n_estimators=320, learning_rate=0.04, num_leaves=31, min_child_samples=60,
    subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=8, verbose=-1,
)
REG_PARAMS = dict(
    objective="huber", alpha=0.02, n_estimators=320, learning_rate=0.04, num_leaves=31,
    min_child_samples=60, subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0,
    n_jobs=8, verbose=-1,
)
TRADE_P = 0.55  # 样本外门槛：概率达到该值才视为"可出手"


@dataclass
class UpModel:
    feats: list
    clf: lgb.LGBMClassifier | None = None
    reg: lgb.LGBMRegressor | None = None
    meta: dict = field(default_factory=dict)

    def fit(self, rows: pd.DataFrame) -> "UpModel":
        d = rows.dropna(subset=["ret_1d"])
        X = d[self.feats]
        y = (d["ret_1d"] > 0).astype(int)
        self.clf = lgb.LGBMClassifier(**CLF_PARAMS).fit(X, y)
        self.reg = lgb.LGBMRegressor(**REG_PARAMS).fit(X, d["ret_1d"].clip(-0.15, 0.15))
        self.meta = {"rows": int(len(d)), "up_base": float(y.mean())}
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.clf is None or self.reg is None:
            raise RuntimeError("UpModel 尚未训练")
        X = df[self.feats]
        return pd.DataFrame({
            "p_up": self.clf.predict_proba(X)[:, 1],
            "exp_ret": self.reg.predict(X),
        }, index=df.index)


def rank_first_page(day: pd.DataFrame, model: UpModel, feats: list, n: int = 20) -> pd.DataFrame:
    """第一页（涨幅降序前 N）按上涨概率重排。day 需含 pct_chg。"""
    fp = day.sort_values("pct_chg", ascending=False).head(n).copy()
    sc = model.predict(fp[feats])
    fp["p_up"] = sc["p_up"].values
    fp["exp_ret"] = sc["exp_ret"].values
    fp["p_up_rank"] = fp["p_up"].rank(ascending=False, method="first").astype(int)
    return fp.sort_values(["p_up", "exp_ret"], ascending=False)


def oos_backtest(trad: pd.DataFrame, feats: list, pool_n: int = 20, n_days: int = 60, block_days: int = 20) -> dict:
    """最近 n_days 个有标签交易日的样本外回测（按块重训，无未来数据）。"""
    labelled = trad.dropna(subset=["ret_1d", "rank_1d"])
    dates = sorted(labelled["date"].unique())
    if len(dates) < 120:
        return {}
    test_dates = dates[-n_days:]
    picks = []
    trad = trad.copy()
    trad["date_str"] = trad["date"].astype(str)
    for start in range(0, len(test_dates), block_days):
        block = test_dates[start:start + block_days]
        tr = labelled[labelled["date"] < block[0]]
        if len(tr) < 4000:
            continue
        model = UpModel(feats).fit(tr)
        block_set = set(str(d) for d in block)
        day_rows = trad[trad["date_str"].isin(block_set)]
        for d, g in day_rows.groupby("date"):
            if g["ret_1d"].isna().all():
                continue
            fp = g.sort_values("pct_chg", ascending=False).head(pool_n).copy()
            sc = model.predict(fp[feats])
            fp["p_up"] = sc["p_up"].values
            fp["exp_ret"] = sc["exp_ret"].values
            pick = fp.sort_values(["p_up", "exp_ret"], ascending=False).iloc[0]
            picks.append({
                "date": d,
                "ret": float(pick["ret_1d"]),
                "p_up": float(pick["p_up"]),
                "fp_avg_up": float(fp["ret_1d"].gt(0).mean()),
            })
    if not picks:
        return {}
    r = pd.DataFrame(picks)
    traded = r[r["p_up"] >= TRADE_P]
    return {
        "days": int(len(r)),
        "up_rate": round(float(r["ret"].gt(0).mean()), 4),
        "avg_ret": round(float(r["ret"].mean()), 4),
        "fp_avg_up": round(float(r["fp_avg_up"].mean()), 4),
        "gate_days": int(len(traded)),
        "gate_up_rate": round(float(traded["ret"].gt(0).mean()), 4) if len(traded) else None,
        "gate_avg_ret": round(float(traded["ret"].mean()), 4) if len(traded) else None,
        "trade_p": TRADE_P,
    }


def confidence_of(p_up: float) -> tuple[str, str]:
    """(置信档位, 行动建议)；低于出手门槛一律不出手。"""
    if p_up >= 0.62:
        return "高置信", "可参与"
    if p_up >= TRADE_P:
        return "偏多", "轻仓试多"
    if p_up >= 0.50:
        return "弱偏多", "不出手"
    return "谨慎", "不出手"


def risk_levels(close: float, exp_ret: float, atr_norm: float) -> dict:
    """基于 ATR 的交易风控参考（非收益承诺）。"""
    atr = max(float(atr_norm) if np.isfinite(atr_norm) else 0.02, 0.01)
    target = close * (1 + max(float(exp_ret), 1.0 * atr))
    stop = close * (1 - max(1.0 * atr, 0.015))
    return {
        "entry": round(close, 2),
        "target": round(float(target), 2),
        "stop": round(float(stop), 2),
        "atr_pct": round(atr * 100, 2),
    }
