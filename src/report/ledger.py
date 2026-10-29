"""预测台账：冻结写入、复盘补全、陈旧数据守卫（daily / run_full 共用）。

治理规则：
- 同一 (T, track) 只保留**首次生成**的记录，后续重跑不覆盖（预测一旦生成即冻结）；
- 面板最新交易日早于今天（工作日 15:30 后）时，拒绝写入新记录，避免把陈旧数据当成新预测；
- 复盘口径：短线=次日实际上涨；长线辅助=方向命中，中性且 |涨跌|<0.5% 记 0.5。
"""
import pandas as pd

NEUTRAL_BAND = 0.005


def stale_note(t_last, now=None) -> str:
    """数据陈旧提示；空字符串表示无异常。节假日可能误报，调用方按提示重试即可。"""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    if now.weekday() >= 5:
        return ""
    if now.hour * 60 + now.minute < 15 * 60 + 30:
        return ""
    if pd.Timestamp(t_last).normalize() < now.normalize():
        return f"面板最新交易日 {pd.Timestamp(t_last):%Y-%m-%d} 早于今天 {now:%Y-%m-%d}：疑似数据未更新"
    return ""


def hit_of(track: str, direction: str, ret: float) -> float:
    if track == "short":
        return 1.0 if ret > 0 else 0.0
    if direction == "中性":
        return 0.5 if abs(ret) < NEUTRAL_BAND else 0.0
    if direction in ("看涨", "偏多"):
        return 1.0 if ret > 0 else 0.0
    if direction in ("看跌", "偏空"):
        return 1.0 if ret < 0 else 0.0
    return 0.0


def fill_reviews(old: pd.DataFrame, panel: pd.DataFrame, dates: list, t_last: str):
    """补全历史记录的 actual/hit，返回 (df, review_text, hits, total)。"""
    old = old.copy()
    mask = (old["actual"].isna() | (old["actual"].astype(str) == "")) & (old["T"].astype(str) < str(t_last))
    for i, r in old[mask].iterrows():
        after = [d for d in dates if str(d) > str(r["T"])]
        if not after:
            continue
        row = panel[(panel["code"] == r["code"]) & (panel["date"] == after[0])]
        if not len(row):
            continue
        ret = float(row["pct_chg"].iloc[0]) / 100
        old.loc[i, "actual"] = round(ret, 4)
        old.loc[i, "hit"] = hit_of(str(r.get("track", "")), str(r.get("direction", "")), ret)

    done = old[pd.to_numeric(old["hit"], errors="coerce").notna()]
    short_done = done[done["track"] == "short"]
    if len(short_done):
        hits = int((pd.to_numeric(short_done["hit"]) >= 0.5).sum())
        total = len(short_done)
        text = f"短线次日上涨命中 {hits}/{total}（{hits/total:.0%}）"
        long_done = done[done["track"] == "long_aux"]
        if len(long_done):
            lh = int((pd.to_numeric(long_done["hit"]) >= 0.5).sum())
            text += f"；长线辅助方向 {lh}/{len(long_done)}"
        return old, text, hits, total
    if len(done):
        hits = int((pd.to_numeric(done["hit"]) >= 0.5).sum())
        total = len(done)
        return old, f"已复盘 {total} 条，方向命中 {hits}（{hits/total:.0%}）", hits, total
    return old, "首日运行，无历史可复盘", 0, 0


def upsert(old: pd.DataFrame, new_rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """按 (T, track) 冻结写入：已存在的键不覆盖，返回 (df, 跳过条数)。"""
    have = set(zip(old["T"].astype(str), old["track"].astype(str)))
    keep = [(str(r["T"]), str(r["track"])) not in have for _, r in new_rows.iterrows()]
    add = new_rows[pd.Series(keep, index=new_rows.index)]
    skipped = len(new_rows) - len(add)
    if not len(add):
        return old, skipped
    return pd.concat([old, add], ignore_index=True), skipped
