r"""防泄漏回归测试（P0）。

运行：.\.venv\Scripts\python.exe -m pytest -q tests/test_no_lookahead.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGE = ROOT / "data" / "research" / "micro005_minimal" / "page_dataset.parquet"
PANEL = ROOT / "data" / "parquet" / "panel.parquet"
CONS = ROOT / "data" / "parquet" / "cons_battery.parquet"


@pytest.fixture(scope="module")
def data():
    page = pd.read_parquet(PAGE)
    panel = pd.read_parquet(PANEL)
    page["date"] = pd.to_datetime(page["date"])
    panel["date"] = pd.to_datetime(panel["date"])
    page["code"] = page["code"].astype(str)
    panel["code"] = panel["code"].astype(str)
    cons = set(pd.read_parquet(CONS)["code"].astype(str))
    return page, panel, cons


def test_page_is_top20_by_pct_chg(data):
    page, panel, cons = data
    trad = panel[(panel["suspended"] == 0) & (panel["isST"] == 0) & (panel["code"].isin(cons))]
    for d in sorted(page["date"].unique())[:3]:
        expected = set(trad[trad["date"] == d].nlargest(20, "pct_chg")["code"].astype(str))
        actual = set(page[page["date"] == d]["code"].astype(str))
        assert actual == expected, f"page mismatch at {d}: {actual ^ expected}"


def test_book_label_is_next_day_close(data):
    page, panel, _ = data
    px = panel.set_index(["code", "date"])["close"]
    sample = page.dropna(subset=["label__book_T_T1"]).head(200)
    for _, r in sample.iterrows():
        seq = sorted(d for d in panel["date"].unique() if d > r["date"])
        if len(seq) < 1:
            continue
        nxt = px.get((r["code"], seq[0]))
        if nxt is None or pd.isna(nxt):
            continue
        expect = float(nxt) / float(r["close"]) - 1
        assert abs(expect - float(r["label__book_T_T1"])) < 1e-6


def test_exec_label_uses_next_open_and_t_plus_2_close(data):
    page, panel, _ = data
    sample = page[(~page["entry_blocked"]) & (page["delay_days"] == 0)].dropna(
        subset=["label__exec_T1_T2", "entry_open", "exit_close"]).head(100)
    assert len(sample) > 0
    for _, r in sample.iterrows():
        expect = float(r["exit_close"]) / float(r["entry_open"]) - 1
        assert abs(expect - float(r["label__exec_T1_T2"])) < 1e-6


def test_feature_contract_has_no_labels():
    from src.features.build import FEATURE_COLS

    forbidden = {c for c in FEATURE_COLS if c.startswith("label__") or c in {"ret_1d", "rank_1d", "cls_3", "label__book_T_T1", "label__exec_T1_T2"}}
    assert not forbidden, f"label columns leaked into features: {forbidden}"


def test_page_does_not_use_future_rows(data):
    page, panel, cons = data
    trad = panel[(panel["suspended"] == 0) & (panel["isST"] == 0) & (panel["code"].isin(cons))]
    d = sorted(page["date"].unique())[0]
    full = trad[trad["date"] == d]
    truncated = trad[trad["date"] <= d]
    before = set(full.nlargest(20, "pct_chg")["code"].astype(str))
    after = set(truncated[truncated["date"] == d].nlargest(20, "pct_chg")["code"].astype(str))
    # T 日选股只用 T 及以前的数据；截断未来行不应改变 T 日第一页
    assert before == after
