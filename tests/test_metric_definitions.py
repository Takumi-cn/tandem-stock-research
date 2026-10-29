"""Metric-definition and feature-contract regression tests.

These tests guard against the exact error this project kept making:
confusing opportunity mean, per-trade mean, and account equity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.metric_reconciliation import account_event  # noqa: E402
from src.data.feature_contract import validate_features, select_features  # noqa: E402


def test_opportunity_vs_per_trade_vs_compounded():
    net = np.array([0.01, 0.0, 0.02])
    trade = np.array([True, False, True])
    opportunity = net.mean()
    per_trade = net[trade].mean()
    compounded = np.prod(1 + net[trade]) - 1
    assert abs(opportunity - 0.01) < 1e-12
    assert abs(per_trade - 0.015) < 1e-12
    assert abs(compounded - 0.0302) < 1e-12
    assert opportunity != per_trade and per_trade != compounded


def test_account_event_one_sleeve_skips_overlap_two_sleeves_does_not():
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    selected = pd.DataFrame({"date": [dates[0], dates[1]], "r_2d": [0.10, 0.10]})
    one = account_event(selected, list(dates), 1, 0.0)
    two = account_event(selected, list(dates), 2, 0.0)
    assert one["trades"] == 1 and one["skipped"] == 1
    assert two["trades"] == 2 and two["skipped"] == 0
    assert abs(one["final_equity"] - 1.10) < 1e-12
    assert abs(two["final_equity"] - 1.10) < 1e-12


def test_feature_contract_rejects_known_labels():
    df = pd.DataFrame({"ret_5": [0.1], "ret_1d": [0.2], "rank_1d": [1.0]})
    ok = validate_features(df, ["ret_5"])
    assert ok.ok
    bad = validate_features(df, ["ret_5", "ret_1d"])
    assert not bad.ok
    assert "ret_1d" in bad.forbidden_present
    assert list(select_features(df, ["ret_5"]).columns) == ["ret_5"]


def test_daily_imputation_and_prediction_guard():
    """NaN daily features must be imputed; bad predictions must fail loudly."""
    from src.eval.micro005_encoder import _prep_daily, _check_pred
    d = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 5.0]])
    tr, te = _prep_daily(d, d)
    assert np.isfinite(tr).all() and np.isfinite(te).all()
    with pytest.raises(RuntimeError):
        _check_pred(np.array([1.0, np.nan]), "nan")
    with pytest.raises(RuntimeError):
        _check_pred(np.array([1.0, 1.0]), "constant")
