"""Tests for the Industrial Employment Risk Monitor.

Run: python -m pytest tests/ -q   (from the repo root)
Covers feature ordering, missing-value handling, risk-tier thresholds, frozen-
score reproducibility, employees-only outcome, exclusion of nominal levels, a
known anchor prediction, the CSV export schema, and insufficient-data behaviour.
"""
import os, sys, json
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import config, features as F, model as M, interpret as I

SCORES = os.path.join(ROOT, "data", "processed", "latest_scores.csv")
MODEL = os.path.join(ROOT, "models", "model.joblib")


@pytest.fixture(scope="module")
def scored():
    return pd.read_csv(SCORES)


@pytest.fixture(scope="module")
def frozen():
    return M.load(MODEL)


# ---- feature ordering & no nominal levels ------------------------------------
def test_feature_ordering_matches_frozen_model(frozen):
    assert frozen["features"] == config.FEATURES          # exact order frozen with the model


def test_no_nominal_level_variables_in_model():
    assert set(config.FEATURES).isdisjoint(set(config.LEVELS))
    for lv in config.LEVELS:
        assert lv not in config.FEATURES


# ---- missing-value handling (never zeroed) -----------------------------------
def _toy_panel():
    rows = []
    for y in [2000, 2001, 2003]:                           # gap at 2002
        rows.append(dict(country_code="X", country="X", activity_code="10", activity="Food",
                         year=y, emp_e=100.0 if y != 2001 else 90.0, emp_pe=np.nan,
                         wages_usd=10.0, output_usd=50.0, output_unc="14",
                         va_usd=20.0, va_unc="18"))
    return pd.DataFrame(rows)


def test_missing_preserved_not_zeroed():
    f = F.add_features(_toy_panel())
    # growth across the 2002 gap (2001->2003 is 2 years, not consecutive) must be NaN, not 0
    r2003 = f[f.year == 2003].iloc[0]
    assert pd.isna(r2003["g_emp_1"])                       # 2002 missing -> 1yr growth undefined
    # a present consecutive step is a real number
    r2001 = f[f.year == 2001].iloc[0]
    assert not pd.isna(r2001["g_emp_1"]) and r2001["g_emp_1"] == pytest.approx(-0.1)


# ---- risk-tier thresholds ----------------------------------------------------
def test_tier_thresholds():
    assert M.assign_tier(0.00) == "Low"
    assert M.assign_tier(0.049) == "Low"
    assert M.assign_tier(0.05) == "Watch"
    assert M.assign_tier(0.10) == "Elevated"
    assert M.assign_tier(0.199) == "Elevated"
    assert M.assign_tier(0.20) == "High"
    assert M.assign_tier(0.95) == "High"
    assert M.assign_tier(np.nan) == config.INSUFFICIENT


# ---- employees-only outcome --------------------------------------------------
def test_label_uses_employees_only():
    # emp_e falls 30% over 3y (event); emp_pe rises (would NOT be an event) -> label must follow emp_e
    rows = [dict(country_code="Z", country="Z", activity_code="13", activity="Textiles", year=y,
                 emp_e=v, emp_pe=200.0 + y, wages_usd=1, output_usd=1, output_unc="14",
                 va_usd=1, va_unc="18")
            for y, v in [(2000, 100.0), (2003, 70.0)]]
    df = pd.DataFrame(rows)
    y = F.make_label(df, "emp_e", 3, 0.20)
    assert y.iloc[0] == 1.0
    y_pe = F.make_label(df, "emp_pe", 3, 0.20)
    assert y_pe.iloc[0] == 0.0                              # persons-engaged rose -> not an event


# ---- frozen score reproducibility --------------------------------------------
def test_scores_reproduce_from_frozen_model(scored, frozen):
    ok = scored[scored.scoreable].copy()
    X = ok[frozen["features"]].values
    p = frozen["model"].predict_proba(X)[:, 1]
    assert np.allclose(p, ok["risk_prob"].values, atol=1e-3)   # committed scores == frozen model output


def test_known_anchor_prediction(scored):
    # a committed high-risk anchor keeps its tier and (approx) probability
    row = scored[(scored.country == "Albania") & (scored.activity == "Basic metals")]
    assert len(row) == 1
    assert row.iloc[0]["tier"] == "High"
    assert row.iloc[0]["risk_prob"] == pytest.approx(0.25, abs=0.03)


# ---- CSV export schema -------------------------------------------------------
def test_csv_schema(scored):
    required = {"country_code", "country", "activity_code", "activity", "year", "risk_prob",
                "tier", "scoreable", "completeness", "in_training", "g_emp_1", "g_emp_3", "top_signals"}
    assert required.issubset(set(scored.columns))
    assert set(config.FEATURES).issubset(set(scored.columns))


# ---- insufficient-data behaviour (never zero) --------------------------------
def test_insufficient_shows_no_score(scored):
    insf = scored[~scored.scoreable]
    assert len(insf) > 0
    # insufficient rows carry no probability (NaN) and the INSUFFICIENT tier; never 0
    assert insf["risk_prob"].isna().all()
    assert (insf["tier"] == config.INSUFFICIENT).all()
    assert not (insf["risk_prob"].fillna(-1) == 0).any()


def test_interpretation_suppressed_when_insufficient(scored):
    insf = scored[~scored.scoreable].iloc[0].to_dict()
    assert I.is_sufficient(insf) is False


def test_tier_counts_match_meta(scored):
    meta = json.load(open(os.path.join(ROOT, "data", "processed", "meta.json")))
    counts = scored[scored.scoreable].tier.value_counts().to_dict()
    for t, n in meta["tier_counts"].items():
        assert counts.get(t, 0) == n
