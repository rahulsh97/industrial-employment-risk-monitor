"""Frozen calibrated gradient-boosted-tree model: train, persist, predict, tier.

No model is trained inside the deployed app; the app loads a frozen artefact.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from .config import (FEATURES, HGB_PARAMS, TIER_BINS, TIER_NAMES, INSUFFICIENT,
                     REQUIRED_FEATURES, MIN_COMPLETENESS, MODEL_VERSION)


def train_calibrated(train_df, cal_df, features=FEATURES):
    """Fit HGB on train_df, isotonic-recalibrate on cal_df. Returns (calibrated, raw)."""
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(train_df[features].values, train_df.y.values)
    cal = CalibratedClassifierCV(FrozenEstimator(hgb), method="isotonic").fit(
        cal_df[features].values, cal_df.y.values)
    return cal, hgb


def train_logistic(train_df, val_df, features=FEATURES):
    """L2 logistic (median-impute + scale), C chosen on val. For model comparison."""
    best = (-1, None)
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        p = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                      ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=0))])
        p.fit(train_df[features].values, train_df.y.values)
        a = roc_auc_score(val_df.y.values, p.predict_proba(val_df[features].values)[:, 1])
        if a > best[0]:
            best = (a, p)
    return best[1]


def recent_trend_score(df):
    s = (-df["g_emp_3"]).fillna(-df["g_emp_1"]).fillna(df["n_consec_decline"] * 0.05)
    return s.fillna(0.0).values


def predict_proba(model, df, features=FEATURES):
    return model.predict_proba(df[features].values)[:, 1]


def assign_tier(prob):
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return INSUFFICIENT
    idx = np.digitize([prob], TIER_BINS[1:-1])[0]
    return TIER_NAMES[min(idx, len(TIER_NAMES) - 1)]


def completeness(df, features=FEATURES):
    return df[features].notna().mean(axis=1)


def scoreable_mask(df, features=FEATURES):
    comp = completeness(df, features)
    req = df[REQUIRED_FEATURES].notna().all(axis=1)
    return req & (comp >= MIN_COMPLETENESS)


def save(cal, path):
    joblib.dump({"model": cal, "features": FEATURES, "version": MODEL_VERSION}, path)


def load(path):
    return joblib.load(path)
