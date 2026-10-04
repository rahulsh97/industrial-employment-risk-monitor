"""End-to-end frozen pipeline: build panel -> performance artefacts -> freeze
scoring model -> generate the compact scored application dataset.

Run once: python -m src.pipeline --data <raw/data.csv>
Writes (committable, derived): data/processed/latest_scores.csv,
data/processed/emp_history.csv, models/model.joblib, models/stats.json,
reports/performance.json. Raw UNIDO data is never written to the repo.
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve, precision_score, recall_score)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from . import features as F, model as M
from .config import (FEATURES, EMP_COL, HORIZON, THRESHOLD, SENSITIVITY_THRESHOLD,
                     EVAL_TRAIN, EVAL_VAL, EVAL_TEST, SCORE_TRAIN, SCORE_CAL,
                     TIER_NAMES, TIER_OBSERVED_RATE, INSUFFICIENT, MODEL_VERSION)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
yr = lambda d, lo, hi: d[(d.year >= lo) & (d.year <= hi)]


def labeled(feat, thr=THRESHOLD):
    y = F.make_label(feat, EMP_COL, HORIZON, thr)
    d = feat.copy(); d["y"] = y
    return d.dropna(subset=["y"]).reset_index(drop=True)


def _rank(y, p):
    return dict(roc_auc=round(roc_auc_score(y, p), 4), pr_auc=round(average_precision_score(y, p), 4),
                brier=round(brier_score_loss(y, np.clip(p, 0, 1)), 4))


def performance(feat):
    d = labeled(feat)
    tr, va, te = yr(d, *EVAL_TRAIN), yr(d, *EVAL_VAL), yr(d, *EVAL_TEST)
    cal, hgb = M.train_calibrated(tr, va)
    logit = M.train_logistic(tr, va)
    y = te.y.values
    pc, pl, prt = M.predict_proba(cal, te), M.predict_proba(logit, te), M.recent_trend_score(te)
    base = round(float(te.y.mean()), 4)

    # curves
    def curve_roc(p): f, t, _ = roc_curve(y, p); return {"fpr": f.round(4).tolist(), "tpr": t.round(4).tolist()}
    def curve_pr(p): pr, rc, _ = precision_recall_curve(y, p); return {"recall": rc.round(4).tolist(), "precision": pr.round(4).tolist()}
    frac, mean_pred = calibration_curve(y, pc, n_bins=8, strategy="quantile")

    # operating point at recall 0.5 (threshold from validation), applied to test
    vp = M.predict_proba(cal, va)
    pr, rc, thr = precision_recall_curve(va.y.values, vp)
    idx = np.where(rc[:-1] >= 0.5)[0]
    op_t = float(thr[idx[-1]]) if len(idx) else 0.5
    yhat = (pc >= op_t).astype(int)
    op = {"threshold": round(op_t, 3), "recall": round(recall_score(y, yhat, zero_division=0), 3),
          "precision": round(precision_score(y, yhat, zero_division=0), 3),
          "n_alerts": int(yhat.sum()), "alert_rate": round(float(yhat.mean()), 3),
          "true_warnings": int(((yhat == 1) & (y == 1)).sum()),
          "false_warnings": int(((yhat == 1) & (y == 0)).sum()), "n_test": int(len(te))}

    # tier event rates on test (confirm approved)
    tiers = {}
    for t in TIER_NAMES:
        m = np.array([M.assign_tier(p) == t for p in pc])
        tiers[t] = {"n": int(m.sum()), "event_rate": round(float(y[m].mean()), 3) if m.sum() else None}

    # grouped holdouts (corrected model, headline)
    def eval_block(trd, vad, ted):
        c, _ = M.train_calibrated(trd, vad)
        return _rank(ted.y.values, M.predict_proba(c, ted)) | {
            "recent_trend_prauc": round(average_precision_score(ted.y.values, M.recent_trend_score(ted)), 4),
            "n_test": int(len(ted)), "base_rate": round(float(ted.y.mean()), 3)}
    ctys = sorted(d.country_code.unique()); hc = set(ctys[::4])
    inds = sorted(d.activity_code.unique()); hi = set(inds[::4])
    holds = {
        "known_units": eval_block(tr, va, te),
        "new_countries": eval_block(yr(d[~d.country_code.isin(hc)], *EVAL_TRAIN), yr(d[~d.country_code.isin(hc)], *EVAL_VAL), yr(d[d.country_code.isin(hc)], *EVAL_TEST)),
        "new_industries": eval_block(yr(d[~d.activity_code.isin(hi)], *EVAL_TRAIN), yr(d[~d.activity_code.isin(hi)], *EVAL_VAL), yr(d[d.activity_code.isin(hi)], *EVAL_TEST)),
        "pre_pandemic": eval_block(yr(d, 2000, 2009), yr(d, 2010, 2012), yr(d, 2013, 2016)),
    }
    # sensitivity 10%
    d10 = labeled(feat, SENSITIVITY_THRESHOLD)
    c10, _ = M.train_calibrated(yr(d10, *EVAL_TRAIN), yr(d10, *EVAL_VAL))
    te10 = yr(d10, *EVAL_TEST)
    sens10 = _rank(te10.y.values, M.predict_proba(c10, te10)) | {"base_rate": round(float(te10.y.mean()), 3)}

    perf = {
        "version": MODEL_VERSION, "target": ">=20% employees fall over 3 years",
        "test_years": list(EVAL_TEST), "base_rate": base,
        "models": {"trees_calibrated": _rank(y, pc), "logistic": _rank(y, pl),
                   "recent_trend": _rank(y, prt)},
        "roc": {"trees": curve_roc(pc), "logistic": curve_roc(pl), "recent_trend": curve_roc(prt)},
        "pr": {"trees": curve_pr(pc), "logistic": curve_pr(pl), "recent_trend": curve_pr(prt)},
        "calibration": {"mean_pred": mean_pred.round(4).tolist(), "obs_freq": frac.round(4).tolist()},
        "tier_event_rates": tiers, "operating_point_recall50": op,
        "holdouts": holds, "sensitivity_10pct": sens10,
    }
    # interpretation stats: importance (test), training means/stds, perf summary
    imp = permutation_importance(hgb, te[FEATURES].values, y, n_repeats=8, random_state=0, scoring="average_precision")
    stats = {
        "importance": {f: round(float(v), 5) for f, v in zip(FEATURES, imp.importances_mean)},
        "mean": {f: float(np.nanmean(tr[f].values)) for f in FEATURES},
        "std": {f: float(np.nanstd(tr[f].values)) or 1.0 for f in FEATURES},
        "base_rate": base, "perf": {"roc_auc": perf["models"]["trees_calibrated"]["roc_auc"],
                                    "precision": op["precision"], "recall": op["recall"]},
    }
    return perf, stats


def freeze_scoring_model(feat):
    d = labeled(feat)
    cal, _ = M.train_calibrated(yr(d, *SCORE_TRAIN), yr(d, *SCORE_CAL))
    return cal


def pick_latest_year(feat):
    sc = M.scoreable_mask(feat)
    by = feat[sc].groupby("year").size()
    recent = by[by.index >= 2015]
    if len(recent) == 0:
        return int(feat.year.max())
    thresh = 0.4 * recent.max()
    ok = recent[recent >= thresh]
    return int(ok.index.max())


def recent_emp_trajectory(feat, emp_col=EMP_COL):
    """Rebased employment index (=100 at earliest point in a trailing window) per unit."""
    rows = []
    for (c, a), g in feat.sort_values("year").groupby(["country_code", "activity_code"]):
        g = g[g[emp_col].notna()].tail(10)
        if len(g) < 2:
            continue
        base = g[emp_col].iloc[0]
        for _, r in g.iterrows():
            rows.append({"country_code": c, "activity_code": a, "year": int(r.year),
                         "emp_index": round(float(r[emp_col] / base * 100), 1)})
    return pd.DataFrame(rows)


def score(feat, cal, stats, latest):
    from .interpret import top_signals
    d = feat[feat.year == latest].copy()
    sc = M.scoreable_mask(d)
    d["completeness"] = M.completeness(d).round(3)
    probs = np.full(len(d), np.nan)
    probs[sc.values] = M.predict_proba(cal, d[sc])
    d["risk_prob"] = probs.round(4)
    train_ccs = set(feat[(feat.year >= SCORE_TRAIN[0]) & (feat.year <= SCORE_TRAIN[1])].country_code.unique())
    # which units ever mixed employment concept (break)
    broke = set()
    for (c, a), g in feat.groupby(["country_code", "activity_code"]):
        if g["emp_e"].notna().any() and g["emp_pe"].notna().any():
            broke.add((c, a))
    out = []
    for _, r in d.iterrows():
        tier = M.assign_tier(r["risk_prob"]) if sc.loc[r.name] else INSUFFICIENT
        row = {"country_code": r.country_code, "country": r.country,
               "activity_code": r.activity_code, "activity": r.activity, "year": int(r.year),
               "risk_prob": None if np.isnan(r.risk_prob) else float(r.risk_prob),
               "tier": tier, "scoreable": bool(sc.loc[r.name]),
               "completeness": float(r.completeness), "n_features": len(FEATURES),
               "in_training": r.country_code in train_ccs,
               "concept_break": (r.country_code, r.activity_code) in broke,
               "g_emp_1": None if pd.isna(r.g_emp_1) else round(float(r.g_emp_1), 4),
               "g_emp_3": None if pd.isna(r.g_emp_3) else round(float(r.g_emp_3), 4)}
        for f in FEATURES:
            row[f] = None if pd.isna(r[f]) else round(float(r[f]), 5)
        row["top_signals"] = ";".join(top_signals(row, stats, k=4)) if tier != INSUFFICIENT else ""
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="UNIDO Rev.4 data.csv (local only)")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "data", "processed"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)

    print("building panel + features...")
    feat = F.add_features(F.build_panel(args.data), EMP_COL)
    print(f"  panel: {len(feat):,} rows | {feat.country_code.nunique()} countries | "
          f"{feat.activity_code.nunique()} industries | {feat.year.min()}-{feat.year.max()}")

    print("computing performance artefacts...")
    perf, stats = performance(feat)
    json.dump(perf, open(os.path.join(ROOT, "reports", "performance.json"), "w"), indent=2)
    json.dump(stats, open(os.path.join(ROOT, "models", "stats.json"), "w"), indent=2)

    print("freezing scoring model...")
    cal = freeze_scoring_model(feat)
    M.save(cal, os.path.join(ROOT, "models", "model.joblib"))

    latest = pick_latest_year(feat)
    print(f"scoring latest usable year: {latest}")
    scored = score(feat, cal, stats, latest)
    scored.to_csv(os.path.join(ROOT, "data", "processed", "latest_scores.csv"), index=False)
    recent_emp_trajectory(feat).to_csv(os.path.join(ROOT, "data", "processed", "emp_history.csv"), index=False)

    meta = {"version": MODEL_VERSION, "latest_year": latest,
            "n_scored": int(scored.scoreable.sum()), "n_rows": int(len(scored)),
            "n_countries": int(scored.country_code.nunique()),
            "n_industries": int(scored.activity_code.nunique()),
            "tier_counts": scored[scored.scoreable].tier.value_counts().to_dict(),
            "source": "UNIDO INDSTAT Rev.4 (licensed); derived indicators only, raw not redistributed"}
    json.dump(meta, open(os.path.join(ROOT, "data", "processed", "meta.json"), "w"), indent=2)
    print(f"  scored {meta['n_scored']} of {meta['n_rows']} units; tiers {meta['tier_counts']}")
    print("  test metrics:", perf["models"]["trees_calibrated"],
          "| op@recall0.5:", perf["operating_point_recall50"]["precision"], "prec")
    print("done.")


if __name__ == "__main__":
    main()
