"""Plain-language interpretation of a scored observation.

Every statement follows directly from displayed data or documented model outputs.
Warning signals are described as "associated with" / "warning signal", never as
causes. Technical variable names are never surfaced; the config PLAIN dictionary
translates them. Interpretation is suppressed when data are insufficient.
"""
import numpy as np
from .config import (PLAIN, HIGHER_IS_WORSE, SEVERE_DEF, TIER_LABELS,
                     TIER_OBSERVED_RATE, INSUFFICIENT)


def pct(p):
    return "-" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{round(p * 100)}%"


def freq(p):
    return "-" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{round(p * 100)} out of 100"


# ---- warning-signal attribution (non-causal) ---------------------------------
def top_signals(row, stats, k=4):
    """Rank features by (global importance) x (how adverse this unit's value is)."""
    imp, mean, std = stats["importance"], stats["mean"], stats["std"]
    scored = []
    for f in imp:
        v = row.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)) or std.get(f, 0) in (0, None):
            continue
        z = (v - mean[f]) / std[f]
        oriented = (-z) if HIGHER_IS_WORSE.get(f, True) else z   # positive => pushes toward risk
        contrib = max(imp[f], 0) * oriented
        if contrib > 0:
            scored.append((contrib, f, v))
    scored.sort(reverse=True)
    return [f for _, f, _ in scored[:k]]


def signals_phrase(signals):
    if not signals:
        return "no single dominant signal; the classification reflects a combination of modest factors"
    phrases = [PLAIN[f] for f in signals]
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


# ---- narrative blocks --------------------------------------------------------
def result_text(row):
    return (f"Based on the latest available data ({int(row['year'])}), "
            f"{row['country']}'s {row['activity'].lower()} industry is in the "
            f"**{TIER_LABELS[row['tier']].lower()}** tier. The model estimates a "
            f"**{pct(row['risk_prob'])} probability** of a severe contraction — "
            f"{SEVERE_DEF}.")


def comparison_text(row, base_rate):
    tr = TIER_OBSERVED_RATE.get(row["tier"])
    s = (f"Across the evaluation sample, about **{freq(base_rate)}** country-industry "
         f"observations experienced such a contraction.")
    if tr is not None:
        s += (f" Among observations placed in this tier, approximately "
              f"**{freq(tr)}** subsequently experienced one.")
    s += " A probability is not a certainty."
    return s


def signals_text(row, stats):
    sig = top_signals(row, stats)
    return ("The warning is mainly associated with " + signals_phrase(sig) +
            ". These are warning signals that contributed to the risk classification, "
            "not causes of distress."), sig


def does_not_mean_text():
    return ("This does not mean employment will definitely fall. Similar patterns have "
            "sometimes been followed by a severe contraction and sometimes have not. The "
            "result indicates where closer investigation may be useful.")


def checklist(row):
    q = ["Does the recent decline reflect a temporary shock or a persistent trend?",
         "Has domestic or export demand for this industry weakened?"]
    ge1, go1 = row.get("g_emp_1"), row.get("g_out_1")
    if ge1 is not None and go1 is not None and not np.isnan(ge1) and not np.isnan(go1) and go1 > 0 and ge1 < 0:
        q.append("Output appears to be growing while employment falls — are firms adopting "
                 "less labour-intensive production?")
    else:
        q.append("Is output falling alongside employment, or are they moving apart?")
    q += ["Is one unusual reporting year driving the warning?",
          "Is the industry concentrated in a small number of firms?",
          "Have trade, energy or input-cost pressures changed recently?",
          "Do workers have realistic transition opportunities if contraction occurs?"]
    return q


def confidence_text(row):
    comp = row.get("completeness")
    n_used = int(round((comp or 0) * row.get("n_features", 16)))
    lines = [f"Latest observation year: **{int(row['year'])}**.",
             f"Data completeness: **{n_used} of {row.get('n_features', 16)}** indicators used."]
    if row.get("in_training"):
        lines.append("This country **was** represented in the model's training data.")
    else:
        lines.append("This country was **not** represented during training. Performance is "
                     "weaker for previously unseen countries, so treat this with extra caution.")
    if row.get("concept_break"):
        lines.append("Note: the employment series for this unit mixes reporting concepts in "
                     "the source; only the employees series is used here.")
    return lines


def track_record_text(perf):
    return (f"When tested on later years it had not seen, the system distinguished higher-risk "
            f"from lower-risk observations reasonably, though imperfectly "
            f"(area under the ROC curve about {perf['roc_auc']:.2f}). At the selected warning "
            f"threshold it identified about {round(perf['recall']*100)} in 100 severe "
            f"contractions, and roughly {round(perf['precision']*100)} in 100 warnings "
            f"corresponded to a contraction.")


def policymaker_summary(row, stats):
    sig = top_signals(row, stats, k=3)
    signal = (f"Employment in {row['country']}'s {row['activity'].lower()} industry shows a "
              f"{TIER_LABELS[row['tier']].split(' (')[0].lower()} risk of a severe contraction "
              f"(at least 20% fewer employees) over the next three years.")
    evidence = ("The classification reflects " + signals_phrase(sig) +
                ". Its risk is " +
                ("above" if TIER_OBSERVED_RATE.get(row["tier"], 0) > 0.11 else "near or below") +
                " the historical average for comparable observations.")
    nxt = ("Examine recent orders, exports, firm closures, technology adoption and "
           "worker-transition options before considering any response.")
    return {"signal": signal, "evidence": evidence, "next_step": nxt}


def is_sufficient(row):
    return row.get("tier") != INSUFFICIENT and row.get("risk_prob") is not None \
        and not (isinstance(row.get("risk_prob"), float) and np.isnan(row.get("risk_prob")))
