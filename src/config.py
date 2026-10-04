"""Frozen configuration for the Industrial Employment Risk Monitor.

Single source of truth for the model specification approved in the feasibility
audit (corrected specification C): employees-only outcome, no nominal monetary
level variables, calibrated gradient-boosted trees, risk tiers.
"""

# ---- target (headline) -------------------------------------------------------
EMP_COL = "emp_e"          # employees only, consistent within every unit
HORIZON = 3
THRESHOLD = 0.20           # severe contraction = >=20% employment fall over 3 years
SENSITIVITY_THRESHOLD = 0.10

# ---- chronological splits (no random split) ----------------------------------
EVAL_TRAIN = (2000, 2013)  # evaluation model (Model-performance page)
EVAL_VAL = (2014, 2016)
EVAL_TEST = (2017, 2020)
SCORE_TRAIN = (2000, 2017)  # production scoring model: more data, recent calibration
SCORE_CAL = (2018, 2020)

# ---- features (specification C: NO nominal level variables) -------------------
FEATURES = [
    "g_emp_1", "g_emp_3", "vol_emp_5", "n_consec_decline", "share_decline_5", "emp_dev5",
    "g_va_1", "g_va_3", "g_out_1", "g_out_3", "g_wages_1", "g_prod_1", "wage_share",
    "g_emp_1_rel_ctry", "g_emp_1_rel_ind", "log_emp_share_ctry",
]
# nominal level variables, deliberately EXCLUDED from the model (audit only)
LEVELS = ["log_prod_va", "log_prod_out", "wage_pe", "log_va", "log_emp"]

# features that must be present for a usable score (key employment dynamics)
REQUIRED_FEATURES = ["g_emp_1", "g_emp_3", "share_decline_5", "wage_share"]
MIN_COMPLETENESS = 0.5     # >= 8 of 16 features present

# ---- risk tiers (calibrated probability bins) + approved historical rates -----
TIER_BINS = [0.0, 0.05, 0.10, 0.20, 1.01]
TIER_NAMES = ["Low", "Watch", "Elevated", "High"]
TIER_LABELS = {
    "Low": "Low (under 5%)",
    "Watch": "Watch (5-10%)",
    "Elevated": "Elevated (10-20%)",
    "High": "High (20% or more)",
}
# observed historical contraction rate within each tier (from the approved test)
TIER_OBSERVED_RATE = {"Low": 0.052, "Watch": 0.095, "Elevated": 0.188, "High": 0.277}
TIER_COLORS = {
    "Low": "#2f8f6b", "Watch": "#e3b23c", "Elevated": "#d98032",
    "High": "#b4452f", "Insufficient data": "#9aa5a1",
}
INSUFFICIENT = "Insufficient data"

# ---- model hyperparameters (frozen) ------------------------------------------
HGB_PARAMS = dict(max_depth=4, learning_rate=0.06, max_iter=400,
                  l2_regularization=1.0, class_weight="balanced", random_state=0)
MODEL_VERSION = "ierm-1.0.0"

# ---- plain-language dictionary (NEVER show raw variable names to users) -------
PLAIN = {
    "g_emp_1": "employment change over the past year",
    "g_emp_3": "employment change over the past three years",
    "vol_emp_5": "how much recent employment growth has swung year to year",
    "n_consec_decline": "consecutive recent years of employment decline",
    "share_decline_5": "share of the last five years with employment decline",
    "emp_dev5": "employment compared with the industry's own recent average",
    "g_va_1": "value-added change over the past year",
    "g_va_3": "value-added change over the past three years",
    "g_out_1": "output change over the past year",
    "g_out_3": "output change over the past three years",
    "g_wages_1": "total wage bill change over the past year",
    "g_prod_1": "labour-productivity change over the past year",
    "wage_share": "share of value added paid out as wages",
    "g_emp_1_rel_ctry": "employment growth relative to the country's manufacturing average",
    "g_emp_1_rel_ind": "employment growth relative to the same industry worldwide",
    "log_emp_share_ctry": "the industry's share of the country's manufacturing employment",
}
# how to read a high value of each signal (for warning-signal phrasing)
HIGHER_IS_WORSE = {  # True = a LOW/negative value raises risk; False = HIGH raises risk
    "g_emp_1": True, "g_emp_3": True, "g_va_1": True, "g_va_3": True,
    "g_out_1": True, "g_out_3": True, "g_wages_1": True, "g_prod_1": True,
    "g_emp_1_rel_ctry": True, "g_emp_1_rel_ind": True, "emp_dev5": True,
    "vol_emp_5": False, "n_consec_decline": False, "share_decline_5": False,
    "wage_share": False, "log_emp_share_ctry": True,
}
SEVERE_DEF = ("a fall of at least 20% in the number of employees over the following "
              "three years")
