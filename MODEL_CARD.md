# Model card — Industrial Employment Risk Monitor

**Model id:** `ierm-1.0.0` · calibrated gradient-boosted trees (isotonic).
**Owner:** Rahul Shukla (concept, research design, interpretation). Engineering
assistance: OpenAI Codex and Anthropic Claude Code.

## Intended use
A **calibrated screening tool** that ranks and tiers country × ISIC Rev.4
manufacturing-industry observations by the estimated risk of a **severe employment
contraction** — a fall of at least 20% in the number of *employees* over the
following three years. Intended for research, teaching and policy *screening*:
deciding where to look more closely.

**Not** intended for: causal inference, exact employment forecasting, automatic
policy decisions (subsidies, protection, finance, worker programmes), or firm-level
conclusions. Outputs are descriptive risk associations, not causes.

## Data
- **Source:** UNIDO INDSTAT Rev.4 (country × ISIC Rev.4 manufacturing division ×
  year, 1991–2023). Variables: employees, wages, output, value added.
- **Units & valuation:** monetary variables are nominal local currency with a
  nominal-USD conversion (`ValueUSD`); **no deflator/PPP exists in the source and
  none was improvised**. Raw UNIDO data is licensed and is **not** redistributed;
  only compact derived indicators and model outputs are shipped.
- **Employment concept:** *employees* only (`UnconsolidatedVariableCode 04`),
  consistent within every unit. The separate *persons engaged* concept (`03`) is
  never spliced in.

## Target
`y = 1` iff employees fall ≥20% over three years (`emp_{t+3}/emp_t − 1 ≤ −0.20`);
labelled only when employment is observed at both `t` and `t+3`. A 10% threshold is
retained as a sensitivity check.

## Features (16; specification C — no nominal level variables)
Within-unit growth (employment, value added, output, wages, productivity),
employment volatility, consecutive-decline count, share of recent declining years,
rolling deviation from the unit's own 5-year employment history, wage share, and
country-year / industry-year relative employment growth and within-country size
share. Every predictor uses only data from years ≤ `t`; monetary growth is computed
on a consistent valuation basis. **Nominal level variables were removed** because
cross-country nominal-USD levels are not comparable real productivity; removing them
did not reduce out-of-time performance. See the feature dictionary in `README.md`.

## Training & calibration
- **Evaluation model** (reported performance): train 2000–2013, select 2014–2016,
  test 2017–2020.
- **Scoring model** (deployed): train 2000–2017, isotonic-recalibrate on 2018–2020,
  then score the latest usable year (2023).
- Base learner: `HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06,
  max_iter=400, l2_regularization=1.0, class_weight="balanced", random_state=0)`;
  calibration: isotonic (`CalibratedClassifierCV` on a frozen estimator).
- Missing values are preserved (native NaN handling); never imputed to zero.

## Performance (out-of-time test, base years 2017–2020; base rate 0.11)
| model | ROC-AUC | PR-AUC | Brier |
|---|--:|--:|--:|
| Majority / no-skill | 0.50 | 0.11 | 0.096 |
| Recent-trend rule | 0.57 | 0.18 | — |
| Logistic regression | 0.65 | 0.25 | — |
| **Calibrated trees (used)** | **0.68** | **0.20** | **0.092** |

Validation (calibrated trees PR-AUC vs recent-trend): known units 0.20 vs 0.18;
new industries 0.22 vs 0.20; **new countries 0.14 vs 0.12** (weakest); pre-pandemic
0.15 vs 0.12. Beats the recent-trend rule in every scenario.

**Calibrated risk tiers** (observed contraction rate in out-of-sample test):
Low <5% → 5%; Watch 5–10% → 10%; Elevated 10–20% → 19%; High ≥20% → 28%.
Monotone and well separated versus the 11% base rate.

**Operating point (50% recall):** precision ≈ 0.21, alert rate ≈ 0.26 — the tool
flags ~1 in 4 observations, catches ~50 in 100 severe contractions, and ~21 in 100
warnings correspond to one.

## Limitations
- Moderate discrimination (ROC-AUC ≈ 0.68); a screening aid, not a precise forecast.
- **Weaker for countries absent from training** (ROC-AUC ≈ 0.60) — flagged in-app.
- Monetary growth remains nominal (mixes price and exchange-rate movement).
- INDSTAT coverage is uneven across countries and years; some units are unscoreable
  ("insufficient data").
- The 2023 scoring year is an out-of-time extrapolation relative to the labelled
  training window; treat recent scores as screening signals, not settled estimates.

## Ethical considerations
Descriptive, non-causal associations. The tool must not be used to justify
automatic policy actions or to attribute blame to firms, workers or governments.
Country and industry names are shown in full; disputed-boundary maps are avoided.
