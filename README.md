# Industrial Employment Risk Monitor

An open, calibrated early-warning **research preview** for identifying country–industry
pairs at elevated risk of **severe manufacturing employment contraction** — a fall of
at least 20% in the number of employees over the following three years.

It is a **screening system**, not a forecast, not a causal model, and not a policy
tool. To our knowledge it is among the first public, reproducible applications to turn
UNIDO INDSTAT into a calibrated, out-of-time-validated employment-risk screen.

**Concept, research design and interpretation: Rahul Shukla. Engineering assistance:
OpenAI Codex and Anthropic Claude Code.** See [`MODEL_CARD.md`](MODEL_CARD.md) and
[`NOTICE.md`](NOTICE.md).

## What it does
For each country × ISIC Rev.4 manufacturing division it shows a calibrated risk
probability, a risk tier (Low / Watch / Elevated / High), data-coverage flags, the
recent employment trajectory, and a plain-language interpretation of the strongest
**warning signals** (never described as causes). Five app pages: Overview, Country
view, Industry view, Case explorer (with the "What does this result mean?" panel),
Model performance, plus Methods & limitations.

## Repository layout
```
app/        Streamlit application (streamlit_app.py) + region lookup
src/        frozen modelling source: config, features, model, interpret, pipeline
models/     model.joblib (frozen calibrated trees) + stats.json
data/processed/  latest_scores.csv, emp_history.csv, meta.json  (derived only)
reports/    performance.json (out-of-time metrics & curves)
tests/      pytest suite
Dockerfile · requirements.txt · MODEL_CARD.md · NOTICE.md
```

## Data & provenance
Built from **UNIDO INDSTAT Rev.4** (country × ISIC Rev.4 manufacturing industry ×
year, 1991–2023; employees, wages, output, value added). Monetary variables are
nominal local currency with a nominal-USD conversion; no deflator/PPP exists in the
source. **Raw UNIDO data is licensed and is not committed to this repository.** Only
compact *derived* indicators (growth rates, ratios, a rebased employment index) and
*model outputs* (calibrated risk, tiers) are shipped, with provenance documented in
`data/processed/meta.json`. Regenerate everything from a local licensed copy with the
pipeline below.

## Reproduce
```bash
pip install -r requirements-dev.txt
# rebuild panel -> performance artefacts -> freeze scoring model -> scored data
python -m src.pipeline --data /path/to/UNIDO_rev4/data.csv
python -m pytest tests/ -q
streamlit run app/streamlit_app.py
```
The pipeline writes `models/model.joblib`, `models/stats.json`,
`reports/performance.json`, `data/processed/latest_scores.csv`,
`data/processed/emp_history.csv` and `meta.json`. The app never trains or scores a
model at runtime — it loads the frozen artefacts.

## Model (summary)
Calibrated gradient-boosted trees (isotonic), **specification C — no nominal level
variables**, employees-only outcome. Out-of-time test (2017–2020): ROC-AUC ≈ 0.68,
PR-AUC ≈ 0.20 (base rate 0.11), Brier 0.092; beats a recent-trend rule in every
validation scenario; weaker for countries unseen in training. Full details and
feature dictionary in [`MODEL_CARD.md`](MODEL_CARD.md).

### Feature dictionary
| internal signal | plain-language meaning |
|---|---|
| g_emp_1 | employment change over the past year |
| g_emp_3 | employment change over the past three years |
| vol_emp_5 | how much recent employment growth has swung year to year |
| n_consec_decline | consecutive recent years of employment decline |
| share_decline_5 | share of the last five years with employment decline |
| emp_dev5 | employment compared with the industry's own recent average |
| g_va_1 / g_va_3 | value-added change over the past year / three years |
| g_out_1 / g_out_3 | output change over the past year / three years |
| g_wages_1 | total wage bill change over the past year |
| g_prod_1 | labour-productivity change over the past year |
| wage_share | share of value added paid out as wages |
| g_emp_1_rel_ctry | employment growth relative to the country's manufacturing average |
| g_emp_1_rel_ind | employment growth relative to the same industry worldwide |
| log_emp_share_ctry | the industry's share of the country's manufacturing employment |

## Deploy
A `Dockerfile` is provided (Cloud Run reads `$PORT`). The app also runs directly on
Streamlit Community Cloud with `app/streamlit_app.py` as the entry point and
`requirements.txt`.

## Limitations
Descriptive, not causal. Moderate discrimination; weakest for countries absent from
training. Monetary growth is nominal. Some observations are unscoreable and show
"insufficient data" (missing is never treated as zero). See `MODEL_CARD.md`.
