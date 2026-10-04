"""Industrial Employment Risk Monitor — Streamlit research preview.

A calibrated screening tool for country x manufacturing-industry observations at
elevated risk of a severe employment contraction (a fall of at least 20% in the
number of employees over the following three years). Not a forecast, not causal,
not a policy prescription. The app loads a frozen model's precomputed outputs; it
never trains or scores models interactively.
"""
import os, sys, json, re
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.config import (TIER_NAMES, TIER_LABELS, TIER_COLORS, TIER_OBSERVED_RATE,
                        PLAIN, SEVERE_DEF, INSUFFICIENT, MODEL_VERSION, FEATURES)
from src import interpret as I
from app.regions import region_of

PORTFOLIO = "https://rahulsh97.github.io"
st.set_page_config(page_title="Industrial Employment Risk Monitor", page_icon="\U0001F4C9",
                   layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
:root{--ink:#173247;--muted:#60727e;--teal:#376b88;--rust:#aa6849;--cream:#fcfcfa;--line:#cfdae0;--wash:#f3f8fa;}
.stApp{background:var(--cream);}
h1,h2,h3,h4{font-family:Georgia,'Times New Roman',serif!important;color:var(--ink);letter-spacing:-.01em;}
.block-container{max-width:1160px;padding-top:1.25rem;padding-bottom:4rem;}
p,li,label,div{color:var(--ink);}
.eyebrow{font-size:.72rem;font-weight:700;letter-spacing:.14em;color:var(--teal);text-transform:uppercase;}
.badge{display:inline-block;padding:.25rem .65rem;border-radius:999px;font-size:.78rem;font-weight:700;color:#fff;}
.preview{display:inline-block;border:1px solid var(--rust);color:var(--rust);padding:.25rem .6rem;border-radius:4px;font-size:.72rem;font-weight:700;letter-spacing:.1em;}
.app-head{display:flex;justify-content:space-between;gap:2rem;align-items:flex-start;border-bottom:1px solid var(--line);padding-bottom:1rem;margin-bottom:1rem;}
.app-head h1{font-size:clamp(1.7rem,3vw,2.6rem);margin:.25rem 0 .35rem;}
.app-head p{margin:0;color:var(--muted);max-width:720px;}
.meta-strip{display:flex;gap:.6rem 1.4rem;flex-wrap:wrap;background:var(--wash);border:1px solid var(--line);padding:.65rem .85rem;margin:.75rem 0 1.2rem;font-size:.82rem;}
.meta-strip b{color:var(--ink);}
.card{background:#fff;border:1px solid var(--line);border-radius:4px;padding:1.15rem 1.3rem;margin:.6rem 0;}
.panel{background:#fff;border-left:3px solid var(--teal);border-radius:0 4px 4px 0;padding:1rem 1.2rem;margin:.65rem 0;}
.warn{background:#f8eee8;border-left:4px solid var(--rust);border-radius:0 8px 8px 0;padding:.8rem 1.1rem;margin:.6rem 0;}
.muted{color:var(--muted);font-size:.9rem;}
.tierdot{display:inline-block;width:.8rem;height:.8rem;border-radius:50%;margin-right:.4rem;vertical-align:middle;}
.stDownloadButton button,.stButton button{border:1px solid var(--teal);color:var(--teal);background:#fff;}
a{color:var(--teal);}
div[data-testid="stMetric"]{background:#fff;border-top:2px solid var(--teal);padding:.85rem 1rem;}
div[data-testid="stSegmentedControl"]{margin-bottom:.35rem;}
div[data-testid="stSegmentedControl"] button{border-radius:0!important;font-weight:650;}
@media(max-width:700px){.app-head{display:block}.meta-strip{gap:.4rem .8rem}.block-container{padding-top:.8rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load():
    p = os.path.join(ROOT, "data", "processed")
    scores = pd.read_csv(os.path.join(p, "latest_scores.csv"))
    scores["region"] = scores["country"].map(region_of)
    meta = json.load(open(os.path.join(p, "meta.json")))
    hist = pd.read_csv(os.path.join(p, "emp_history.csv"))
    perf = json.load(open(os.path.join(ROOT, "reports", "performance.json")))
    stats = json.load(open(os.path.join(ROOT, "models", "stats.json")))
    return scores, meta, hist, perf, stats


scores, meta, hist, perf, stats = load()
BASE = stats["base_rate"]
TIER_ORDER = {t: i for i, t in enumerate(TIER_NAMES + [INSUFFICIENT])}


def tier_badge(tier):
    c = TIER_COLORS.get(tier, "#9aa5a1")
    label = TIER_LABELS.get(tier, tier)
    return f'<span class="badge" style="background:{c}">{label}</span>'


def tier_dot(tier):
    return f'<span class="tierdot" style="background:{TIER_COLORS.get(tier,"#9aa5a1")}"></span>'


def b(s):  # markdown **bold** -> HTML <b> for text injected into raw HTML panels
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(s))


def pctw(x):
    return "insufficient data" if x is None or pd.isna(x) else f"{round(x*100)}%"


def signbars(df, value="risk_prob", label="country", title=""):
    df = df.copy()
    df = df[df[value].notna()]
    x = (df[value] * 100).clip(lower=0)
    colors = [TIER_COLORS.get(t, "#9aa5a1") for t in df["tier"]]
    fig = go.Figure(go.Bar(
        x=x, y=df[label], orientation="h", marker_color=colors,
        text=[pctw(v) for v in df[value]], textposition="auto",
        hovertext=[f"{n}: {pctw(v)} ({t})" for n, v, t in zip(df[label], df[value], df["tier"])],
        hoverinfo="text"))
    fig.update_layout(title=title, height=max(260, 26 * len(df) + 90),
                      margin=dict(l=10, r=30, t=40, b=10), xaxis_title="Calibrated risk probability (%)",
                      plot_bgcolor="#fff", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#17252c"))
    fig.update_xaxes(gridcolor="#eceee9", range=[0, max(55, float(x.max()) + 12 if len(x) else 55)])
    fig.update_yaxes(autorange="reversed")
    return fig


def coverage_note(row):
    bits = []
    if not row["in_training"]:
        bits.append("this country was **not** in the model's training data — results for previously "
                    "unseen countries are weaker and should be treated with extra caution")
    if row["completeness"] < 0.75:
        bits.append(f"only **{round(row['completeness']*row['n_features'])} of {row['n_features']}** "
                    "indicators are available for this observation")
    if row.get("concept_break"):
        bits.append("the source employment series for this unit mixes reporting concepts; only the "
                    "employees series is used")
    return bits


# ---- product header and simple navigation -----------------------------------
st.markdown(
    '<div class="app-head"><div><span class="eyebrow">UNIDO INDSTAT · EMPLOYMENT RISK</span>'
    '<h1>Industrial Employment Risk Monitor</h1>'
    '<p>A transparent screening tool for finding country–industry cases that may deserve '
    'closer investigation. It is not a forecast or a causal model.</p></div>'
    f'<div><a href="{PORTFOLIO}/industrial-employment-risk-monitor/" target="_top">'
    'Project overview ↗</a></div></div>', unsafe_allow_html=True)

mode = st.segmented_control("Choose a view", ["Briefing", "Explore", "Evidence"],
                            default="Briefing", label_visibility="collapsed")
if mode == "Explore":
    explore_page = st.segmented_control(
        "Explore by", ["A single case", "A country", "An industry"],
        default="A single case", label_visibility="collapsed")
    page = {"A single case": "Case explorer", "A country": "Country view",
            "An industry": "Industry view"}[explore_page]
elif mode == "Evidence":
    evidence_page = st.segmented_control(
        "Evidence view", ["Model performance", "Methods & limitations"],
        default="Model performance", label_visibility="collapsed")
    page = evidence_page
else:
    page = "Overview"

st.markdown(
    f'<div class="meta-strip"><span><b>{meta["n_countries"]}</b> economies</span>'
    f'<span><b>{meta["n_industries"]}</b> industries</span>'
    f'<span><b>{meta["n_scored"]:,}</b> scored cases</span>'
    f'<span>latest year <b>{meta["latest_year"]}</b></span>'
    f'<span>model <b>{MODEL_VERSION}</b></span></div>', unsafe_allow_html=True)


def tier_legend():
    cols = st.columns(4)
    for i, t in enumerate(TIER_NAMES):
        with cols[i]:
            st.markdown(f"{tier_dot(t)} **{TIER_LABELS[t]}**  \n"
                        f'<span class="muted">~{round(TIER_OBSERVED_RATE[t]*100)} in 100 later '
                        f'had a severe contraction</span>', unsafe_allow_html=True)


# ============================ OVERVIEW =======================================
if page == "Overview":
    st.markdown('<span class="eyebrow">Global value chains · Labour markets</span>', unsafe_allow_html=True)
    st.title("Where might manufacturing employment contract severely?")
    st.markdown(
        f"This research preview screens **country × manufacturing-industry** observations for an "
        f"**elevated risk of a severe employment contraction** — defined as {SEVERE_DEF}. "
        f"It is a **calibrated screening system**: it ranks and tiers observations by estimated risk "
        f"so analysts can decide where to look more closely. It does **not** provide causal estimates, "
        f"exact employment forecasts or policy prescriptions.")
    c = st.columns(4)
    c[0].metric("Countries", meta["n_countries"])
    c[1].metric("Manufacturing industries", meta["n_industries"])
    c[2].metric("Latest scoring year", meta["latest_year"])
    c[3].metric("Observations scored", f"{meta['n_scored']}")
    st.markdown("### The four risk tiers")
    st.markdown('<span class="muted">Tiers come from the model\'s calibrated probability. The figure '
                'beside each is the share of observations in that tier that actually went on to have a '
                'severe contraction, in out-of-sample testing.</span>', unsafe_allow_html=True)
    tier_legend()
    st.markdown("### What a severe contraction means")
    st.markdown(f"A **severe contraction** is {SEVERE_DEF}. Across the evaluation sample, about "
                f"**{I.freq(BASE)}** country-industry observations experienced one over a three-year window.")
    counts = scores[scores.scoreable].tier.value_counts()
    fig = go.Figure(go.Bar(x=[TIER_LABELS[t] for t in TIER_NAMES],
                           y=[int(counts.get(t, 0)) for t in TIER_NAMES],
                           marker_color=[TIER_COLORS[t] for t in TIER_NAMES]))
    fig.update_layout(title=f"Scored observations by tier ({meta['latest_year']})", height=330,
                      plot_bgcolor="#fff", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10),
                      font=dict(color="#17252c"))
    st.plotly_chart(fig, width="stretch")
    st.info("This is a research preview built from UNIDO INDSTAT Rev.4 for methods demonstration. "
            "Results are descriptive risk associations, not causal findings.")

# ============================ COUNTRY VIEW ===================================
elif page == "Country view":
    st.title("Country view")
    country = st.selectbox("Select a country", sorted(scores.country.unique()))
    cdf = scores[scores.country == country].copy()
    cdf["torder"] = cdf["tier"].map(TIER_ORDER)
    cdf = cdf.sort_values(["scoreable", "risk_prob"], ascending=[False, False])
    row0 = cdf.iloc[0]
    if not row0["in_training"]:
        st.markdown('<div class="warn"><b>Caution:</b> this country was <b>not</b> represented in the '
                    'model\'s training data. Performance is weaker for previously unseen countries.</div>',
                    unsafe_allow_html=True)
    nlow = cdf["completeness"].lt(0.75).sum()
    if nlow:
        st.markdown(f'<div class="warn"><b>Coverage:</b> {nlow} of this country\'s industries have weak '
                    f'data completeness; read those scores with caution.</div>', unsafe_allow_html=True)
    st.markdown(f"**{country}** — {int(cdf.scoreable.sum())} of {len(cdf)} manufacturing industries scored "
                f"for {meta['latest_year']}. Industries ranked by calibrated risk:")
    show = cdf.copy()
    show["Risk"] = show["risk_prob"].map(pctw)
    show["Tier"] = show["tier"].map(lambda t: TIER_LABELS.get(t, t))
    show["Completeness"] = (show["completeness"] * 100).round().astype(int).astype(str) + "%"
    show["Recent 3-yr employment change"] = show["g_emp_3"].map(
        lambda v: "n/a" if pd.isna(v) else f"{v*100:+.0f}%")
    st.dataframe(show[["activity", "Tier", "Risk", "Completeness", "Recent 3-yr employment change"]]
                 .rename(columns={"activity": "Industry"}), width="stretch", hide_index=True)
    st.plotly_chart(signbars(cdf[cdf.scoreable], label="activity",
                             title=f"{country}: calibrated risk by industry ({meta['latest_year']})"),
                    width="stretch")
    dl = cdf[["country", "activity_code", "activity", "year", "risk_prob", "tier",
              "completeness", "in_training", "g_emp_1", "g_emp_3"]]
    st.download_button(f"Download {country} scores (CSV)", dl.to_csv(index=False),
                       file_name=f"ierm_{country}_{meta['latest_year']}.csv", mime="text/csv")

# ============================ INDUSTRY VIEW ==================================
elif page == "Industry view":
    st.title("Industry view")
    industry = st.selectbox("Select a manufacturing industry (full ISIC Rev.4 name)",
                            sorted(scores.activity.unique()))
    idf = scores[scores.activity == industry].copy()
    cc = st.columns(3)
    regions = ["All regions"] + sorted(idf.region.unique())
    reg = cc[0].selectbox("Region", regions)
    tiersel = cc[1].multiselect("Risk tier", [TIER_LABELS[t] for t in TIER_NAMES],
                                default=[TIER_LABELS[t] for t in TIER_NAMES])
    mincomp = cc[2].slider("Minimum data completeness", 0, 100, 0, 5) / 100
    f = idf[idf.scoreable].copy()
    if reg != "All regions":
        f = f[f.region == reg]
    keep_tiers = [t for t in TIER_NAMES if TIER_LABELS[t] in tiersel]
    f = f[f.tier.isin(keep_tiers) & (f.completeness >= mincomp)]
    f = f.sort_values("risk_prob", ascending=False)
    st.markdown(f"**{industry}** — {len(f)} country observations shown "
                f"(of {int(idf.scoreable.sum())} scored). A map is not shown: no reliable open boundary "
                f"file is bundled, so a ranked chart is used instead (see Methods).")
    if len(f):
        st.plotly_chart(signbars(f, label="country",
                                 title=f"{industry}: calibrated risk across countries ({meta['latest_year']})"),
                        width="stretch")
        show = f.copy(); show["Risk"] = show["risk_prob"].map(pctw)
        show["Tier"] = show["tier"].map(lambda t: TIER_LABELS[t])
        show["In training"] = show["in_training"].map({True: "yes", False: "no"})
        st.dataframe(show[["country", "region", "Tier", "Risk", "In training"]]
                     .rename(columns={"country": "Country", "region": "Region"}),
                     width="stretch", hide_index=True)
        st.download_button("Download selection (CSV)",
                           f[["country", "activity", "year", "risk_prob", "tier", "completeness", "in_training"]].to_csv(index=False),
                           file_name=f"ierm_{industry[:20]}_{meta['latest_year']}.csv", mime="text/csv")
    else:
        st.info("No observations match the current filters.")

# ============================ CASE EXPLORER ==================================
elif page == "Case explorer":
    st.title("Case explorer")
    c1, c2 = st.columns(2)
    country = c1.selectbox("Country", sorted(scores.country.unique()))
    inds = scores[scores.country == country].activity.unique()
    industry = c2.selectbox("Manufacturing industry", sorted(inds))
    r = scores[(scores.country == country) & (scores.activity == industry)]
    if r.empty:
        st.info("No observation for this selection.")
        st.stop()
    row = r.iloc[0].to_dict()

    # employment history (rebased index, derived)
    h = hist[(hist.country_code == row["country_code"]) & (hist.activity_code == row["activity_code"])]
    left, right = st.columns([3, 2])
    with left:
        if len(h) >= 2:
            fig = go.Figure(go.Scatter(x=h.year, y=h.emp_index, mode="lines+markers",
                                       line=dict(color="#11718b", width=2)))
            fig.update_layout(title=f"Employment index (rebased to 100 at {int(h.year.min())})",
                              height=300, plot_bgcolor="#fff", paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=40, b=10), font=dict(color="#17252c"),
                              yaxis_title="index", xaxis_title="year")
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("Not enough employment history to chart.")
    with right:
        st.markdown(f"#### {tier_badge(row['tier'])}", unsafe_allow_html=True)
        st.markdown(f"**{country} · {industry}**  \n<span class='muted'>ISIC {row['activity_code']} · "
                    f"{row['year']}</span>", unsafe_allow_html=True)
        if I.is_sufficient(row):
            st.metric("Calibrated risk probability", pctw(row["risk_prob"]))

    st.markdown("---")
    # ===== interpretation panel =====
    st.markdown("## What does this result mean?")
    if not I.is_sufficient(row):
        st.markdown('<div class="warn"><b>Insufficient data.</b> The required indicators are not '
                    'available for this observation, so no risk score is shown. Missing data is never '
                    'treated as zero.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="panel"><b>1 · The result.</b> {b(I.result_text(row))}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="panel"><b>2 · The right comparison.</b> {b(I.comparison_text(row, BASE))}</div>', unsafe_allow_html=True)
        sig_text, sigs = I.signals_text(row, stats)
        st.markdown(f'<div class="panel"><b>3 · Why it has been flagged.</b> {b(sig_text)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="panel"><b>4 · What it does not mean.</b> {b(I.does_not_mean_text())}</div>', unsafe_allow_html=True)
        checks = "".join(f"<li>{q}</li>" for q in I.checklist(row))
        st.markdown('<div class="panel"><b>5 · What to check next</b> — questions for investigation, '
                    f'not recommendations:<ul>{checks}</ul></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><b>6 · Data confidence.</b><br>' +
                    "<br>".join(b(x) for x in I.confidence_text(row)) + "</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="panel"><b>7 · Model track record.</b> {b(I.track_record_text(stats["perf"]))}</div>', unsafe_allow_html=True)
        with st.expander("Technical note (model metrics)"):
            tm = perf["models"]["trees_calibrated"]
            st.write(f"ROC-AUC {tm['roc_auc']}, PR-AUC {tm['pr_auc']}, Brier {tm['brier']} "
                     f"(out-of-time test, base rate {perf['base_rate']}). At the 50% recall operating "
                     f"point: precision {perf['operating_point_recall50']['precision']}, "
                     f"recall {perf['operating_point_recall50']['recall']}.")
        s = I.policymaker_summary(row, stats)
        st.markdown(f'<div class="card"><b>8 · Policymaker summary.</b><br><br>'
                    f'<b>Signal.</b> {s["signal"]}<br><br>'
                    f'<b>Evidence.</b> {s["evidence"]}<br><br>'
                    f'<b>Next step.</b> {s["next_step"]}</div>', unsafe_allow_html=True)
        st.caption("Warning signals are statistical associations, not causes. A probability is not a "
                   "certainty. This tool does not recommend subsidies, protection, finance or worker "
                   "programmes.")

# ============================ MODEL PERFORMANCE ==============================
elif page == "Model performance":
    st.title("Model performance")
    st.markdown("All figures are **out-of-time**: the model was trained on earlier years and tested on "
                "later years it had not seen (base years 2017-2020). The headline target is a severe "
                f"contraction ({SEVERE_DEF}).")
    m = perf["models"]
    c = st.columns(3)
    c[0].metric("ROC-AUC (calibrated trees)", m["trees_calibrated"]["roc_auc"])
    c[1].metric("PR-AUC (trees)", m["trees_calibrated"]["pr_auc"], f"no-skill {perf['base_rate']}")
    c[2].metric("Brier (trees)", m["trees_calibrated"]["brier"], "lower is better")

    colA, colB = st.columns(2)
    with colA:
        fig = go.Figure()
        for nm, key, col in [("Calibrated trees", "trees", "#11718b"), ("Logistic", "logistic", "#e3b23c"),
                             ("Recent-trend rule", "recent_trend", "#b4452f")]:
            fig.add_trace(go.Scatter(x=perf["roc"][key]["fpr"], y=perf["roc"][key]["tpr"], name=nm, line=dict(color=col)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#999"), showlegend=False))
        fig.update_layout(title="ROC curve", height=360, plot_bgcolor="#fff", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#17252c"), xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, width="stretch")
    with colB:
        fig = go.Figure()
        for nm, key, col in [("Calibrated trees", "trees", "#11718b"), ("Logistic", "logistic", "#e3b23c"),
                             ("Recent-trend rule", "recent_trend", "#b4452f")]:
            fig.add_trace(go.Scatter(x=perf["pr"][key]["recall"], y=perf["pr"][key]["precision"], name=nm, line=dict(color=col)))
        fig.add_hline(y=perf["base_rate"], line_dash="dash", line_color="#999", annotation_text="no-skill")
        fig.update_layout(title="Precision-Recall curve", height=360, plot_bgcolor="#fff", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#17252c"), xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, width="stretch")

    colC, colD = st.columns(2)
    with colC:
        cal = perf["calibration"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cal["mean_pred"], y=cal["obs_freq"], mode="lines+markers",
                                 name="calibrated trees", line=dict(color="#11718b")))
        fig.add_trace(go.Scatter(x=[0, max(cal["mean_pred"])], y=[0, max(cal["mean_pred"])],
                                 line=dict(dash="dash", color="#999"), name="perfect"))
        fig.update_layout(title="Calibration (reliability)", height=340, plot_bgcolor="#fff",
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#17252c"),
                          xaxis_title="Predicted probability", yaxis_title="Observed frequency")
        st.plotly_chart(fig, width="stretch")
    with colD:
        tr = perf["tier_event_rates"]
        fig = go.Figure(go.Bar(x=[TIER_LABELS[t] for t in TIER_NAMES],
                               y=[tr[t]["event_rate"] for t in TIER_NAMES],
                               marker_color=[TIER_COLORS[t] for t in TIER_NAMES],
                               text=[f"{round(tr[t]['event_rate']*100)} in 100" for t in TIER_NAMES]))
        fig.update_layout(title="Observed contraction rate by risk tier (test)", height=340,
                          plot_bgcolor="#fff", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#17252c"))
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Model comparison and validation (out-of-time PR-AUC)")
    cmp = pd.DataFrame({
        "Model": ["Recent-trend rule", "Logistic regression", "Calibrated trees (used)"],
        "ROC-AUC": [m["recent_trend"]["roc_auc"], m["logistic"]["roc_auc"], m["trees_calibrated"]["roc_auc"]],
        "PR-AUC": [m["recent_trend"]["pr_auc"], m["logistic"]["pr_auc"], m["trees_calibrated"]["pr_auc"]],
        "Brier": [m["recent_trend"]["brier"], m["logistic"]["brier"], m["trees_calibrated"]["brier"]]})
    st.dataframe(cmp, width="stretch", hide_index=True)
    h = perf["holdouts"]
    hold = pd.DataFrame([{"Validation": k.replace("_", " "), "n_test": v["n_test"], "base rate": v["base_rate"],
                         "trees ROC-AUC": v["roc_auc"], "trees PR-AUC": v["pr_auc"],
                         "recent-trend PR-AUC": v["recent_trend_prauc"]}
                        for k, v in h.items()])
    st.dataframe(hold, width="stretch", hide_index=True)
    st.caption("The model beats the recent-trend rule on PR-AUC in every scenario, and is weakest for "
               "countries absent from training (new countries).")

    op = perf["operating_point_recall50"]
    st.markdown("### At the 50% recall operating point")
    cc = st.columns(4)
    cc[0].metric("Recall (caught)", f"{round(op['recall']*100)}%")
    cc[1].metric("Precision", f"{round(op['precision']*100)}%")
    cc[2].metric("True warnings", op["true_warnings"])
    cc[3].metric("False warnings", op["false_warnings"])
    st.markdown(f"In plain terms: at this setting the tool flags **{round(op['alert_rate']*100)}% of "
                f"observations**. It catches about **{round(op['recall']*100)} in 100** severe "
                f"contractions, and roughly **{round(op['precision']*100)} in 100 warnings** correspond "
                f"to one. Raising recall catches more true cases but raises false warnings — a screening "
                f"trade-off the analyst should set deliberately.")

# ============================ METHODS ========================================
elif page == "Methods & limitations":
    st.title("Methods & limitations")
    st.markdown(f"""
**Source.** UNIDO INDSTAT Rev.4 (country × ISIC Rev.4 manufacturing industry × year, 1991-2023).
Variables: employees, wages, output, value added. Monetary variables are nominal local currency with a
nominal-USD conversion; **no deflator or PPP exists in the source, so none was improvised**. Raw UNIDO
data is licensed and is **not** redistributed here — only compact derived indicators and model outputs.

**Employees vs persons engaged.** UNIDO reports employment under two underlying concepts — *employees*
and *persons engaged*. Some units switch concept across years. The model uses **employees only**,
consistently within every unit; the two concepts are never spliced.

**Why nominal levels were removed.** Cross-country nominal-USD levels are not comparable real
productivity. The model therefore uses **no nominal level variables** — only within-unit growth rates,
ratios (e.g. wage share), a rolling deviation from the unit's own history, and country-year / industry-year
relative measures. Removing levels did not reduce out-of-time performance.

**Target.** {SEVERE_DEF.capitalize()}. Built from the employees series; an observation is labelled only
when employment is observed at both the base year and three years later. The 10% threshold is retained as
a sensitivity check.

**Features & windows.** Every predictor at year *t* uses only data from years ≤ *t*. Growth is computed
across the exact calendar gap and, for monetary variables, only on a consistent valuation basis. Missing
values are preserved as NaN and never converted to zero; an observation is scored only when the key
indicators are present and at least half of all indicators are available, otherwise it shows
**"insufficient data"**.

**Time splits & calibration.** No random split. The evaluation model trains on 2000-2013, selects on
2014-2016 and is tested on 2017-2020. The deployed scoring model trains on 2000-2017 and is isotonic-
recalibrated on 2018-2020, then scores the latest usable year ({meta['latest_year']}). Reported
performance is the out-of-time test.

**It is descriptive, not causal.** The model identifies statistical associations useful for screening.
It does not estimate causal effects, and it performs **more weakly for countries absent from training**.

**Map.** No reliable open boundary file is bundled, so the Industry view uses a ranked chart rather than
a choropleth, to avoid a delayed build or an unverified/disputed-boundary map.
""")
    st.markdown("### Feature dictionary")
    st.dataframe(pd.DataFrame({"model signal (internal)": FEATURES,
                               "plain-language meaning": [PLAIN[f] for f in FEATURES]}),
                 width="stretch", hide_index=True)
    st.markdown(f"**Model:** `{MODEL_VERSION}` · calibrated gradient-boosted trees (isotonic). "
                "See `MODEL_CARD.md` in the repository for the full model card.")
    st.caption("Concept, research design and interpretation: Rahul Shukla. "
               "Engineering assistance: OpenAI Codex and Anthropic Claude Code.")
