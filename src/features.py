"""Panel construction + leakage-safe features + labels.

Authoritative modelling source, unchanged from the approved feasibility audit
(specification C). Employment concepts are kept separate (emp_e = employees,
primary; emp_pe = persons engaged) and never spliced. Monetary variables are
nominal-USD and used only for within-unit growth and ratios; monetary growth is
computed on a consistent valuation basis. Every predictor at t uses only years
<= t. Missing values are preserved as NaN; never zeroed.
"""
import csv
import numpy as np
import pandas as pd

from .config import FEATURES, LEVELS, EMP_COL, HORIZON, THRESHOLD

csv.field_size_limit(10 ** 7)
MFG = {str(d) for d in range(10, 34)}  # ISIC Rev.4 manufacturing divisions 10-33


# ---------------------------------------------------------------- panel build
def build_panel(data_path):
    """Build the (country, industry, year) panel from the UNIDO Rev.4 data.csv."""
    cell, meta_c, meta_a = {}, {}, {}
    with open(data_path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        ix = {n: i for i, n in enumerate(next(r))}
        for row in r:
            ac, vc = row[ix["ActivityCode"]], row[ix["VariableCode"]]
            if ac not in MFG or vc not in {"04", "05", "14", "20"}:
                continue
            cc, yr = row[ix["CountryCode"]], int(row[ix["Year"]])
            unc = row[ix["UnconsolidatedVariableCode"]]
            d = cell.setdefault((cc, ac, yr), {})
            try:
                if vc == "04":
                    v = float(row[ix["Value"]])
                    if unc == "04":
                        d["emp_e"] = v
                    elif unc == "03":
                        d["emp_pe"] = v
                else:
                    v = float(row[ix["ValueUSD"]])
                    if vc == "05":
                        d["wages_usd"] = v
                    elif vc == "14":
                        d["output_usd"] = v; d["output_unc"] = unc
                    elif vc == "20":
                        d["va_usd"] = v; d["va_unc"] = unc
            except ValueError:
                pass
            meta_c[cc] = row[ix["Country"]]; meta_a[ac] = row[ix["Activity"]]
    rows = []
    for (cc, ac, yr), d in cell.items():
        rows.append({"country_code": cc, "country": meta_c[cc], "activity_code": ac,
                     "activity": meta_a[ac], "year": yr,
                     "emp_e": d.get("emp_e"), "emp_pe": d.get("emp_pe"),
                     "wages_usd": d.get("wages_usd"),
                     "output_usd": d.get("output_usd"), "output_unc": d.get("output_unc"),
                     "va_usd": d.get("va_usd"), "va_unc": d.get("va_unc")})
    return pd.DataFrame(rows).sort_values(["country_code", "activity_code", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------- features
def _lk(df, col):
    return {(c, a, y): v for c, a, y, v
            in zip(df.country_code, df.activity_code, df.year, df[col]) if pd.notna(v)}


def add_features(df, emp_col=EMP_COL):
    df = df.sort_values(["country_code", "activity_code", "year"]).reset_index(drop=True)
    E = _lk(df, emp_col)
    VA, OUT, W = _lk(df, "va_usd"), _lk(df, "output_usd"), _lk(df, "wages_usd")
    VAU, OUTU = _lk(df, "va_unc"), _lk(df, "output_unc")

    def g_emp(c, a, y, k):
        x, b = E.get((c, a, y), np.nan), E.get((c, a, y - k), np.nan)
        return x / b - 1.0 if (not np.isnan(x) and not np.isnan(b) and b > 0) else np.nan

    def g_mon(L, U, c, a, y, k):
        x, b = L.get((c, a, y), np.nan), L.get((c, a, y - k), np.nan)
        ua, ub = U.get((c, a, y)), U.get((c, a, y - k))
        if np.isnan(x) or np.isnan(b) or b <= 0 or ua is None or ua != ub:
            return np.nan
        return x / b - 1.0

    rows = []
    for c, a, y in zip(df.country_code, df.activity_code, df.year):
        emp = E.get((c, a, y), np.nan)
        va, out, w = VA.get((c, a, y), np.nan), OUT.get((c, a, y), np.nan), W.get((c, a, y), np.nan)
        gs = [g_emp(c, a, y - j, 1) for j in range(5)]
        gv = [g for g in gs if not np.isnan(g)]
        vol = float(np.std(gv)) if len(gv) >= 3 else np.nan
        share_dec = float(np.mean([g < 0 for g in gv])) if gv else np.nan
        consec = 0
        for g in gs:
            if not np.isnan(g) and g < 0:
                consec += 1
            else:
                break
        hist = [E.get((c, a, y - j), np.nan) for j in range(5)]
        hv = [h for h in hist if not np.isnan(h) and h > 0]
        emp_dev5 = (np.log(emp / np.mean(hv)) if (not np.isnan(emp) and emp > 0 and len(hv) >= 2) else np.nan)
        prod = va / emp if (not np.isnan(va) and not np.isnan(emp) and emp > 0) else np.nan
        pe, ve = E.get((c, a, y - 1), np.nan), VA.get((c, a, y - 1), np.nan)
        prod_p = ve / pe if (not np.isnan(ve) and not np.isnan(pe) and pe > 0) else np.nan
        same = VAU.get((c, a, y)) is not None and VAU.get((c, a, y)) == VAU.get((c, a, y - 1))
        g_prod_1 = (prod / prod_p - 1.0 if (not np.isnan(prod) and not np.isnan(prod_p) and prod_p > 0 and same) else np.nan)
        wprev = W.get((c, a, y - 1), np.nan)
        rows.append({
            "g_emp_1": g_emp(c, a, y, 1), "g_emp_3": g_emp(c, a, y, 3),
            "vol_emp_5": vol, "n_consec_decline": consec, "share_decline_5": share_dec, "emp_dev5": emp_dev5,
            "g_va_1": g_mon(VA, VAU, c, a, y, 1), "g_va_3": g_mon(VA, VAU, c, a, y, 3),
            "g_out_1": g_mon(OUT, OUTU, c, a, y, 1), "g_out_3": g_mon(OUT, OUTU, c, a, y, 3),
            "g_wages_1": (w / wprev - 1.0 if (not np.isnan(w) and not np.isnan(wprev) and wprev > 0) else np.nan),
            "g_prod_1": g_prod_1,
            "wage_share": (w / va if (not np.isnan(w) and not np.isnan(va) and va > 0) else np.nan),
            "log_prod_va": np.log(prod) if (not np.isnan(prod) and prod > 0) else np.nan,
            "log_prod_out": (np.log(out / emp) if (not np.isnan(out) and out > 0 and not np.isnan(emp) and emp > 0) else np.nan),
            "wage_pe": (w / emp if (not np.isnan(w) and not np.isnan(emp) and emp > 0) else np.nan),
            "log_va": np.log(va) if (not np.isnan(va) and va > 0) else np.nan,
            "log_emp": np.log(emp) if (not np.isnan(emp) and emp > 0) else np.nan,
            "_emp": emp,
        })
    out = pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)
    out["g_emp_1_rel_ctry"] = out["g_emp_1"] - out.groupby(["country_code", "year"])["g_emp_1"].transform("median")
    out["g_emp_1_rel_ind"] = out["g_emp_1"] - out.groupby(["activity_code", "year"])["g_emp_1"].transform("median")
    tot = out.groupby(["country_code", "year"])["_emp"].transform("sum")
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where((out["_emp"] > 0) & (tot > 0), np.log(out["_emp"] / tot), np.nan)
    out["log_emp_share_ctry"] = share
    return out


def make_label(df, emp_col=EMP_COL, horizon=HORIZON, threshold=THRESHOLD):
    emp = {(c, a, y): v for c, a, y, v
           in zip(df.country_code, df.activity_code, df.year, df[emp_col]) if pd.notna(v) and v > 0}
    y = np.full(len(df), np.nan)
    for i, (c, a, yr) in enumerate(zip(df.country_code, df.activity_code, df.year)):
        e0 = emp.get((c, a, yr))
        if e0 is None:
            continue
        ef = emp.get((c, a, yr + horizon))
        if ef is None:
            continue
        y[i] = 1.0 if (ef / e0 - 1.0) <= -threshold else 0.0
    return pd.Series(y, index=df.index, name="y")
