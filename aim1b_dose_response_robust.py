"""
Aim 1b - Dose-response designs robust to confounding by indication.

Why: in the interval-level analysis (aim1) more opioid was followed by MORE pain, because patients
in more pain receive more opioid (confounding by indication). Two designs reduce this bias:

A) Dose-level "conditional" analysis (MIMIC only, administered doses):
   each IV bolus or oral dose given for pain (pre-dose score >= 4, measured <= 60 min before the dose),
   with the nurse reassessment 15-120 min (IV) or 30-180 min (oral) after the dose.
   Patients are compared at the SAME pre-dose pain level, so the indication is held fixed.
   Outcomes: pain reduction (points), clinically meaningful relief (>= 2 points or >= 30%),
   oversedation (RASS <= -3 within 2 h).
   Models: (i) adjusted GLM with patient-cluster SEs; (ii) within-patient fixed-effects model
   (each patient is his/her own control: larger vs smaller doses in the same patient).
   Ceiling: two-segment model on log dose (breakpoint with profile range).

B) Interval-level within-patient fixed effects (both cohorts):
   pain in t+1 on opioid in t, demeaned within each stay, adjusted for time-varying factors
   (pain in t-1, interval, adjuncts, sedatives, infusion). Stable patient differences
   (tolerance, surgery, frailty) cancel out.

Only aggregate outputs; cells < 11 suppressed.

Usage:
  python aim1b_dose_response_robust.py --data mimic --dir D:\\MIMIC_analytic --out D:\\MIMIC_analytic\\aim1b
  python aim1b_dose_response_robust.py --data eicu  --dir D:\\EICU_analytic  --out D:\\EICU_analytic\\aim1b
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(2026)
ADJ = ["paracetamol", "nsaid", "gabapentinoid", "ketamine", "dexmedetomidine", "propofol", "benzodiazepine"]
MIN_CELL = 11
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def varies(d, cols):
    return [c for c in cols if c in d and d[c].nunique(dropna=True) > 1]


def glm_cluster(y, X, groups, family):
    return sm.GLM(y, X, family=family).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})


def curve_from(res, di, base, xname, grid, family, n_draws=300):
    beta = res.params.values
    draws = RNG.multivariate_normal(beta, res.cov_params().values, size=n_draws, method="eigh")
    inv = family.link.inverse
    out = []
    for v in grid:
        nd = base.copy()
        nd[xname] = np.log1p(v)
        X = np.asarray(patsy.build_design_matrices([di], nd)[0])
        out.append((v, inv(X @ beta).mean(), *np.percentile(inv(X @ draws.T).mean(axis=0), [2.5, 97.5])))
    return pd.DataFrame(out, columns=["mme", "estimate", "lo95", "hi95"])


def within(df, cols, gid):
    g = df.groupby(gid)[cols].transform("mean")
    return df[cols] - g


def fe_curve(d, y, xcol, tv_terms, gid, cluster, grid, ref):
    """Within-stay fixed-effects linear model; curve = difference from reference dose."""
    d = d[d.groupby(gid)[gid].transform("size") >= 2]
    if len(d) < 500:
        return None, None
    f = f"{y} ~ cr({xcol}, df=4, constraints='center')" + ("" if not tv_terms else " + " + " + ".join(tv_terms))
    yy, X = patsy.dmatrices(f, d, return_type="dataframe")
    di = X.design_info
    X = X.drop(columns=["Intercept"])
    keep = yy.index
    dd = d.loc[keep]
    Xw = within(pd.concat([X, dd[[gid]]], axis=1), list(X.columns), gid)
    yw = within(pd.concat([yy, dd[[gid]]], axis=1), [y], gid)[y]
    res = sm.OLS(yw, Xw).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd[cluster])[0]})
    sp = [c for c in X.columns if c.startswith("cr(")]
    base = dd.iloc[:1].copy()

    def basis(v):
        nd = base.copy()
        nd[xcol] = np.log1p(v)
        M = patsy.build_design_matrices([di], nd, return_type="dataframe")[0]
        return M[sp].values[0]
    b = res.params[sp].values
    V = res.cov_params().loc[sp, sp].values
    r0 = basis(ref)
    rows = []
    for v in grid:
        diff = basis(v) - r0
        est = diff @ b
        se = np.sqrt(diff @ V @ diff)
        rows.append((v, est, est - 1.96 * se, est + 1.96 * se))
    return pd.DataFrame(rows, columns=["mme", "diff_vs_ref", "lo95", "hi95"]), (len(dd), dd[gid].nunique())


def hinge(d, y, xcol, cov, family, groups):
    pos = d[xcol]
    cands = np.unique(np.round(np.quantile(pos, np.linspace(0.10, 0.90, 33)), 3))
    t = []
    for c in cands:
        dd = d.assign(h=np.maximum(d[xcol] - c, 0))
        yy, X = patsy.dmatrices(f"{y} ~ {xcol} + h + {cov}", dd, return_type="dataframe")
        t.append((c, sm.GLM(yy, X, family=family).fit().deviance))
    t = pd.DataFrame(t, columns=["c", "dev"])
    best = t.loc[t["dev"].idxmin(), "c"]
    rng = t.loc[t["dev"] - t["dev"].min() <= 3.84, "c"]
    dd = d.assign(h=np.maximum(d[xcol] - best, 0))
    yy, X = patsy.dmatrices(f"{y} ~ {xcol} + h + {cov}", dd, return_type="dataframe")
    r = glm_cluster(yy, X, dd.loc[X.index, groups], family)
    ln2 = np.log(2)
    a = r.params[xcol]
    s_after = r.params[xcol] + r.params["h"]
    C = r.cov_params()
    se_after = np.sqrt(C.loc[xcol, xcol] + C.loc["h", "h"] + 2 * C.loc[xcol, "h"])
    return {"outcome": y, "breakpoint_mme": round(float(np.expm1(best)), 1),
            "bp_range_low": round(float(np.expm1(rng.min())), 1), "bp_range_high": round(float(np.expm1(rng.max())), 1),
            "per_doubling_before": round(a * ln2, 4), "before_lo": round((a - 1.96 * r.bse[xcol]) * ln2, 4),
            "before_hi": round((a + 1.96 * r.bse[xcol]) * ln2, 4),
            "per_doubling_after": round(s_after * ln2, 4), "after_lo": round((s_after - 1.96 * se_after) * ln2, 4),
            "after_hi": round((s_after + 1.96 * se_after) * ln2, 4)}


# ---------------------------------------------------------------- A) dose level (MIMIC)
def dose_level(dirp, out):
    ev = pd.read_csv(os.path.join(dirp, "mimic_opioid_events.csv.gz"))
    if "mme" not in ev:
        log("mimic_opioid_events has no 'mme' column - rerun the updated mimic_step2_build.py first")
        return
    st = pd.read_csv(os.path.join(dirp, "mimic_stays.csv.gz"))
    bn = pd.read_csv(os.path.join(dirp, "mimic_bins.csv.gz"))
    st = st[st["core_cohort"] == 1]
    ev = ev[ev["stay_id"].isin(st["stay_id"])].copy()
    log(f"Dose-level: {len(ev):,} doses in core cohort")
    ev["bin"] = (ev["hrs"] // 6).astype(int)
    bcols = ["stay_id", "bin", "opioid_infusion"] + [c for c in ADJ if c in bn]
    ev = ev.merge(bn[bcols], on=["stay_id", "bin"], how="left")
    ev = ev.merge(st[["stay_id", "age_num", "female", "egfr_cat", "surgery_group", "intubated_day1"]], on="stay_id")
    ev["egfr_cat"] = ev["egfr_cat"].fillna("missing")
    ev["intubated_day1"] = ev["intubated_day1"].fillna(0)
    for c in ["opioid_infusion"] + ADJ:
        ev[c] = ev[c].fillna(0) if c in ev else 0
    # doses stacked within 60 min of a previous dose in the same patient
    ev = ev.sort_values(["stay_id", "hrs"])
    ev["prev_gap_min"] = ev.groupby("stay_id")["hrs"].diff() * 60
    ev["stacked"] = (ev["prev_gap_min"] < 60).astype(int)
    win = {"iv_bolus": (15, 120), "oral": (30, 180)}
    ev["in_window"] = [win[s][0] <= m <= win[s][1] if pd.notna(m) else False
                       for s, m in zip(ev["src"], ev["min_to_pain_reassess"])]
    flow = [("all doses", len(ev))]
    q = ev[ev["pain_before"].notna() & (ev["min_from_pain_before"] <= 60)]
    flow.append(("pre-dose pain <= 60 min before", len(q)))
    q = q[q["pain_before"] >= 4]
    flow.append(("pre-dose pain >= 4 (indication)", len(q)))
    q = q[q["in_window"]]
    flow.append(("reassessed in pharmacological window", len(q)))
    q = q[(q["mme"] > 0)].copy()
    pd.DataFrame(flow, columns=["step", "doses"]).to_csv(os.path.join(out, "aim1b_mimic_dose_flow.csv"), index=False)
    for s, n in flow:
        log(f"  {s}: {n:,}")
    q["delta"] = q["pain_before"] - q["pain_after"]
    q["relief"] = ((q["delta"] >= 2) | (q["delta"] >= 0.3 * q["pain_before"])).astype(int)
    q["overs2h"] = ((q["rass_after"] <= -3) & (q["min_to_rass"] <= 120)).astype(float).where(q["min_to_rass"] <= 120)
    q["x"] = np.log1p(q["mme"])
    q["hrs_c"] = q["hrs"]

    cats = pd.cut(q["mme"], [0, 5, 7.5, 10, 15, 22.5, 30, np.inf], labels=["<=5", "5-7.5", "7.5-10", "10-15", "15-22.5", "22.5-30", ">30"])
    t = q.assign(cat=cats).groupby(["src", "cat"], observed=False).agg(
        doses=("stay_id", "size"), stays=("stay_id", "nunique"), pain_before=("pain_before", "mean"),
        pain_after=("pain_after", "mean"), delta=("delta", "mean"), relief_pct=("relief", "mean"),
        overs2h_pct=("overs2h", "mean"), min_to_reassess=("min_to_pain_reassess", "median")).reset_index()
    t["relief_pct"] = (100 * t["relief_pct"]).round(1)
    t["overs2h_pct"] = (100 * t["overs2h_pct"]).round(1)
    t.loc[t["doses"] < MIN_CELL, t.columns[3:]] = np.nan
    t.round(2).to_csv(os.path.join(out, "aim1b_mimic_dose_categories.csv"), index=False)

    gauss, binom = sm.families.Gaussian(), sm.families.Binomial()
    curves, fes, hs = [], [], []
    for src, grid, ref in [("iv_bolus", [2.5, 5, 7.5, 10, 15, 20, 30, 45, 60], 7.5),
                           ("oral", [1, 2.5, 5, 7.5, 10, 15, 22.5, 30, 45], 7.5)]:
        d = q[q["src"] == src].copy()
        if len(d) < 500:
            continue
        hi = d["mme"].quantile(0.99)
        grid = [g for g in grid if g <= hi]
        cov_terms = ["cr(pain_before, df=3)", "hrs_c", "age_num", "female", "C(egfr_cat)", "C(surgery_group)",
                     "intubated_day1", "C(agent)", "stacked"] + varies(d, ["opioid_infusion"] + ADJ)
        cov_terms = [c for c in cov_terms if not c.startswith("C(") or d[c[2:-1]].nunique() > 1]
        cov = " + ".join(cov_terms)
        base = d.sample(min(len(d), 20000), random_state=1)
        for y, fam, dd in [("delta", gauss, d), ("relief", binom, d), ("overs2h", binom, d.dropna(subset=["overs2h"]))]:
            if dd[y].nunique() < 2:
                continue
            yy, X = patsy.dmatrices(f"{y} ~ cr(x, df=4, constraints='center') + {cov}", dd, return_type="dataframe")
            r = glm_cluster(yy, X, dd.loc[X.index, "stay_id"], fam)
            c = curve_from(r, X.design_info, base if y != "overs2h" else dd.sample(min(len(dd), 20000), random_state=1),
                           "x", grid, fam)
            c.insert(0, "outcome", y)
            c.insert(0, "route", src)
            c["n_doses"], c["n_stays"] = len(dd), dd["stay_id"].nunique()
            curves.append(c)
            h = hinge(dd, y, "x", cov, fam, "stay_id")
            h["route"] = src
            hs.append(h)
            log(f"  [{src}] {y}: n={len(dd):,} breakpoint {h['breakpoint_mme']} MME ({h['bp_range_low']}-{h['bp_range_high']}); "
                f"per doubling before {h['per_doubling_before']} after {h['per_doubling_after']}")
            tv = ["cr(pain_before, df=3)", "hrs_c", "C(agent)", "stacked"] + varies(dd, ["opioid_infusion"] + ADJ)
            if y != "overs2h":
                fc, n = fe_curve(dd, y, "x", tv, "stay_id", "stay_id", grid, ref)
                if fc is not None:
                    fc.insert(0, "outcome", y)
                    fc.insert(0, "route", src)
                    fc["ref_mme"], fc["n_doses"], fc["n_stays"] = ref, n[0], n[1]
                    fes.append(fc)
    if curves:
        pd.concat(curves).round(4).to_csv(os.path.join(out, "aim1b_mimic_dose_curves.csv"), index=False)
    if fes:
        pd.concat(fes).round(4).to_csv(os.path.join(out, "aim1b_mimic_dose_within_patient.csv"), index=False)
    pd.DataFrame(hs).to_csv(os.path.join(out, "aim1b_mimic_dose_breakpoints.csv"), index=False)
    plot_dose(curves, out)


def plot_dose(curves, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not curves:
        return
    c = pd.concat(curves)
    ttl = {"delta": "Pain reduction after dose (points)", "relief": "Meaningful relief (%)", "overs2h": "Oversedation within 2 h (%)"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for i, src in enumerate(["iv_bolus", "oral"]):
        for j, y in enumerate(ttl):
            ax = axes[i, j]
            s = c[(c["route"] == src) & (c["outcome"] == y)]
            if s.empty:
                ax.set_visible(False)
                continue
            k = 1 if y == "delta" else 100
            ax.fill_between(s["mme"], s["lo95"] * k, s["hi95"] * k, color="#2a6fdb", alpha=0.15, linewidth=0)
            ax.plot(s["mme"], s["estimate"] * k, color="#2a6fdb", linewidth=2)
            ax.set_title(f"{'IV bolus' if src == 'iv_bolus' else 'Oral'}: {ttl[y]}", fontsize=10)
            ax.set_xlabel("MME per dose")
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "aim1b_mimic_dose_curves.png"), dpi=200)


# ---------------------------------------------------------------- B) interval level, within patient
def interval_fe(data, dirp, out):
    pre = "eicu" if data == "eicu" else "mimic"
    b = pd.read_csv(os.path.join(dirp, f"{pre}_bins.csv.gz"))
    s = pd.read_csv(os.path.join(dirp, f"{pre}_stays.csv.gz"))
    idc = "sid" if data == "eicu" else "stay_id"
    b, s = b.rename(columns={idc: "id"}), s.rename(columns={idc: "id"})
    s = s[s["core_cohort"] == 1]
    b = b[b["id"].isin(s["id"])].sort_values(["id", "bin"])
    b["mme"] = b["mme_prn_avail"] + b["mme_given_or_scheduled"] if data == "eicu" else b["mme_admin"]
    for c in ADJ + ["opioid_infusion"]:
        b[c] = b[c].fillna(0) if c in b else 0
    nx = b.groupby("id")[["bin", "pain_mean", "pain_max"]].shift(-1)
    pv = b.groupby("id")[["bin", "pain_mean"]].shift(1)
    b["pain_next"] = nx["pain_mean"].where(nx["bin"] == b["bin"] + 1)
    b["severe_next"] = (nx["pain_max"] >= 7).astype(float).where((nx["bin"] == b["bin"] + 1) & nx["pain_max"].notna())
    b["pain_prev"] = pv["pain_mean"].where(pv["bin"] == b["bin"] - 1)
    b["x"] = np.log1p(b["mme"])
    b = b.dropna(subset=["pain_next", "pain_prev"])
    clus = "id"
    if data == "eicu":
        b = b.merge(s[["id", "hospitalid"]], on="id")
        clus = "hospitalid"
    grid = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 250]
    grid = [g for g in grid if g <= b["mme"].quantile(0.99)]
    tv = ["pain_prev", "C(bin)"] + varies(b, ["opioid_infusion"] + ADJ)
    res = []
    for y in ["pain_next", "severe_next"]:
        fc, n = fe_curve(b.dropna(subset=[y]), y, "x", tv, "id", clus, grid, 0)
        if fc is None:
            continue
        fc.insert(0, "outcome", y)
        fc["ref_mme"], fc["n_intervals"], fc["n_stays"] = 0, n[0], n[1]
        res.append(fc)
        log(f"  [within-patient, {data}] {y}: n={n[0]:,} intervals / {n[1]:,} stays")
        for _, r in fc.iterrows():
            log(f"     {r['mme']:>6.1f} MME vs 0: {r['diff_vs_ref']:+.3f} ({r['lo95']:+.3f} to {r['hi95']:+.3f})")
    if res:
        pd.concat(res).round(4).to_csv(os.path.join(out, f"aim1b_{data}_interval_within_patient.csv"), index=False)


def main(a):
    os.makedirs(a.out, exist_ok=True)
    if a.data == "mimic":
        log("== A) dose-level conditional analysis")
        dose_level(a.dir, a.out)
    log(f"\n== B) interval-level within-patient fixed effects ({a.data})")
    interval_fe(a.data, a.dir, a.out)
    with open(os.path.join(a.out, f"aim1b_{a.data}_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    log(f"\nDone. Aggregate results in {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["eicu", "mimic"], required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
