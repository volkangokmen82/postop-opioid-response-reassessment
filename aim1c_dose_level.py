"""
Aim 1c - Single-dose analgesic ceiling (MIMIC-IV, administered doses). Refines aim1b part A.

Design: doses given for pain (pre-dose score >= 4, measured <= 60 min before the dose).
Outcome measured at the nurse reassessment 15-120 min (IV) / 30-180 min (oral) after the dose:
  delta  = pain reduction (points);  relief = >= 2 points or >= 30% reduction.
Changes vs aim1b:
  * effects are reported as CONTRASTS vs a reference dose (7.5 MME) using only the dose-spline
    coefficients (delta method, patient-cluster SEs) -> stable confidence intervals;
    relief uses a linear probability model (risk difference).
  * inverse-probability-of-reassessment weights (IPW): only ~half of indicated doses were reassessed
    in the window; the weights rebalance reassessed doses to all indicated doses.
  * agent-specific curves (fentanyl, hydromorphone, morphine) with doses in native units.
  * oversedation after a dose is rare -> reported descriptively only.
  * within-patient (fixed-effects) contrasts repeated with the same contrast method.

Usage (the aim1b script must be in the same folder):
  python aim1c_dose_level.py --dir D:\\MIMIC_analytic --out D:\\MIMIC_analytic\\aim1c
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from aim1b_dose_response_robust import ADJ, MIN_CELL, varies, within

warnings.filterwarnings("ignore")
LOG = []
NATIVE = {"fentanyl": ("mcg", 0.3), "hydromorphone": ("mg", 20), "morphine": ("mg", 3)}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def prepare(dirp):
    ev = pd.read_csv(os.path.join(dirp, "mimic_opioid_events.csv.gz"))
    st = pd.read_csv(os.path.join(dirp, "mimic_stays.csv.gz"))
    bn = pd.read_csv(os.path.join(dirp, "mimic_bins.csv.gz"))
    st = st[st["core_cohort"] == 1]
    ev = ev[ev["stay_id"].isin(st["stay_id"]) & (ev["mme"] > 0)].copy()
    ev["bin"] = (ev["hrs"] // 6).astype(int)
    ev = ev.merge(bn[["stay_id", "bin", "opioid_infusion"] + [c for c in ADJ if c in bn]], on=["stay_id", "bin"], how="left")
    ev = ev.merge(st[["stay_id", "age_num", "female", "egfr_cat", "surgery_group", "intubated_day1"]], on="stay_id")
    ev["egfr_cat"] = ev["egfr_cat"].fillna("missing")
    ev["intubated_day1"] = ev["intubated_day1"].fillna(0)
    for c in ["opioid_infusion"] + ADJ:
        ev[c] = ev[c].fillna(0) if c in ev else 0
    ev = ev.sort_values(["stay_id", "hrs"])
    ev["stacked"] = (ev.groupby("stay_id")["hrs"].diff() * 60 < 60).astype(int)
    win = {"iv_bolus": (15, 120), "oral": (30, 180)}
    ev["R"] = [int(win[s][0] <= m <= win[s][1]) if pd.notna(m) else 0 for s, m in zip(ev["src"], ev["min_to_pain_reassess"])]
    ind = ev[ev["pain_before"].notna() & (ev["min_from_pain_before"] <= 60) & (ev["pain_before"] >= 4)].copy()
    ind["x"] = np.log1p(ind["mme"])
    ind["delta"] = ind["pain_before"] - ind["pain_after"]
    ind["relief"] = ((ind["delta"] >= 2) | (ind["delta"] >= 0.3 * ind["pain_before"])).astype(float)
    ind.loc[ind["R"] == 0, ["delta", "relief"]] = np.nan
    return ind


def cov_terms(d):
    t = ["cr(pain_before, df=3)", "hrs", "age_num", "female", "C(egfr_cat)", "C(surgery_group)", "intubated_day1",
         "stacked"] + varies(d, ["opioid_infusion"] + ADJ)
    if d["agent"].nunique() > 1:
        t.append("C(agent)")
    return [x for x in t if not x.startswith("C(") or d[x[2:-1]].nunique() > 1]


def ipw(d):
    f = "R ~ cr(x, df=3) + " + " + ".join(cov_terms(d))
    y, X = patsy.dmatrices(f, d, return_type="dataframe")
    p = sm.GLM(y, X, family=sm.families.Binomial()).fit().predict(X)
    w = pd.Series(np.nan, index=d.index)
    w.loc[X.index] = d.loc[X.index, "R"].mean() / p
    return w.clip(upper=w.quantile(0.99))


def contrast(d, y, grid, ref, weights=None, fe=False):
    terms = cov_terms(d)
    if fe:
        d = d[d.groupby("stay_id")["stay_id"].transform("size") >= 2]
        terms = [t for t in terms if t in ["cr(pain_before, df=3)", "hrs", "stacked", "C(agent)", "opioid_infusion"] + ADJ]
    if len(d) < 300:
        return None
    f = f"{y} ~ cr(x, df=4, constraints='center') + " + " + ".join(terms)
    yy, X = patsy.dmatrices(f, d, return_type="dataframe")
    di = X.design_info
    dd = d.loc[X.index]
    g = pd.factorize(dd["stay_id"])[0]
    if fe:
        X = X.drop(columns=["Intercept"])
        Xw = within(pd.concat([X, dd[["stay_id"]]], axis=1), list(X.columns), "stay_id")
        yw = within(pd.concat([yy, dd[["stay_id"]]], axis=1), [y], "stay_id")[y]
        res = sm.OLS(yw, Xw).fit(cov_type="cluster", cov_kwds={"groups": g})
    else:
        w = None if weights is None else weights.loc[X.index].values
        res = (sm.WLS(yy, X, weights=w) if w is not None else sm.OLS(yy, X)).fit(cov_type="cluster", cov_kwds={"groups": g})
    sp = [c for c in X.columns if c.startswith("cr(x")]
    base = dd.iloc[:1].copy()

    def basis(v):
        nd = base.copy()
        nd["x"] = np.log1p(v)
        return patsy.build_design_matrices([di], nd, return_type="dataframe")[0][sp].values[0]
    b, V, r0 = res.params[sp].values, res.cov_params().loc[sp, sp].values, basis(ref)
    rows = []
    for v in grid:
        k = basis(v) - r0
        e, se = k @ b, np.sqrt(max(k @ V @ k, 0))
        rows.append((v, e, e - 1.96 * se, e + 1.96 * se))
    out = pd.DataFrame(rows, columns=["mme", "diff_vs_ref", "lo95", "hi95"])
    out["n_doses"], out["n_stays"] = len(dd), dd["stay_id"].nunique()
    return out


def run(dirp, out):
    ind = prepare(dirp)
    log(f"Indicated doses (pain >= 4, score <= 60 min before): {len(ind):,} | reassessed in window: {int(ind['R'].sum()):,} "
        f"({100 * ind['R'].mean():.1f}%)")
    # who gets reassessed? (aggregate)
    rr = ind.assign(pb=pd.cut(ind["pain_before"], [3.9, 6, 8, 10], labels=["4-6", "7-8", "9-10"])).groupby(
        ["src", "pb"], observed=False)["R"].agg(["size", "mean"]).reset_index()
    rr["mean"] = (100 * rr["mean"]).round(1)
    rr.columns = ["route", "pain_before", "doses", "reassessed_in_window_pct"]
    rr.to_csv(os.path.join(out, "aim1c_reassessment_by_pain.csv"), index=False)

    res = []
    specs = [("iv_bolus", "all", [2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30]),
             ("oral", "all", [2.5, 5, 7.5, 10, 15, 20, 30])]
    for a in NATIVE:
        specs.append(("iv_bolus", a, [2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30]))
    for src, agent, grid in specs:
        d = ind[ind["src"] == src]
        if agent != "all":
            d = d[d["agent"] == agent]
        if len(d) < 500:
            log(f"  [{src}/{agent}] too few doses ({len(d)})")
            continue
        grid = [g for g in grid if d["mme"].quantile(0.01) <= g <= d["mme"].quantile(0.99)] or [7.5]
        if 7.5 not in grid:
            grid = sorted(set(grid) | {7.5})
        d = d.copy()
        d["w"] = ipw(d)
        r = d[d["R"] == 1]
        for y in ["delta", "relief"]:
            for label, kw in [("adjusted", {}), ("adjusted_ipw", {"weights": r["w"]}), ("within_patient", {"fe": True})]:
                c = contrast(r, y, grid, 7.5, **kw)
                if c is None:
                    continue
                c.insert(0, "model", label)
                c.insert(0, "outcome", y)
                c.insert(0, "agent", agent)
                c.insert(0, "route", src)
                if agent in NATIVE:
                    u, f = NATIVE[agent]
                    c[f"dose_{u}"] = (c["mme"] / f).round(2)
                res.append(c)
                top = c.iloc[-1]
                log(f"  [{src}/{agent}] {y} {label}: n={int(top['n_doses']):,} | "
                    + "; ".join(f"{m:g}:{e:+.2f}" for m, e in zip(c["mme"], c["diff_vs_ref"])))
    allc = pd.concat(res)
    allc.round(4).to_csv(os.path.join(out, "aim1c_dose_contrasts.csv"), index=False)

    # descriptive safety table
    s = ind.copy()
    s["overs2h"] = ((s["rass_after"] <= -3) & (s["min_to_rass"] <= 120)).astype(float).where(s["min_to_rass"] <= 120)
    cats = pd.cut(s["mme"], [0, 5, 10, 15, 22.5, np.inf], labels=["<=5", "5-10", "10-15", "15-22.5", ">22.5"])
    t = s.assign(cat=cats).groupby(["src", "cat"], observed=False).agg(
        doses=("stay_id", "size"), rass_within_2h=("overs2h", "count"), oversed_events=("overs2h", "sum")).reset_index()
    t["oversed_pct_of_assessed"] = (100 * t["oversed_events"] / t["rass_within_2h"]).round(2)
    small = t["oversed_events"].between(1, MIN_CELL - 1)
    t.loc[small, ["oversed_events", "oversed_pct_of_assessed"]] = np.nan
    t.to_csv(os.path.join(out, "aim1c_sedation_descriptive.csv"), index=False)
    plot(allc, out)


def plot(c, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    panels = [("iv_bolus", "all", "delta", "IV bolus: extra pain reduction vs 7.5 MME (points)"),
              ("iv_bolus", "all", "relief", "IV bolus: extra meaningful relief vs 7.5 MME (%)"),
              ("oral", "all", "delta", "Oral: extra pain reduction vs 7.5 MME (points)"),
              ("oral", "all", "relief", "Oral: extra meaningful relief vs 7.5 MME (%)")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    for ax, (src, ag, y, ttl) in zip(axes, panels):
        s = c[(c["route"] == src) & (c["agent"] == ag) & (c["outcome"] == y) & (c["model"] == "adjusted_ipw")]
        if s.empty:
            ax.set_visible(False)
            continue
        k = 1 if y == "delta" else 100
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.fill_between(s["mme"], s["lo95"] * k, s["hi95"] * k, color="#2a6fdb", alpha=0.15, linewidth=0)
        ax.plot(s["mme"], s["diff_vs_ref"] * k, color="#2a6fdb", linewidth=2, marker="o", markersize=4)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("MME per dose")
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "aim1c_dose_contrasts.png"), dpi=200)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run(a.dir, a.out)
    with open(os.path.join(a.out, "aim1c_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    print(f"Done. Aggregate results in {a.out}")
