"""
Aim 3 - Opioid-sparing and predictive analyses at the single-dose level (MIMIC-IV).

(A) Non-opioid adjuncts and single-dose relief: does a dose given while a non-opioid adjunct
    (paracetamol, NSAID, gabapentinoid, ketamine or dexmedetomidine) is active in the same 6-h
    interval produce more relief? Within-patient fixed-effects linear model + between-patient model.
(B) Escalation after a failed dose: among doses that follow a previous dose without meaningful
    relief (within 6 h), relief after an escalated dose (>20% more MME) vs the same/lower dose,
    and with vs without an adjunct.
(C) Predicting non-response before the dose: gradient boosting vs logistic regression, 5-fold
    cross-validation grouped by patient; AUROC (patient-bootstrap 95% CI), Brier score,
    calibration intercept/slope, observed relief by risk quintile, permutation importance.

Only aggregate outputs. Needs aim1b_dose_response_robust.py and aim1c_dose_level.py in the
same folder, and scikit-learn (python -m pip install scikit-learn).

Usage:
  python aim3_opioid_sparing_prediction.py --dir D:\\MIMIC_analytic --out D:\\MIMIC_analytic\\aim3
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from aim1b_dose_response_robust import within
from aim1c_dose_level import prepare

warnings.filterwarnings("ignore")
NONOP = ["paracetamol", "nsaid", "gabapentinoid", "ketamine", "dexmedetomidine"]
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def ols_cluster(f, d, fe=False):
    y, X = patsy.dmatrices(f, d, return_type="dataframe")
    dd = d.loc[X.index]
    g = pd.factorize(dd["stay_id"])[0]
    if fe:
        X = X.drop(columns=["Intercept"])
        Xw = within(pd.concat([X, dd[["stay_id"]]], axis=1), list(X.columns), "stay_id")
        yw = within(pd.concat([y, dd[["stay_id"]]], axis=1), [y.columns[0]], "stay_id")[y.columns[0]]
        return sm.OLS(yw, Xw).fit(cov_type="cluster", cov_kwds={"groups": g}), len(dd), dd["stay_id"].nunique()
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g}), len(dd), dd["stay_id"].nunique()


def main(a):
    os.makedirs(a.out, exist_ok=True)
    ind = prepare(a.dir).sort_values(["stay_id", "hrs"])
    ind["adjunct"] = (ind[NONOP].sum(axis=1) > 0).astype(int)
    ind["sedative"] = ((ind["propofol"] + ind["benzodiazepine"]) > 0).astype(int)
    # previous indicated dose in the same patient
    g = ind.groupby("stay_id")
    ind["prev_hrs"] = g["hrs"].shift(1)
    ind["prev_relief"] = g["relief"].shift(1)
    ind["prev_mme"] = g["mme"].shift(1)
    ind["prev_R"] = g["R"].shift(1)
    ind["gap_prev_h"] = ind["hrs"] - ind["prev_hrs"]
    ind["n_prior"] = g.cumcount()
    ind["prev_status"] = np.select(
        [ind["prev_hrs"].isna() | (ind["gap_prev_h"] > 6), ind["prev_R"] != 1, ind["prev_relief"] == 1],
        ["none_6h", "not_reassessed", "relief"], "no_relief")
    r = ind[ind["R"] == 1].copy()
    log(f"Reassessed indicated doses: {len(r):,} in {r['stay_id'].nunique():,} patients; adjunct active for "
        f"{100 * r['adjunct'].mean():.1f}%")

    # ---------------- (A) adjunct and relief
    rows = []
    base = "cr(pain_before, df=3) + cr(x, df=3) + hrs + stacked + C(agent) + opioid_infusion + sedative"
    for src in ["iv_bolus", "oral"]:
        d = r[r["src"] == src]
        for y in ["relief", "delta"]:
            for label, fe, extra in [("between_adjusted", False,
                                      " + age_num + female + C(egfr_cat) + C(surgery_group) + intubated_day1"),
                                     ("within_patient", True, "")]:
                try:
                    res, n, ns = ols_cluster(f"{y} ~ adjunct + {base}{extra}", d, fe)
                    e, (lo, hi) = res.params["adjunct"], res.conf_int().loc["adjunct"]
                    k = 100 if y == "relief" else 1
                    rows.append({"route": src, "outcome": y, "model": label, "estimate": e * k, "lo95": lo * k,
                                 "hi95": hi * k, "p": res.pvalues["adjunct"], "n_doses": n, "n_patients": ns,
                                 "unit": "percentage points" if y == "relief" else "points"})
                    log(f"  (A) {src} {y} {label}: {e * k:+.2f} ({lo * k:+.2f} to {hi * k:+.2f}), n={n:,}")
                except Exception as ex:
                    log(f"  (A) {src} {y} {label}: failed {type(ex).__name__}: {ex}")
    pd.DataFrame(rows).round(4).to_csv(os.path.join(a.out, "aim3A_adjunct_relief.csv"), index=False)

    # ---------------- (B) after a failed dose
    f = r[r["prev_status"] == "no_relief"].copy()
    f["strategy"] = np.select([f["mme"] > 1.2 * f["prev_mme"], f["mme"] < 0.8 * f["prev_mme"]],
                              ["escalated", "reduced"], "same")
    desc = f.groupby(["src", "strategy", "adjunct"]).agg(doses=("relief", "size"), patients=("stay_id", "nunique"),
                                                         relief_pct=("relief", "mean"), delta=("delta", "mean")).reset_index()
    desc["relief_pct"] = (100 * desc["relief_pct"]).round(1)
    desc.loc[desc["doses"] < 11, ["relief_pct", "delta"]] = np.nan
    desc.round(2).to_csv(os.path.join(a.out, "aim3B_after_failed_dose_descriptive.csv"), index=False)
    log(f"(B) doses after a failed dose: {len(f):,}; overall relief {100 * f['relief'].mean():.1f}% vs "
        f"{100 * r.loc[r['prev_status'] == 'relief', 'relief'].mean():.1f}% after a successful dose")
    rowsb = []
    for src in ["iv_bolus", "oral"]:
        d = f[f["src"] == src].copy()
        d["escalated"] = (d["strategy"] == "escalated").astype(int)
        if len(d) < 300:
            continue
        try:
            res, n, ns = ols_cluster("relief ~ escalated + adjunct + cr(pain_before, df=3) + hrs + C(agent) + "
                                     "opioid_infusion + sedative + age_num + female + C(surgery_group) + intubated_day1", d)
            for t in ["escalated", "adjunct"]:
                e, (lo, hi) = res.params[t], res.conf_int().loc[t]
                rowsb.append({"route": src, "term": t, "risk_difference_pp": 100 * e, "lo95": 100 * lo,
                              "hi95": 100 * hi, "p": res.pvalues[t], "n_doses": n, "n_patients": ns})
                log(f"  (B) {src} {t}: {100 * e:+.1f} pp ({100 * lo:+.1f} to {100 * hi:+.1f}), n={n:,}")
        except Exception as ex:
            log(f"  (B) {src}: failed {type(ex).__name__}: {ex}")
    pd.DataFrame(rowsb).round(4).to_csv(os.path.join(a.out, "aim3B_after_failed_dose_models.csv"), index=False)
    rel = r.groupby(["src", "prev_status"]).agg(doses=("relief", "size"), relief_pct=("relief", "mean")).reset_index()
    rel["relief_pct"] = (100 * rel["relief_pct"]).round(1)
    rel.to_csv(os.path.join(a.out, "aim3B_relief_by_previous_response.csv"), index=False)

    # ---------------- (C) prediction of non-response
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.inspection import permutation_importance

    r["no_relief"] = 1 - r["relief"]
    num = ["pain_before", "x", "hrs", "age_num", "n_prior", "gap_prev_h"]
    binf = ["stacked", "opioid_infusion", "female", "intubated_day1"] + NONOP + ["propofol", "benzodiazepine"]
    cat = ["agent", "src", "egfr_cat", "surgery_group", "prev_status"]
    X = r[num + binf + cat].copy()
    X["gap_prev_h"] = X["gap_prev_h"].fillna(24).clip(upper=24)
    X["age_num"] = X["age_num"].fillna(X["age_num"].median())
    X = pd.get_dummies(X, columns=cat, drop_first=False, dtype=float).fillna(0)
    y = r["no_relief"].values
    groups = r["stay_id"].values
    gkf = GroupKFold(n_splits=5)
    models = {"logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
              "gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                          max_leaf_nodes=31, l2_regularization=1.0,
                                                                          random_state=2026)}
    oof = {k: np.zeros(len(y)) for k in models}
    imps = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        for k, mk in models.items():
            m = mk().fit(X.iloc[tr], y[tr])
            oof[k][te] = m.predict_proba(X.iloc[te])[:, 1]
            if k == "gradient_boosting":
                pi = permutation_importance(m, X.iloc[te], y[te], scoring="roc_auc", n_repeats=5, random_state=fold)
                imps.append(pd.Series(pi.importances_mean, index=X.columns))
    rng = np.random.default_rng(2026)
    pats = np.unique(groups)
    idx_by = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array)
    res = []
    for k, p in oof.items():
        auc = roc_auc_score(y, p)
        boots = []
        for _ in range(200):
            s = rng.choice(pats, size=len(pats), replace=True)
            ii = np.concatenate([idx_by[x] for x in s])
            if len(np.unique(y[ii])) == 2:
                boots.append(roc_auc_score(y[ii], p[ii]))
        lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        cal = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
        res.append({"model": k, "auroc": auc, "auroc_lo95": np.percentile(boots, 2.5), "auroc_hi95": np.percentile(boots, 97.5),
                    "brier": brier_score_loss(y, p), "brier_null": brier_score_loss(y, np.full(len(y), y.mean())),
                    "calibration_slope": cal.params[1], "calibration_in_the_large": citl.params[0],
                    "n_doses": len(y), "n_patients": len(pats), "non_response_pct": 100 * y.mean()})
        log(f"(C) {k}: AUROC {auc:.3f} ({np.percentile(boots, 2.5):.3f}-{np.percentile(boots, 97.5):.3f}), "
            f"Brier {brier_score_loss(y, p):.3f}, slope {cal.params[1]:.2f}")
    pd.DataFrame(res).round(4).to_csv(os.path.join(a.out, "aim3C_prediction_performance.csv"), index=False)
    best = max(oof, key=lambda k: roc_auc_score(y, oof[k]))
    q = pd.qcut(oof[best], 5, labels=["Q1 lowest risk", "Q2", "Q3", "Q4", "Q5 highest risk"])
    qt = pd.DataFrame({"q": q, "y": y, "p": oof[best]}).groupby("q", observed=False).agg(
        doses=("y", "size"), predicted_non_response=("p", "mean"), observed_non_response=("y", "mean"))
    qt[["predicted_non_response", "observed_non_response"]] *= 100
    qt["observed_relief_pct"] = 100 - qt["observed_non_response"]
    qt.round(1).to_csv(os.path.join(a.out, f"aim3C_risk_quintiles_{best}.csv"))
    log(qt.round(1).to_string())
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False).head(20)
    imp.round(4).to_csv(os.path.join(a.out, "aim3C_permutation_importance.csv"), header=["mean_auroc_drop"])
    log("Top predictors: " + ", ".join(f"{i} ({v:.3f})" for i, v in imp.head(8).items()))
    plot(oof, y, qt, out=a.out)
    with open(os.path.join(a.out, "aim3_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG))
    log(f"Done. Aggregate results in {a.out}")


def plot(oof, y, qt, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for k, col in [("gradient_boosting", "#0072B2"), ("logistic", "#7f7f7f")]:
        p = oof[k]
        bins = pd.qcut(p, 10, duplicates="drop")
        t = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
        ax.plot(t["p"] * 100, t["y"] * 100, marker="o", ms=4, lw=1.8, color=col, label=k.replace("_", " "))
    ax.plot([0, 100], [0, 100], color="#bbbbbb", lw=1, ls="--")
    ax.set_xlabel("Predicted non-response (%)")
    ax.set_ylabel("Observed non-response (%)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("A  Calibration (cross-validated)", loc="left", fontsize=10)
    ax = axes[1]
    ax.bar(range(len(qt)), qt["observed_relief_pct"], color="#0072B2", alpha=0.8)
    ax.set_xticks(range(len(qt)))
    ax.set_xticklabels([s.split(" ")[0] for s in qt.index])
    ax.set_xlabel("Predicted non-response quintile")
    ax.set_ylabel("Observed meaningful relief (%)")
    ax.set_title("B  Relief by predicted risk", loc="left", fontsize=10)
    for a_ in axes:
        a_.grid(axis="y", color="#e5e5e5", lw=0.8)
        for sp in ["top", "right"]:
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "aim3C_prediction.png"), dpi=300)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
