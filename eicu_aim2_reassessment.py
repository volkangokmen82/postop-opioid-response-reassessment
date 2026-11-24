"""
Aim 2 - Nurse-led pain reassessment across hospitals (eICU-CRD, multicentre).

Questions
  1) After a moderate-severe pain score (>= 4), how quickly do nurses reassess pain?
  2) How much does this vary between hospitals beyond case mix (ICC, median odds ratio,
     case-mix-adjusted hospital rates)?
  3) Do patients treated in hospitals with a stronger reassessment practice have better
     analgesic outcomes (less persistent severe pain, lower pain on days 2-3) without more
     oversedation, and what about ICU length of stay and mortality?

Exposure for (3) is the hospital's leave-one-out reassessment rate (the patient's own episodes
are removed), in quartiles; models are GEE with hospital clusters, adjusted for age, sex, APACHE,
surgery group, intubation on day 1, eGFR category and first-interval pain.

Reads nurseCharting (pain scores) for core-cohort stays only. Only aggregate outputs;
hospital-level tables carry no hospital identifiers; hospitals with < 50 episodes are excluded
from hospital-level estimates.

Usage:
  python eicu_aim2_reassessment.py --eicu D:\\EICU --dir D:\\EICU_analytic --out D:\\EICU_analytic\\aim2
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
H = 72
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def find_table(folder, name):
    for f in os.listdir(folder):
        if f.lower() in (name.lower() + ".csv", name.lower() + ".csv.gz"):
            return os.path.join(folder, f)
    raise FileNotFoundError(name)


def read_pain(eicu, sids):
    parts = []
    cols = ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel",
            "nursingchartcelltypevalname", "nursingchartvalue"]
    for ch in pd.read_csv(find_table(eicu, "nurseCharting"), usecols=cols, chunksize=1_000_000, dtype=str):
        ch["sid"] = pd.to_numeric(ch["patientunitstayid"], errors="coerce")
        ch["off"] = pd.to_numeric(ch["nursingchartoffset"], errors="coerce")
        ch = ch[ch["sid"].isin(sids) & ch["off"].between(0, H * 60 + 360)]
        if ch.empty:
            continue
        p = ch[(ch["nursingchartcelltypevalname"] == "Pain Score")
               & ch["nursingchartcelltypevallabel"].fillna("").str.contains("Pain Score")]
        p = p.assign(v=pd.to_numeric(p["nursingchartvalue"], errors="coerce"))
        parts.append(p.loc[p["v"].between(0, 10), ["sid", "off", "v"]])
    pain = pd.concat(parts).drop_duplicates(["sid", "off"]).sort_values(["sid", "off"])
    return pain


def episodes(pain, thr=60):
    pain = pain.copy()
    pain["next_off"] = pain.groupby("sid")["off"].shift(-1)
    pain["next_v"] = pain.groupby("sid")["v"].shift(-1)
    ep = pain[(pain["v"] >= 4) & (pain["off"] < H * 60)].copy()
    ep["gap"] = ep["next_off"] - ep["off"]
    ep.loc[ep["gap"] > 360, ["gap", "next_v"]] = np.nan          # no reassessment within 6 h
    ep["r60"] = (ep["gap"] <= thr).astype(int)   # "r60" = reassessed within the chosen threshold
    ep["r120"] = (ep["gap"] <= 120).astype(int)
    ep["improved_next"] = (ep["next_v"] < 4).astype(float).where(ep["next_v"].notna())
    ep["pain_band"] = pd.cut(ep["v"], [3.9, 6, 8, 10], labels=["4-6", "7-8", "9-10"]).astype(str)
    ep["t_bin"] = (ep["off"] // 360).astype(int)
    return ep


def main(a):
    os.makedirs(a.out, exist_ok=True)
    st = pd.read_csv(os.path.join(a.dir, "eicu_stays.csv.gz"))
    bn = pd.read_csv(os.path.join(a.dir, "eicu_bins.csv.gz"))
    st = st[st["core_cohort"] == 1].copy()
    sids = set(st["sid"])
    log(f"Core stays: {len(st):,} in {st['hospitalid'].nunique()} hospitals")
    pain = read_pain(a.eicu, sids)
    ep = episodes(pain, a.threshold).merge(st[["sid", "hospitalid", "age_num", "female", "apachescore", "surgery_group",
                                  "intubated_day1", "egfr_cat"]], on="sid")
    for c, f in [("apachescore", ep["apachescore"].median()), ("age_num", ep["age_num"].median())]:
        ep[c] = ep[c].fillna(f)
    ep["intubated_day1"] = ep["intubated_day1"].fillna(0)
    ep["egfr_cat"] = ep["egfr_cat"].fillna("missing")
    log(f"Pain >= 4 episodes: {len(ep):,} in {ep['sid'].nunique():,} stays")

    # ---- 1) overall and by pain band / time
    def summ(g):
        return pd.Series({"episodes": len(g), "stays": g["sid"].nunique(),
                          "reassessed_6h_pct": 100 * g["gap"].notna().mean(),
                          "median_gap_min": g["gap"].median(),
                          "within_60_pct": 100 * g["r60"].mean(), "within_120_pct": 100 * g["r120"].mean(),
                          "improved_at_next_pct": 100 * g["improved_next"].mean()})
    tabs = [summ(ep).to_frame("overall").T]
    for c in ["pain_band", "t_bin", "surgery_group", "intubated_day1"]:
        t = ep.groupby(c).apply(summ)
        t.index = [f"{c}={i}" for i in t.index]
        tabs.append(t)
    t1 = pd.concat(tabs).round(1)
    t1.loc[t1["episodes"] < 11, t1.columns[1:]] = np.nan
    t1.to_csv(os.path.join(a.out, "aim2_reassessment_overall.csv"))
    log(t1.head(4).to_string())

    # ---- 2) between-hospital variation
    hc = ep.groupby("hospitalid").agg(n=("sid", "size"), stays=("sid", "nunique"))
    elig = hc[(hc["n"] >= 50) & (hc["stays"] >= 20)].index
    if a.min_rate > 0:   # sensitivity: drop hospitals whose crude rate suggests fixed charting intervals
        crude = ep.groupby("hospitalid")["r60"].mean()
        elig = [h for h in elig if crude.loc[h] * 100 >= a.min_rate]
    log(f"Settings: threshold={a.threshold} min, min_rate={a.min_rate}%")
    eh = ep[ep["hospitalid"].isin(elig)].copy()
    log(f"Hospitals with >= 50 episodes and >= 20 stays: {len(elig)} ({len(eh):,} episodes)")
    f = ("r60 ~ C(pain_band) + C(t_bin) + age_num + female + apachescore + C(surgery_group) + "
         "intubated_day1 + C(egfr_cat)")
    y, X = patsy.dmatrices(f, eh, return_type="dataframe")
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    eh.loc[X.index, "expected"] = m.predict(X)
    hosp = eh.groupby("hospitalid").agg(episodes=("r60", "size"), stays=("sid", "nunique"),
                                        observed=("r60", "mean"), expected=("expected", "mean"),
                                        median_gap=("gap", "median"))
    overall = eh["r60"].mean()
    hosp["adjusted_rate"] = (hosp["observed"] / hosp["expected"] * overall).clip(upper=1)
    hosp_out = hosp.reset_index(drop=True).sort_values("adjusted_rate").reset_index(drop=True)
    hosp_out.index.name = "hospital_rank"
    (hosp_out[["episodes", "stays", "observed", "adjusted_rate", "median_gap"]] * [1, 1, 100, 100, 1]).round(1).to_csv(
        os.path.join(a.out, "aim2_hospital_rates_anonymised.csv"))
    q = hosp["adjusted_rate"].quantile([0, .1, .25, .5, .75, .9, 1]) * 100
    log("Case-mix-adjusted hospital reassessment within 60 min (%): " + ", ".join(f"p{int(k*100)}={v:.1f}" for k, v in q.items()))

    # random-intercept logistic model (variational Bayes) -> ICC and median odds ratio
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        eh2 = eh.loc[X.index].copy()
        eh2["h"] = pd.factorize(eh2["hospitalid"])[0]
        mm = BinomialBayesMixedGLM.from_formula(f, {"h": "0 + C(h)"}, eh2).fit_vb()
        sd = float(np.exp(mm.vcp_mean[0]))
        var = sd ** 2
        icc = var / (var + np.pi ** 2 / 3)
        mor = float(np.exp(np.sqrt(2 * var) * 0.6745))
        log(f"Random-intercept logistic: hospital SD={sd:.3f}, ICC={icc:.3f}, median odds ratio={mor:.2f}")
        pd.DataFrame([{"hospital_sd_logit": sd, "icc": icc, "median_odds_ratio": mor, "hospitals": len(elig),
                       "episodes": len(eh2)}]).round(4).to_csv(os.path.join(a.out, "aim2_variation.csv"), index=False)
    except Exception as e:
        log(f"mixed model failed: {type(e).__name__}: {e}")

    # ---- 3) hospital practice (leave-one-out) and patient outcomes
    s_h = eh.groupby("hospitalid")["r60"].agg(["sum", "size"])
    s_i = eh.groupby("sid").agg(hid=("hospitalid", "first"), si=("r60", "sum"), ni=("r60", "size"))
    s_i = s_i.join(s_h, on="hid")
    s_i["loo_rate"] = (s_i["sum"] - s_i["si"]) / (s_i["size"] - s_i["ni"])
    hosp_rate = hosp["observed"]
    cut = hosp_rate.quantile([.25, .5, .75]).values
    pt = st[st["hospitalid"].isin(elig)].copy()
    pt = pt.merge(s_i[["loo_rate"]], left_on="sid", right_index=True, how="left")
    pt["loo_rate"] = pt["loo_rate"].fillna(pt["hospitalid"].map(hosp_rate))
    pt["practice_q"] = pd.cut(pt["loo_rate"], [-1, *cut, 2], labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]).astype(str)
    pt["practice_10pct"] = pt["loo_rate"] * 10        # per 10 percentage points

    g = bn[bn["sid"].isin(pt["sid"])]
    late = g[g["bin"] >= 4]
    o = pd.DataFrame({
        "persistent_severe": late.groupby("sid")["pain_max"].max().ge(7).astype(float).where(late.groupby("sid")["pain_max"].max().notna()),
        "pain_mean_24_72": late.groupby("sid")["pain_mean"].mean(),
        "oversed_12_72": g[g["bin"] >= 2].groupby("sid")["oversedation"].max(),
        "pain_first": g[g["bin"] == 0].set_index("sid")["pain_mean"]})
    pt = pt.merge(o, left_on="sid", right_index=True, how="left")
    pt["log_los"] = np.log(pt["icu_los_h"].clip(lower=1))
    pt["pain_first"] = pt["pain_first"].fillna(pt["pain_first"].median())
    pt["apachescore"] = pt["apachescore"].fillna(pt["apachescore"].median())
    pt["intubated_day1"] = pt["intubated_day1"].fillna(0)
    pt["egfr_cat"] = pt["egfr_cat"].fillna("missing")
    desc = pt.groupby("practice_q").agg(stays=("sid", "size"), hospitals=("hospitalid", "nunique"),
                                        practice_rate=("loo_rate", "mean"),
                                        persistent_severe=("persistent_severe", "mean"),
                                        pain_mean_24_72=("pain_mean_24_72", "mean"),
                                        oversed_12_72=("oversed_12_72", "mean"),
                                        icu_los_h_median=("icu_los_h", "median"), hosp_death=("hosp_death", "mean"))
    for c in ["practice_rate", "persistent_severe", "oversed_12_72", "hosp_death"]:
        desc[c] = 100 * desc[c]
    desc.round(2).to_csv(os.path.join(a.out, "aim2_outcomes_by_practice_quartile.csv"))
    log(desc.round(2).to_string())

    adj = "age_num + female + apachescore + C(surgery_group) + intubated_day1 + C(egfr_cat) + pain_first"
    rows = []
    for y, fam in [("persistent_severe", sm.families.Binomial()), ("pain_mean_24_72", sm.families.Gaussian()),
                   ("oversed_12_72", sm.families.Binomial()), ("hosp_death", sm.families.Binomial()),
                   ("log_los", sm.families.Gaussian())]:
        d = pt.dropna(subset=[y])
        for expo in ["practice_10pct", "C(practice_q, Treatment('Q1 (lowest)'))"]:
            try:
                r = smf.gee(f"{y} ~ {expo} + {adj}", groups="hospitalid", data=d, family=fam,
                            cov_struct=sm.cov_struct.Exchangeable()).fit()
                for k in [k for k in r.params.index if "practice" in k]:
                    e, lo, hi = r.params[k], *r.conf_int().loc[k]
                    tr = np.exp if isinstance(fam, sm.families.Binomial) or y == "log_los" else (lambda z: z)
                    rows.append({"outcome": y, "term": k, "estimate": tr(e), "lo95": tr(lo), "hi95": tr(hi),
                                 "p": r.pvalues[k], "scale": "OR" if isinstance(fam, sm.families.Binomial) else
                                 ("ratio of geometric means" if y == "log_los" else "difference (points)"),
                                 "n": len(d), "hospitals": d["hospitalid"].nunique()})
            except Exception as e:
                log(f"  {y}/{expo}: failed {type(e).__name__}: {e}")
    res = pd.DataFrame(rows)
    res.round(4).to_csv(os.path.join(a.out, "aim2_practice_outcome_models.csv"), index=False)
    log(res[res["term"] == "practice_10pct"].round(3).to_string(index=False))

    plot(hosp, a.out)
    with open(os.path.join(a.out, "aim2_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG))
    log(f"Done. Aggregate results in {a.out}")


def plot(hosp, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    h = hosp.sort_values("adjusted_rate")
    se = np.sqrt(h["adjusted_rate"] * (1 - h["adjusted_rate"]) / h["episodes"])
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(h))
    ax.errorbar(x, h["adjusted_rate"] * 100, yerr=1.96 * se * 100, fmt="o", color="#2a6fdb", markersize=4,
                ecolor="#a9c3ef", elinewidth=1)
    ax.axhline(h["observed"].mean() * 100, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel("Hospitals (ranked)")
    ax.set_ylabel("Reassessed within 60 min (%)")
    ax.set_title("Case-mix-adjusted pain reassessment after a score >= 4, by hospital", fontsize=10)
    ax.set_xticks([])
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "aim2_hospital_caterpillar.png"), dpi=200)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eicu", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=int, default=60)
    p.add_argument("--min_rate", type=float, default=0)
    main(p.parse_args())
