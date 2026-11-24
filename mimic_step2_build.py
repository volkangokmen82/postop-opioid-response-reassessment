"""
MIMIC-IV (v3.1) - Step 2: build the analytic dataset (same structure as eICU step 2)
Unit of analysis: postoperative ICU stay x 6-hour interval (first 72 h; 12 intervals)

Opioid exposure in MIMIC-IV = ADMINISTERED doses:
  - parenteral opioids (boluses and infusions): icu/inputevents amounts, spread over start-end time
  - oral/enteral opioids: hosp/emar 'Administered' events with dose_given from hosp/emar_detail
  (IV doses in eMAR are not used during the ICU stay, to avoid double counting with inputevents)
  All expressed as ORAL MME with the same factors as the eICU script.
Nurse reassessment: for each opioid bolus / oral dose, minutes to the next nurse pain assessment
  (Pain Level 223791 or Pain Level Response 224409) and to the next RASS (228096).

Outputs (in --out):
  PATIENT-LEVEL (keep on your computer, never share):
    mimic_stays.csv.gz, mimic_bins.csv.gz, mimic_opioid_events.csv.gz
  AGGREGATE (safe to share):
    step2_summary.txt, table1.csv, bins_summary.csv, reassessment_summary.csv

Usage (Windows):
    cd /d D:\\MIMICIV
    python mimic_step2_build.py --mimic D:\\MIMICIV --out D:\\MIMIC_analytic
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

H, BIN = 72, 6
NBIN = H // BIN
SURG_ICUS = re.compile(r"SICU|Surgical|CVICU|Cardiac Vascular|Trauma", re.I)
SURG_SERVICES = {"SURG", "CSURG", "NSURG", "ORTHO", "PSURG", "TSURG", "VSURG", "GU", "ENT", "GYN", "TRAUM"}
SERVICE_GROUP = {"CSURG": "Cardiovascular", "VSURG": "Cardiovascular", "NSURG": "Neurology",
                 "SURG": "Gastrointestinal", "TSURG": "Respiratory", "ORTHO": "Musculoskeletal/Skin",
                 "TRAUM": "Trauma", "GU": "Genitourinary", "PSURG": "Other", "ENT": "Other", "GYN": "Genitourinary"}
PAIN_IDS = {223791}
PAIN_RESP_IDS = {224409}
RASS_IDS = {228096}
WEIGHT_IDS = {226512, 224639}
VENT_PROC = {225792}  # invasive ventilation (procedureevents)
LAB_IDS = {50912: "creatinine", 50885: "bilirubin", 50861: "alt", 50862: "albumin"}

ORAL_F = {"morphine": 1, "oxycodone": 1.5, "hydrocodone": 1, "hydromorphone": 5, "tramadol": 0.2,
          "codeine": 0.15, "tapentadol": 0.4, "methadone": 4.7, "meperidine": 0.1}
PAR_F = {"morphine": 3, "hydromorphone": 20, "fentanyl": 0.3, "meperidine": 0.4}
MAX_RATE = {"fentanyl": 400, "morphine": 20, "hydromorphone": 5, "meperidine": 100}  # per hour (mcg fentanyl, mg others)
MAX_ORAL = {"oxycodone": 40, "hydrocodone": 20, "tramadol": 200, "morphine": 60, "hydromorphone": 8,
            "tapentadol": 200, "codeine": 120, "methadone": 40}
ADJ = {
    "paracetamol": r"acetaminophen|paracetamol|tylenol|ofirmev",
    "nsaid": r"ketorolac|toradol|ibuprofen|celecoxib|diclofenac|naproxen|meloxicam",
    "gabapentinoid": r"gabapentin|pregabalin",
    "ketamine": r"ketamine",
    "dexmedetomidine": r"dexmedetomidine|precedex",
    "propofol": r"propofol",
    "benzodiazepine": r"midazolam|lorazepam|diazepam|alprazolam|clonazepam",
    "antipsychotic": r"haloperidol|quetiapine|olanzapine|risperidone",
    "naloxone": r"naloxone|narcan",
}
ORAL_ROUTE = re.compile(r"\bpo\b|oral|\bng\b|\bog\b|tube|\bpeg\b|\bj\b|\bsl\b|sublingual|\bpr\b", re.I)


def find_table(root, name):
    t = {name.lower() + ".csv", name.lower() + ".csv.gz"}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower() in t:
                return os.path.join(dp, f)
    raise FileNotFoundError(f"{name}.csv(.gz) not found under {root}")


def agent_of(name):
    n = str(name).lower()
    for a, rx in [("oxycodone", r"oxycodone|percocet|roxicodone"), ("hydrocodone", r"hydrocodone|norco|vicodin"),
                  ("hydromorphone", r"hydromorphone|dilaudid"), ("fentanyl", r"fentanyl"), ("morphine", r"morphine"),
                  ("meperidine", r"meperidine|demerol"), ("tramadol", r"tramadol"), ("methadone", r"methadone"),
                  ("tapentadol", r"tapentadol"), ("codeine", r"codeine")]:
        if re.search(rx, n):
            return a
    return None


def egfr_ckdepi2021(scr, age, female):
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    r = scr / k
    return 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.2 * 0.9938 ** age * np.where(female, 1.012, 1)


def spread(df, t0, t1, val):
    """spread df[val] over [t0, t1) hours (relative to ICU admission) into 6-h bins -> rows (stay_id, bin, value)"""
    out = []
    edges = np.arange(NBIN + 1) * BIN
    for s, a, b, v in zip(df["stay_id"], df[t0], df[t1], df[val]):
        a, b = max(a, 0.0), min(max(b, a + 1 / 60), H)
        if b <= 0 or a >= H:
            continue
        dur = b - a
        lo = np.clip(a, edges[:-1], edges[1:])
        hi = np.clip(b, edges[:-1], edges[1:])
        ov = np.maximum(hi - lo, 0)
        for k in np.nonzero(ov)[0]:
            out.append((s, int(k), v * ov[k] / dur))
    return out


def main(root, out):
    os.makedirs(out, exist_ok=True)
    L = []

    def log(m=""):
        print(m)
        L.append(str(m))

    # ---------- cohort
    icu = pd.read_csv(find_table(root, "icustays"),
                      usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime", "los"],
                      parse_dates=["intime", "outtime"]).sort_values("intime")
    tr = pd.read_csv(find_table(root, "transfers"), usecols=["hadm_id", "careunit", "intime"], parse_dates=["intime"])
    pacu = tr[tr["careunit"].fillna("").str.contains("PACU", case=False)].rename(columns={"intime": "pacu_time"})
    m = icu.merge(pacu[["hadm_id", "pacu_time"]], on="hadm_id", how="left")
    pacu_stays = set(m.loc[((m["intime"] - m["pacu_time"]).dt.total_seconds() / 3600).between(0, 24), "stay_id"])
    sv = pd.read_csv(find_table(root, "services"), usecols=["hadm_id", "transfertime", "curr_service"],
                     parse_dates=["transfertime"])
    ms = icu.merge(sv, on="hadm_id")
    ms = ms[ms["transfertime"] <= ms["intime"] + pd.Timedelta(hours=6)].sort_values("transfertime")
    svc = ms.drop_duplicates("stay_id", keep="last").set_index("stay_id")["curr_service"]
    surg_svc = set(svc[svc.isin(SURG_SERVICES)].index)
    surg_icu = set(icu.loc[icu["first_careunit"].fillna("").str.contains(SURG_ICUS), "stay_id"])
    icu["postop"] = icu["stay_id"].isin(pacu_stays) | (icu["stay_id"].isin(surg_svc) & icu["stay_id"].isin(surg_icu))
    st = icu[icu["postop"]].drop_duplicates("subject_id", keep="first").copy()

    pts = pd.read_csv(find_table(root, "patients"), usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
    adm = pd.read_csv(find_table(root, "admissions"), usecols=["hadm_id", "admittime", "hospital_expire_flag"],
                      parse_dates=["admittime"])
    st = st.merge(pts, on="subject_id").merge(adm, on="hadm_id")
    st["age_num"] = (st["anchor_age"] + (st["admittime"].dt.year - st["anchor_year"])).clip(upper=91)
    st["female"] = (st["gender"] == "F").astype(int)
    st["icu_los_h"] = st["los"] * 24
    st = st[(st["age_num"] >= 18) & (st["icu_los_h"] >= 6)]
    st["surgery_group"] = st["stay_id"].map(svc).map(SERVICE_GROUP).fillna("unknown")
    st["hosp_death"] = st["hospital_expire_flag"].astype(int)
    sids = set(st["stay_id"])
    hadms = set(st["hadm_id"])
    intime = st.set_index("stay_id")["intime"]
    hadm2stay = st.set_index("hadm_id")["stay_id"]
    log(f"Adult postoperative stays (first per patient, ICU >= 6 h): {len(sids):,}")

    # ---------- chartevents: pain, pain response, RASS, weight
    ids = PAIN_IDS | PAIN_RESP_IDS | RASS_IDS | WEIGHT_IDS
    parts = []
    for ch in pd.read_csv(find_table(root, "chartevents"), usecols=["stay_id", "itemid", "charttime", "value", "valuenum"],
                          chunksize=2_000_000, dtype={"value": str}):
        ch = ch[ch["itemid"].isin(ids) & ch["stay_id"].isin(sids)]
        if len(ch):
            parts.append(ch)
    ce = pd.concat(parts)
    ce["hrs"] = (pd.to_datetime(ce["charttime"]) - ce["stay_id"].map(intime)).dt.total_seconds() / 3600
    lead = pd.to_numeric(ce["value"].str.extract(r"^\s*(-?\d+)")[0], errors="coerce")
    ce["v"] = ce["valuenum"].where(ce["valuenum"].notna(), lead)
    w = ce[ce["itemid"].isin(WEIGHT_IDS) & ce["v"].between(30, 250)].sort_values("hrs").drop_duplicates("stay_id")
    st["weight"] = st["stay_id"].map(w.set_index("stay_id")["v"])
    ce = ce[ce["hrs"].between(0, H, inclusive="left")]
    pain = ce[ce["itemid"].isin(PAIN_IDS) & ce["v"].between(0, 10)][["stay_id", "hrs", "v"]]
    presp = ce[ce["itemid"].isin(PAIN_RESP_IDS) & ce["v"].between(0, 10)][["stay_id", "hrs", "v"]]
    rass = ce[ce["itemid"].isin(RASS_IDS) & ce["v"].between(-5, 4)][["stay_id", "hrs", "v"]]
    assess = pd.concat([pain, presp])  # any nurse numeric pain assessment
    log(f"Pain Level: {len(pain):,} | Pain Level Response: {len(presp):,} | RASS: {len(rass):,}")

    # ---------- ventilation day 1 (procedureevents)
    try:
        pe = pd.read_csv(find_table(root, "procedureevents"), usecols=["stay_id", "itemid", "starttime"])
        pe = pe[pe["itemid"].isin(VENT_PROC) & pe["stay_id"].isin(sids)]
        pe["hrs"] = (pd.to_datetime(pe["starttime"]) - pe["stay_id"].map(intime)).dt.total_seconds() / 3600
        st["intubated_day1"] = st["stay_id"].isin(set(pe.loc[pe["hrs"] < 24, "stay_id"])).astype(int)
    except FileNotFoundError:
        st["intubated_day1"] = np.nan

    # ---------- labs (closest to ICU admission, -24..+24 h)
    lparts = []
    for ch in pd.read_csv(find_table(root, "labevents"), usecols=["hadm_id", "itemid", "charttime", "valuenum"],
                          chunksize=2_000_000):
        ch = ch[ch["itemid"].isin(LAB_IDS) & ch["hadm_id"].isin(hadms)]
        if len(ch):
            lparts.append(ch)
    lab = pd.concat(lparts)
    lab["stay_id"] = lab["hadm_id"].map(hadm2stay)
    lab["hrs"] = (pd.to_datetime(lab["charttime"]) - lab["stay_id"].map(intime)).dt.total_seconds() / 3600
    lab = lab[lab["hrs"].between(-24, 24) & lab["valuenum"].notna()].assign(a=lambda d: d["hrs"].abs())
    lab = lab.sort_values("a").drop_duplicates(["stay_id", "itemid"])
    lw = lab.pivot(index="stay_id", columns="itemid", values="valuenum").rename(columns=LAB_IDS)
    st = st.merge(lw, left_on="stay_id", right_index=True, how="left")
    st["egfr"] = egfr_ckdepi2021(st["creatinine"].clip(0.2, 15), st["age_num"], st["female"] == 1)
    st["egfr_cat"] = pd.cut(st["egfr"], [-np.inf, 30, 60, 90, np.inf], labels=["<30", "30-59", "60-89", ">=90"])

    # ---------- inputevents: parenteral opioids (true amounts) and adjunct infusions
    items = pd.read_csv(find_table(root, "d_items"), usecols=["itemid", "label"])
    ie = pd.read_csv(find_table(root, "inputevents"),
                     usecols=["stay_id", "itemid", "starttime", "endtime", "amount", "amountuom",
                              "ordercategoryname", "statusdescription"])
    ie = ie[ie["stay_id"].isin(sids)].merge(items, on="itemid")
    ie = ie[ie["statusdescription"].fillna("") != "Rewritten"]
    ie["h0"] = (pd.to_datetime(ie["starttime"]) - ie["stay_id"].map(intime)).dt.total_seconds() / 3600
    ie["h1"] = (pd.to_datetime(ie["endtime"]) - ie["stay_id"].map(intime)).dt.total_seconds() / 3600
    ie = ie[(ie["h1"] > 0) & (ie["h0"] < H)]
    oie = ie[ie["label"].map(agent_of).notna()].copy()
    oie["agent"] = oie["label"].map(agent_of)
    u = oie["amountuom"].fillna("").str.lower()
    oie["amt"] = np.where(oie["agent"] == "fentanyl",
                          np.where(u == "mg", oie["amount"] * 1000, np.where(u.isin(["mcg", "ug"]), oie["amount"], np.nan)),
                          np.where(u == "mg", oie["amount"], np.where(u.isin(["mcg", "ug"]), oie["amount"] / 1000, np.nan)))
    dur_h = (oie["h1"] - oie["h0"]).clip(lower=1 / 60)
    oie["bolus"] = (dur_h <= 2 / 60) | oie["ordercategoryname"].fillna("").str.contains("Bolus|Push", case=False)
    rate = oie["amt"] / dur_h
    bad = (~oie["bolus"]) & (rate > oie["agent"].map(MAX_RATE))
    bad_bolus = oie["bolus"] & (oie["amt"] > oie["agent"].map({"fentanyl": 200, "morphine": 20, "hydromorphone": 4,
                                                                  "meperidine": 100}))
    log(f"Parenteral opioid records: {len(oie):,} | boluses: {int(oie['bolus'].sum()):,} | "
        f"implausible rate/dose excluded: {int((bad | bad_bolus).sum()):,}")
    oie = oie[~(bad | bad_bolus) & oie["amt"].notna() & (oie["amt"] > 0)]
    oie["mme"] = oie["amt"] * oie["agent"].map(PAR_F)
    oie = oie[oie["mme"].notna()]
    rows_bolus = spread(oie[oie["bolus"]], "h0", "h1", "mme")
    rows_inf = spread(oie[~oie["bolus"]], "h0", "h1", "mme")

    flags = []
    for k, rx in ADJ.items():
        sub = ie[ie["label"].str.contains(rx, case=False, na=False)]
        for s, a, b in zip(sub["stay_id"], sub["h0"], sub["h1"]):
            for kk in range(max(int(a // BIN), 0), min(int(max(b, a) // BIN), NBIN - 1) + 1):
                flags.append((s, kk, k))

    # ---------- eMAR: oral opioids with dose_given; adjuncts and naloxone
    emar = pd.read_csv(find_table(root, "emar"), usecols=["emar_id", "hadm_id", "charttime", "medication", "event_txt"])
    emar = emar[emar["hadm_id"].isin(hadms) & emar["event_txt"].fillna("").str.startswith("Administered")]
    emar["stay_id"] = emar["hadm_id"].map(hadm2stay)
    emar["hrs"] = (pd.to_datetime(emar["charttime"]) - emar["stay_id"].map(intime)).dt.total_seconds() / 3600
    emar = emar[emar["hrs"].between(0, H, inclusive="left")]
    for k, rx in ADJ.items():
        sub = emar[emar["medication"].str.contains(rx, case=False, na=False)]
        flags += [(s, int(h // BIN), k) for s, h in zip(sub["stay_id"], sub["hrs"])]
    oem = emar[emar["medication"].map(agent_of).notna()].copy()
    oem["agent"] = oem["medication"].map(agent_of)
    oem["emar_id"] = oem["emar_id"].astype(str)
    keep_ids = set(oem["emar_id"])
    dparts = []
    for ch in pd.read_csv(find_table(root, "emar_detail"),
                          usecols=["emar_id", "parent_field_ordinal", "dose_given", "dose_given_unit", "route"],
                          chunksize=2_000_000, dtype=str):
        ch = ch[ch["emar_id"].isin(keep_ids)]
        if len(ch):
            dparts.append(ch)
    det = pd.concat(dparts) if dparts else pd.DataFrame(
        columns=["emar_id", "parent_field_ordinal", "dose_given", "dose_given_unit", "route"])
    # In MIMIC-IV the parent row (no parent_field_ordinal) usually carries the route, while the
    # child rows (1.1, 1.2 ...) carry dose_given per product. Combine both levels per emar_id.
    det["is_child"] = det["parent_field_ordinal"].notna()
    det["dose"] = pd.to_numeric(det["dose_given"].fillna("").str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    du = det["dose_given_unit"].fillna("").str.strip().str.lower()
    det["mg"] = np.where(du == "mg", det["dose"], np.where(du.isin(["mcg", "ug"]), det["dose"] / 1000, np.nan))
    route = (det.dropna(subset=["route"]).sort_values("is_child")
             .drop_duplicates("emar_id").set_index("emar_id")["route"])
    mg_child = det[det["is_child"]].groupby("emar_id")["mg"].sum(min_count=1)
    mg_parent = det[~det["is_child"]].groupby("emar_id")["mg"].sum(min_count=1)
    oem["route"] = oem["emar_id"].map(route)
    oem["mg"] = oem["emar_id"].map(mg_child).fillna(oem["emar_id"].map(mg_parent))
    # aggregate-only diagnostic of how routes / units were recorded (no patient-level content)
    diag = pd.concat([
        oem["route"].fillna("<missing>").value_counts().head(25).rename_axis("value").reset_index().assign(field="route"),
        du[det["emar_id"].isin(keep_ids)].replace("", "<missing>").value_counts().head(25)
          .rename_axis("value").reset_index().assign(field="dose_given_unit")])
    diag.to_csv(os.path.join(out, "emar_route_unit_diagnostic.csv"), index=False)
    # agents that exist only as enteral products in ICU practice count as oral when the route is missing
    ORAL_ONLY = {"oxycodone", "hydrocodone", "tramadol", "tapentadol", "codeine", "methadone"}
    oem.loc[oem["route"].isna() & oem["agent"].isin(ORAL_ONLY), "route"] = "PO (assumed)"
    oem["oral"] = oem["route"].fillna("").str.contains(ORAL_ROUTE)
    oral = oem[oem["oral"] & oem["mg"].notna()].copy()
    oral = oral[oral["mg"] <= oral["agent"].map(MAX_ORAL).fillna(np.inf)]
    oral["mme"] = oral["mg"] * oral["agent"].map(ORAL_F)
    oral = oral[oral["mme"].notna()]
    log(f"eMAR opioid administrations: {len(oem):,} | oral with usable dose: {len(oral):,}")
    rows_oral = [(s, int(h // BIN), v) for s, h, v in zip(oral["stay_id"], oral["hrs"], oral["mme"])]

    # ---------- nurse reassessment after each opioid dose (bolus or oral)
    ev = pd.concat([oie.loc[oie["bolus"], ["stay_id", "h0", "agent", "mme"]].rename(columns={"h0": "hrs"}).assign(src="iv_bolus"),
                    oral[["stay_id", "hrs", "agent", "mme"]].assign(src="oral")])
    ev = ev[ev["hrs"].between(0, H, inclusive="left")].sort_values("hrs")
    a1 = assess.sort_values("hrs").rename(columns={"hrs": "t_pain", "v": "pain_after"})
    a0 = assess.sort_values("hrs").rename(columns={"hrs": "t_before", "v": "pain_before"})
    r1 = rass.sort_values("hrs").rename(columns={"hrs": "t_rass", "v": "rass_after"})
    ev = pd.merge_asof(ev, a1.assign(t_pain_key=a1["t_pain"]), left_on="hrs", right_on="t_pain_key", by="stay_id",
                       direction="forward", allow_exact_matches=False, tolerance=6.0)
    ev = pd.merge_asof(ev.sort_values("hrs"), a0.assign(k=a0["t_before"]), left_on="hrs", right_on="k", by="stay_id",
                       direction="backward", tolerance=4.0)
    ev = pd.merge_asof(ev.sort_values("hrs"), r1.assign(k2=r1["t_rass"]), left_on="hrs", right_on="k2", by="stay_id",
                       direction="forward", allow_exact_matches=False, tolerance=6.0)
    ev["min_to_pain_reassess"] = (ev["t_pain"] - ev["hrs"]) * 60
    ev["min_to_rass"] = (ev["t_rass"] - ev["hrs"]) * 60
    ev["min_from_pain_before"] = (ev["hrs"] - ev["t_before"]) * 60
    ev = ev[["stay_id", "hrs", "agent", "src", "mme", "pain_before", "min_from_pain_before", "min_to_pain_reassess", "pain_after",
             "min_to_rass", "rass_after"]]
    ev.to_csv(os.path.join(out, "mimic_opioid_events.csv.gz"), index=False)
    rs = ev.groupby("src").agg(events=("stay_id", "size"), stays=("stay_id", "nunique"),
                               reassessed_6h_pct=("min_to_pain_reassess", lambda s: 100 * s.notna().mean()),
                               median_min=("min_to_pain_reassess", "median"),
                               within_60_pct=("min_to_pain_reassess", lambda s: 100 * (s <= 60).mean()),
                               within_120_pct=("min_to_pain_reassess", lambda s: 100 * (s <= 120).mean())).round(1)
    rs.to_csv(os.path.join(out, "reassessment_summary.csv"))
    log("\nNurse pain reassessment after opioid doses (Pain Level or Pain Level Response, max 6 h):")
    log(rs.to_string())

    # ---------- assemble stay x bin
    grid = pd.MultiIndex.from_product([sorted(sids), range(NBIN)], names=["stay_id", "bin"]).to_frame(index=False)
    grid = grid[grid["bin"] * BIN < grid["stay_id"].map(st.set_index("stay_id")["icu_los_h"])]
    for df, nm in [(pain, "pain"), (rass, "rass"), (assess, "assess")]:
        d = df.assign(bin=(df["hrs"] // BIN).astype(int))
        agg = d.groupby(["stay_id", "bin"])["v"].agg(["mean", "max", "min", "count"])
        agg.columns = [f"{nm}_{c}" for c in agg.columns]
        if nm == "assess":
            agg = agg[["assess_count"]]
        grid = grid.merge(agg.reset_index(), on=["stay_id", "bin"], how="left")
    for rows, nm in [(rows_bolus, "mme_iv_bolus"), (rows_inf, "mme_inf"), (rows_oral, "mme_oral")]:
        d = pd.DataFrame(rows, columns=["stay_id", "bin", nm]).groupby(["stay_id", "bin"], as_index=False).sum()
        grid = grid.merge(d, on=["stay_id", "bin"], how="left")
        grid[nm] = grid[nm].fillna(0.0)
    grid["mme_admin"] = grid["mme_iv_bolus"] + grid["mme_inf"] + grid["mme_oral"]
    nd = ev.assign(bin=(ev["hrs"] // BIN).astype(int)).groupby(["stay_id", "bin"]).agg(
        n_opioid_doses=("hrs", "size"), median_min_reassess=("min_to_pain_reassess", "median")).reset_index()
    grid = grid.merge(nd, on=["stay_id", "bin"], how="left")
    grid["n_opioid_doses"] = grid["n_opioid_doses"].fillna(0).astype(int)
    fl = pd.DataFrame(flags, columns=["stay_id", "bin", "flag"]).drop_duplicates()
    fl = fl[fl["bin"].between(0, NBIN - 1)]
    fw = pd.crosstab([fl["stay_id"], fl["bin"]], fl["flag"]).clip(upper=1).reset_index()
    grid = grid.merge(fw, on=["stay_id", "bin"], how="left")
    for c in ADJ:
        if c not in grid:
            grid[c] = 0
        grid[c] = grid[c].fillna(0).astype(int)
    grid["opioid_infusion"] = (grid["mme_inf"] > 0).astype(int)
    grid["oversedation"] = (grid["rass_min"] <= -3).astype(float).where(grid["rass_min"].notna())
    grid["severe_pain"] = (grid["pain_max"] >= 7).astype(float).where(grid["pain_max"].notna())

    g = grid.groupby("stay_id")
    st = st.merge(g.agg(n_pain=("pain_count", "sum"), n_rass=("rass_count", "sum"),
                        any_severe_pain=("severe_pain", "max"), any_oversed=("oversedation", "max"),
                        mme_admin_72h=("mme_admin", "sum"), mme_inf_72h=("mme_inf", "sum"),
                        any_naloxone=("naloxone", "max")).reset_index(), on="stay_id", how="left")
    st["any_opioid"] = (st["mme_admin_72h"] > 0).astype(int)
    early = grid[grid["bin"] <= 1].groupby("stay_id")[["paracetamol", "nsaid", "gabapentinoid", "ketamine",
                                                       "dexmedetomidine"]].max()
    early["multimodal_0_12h"] = (early.sum(axis=1) >= 1).astype(int)
    st = st.merge(early.rename(columns={c: c + "_0_12h" for c in early.columns if c != "multimodal_0_12h"}),
                  left_on="stay_id", right_index=True, how="left")
    st["core_cohort"] = ((st["n_pain"] >= 3) & (st["any_opioid"] == 1)).astype(int)
    keep = ["stay_id", "age_num", "female", "weight", "first_careunit", "surgery_group", "intubated_day1",
            "creatinine", "egfr", "egfr_cat", "bilirubin", "alt", "albumin", "icu_los_h", "hosp_death", "n_pain",
            "n_rass", "any_severe_pain", "any_oversed", "mme_admin_72h", "mme_inf_72h", "any_naloxone", "any_opioid",
            "paracetamol_0_12h", "nsaid_0_12h", "gabapentinoid_0_12h", "ketamine_0_12h", "dexmedetomidine_0_12h",
            "multimodal_0_12h", "core_cohort"]
    st = st[[c for c in keep if c in st.columns]]
    st.to_csv(os.path.join(out, "mimic_stays.csv.gz"), index=False)
    grid.to_csv(os.path.join(out, "mimic_bins.csv.gz"), index=False)

    # ---------- aggregate outputs
    core = st[st["core_cohort"] == 1]
    log(f"\nStays: {len(st):,} | core cohort (>=3 pain scores + administered opioid): {len(core):,}")

    def desc(df, name):
        r = {"group": name, "n": len(df), "age_median": df["age_num"].median(), "female_%": 100 * df["female"].mean(),
             "intubated_day1_%": 100 * df["intubated_day1"].mean(), "egfr_<60_%": 100 * (df["egfr"] < 60).mean(),
             "creatinine_available_%": 100 * df["creatinine"].notna().mean(),
             "severe_pain_72h_%": 100 * df["any_severe_pain"].mean(), "oversedation_72h_%": 100 * df["any_oversed"].mean(),
             "naloxone_72h_%": 100 * df["any_naloxone"].mean(), "mme_72h_median": df["mme_admin_72h"].median(),
             "opioid_infusion_%": 100 * (df["mme_inf_72h"] > 0).mean(),
             "multimodal_0_12h_%": 100 * df["multimodal_0_12h"].mean(),
             "icu_los_h_median": df["icu_los_h"].median(), "hosp_death_%": 100 * df["hosp_death"].mean()}
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}
    t1 = pd.DataFrame([desc(st, "all postoperative"), desc(core, "core cohort")])
    t1.to_csv(os.path.join(out, "table1.csv"), index=False)
    log(t1.T.to_string())
    log("\nSurgery groups (core):")
    log(core["surgery_group"].value_counts().to_string())
    log("\neGFR categories (core):")
    log(core["egfr_cat"].value_counts(dropna=False).to_string())
    cg = grid[grid["stay_id"].isin(core["stay_id"])]
    bs = cg.groupby("bin").agg(stays=("stay_id", "nunique"), pain_mean=("pain_mean", "mean"),
                               pain_assess_per_bin=("assess_count", "mean"),
                               rass_documented_pct=("rass_count", lambda s: 100 * s.notna().mean()),
                               oversed_pct=("oversedation", lambda s: 100 * s.mean()),
                               mme_admin_median=("mme_admin", "median"), mme_admin_p90=("mme_admin", lambda s: s.quantile(.9)),
                               mme_admin_max=("mme_admin", "max"),
                               infusion_pct=("opioid_infusion", lambda s: 100 * s.mean())).round(2)
    bs.to_csv(os.path.join(out, "bins_summary.csv"))
    log("\nCore cohort by 6-h interval:")
    log(bs.to_string())
    with open(os.path.join(out, "step2_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\nDone. Results in: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic", required=True)
    ap.add_argument("--out", default="mimic_analytic")
    a = ap.parse_args()
    main(a.mimic, a.out)
