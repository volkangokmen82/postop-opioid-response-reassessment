"""
eICU-CRD - Step 2: build the analytic dataset
Unit of analysis: postoperative ICU stay x 6-hour interval (first 72 h; 12 intervals)

Outputs (in --out):
  PATIENT-LEVEL (keep on your computer, never share):
    eicu_stays.csv.gz   one row per stay (covariates, organ function, outcomes)
    eicu_bins.csv.gz    one row per stay x 6-h interval (pain, RASS, opioid MME, adjuncts)
  AGGREGATE (safe to share):
    step2_summary.txt, table1.csv, bins_summary.csv, mme_parse_report.csv

Opioid exposure in eICU = PRESCRIBED (ordered) opioid, not administered:
  - mme_prn_avail : maximum ordered PRN opioid available in the interval
                    (max dose x permitted frequency); PACU/procedural titration orders excluded
  - mme_sched     : scheduled (non-PRN) repeated orders, same rule
  - mme_once      : one-time orders, at their start time
  - mme_inf       : continuous opioid infusions (charted rate x time; true dose)
  All expressed as ORAL morphine milligram equivalents (MME).
MME factors (oral morphine mg per unit): oral morphine 1/mg, oral oxycodone 1.5/mg,
  oral hydrocodone 1/mg, oral hydromorphone 5/mg, oral tramadol 0.2/mg, oral codeine 0.15/mg,
  oral tapentadol 0.4/mg, oral methadone 4.7/mg, oral meperidine 0.1/mg (CDC 2022);
  IV morphine 3/mg, IV hydromorphone 20/mg, IV fentanyl 0.3/mcg, IV meperidine 0.4/mg
  (standard parenteral equianalgesic ratios). Neuraxial (epidural) opioids are flagged, not converted.

Usage (Windows):
    cd /d D:\\EICU
    python eicu_step2_build.py --eicu D:\\EICU --out D:\\EICU_analytic
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

H = 72
BIN = 6
NBIN = H // BIN
POSTOP_SOURCES = {"operating room", "recovery room", "pacu"}

ORAL_F = {"morphine": 1, "oxycodone": 1.5, "hydrocodone": 1, "hydromorphone": 5, "tramadol": 0.2,
          "codeine": 0.15, "tapentadol": 0.4, "methadone": 4.7, "meperidine": 0.1}
PAR_F = {"morphine": 3, "hydromorphone": 20, "fentanyl": 0.3, "meperidine": 0.4}  # fentanyl per mcg

ADJ = {
    "paracetamol": r"acetaminophen|paracetamol|tylenol|ofirmev",
    "nsaid": r"ketorolac|toradol|ibuprofen|celecoxib|diclofenac|naproxen|meloxicam",
    "gabapentinoid": r"gabapentin|pregabalin",
    "ketamine": r"ketamine",
    "dexmedetomidine": r"dexmedetomidine|precedex",
    "propofol": r"propofol|diprivan",
    "benzodiazepine": r"midazolam|versed|lorazepam|ativan|diazepam|valium|alprazolam|clonazepam",
    "antipsychotic": r"haloperidol|haldol|quetiapine|olanzapine|risperidone",
}


def find_table(folder, name):
    for f in os.listdir(folder):
        if f.lower() in (name.lower() + ".csv", name.lower() + ".csv.gz"):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"{name}.csv(.gz) not found in {folder}")


# ---------------------------------------------------------------- parsing helpers
def agent_of(name):
    n = str(name).lower()
    if re.search(r"oxycodone|percocet|oxy ir|roxicodone", n):
        return "oxycodone"
    if re.search(r"hydrocodone|norco|vicodin|lortab", n):
        return "hydrocodone"
    if re.search(r"hydromorphone|dilaudid", n):
        return "hydromorphone"
    if re.search(r"fentanyl|sublimaze", n):
        return "fentanyl"
    if re.search(r"morphine", n):
        return "morphine"
    if re.search(r"meperidine|demerol", n):
        return "meperidine"
    if re.search(r"tramadol", n):
        return "tramadol"
    if re.search(r"methadone", n):
        return "methadone"
    if re.search(r"tapentadol", n):
        return "tapentadol"
    if re.search(r"codeine", n):
        return "codeine"
    return None


def route_of(route, name):
    t = (str(route) + " " + str(name)).lower()
    if re.search(r"epid|caudal|intrathecal|\bep\b", t):
        return "neuraxial"
    if re.search(r"transderm|patch|\btd\b", t):
        return "transdermal"
    if re.search(r"\bpo\b|oral|tube|\bng\b|ngt|gastr|\bpeg\b|feeding|\bj\b|\bog\b|\bft\b|tab|sublingual", t):
        return "oral"
    return "parenteral"


EXCL_FREQ = re.compile(r"pacu|post-?proc|postop|stk-med|pyx|manual|procedure|pre-?proc|intra-?proc|"
                       r"on ?call|\.disp|\.rrt|sedation|cath lab", re.I)


def parse_freq(f):
    """returns (kind, interval_h, ndoses). kind in once/repeat/continuous/exclude/unknown"""
    t = str(f).lower().strip()
    if t in ("", "nan"):
        return ("unknown", None, None)
    if EXCL_FREQ.search(t):
        return ("exclude", None, None)
    if re.search(r"contin|titrat|infuse|\bcont\b", t):
        return ("continuous", None, None)
    m = re.search(r"(?:q|every)\s*(\d+)\s*-?\s*(?:min|m\b)", t) or re.search(r"q(\d+)m(?:in)?", t)
    if m:
        x = re.search(r"x\s*(\d+)", t)
        return ("repeat_min", int(m.group(1)) / 60, int(x.group(1)) if x else None)
    m = (re.search(r"(?:q|every)\s*(\d+)\s*-?\s*(?:h|hr|hour)", t) or re.search(r"q(\d+)h", t)
         or re.search(r"prn(\d+)\s*indef", t) or re.search(r"q(\d+)\s*hrs", t))
    if m and int(m.group(1)) > 0:
        return ("repeat", float(m.group(1)), None)
    for pat, iv in [(r"\bqid\b|4\s*x|4 times", 6), (r"\btid\b|3\s*x|3 times", 8),
                    (r"\bbid\b|2 times", 12), (r"daily|\bqd\b|qhs|nightly|1xdaily", 24)]:
        if re.search(pat, t):
            return ("repeat", float(iv), None)
    if re.search(r"once|onetime|one time|\bone\b|1xonly|\bx1\b|\bstat\b|\bnow\b|\b1x\b|oncerx|\bot\b|\boto\b", t):
        return ("once", None, 1)
    return ("unknown", None, None)


UNIT_CODES = [(r"\s3$", " mg"), (r"\s8$", " mcg"), (r"\s5002$", " tab")]


def mg_per_tab(name):
    n = str(name).lower()
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:mg)?\s*[-/]\s*(?=\d)", n)]
    allmg = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*mg", n)]
    cand = [x for x in nums + allmg if x not in (325.0, 300.0, 500.0)]
    if cand:
        return cand[0]
    if re.search(r"percocet|norco|vicodin", n):
        return 5.0
    return None


def conc_per_ml(name):
    """amount per mL from drug name, returns (value, unit) e.g. 50 MCG/ML, 0.05 MG/ML, 50 MCG/1 ML"""
    n = str(name).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg)\s*/\s*(\d+(?:\.\d+)?)?\s*ml", n)
    if m:
        vol = float(m.group(3)) if m.group(3) else 1.0
        return float(m.group(1)) / vol, m.group(2)
    return None


def parse_dose(dosage, name):
    """max amount per administration -> (amount, unit 'mg'|'mcg') or (None, reason)"""
    d = str(dosage).strip().replace(",", "")
    if d.lower() in ("", "nan"):
        return None, "missing"
    for pat, rep in UNIT_CODES:
        d = re.sub(pat, rep, d)
    m = re.search(r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*(mcg|mg|tab|tablet|ml|ea|each)", d.lower())
    if not m:
        return None, "unparsed"
    hi = float(m.group(2) or m.group(1))
    unit = m.group(3)
    if hi <= 0:
        return None, "zero"
    if unit in ("mg", "mcg"):
        return (hi, unit), "ok"
    if unit in ("tab", "tablet"):
        mpt = mg_per_tab(name)
        return ((hi * mpt, "mg"), "ok_tab") if mpt else (None, "tab_strength_unknown")
    if unit == "ml":
        c = conc_per_ml(name)
        return ((hi * c[0], c[1]), "ok_ml") if c else (None, "ml_conc_unknown")
    return None, "unit_" + unit


def to_mme(agent, route, amount, unit):
    if agent is None or amount is None:
        return None
    if route == "oral":
        f = ORAL_F.get(agent)
        if f is None:
            return None
        mg = amount / 1000 if unit == "mcg" else amount
        return mg * f
    if route == "parenteral":
        f = PAR_F.get(agent)
        if f is None:
            return None
        if agent == "fentanyl":
            mcg = amount * 1000 if unit == "mg" else amount
            return mcg * f
        mg = amount / 1000 if unit == "mcg" else amount
        return mg * f
    return None


MAX_DOSE = {  # plausible single administration; above = bag/charge/entry error -> excluded
    ("fentanyl", "parenteral"): ("mcg", 200), ("morphine", "parenteral"): ("mg", 20),
    ("hydromorphone", "parenteral"): ("mg", 4), ("meperidine", "parenteral"): ("mg", 100),
    ("oxycodone", "oral"): ("mg", 40), ("hydrocodone", "oral"): ("mg", 20), ("tramadol", "oral"): ("mg", 200),
    ("morphine", "oral"): ("mg", 60), ("hydromorphone", "oral"): ("mg", 8), ("tapentadol", "oral"): ("mg", 200),
    ("codeine", "oral"): ("mg", 120), ("methadone", "oral"): ("mg", 40),
}
MAX_RATE = {"fentanyl": 400, "morphine": 20, "hydromorphone": 5}  # mcg/h fentanyl, mg/h others


def plausible(agent, route, amount, unit):
    lim = MAX_DOSE.get((agent, route))
    if lim is None or amount is None:
        return True
    u, mx = lim
    a = amount * 1000 if (unit == "mg" and u == "mcg") else amount / 1000 if (unit == "mcg" and u == "mg") else amount
    return a <= mx


def egfr_ckdepi2021(scr, age, female):
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    r = scr / k
    return 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.2 * 0.9938 ** age * np.where(female, 1.012, 1)


def overlap_bins(start, stop):
    """minutes of overlap of [start, stop) with each 6-h bin (array of NBIN)"""
    edges = np.arange(NBIN + 1) * BIN * 60
    lo = np.clip(start, edges[:-1], edges[1:])
    hi = np.clip(stop, edges[:-1], edges[1:])
    return np.maximum(hi - lo, 0)


# ---------------------------------------------------------------- main
def main(eicu, out):
    os.makedirs(out, exist_ok=True)
    log_lines = []

    def log(m=""):
        print(m)
        log_lines.append(str(m))

    # ---------- stays and covariates
    pat = pd.read_csv(find_table(eicu, "patient"),
                      usecols=["patientunitstayid", "hospitalid", "gender", "age", "admissionweight",
                               "admissionheight", "unittype", "unitadmitsource", "unitdischargeoffset",
                               "hospitaldischargestatus", "unitvisitnumber"])
    dx = pd.read_csv(find_table(eicu, "admissionDx"), usecols=["patientunitstayid", "admitdxpath"])
    opdx = dx[dx["admitdxpath"].str.contains(r"\|Operative\|", na=False)].copy()
    opdx["surgery_group"] = opdx["admitdxpath"].str.extract(r"\|Operative\|Diagnosis\|([^|]+)")[0]
    opdx = opdx.drop_duplicates("patientunitstayid")[["patientunitstayid", "surgery_group"]]
    src = pat["unitadmitsource"].fillna("").str.strip().str.lower()
    pat = pat[src.isin(POSTOP_SOURCES) | pat["patientunitstayid"].isin(opdx["patientunitstayid"])]
    pat = pat.merge(opdx, on="patientunitstayid", how="left")
    pat["age_num"] = pd.to_numeric(pat["age"].replace("> 89", "90"), errors="coerce")
    pat = pat[(pat["age_num"] >= 18) & (pat["unitdischargeoffset"] >= 6 * 60)]
    pat["female"] = (pat["gender"] == "Female").astype(int)
    pat["hosp_death"] = (pat["hospitaldischargestatus"] == "Expired").astype(int)
    pat["icu_los_h"] = pat["unitdischargeoffset"] / 60
    st = pat.rename(columns={"patientunitstayid": "sid"})
    sids = set(st["sid"])
    log(f"Adult postoperative stays with ICU stay >= 6 h: {len(sids):,}")

    ap = pd.read_csv(find_table(eicu, "apachePatientResult"),
                     usecols=["patientunitstayid", "apacheversion", "apachescore", "predictedhospitalmortality"])
    ap = ap[ap["apacheversion"] == "IVa"].drop_duplicates("patientunitstayid")
    ap["predictedhospitalmortality"] = pd.to_numeric(ap["predictedhospitalmortality"], errors="coerce").where(lambda s: s >= 0)
    ap["apachescore"] = ap["apachescore"].where(ap["apachescore"] >= 0)
    st = st.merge(ap.rename(columns={"patientunitstayid": "sid"})[["sid", "apachescore", "predictedhospitalmortality"]],
                  on="sid", how="left")
    aps = pd.read_csv(find_table(eicu, "apacheApsVar"), usecols=["patientunitstayid", "intubated"])
    st = st.merge(aps.rename(columns={"patientunitstayid": "sid", "intubated": "intubated_day1"})
                  .drop_duplicates("sid"), on="sid", how="left")

    # ---------- labs (baseline = closest to ICU admission within -24..+24 h)
    lab = pd.read_csv(find_table(eicu, "lab"), usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    lab = lab[lab["patientunitstayid"].isin(sids) & lab["labresultoffset"].between(-24 * 60, 24 * 60)
              & lab["labname"].isin(["creatinine", "total bilirubin", "ALT (SGPT)", "albumin"])].copy()
    lab["absoff"] = lab["labresultoffset"].abs()
    lab = lab.sort_values("absoff").drop_duplicates(["patientunitstayid", "labname"])
    lw = lab.pivot(index="patientunitstayid", columns="labname", values="labresult")
    lw.columns = [c.split(" ")[0].lower() if c != "total bilirubin" else "bilirubin" for c in lw.columns]
    st = st.merge(lw, left_on="sid", right_index=True, how="left")
    if "creatinine" in st:
        st["egfr"] = egfr_ckdepi2021(st["creatinine"].clip(0.2, 15), st["age_num"], st["female"] == 1)
        st["egfr_cat"] = pd.cut(st["egfr"], [-np.inf, 30, 60, 90, np.inf], labels=["<30", "30-59", "60-89", ">=90"])

    # ---------- nurse charting: pain and sedation
    pain_parts, sed_parts = [], []
    cols = ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel",
            "nursingchartcelltypevalname", "nursingchartvalue"]
    for ch in pd.read_csv(find_table(eicu, "nurseCharting"), usecols=cols, chunksize=1_000_000, dtype=str):
        ch["sid"] = pd.to_numeric(ch["patientunitstayid"], errors="coerce")
        ch["off"] = pd.to_numeric(ch["nursingchartoffset"], errors="coerce")
        ch = ch[ch["sid"].isin(sids) & ch["off"].between(0, H * 60 - 1)]
        if ch.empty:
            continue
        lab_ = ch["nursingchartcelltypevallabel"].fillna("")
        nam_ = ch["nursingchartcelltypevalname"].fillna("")
        val = ch["nursingchartvalue"]
        p = ch[(nam_ == "Pain Score") & lab_.str.contains("Pain Score", na=False)]
        p = p.assign(v=pd.to_numeric(p["nursingchartvalue"], errors="coerce"))
        pain_parts.append(p.loc[p["v"].between(0, 10), ["sid", "off", "v"]])
        s = ch[(lab_ == "Sedation Scale/Score/Goal") & nam_.isin(["Sedation Scale", "Sedation Score"])
               | lab_.isin(["RASS", "SEDATION SCORE"])]
        sed_parts.append(s[["sid", "off", "nursingchartcelltypevallabel", "nursingchartcelltypevalname",
                            "nursingchartvalue"]])
    pain = pd.concat(pain_parts)
    sed = pd.concat(sed_parts)
    # scale type for "Sedation Score": the "Sedation Scale" entry at the same stay/time
    scale = sed[sed["nursingchartcelltypevalname"] == "Sedation Scale"][["sid", "off", "nursingchartvalue"]] \
        .rename(columns={"nursingchartvalue": "scale"}).drop_duplicates(["sid", "off"])
    sc = sed[sed["nursingchartcelltypevalname"] == "Sedation Score"].merge(scale, on=["sid", "off"], how="left")
    sc = sc[sc["scale"].fillna("RASS").str.upper() == "RASS"]
    direct = sed[sed["nursingchartcelltypevallabel"].isin(["RASS", "SEDATION SCORE"])]
    rass = pd.concat([sc[["sid", "off", "nursingchartvalue"]], direct[["sid", "off", "nursingchartvalue"]]])
    rass["v"] = pd.to_numeric(rass["nursingchartvalue"], errors="coerce")
    rass = rass[rass["v"].between(-5, 4)][["sid", "off", "v"]].drop_duplicates()
    log(f"Pain scores: {len(pain):,} in {pain['sid'].nunique():,} stays | "
        f"RASS: {len(rass):,} in {rass['sid'].nunique():,} stays")

    # ---------- medication orders
    med = pd.read_csv(find_table(eicu, "medication"),
                      usecols=["patientunitstayid", "drugstartoffset", "drugstopoffset", "drugname",
                               "dosage", "routeadmin", "frequency", "prn", "drugordercancelled"], dtype=str)
    med["sid"] = pd.to_numeric(med["patientunitstayid"], errors="coerce")
    med["start"] = pd.to_numeric(med["drugstartoffset"], errors="coerce")
    med["stop"] = pd.to_numeric(med["drugstopoffset"], errors="coerce")
    med = med[med["sid"].isin(sids) & (med["drugordercancelled"].fillna("No").str.lower() != "yes")
              & (med["start"] < H * 60)]
    med["stop"] = np.where(med["stop"].isna() | (med["stop"] <= med["start"]), med["start"] + 24 * 60, med["stop"])

    opi = med[med["drugname"].map(agent_of).notna()].copy()
    n0 = len(opi)
    opi = opi.drop_duplicates(["sid", "drugname", "dosage", "frequency", "routeadmin", "start"])
    print(f"  duplicate opioid order rows removed: {n0 - len(opi):,}")
    opi["agent"] = opi["drugname"].map(agent_of)
    opi["route"] = [route_of(r, n) for r, n in zip(opi["routeadmin"], opi["drugname"])]
    fr = opi["frequency"].map(parse_freq)
    opi["fkind"] = [f[0] for f in fr]
    opi["interval_h"] = [f[1] for f in fr]
    opi["ndoses"] = [f[2] for f in fr]
    pdz = [parse_dose(d, n) for d, n in zip(opi["dosage"], opi["drugname"])]
    impl = [p[0] is not None and not plausible(a, r, *p[0]) for a, r, p in zip(opi["agent"], opi["route"], pdz)]
    pdz = [(None, "implausible_dose") if bad else p for p, bad in zip(pdz, impl)]
    opi["dose_status"] = [p[1] for p in pdz]
    opi["mme_dose"] = [to_mme(a, r, *(p[0] if p[0] else (None, None)))
                       for a, r, p in zip(opi["agent"], opi["route"], pdz)]
    opi["is_prn"] = opi["prn"].fillna("").str.lower().eq("yes") | opi["frequency"].fillna("").str.contains("prn", case=False)

    # parse report (aggregate)
    rep = (opi.assign(route=opi["route"]).groupby(["agent", "route", "fkind", "dose_status"]).size()
           .rename("orders").reset_index().sort_values("orders", ascending=False))
    rep.to_csv(os.path.join(out, "mme_parse_report.csv"), index=False)
    log(f"Opioid orders with implausible single dose (excluded): {sum(impl):,}")
    usable = opi["mme_dose"].notna() & opi["fkind"].isin(["once", "repeat", "repeat_min"]) & opi["route"].isin(["oral", "parenteral"])
    log(f"Opioid orders: {len(opi):,} | converted to MME: {usable.sum():,} ({100 * usable.mean():.1f}%) | "
        f"excluded procedural/PACU: {(opi['fkind'] == 'exclude').sum():,} | continuous (infusion table): "
        f"{(opi['fkind'] == 'continuous').sum():,} | neuraxial: {(opi['route'] == 'neuraxial').sum():,}")

    # per-bin prescribed MME
    rows = []
    for r in opi[usable].itertuples(index=False):
        if r.fkind == "once" or (r.fkind == "repeat_min" and r.ndoses):
            n = 1 if r.fkind == "once" else r.ndoses
            if 0 <= r.start < H * 60:
                b = int(r.start // (BIN * 60))
                rows.append((r.sid, b, "once", r.mme_dose * n))
            continue
        if r.fkind == "repeat_min":  # open-ended minute-level titration -> not counted
            continue
        ov = overlap_bins(r.start, min(r.stop, H * 60))
        for b in np.nonzero(ov)[0]:
            doses = ov[b] / (r.interval_h * 60)
            rows.append((r.sid, int(b), "prn_avail" if r.is_prn else "sched", r.mme_dose * doses))
    mb = pd.DataFrame(rows, columns=["sid", "bin", "kind", "mme"])
    mb = mb.pivot_table(index=["sid", "bin"], columns="kind", values="mme", aggfunc="sum").add_prefix("mme_").reset_index()

    # order-based flags per bin (any active order), opioid regimen and adjuncts
    def active_flags(df, label):
        out_ = []
        for r in df.itertuples(index=False):
            ov = overlap_bins(max(r.start, 0), min(r.stop, H * 60))
            for b in np.nonzero(ov)[0]:
                out_.append((r.sid, int(b), label))
        return out_

    flags = []
    flags += active_flags(opi[opi["is_prn"] & opi["route"].isin(["oral", "parenteral"]) & (opi["fkind"] != "exclude")], "opioid_prn")
    flags += active_flags(opi[~opi["is_prn"] & opi["fkind"].isin(["repeat"])
                              & (opi["dose_status"] != "implausible_dose")], "opioid_sched")
    flags += active_flags(opi[opi["route"] == "neuraxial"], "neuraxial")
    for k, rx in ADJ.items():
        flags += active_flags(med[med["drugname"].str.contains(rx, case=False, na=False)], k)

    # ---------- infusions (true administered rate)
    inf = pd.read_csv(find_table(eicu, "infusionDrug"),
                      usecols=["patientunitstayid", "infusionoffset", "drugname", "drugrate"], dtype=str)
    inf["sid"] = pd.to_numeric(inf["patientunitstayid"], errors="coerce")
    inf["off"] = pd.to_numeric(inf["infusionoffset"], errors="coerce")
    inf["rate"] = pd.to_numeric(inf["drugrate"], errors="coerce")
    inf = inf[inf["sid"].isin(sids) & inf["off"].between(-12 * 60, H * 60)]
    for k in ["dexmedetomidine", "propofol", "benzodiazepine", "ketamine"]:
        sub = inf[inf["drugname"].str.contains(ADJ[k], case=False, na=False) & (inf["rate"] > 0)
                  & inf["off"].between(0, H * 60 - 1)]
        flags += [(s, int(o // (BIN * 60)), k) for s, o in zip(sub["sid"], sub["off"])]
    oinf = inf[inf["drugname"].map(agent_of).notna()].copy()
    oinf["agent"] = oinf["drugname"].map(agent_of)
    nlow = oinf["drugname"].str.lower()
    oinf["neuraxial"] = nlow.str.contains(r"epid|bupiv")
    unit = nlow.str.extract(r"\(([^)]*)\)\s*$")[0].fillna("")
    w = oinf["sid"].map(st.set_index("sid")["admissionweight"])
    conc = [conc_per_ml(n.replace(" ", "")) or None for n in nlow]
    bag = nlow.str.extract(r"(\d[\d.]*)\s*(mcg|mg)\D+?(\d[\d.]*)\s*ml")
    bagc = pd.to_numeric(bag[0], errors="coerce") / pd.to_numeric(bag[2], errors="coerce")
    per_hr = pd.Series(np.nan, index=oinf.index)  # mcg/h for fentanyl, mg/h otherwise
    fen = oinf["agent"] == "fentanyl"
    per_hr[fen & (unit == "mcg/hr")] = oinf["rate"]
    per_hr[fen & (unit == "mg/hr")] = oinf["rate"] * 1000
    per_hr[fen & (unit == "mcg/kg/hr")] = oinf["rate"] * w
    per_hr[fen & (unit == "mcg/kg/min")] = oinf["rate"] * w * 60
    per_hr[fen & (unit == "ml/hr")] = oinf["rate"] * bagc.where(bagc.notna(), 10.0)  # standard 10 mcg/mL premix
    nf = oinf["agent"].isin(["morphine", "hydromorphone"])
    per_hr[nf & (unit == "mg/hr")] = oinf["rate"]
    per_hr[nf & (unit == "ml/hr") & bagc.notna()] = oinf["rate"] * bagc
    implr = per_hr > oinf["agent"].map(MAX_RATE)
    log(f"Opioid infusion records with implausible rate (excluded): {int(implr.sum()):,}")
    per_hr[implr] = np.nan
    oinf["per_hr"] = per_hr
    oinf.loc[oinf["neuraxial"], "per_hr"] = np.nan
    log(f"Opioid infusion records: {len(oinf):,} | with convertible rate: {oinf['per_hr'].notna().sum():,} "
        f"| neuraxial: {oinf['neuraxial'].sum():,}")
    flags += [(s, max(int(o // (BIN * 60)), 0), "neuraxial") for s, o in
              zip(oinf.loc[oinf["neuraxial"] & oinf["off"].between(0, H * 60 - 1), "sid"],
                  oinf.loc[oinf["neuraxial"] & oinf["off"].between(0, H * 60 - 1), "off"])]
    # step-function integration: each charted rate holds until the next record (max 4 h)
    irows = []
    for (s, a), g in oinf[oinf["per_hr"].notna()].sort_values("off").groupby(["sid", "agent"]):
        offs = g["off"].to_numpy()
        rates = g["per_hr"].to_numpy()
        ends = np.append(offs[1:], offs[-1] + 240)
        ends = np.minimum(ends, offs + 240)
        for o, e, r_ in zip(offs, ends, rates):
            ov = overlap_bins(max(o, 0), min(e, H * 60))
            for b in np.nonzero(ov)[0]:
                amt = r_ * ov[b] / 60
                irows.append((s, int(b), amt * PAR_F[a]))
    ib = pd.DataFrame(irows, columns=["sid", "bin", "mme_inf"]).groupby(["sid", "bin"], as_index=False).sum()

    # ---------- assemble stay x bin
    grid = pd.MultiIndex.from_product([sorted(sids), range(NBIN)], names=["sid", "bin"]).to_frame(index=False)
    los_min = st.set_index("sid")["unitdischargeoffset"]
    grid = grid[grid["bin"] * BIN * 60 < grid["sid"].map(los_min)]
    for df, nm in [(pain, "pain"), (rass, "rass")]:
        df = df.assign(bin=(df["off"] // (BIN * 60)).astype(int))
        agg = df.groupby(["sid", "bin"])["v"].agg(["mean", "max", "min", "count"])
        agg.columns = [f"{nm}_{c}" for c in agg.columns]
        grid = grid.merge(agg.reset_index(), on=["sid", "bin"], how="left")
    grid = grid.merge(mb, on=["sid", "bin"], how="left").merge(ib, on=["sid", "bin"], how="left")
    for c in ["mme_prn_avail", "mme_sched", "mme_once", "mme_inf"]:
        if c not in grid:
            grid[c] = 0.0
        grid[c] = grid[c].fillna(0.0)
    grid["mme_given_or_scheduled"] = grid["mme_sched"] + grid["mme_once"] + grid["mme_inf"]
    fl = pd.DataFrame(flags, columns=["sid", "bin", "flag"]).drop_duplicates()
    fl = fl[fl["bin"].between(0, NBIN - 1)]
    fw = pd.crosstab([fl["sid"], fl["bin"]], fl["flag"]).clip(upper=1).reset_index()
    grid = grid.merge(fw, on=["sid", "bin"], how="left")
    flag_cols = ["opioid_prn", "opioid_sched", "neuraxial"] + list(ADJ)
    for c in flag_cols:
        if c not in grid:
            grid[c] = 0
        grid[c] = grid[c].fillna(0).astype(int)
    grid["opioid_infusion"] = (grid["mme_inf"] > 0).astype(int)
    grid["opioid_regimen"] = np.select(
        [grid["opioid_infusion"] == 1, grid["opioid_sched"] == 1, grid["opioid_prn"] == 1],
        ["infusion", "scheduled", "prn_only"], "none")
    grid["oversedation"] = (grid["rass_min"] <= -3).astype(float).where(grid["rass_min"].notna())
    grid["severe_pain"] = (grid["pain_max"] >= 7).astype(float).where(grid["pain_max"].notna())

    # ---------- stay-level derived
    g72 = grid.groupby("sid")
    st = st.merge(g72.agg(n_pain=("pain_count", "sum"), n_rass=("rass_count", "sum"),
                          any_severe_pain=("severe_pain", "max"), any_oversed=("oversedation", "max"),
                          mme_inf_72h=("mme_inf", "sum"), mme_prn_avail_72h=("mme_prn_avail", "sum"),
                          any_opioid=("opioid_regimen", lambda s: int((s != "none").any())),
                          any_neuraxial=("neuraxial", "max")).reset_index(), on="sid", how="left")
    early = grid[grid["bin"] <= 1].groupby("sid")[["paracetamol", "nsaid", "gabapentinoid", "ketamine", "dexmedetomidine"]].max()
    early["multimodal_early"] = (early.sum(axis=1) >= 1).astype(int)
    st = st.merge(early.add_suffix("_0_12h").rename(columns={"multimodal_early_0_12h": "multimodal_0_12h"}),
                  left_on="sid", right_index=True, how="left")
    st["core_cohort"] = ((st["n_pain"] >= 3) & (st["any_opioid"] == 1)).astype(int)

    keep_st = ["sid", "hospitalid", "age_num", "female", "admissionweight", "admissionheight", "unittype",
               "surgery_group", "apachescore", "predictedhospitalmortality", "intubated_day1", "creatinine",
               "egfr", "egfr_cat", "bilirubin", "alt", "albumin", "icu_los_h", "hosp_death", "n_pain", "n_rass",
               "any_severe_pain", "any_oversed", "mme_inf_72h", "mme_prn_avail_72h", "any_opioid", "any_neuraxial",
               "paracetamol_0_12h", "nsaid_0_12h", "gabapentinoid_0_12h", "ketamine_0_12h",
               "dexmedetomidine_0_12h", "multimodal_0_12h", "core_cohort"]
    st = st[[c for c in keep_st if c in st.columns]]
    st.to_csv(os.path.join(out, "eicu_stays.csv.gz"), index=False)
    grid.to_csv(os.path.join(out, "eicu_bins.csv.gz"), index=False)

    # ---------- aggregate outputs (shareable)
    core = st[st["core_cohort"] == 1]
    log(f"\nStays: {len(st):,} | core cohort (>=3 pain scores + opioid order): {len(core):,} "
        f"from {core['hospitalid'].nunique()} hospitals")
    def desc(df, name):
        r = {"group": name, "n": len(df),
             "age_median": df["age_num"].median(), "female_%": 100 * df["female"].mean(),
             "apache_median": df["apachescore"].median(), "intubated_day1_%": 100 * df["intubated_day1"].mean(),
             "egfr_<60_%": 100 * (df["egfr"] < 60).mean(), "creatinine_available_%": 100 * df["creatinine"].notna().mean(),
             "severe_pain_72h_%": 100 * df["any_severe_pain"].mean(), "oversedation_72h_%": 100 * df["any_oversed"].mean(),
             "opioid_infusion_%": 100 * (df["mme_inf_72h"] > 0).mean(), "neuraxial_%": 100 * df["any_neuraxial"].mean(),
             "multimodal_0_12h_%": 100 * df["multimodal_0_12h"].mean(),
             "icu_los_h_median": df["icu_los_h"].median(), "hosp_death_%": 100 * df["hosp_death"].mean()}
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}
    t1 = pd.DataFrame([desc(st, "all postoperative"), desc(core, "core cohort")])
    t1.to_csv(os.path.join(out, "table1.csv"), index=False)
    log(t1.T.to_string())
    log("\nSurgery groups (core cohort):")
    log(core["surgery_group"].fillna("unknown").value_counts().head(15).to_string())
    log("\neGFR categories (core cohort):")
    log(core["egfr_cat"].value_counts(dropna=False).to_string())
    cg = grid[grid["sid"].isin(core["sid"])]
    bs = cg.groupby("bin").agg(stays=("sid", "nunique"), pain_mean=("pain_mean", "mean"),
                               pain_assess_per_bin=("pain_count", "mean"),
                               rass_documented_pct=("rass_count", lambda s: 100 * s.notna().mean()),
                               oversed_pct=("oversedation", lambda s: 100 * s.mean()),
                               mme_prn_avail_median=("mme_prn_avail", "median"),
                               mme_given_or_sched_median=("mme_given_or_scheduled", "median"),
                               prn_only_pct=("opioid_regimen", lambda s: 100 * (s == "prn_only").mean()),
                               infusion_pct=("opioid_regimen", lambda s: 100 * (s == "infusion").mean())).round(2)
    bs.to_csv(os.path.join(out, "bins_summary.csv"))
    log("\nCore cohort by 6-h interval:")
    log(bs.to_string())
    with open(os.path.join(out, "step2_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nDone. Results in: {out}")


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--eicu", required=True)
    ap_.add_argument("--out", default="eicu_analytic")
    a = ap_.parse_args()
    main(a.eicu, a.out)
