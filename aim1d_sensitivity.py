"""
Aim 1d - Sensitivity analyses for the within-patient single-dose contrasts (MIMIC-IV).
Same model as the primary analysis (aim1c, within-patient fixed effects, reference 7.5 MME),
repeated in four subsets:
  S1 doses not given within 60 min of a previous dose (no stacking)
  S2 no opioid infusion running in the same 6-h interval
  S3 stricter reassessment window: IV 15-60 min, oral 30-120 min
  S4 patients not intubated on day 1 (self-reported pain more reliable)
Needs aim1b_dose_response_robust.py and aim1c_dose_level.py in the same folder.

Usage:
  python aim1d_sensitivity.py --dir D:\\MIMIC_analytic --out D:\\MIMIC_analytic\\aim1d
"""
import argparse
import os

import numpy as np
import pandas as pd

from aim1c_dose_level import prepare, contrast

SPECS = [("iv_bolus", "all", [5, 7.5, 10, 12.5, 15, 20, 30]),
         ("iv_bolus", "fentanyl", [5, 7.5, 10, 12.5, 15, 20, 25]),
         ("iv_bolus", "hydromorphone", [5, 7.5, 10, 15, 20, 30]),
         ("iv_bolus", "morphine", [5, 7.5, 10, 12.5, 15]),
         ("oral", "all", [5, 7.5, 10, 15, 20, 30])]


def main(a):
    os.makedirs(a.out, exist_ok=True)
    ind = prepare(a.dir)
    # stricter window recomputed from minutes to reassessment
    strict = {"iv_bolus": (15, 60), "oral": (30, 120)}
    m = ind["min_to_pain_reassess"]
    ind["R_strict"] = [int(strict[s][0] <= v <= strict[s][1]) if pd.notna(v) else 0 for s, v in zip(ind["src"], m)]
    subsets = {
        "primary": ind[ind["R"] == 1],
        "S1_no_stacking": ind[(ind["R"] == 1) & (ind["stacked"] == 0)],
        "S2_no_infusion": ind[(ind["R"] == 1) & (ind["opioid_infusion"] == 0)],
        "S3_strict_window": ind[ind["R_strict"] == 1],
        "S4_not_intubated_day1": ind[(ind["R"] == 1) & (ind["intubated_day1"] == 0)],
    }
    out, log = [], []
    for name, d in subsets.items():
        for src, agent, grid in SPECS:
            dd = d[d["src"] == src]
            if agent != "all":
                dd = dd[dd["agent"] == agent]
            lo, hi = dd["mme"].quantile([0.01, 0.99]) if len(dd) else (0, 0)
            g = sorted({x for x in grid if lo <= x <= hi} | {7.5})
            for y in ["delta", "relief"]:
                c = contrast(dd, y, g, 7.5, fe=True) if len(dd) >= 300 else None
                if c is None:
                    log.append(f"{name} {src}/{agent} {y}: too few doses ({len(dd)})")
                    continue
                c.insert(0, "outcome", y)
                c.insert(0, "agent", agent)
                c.insert(0, "route", src)
                c.insert(0, "analysis", name)
                out.append(c)
                log.append(f"{name} {src}/{agent} {y}: n={int(c['n_doses'].iloc[0]):,} | "
                           + "; ".join(f"{x:g}:{e:+.2f} ({l:+.2f},{h:+.2f})"
                                       for x, e, l, h in zip(c["mme"], c["diff_vs_ref"], c["lo95"], c["hi95"])))
                print(log[-1])
    pd.concat(out).round(4).to_csv(os.path.join(a.out, "aim1d_sensitivity_contrasts.csv"), index=False)
    with open(os.path.join(a.out, "aim1d_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print(f"Done. Aggregate results in {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
