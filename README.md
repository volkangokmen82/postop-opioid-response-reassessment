# Postoperative opioid response and nurse-led pain reassessment in intensive care

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24486228.svg)](https://doi.org/10.5281/zenodo.24486228)

Analysis code for the article "Beyond Dose Escalation: Agent-Specific Opioid Response, Nurse-Led Pain Reassessment and Prediction of Non-Response in Postoperative Intensive Care".

## Data

The code uses two credentialed-access PhysioNet databases. No patient-level data are included in this repository.

- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/
- eICU Collaborative Research Database v2.0: https://physionet.org/content/eicu-crd/

Access requires a PhysioNet account, completion of the required training and signing of the data use agreement.

## Requirements

Python 3.10 or later with pandas, numpy, statsmodels, patsy, scikit-learn and matplotlib.

```
python -m pip install pandas numpy statsmodels patsy scikit-learn matplotlib
```

## Order of scripts

| Step | Script | Purpose |
| --- | --- | --- |
| 1 | `mimic_step2_build.py` | Builds the MIMIC-IV analytic files (stays, 6-h intervals, single opioid doses with pre- and post-dose pain) |
| 1 | `eicu_step2_build.py` | Builds the eICU analytic files (stays, 6-h intervals, prescribed opioid intensity) |
| 2 | `aim1c_dose_level.py` | Single-dose dose-response: adjusted, inverse-probability-weighted and within-patient models (primary analysis) |
| 3 | `aim1d_sensitivity.py` | Sensitivity analyses of the within-patient dose contrasts |
| 4 | `aim3_opioid_sparing_prediction.py` | Non-opioid adjuncts, response to the previous dose and prediction of single-dose non-response |
| 5 | `eicu_aim2_reassessment.py` | Pain reassessment after scores of 4 or more, between-hospital variation and patient outcomes |

`aim1b_dose_response_robust.py` holds helper functions imported by steps 2 to 4 and must be in the same folder.

## Example

```
python mimic_step2_build.py --mimic /path/to/mimiciv --out /path/to/MIMIC_analytic
python eicu_step2_build.py --eicu /path/to/eicu --out /path/to/EICU_analytic
python aim1c_dose_level.py --dir /path/to/MIMIC_analytic --out /path/to/MIMIC_analytic/aim1c
python aim1d_sensitivity.py --dir /path/to/MIMIC_analytic --out /path/to/MIMIC_analytic/aim1d
python aim3_opioid_sparing_prediction.py --dir /path/to/MIMIC_analytic --out /path/to/MIMIC_analytic/aim3
python eicu_aim2_reassessment.py --eicu /path/to/eicu --dir /path/to/EICU_analytic --out /path/to/EICU_analytic/aim2
python eicu_aim2_reassessment.py --eicu /path/to/eicu --dir /path/to/EICU_analytic --out /path/to/EICU_analytic/aim2_t120 --threshold 120
python eicu_aim2_reassessment.py --eicu /path/to/eicu --dir /path/to/EICU_analytic --out /path/to/EICU_analytic/aim2_excl --min_rate 5
```

The build scripts write patient-level files that must stay on the user's own computer under the data use agreement. The analysis scripts write aggregate tables and figures only, and cells with fewer than 11 observations are suppressed.

## Licence

MIT. See `LICENSE`.
