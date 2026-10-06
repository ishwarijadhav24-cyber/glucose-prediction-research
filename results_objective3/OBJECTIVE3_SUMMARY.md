# Objective 3 — External validation on HUPA-UCM

OhioT1DM-trained models (all 12 patients, Objective 1 settings) were applied to HUPA-UCM **without retraining or refitting** (imputer/scaler: transform only). Metrics are computed per patient, then averaged across patients (each patient counts once), at evaluable points only (real Libre reading at t and at t+30 min).

## Results (mean across patients)

| Model | Ohio LOSO MARD % | HUPA MAIN MARD % | HUPA ALL MARD % | HUPA MAIN RMSE mg/dL | HUPA MAIN Clarke A+B % |
|---|---|---|---|---|---|
| Persistence | 12.11 | 13.03 | 13.16 | 24.44 | 97.89 |
| Ridge | 10.70 | **12.22** | **12.20** | 21.98 | **98.04** |
| RandomForest | 10.13 | 13.24 | 13.37 | 22.05 | 95.53 |
| MLP | 10.03 | 14.21 | 14.35 | 23.81 | 95.60 |
| LightGBM | 10.03 | 13.38 | 13.49 | 21.95 | 95.74 |
| Stacking | 10.03 | 12.83 | 12.91 | **21.46** | 96.23 |

SDs across patients, plus MAE, R², Clarke A and B, are in `hupa_summary_main.csv` / `hupa_summary_all.csv`. Per-patient values are in `hupa_patient_results.csv`.

## Patients
- 25 HUPA-UCM patients. **MAIN = 19** patients (99,855 evaluable points); **ALL = 23** patients (105,423 points).
- Excluded from both, no raw FreeStyle Libre CGM: HUPA0009P, HUPA0010P.
- Excluded from MAIN only (in ALL), missing insulin or meal events: HUPA0011P, HUPA0015P, HUPA0018P, HUPA0020P.

## Data notes
- Glucose comes only from raw Libre type-0 (historic, ~15-minute) readings. The interpolated Preprocessed glucose is not used.
- HUPA0017P: 4 negative recorded boluses (−1, −1, −3, −1 U) are invalid records and were treated as no event (general rule: bolus ≤ 0 means no event).
- HUPA0027P: two identical Libre files were de-duplicated, and 12 timestamps had conflicting values (the last one was kept). The patient has a 574-day window with 59,671 points (~60% of MAIN), which is why patient-level averaging is the primary metric.
- HUPA0028P: 66% Libre coverage (raw data ends 24 Apr 2022, window ends 18 May 2022). This is the only patient below 70%.
- Parser validation: all 13 checks PASS (`hupa_parser_check_log.txt`). Objective 3 audit: 11/11 PASS (`audit_objective3_log.txt`).

## Reproducibility and environment
- Objective 1 was run in a different environment, and its LightGBM version was not recorded. Objective 3 used Python 3.13.9, LightGBM 4.7.0, scikit-learn 1.7.2, numpy 2.3.5 and pandas 2.3.3 (`models_objective3/settings.json`).
- Check: retraining on 11 OhioT1DM patients and testing on 559 reproduced Objective 1 exactly for Ridge (MARD 11.2153 vs 11.2153, RMSE 23.9318 vs 23.9318). LightGBM was within 0.012 MARD points (10.527 vs 10.539; RMSE 22.849 vs 22.843). See `reproducibility_check_559.csv`.
- Model settings were copied from `evaluate_loso.py` (its models are built inline and cannot be imported). Metrics and the Clarke grid were imported from `evaluate_loso.py`. Objective 1's LightGBM has no early stopping. The MLP's built-in early stopping uses an internal split of the OhioT1DM training data only.

## Findings
1. **Performance drops on an external dataset.** The best OhioT1DM models (MARD ≈ 10.0%) reach 12.2–14.4% MARD on HUPA-UCM MAIN, which is 2–4 points worse. RMSE changes much less (≈ 20–21 vs 21.5–24 mg/dL).
2. **The ranking changes.** On OhioT1DM the complex models (LightGBM, MLP, Stacking) were best. On HUPA-UCM the simple Ridge model has the lowest MARD (12.22%) and the highest Clarke A+B (98.0%), and Stacking has the lowest RMSE. MLP generalizes worst (14.2% MARD, worse than Persistence).
3. **The gain over Persistence mostly disappears for MARD.** On MARD, Ridge and Stacking beat Persistence for 13 of 19 patients, but LightGBM only for 10/19 and MLP for 4/19. On RMSE, most models beat Persistence for 16–17 of 19 patients.
4. **A likely cause is the sensor difference.** HUPA-UCM uses a FreeStyle Libre with ~15-minute historic readings, while OhioT1DM uses a 5-minute CGM. The 5-minute lag and difference features are therefore built from forward-filled values. Flexible models trained on 5-minute dynamics appear to transfer worse than linear ones. (The evaluation masks also differ: OhioT1DM LOSO evaluated all rows with glucose at t, while HUPA-UCM requires a real reading at t and t+30.)
5. **The ALL analysis does not change the conclusions.** Adding the 4 patients with missing events moves MARD by at most 0.14 points and leaves the model ranking identical (Ridge < Stacking < Persistence < RandomForest < LightGBM < MLP).
