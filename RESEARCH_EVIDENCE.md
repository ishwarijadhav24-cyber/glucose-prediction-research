# Research evidence report

Facts and numbers only, from existing results files and data (no retraining). Tests are two-sided: Wilcoxon signed-rank for paired per-patient comparisons, Mann-Whitney U for unpaired ones. 95% CIs are bootstrap over patients (2000 resamples, random_state=42). All CSVs are in `results_evidence/`.

## A. Objective 1: OhioT1DM leave-one-subject-out (12 patients)

| model | n | MARD_mean | MARD_sd | MARD_ci_lo | MARD_ci_hi | RMSE_mean | RMSE_ci_lo | RMSE_ci_hi | MAE_mean | EGA_A+B_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| Persistence | 12 | 12.11 | 1.923 | 11.04 | 13.10 | 24.82 | 23.06 | 26.43 | 17.68 | 98.61 |
| Ridge | 12 | 10.70 | 1.549 | 9.884 | 11.51 | 21.66 | 20.23 | 23.13 | 15.29 | 98.36 |
| RandomForest | 12 | 10.13 | 1.543 | 9.283 | 10.94 | 20.58 | 19.19 | 22.04 | 14.40 | 98.02 |
| MLP | 12 | 10.03 | 1.453 | 9.236 | 10.78 | 20.53 | 19.06 | 22.08 | 14.29 | 98.22 |
| LightGBM | 12 | 10.03 | 1.522 | 9.201 | 10.83 | 20.33 | 18.93 | 21.80 | 14.24 | 98.00 |
| Stacking | 12 | 10.03 | 1.545 | 9.194 | 10.84 | 20.42 | 19.01 | 21.88 | 14.29 | 98.09 |

| model | n_beats_persistence | n_patients | mean_diff | diff_ci_lo | diff_ci_hi | p |
|---|---|---|---|---|---|---|
| Ridge | 12 | 12 | -1.411 | -1.677 | -1.134 | 4.88e-04 |
| RandomForest | 12 | 12 | -1.986 | -2.266 | -1.684 | 4.88e-04 |
| MLP | 12 | 12 | -2.083 | -2.455 | -1.735 | 4.88e-04 |
| LightGBM | 12 | 12 | -2.082 | -2.366 | -1.780 | 4.88e-04 |
| Stacking | 12 | 12 | -2.086 | -2.351 | -1.799 | 4.88e-04 |

| comparison | n_patients | mean_diff | diff_ci_lo | diff_ci_hi | p |
|---|---|---|---|---|---|
| LightGBM vs Ridge | 12 | -0.671 | -0.834 | -0.523 | 4.88e-04 |
| LightGBM vs Stacking | 12 | 0.004 | -0.024 | 0.030 | 0.970 |

- Ridge has lower MARD than Persistence in 12/12 patients (mean difference -1.41 points, Wilcoxon p=0.0005).
- RandomForest has lower MARD than Persistence in 12/12 patients (mean difference -1.99 points, Wilcoxon p=0.0005).
- MLP has lower MARD than Persistence in 12/12 patients (mean difference -2.08 points, Wilcoxon p=0.0005).
- LightGBM has lower MARD than Persistence in 12/12 patients (mean difference -2.08 points, Wilcoxon p=0.0005).
- Stacking has lower MARD than Persistence in 12/12 patients (mean difference -2.09 points, Wilcoxon p=0.0005).
- LightGBM vs Ridge: mean MARD difference -0.671 points (95% CI -0.834 to -0.523), Wilcoxon p=0.0005, n=12.
- LightGBM vs Stacking: mean MARD difference 0.004 points (95% CI -0.024 to 0.030), Wilcoxon p=0.9697, n=12.

## B. Objective 2: personalization (12 patients)

- p_level is the personalization level as stored in `personalization_patient_results.csv`; days = n_personal_samples x 5 min / 1440.

| p_level | median | min | max |
|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0.000 |
| 5.000 | 2.401 | 2.028 | 2.688 |
| 10.00 | 4.804 | 4.059 | 5.378 |
| 20.00 | 9.608 | 8.122 | 10.76 |
| 30.00 | 14.41 | 12.18 | 16.14 |
| 50.00 | 24.02 | 20.31 | 26.91 |

| model | p_level | n | MARD_mean | MARD_sd | MARD_ci_lo | MARD_ci_hi | RMSE_mean | RMSE_ci_lo | RMSE_ci_hi | Delta_MARD_mean | Delta_MARD_ci_lo | Delta_MARD_ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM | 0 | 12 | 9.831 | 2.039 | 8.761 | 10.90 | 20.53 | 18.69 | 22.42 | 0.000 | 0.000 | 0.000 |
| LightGBM | 5 | 12 | 9.921 | 2.230 | 8.764 | 11.11 | 20.55 | 18.70 | 22.43 | -0.090 | -0.278 | 0.085 |
| LightGBM | 10 | 12 | 9.768 | 2.235 | 8.645 | 10.99 | 20.32 | 18.52 | 22.20 | 0.063 | -0.137 | 0.236 |
| LightGBM | 20 | 12 | 9.655 | 2.056 | 8.592 | 10.75 | 20.27 | 18.42 | 22.19 | 0.176 | 0.055 | 0.300 |
| LightGBM | 30 | 12 | 9.667 | 2.055 | 8.601 | 10.75 | 20.24 | 18.39 | 22.16 | 0.165 | 0.060 | 0.290 |
| LightGBM | 50 | 12 | 9.631 | 2.111 | 8.531 | 10.77 | 20.14 | 18.28 | 22.11 | 0.201 | 0.080 | 0.329 |
| Persistence | 0 | 12 | 11.93 | 2.330 | 10.68 | 13.11 | 24.95 | 22.80 | 27.06 | 0.000 | 0.000 | 0.000 |
| Persistence | 5 | 12 | 11.93 | 2.330 | 10.68 | 13.11 | 24.95 | 22.80 | 27.06 | 0.000 | 0.000 | 0.000 |
| Persistence | 10 | 12 | 11.93 | 2.330 | 10.68 | 13.11 | 24.95 | 22.80 | 27.06 | 0.000 | 0.000 | 0.000 |
| Persistence | 20 | 12 | 11.93 | 2.330 | 10.68 | 13.11 | 24.95 | 22.80 | 27.06 | 0.000 | 0.000 | 0.000 |
| Persistence | 30 | 12 | 11.93 | 2.330 | 10.68 | 13.11 | 24.95 | 22.80 | 27.06 | 0.000 | 0.000 | 0.000 |
| Persistence | 50 | 12 | 11.93 | 2.330 | 10.68 | 13.11 | 24.95 | 22.80 | 27.06 | 0.000 | 0.000 | 0.000 |
| Ridge | 0 | 12 | 10.50 | 2.070 | 9.412 | 11.59 | 21.89 | 19.96 | 23.95 | 0.000 | 0.000 | 0.000 |
| Ridge | 5 | 12 | 11.03 | 2.397 | 9.754 | 12.26 | 22.93 | 20.78 | 25.11 | -0.530 | -1.012 | -0.027 |
| Ridge | 10 | 12 | 10.70 | 2.507 | 9.380 | 12.03 | 22.23 | 20.15 | 24.34 | -0.199 | -0.682 | 0.295 |
| Ridge | 20 | 12 | 10.35 | 2.318 | 9.137 | 11.58 | 21.67 | 19.69 | 23.75 | 0.152 | -0.137 | 0.458 |
| Ridge | 30 | 12 | 10.24 | 2.122 | 9.133 | 11.36 | 21.54 | 19.52 | 23.57 | 0.261 | 0.024 | 0.533 |
| Ridge | 50 | 12 | 10.15 | 2.146 | 9.028 | 11.31 | 21.36 | 19.32 | 23.43 | 0.352 | 0.087 | 0.645 |

| model | p_level | n_improved | n_patients | mean_diff | diff_ci_lo | diff_ci_hi | p |
|---|---|---|---|---|---|---|---|
| LightGBM | 5 | 4 | 12 | 0.090 | -0.085 | 0.278 | 0.424 |
| LightGBM | 10 | 9 | 12 | -0.063 | -0.236 | 0.137 | 0.233 |
| LightGBM | 20 | 9 | 12 | -0.176 | -0.300 | -0.055 | 0.042 |
| LightGBM | 30 | 11 | 12 | -0.165 | -0.290 | -0.060 | 0.002 |
| LightGBM | 50 | 11 | 12 | -0.201 | -0.329 | -0.080 | 0.012 |
| Ridge | 5 | 3 | 12 | 0.530 | 0.027 | 1.012 | 0.092 |
| Ridge | 10 | 5 | 12 | 0.199 | -0.295 | 0.682 | 0.519 |
| Ridge | 20 | 6 | 12 | -0.152 | -0.458 | 0.137 | 0.470 |
| Ridge | 30 | 8 | 12 | -0.261 | -0.533 | -0.024 | 0.110 |
| Ridge | 50 | 8 | 12 | -0.352 | -0.645 | -0.087 | 0.052 |

| model | from_level | to_level | n_patients | mean_diff | diff_ci_lo | diff_ci_hi | p |
|---|---|---|---|---|---|---|---|
| LightGBM | 0 | 5 | 12 | 0.090 | -0.085 | 0.278 | 0.424 |
| LightGBM | 5 | 10 | 12 | -0.153 | -0.257 | -0.048 | 0.027 |
| LightGBM | 10 | 20 | 12 | -0.113 | -0.254 | -0.022 | 0.007 |
| LightGBM | 20 | 30 | 12 | 0.011 | -0.050 | 0.074 | 0.970 |
| LightGBM | 30 | 50 | 12 | -0.036 | -0.116 | 0.035 | 0.424 |
| Ridge | 0 | 5 | 12 | 0.530 | 0.027 | 1.012 | 0.092 |
| Ridge | 5 | 10 | 12 | -0.331 | -0.614 | -0.048 | 0.077 |
| Ridge | 10 | 20 | 12 | -0.350 | -0.673 | -0.089 | 0.021 |
| Ridge | 20 | 30 | 12 | -0.110 | -0.257 | 0.034 | 0.176 |
| Ridge | 30 | 50 | 12 | -0.091 | -0.178 | -0.008 | 0.077 |

- LightGBM: the smallest level with a significant MARD improvement is p_level 20 (median 9.6 days of data; 9/12 improved; mean change -0.18 points, p=0.0425).
- LightGBM: mean MARD change between consecutive levels: 0->5: 0.09 (p=0.424); 5->10: -0.15 (p=0.027); 10->20: -0.11 (p=0.007); 20->30: 0.01 (p=0.970); 30->50: -0.04 (p=0.424).
- Ridge: no level shows a significant MARD improvement over P=0 (p<0.05).
- Ridge: mean MARD change between consecutive levels: 0->5: 0.53 (p=0.092); 5->10: -0.33 (p=0.077); 10->20: -0.35 (p=0.021); 20->30: -0.11 (p=0.176); 30->50: -0.09 (p=0.077).

## C. Objective 3: external validation on HUPA-UCM

| analysis | model | n | MARD_mean | MARD_sd | MARD_ci_lo | MARD_ci_hi | RMSE_mean | RMSE_ci_lo | RMSE_ci_hi | EGA_A+B_mean | EGA_A+B_ci_lo | EGA_A+B_ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MAIN | Persistence | 19 | 13.03 | 1.998 | 12.19 | 13.89 | 24.44 | 21.76 | 27.00 | 97.89 | 97.56 | 98.21 |
| MAIN | Ridge | 19 | 12.22 | 1.487 | 11.56 | 12.84 | 21.98 | 19.90 | 24.04 | 98.04 | 97.52 | 98.49 |
| MAIN | RandomForest | 19 | 13.24 | 1.846 | 12.46 | 14.11 | 22.05 | 20.25 | 23.86 | 95.53 | 94.47 | 96.41 |
| MAIN | MLP | 19 | 14.21 | 2.133 | 13.28 | 15.23 | 23.81 | 22.07 | 25.58 | 95.60 | 94.49 | 96.48 |
| MAIN | LightGBM | 19 | 13.38 | 1.924 | 12.55 | 14.26 | 21.95 | 20.21 | 23.72 | 95.74 | 94.74 | 96.55 |
| MAIN | Stacking | 19 | 12.83 | 1.836 | 12.04 | 13.69 | 21.46 | 19.73 | 23.23 | 96.23 | 95.32 | 96.98 |
| ALL | Persistence | 23 | 13.16 | 1.846 | 12.42 | 13.88 | 24.97 | 22.58 | 27.19 | 97.89 | 97.60 | 98.18 |
| ALL | Ridge | 23 | 12.20 | 1.439 | 11.61 | 12.76 | 22.30 | 20.53 | 24.07 | 98.10 | 97.64 | 98.51 |
| ALL | RandomForest | 23 | 13.37 | 1.930 | 12.60 | 14.15 | 22.51 | 20.91 | 24.12 | 95.54 | 94.58 | 96.39 |
| ALL | MLP | 23 | 14.35 | 2.252 | 13.47 | 15.30 | 24.22 | 22.68 | 25.82 | 95.61 | 94.58 | 96.49 |
| ALL | LightGBM | 23 | 13.49 | 2.002 | 12.70 | 14.31 | 22.46 | 20.86 | 24.07 | 95.82 | 94.98 | 96.59 |
| ALL | Stacking | 23 | 12.91 | 1.876 | 12.17 | 13.67 | 21.91 | 20.34 | 23.49 | 96.31 | 95.56 | 96.99 |

| analysis | model | n_beats_persistence | n_patients | mean_diff | diff_ci_lo | diff_ci_hi | p |
|---|---|---|---|---|---|---|---|
| MAIN | Ridge | 13 | 19 | -0.809 | -1.429 | -0.186 | 0.036 |
| MAIN | RandomForest | 11 | 19 | 0.209 | -0.491 | 1.028 | 0.829 |
| MAIN | MLP | 4 | 19 | 1.180 | 0.317 | 2.244 | 0.014 |
| MAIN | LightGBM | 10 | 19 | 0.343 | -0.383 | 1.235 | 0.953 |
| MAIN | Stacking | 13 | 19 | -0.208 | -0.923 | 0.638 | 0.182 |
| ALL | Ridge | 17 | 23 | -0.959 | -1.500 | -0.436 | 0.004 |
| ALL | RandomForest | 14 | 23 | 0.210 | -0.417 | 0.929 | 0.754 |
| ALL | MLP | 5 | 23 | 1.189 | 0.406 | 2.166 | 0.009 |
| ALL | LightGBM | 13 | 23 | 0.337 | -0.314 | 1.107 | 0.823 |
| ALL | Stacking | 16 | 23 | -0.246 | -0.880 | 0.470 | 0.170 |

| comparison | model | n_ohio | n_hupa | ohio_mean | hupa_mean | U | p |
|---|---|---|---|---|---|---|---|
| Ohio LOSO vs HUPA MAIN | Persistence | 12 | 19 | 12.11 | 13.03 | 85.00 | 0.248 |
| Ohio LOSO vs HUPA MAIN | Ridge | 12 | 19 | 10.70 | 12.22 | 52.00 | 0.013 |
| Ohio LOSO vs HUPA MAIN | RandomForest | 12 | 19 | 10.13 | 13.24 | 22.00 | 2.07e-04 |
| Ohio LOSO vs HUPA MAIN | MLP | 12 | 19 | 10.03 | 14.21 | 7.000 | 1.57e-05 |
| Ohio LOSO vs HUPA MAIN | LightGBM | 12 | 19 | 10.03 | 13.38 | 17.00 | 9.09e-05 |
| Ohio LOSO vs HUPA MAIN | Stacking | 12 | 19 | 10.03 | 12.83 | 27.00 | 4.51e-04 |
| Ohio LOSO vs HUPA ALL | Persistence | 12 | 23 | 12.11 | 13.16 | 96.00 | 0.149 |
| Ohio LOSO vs HUPA ALL | Ridge | 12 | 23 | 10.70 | 12.20 | 64.00 | 0.011 |
| Ohio LOSO vs HUPA ALL | RandomForest | 12 | 23 | 10.13 | 13.37 | 24.00 | 8.00e-05 |
| Ohio LOSO vs HUPA ALL | MLP | 12 | 23 | 10.03 | 14.35 | 7.000 | 5.76e-06 |
| Ohio LOSO vs HUPA ALL | LightGBM | 12 | 23 | 10.03 | 13.49 | 19.00 | 3.82e-05 |
| Ohio LOSO vs HUPA ALL | Stacking | 12 | 23 | 10.03 | 12.91 | 30.00 | 1.87e-04 |

- Gap decomposition (from `results_objective3/followup/`). The eval-rule and sampling effects are paired per OhioT1DM patient (Wilcoxon, n=12). The remaining effect is HUPA MAIN (19) minus Ohio C (12), unpaired (independent bootstrap CI, Mann-Whitney p).

| model | ohio_A | ohio_B | ohio_C | hupa_main | eval_rule_effect | eval_rule_p | sampling_effect | sampling_ci_lo | sampling_ci_hi | sampling_p | remaining_effect | remaining_ci_lo | remaining_ci_hi | remaining_mannwhitney_p | total_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistence | 12.11 | 12.14 | 12.14 | 13.03 | 0.025 | 0.233 | -0.004 | -0.042 | 0.036 | 0.519 | 0.897 | -0.474 | 2.279 | 0.265 | 0.919 |
| Ridge | 10.70 | 10.68 | 11.87 | 12.22 | -0.021 | 0.569 | 1.184 | 0.893 | 1.504 | 4.88e-04 | 0.357 | -0.748 | 1.451 | 0.612 | 1.520 |
| RandomForest | 10.13 | 10.08 | 11.49 | 13.24 | -0.047 | 0.052 | 1.408 | 1.203 | 1.609 | 4.88e-04 | 1.753 | 0.516 | 3.028 | 0.016 | 3.114 |
| MLP | 10.03 | 9.969 | 11.72 | 14.21 | -0.063 | 0.021 | 1.754 | 1.512 | 1.983 | 4.88e-04 | 2.491 | 1.130 | 3.881 | 0.005 | 4.182 |
| LightGBM | 10.02 | 9.974 | 11.57 | 13.38 | -0.051 | 0.064 | 1.594 | 1.380 | 1.818 | 4.88e-04 | 1.808 | 0.530 | 3.131 | 0.014 | 3.352 |
| Stacking | 10.03 | 9.982 | 11.27 | 12.83 | -0.045 | 0.092 | 1.287 | 1.154 | 1.406 | 4.88e-04 | 1.556 | 0.375 | 2.796 | 0.027 | 2.798 |

- Glucose-range metrics are pooled over points; CIs come from a patient-level (cluster) bootstrap. Ohio predictions are restricted to the Objective 1 common mask (row counts equal n_test_samples: True).

| dataset | model | range | n_points | n_patients_with_points | MARD | MARD_ci_lo | MARD_ci_hi | RMSE | RMSE_ci_lo | RMSE_ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| Ohio LOSO | Persistence | <70 | 5536 | 12 | 24.02 | 20.01 | 29.15 | 21.69 | 18.22 | 26.17 |
| Ohio LOSO | Persistence | 70-180 | 106187 | 12 | 13.00 | 12.02 | 13.97 | 22.76 | 21.19 | 24.15 |
| Ohio LOSO | Persistence | >180 | 56486 | 12 | 9.289 | 8.210 | 10.26 | 29.08 | 26.09 | 31.70 |
| Ohio LOSO | Ridge | <70 | 5536 | 12 | 26.99 | 24.40 | 30.91 | 21.38 | 18.45 | 25.33 |
| Ohio LOSO | Ridge | 70-180 | 106187 | 12 | 11.31 | 10.47 | 12.15 | 19.39 | 17.99 | 20.72 |
| Ohio LOSO | Ridge | >180 | 56486 | 12 | 7.975 | 7.334 | 8.566 | 25.96 | 23.72 | 28.04 |
| Ohio LOSO | RandomForest | <70 | 5536 | 12 | 32.59 | 28.65 | 37.45 | 24.01 | 20.83 | 27.98 |
| Ohio LOSO | RandomForest | 70-180 | 106187 | 12 | 10.33 | 9.581 | 11.09 | 18.17 | 16.83 | 19.42 |
| Ohio LOSO | RandomForest | >180 | 56486 | 12 | 7.576 | 6.877 | 8.178 | 24.71 | 22.50 | 26.71 |
| Ohio LOSO | MLP | <70 | 5536 | 12 | 30.26 | 25.30 | 36.26 | 23.23 | 19.65 | 27.69 |
| Ohio LOSO | MLP | 70-180 | 106187 | 12 | 10.33 | 9.625 | 11.03 | 18.16 | 16.89 | 19.46 |
| Ohio LOSO | MLP | >180 | 56486 | 12 | 7.509 | 6.716 | 8.194 | 24.71 | 22.14 | 26.99 |
| Ohio LOSO | LightGBM | <70 | 5536 | 12 | 33.13 | 29.41 | 37.68 | 23.81 | 20.72 | 27.67 |
| Ohio LOSO | LightGBM | 70-180 | 106187 | 12 | 10.21 | 9.485 | 10.94 | 17.93 | 16.65 | 19.17 |
| Ohio LOSO | LightGBM | >180 | 56486 | 12 | 7.468 | 6.775 | 8.067 | 24.44 | 22.19 | 26.48 |
| Ohio LOSO | Stacking | <70 | 5536 | 12 | 32.27 | 28.77 | 36.60 | 23.32 | 20.34 | 27.12 |
| Ohio LOSO | Stacking | 70-180 | 106187 | 12 | 10.22 | 9.467 | 10.97 | 18.06 | 16.74 | 19.31 |
| Ohio LOSO | Stacking | >180 | 56486 | 12 | 7.519 | 6.814 | 8.129 | 24.54 | 22.29 | 26.56 |
| HUPA MAIN | Persistence | <70 | 6600 | 19 | 15.92 | 14.86 | 19.28 | 14.74 | 12.79 | 16.72 |
| HUPA MAIN | Persistence | 70-180 | 73678 | 19 | 11.64 | 10.97 | 14.68 | 19.37 | 18.04 | 24.37 |
| HUPA MAIN | Persistence | >180 | 19990 | 19 | 8.821 | 7.999 | 10.57 | 26.70 | 23.34 | 32.66 |
| HUPA MAIN | Ridge | <70 | 6600 | 19 | 21.83 | 21.27 | 22.95 | 16.11 | 14.14 | 17.12 |
| HUPA MAIN | Ridge | 70-180 | 73678 | 19 | 11.29 | 10.84 | 13.20 | 18.17 | 17.14 | 21.97 |
| HUPA MAIN | Ridge | >180 | 19990 | 19 | 7.400 | 6.696 | 8.891 | 22.76 | 19.91 | 27.97 |
| HUPA MAIN | RandomForest | <70 | 6600 | 19 | 30.88 | 29.82 | 35.78 | 21.83 | 20.37 | 23.87 |
| HUPA MAIN | RandomForest | 70-180 | 73678 | 19 | 11.30 | 10.89 | 13.32 | 17.80 | 17.04 | 21.00 |
| HUPA MAIN | RandomForest | >180 | 19990 | 19 | 7.179 | 6.500 | 8.722 | 22.27 | 19.30 | 27.65 |
| HUPA MAIN | MLP | <70 | 6600 | 19 | 33.05 | 30.88 | 39.97 | 24.03 | 22.80 | 28.33 |
| HUPA MAIN | MLP | 70-180 | 73678 | 19 | 12.02 | 11.59 | 14.21 | 19.05 | 18.23 | 22.76 |
| HUPA MAIN | MLP | >180 | 19990 | 19 | 7.514 | 6.820 | 9.081 | 23.21 | 20.24 | 28.76 |
| HUPA MAIN | LightGBM | <70 | 6600 | 19 | 31.57 | 29.84 | 37.02 | 21.60 | 20.53 | 23.97 |
| HUPA MAIN | LightGBM | 70-180 | 73678 | 19 | 11.35 | 10.91 | 13.49 | 17.89 | 17.05 | 21.45 |
| HUPA MAIN | LightGBM | >180 | 19990 | 19 | 7.038 | 6.401 | 8.492 | 21.58 | 18.87 | 26.66 |
| HUPA MAIN | Stacking | <70 | 6600 | 19 | 28.96 | 27.19 | 33.87 | 19.77 | 18.86 | 21.65 |
| HUPA MAIN | Stacking | 70-180 | 73678 | 19 | 10.94 | 10.53 | 12.93 | 17.46 | 16.64 | 20.82 |
| HUPA MAIN | Stacking | >180 | 19990 | 19 | 7.065 | 6.440 | 8.488 | 21.66 | 18.97 | 26.65 |

- HUPA MAIN: Ridge has lower MARD than Persistence in 13/19 patients (mean difference -0.81, Wilcoxon p=0.0361).
- HUPA MAIN: RandomForest has lower MARD than Persistence in 11/19 patients (mean difference 0.21, Wilcoxon p=0.8288).
- HUPA MAIN: MLP has lower MARD than Persistence in 4/19 patients (mean difference 1.18, Wilcoxon p=0.0141).
- HUPA MAIN: LightGBM has lower MARD than Persistence in 10/19 patients (mean difference 0.34, Wilcoxon p=0.9530).
- HUPA MAIN: Stacking has lower MARD than Persistence in 13/19 patients (mean difference -0.21, Wilcoxon p=0.1819).
- HUPA ALL: Ridge has lower MARD than Persistence in 17/23 patients (mean difference -0.96, Wilcoxon p=0.0043).
- HUPA ALL: RandomForest has lower MARD than Persistence in 14/23 patients (mean difference 0.21, Wilcoxon p=0.7540).
- HUPA ALL: MLP has lower MARD than Persistence in 5/23 patients (mean difference 1.19, Wilcoxon p=0.0091).
- HUPA ALL: LightGBM has lower MARD than Persistence in 13/23 patients (mean difference 0.34, Wilcoxon p=0.8229).
- HUPA ALL: Stacking has lower MARD than Persistence in 16/23 patients (mean difference -0.25, Wilcoxon p=0.1695).
- Persistence: per-patient MARD Ohio 12.11 vs HUPA MAIN 13.03 (Mann-Whitney p=0.2478, n=12 vs 19).
- Ridge: per-patient MARD Ohio 10.70 vs HUPA MAIN 12.22 (Mann-Whitney p=0.0126, n=12 vs 19).
- RandomForest: per-patient MARD Ohio 10.13 vs HUPA MAIN 13.24 (Mann-Whitney p=0.0002, n=12 vs 19).
- MLP: per-patient MARD Ohio 10.03 vs HUPA MAIN 14.21 (Mann-Whitney p=0.0000, n=12 vs 19).
- LightGBM: per-patient MARD Ohio 10.03 vs HUPA MAIN 13.38 (Mann-Whitney p=0.0001, n=12 vs 19).
- Stacking: per-patient MARD Ohio 10.03 vs HUPA MAIN 12.83 (Mann-Whitney p=0.0005, n=12 vs 19).
- Persistence: sampling effect (C-B) -0.00 MARD points (95% CI -0.04 to 0.04, Wilcoxon p=0.5186, n=12).
- Ridge: sampling effect (C-B) 1.18 MARD points (95% CI 0.89 to 1.50, Wilcoxon p=0.0005, n=12).
- RandomForest: sampling effect (C-B) 1.41 MARD points (95% CI 1.20 to 1.61, Wilcoxon p=0.0005, n=12).
- MLP: sampling effect (C-B) 1.75 MARD points (95% CI 1.51 to 1.98, Wilcoxon p=0.0005, n=12).
- LightGBM: sampling effect (C-B) 1.59 MARD points (95% CI 1.38 to 1.82, Wilcoxon p=0.0005, n=12).
- Stacking: sampling effect (C-B) 1.29 MARD points (95% CI 1.15 to 1.41, Wilcoxon p=0.0005, n=12).

## D. HUPA-UCM data quality: Preprocessed glucose vs raw Libre readings

- For each Preprocessed timestamp: is a raw reading within 2 min; is the value within 1 mg/dL of the linear interpolation between the surrounding raw readings; does it need a FUTURE reading (it differs by more than 1 mg/dL from the last earlier reading but matches the interpolation). Percentages are of the timestamps inside the raw-reading range.

| patient | reference | n_preprocessed | n_inside_raw_range | pct_timestamp_within_2min_of_raw | pct_within_1_of_linear_interp | pct_within_1_of_last_earlier | pct_needs_future_reading | pct_in_gaps_over_30min | median_abs_diff_to_interp |
|---|---|---|---|---|---|---|---|---|---|
| HUPA0001P | raw Libre type-0 | 4096 | 4001 | 32.35 | 11.75 | 10.70 | 5.099 | 0.175 | 9.533 |
| HUPA0002P | raw Libre type-0 | 3181 | 3177 | 32.91 | 41.01 | 25.50 | 18.35 | 0.787 | 1.467 |
| HUPA0003P | raw Libre type-0 | 3770 | 3768 | 33.18 | 32.70 | 17.94 | 17.81 | 0.000 | 2.067 |
| HUPA0004P | raw Libre type-0 | 3184 | 3181 | 28.36 | 28.48 | 15.91 | 14.74 | 15.53 | 2.933 |
| HUPA0005P | raw Libre type-0 | 3858 | 3841 | 33.02 | 31.53 | 19.08 | 15.91 | 0.000 | 2.200 |
| HUPA0006P | raw Libre type-0 | 2290 | 2281 | 31.75 | 35.99 | 20.25 | 18.15 | 4.910 | 1.867 |
| HUPA0007P | raw Libre type-0 | 3857 | 3845 | 33.11 | 24.99 | 15.58 | 13.63 | 0.000 | 3.000 |
| HUPA0011P | raw Libre type-0 | 3839 | 3835 | 33.26 | 27.54 | 16.19 | 14.47 | 0.000 | 2.667 |
| HUPA0014P | raw Libre type-0 | 3829 | 3825 | 28.05 | 31.27 | 15.53 | 18.07 | 16.84 | 2.133 |
| HUPA0015P | raw Libre type-0 | 3792 | 3789 | 32.20 | 38.35 | 19.08 | 21.14 | 3.352 | 1.733 |
| HUPA0016P | raw Libre type-0 | 3835 | 3831 | 28.42 | 32.76 | 17.18 | 18.38 | 15.30 | 2.000 |
| HUPA0017P | raw Libre type-0 | 3599 | 3578 | 30.90 | 36.98 | 19.76 | 19.96 | 7.239 | 2.000 |
| HUPA0018P | raw Libre type-0 | 3895 | 3893 | 32.66 | 36.09 | 20.86 | 17.16 | 1.207 | 1.867 |
| HUPA0019P | raw Libre type-0 | 3711 | 3706 | 33.23 | 24.88 | 21.78 | 11.36 | 0.000 | 2.933 |
| HUPA0020P | raw Libre type-0 | 2862 | 2857 | 33.05 | 32.80 | 17.50 | 18.34 | 0.000 | 2.200 |
| HUPA0021P | raw Libre type-0 | 2343 | 2338 | 33.12 | 33.32 | 17.45 | 18.31 | 0.000 | 2.000 |
| HUPA0022P | raw Libre type-0 | 4023 | 3907 | 32.19 | 43.36 | 25.39 | 19.84 | 0.384 | 1.333 |
| HUPA0023P | raw Libre type-0 | 3919 | 3914 | 33.12 | 41.19 | 23.12 | 21.18 | 0.230 | 1.467 |
| HUPA0024P | raw Libre type-0 | 2902 | 2898 | 33.05 | 37.78 | 20.36 | 20.12 | 0.000 | 1.667 |
| HUPA0025P | raw Libre type-0 | 4006 | 4000 | 33.03 | 40.05 | 23.97 | 19.40 | 0.525 | 1.467 |
| HUPA0026P | raw Libre type-0 | 40605 | 40578 | 27.46 | 33.10 | 18.22 | 17.33 | 18.13 | 2.267 |
| HUPA0027P | raw Libre type-0 | 165306 | 165301 | 31.20 | 36.85 | 22.05 | 17.27 | 6.335 | 1.800 |
| HUPA0028P | raw Libre type-0 | 25902 | 19007 | 21.99 | 41.58 | 23.72 | 19.84 | 10.61 | 1.400 |
| HUPA0001P | Medtronic sensor glucose | 4096 | 4095 | 82.74 | 19.88 | 16.68 | 3.956 | 17.36 | 4.400 |

- For the 22 patients other than HUPA0001P, 34.7% of Preprocessed glucose values on average (range 24.9-43.4%; 95% CI 32.5-36.8) equal the linear interpolation of raw Libre readings, and 17.8% (range 11.4-21.2%) can only be reproduced by using a later raw reading.
- Only 31.3% of Preprocessed timestamps lie within 2 min of a real raw reading (mean over the 22 patients).
- HUPA0001P: only 11.7% of its Preprocessed values match the Libre interpolation (median difference 9.5 mg/dL), but 19.9% match the interpolation of its Medtronic pump sensor glucose (median difference 4.4 mg/dL, 82.7% of timestamps within 2 min of a Medtronic reading).

## E. Integrity

**Parser check (results_objective3/hupa_parser_check_log.txt)**

```
CHECK  1 SCHEMA            PASS
CHECK  2 FEATURES          PASS
CHECK  3 CAUSALITY         PASS
CHECK  4 RAW GLUCOSE ONLY  PASS
CHECK  5 CLIPPING          PASS
CHECK  6 UNITS             PASS
CHECK  7 EXCLUSIONS        PASS
CHECK  8 NO IMPUTATION     PASS
CHECK  9 NON-EMPTY         PASS
CHECK 10 DATES PARSED      PASS
CHECK 11 DATE SANITY       PASS
CHECK 12 WINDOW            PASS
CHECK 13 COUNTS            PASS
OVERALL: PASS
```

**Objective 3 audit (results_objective3/audit_objective3_log.txt)**

```
PASS  23 x 6 patient-model rows  (138 rows, 23 patients)
PASS  MAIN subset 19 x 6  (114 rows)
PASS  no duplicate patient-model rows
PASS  no NaN or infinite metric values  (0 bad)
PASS  no duplicate prediction rows
PASS  summary main: means and SDs equal recomputed patient-level values
PASS  summary all: means and SDs equal recomputed patient-level values
PASS  Persistence predictions equal glucose at t for every evaluated row  (0 mismatches over 105,423 rows)
PASS  model file hashes equal model_hashes.json  (9 files; mismatched []; unhashed [])
PASS  train_final_ohio.py (and objective3_common.py) contain no HUPA-UCM path or import  ([])
PASS  n_samples per patient equals n_evaluable_points in hupa_parser_check.csv  (differs: [])
OVERALL: PASS (11/11)
```

**Objective 3 follow-up (results_objective3/followup/followup_checks_log.txt)**

```
PASS  1 condition A reproduces Objective 1 (mean MARD within 0.1)  (differences {'Persistence': 0.0, 'Ridge': -0.0, 'RandomForest': 0.0, 'MLP': 0.0, 'LightGBM': -0.0086, 'Stacking': -0.0017})
PASS  2 12 patients x 6 models x 3 conditions, no duplicates, no NaN/inf  (216 rows)
PASS  3 condition C median interval 14-16 min for every patient and offset  (range 15.00-15.00 min)
PASS  4 Persistence equals glucose at t on every evaluated row (A, B, C)  (0 mismatches)
PASS  5 no HUPA-UCM data path or parser import in this script  (lines [])
OVERALL: PASS
```

- Objective 2: no audit log file exists in results_objective2/ (audit_objective2.py saves no log).
- Tags in data/OhioT1DM/559-ws-training.xml: basal, basis_air_temperature, basis_gsr, basis_heart_rate, basis_skin_temperature, basis_sleep, basis_steps, bolus, exercise, finger_stick, glucose_level, hypo_event, illness, meal, sleep, stressors, temp_basal, work.
- parse_xml_file looks for `.//heart_rate` elements. Across the 24 OhioT1DM XML files there are 0 such elements and 90020 `basis_heart_rate` events (12 files; the 2020 cohort has none).
- parse_xml_file on 559-ws-training.xml returns 0 non-NaN heart_rate values; 0.00% of heart_rate values in the Objective 1 features are non-NaN (results_causal/patient_features_causal.pkl).
- Objective 3's saved imputer fills heart_rate with 0.0 (keep_empty_features=True on an all-NaN column).
- No model in Objectives 1-3 received real heart-rate data: the heart_rate feature is NaN in all rows and becomes a constant after imputation.

## F. Methods facts

- Features: 47 (glucose, bolus, carbs, heart_rate, iob, glucose_diff1, glucose_diff2, glucose_roll_mean_30, glucose_roll_std_30, hour_sin, hour_cos, glucose_lag_1, glucose_lag_2, glucose_lag_3, glucose_lag_4, glucose_lag_5, glucose_lag_6, glucose_lag_7, glucose_lag_8, glucose_lag_9, glucose_lag_10, glucose_lag_11, glucose_lag_12, glucose_diff1_lag_1, glucose_diff1_lag_2, glucose_diff1_lag_3, glucose_diff1_lag_4, glucose_diff1_lag_5, glucose_diff1_lag_6, glucose_diff1_lag_7, glucose_diff1_lag_8, glucose_diff1_lag_9, glucose_diff1_lag_10, glucose_diff1_lag_11, glucose_diff1_lag_12, iob_lag_1, iob_lag_2, iob_lag_3, iob_lag_4, iob_lag_5, iob_lag_6, iob_lag_7, iob_lag_8, iob_lag_9, iob_lag_10, iob_lag_11, iob_lag_12).
- IOB: exponential kernel exp(-t/t_peak), normalized, DIA = 240 min, t_peak = 60 min, 5-min steps (create_features defaults).
- Prediction horizon: target = glucose shifted by 6 steps = 30 minutes.

| model | settings |
|---|---|
| imputer | SimpleImputer(strategy='mean', keep_empty_features=True) + np.nan_to_num(nan=0.0) |
| scaler | StandardScaler() |
| Ridge | {"alpha": 1.0} |
| RandomForest | {"n_estimators": 50, "max_depth": 15, "min_samples_leaf": 5, "max_samples": 0.5, "n_jobs": -1} |
| MLP | {"hidden_layer_sizes": [128, 64], "max_iter": 150, "early_stopping": true, "n_iter_no_change": 10} |
| LightGBM | {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "n_jobs": -1, "verbose": -1, "early_stopping": "none (plain fit, as Objective 1)"} |
| Stacking | {"estimators": "Ridge(1.0) + RF(30, depth 12, leaf 5, max_samples 0.5) + LGBM(100, lr 0.05, 31 leaves)", "final_estimator": "Ridge(alpha=1.0)", "cv": 3, "n_jobs": 1} |
| Persistence | prediction = glucose at time t |
| random_state | 42 |

| objective | patients | rows | note |
|---|---|---|---|
| 1 LOSO | 12 | 167376 | evaluated test rows summed over the 12 folds (common mask) |
| 1/3 training (all 12) | 12 | 169221 | feature rows, OhioT1DM all files |
| 2 personalization | 12 | 83646 | evaluation rows per level; levels [np.int64(0), np.int64(5), np.int64(10), np.int64(20), np.int64(30), np.int64(50)] |
| 3 HUPA MAIN | 19 | 99855 | evaluable points |
| 3 HUPA ALL | 23 | 105423 | evaluable points |

- Exclusions (HUPA-UCM): HUPA0009P and HUPA0010P have no raw Libre CGM (excluded from both analyses); HUPA0011P, HUPA0015P, HUPA0018P and HUPA0020P have missing insulin or meal events (excluded from MAIN only).
- HUPA-UCM study window: only raw Libre readings between the first and last `time` of Preprocessed/<PID>.csv are used. Glucose is clipped to [40, 400]. Bolus <= 0 is treated as no event.
- Objective 1 was run in an earlier environment whose LightGBM version was not recorded.

| library | version |
|---|---|
| python | 3.13.9 |
| lightgbm | 4.7.0 |
| scikit-learn | 1.7.2 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |

