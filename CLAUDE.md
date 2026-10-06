# Project: 30-minute blood glucose prediction (Type 1 diabetes)
## Status
- Objectives 1 (LOSO) and 2 (personalization) are COMPLETE and LOCKED (git tag: obj1-obj2-locked). Trained on OhioT1DM.
- Objective 3 (in progress, branch "objective3"): external validation on HUPA-UCM. OhioT1DM-trained models are tested on HUPA-UCM WITHOUT retraining. Same features, same preprocessing, same models.
## Never do this
- Never modify, move or delete: data_processing.py, evaluate_loso.py, evaluate_personalization.py, evaluate_ohio_official_test.py, audit_objective2.py, eda_pca.py, train.py, preprocess_test.py, explore_hupa.py, diagnose_hupa_formats.py, .gitignore, .gitattributes, or anything in results/, results_causal/, results_objective2/, results_ohio_official/, results_pre_causal/, outputs/, outputs_causal/, outputs_objective2/, outputs_pre_causal/.
- Never write into data/OhioT1DM/. Never run "git add" on anything in data/.
- Never commit or push unless I explicitly ask.
- Never weaken, skip or delete a check to make it pass. Never add patients to an exclusion list to make a check pass. If a check fails, report it and explain why.
- Never impute or estimate missing insulin or meal data.
## Objective 3 data rules (decided before any results)
- HUPA-UCM glucose comes ONLY from raw FreeStyle Libre files (data/HUPA-UCM/Raw_Data/<PID>/free_style_sensor/), type-0 historic readings. Never use the glucose column of data/HUPA-UCM/Preprocessed/ (it is interpolated and leaks future data).
- Bolus and carbs come from Preprocessed/<PID>.csv (separator ";"): bolus = bolus_volume_delivered, carbs = carb_input x 10 (servings to grams).
- Causal glucose rule, identical to the OhioT1DM parser (parse_xml_file in data_processing.py): Step A reindex(grid, method="ffill", tolerance=5min); Step B ffill(limit=6). 5-minute grid, index named "ts", float columns glucose, bolus, carbs, heart_rate (heart_rate = NaN).
- Only use raw readings inside each patient's study window (first to last "time" in Preprocessed/<PID>.csv).
- Clip glucose to [40, 400].
- Bolus values <= 0 are invalid records and are treated as no event (HUPA0017P: 4 records).
- EXCLUDED_NO_CGM = HUPA0009P, HUPA0010P (no raw Libre CGM; excluded from both analyses).
- EXCLUDED_MISSING_EVENTS = HUPA0011P, HUPA0015P, HUPA0018P, HUPA0020P (excluded from MAIN, included in ALL).
- MAIN analysis = 19 patients. ALL analysis = 23 patients.
## Environment
- Windows, PowerShell. Run scripts from the project root.
