# GlucoCast frontend

Next.js 14 (App Router) + Tailwind + Plotly website for the 30-minute glucose forecasting research project.
It talks to the FastAPI backend in `../backend` (see `../FRONTEND_HANDOFF.md`).

> Research demonstration only. Not a medical device. Not for diagnosis, treatment decisions or insulin dosing.
> Does not replace a CGM.

## Run locally

```bash
npm ci
```

```bash
npm run dev
```

The site runs at http://localhost:3000. It expects the backend at `NEXT_PUBLIC_API_BASE_URL`
(default `http://127.0.0.1:7860`; see `.env.example`). The backend allows `http://localhost:3000` in development.

## Pages

| Route | What it shows |
|---|---|
| `/` | Overview: the question, headline results, the three objectives, the pipeline |
| `/method` | Datasets (used / excluded and why), causal processing, the 47 features, the 6 models, an animated LOSO fold player, an interactive error / Clarke-zone explainer |
| `/research/loso` | Objective 1: all 6 models with a metric switcher, per-patient comparison, Wilcoxon tests, research figures |
| `/research/personalization` | Objective 2: personal-data slider, personalization curve (average or one patient), ΔMARD heatmap |
| `/research/external` | Objective 3: OhioT1DM vs HUPA-UCM for every model, gap-decomposition waterfall, per-patient model-vs-baseline scatter |
| `/research/analyses` | Error by glucose range, low-glucose alert test, event-bucketing leakage audit, heart-rate finding, limitations |
| `/lab` | Forecast Lab: upload CGM data (or the synthetic demo patient), forecasts at every 5-minute point vs real glucose, replay slider, scores vs the no-change baseline, Clarke grid of the file |

`/demo` and `/upload` redirect to `/lab`.

## Where the numbers come from

All research numbers come from `src/data/research.json`, generated from the project's result files — never type
numbers into pages by hand. Regenerate after any change to the results (from the repository root):

```bash
python frontend/scripts/build_research_data.py
```

`public/figures/` holds the original summary figures from the research scripts (no single-patient traces, because
of the dataset licences). `public/samples/synthetic_01.csv` is the synthetic demo patient and CSV template.

## Forecast Lab details

- Forecasts always come from the backend (`POST /api/v1/predict/upload?mode=all`).
- `src/lib/parser.ts` reads the same files in the browser only to draw the real glucose line and to score each
  forecast against the real reading at its target time (latest reading in the 5 minutes before it). It follows the
  backend's reading rules: Libre type-0 historic glucose only, HUPA Preprocessed file for insulin/carbs only
  (servings × 10), glucose clipped to 40–400.
- `src/lib/metrics.ts` implements MARD, RMSE, MAE, R² and the Clarke zones exactly as in `evaluate_loso.py`.
- Scores on one uploaded file describe one person and are not research results; OhioT1DM files are in-sample for
  the deployed model (it was trained on all 12 patients).

## Wording rules

Use "Upload CGM Data", never "Upload Medical Record". No hypoglycemia alerts, no safe/unsafe labels, no dosing
advice. Always show the disclaimer.
