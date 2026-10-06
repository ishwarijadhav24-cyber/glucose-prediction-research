"""Single 30-minute forecast from real observations (no recursion, no retraining).

Pipeline for a prediction at grid time t:
  1. keep only events with timestamp <= t (and within the context window)
  2. a real glucose reading must exist in (t - 5 min, t]; the research grid ends at
     ceil(last reading), so without a reading in that interval row t would only exist
     if a later (future) reading extended the grid
  3. events -> research grid (parse_xml_file via in-memory XML)
  4. the last min_history_rows grid rows ending at t must all have glucose
  5. 47 features at t (features_at adapter), schema/order and finiteness checks
  6. saved imputer -> nan_to_num -> scaler (objective3_common.apply_preprocessing, transform only)
  7. model.predict -> one value for t + 30 min

Step 1 makes serving strictly causal. Note: the research grid assigns a bolus/meal
to bucket floor(event, 5 min), so in training data row t also contained events from
(t, t + 5 min). At serving time those events are in the future and are excluded.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backend.app.errors import BackendError, ErrorCode
from backend.app.services.artifacts import LoadedArtifacts
from backend.app.services.features import (CONTEXT, EVENT_COLUMNS, FIVE_MIN, PLACEHOLDER_GLUCOSE, build_grid,
                                           features_at, pipeline_constants)
from data_processing import create_features
from objective3_common import apply_preprocessing

IOB_HISTORY = pd.Timedelta("240min")   # create_features default DIA


@dataclass(frozen=True)
class PredictionResult:
    predicted_glucose_mg_dl: float
    prediction_time: pd.Timestamp          # grid time t the forecast is made at
    latest_observation_time: pd.Timestamp  # last real glucose reading used (<= t)
    forecast_time: pd.Timestamp            # t + horizon
    horizon_minutes: int
    warnings: tuple = field(default_factory=tuple)
    features: pd.Series = field(default=None, repr=False, compare=False)   # internal, not for API output


class InferenceService:
    def __init__(self, artifacts: LoadedArtifacts):
        self.artifacts = artifacts
        self.constants = pipeline_constants()
        self.feature_list = list(artifacts.feature_list)

    def default_prediction_time(self, events: pd.DataFrame) -> pd.Timestamp:
        g = events.loc[events["glucose_mg_dl"].notna(), "timestamp"]
        if g.empty:
            raise BackendError(ErrorCode.MISSING_CURRENT_GLUCOSE, "No glucose readings in the input.")
        return g.max().ceil("5min")    # last grid row parse_xml_file builds

    def predict(self, events: pd.DataFrame, t: pd.Timestamp | None = None) -> PredictionResult:
        missing = [c for c in EVENT_COLUMNS if c not in events.columns]
        if missing:
            raise BackendError(ErrorCode.MISSING_COLUMNS, f"Missing columns: {missing}")
        if t is None:
            t = self.default_prediction_time(events)
        t = pd.Timestamp(t)
        if t != t.floor("5min"):
            raise BackendError(ErrorCode.INVALID_TIMESTAMP, "Prediction time must be on a 5-minute boundary.")

        # 1. causal cut: nothing after t is ever seen
        ev = events.loc[(events["timestamp"] <= t) & (events["timestamp"] >= t - CONTEXT), list(EVENT_COLUMNS)]
        g_times = ev.loc[ev["glucose_mg_dl"].notna(), "timestamp"]
        # 2. real reading in (t - 5 min, t]
        if g_times.empty or t - g_times.max() >= FIVE_MIN:
            raise BackendError(ErrorCode.MISSING_CURRENT_GLUCOSE,
                               "No glucose reading within the 5 minutes before the prediction time.")
        latest_obs = g_times.max()

        # 3. research grid
        grid = build_grid(ev)
        # 4. history
        need = self.constants["min_history_rows"]
        window = grid.loc[:t].iloc[-need:]
        if (len(window) < need or window.index[-1] != t
                or (window.index[-1] - window.index[0]) != (need - 1) * FIVE_MIN
                or window["glucose"].isna().any()):
            raise BackendError(ErrorCode.INSUFFICIENT_HISTORY,
                               f"At least {need} consecutive 5-minute glucose values "
                               f"({(need - 1) * 5} minutes) are needed up to the prediction time.")

        # 5. features
        feats = features_at(grid, t, self.feature_list)
        if list(feats.index) != self.feature_list:
            raise BackendError(ErrorCode.FEATURE_SCHEMA_MISMATCH, "Feature names/order differ from the model.")
        values = feats.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise BackendError(ErrorCode.INFINITE_FEATURE, "A feature value is infinite.")
        expected_nan = {"heart_rate"}   # never available, as in training (imputed constant)
        bad_nan = [n for n, v in feats.items() if np.isnan(v) and n not in expected_nan]
        if bad_nan:
            raise BackendError(ErrorCode.INSUFFICIENT_HISTORY, "Features could not be computed from the history.")

        # 6. preprocessing: transform only
        X = apply_preprocessing(self.artifacts.imputer, self.artifacts.scaler, values.reshape(1, -1))
        if not np.isfinite(X).all():
            raise BackendError(ErrorCode.NAN_AFTER_PREPROCESSING, "Non-finite value after preprocessing.")

        # 7. one prediction
        try:
            pred = float(self.artifacts.model.predict(X)[0])
        except Exception as exc:    # noqa: BLE001 - surfaced as a structured error
            raise BackendError(ErrorCode.PREDICTION_FAILED, "The model could not produce a prediction.") from exc
        if not np.isfinite(pred):
            raise BackendError(ErrorCode.PREDICTION_FAILED, "The model produced a non-finite prediction.")

        warnings = []
        if t - ev["timestamp"].min() < IOB_HISTORY:
            warnings.append("Less than 4 hours of history: insulin on board may be underestimated.")
        if ev["bolus_units"].notna().sum() == 0 and ev["carbs_g"].notna().sum() == 0:
            warnings.append("No insulin or carbohydrate records in the input window.")
        horizon = self.constants["horizon_minutes"]
        return PredictionResult(predicted_glucose_mg_dl=pred, prediction_time=t, latest_observation_time=latest_obs,
                                forecast_time=t + pd.Timedelta(minutes=horizon), horizon_minutes=horizon,
                                warnings=tuple(warnings), features=feats)


    # ------------------------------------------------------------------ many prediction times
    def eligible_times(self, events: pd.DataFrame, grid: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict]:
        """Grid times t with a real reading in (t - 5 min, t] and min_history_rows of glucose ending at t."""
        g = events.loc[events["glucose_mg_dl"].notna(), "timestamp"]
        cand = pd.DatetimeIndex(sorted(set(g.dt.ceil("5min")))).intersection(grid.index)
        need = self.constants["min_history_rows"]
        has = grid["glucose"].notna().astype(int).rolling(need).sum() == need   # regular 5-min grid
        ok = has.reindex(cand).fillna(False).values
        return cand[ok], {"INSUFFICIENT_HISTORY": int((~ok).sum())}

    def predict_many(self, events: pd.DataFrame, max_predictions: int) -> tuple[list, dict, bool]:
        """Independent strictly-causal forecasts at every eligible time (latest max_predictions).

        Equivalent to calling predict(events, t) for each t (tested). For speed the features come
        from one create_features pass over the full grid; rows whose research bucket would contain
        a bolus/meal from (t, t + 5 min) are recomputed with the strictly causal single path.
        Predictions are never fed back as inputs.
        """
        missing = [c for c in EVENT_COLUMNS if c not in events.columns]
        if missing:
            raise BackendError(ErrorCode.MISSING_COLUMNS, f"Missing columns: {missing}")
        events = events[list(EVENT_COLUMNS)].sort_values("timestamp", kind="mergesort")
        grid = build_grid(events)
        times, skipped = self.eligible_times(events, grid)
        truncated = len(times) > max_predictions
        times = times[-max_predictions:]
        if len(times) == 0:
            raise BackendError(ErrorCode.INSUFFICIENT_HISTORY,
                               f"No time point has {self.constants['min_history_rows']} consecutive 5-minute glucose "
                               "values ending at a real reading.")
        k = self.constants["horizon_rows"]
        future = pd.DataFrame({"glucose": PLACEHOLDER_GLUCOSE, "bolus": 0.0, "carbs": 0.0, "heart_rate": np.nan},
                              index=pd.date_range(grid.index[-1] + FIVE_MIN, periods=k, freq="5min", name="ts"))
        ext = pd.concat([grid[["glucose", "bolus", "carbs", "heart_rate"]], future])
        ext.index.name = "ts"
        full = create_features(ext).set_index("ts")[self.feature_list]
        ev_t = events.loc[events["bolus_units"].notna() | events["carbs_g"].notna(), "timestamp"].values
        g_t = events.loc[events["glucose_mg_dl"].notna(), "timestamp"].values
        rows, latest, recompute = [], [], 0
        for t in times:
            t64 = np.datetime64(t)
            after = ((ev_t > t64) & (ev_t < t64 + np.timedelta64(5, "m"))).any()
            if after:
                res = self.predict(events, t)        # strictly causal single path
                rows.append(res.features.values)
                recompute += 1
            else:
                rows.append(full.loc[t].to_numpy(dtype=float))
            latest.append(pd.Timestamp(g_t[np.searchsorted(g_t, t64, side="right") - 1]))
        F = np.vstack(rows)
        hr = self.feature_list.index("heart_rate")
        other = np.delete(F, hr, axis=1)
        if np.isinf(F).any():
            raise BackendError(ErrorCode.INFINITE_FEATURE, "A feature value is infinite.")
        if np.isnan(other).any():
            raise BackendError(ErrorCode.INSUFFICIENT_HISTORY, "Features could not be computed from the history.")
        X = apply_preprocessing(self.artifacts.imputer, self.artifacts.scaler, F)
        if not np.isfinite(X).all():
            raise BackendError(ErrorCode.NAN_AFTER_PREPROCESSING, "Non-finite value after preprocessing.")
        try:
            preds = self.artifacts.model.predict(X)
        except Exception as exc:    # noqa: BLE001
            raise BackendError(ErrorCode.PREDICTION_FAILED, "The model could not produce a prediction.") from exc
        if not np.isfinite(preds).all():
            raise BackendError(ErrorCode.PREDICTION_FAILED, "The model produced a non-finite prediction.")
        horizon = pd.Timedelta(minutes=self.constants["horizon_minutes"])
        out = [PredictionResult(predicted_glucose_mg_dl=float(p), prediction_time=t, latest_observation_time=lo,
                                forecast_time=t + horizon, horizon_minutes=self.constants["horizon_minutes"],
                                features=pd.Series(f, index=self.feature_list))
               for p, t, lo, f in zip(preds, times, latest, F)]
        skipped["recomputed_causal_bucket"] = recompute
        return out, skipped, truncated
