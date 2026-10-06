from fastapi import APIRouter, Request

from backend.app.routers.common import service
from backend.app.schemas import ModelInfo
from backend.app.services.csv_input import RANGES, TS_FORMATS
from backend.app.services.features import EVENT_COLUMNS

router = APIRouter(prefix="/api/v1", tags=["model"])


@router.get("/model", response_model=ModelInfo, summary="Deployed model and its research evaluation")
def model_info(request: Request):
    svc = service(request)
    card = request.app.state.model_card
    c = svc.constants
    return ModelInfo(
        model_name=card["model_name"], model_version=svc.artifacts.model_version,
        feature_version=svc.artifacts.feature_version, n_features=svc.artifacts.n_features,
        features=list(svc.artifacts.feature_list), prediction_horizon_minutes=c["horizon_minutes"],
        sampling_interval_minutes=card["sampling_interval_minutes"],
        minimum_history_minutes=(c["min_history_rows"] - 1) * 5,
        training_data=card["training_data"], library_versions=card["library_versions"],
        research_evaluation=card["research_evaluation"], external_validation=card["external_validation"],
        limitations=card["limitations"],
        input_format={
            "upload": "POST /api/v1/predict/upload, multipart field 'file' (1-4 files of ONE dataset), "
                      "query mode=latest|all",
            "formats": {
                "ohiot1dm_xml": "OhioT1DM XML, 1-2 files of the same patient (training/testing). Events: "
                                "glucose_level value, bolus dose at ts_begin, meal carbs (g); "
                                "timestamps DD-MM-YYYY HH:MM:SS. DOCTYPE/ENTITY declarations are rejected.",
                "hupa_ucm": "FreeStyle Libre export(s) (OLD 'ID;Hora;...' or NEW 'Dispositivo;...' layout; delimiter "
                            "; , or tab) + optional HUPA-UCM Preprocessed CSV (time;...;bolus_volume_delivered;"
                            "carb_input). Type-0 historic glucose only; Preprocessed glucose is never read.",
                "normalized_csv": {"columns": list(EVENT_COLUMNS), "timestamp_formats": list(TS_FORMATS),
                                   "timezone": "local time, no time zone offset",
                                   "notes": "UTF-8, comma-separated; one row per event; empty cells mean no event; "
                                            "duplicate timestamps within a signal are rejected."},
            },
            "value_ranges": {k: list(v) for k, v in RANGES.items()},
            "sampling": "glucose about every 5 minutes (native, as in training) or about every 15 minutes "
                        "(FreeStyle Libre historic data; accepted with external-validation warnings); others rejected",
            "not_used": "heart rate and any other signal (the model never received them)",
        },
    )
