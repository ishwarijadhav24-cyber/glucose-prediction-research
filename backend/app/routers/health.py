from fastapi import APIRouter, Request

from backend.app.routers.common import service
from backend.app.schemas import ApiHealth, Health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health, summary="Liveness check")
def health():
    return Health()


@router.get("/api/v1/health", response_model=ApiHealth, summary="API and model status")
def api_health(request: Request):
    svc = service(request)
    art = svc.artifacts
    return ApiHealth(model_loaded=True, model_name=art.model_name, model_version=art.model_version,
                     environment=request.app.state.settings.environment)
