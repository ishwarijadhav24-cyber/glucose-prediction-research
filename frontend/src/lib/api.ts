/* eslint-disable @typescript-eslint/no-unused-vars */
import { z } from "zod";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:7860";

export const ModelRefSchema = z.object({
  name: z.string(),
  version: z.string(),
  feature_version: z.string(),
});
export type ModelRef = z.infer<typeof ModelRefSchema>;

export const PredictionSchema = z.object({
  predicted_glucose_mg_dl: z.number(),
  prediction_time: z.string(),
  latest_observation_time: z.string(),
  forecast_time: z.string(),
  horizon_minutes: z.number(),
  actual_glucose_mg_dl_at_forecast_time: z.number().nullable().optional(),
});
export type Prediction = z.infer<typeof PredictionSchema>;
export const API_BASE_URL = API_BASE;

export const DemoPatientSummarySchema = z.object({
  patient_id: z.string(),
  n_glucose_readings: z.number(),
  start: z.string(),
  end: z.string(),
});
export type DemoPatientSummary = z.infer<typeof DemoPatientSummarySchema>;

export const GlucosePointSchema = z.object({
  timestamp: z.string(),
  glucose_mg_dl: z.number(),
});
export type GlucosePoint = z.infer<typeof GlucosePointSchema>;

export const EventPointSchema = z.object({
  timestamp: z.string(),
  value: z.number(),
});
export type EventPoint = z.infer<typeof EventPointSchema>;

export const DemoPatientDataSchema = DemoPatientSummarySchema.extend({
  glucose: z.array(GlucosePointSchema),
  bolus_units: z.array(EventPointSchema),
  carbs_g: z.array(EventPointSchema),
});
export type DemoPatientData = z.infer<typeof DemoPatientDataSchema>;

export const ModelInfoSchema = z.object({
  model_name: z.string(),
  model_version: z.string(),
  feature_version: z.string(),
  n_features: z.number(),
  features: z.array(z.string()),
  prediction_horizon_minutes: z.number(),
  sampling_interval_minutes: z.number(),
  minimum_history_minutes: z.number(),
  training_data: z.string(),
  library_versions: z.record(z.string(), z.any()),
  research_evaluation: z.record(z.string(), z.any()),
  external_validation: z.record(z.string(), z.any()),
  limitations: z.array(z.string()),
  input_format: z.record(z.string(), z.any()),
  disclaimer: z.string(),
});
export type ModelInfo = z.infer<typeof ModelInfoSchema>;

export const UploadPredictionResponseSchema = z.object({
  status: z.literal("success"),
  dataset: z.object({
    detected_type: z.string(),
    file_types: z.array(z.string()),
    sampling_profile: z.string(),
    median_interval_minutes: z.number(),
    n_glucose_readings: z.number(),
    n_bolus_events: z.number(),
    n_carb_events: z.number(),
    start: z.string(),
    end: z.string(),
    glucose_coverage_percent: z.number(),
  }),
  validation: z.object({
    status: z.literal("valid"),
    checks: z.array(z.string()),
  }),
  mode: z.enum(["latest", "all"]),
  predictions: z.array(PredictionSchema),
  prediction: PredictionSchema,
  n_predictions: z.number(),
  truncated: z.boolean(),
  skipped: z.record(z.string(), z.number()),
  preprocessing: z.array(z.string()),
  model: ModelRefSchema,
  warnings: z.array(z.string()),
  disclaimer: z.string(),
});
export type UploadPredictionResponse = z.infer<typeof UploadPredictionResponseSchema>;

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public retryAfter?: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, options);

  if (!response.ok) {
    let code = "UNKNOWN_ERROR";
    let message = "An unknown error occurred.";
    let retryAfter: number | undefined;

    if (response.status === 429) {
      retryAfter = parseInt(response.headers.get("Retry-After") || "60", 10);
      message = `Rate limit exceeded. Try again in ${retryAfter}s.`;
    } else if (response.status === 400) {
      message = "Bad request (400). Please check your file.";
    } else if (response.status === 413) {
      message = "The upload is too large (413). The backend limit is 25 MB in total.";
    } else if (response.status === 415) {
      message = "Unsupported media type (415). Please upload CSV or XML.";
    } else if (response.status === 422) {
      message = "Unprocessable entity (422). File format is invalid.";
    } else if (response.status === 500) {
      message = "Internal server error (500). The backend failed to process this file.";
    } else if (response.status === 503) {
      message = "Service unavailable (503). The prediction engine is busy.";
    } else if (response.status === 504) {
      message = "Gateway timeout (504). The prediction took too long.";
    }

    try {
      const errorData = await response.json();
      if (errorData.error) {
        code = errorData.error.code;
        message = errorData.error.message || message;
      }
    } catch (e) {
      // Keep default mapped message if no JSON
    }

    throw new ApiError(response.status, code, message, retryAfter);
  }

  return response.json();
}

export const api = {
  getHealth: () => fetchApi<{ status: string }>("/health"),
  getApiHealth: () => fetchApi<{ status: string; model_loaded: boolean; model_name: string; model_version: string; environment: string }>("/api/v1/health"),
  getModel: () => fetchApi<ModelInfo>("/api/v1/model"),
  getDemoPatients: () => fetchApi<{ patients: DemoPatientSummary[]; note: string }>("/api/v1/demo/patients"),
  getDemoPatient: (patientId: string) => fetchApi<DemoPatientData>(`/api/v1/demo/patients/${patientId}`),
  predictDemo: (patientId: string, predictionTime?: string) => {
    const body = predictionTime ? JSON.stringify({ prediction_time: predictionTime }) : undefined;
    return fetchApi<{
      status: "success";
      prediction: Prediction;
      model: ModelRef;
      warnings: string[];
      disclaimer: string;
      actual_glucose_mg_dl_at_forecast_time: number | null;
    }>(`/api/v1/predict/demo/${patientId}`, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body,
    });
  },
  predictUpload: (files: File[], mode: "latest" | "all" = "all") => {
    const formData = new FormData();
    files.forEach((f) => formData.append("file", f));
    
    // Do NOT set Content-Type header when sending FormData!
    // fetch will automatically set it to multipart/form-data with the correct boundary
    return fetchApi<UploadPredictionResponse>(`/api/v1/predict/upload?mode=${mode}`, {
      method: "POST",
      body: formData,
    });
  },
};
