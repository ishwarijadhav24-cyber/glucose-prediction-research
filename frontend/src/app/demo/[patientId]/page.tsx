"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import { useQuery, useMutation } from "@tanstack/react-query";
import { api, Prediction, ModelRef } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { GlucoseChart } from "@/components/glucose-chart";
import { Activity, Brain, AlertTriangle, ShieldCheck, TrendingDown, TrendingUp, Info } from "lucide-react";
import { format } from "date-fns";

export default function DemoPatientDashboard() {
  const params = useParams();
  const patientId = params.patientId as string;

  const [predictionResult, setPredictionResult] = useState<{
    prediction: Prediction;
    model: ModelRef;
    warnings: string[];
    actual_glucose_mg_dl_at_forecast_time: number | null;
  } | null>(null);

  // Fetch Patient History
  const { data: patient, isLoading, isError } = useQuery({
    queryKey: ["demo-patient", patientId],
    queryFn: () => api.getDemoPatient(patientId),
  });

  // Fetch full model info for research performance
  const { data: modelInfoFull } = useQuery({
    queryKey: ["model-info"],
    queryFn: () => api.getModel(),
  });

  // Run Inference Mutation
  const predictMutation = useMutation({
    mutationFn: () => api.predictDemo(patientId),
    onSuccess: (data) => {
      setPredictionResult(data);
    }
  });

  if (isLoading) {
    return (
      <div className="container py-8 space-y-6 max-w-screen-xl">
        <Skeleton className="h-10 w-1/3" />
        <Skeleton className="h-[400px] w-full" />
      </div>
    );
  }

  if (isError || !patient) {
    return (
      <div className="container py-8 max-w-screen-xl">
        <div className="p-6 bg-destructive/10 text-destructive rounded-xl">
          Failed to load patient data.
        </div>
      </div>
    );
  }

  const chartHistoricalGlucose = patient.glucose.map(g => ({ timestamp: g.timestamp, value: g.glucose_mg_dl }));
  const chartPredictions = predictionResult?.prediction ? [{
    forecast_time: predictionResult.prediction.forecast_time,
    value: predictionResult.prediction.predicted_glucose_mg_dl,
    prediction_time: predictionResult.prediction.prediction_time
  }] : [];

  const prediction = predictionResult?.prediction;
  const actual = predictionResult?.actual_glucose_mg_dl_at_forecast_time;
  const modelInfo = predictionResult?.model;
  const warnings = predictionResult?.warnings || [];

  let absError = null;
  let absPercError = null;
  if (actual != null && prediction != null) {
    absError = Math.abs(prediction.predicted_glucose_mg_dl - actual);
    absPercError = (absError / actual) * 100;
  }

  let safetyStatus = "";
  let safetyMsg = "";
  let SafetyIcon = Activity;
  let safetyColor = "";

  if (prediction) {
    if (prediction.predicted_glucose_mg_dl < 70) {
      safetyStatus = "Low glucose";
      safetyMsg = "Predicted glucose is below the displayed 70–180 mg/dL target range. Review your CGM reading and follow your prescribed hypoglycemia care plan.";
      SafetyIcon = TrendingDown;
      safetyColor = "text-orange-500 bg-orange-500/10 border-orange-500/20";
    } else if (prediction.predicted_glucose_mg_dl > 180) {
      safetyStatus = "Elevated glucose";
      safetyMsg = "Predicted glucose is above the displayed 70–180 mg/dL target range. Continue monitoring and follow your prescribed diabetes care plan.";
      SafetyIcon = TrendingUp;
      safetyColor = "text-yellow-600 dark:text-yellow-500 bg-yellow-500/10 border-yellow-500/20";
    } else {
      safetyStatus = "Within target range";
      safetyMsg = "Predicted glucose is within the displayed 70–180 mg/dL target range. Continue monitoring according to your usual care plan.";
      SafetyIcon = ShieldCheck;
      safetyColor = "text-emerald-600 dark:text-emerald-500 bg-emerald-500/10 border-emerald-500/20";
    }
  }

  return (
    <div className="container max-w-screen-xl py-8 space-y-8">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            Patient {patient.patient_id}
            <Badge variant="secondary">Demo Data</Badge>
          </h1>
          <p className="text-muted-foreground mt-1">
            {patient.n_glucose_readings.toLocaleString()} Readings • 
            Ends at {format(new Date(patient.end), "MMM d, yyyy HH:mm")}
          </p>
        </div>
        
        <Button 
          onClick={() => predictMutation.mutate()}
          disabled={predictMutation.isPending}
          size="lg"
          className="w-full md:w-auto"
        >
          {predictMutation.isPending ? (
            <Activity className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <Brain className="w-4 h-4 mr-2" />
          )}
          Predict 30-Min Glucose
        </Button>
      </div>

      {warnings.length > 0 && (
        <div className="bg-yellow-500/10 border border-yellow-500/20 text-yellow-600 dark:text-yellow-500 p-4 rounded-xl flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
          <ul className="list-disc list-inside text-sm space-y-1">
            {warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Continuous Glucose Monitoring (CGM)</CardTitle>
          <CardDescription>Target Range: 70 - 180 mg/dL</CardDescription>
        </CardHeader>
        <CardContent>
          <GlucoseChart
            historicalGlucose={chartHistoricalGlucose}
            carbs={patient.carbs_g}
            bolus={patient.bolus_units}
            predictions={chartPredictions}
            currentPredictionTime={prediction?.prediction_time}
          />
        </CardContent>
      </Card>

      {prediction && modelInfo && (
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="border-primary/20 shadow-md">
            <CardHeader className="pb-4">
              <CardTitle>Prediction Result</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-col gap-1 pb-4 border-b">
                <div className="flex items-baseline gap-2">
                  <span className="text-5xl font-bold text-primary">
                    {Intl.NumberFormat('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 }).format(prediction.predicted_glucose_mg_dl)}
                  </span>
                  <span className="text-muted-foreground font-medium text-xl">mg/dL</span>
                </div>
                <p className="text-sm text-muted-foreground font-medium mt-1">
                  Predicted glucose
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm pt-2">
                <div>
                  <span className="text-muted-foreground block mb-1">Forecast Time</span>
                  <span className="font-medium">{format(new Date(prediction.forecast_time), "MMM d, yyyy • HH:mm")}</span>
                </div>
                <div>
                  <span className="text-muted-foreground block mb-1">Horizon</span>
                  <span className="font-medium">+{prediction.horizon_minutes} minutes</span>
                </div>
              </div>

              {actual != null && (
                <div className="bg-muted/40 rounded-lg p-3 grid grid-cols-3 gap-3 text-sm mt-2 border">
                  <div>
                    <span className="text-muted-foreground block text-xs mb-1">Actual</span>
                    <span className="font-semibold">{actual.toFixed(1)} mg/dL</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block text-xs mb-1">Prediction Error</span>
                    <span className="font-medium">{absError?.toFixed(1)} mg/dL</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block text-xs mb-1">Absolute Percentage Error</span>
                    <span className="font-medium">{absPercError?.toFixed(1)}%</span>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="space-y-6">
            <Card className={`border ${safetyColor}`}>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <SafetyIcon className="w-5 h-5" />
                  Safety Interpretation
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="font-semibold text-lg mb-2 leading-tight">
                    {Intl.NumberFormat('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 }).format(prediction.predicted_glucose_mg_dl)} mg/dL<br/>
                    <span className="opacity-90 font-medium text-base">{safetyStatus}</span>
                  </div>
                  <p className="text-sm leading-relaxed opacity-90">{safetyMsg}</p>
                </div>
                <div className="bg-background/50 rounded-md p-3 text-xs flex items-start gap-2 border border-current/10">
                  <Info className="w-4 h-4 shrink-0 mt-0.5" />
                  <p>
                    This research tool does not provide insulin, medication, dosing, diagnosis, or treatment recommendations.
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-lg">Model Metadata</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="grid grid-cols-2 gap-y-2">
                  <span className="text-muted-foreground">Architecture</span>
                  <span className="font-medium">{modelInfo.name}</span>
                  
                  <span className="text-muted-foreground">Model Version</span>
                  <span className="font-medium">{modelInfo.version}</span>
                  
                  <span className="text-muted-foreground">Feature Pipeline</span>
                  <span className="font-medium">{modelInfo.feature_version}</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {prediction && modelInfoFull?.research_evaluation && (
        <Card className="mt-8 bg-muted/30 border-muted-foreground/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              Research Model Performance
              <Badge variant="outline" className="font-normal text-xs bg-background">12-patient LOSO evaluation</Badge>
            </CardTitle>
            <CardDescription className="max-w-3xl">
              Research evaluation performance calculated over multiple samples (population averages). 
              These metrics represent overall model capability during cross-validation, not the accuracy of the single prediction above.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {modelInfoFull.research_evaluation.MARD_percent && (
                <div className="bg-background rounded-lg p-4 border shadow-sm">
                  <div className="text-sm text-muted-foreground font-medium mb-1">MARD</div>
                  <div className="text-2xl font-bold">
                    {modelInfoFull.research_evaluation.MARD_percent.mean?.toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    ±{modelInfoFull.research_evaluation.MARD_percent.sd?.toFixed(2)}
                  </div>
                </div>
              )}
              {modelInfoFull.research_evaluation.RMSE_mg_dl && (
                <div className="bg-background rounded-lg p-4 border shadow-sm">
                  <div className="text-sm text-muted-foreground font-medium mb-1">RMSE</div>
                  <div className="text-2xl font-bold">
                    {modelInfoFull.research_evaluation.RMSE_mg_dl.mean?.toFixed(1)} <span className="text-sm font-normal">mg/dL</span>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    ±{modelInfoFull.research_evaluation.RMSE_mg_dl.sd?.toFixed(2)}
                  </div>
                </div>
              )}
              {modelInfoFull.research_evaluation.R2 && (
                <div className="bg-background rounded-lg p-4 border shadow-sm">
                  <div className="text-sm text-muted-foreground font-medium mb-1">R²</div>
                  <div className="text-2xl font-bold">
                    {modelInfoFull.research_evaluation.R2.mean?.toFixed(3)}
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    ±{modelInfoFull.research_evaluation.R2.sd?.toFixed(3)}
                  </div>
                </div>
              )}
              {modelInfoFull.research_evaluation.clarke_A_plus_B_percent && (
                <div className="bg-background rounded-lg p-4 border shadow-sm">
                  <div className="text-sm text-muted-foreground font-medium mb-1">Clarke Error Grid A+B</div>
                  <div className="text-2xl font-bold">
                    {modelInfoFull.research_evaluation.clarke_A_plus_B_percent.mean?.toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    ±{modelInfoFull.research_evaluation.clarke_A_plus_B_percent.sd?.toFixed(2)}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
