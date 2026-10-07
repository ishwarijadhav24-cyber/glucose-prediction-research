/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useState, useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { api, UploadPredictionResponse } from "@/lib/api";
import { parseFileLocal, ParsedData } from "@/lib/parser";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { GlucoseChart } from "@/components/glucose-chart";
import { Activity, UploadCloud, Brain, AlertTriangle, FileUp, X, CheckCircle2 } from "lucide-react";
import { format } from "date-fns";

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [localData, setLocalData] = useState<ParsedData | null>(null);
  const [predictionResponse, setPredictionResponse] = useState<UploadPredictionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setSelectedFile(file);
    setPredictionResponse(null);
    setErrorMsg(null);
    setLocalData(null);

    // Locally parse file for rendering
    try {
      const data = await parseFileLocal(file);
      setLocalData(data);
    } catch (e) {
      console.error("Local parse failed", e);
      // We don't block upload if local parse fails, backend is source of truth
    }
  };

  const uploadMutation = useMutation({
    mutationFn: () => {
      if (!selectedFile) throw new Error("No file selected");
      // Use mode="latest" since the UI only visualizes the latest prediction right now
      return api.predictUpload([selectedFile], "latest");
    },
    onSuccess: (data) => {
      setPredictionResponse(data);
      setErrorMsg(null);
    },
    onError: (err: any) => {
      if (err.retryAfter) {
        setErrorMsg(`Rate limit exceeded. Please try again in ${err.retryAfter} seconds.`);
      } else {
        setErrorMsg(err.message || "An error occurred during prediction.");
      }
    }
  });

  const clearSelection = () => {
    setSelectedFile(null);
    setLocalData(null);
    setPredictionResponse(null);
    setErrorMsg(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="container max-w-screen-xl py-12 space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Upload & Predict</h1>
        <p className="text-muted-foreground mt-2">
          Upload custom CGM exports (CSV/XML) to run inference using the LightGBM research model.
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        
        {/* Sidebar / Upload Controls */}
        <div className="lg:col-span-1 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Data Source</CardTitle>
              <CardDescription>Select a CSV or XML file</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {!selectedFile ? (
                <div 
                  className="border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer hover:bg-muted/50 transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <UploadCloud className="w-10 h-10 text-muted-foreground mb-4" />
                  <p className="font-medium">Click to select file</p>
                  <p className="text-sm text-muted-foreground mt-1">Maximum 50MB</p>
                  <input 
                    type="file" 
                    className="hidden" 
                    ref={fileInputRef} 
                    accept=".csv,.xml" 
                    onChange={handleFileChange} 
                  />
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="border rounded-xl p-4 flex items-center justify-between bg-muted/30">
                    <div className="flex items-center gap-3 overflow-hidden">
                      <FileUp className="w-5 h-5 text-primary shrink-0" />
                      <div className="truncate">
                        <p className="font-medium text-sm truncate">{selectedFile.name}</p>
                        <p className="text-xs text-muted-foreground">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                      </div>
                    </div>
                    <Button variant="ghost" size="icon" onClick={clearSelection}>
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                  {localData && !predictionResponse && (
                    <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-500 font-medium px-1">
                      <CheckCircle2 className="w-4 h-4" />
                      File ready for inference
                    </div>
                  )}
                </div>
              )}

              <Button 
                className="w-full" 
                size="lg"
                disabled={!selectedFile || uploadMutation.isPending}
                onClick={() => uploadMutation.mutate()}
              >
                {uploadMutation.isPending ? (
                  <>
                    <Activity className="w-4 h-4 mr-2 animate-spin" />
                    Running 30-minute prediction...
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4 mr-2" />
                    Analyze & Predict
                  </>
                )}
              </Button>

              {errorMsg && (
                <div className="p-3 bg-destructive/10 border border-destructive/20 text-destructive text-sm rounded-md">
                  {errorMsg}
                </div>
              )}
            </CardContent>
          </Card>

          {predictionResponse?.dataset && (
            <Card>
              <CardHeader>
                <CardTitle>Dataset Profile</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Type</span>
                  <span className="font-medium capitalize">{predictionResponse.dataset.detected_type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Glucose Readings</span>
                  <span className="font-medium">{predictionResponse.dataset.n_glucose_readings.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Interval</span>
                  <span className="font-medium">{predictionResponse.dataset.median_interval_minutes} mins</span>
                </div>
                <div className="flex justify-between">
                  <span 
                    className="text-muted-foreground cursor-help underline decoration-dotted" 
                    title={`Percentage of the expected ${predictionResponse.dataset.median_interval_minutes}-minute glucose timeline containing valid readings.`}
                  >
                    Coverage
                  </span>
                  <span className="font-medium">{predictionResponse.dataset.glucose_coverage_percent.toFixed(1)}%</span>
                </div>
                {predictionResponse.dataset.start && predictionResponse.dataset.end && (
                  <div className="flex justify-between pt-2 border-t mt-3">
                    <span className="text-muted-foreground">Date Range</span>
                    <span className="font-medium text-right ml-4">
                      {format(new Date(predictionResponse.dataset.start), "MMM d")} - {format(new Date(predictionResponse.dataset.end), "MMM d, yyyy")}
                    </span>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Main Content Area */}
        <div className="lg:col-span-2 space-y-6">
          {predictionResponse?.warnings && predictionResponse.warnings.length > 0 && (
            <div className="bg-yellow-500/10 border border-yellow-500/20 text-yellow-600 dark:text-yellow-500 p-4 rounded-xl flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
              <ul className="list-disc list-inside text-sm space-y-1">
                {predictionResponse.warnings.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </div>
          )}

          {predictionResponse && (
            <div className="grid sm:grid-cols-2 gap-4">
              {/* Prediction Result Card */}
              <Card className="border-primary/50 shadow-sm shadow-primary/10 overflow-hidden relative">
                <CardContent className="p-6">
                  <div className="text-sm font-semibold tracking-wider text-muted-foreground mb-2 uppercase">
                    Predicted Glucose
                  </div>
                  <div className="text-5xl font-bold text-primary tracking-tighter flex items-baseline gap-2 drop-shadow-md">
                    {predictionResponse.prediction.predicted_glucose_mg_dl.toFixed(1)}
                    <span className="text-xl font-normal text-muted-foreground">mg/dL</span>
                  </div>
                  <div className="text-sm font-medium text-muted-foreground mt-1">
                    30-MINUTE FORECAST
                  </div>
                  <div className="text-sm text-muted-foreground mt-4 flex items-center gap-2 border-t pt-3">
                    <span>Forecast time:</span>
                    <span className="font-medium text-foreground">
                      {format(new Date(predictionResponse.prediction.forecast_time), "MMM d, HH:mm")}
                    </span>
                  </div>
                </CardContent>
              </Card>

              {/* Result Summary / Model */}
              <Card>
                <CardContent className="p-6 flex flex-col justify-center space-y-3 h-full">
                   <div className="flex justify-between items-center">
                     <span className="text-muted-foreground text-sm">Model</span>
                     <span className="font-medium text-sm text-right">{predictionResponse.model.name} {predictionResponse.model.version}</span>
                   </div>
                   <div className="flex justify-between items-center">
                     <span className="text-muted-foreground text-sm">Data Type</span>
                     <span className="font-medium text-sm text-right">{predictionResponse.dataset.detected_type}</span>
                   </div>
                   <div className="flex justify-between items-center">
                     <span className="text-muted-foreground text-sm">Sampling</span>
                     <span className="font-medium text-sm text-right">{predictionResponse.dataset.median_interval_minutes} mins</span>
                   </div>
                   <div className="flex justify-between items-center border-t pt-2 mt-1">
                     <span className="text-muted-foreground text-sm">Horizon</span>
                     <span className="font-medium text-sm text-right">30 minutes</span>
                   </div>
                </CardContent>
              </Card>
            </div>
          )}

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <div className="space-y-1">
                <CardTitle>30-Minute Glucose Forecast</CardTitle>
                <CardDescription>Visualizing localized data alongside backend inference</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              {localData ? (
                <GlucoseChart
                  historicalGlucose={localData.glucose}
                  carbs={localData.carbs}
                  bolus={localData.bolus}
                  predictions={predictionResponse ? predictionResponse.predictions.map(p => ({
                    forecast_time: p.forecast_time,
                    prediction_time: p.prediction_time,
                    value: p.predicted_glucose_mg_dl
                  })) : []}
                  currentPredictionTime={predictionResponse?.prediction?.prediction_time}
                  currentForecastTime={predictionResponse?.prediction?.forecast_time}
                />
              ) : (
                <div className="h-[400px] flex flex-col items-center justify-center border-2 border-dashed rounded-xl bg-muted/20">
                  <Activity className="w-10 h-10 text-muted-foreground mb-4 opacity-50" />
                  <p className="text-muted-foreground">Upload a glucose dataset to visualize the forecast.</p>
                </div>
              )}
              {predictionResponse && (
                <div className="mt-4 text-xs text-muted-foreground bg-muted/50 p-3 rounded-md border text-center">
                  Observed glucose at the forecast time is not available in this upload, so prediction error cannot be calculated for this forecast.
                </div>
              )}
            </CardContent>
          </Card>

          {predictionResponse && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Safety Interpretation</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-sm">
                  {predictionResponse.prediction.predicted_glucose_mg_dl < 70 ? (
                    <span className="text-red-500 font-medium">Predicted glucose is below the 70 mg/dL reference threshold. Review your CGM reading and follow your prescribed hypoglycemia care plan.</span>
                  ) : predictionResponse.prediction.predicted_glucose_mg_dl > 180 ? (
                    <span className="text-orange-500 font-medium">Predicted glucose is above the 180 mg/dL reference threshold. Review your CGM reading and follow your prescribed care plan.</span>
                  ) : (
                    <span className="text-green-500 font-medium">Predicted glucose is within the 70–180 mg/dL reference range.</span>
                  )}
                </div>
                <div className="text-xs text-muted-foreground bg-muted p-3 rounded-md">
                  Research interpretation only. This prediction is not intended for diagnosis or treatment decisions.
                </div>
              </CardContent>
            </Card>
          )}

        </div>

      </div>
    </div>
  );
}
