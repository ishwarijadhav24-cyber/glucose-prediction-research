/* eslint-disable @typescript-eslint/no-unused-vars */
"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Users, Calendar, Activity, ArrowRight } from "lucide-react";
import Link from "next/link";
import { format } from "date-fns";

export default function DemoPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["demo-patients"],
    queryFn: api.getDemoPatients,
  });

  return (
    <div className="container max-w-screen-xl py-12 space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Demo Patients</h1>
        <p className="text-muted-foreground mt-2">
          Select a pre-loaded patient dataset to run inference and explore the predictive model.
        </p>
      </div>

      {isLoading && (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Card key={i} className="flex flex-col">
              <CardHeader className="gap-2">
                <Skeleton className="h-6 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
              </CardHeader>
              <CardContent className="flex-1 space-y-4">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-10 w-full mt-4" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {isError && (
        <div className="p-6 bg-destructive/10 text-destructive rounded-xl border border-destructive/20">
          Failed to load demo patients. Ensure the backend is running at {process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:7860"}.
        </div>
      )}

      {data && (
        <>
          <div className="bg-muted/50 p-4 rounded-xl border text-sm mb-6 flex items-start gap-3">
            <Activity className="w-5 h-5 text-primary shrink-0" />
            <p className="text-muted-foreground">{data.note}</p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {data.patients.map((patient) => (
              <Card key={patient.patient_id} className="flex flex-col hover:border-primary/50 transition-colors group">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Patient {patient.patient_id}</span>
                    <Badge variant="secondary">Demo</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col justify-between">
                  <div className="space-y-4 mb-6 text-sm text-muted-foreground">
                    <div className="flex items-center gap-3">
                      <Activity className="w-4 h-4 text-primary" />
                      <span>{patient.n_glucose_readings.toLocaleString()} Readings</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <Calendar className="w-4 h-4 text-primary" />
                      <span>
                        {format(new Date(patient.start), "MMM d, yyyy")} - {format(new Date(patient.end), "MMM d, yyyy")}
                      </span>
                    </div>
                  </div>
                  
                  <Button className="w-full group-hover:bg-primary group-hover:text-primary-foreground transition-all" asChild>
                    <Link href={`/demo/${patient.patient_id}`}>
                      Analyze Patient
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
