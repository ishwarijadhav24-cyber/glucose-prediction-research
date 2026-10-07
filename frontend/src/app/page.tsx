"use client";

import Link from "next/link";
import { ArrowRight, Activity, BrainCircuit, ShieldAlert, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";

export default function Home() {
  return (
    <div className="flex-1 flex flex-col">
      {/* Hero Section */}
      <section className="relative px-6 pt-24 pb-32 md:pt-32 md:pb-40 overflow-hidden flex flex-col items-center justify-center min-h-[70vh]">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-primary/10 via-background to-background" />
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="container flex flex-col items-center text-center space-y-8 max-w-4xl"
        >
          <div className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 border-transparent bg-secondary text-secondary-foreground">
            <Activity className="w-3.5 h-3.5 mr-1.5 text-primary" />
            Research Preview
          </div>
          
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-b from-foreground to-foreground/70">
            Causal Glucose Forecasting
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-[42rem] leading-normal sm:leading-8">
            A state-of-the-art LightGBM architecture for continuous glucose monitoring (CGM). 
            Predicting 30 minutes into the future with strict causal adherence.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 mt-8 w-full sm:w-auto">
            <Button size="lg" className="h-12 px-8 text-base group" asChild>
              <Link href="/demo">
                View Live Demo
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Link>
            </Button>
            <Button size="lg" variant="outline" className="h-12 px-8 text-base" asChild>
              <Link href="/upload">
                Upload Custom Data
              </Link>
            </Button>
          </div>
        </motion.div>
      </section>

      {/* Capabilities Section */}
      <section className="py-24 bg-muted/30 border-t">
        <div className="container max-w-screen-xl">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold tracking-tight mb-4">Architecture & Capabilities</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Built on rigorous causal inference principles to ensure predictions rely solely on strictly historical data.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-8">
            <FeatureCard
              icon={<BrainCircuit className="h-8 w-8 text-primary" />}
              title="LightGBM Regressor"
              description="Tree-based gradient boosting optimized for tabular time-series data with handling for missing events."
            />
            <FeatureCard
              icon={<Zap className="h-8 w-8 text-primary" />}
              title="5-Min Grid Alignment"
              description="Data is automatically aligned to a strict 5-minute causal forward-fill grid before inference."
            />
            <FeatureCard
              icon={<ShieldAlert className="h-8 w-8 text-primary" />}
              title="Strict Causality"
              description="No future leakage. All features are constructed using only data available at the exact prediction time."
            />
          </div>
        </div>
      </section>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="p-6 rounded-2xl border bg-card text-card-foreground shadow-sm flex flex-col gap-4"
    >
      <div className="p-3 bg-primary/10 rounded-xl w-fit">
        {icon}
      </div>
      <h3 className="text-xl font-semibold">{title}</h3>
      <p className="text-muted-foreground leading-relaxed">
        {description}
      </p>
    </motion.div>
  );
}
