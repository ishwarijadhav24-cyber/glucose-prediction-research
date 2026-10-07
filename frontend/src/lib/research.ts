// Research results for the website. Generated from the project's result files by
// frontend/scripts/build_research_data.py — never edit research.json by hand.
import data from "@/data/research.json";

export const research = data;

export type ModelName = "Persistence" | "Ridge" | "RandomForest" | "MLP" | "LightGBM" | "Stacking";

export const MODELS: ModelName[] = ["Persistence", "Ridge", "RandomForest", "MLP", "LightGBM", "Stacking"];

export const MODEL_INFO: Record<ModelName, { label: string; color: string; idea: string; detail: string }> = {
  Persistence: {
    label: "Persistence",
    color: "#94a3b8",
    idea: "Glucose in 30 min = glucose now",
    detail: "The baseline. It needs no training; any useful model must beat it.",
  },
  Ridge: {
    label: "Ridge",
    color: "#14b8a6",
    idea: "Weighted sum of the 47 features, weights kept small",
    detail: "Linear regression with an L2 penalty (alpha = 1.0). Simple, stable and interpretable.",
  },
  RandomForest: {
    label: "Random Forest",
    color: "#22c55e",
    idea: "Average of 50 decision trees",
    detail: "50 trees, max depth 15, min 5 samples per leaf, each tree sees half of the data.",
  },
  MLP: {
    label: "MLP",
    color: "#a855f7",
    idea: "Small neural network (128 → 64 neurons)",
    detail: "Feed-forward network, 150 iterations with early stopping on an internal training split.",
  },
  LightGBM: {
    label: "LightGBM",
    color: "#3b82f6",
    idea: "150 boosted trees, each fixing the previous errors",
    detail: "Gradient boosting: learning rate 0.05, 31 leaves, min 20 samples per leaf. The deployed model.",
  },
  Stacking: {
    label: "Stacking",
    color: "#f59e0b",
    idea: "Ridge + Random Forest + LightGBM combined by a Ridge",
    detail: "Base models' predictions (3-fold) are combined by a final Ridge regression.",
  },
};

export const fmt = (v: number | null | undefined, digits = 2) =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const fmtP = (p: number) => (p < 0.001 ? "< 0.001" : p.toFixed(p < 0.01 ? 4 : 3));
