import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Local imports from the project
from data_processing import load_subject_data, create_features

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
subject_id = "559"
data_dir = os.path.join(os.getcwd(), "data", "OhioT1DM_2018")
# Use pathlib for robust directory handling
from pathlib import Path
output_dir = Path(os.getcwd()) / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Load and process data
# -----------------------------------------------------------------------------
df = load_subject_data(subject_id, data_dir=data_dir)
features_df = create_features(df)

# -----------------------------------------------------------------------------
# Helper to save figures consistently
# -----------------------------------------------------------------------------
def save_fig(fig, filename, dpi=300):
    path = output_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# -----------------------------------------------------------------------------
# 1. Dataset summary
# -----------------------------------------------------------------------------
raw_records = len(df)
feature_rows = len(features_df)
num_features = features_df.shape[1] - 2  # exclude ts and target
missing_total = features_df.isna().sum().sum()

fig1, ax1 = plt.subplots(figsize=(6, 2))
ax1.axis('off')
summary_table = [["Raw records", raw_records],
                 ["Feature rows", feature_rows],
                 ["Number of features (excluding ts/target)", num_features],
                 ["Total missing values", missing_total]]
ax1.table(cellText=summary_table, colLabels=None, loc='center')
ax1.set_title('Dataset Summary')
save_fig(fig1, "dataset_summary.png")

# -----------------------------------------------------------------------------
# 2. Glucose distribution
# -----------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.histplot(features_df["glucose"].dropna(), bins=30, kde=True, ax=ax2, color='steelblue')
ax2.set_xlabel('Glucose (mg/dL)')
ax2.set_ylabel('Count')
ax2.set_title('Glucose Distribution')
save_fig(fig2, "glucose_distribution.png")

# -----------------------------------------------------------------------------
# 3. Glucose timeseries
# -----------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 4))
# Ensure chronological order
features_df_sorted = features_df.sort_values("ts")
ax3.plot(features_df_sorted["ts"], features_df_sorted["glucose"], color='darkorange')
ax3.set_xlabel('Time')
ax3.set_ylabel('Glucose (mg/dL)')
ax3.set_title('Glucose Time‑Series')
ax3.tick_params(axis='x', rotation=45)
save_fig(fig3, "glucose_timeseries.png")

# -----------------------------------------------------------------------------
# 4. Correlation matrix (excluding ts and target)
# -----------------------------------------------------------------------------
corr_features = features_df.drop(columns=["ts", "target"]).select_dtypes(include=[np.number])
corr = corr_features.corr()
fig4, ax4 = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, fmt=".2f", ax=ax4)
ax4.set_title('Feature Correlation Matrix')
save_fig(fig4, "correlation_matrix.png")

# -----------------------------------------------------------------------------
# 5. Missing values per feature (before imputation)
# -----------------------------------------------------------------------------
missing_counts = features_df.isna().sum()
missing_counts = missing_counts[missing_counts > 0]
fig5, ax5 = plt.subplots(figsize=(10, 6))
missing_counts.sort_values(ascending=False).plot.bar(ax=ax5, color='tomato')
ax5.set_ylabel('Number of missing values')
ax5.set_title('Missing Values by Feature (pre‑imputation)')
save_fig(fig5, "missing_values.png")

# -----------------------------------------------------------------------------
# 6. Feature distributions (selected important features)
# -----------------------------------------------------------------------------
selected = ["glucose", "bolus", "carbs", "heart_rate", "iob", "glucose_diff1",
            "glucose_roll_mean_30", "glucose_roll_std_30"]
available = [col for col in selected if col in features_df.columns]
num_plots = len(available)
cols = 4
rows = (num_plots + cols - 1) // cols
fig6, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
axes = axes.flat
for i, col in enumerate(available):
    ax = axes[i]
    sns.histplot(features_df[col].dropna(), bins=30, kde=True, ax=ax, color='seagreen')
    ax.set_title(col)
    ax.set_xlabel('')
    ax.set_ylabel('')
# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')
fig6.suptitle('Selected Feature Distributions', y=1.02)
save_fig(fig6, "feature_distributions.png")

# -----------------------------------------------------------------------------
# 7. PCA explained variance
# -----------------------------------------------------------------------------
# Prepare data: drop ts and target, then standardize
pca_features = features_df.drop(columns=["ts", "target"]).select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pca_features)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

fig7, ax7 = plt.subplots(figsize=(8, 5))
indices = np.arange(1, len(explained) + 1)
ax7.bar(indices, explained, alpha=0.7, label='Individual')
ax7.plot(indices, cum_explained, color='red', marker='o', label='Cumulative')
ax7.set_xlabel('Principal Component')
ax7.set_ylabel('Explained Variance Ratio')
ax7.set_title('PCA Explained Variance')
ax7.legend()
save_fig(fig7, "pca_explained_variance.png")

# -----------------------------------------------------------------------------
# 8. PCA 2‑D scatter plot (first two components)
# -----------------------------------------------------------------------------
fig8, ax8 = plt.subplots(figsize=(8, 6))
scatter = ax8.scatter(X_pca[:, 0], X_pca[:, 1], c=features_df["target"], cmap='viridis', s=30, alpha=0.8)
ax8.set_xlabel('PC 1')
ax8.set_ylabel('PC 2')
ax8.set_title('PCA Projection (First 2 Components)')
cb = fig8.colorbar(scatter, ax=ax8)
cb.set_label('Target')
save_fig(fig8, "pca_2d.png")

print('EDA and PCA visualizations saved to', output_dir)
