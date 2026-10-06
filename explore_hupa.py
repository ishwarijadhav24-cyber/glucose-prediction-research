"""
Phase 1: Explore HUPA-UCM dataset for Objective 3
Generates an inventory of patients, detecting if glucose is native 5-minute or interpolated,
and produces summary statistics and plots.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

THRESHOLD_NON_INTEGER = 0.10
THRESHOLD_ZERO_DIFF = 0.20

def load_preprocessed(path):
    # The separator is a SEMICOLON ";"
    try:
        df = pd.read_csv(path, sep=';')
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return pd.DataFrame()

def interpolation_signals(glucose):
    # frac_non_integer_glucose: fraction of non-missing glucose values that are not whole numbers (tolerance 1e-6)
    valid_glucose = glucose.dropna()
    if len(valid_glucose) == 0:
        return np.nan, np.nan
    
    diff_from_round = np.abs(valid_glucose - np.round(valid_glucose))
    frac_non_integer_glucose = np.mean(diff_from_round > 1e-6)
    
    # frac_zero_second_diff: fraction of non-missing second differences of glucose (glucose.diff().diff()) that are exactly zero
    second_diff = valid_glucose.diff().diff().dropna()
    if len(second_diff) == 0:
        frac_zero_second_diff = np.nan
    else:
        frac_zero_second_diff = np.mean(np.abs(second_diff) <= 1e-6)
        
    return frac_non_integer_glucose, frac_zero_second_diff

def raw_devices(patient_id):
    raw_dir = os.path.join('data', 'HUPA-UCM', 'Raw_Data', patient_id)
    raw_libre = False
    raw_medtronic = False
    
    libre_dir = os.path.join(raw_dir, 'free_style_sensor')
    if os.path.exists(libre_dir):
        if len(glob.glob(os.path.join(libre_dir, '*.csv'))) > 0:
            raw_libre = True
            
    medtronic_dir = os.path.join(raw_dir, 'medtronic_insulin_pump')
    if os.path.exists(medtronic_dir):
        if len(glob.glob(os.path.join(medtronic_dir, '*.csv'))) > 0:
            raw_medtronic = True
            
    return raw_libre, raw_medtronic

def summarise_patient(path):
    patient_id = os.path.basename(path).replace('.csv', '')
    df = load_preprocessed(path)
    
    if df.empty or 'glucose' not in df.columns or 'time' not in df.columns:
        return {'patient': patient_id}
        
    start = df['time'].min()
    end = df['time'].max()
    days = (end - start).total_seconds() / 86400.0 if not pd.isna(start) else 0.0
    
    rows = len(df)
    
    time_diffs = df['time'].diff().dt.total_seconds() / 60.0
    median_step_min = time_diffs.median()
    time_gaps_over_30min = (time_diffs > 30).sum()
    
    expected_points = int(days * 24 * 60 / 5) if days > 0 else 0
    non_missing_glucose = df['glucose'].notna().sum()
    glucose_coverage_pct = (non_missing_glucose / expected_points * 100) if expected_points > 0 else 0.0
    
    glucose_min = df['glucose'].min()
    glucose_max = df['glucose'].max()
    glucose_mean = df['glucose'].mean()
    
    valid_g = df['glucose'].dropna()
    pct_below_70 = (valid_g < 70).mean() * 100 if len(valid_g) > 0 else 0
    pct_70_180 = ((valid_g >= 70) & (valid_g <= 180)).mean() * 100 if len(valid_g) > 0 else 0
    pct_above_180 = (valid_g > 180).mean() * 100 if len(valid_g) > 0 else 0
    
    n_bolus_events = (df['bolus_volume_delivered'] > 0).sum() if 'bolus_volume_delivered' in df.columns else 0
    bolus_units_per_day = df['bolus_volume_delivered'].sum() / days if days > 0 and 'bolus_volume_delivered' in df.columns else 0
    
    n_meal_events = (df['carb_input'] > 0).sum() if 'carb_input' in df.columns else 0
    carbs_g_per_day = (df['carb_input'].sum() * 10) / days if days > 0 and 'carb_input' in df.columns else 0
    
    hr_available_pct = (df['heart_rate'].notna().mean() * 100) if 'heart_rate' in df.columns else 0.0
    
    raw_libre, raw_medtronic = raw_devices(patient_id)
    
    frac_non_integer_glucose, frac_zero_second_diff = interpolation_signals(df['glucose'])
    
    if pd.isna(frac_non_integer_glucose) or pd.isna(frac_zero_second_diff):
        glucose_type = "UNKNOWN"
    elif frac_non_integer_glucose > THRESHOLD_NON_INTEGER or frac_zero_second_diff > THRESHOLD_ZERO_DIFF:
        glucose_type = "INTERPOLATED?"
    else:
        glucose_type = "native 5-min"
        
    include_candidate = (days >= 7) and (glucose_coverage_pct >= 70) and (n_bolus_events > 0) and (n_meal_events > 0)
    
    return {
        'patient': patient_id,
        'start': start,
        'end': end,
        'days': round(days, 1),
        'rows': rows,
        'median_step_min': median_step_min,
        'time_gaps_over_30min': time_gaps_over_30min,
        'glucose_coverage_pct': glucose_coverage_pct,
        'glucose_min': glucose_min,
        'glucose_max': glucose_max,
        'glucose_mean': glucose_mean,
        'pct_below_70': pct_below_70,
        'pct_70_180': pct_70_180,
        'pct_above_180': pct_above_180,
        'n_bolus_events': n_bolus_events,
        'bolus_units_per_day': bolus_units_per_day,
        'n_meal_events': n_meal_events,
        'carbs_g_per_day': carbs_g_per_day,
        'hr_available_pct': hr_available_pct,
        'raw_libre': raw_libre,
        'raw_medtronic': raw_medtronic,
        'frac_non_integer_glucose': frac_non_integer_glucose,
        'frac_zero_second_diff': frac_zero_second_diff,
        'glucose_type': glucose_type,
        'include_candidate': include_candidate
    }

def make_plots(inventory, data_dict):
    out_dir = 'outputs_objective3'
    os.makedirs(out_dir, exist_ok=True)
    
    # hupa_days_and_coverage.png
    plt.figure(figsize=(12, 6))
    bars = plt.bar(inventory['patient'], inventory['days'])
    plt.xticks(rotation=90)
    plt.ylabel('Days')
    plt.title('Days per Patient with Glucose Coverage %')
    for bar, cov in zip(bars, inventory['glucose_coverage_pct']):
        yval = bar.get_height()
        if not pd.isna(cov):
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{cov:.0f}%', ha='center', va='bottom', rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hupa_days_and_coverage.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # hupa_glucose_type.png
    plt.figure(figsize=(10, 8))
    for _, row in inventory.iterrows():
        plt.scatter(row['frac_non_integer_glucose'], row['frac_zero_second_diff'])
        plt.text(row['frac_non_integer_glucose'], row['frac_zero_second_diff'], row['patient'], fontsize=8)
    plt.axvline(THRESHOLD_NON_INTEGER, color='r', linestyle='--', label=f'frac_non_int > {THRESHOLD_NON_INTEGER}')
    plt.axhline(THRESHOLD_ZERO_DIFF, color='b', linestyle='--', label=f'frac_zero_diff > {THRESHOLD_ZERO_DIFF}')
    plt.xlabel('frac_non_integer_glucose')
    plt.ylabel('frac_zero_second_diff')
    plt.title('Glucose Type Detection')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hupa_glucose_type.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # hupa_glucose_distribution.png
    all_glucose = []
    for pat, df in data_dict.items():
        if 'glucose' in df.columns:
            all_glucose.extend(df['glucose'].dropna().values)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_glucose, bins=100, color='skyblue', edgecolor='black')
    plt.axvline(70, color='red', linestyle='dashed', linewidth=2, label='70 mg/dL')
    plt.axvline(180, color='red', linestyle='dashed', linewidth=2, label='180 mg/dL')
    plt.title('Pooled Glucose Distribution (All Patients)')
    plt.xlabel('Glucose (mg/dL)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hupa_glucose_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # hupa_example_traces.png
    native_pat = inventory[inventory['glucose_type'] == 'native 5-min']['patient'].first_valid_index()
    interp_pat = inventory[inventory['glucose_type'] == 'INTERPOLATED?']['patient'].first_valid_index()
    
    if native_pat is not None and interp_pat is not None:
        p1 = inventory.loc[native_pat, 'patient']
        p2 = inventory.loc[interp_pat, 'patient']
        
        df1 = data_dict[p1]
        df2 = data_dict[p2]
        
        # Take 24 hours (24 * 12 = 288 points) from somewhere with data
        # To ensure we have valid points, we can find a window of non-null glucose values
        df1_sub = df1.dropna(subset=['glucose']).head(288)
        df2_sub = df2.dropna(subset=['glucose']).head(288)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        
        if not df1_sub.empty:
            axes[0].plot(df1_sub['time'], df1_sub['glucose'], marker='o', linestyle='-')
            axes[0].set_title(f'{p1} (Native 5-min)')
            axes[0].set_ylabel('Glucose')
            
        if not df2_sub.empty:
            axes[1].plot(df2_sub['time'], df2_sub['glucose'], marker='o', linestyle='-')
            axes[1].set_title(f'{p2} (INTERPOLATED?)')
            axes[1].set_ylabel('Glucose')
            
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'hupa_example_traces.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # If we don't have both, just create an empty figure to satisfy requirement
        plt.figure()
        plt.text(0.5, 0.5, "Could not find both a native and interpolated patient to plot", ha="center")
        plt.savefig(os.path.join(out_dir, 'hupa_example_traces.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    prep_dir = os.path.join('data', 'HUPA-UCM', 'Preprocessed')
    files = glob.glob(os.path.join(prep_dir, '*.csv'))
    
    if not files:
        raise ValueError(f"No CSV files found in {prep_dir}")
        
    os.makedirs('results_objective3', exist_ok=True)
    
    records = []
    data_dict = {}
    for f in sorted(files):
        df = load_preprocessed(f)
        patient_id = os.path.basename(f).replace('.csv', '')
        data_dict[patient_id] = df
        summary = summarise_patient(f)
        records.append(summary)
        
    inventory = pd.DataFrame(records)
    inventory.to_csv(os.path.join('results_objective3', 'hupa_inventory.csv'), index=False)
    
    # Console output 1: Table
    cols_to_print = ['patient', 'days', 'glucose_coverage_pct', 'glucose_min', 'glucose_max',
                     'n_bolus_events', 'n_meal_events', 'raw_libre', 'raw_medtronic',
                     'frac_non_integer_glucose', 'frac_zero_second_diff', 'glucose_type', 'include_candidate']
    print(inventory[cols_to_print].to_string(index=False))
    
    # Console output 2: Summary block
    total_patients = len(inventory)
    native_count = (inventory['glucose_type'] == 'native 5-min').sum()
    interp_count = (inventory['glucose_type'] == 'INTERPOLATED?').sum()
    raw_med_count = inventory['raw_medtronic'].sum()
    raw_libre_count = inventory['raw_libre'].sum()
    include_count = inventory['include_candidate'].sum()
    max_glucose = inventory['glucose_max'].max()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total patients: {total_patients}")
    print(f"Number 'native 5-min': {native_count}")
    print(f"Number 'INTERPOLATED?': {interp_count}")
    print(f"Number with raw Medtronic data: {raw_med_count}")
    print(f"Number with raw Libre data: {raw_libre_count}")
    print(f"Number of include candidates: {include_count}")
    print(f"Maximum glucose across the cohort: {max_glucose} mg/dL (Note: OhioT1DM's range is 40-400 mg/dL)")
    
    # Console output 3: Cross-tab
    print("\nCross-tab of glucose_type vs raw_medtronic:")
    print(pd.crosstab(inventory['glucose_type'], inventory['raw_medtronic']))
    print("\n")
    
    # Plots
    make_plots(inventory, data_dict)

if __name__ == "__main__":
    main()
