import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lxml import etree

def parse_xml_file(filepath: str) -> pd.DataFrame:
    """Parse a single OhioT1DM XML patient file into a DataFrame.

    Returns a DataFrame with columns:
        - ts (datetime)
        - glucose (float, mg/dL)
        - bolus (float, insulin units)
        - carbs (float, grams)
        - heart_rate (float, bpm) – optional, may contain NaNs
    """
    tree = etree.parse(filepath)
    root = tree.getroot()

    def extract_events(tag, value_attr, time_attr):
        events = []
        for elem in root.findall(f'.//{tag}/event'):
            val = elem.get(value_attr)
            ts = elem.get(time_attr)
            if val and ts:
                try:
                    value = float(val)
                except ValueError:
                    continue
                try:
                    timestamp = pd.to_datetime(ts, dayfirst=True)
                except Exception:
                    continue
                events.append((timestamp, value))
        return events

    glucose_events = extract_events('glucose_level', 'value', 'ts')
    bolus_events = extract_events('bolus', 'dose', 'ts_begin')
    meal_events = extract_events('meal', 'carbs', 'ts')
    hr_events = []
    for elem in root.findall('.//heart_rate'):
        val = elem.get('value')
        ts = elem.get('ts')
        if val and ts:
            try:
                hr = float(val)
                timestamp = pd.to_datetime(ts, dayfirst=True)
                hr_events.append((timestamp, hr))
            except Exception:
                continue

    # Create 5‑min uniform time grid
    all_ts = [t for t, _ in glucose_events]
    if not all_ts:
        raise ValueError(f'No glucose data in {filepath}')
    start = min(all_ts)
    end = max(all_ts)
    grid = pd.date_range(start=start.floor('5min'), end=end.ceil('5min'), freq='5min')
    df = pd.DataFrame(index=grid)
    df.index.name = 'ts'

    # Convert raw glucose events to a sorted Series with unique timestamps
    g_times = [t for t, _ in glucose_events]
    g_vals = [g for _, g in glucose_events]
    glucose_series = pd.Series(data=g_vals, index=pd.DatetimeIndex(g_times)).sort_index()
    glucose_series = glucose_series[~glucose_series.index.duplicated(keep='last')]

    # Causal glucose alignment:
    # At grid timestamp t, glucose is aligned using the most recent raw observation at or before t
    # (tau_raw <= t) within a 5-minute backward tolerance window (i.e. in (t - 5min, t]).
    # Under this causal rule, no observation recorded after t can ever populate grid row t.
    df['glucose'] = glucose_series.reindex(df.index, method='ffill', tolerance=pd.Timedelta('5min'))

    # Causal missing-value handling:
    # Forward-fill short missing segments up to 6 steps (30 min) without look-ahead.
    # Bidirectional interpolation (limit_direction='both') is removed so earlier timestamps
    # never depend on observations occurring after them.
    df['glucose'] = df['glucose'].ffill(limit=6)

    df['bolus'] = 0.0
    for t, d in bolus_events:
        bucket = t.floor('5min')
        if bucket in df.index:
            df.at[bucket, 'bolus'] += d
    df['carbs'] = 0.0
    for t, c in meal_events:
        bucket = t.floor('5min')
        if bucket in df.index:
            df.at[bucket, 'carbs'] += c

    df['heart_rate'] = np.nan
    if hr_events:
        hr_series = pd.Series({t: h for t, h in hr_events})
        df['heart_rate'] = hr_series.resample('5T').mean().reindex(df.index)

    df = df.astype(float)
    return df

def load_subject_data(subject_id: str, data_dir: str = "data/OhioT1DM") -> pd.DataFrame:
    """Load all XML files for a subject and concatenate them.

    The OhioT1DM dataset stores training and testing files separately. This function loads every file that starts with the subject id.
    """
    if not os.path.exists(data_dir):
        for alt in ["data/OhioT1DM", "data/OhioT1DM_2018", "data/2018"]:
            if os.path.exists(alt):
                data_dir = alt
                break
    pattern = os.path.join(data_dir, f"{subject_id}*ws-*.xml")
    file_list = glob.glob(pattern)
    if not file_list:
        raise FileNotFoundError(f"No XML files found for subject {subject_id} in {data_dir}")
    dfs = [parse_xml_file(fp) for fp in file_list]
    full = pd.concat(dfs).sort_index()
    full = full[~full.index.duplicated(keep='first')]
    return full

def create_features(df: pd.DataFrame, iob_params: dict = None) -> pd.DataFrame:
    """Generate modeling features.

    Parameters
    ----------
    df : DataFrame with columns ['glucose', 'bolus', 'carbs', 'heart_rate']
    iob_params : dict with keys 'dia' (minutes) and 't_peak' (minutes). Defaults to DIA=240, t_peak=60.
    """
    if iob_params is None:
        iob_params = {'dia': 240, 't_peak': 60}
    df = df.copy()
    dt = 5
    max_steps = int(iob_params['dia'] / dt)
    times = np.arange(max_steps)
    tau = iob_params['t_peak']
    kernel = np.exp(-times * dt / tau)
    kernel = kernel / kernel.sum()
    iob = np.convolve(df['bolus'].values, kernel, mode='full')[:len(df)]
    df['iob'] = iob

    df['glucose_diff1'] = df['glucose'].diff()
    df['glucose_diff2'] = df['glucose_diff1'].diff()
    df['glucose_roll_mean_30'] = df['glucose'].rolling(window=6, min_periods=1).mean()
    df['glucose_roll_std_30'] = df['glucose'].rolling(window=6, min_periods=1).std()
    df['glucose_roll_std_30'] = df['glucose_roll_std_30'].fillna(0)

    hour = df.index.hour + df.index.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    lag_steps = 12
    for col in ['glucose', 'glucose_diff1', 'iob']:
        for lag in range(1, lag_steps + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    df['target'] = df['glucose'].shift(-6)
    # Causal heart-rate handling:
    # Forward-fill historical heart-rate observations without look-ahead.
    # Remove .bfill() so measurements at time t never depend on any future heart-rate readings.
    # Initial unobserved heart-rate values remain NaN and are imputed during training in evaluate_loso.py.
    if 'heart_rate' in df.columns:
        df['heart_rate'] = df['heart_rate'].ffill()
    # Drop rows where target is NaN (usually the last few rows) but keep others
    df = df.dropna(subset=['target']).reset_index()
    return df

if __name__ == "__main__":
    subject = "559"
    data = load_subject_data(subject)
    features = create_features(data)
    print(features.head())
