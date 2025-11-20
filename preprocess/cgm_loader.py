import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

cgm_high_value = 180.0
cgm_low_value = 70.0
len_cgm = 2880  # 2880 values = 10 days at 5-min intervals (as per TA requirements)
"""
def pad_and_mask(values, seq_len):
    values = np.array(values, dtype=np.float32)
    mask = ~np.isnan(values)
    values = np.nan_to_num(values, nan=0.0)
    length = len(values)
    if length >= seq_len:
        return values[:seq_len], mask[:seq_len].astype(np.float32)
    else:
        pad_len = seq_len - length
        padded_values = np.pad(values, (0, pad_len), constant_values=0.0)
        padded_mask = np.pad(mask.astype(np.float32), (0, pad_len), constant_values=0.0)
        return padded_values, padded_mask

def load_cgm(json_path):
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        glucose_values = []
        for entry in data['body']['cgm']:
            val = entry['blood_glucose']['value']
            if val == 'High':
                glucose_values.append(cgm_high_value)
            elif val == 'Low':
                glucose_values.append(cgm_low_value)
            else:
                glucose_values.append(float(val))
        return pad_and_mask(glucose_values, len_cgm)
    except:
        return [[], []]

def downsample_by_averaging(values, target_len):
   
    #Downsample a sequence by average pooling into target_len bins.
    
    values = np.array(values, dtype=np.float32)
    bins = np.array_split(values, target_len)
    downsampled = np.array([np.mean(bin) for bin in bins], dtype=np.float32)
    return downsampled
"""

def interpolate_downsample_pad(times, values, target_len, freq='5min'):
    """Interpolate, downsample, and pad time series data"""
    if not times or not values:
        return np.zeros(target_len, dtype=np.float32), np.zeros(target_len, dtype=np.float32)

    # Ensure values are numpy array and handle any remaining invalid values
    values = np.array(values, dtype=np.float32)
    times = np.array(times)
    
    # Remove any NaN or inf values
    valid_mask = np.isfinite(values)
    if np.sum(valid_mask) == 0:
        return np.zeros(target_len, dtype=np.float32), np.zeros(target_len, dtype=np.float32)
    
    times = times[valid_mask]
    values = values[valid_mask]

    # 1. Interpolation on uniform timeline
    start_time = min(times)
    full_times = pd.date_range(start=start_time, periods=max(len(times), target_len), freq=freq)
    series = pd.Series(np.nan, index=full_times)

    for ts, val in zip(times, values):
        if ts in series.index:
            series[ts] = val
        else:
            closest_idx = np.argmin(np.abs(series.index - ts))
            series.iloc[closest_idx] = val

    # Create mask before interpolation
    mask = (~series.isna()).astype(np.float32)
    
    # Interpolate missing values
    series = series.interpolate(method='linear', limit_direction='both')
    # Fill remaining NaNs at boundaries
    series = series.fillna(method='bfill').fillna(method='ffill')
    
    # If still NaN, fill with 0
    series = series.fillna(0.0)

    # 2. Downsample if needed
    if len(series) > target_len:
        factor = len(series) // target_len
        series_values = series[:factor * target_len].values.reshape(-1, factor).mean(axis=1)
        mask_values = mask[:factor * target_len].values.reshape(-1, factor).mean(axis=1)
        # Ensure mask is binary after averaging
        mask_values = (mask_values > 0.5).astype(np.float32)
    else:
        series_values = series.values
        mask_values = mask.values

    # 3. Pad if needed
    if len(series_values) < target_len:
        pad_len = target_len - len(series_values)
        series_values = np.pad(series_values, (0, pad_len), constant_values=0.0)
        mask_values = np.pad(mask_values, (0, pad_len), constant_values=0.0)

    return series_values.astype(np.float32), mask_values.astype(np.float32)

# ====== CGM ======
def load_cgm(json_path, target_len=len_cgm):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['cgm']:
            ts = entry['effective_time_frame']['time_interval']['start_date_time']
            val = entry['blood_glucose']['value']
            if val == 'High':
                val = cgm_high_value
            elif val == 'Low':
                val = cgm_low_value
            else:
                try:
                    val = float(val)
                    # Filter invalid values (should be 40-400 mg/dL)
                    if val < 40 or val > 400:
                        continue
                except (ValueError, TypeError):
                    continue
            times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
            values.append(val)
        return interpolate_downsample_pad(times, values, target_len)
    except Exception as e:
        return np.zeros(target_len, dtype=np.float32), np.zeros(target_len, dtype=np.float32)