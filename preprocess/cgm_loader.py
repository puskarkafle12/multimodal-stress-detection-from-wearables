import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

cgm_high_value = 180.0
cgm_low_value = 70.0
len_cgm = 2856  # set based on your requirement
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
    if not times:
        return np.zeros(target_len), np.zeros(target_len)

    # 1. Interpolation on uniform timeline
    start_time = min(times)
    full_times = pd.date_range(start=start_time, periods=len(times), freq=freq)
    series = pd.Series(np.nan, index=full_times)

    for ts, val in zip(times, values):
        if ts in series.index:
            series[ts] = val
        else:
            closest_idx = np.argmin(np.abs(series.index - ts))
            series.iloc[closest_idx] = val

    series = series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    mask = (~series.isna()).astype(np.float32)

    # 2. Downsample if needed
    if len(series) > target_len:
        factor = len(series) // target_len
        series = series[:factor * target_len].values.reshape(-1, factor).mean(axis=1)
        mask = mask[:factor * target_len].values.reshape(-1, factor).mean(axis=1)
    else:
        series = series.values
        mask = mask.values

    # 3. Pad if needed
    if len(series) < target_len:
        pad_len = target_len - len(series)
        series = np.pad(series, (0, pad_len), constant_values=0.0)
        mask = np.pad(mask, (0, pad_len), constant_values=0.0)

    return series, mask

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
                val = float(val)
            times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
            values.append(val)
        return interpolate_downsample_pad(times, values, target_len)
    except:
        return np.zeros(target_len), np.zeros(target_len)