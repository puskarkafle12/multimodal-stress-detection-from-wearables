import pandas as pd
import json
import numpy as np
from datetime import datetime

# Set the max length per modality
len_calories = 1895
len_sleep = 403
len_oxygen_saturation = 2411

len_heart_rate = 3000
len_stress = 3000
len_respiratory_rate = 3000
len_activity = 3000


def interpolate_downsample_pad(times, values, target_len, freq='5min'):
    if not times:
        return np.zeros(target_len), np.zeros(target_len)

    start_time = min(times)
    full_times = pd.date_range(start=start_time, periods=len(times), freq=freq)
    series = pd.Series(np.nan, index=full_times)

    for ts, val in zip(times, values):
        if ts in series.index:
            series[ts] = val
        else:
            closest_idx = np.argmin(np.abs(series.index - ts))
            series.iloc[closest_idx] = val

    mask = (~series.isna()).astype(np.float32)
    series = series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    if len(series) > target_len:
        factor = len(series) // target_len
        series = series[:factor * target_len].values.reshape(-1, factor).mean(axis=1)
        mask = mask[:factor * target_len].values.reshape(-1, factor).mean(axis=1)
    else:
        series = series.values
        mask = mask.values
    if len(series) < target_len:
        pad_len = target_len - len(series)
        series = np.pad(series, (0, pad_len), constant_values=0.0)
        mask = np.pad(mask, (0, pad_len), constant_values=0.0)

    return series, mask

def load_oxygen(json_path, target_len=len_oxygen_saturation):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['breathing']:
            ts = entry['effective_time_frame']['date_time']
            val = entry['oxygen_saturation']['value']
            times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
            values.append(float(val))
        return interpolate_downsample_pad(times, values, target_len)
    except:
        return np.zeros(target_len), np.zeros(target_len)

# ====== Calories ======
def load_calories(json_path, target_len=len_calories):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['activity']:
            if entry.get('activity_name') == 'kcal_burned':
                ts = entry['effective_time_frame']['time_interval']['start_date_time']
                val = entry['calories_value']['value']
                times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                values.append(float(val))
        return interpolate_downsample_pad(times, values, target_len)
    except:
        return np.zeros(target_len), np.zeros(target_len)

# ====== Sleep ======
def load_sleep(json_path, target_len=len_sleep):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['sleep']:
            ts = entry['sleep_stage_time_frame']['time_interval']['start_date_time']
            duration = (datetime.fromisoformat(entry['sleep_stage_time_frame']['time_interval']['end_date_time'].replace('Z', '+00:00'))
                        - datetime.fromisoformat(ts.replace('Z', '+00:00'))).total_seconds() / 60.0
            times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
            values.append(duration)
        return interpolate_downsample_pad(times, values, target_len)
    except:
        return np.zeros(target_len), np.zeros(target_len)

# ====== Heart Rate ======
def load_heartrate(json_path, target_len=len_heart_rate):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['heart_rate']:
            ts = entry['effective_time_frame']['date_time']
            val = entry['heart_rate']['value']
            times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
            values.append(float(val))
        return interpolate_downsample_pad(times, values, target_len)
    except:
        return np.zeros(target_len), np.zeros(target_len)

# ====== Stress ======
def load_stress(json_path, target_len=len_stress):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['stress']:
            ts = entry['effective_time_frame']['date_time']
            val = entry['stress']['value']
            times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
            values.append(float(val))
        return interpolate_downsample_pad(times, values, target_len)
    except:
        return np.zeros(target_len), np.zeros(target_len)

# ====== Respiratory Rate ======
def load_resp(json_path, target_len=len_respiratory_rate):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['breathing']:
            ts = entry['effective_time_frame']['date_time']
            val = entry['respiratory_rate']['value']
            times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
            values.append(float(val))
        return interpolate_downsample_pad(times, values, target_len)
    except:
        return np.zeros(target_len), np.zeros(target_len)

# ====== Activity (Steps) ======
def load_activity(json_path, target_len=len_activity):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['activity']:
            ts = entry['effective_time_frame']['time_interval']['start_date_time']
            val = entry['base_movement_quantity']['value']
            if val != '':
                times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                values.append(float(val))
        return interpolate_downsample_pad(times, values, target_len)
    except:
        return np.zeros(target_len), np.zeros(target_len)