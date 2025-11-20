import pandas as pd
import json
import numpy as np
from datetime import datetime

# Set the max length per modality - Updated for TA requirements (2880 = 10 days at 5-min intervals)
len_calories = 2880
len_sleep = 2880
len_oxygen_saturation = 2880

len_heart_rate = 2880  # 2880 values = 10 days at 5-min intervals
len_stress = 2880
len_respiratory_rate = 2880
len_activity = 2880


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

    start_time = min(times)
    # Create uniform time grid
    full_times = pd.date_range(start=start_time, periods=max(len(times), target_len), freq=freq)
    series = pd.Series(np.nan, index=full_times)

    # Map actual timestamps to grid
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

    # Downsample if needed
    if len(series) > target_len:
        factor = len(series) // target_len
        series_values = series[:factor * target_len].values.reshape(-1, factor).mean(axis=1)
        mask_values = mask[:factor * target_len].values.reshape(-1, factor).mean(axis=1)
        # Ensure mask is binary after averaging
        mask_values = (mask_values > 0.5).astype(np.float32)
    else:
        series_values = series.values
        mask_values = mask.values

    # Pad if needed
    if len(series_values) < target_len:
        pad_len = target_len - len(series_values)
        series_values = np.pad(series_values, (0, pad_len), constant_values=0.0)
        mask_values = np.pad(mask_values, (0, pad_len), constant_values=0.0)

    return series_values.astype(np.float32), mask_values.astype(np.float32)

def load_oxygen(json_path, target_len=len_oxygen_saturation):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['breathing']:
            # Check if oxygen_saturation exists in this entry
            if 'oxygen_saturation' not in entry:
                continue
                
            ts = entry['effective_time_frame']['date_time']
            val = entry['oxygen_saturation']['value']
            # Filter invalid values (should be 70-100 for SpO2)
            if val is not None:
                try:
                    val_float = float(val)
                    if 70 <= val_float <= 100:
                        times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                        values.append(val_float)
                except (ValueError, TypeError):
                    continue
        return interpolate_downsample_pad(times, values, target_len)
    except Exception as e:
        return np.zeros(target_len, dtype=np.float32), np.zeros(target_len, dtype=np.float32)

# ====== Calories ======
def load_calories(json_path, target_len=len_calories):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['activity']:
            if entry.get('activity_name') == 'kcal_burned':
                # Handle both date_time and time_interval structures
                time_frame = entry.get('effective_time_frame', {})
                if 'date_time' in time_frame:
                    ts = time_frame['date_time']
                elif 'time_interval' in time_frame:
                    ts = time_frame['time_interval'].get('start_date_time')
                else:
                    continue
                
                val = entry.get('calories_value', {}).get('value')
                # Filter invalid values: non-negative calories
                if val is not None and val != '':
                    try:
                        val_float = float(val)
                        if val_float >= 0:
                            times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                            values.append(val_float)
                    except (ValueError, TypeError):
                        continue
        return interpolate_downsample_pad(times, values, target_len)
    except Exception as e:
        return np.zeros(target_len, dtype=np.float32), np.zeros(target_len, dtype=np.float32)

# ====== Sleep ======
def load_sleep(json_path, target_len=len_sleep):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        # Map sleep stages to numeric values for better model learning
        # 0=awake, 1=light, 2=deep, 3=rem
        stage_map = {'awake': 0.0, 'light': 1.0, 'deep': 2.0, 'rem': 3.0}
        
        for entry in data['body']['sleep']:
            time_frame = entry.get('sleep_stage_time_frame', {})
            if 'time_interval' not in time_frame:
                continue
                
            ts = time_frame['time_interval']['start_date_time']
            end_ts = time_frame['time_interval']['end_date_time']
            
            # Calculate duration in minutes
            try:
                start_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_ts.replace('Z', '+00:00'))
                duration = (end_dt - start_dt).total_seconds() / 60.0
                
                # Filter invalid durations: positive and reasonable (0-600 minutes = 0-10 hours)
                if duration > 0 and duration <= 600:
                    # Use sleep stage as value (encoded numerically)
                    stage = entry.get('sleep_stage_state', 'awake')
                    stage_value = stage_map.get(stage, 0.0)
                    
                    # Use midpoint of time interval for timestamp
                    midpoint_dt = start_dt + (end_dt - start_dt) / 2
                    times.append(midpoint_dt)
                    values.append(stage_value)  # Use stage value instead of duration
            except (ValueError, TypeError, KeyError):
                continue
                
        return interpolate_downsample_pad(times, values, target_len)
    except Exception as e:
        return np.zeros(target_len, dtype=np.float32), np.zeros(target_len, dtype=np.float32)

# ====== Heart Rate ======
def load_heartrate(json_path, target_len=len_heart_rate):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['heart_rate']:
            ts = entry['effective_time_frame']['date_time']
            val = entry['heart_rate']['value']
            # Filter invalid values: exclude 0 (missing data) and ensure 30-220 bpm range
            if val is not None and val != 0 and 30 <= float(val) <= 220:
                times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                values.append(float(val))
        return interpolate_downsample_pad(times, values, target_len)
    except Exception as e:
        return np.zeros(target_len, dtype=np.float32), np.zeros(target_len, dtype=np.float32)

# ====== Stress ======
def load_stress(json_path, target_len=len_stress):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['stress']:
            ts = entry['effective_time_frame']['date_time']
            val = entry['stress']['value']
            # Filter invalid values: negative values (-2, -1) indicate missing/invalid data
            # Valid stress values are 0-100
            if val is not None:
                try:
                    val_float = float(val)
                    if val_float >= 0:  # Only include non-negative values
                        times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                        values.append(val_float)
                except (ValueError, TypeError):
                    continue
        return interpolate_downsample_pad(times, values, target_len)
    except Exception as e:
        return np.zeros(target_len, dtype=np.float32), np.zeros(target_len, dtype=np.float32)

# ====== Respiratory Rate ======
def load_resp(json_path, target_len=len_respiratory_rate):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['breathing']:
            # Check if respiratory_rate exists in this entry
            if 'respiratory_rate' not in entry:
                continue
                
            ts = entry['effective_time_frame']['date_time']
            val = entry['respiratory_rate']['value']
            # Filter invalid values: exclude negative values (-1, -2 indicate missing data)
            # Valid range: 8-40 breaths/min
            if val is not None:
                try:
                    val_float = float(val)
                    if val_float >= 0 and 8 <= val_float <= 40:
                        times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                        values.append(val_float)
                except (ValueError, TypeError):
                    continue
        return interpolate_downsample_pad(times, values, target_len)
    except Exception as e:
        return np.zeros(target_len, dtype=np.float32), np.zeros(target_len, dtype=np.float32)

# ====== Activity (Steps) ======
def load_activity(json_path, target_len=len_activity):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        times, values = [], []
        for entry in data['body']['activity']:
            # Skip entries with empty activity_name (these are invalid)
            activity_name = entry.get('activity_name', '')
            if activity_name == '':
                continue
            
            # Get timestamp from time_interval
            time_frame = entry.get('effective_time_frame', {})
            if 'time_interval' not in time_frame:
                continue
            ts = time_frame['time_interval'].get('start_date_time')
            if not ts:
                continue
            
            # Get step value
            movement = entry.get('base_movement_quantity', {})
            val = movement.get('value', '')
            
            # Filter invalid values: non-empty and non-negative
            if val != '' and val is not None:
                try:
                    val_float = float(val)
                    if val_float >= 0:  # Steps should be non-negative
                        times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                        values.append(val_float)
                except (ValueError, TypeError):
                    continue
        return interpolate_downsample_pad(times, values, target_len)
    except Exception as e:
        return np.zeros(target_len, dtype=np.float32), np.zeros(target_len, dtype=np.float32)