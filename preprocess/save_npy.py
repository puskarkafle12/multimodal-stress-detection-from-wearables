# npy_saving.py
import os
import numpy as np
import pandas as pd
from data_loader import load_cgm, load_oxygen, load_calories, load_sleep, load_heartrate, load_stress, load_resp, load_activity

DATASET_ROOT = '../../dataset/wearable_activity_monitor'
CGM_ROOT = '../../dataset/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6'
CLINICAL_PATH = '../../dataset/clinical_data/condition_occurrence.csv'
OUTPUT_FOLDER = '../../npy/transformer-npy/' #transformer-npy

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

clinical_df = pd.read_csv(CLINICAL_PATH)
clinical_data = clinical_df.groupby('person_id')['condition_concept_id'].apply(list).to_dict()

def get_label(person_id, clinical_data):
    codes = clinical_data.get(person_id, [])
    if 201826 in codes:
        return 2
    elif 37018196 in codes:
        return 1
    else:
        return 0

# Preprocess and Save
for pid in clinical_data.keys():
    try:
        print(f"Processing participant {pid}...")

        paths = {
            'heartrate': f'{DATASET_ROOT}/heart_rate/garmin_vivosmart5/{pid}/{pid}_heartrate.json',
            'oxygen': f'{DATASET_ROOT}/oxygen_saturation/garmin_vivosmart5/{pid}/{pid}_oxygensaturation.json',
            'stress': f'{DATASET_ROOT}/stress/garmin_vivosmart5/{pid}/{pid}_stress.json',
            'resp': f'{DATASET_ROOT}/respiratory_rate/garmin_vivosmart5/{pid}/{pid}_respiratoryrate.json',
            'activity': f'{DATASET_ROOT}/physical_activity/garmin_vivosmart5/{pid}/{pid}_activity.json',
            'calorie': f'{DATASET_ROOT}/physical_activity_calorie/garmin_vivosmart5/{pid}/{pid}_calorie.json',
            'sleep': f'{DATASET_ROOT}/sleep/garmin_vivosmart5/{pid}/{pid}_sleep.json',
            'cgm': f'{CGM_ROOT}/{pid}/{pid}_DEX.json'
        }

        # Load each modality
        modalities = {
            'cgm': load_cgm(paths['cgm']),
            'oxygen': load_oxygen(paths['oxygen']),
            'calorie': load_calories(paths['calorie']),
            'sleep': load_sleep(paths['sleep']),
            'heartrate': load_heartrate(paths['heartrate']),
            'stress': load_stress(paths['stress']),
            'resp': load_resp(paths['resp']),
            'activity': load_activity(paths['activity']),
        }

        # Check if any modality is completely empty (all zeros)
        if any(np.all(modalities[key] == 0) for key in modalities.keys()):
            print(f"⚠️ Skipping participant {pid} due to empty modality.")
            continue

        # Get label
        y_label = get_label(pid, clinical_data)

        # Save npz file
        np.savez_compressed(
            os.path.join(OUTPUT_FOLDER, f"{pid}.npz"),
            **modalities,
            label=np.array([y_label], dtype=np.int64)
        )

    except Exception as e:
        print(f"⚠️ Skipped participant {pid} due to error: {e}")
        continue

print(f"\n✅ Finished preprocessing all participants into {OUTPUT_FOLDER}")