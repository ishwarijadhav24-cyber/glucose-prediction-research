import os
from data_processing import load_subject_data, create_features

DATA_DIR = os.path.join('data', 'OhioT1DM_2018')
subjects = ['559','563','570','575','588','591']
for sub in subjects:
    try:
        df = load_subject_data(sub, data_dir=DATA_DIR)
        print(f'Subject {sub}: records = {len(df)}')
        feats = create_features(df)
        print(f'Subject {sub}: feature rows = {len(feats)}')
    except Exception as e:
        print(f'Error processing subject {sub}: {e}')
