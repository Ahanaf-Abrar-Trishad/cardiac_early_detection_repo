#!/usr/bin/env python3
"""
Extract ACDC diagnosis labels from Info.cfg files.
"""
import pandas as pd
import pathlib

def main():
    # Read metadata
    df = pd.read_csv('meta/master_metadata.csv')
    acdc_patients = df[df['dataset'] == 'acdc']['patient_id'].unique()
    
    # Extract diagnosis from Info.cfg files
    labels = []
    for patient in acdc_patients:
        cfg_path = f'cardio_data/raw/acdc/{patient}/Info.cfg'
        if pathlib.Path(cfg_path).exists():
            # Read the config file manually since it doesn't have section headers
            group = 'Unknown'
            with open(cfg_path, 'r') as f:
                for line in f:
                    if line.startswith('Group:'):
                        group = line.split(':')[1].strip()
                        break
            labels.append({'patient_id': patient, 'diagnosis': group})
        else:
            print(f"Warning: {cfg_path} not found")
    
    # Create DataFrame and save
    lab = pd.DataFrame(labels).sort_values('patient_id')
    pathlib.Path('results').mkdir(parents=True, exist_ok=True)
    lab.to_csv('results/acdc_labels.csv', index=False)
    
    print('Saved results/acdc_labels.csv')
    print(f"Found {len(lab)} patients")
    print("Diagnosis distribution:")
    print(lab['diagnosis'].value_counts())

if __name__ == '__main__':
    main()