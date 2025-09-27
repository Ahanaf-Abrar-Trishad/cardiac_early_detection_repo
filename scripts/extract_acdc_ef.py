#!/usr/bin/env python3
"""
Extract ACDC Ejection Fraction (EF) from segmentation masks.

Calculates EF using the formula: EF = (EDV - ESV) / EDV * 100
where EDV/ESV are calculated from LV segmentation masks at ED/ES frames.

Usage:
    python scripts/extract_acdc_ef.py [--meta path/to/metadata.csv] [--data_root path/to/acdc]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional


def parse_acdc_info(info_path: str) -> dict:
    """Parse ACDC Info.cfg file to get ED/ES frame numbers and patient info."""
    # ACDC Info.cfg format is simple key:value pairs without sections
    info = {}
    with open(info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, value = line.split(':', 1)
                key, value = key.strip(), value.strip()
                info[key] = value
    
    return {
        'ed_frame': int(info.get('ED', 1)),
        'es_frame': int(info.get('ES', 1)),
        'group': info.get('Group', 'Unknown'),
        'height': float(info.get('Height', 0)) if info.get('Height') else None,
        'weight': float(info.get('Weight', 0)) if info.get('Weight') else None,
        'nb_frames': int(info.get('NbFrame', 1))
    }


def calculate_lv_volume(mask_path: str, voxel_spacing: Optional[Tuple[float, float, float]] = None) -> float:
    """
    Calculate LV volume from segmentation mask.
    
    Args:
        mask_path: Path to segmentation mask (.nii.gz)
        voxel_spacing: (x, y, z) spacing in mm. If None, extracted from NIfTI header.
    
    Returns:
        LV volume in mL
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    # Load segmentation mask
    img = nib.load(mask_path)
    data = img.get_fdata()
    
    # Get voxel spacing from header if not provided
    if voxel_spacing is None:
        # NIfTI pixdim: [0, x, y, z, t, ...]
        pixdim = img.header.get_zooms()
        voxel_spacing = pixdim[:3]  # (x, y, z) in mm
    
    # Extract LV mask (label = 3)
    lv_mask = (data == 3).astype(np.float32)
    
    # Calculate volume: num_voxels * voxel_volume
    num_lv_voxels = np.sum(lv_mask)
    voxel_volume_mm3 = np.prod(voxel_spacing)  # mm³
    lv_volume_mm3 = num_lv_voxels * voxel_volume_mm3
    
    # Convert mm³ to mL (1 mL = 1000 mm³)
    lv_volume_ml = lv_volume_mm3 / 1000.0
    
    return lv_volume_ml


def calculate_ef(edv: float, esv: float) -> float:
    """Calculate Ejection Fraction: EF = (EDV - ESV) / EDV * 100"""
    if edv <= 0:
        return 0.0
    return ((edv - esv) / edv) * 100.0


def categorize_ef(ef: float) -> str:
    """Categorize EF into clinical ranges."""
    if ef >= 55:
        return 'normal'
    elif ef >= 40:
        return 'mid'  # mildly reduced
    else:
        return 'reduced'  # severely reduced


def extract_acdc_ef(data_root: str = "cardio_data/raw/acdc") -> pd.DataFrame:
    """
    Extract EF values for all ACDC patients.
    
    Returns:
        DataFrame with columns: patient_id, ef, ef_category, edv, esv, group, height, weight
    """
    results = []
    data_root = Path(data_root)
    
    if not data_root.exists():
        raise FileNotFoundError(f"ACDC data root not found: {data_root}")
    
    # Find all patient directories
    patient_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith('patient')])
    
    print(f"Found {len(patient_dirs)} ACDC patients")
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        info_path = patient_dir / "Info.cfg"
        
        if not info_path.exists():
            print(f"Warning: No Info.cfg found for {patient_id}")
            continue
        
        try:
            # Parse patient info
            info = parse_acdc_info(str(info_path))
            ed_frame = info['ed_frame']
            es_frame = info['es_frame']
            
            # Construct mask paths
            ed_mask_path = patient_dir / f"{patient_id}_frame{ed_frame:02d}_gt.nii.gz"
            es_mask_path = patient_dir / f"{patient_id}_frame{es_frame:02d}_gt.nii.gz"
            
            # Calculate volumes
            if ed_mask_path.exists() and es_mask_path.exists():
                edv = calculate_lv_volume(str(ed_mask_path))
                esv = calculate_lv_volume(str(es_mask_path))
                
                # Calculate EF
                ef = calculate_ef(edv, esv)
                ef_category = categorize_ef(ef)
                
                results.append({
                    'patient_id': patient_id,
                    'ef': round(ef, 2),
                    'ef_category': ef_category,
                    'edv': round(edv, 2),
                    'esv': round(esv, 2),
                    'group': info['group'],
                    'height': info['height'],
                    'weight': info['weight'],
                    'ed_frame': ed_frame,
                    'es_frame': es_frame
                })
                
                print(f"{patient_id}: EF={ef:.1f}% ({ef_category}), EDV={edv:.1f}mL, ESV={esv:.1f}mL")
            else:
                print(f"Warning: Missing segmentation masks for {patient_id}")
                
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            continue
    
    return pd.DataFrame(results)


def update_metadata_with_ef(df_ef: pd.DataFrame, meta_path: str = "meta/master_metadata.csv") -> None:
    """Update master metadata CSV with ACDC EF values."""
    if not os.path.exists(meta_path):
        print(f"Warning: Metadata file not found: {meta_path}")
        return
    
    # Load existing metadata
    df_meta = pd.read_csv(meta_path)
    print(f"Loaded metadata: {len(df_meta)} rows")
    
    # Filter for ACDC patients
    acdc_mask = df_meta['dataset'] == 'acdc'
    acdc_patients = df_meta[acdc_mask].copy()
    print(f"Found {len(acdc_patients)} ACDC entries in metadata")
    
    # Merge EF data
    # Use existing patient_id column in metadata
    # Merge with EF data based on patient_id
    merged = acdc_patients.merge(
        df_ef[['patient_id', 'ef', 'ef_category', 'edv', 'esv', 'group']], 
        on='patient_id', 
        how='left'
    )
    
    # Update original metadata using ef_lv column
    for idx, row in merged.iterrows():
        meta_idx = df_meta.index[(df_meta['dataset'] == 'acdc') & 
                                (df_meta['patient_id'] == row['patient_id'])].tolist()
        if meta_idx:
            for mid in meta_idx:  # Update all entries for this patient
                df_meta.loc[mid, 'ef_lv'] = row['ef']
                # Add EF categorization columns if they don't exist
                for col in ['EF_normal', 'EF_mid', 'EF_reduced']:
                    if col not in df_meta.columns:
                        df_meta[col] = 0
                
                # Set EF category flags
                if pd.notna(row['ef_category']):
                    df_meta.loc[mid, f'EF_{row["ef_category"]}'] = 1
    
    # Save updated metadata
    backup_path = meta_path + ".backup"
    df_meta.to_csv(backup_path, index=False)
    df_meta.to_csv(meta_path, index=False)
    
    # Report updates
    updated_ef = df_meta[acdc_mask]['ef_lv'].notna().sum()
    print(f"Updated EF values for {updated_ef} ACDC entries")
    print(f"Backup saved to: {backup_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract ACDC EF from segmentation masks")
    parser.add_argument("--meta", default="meta/master_metadata.csv", 
                      help="Path to metadata CSV file")
    parser.add_argument("--data_root", default="cardio_data/raw/acdc", 
                      help="Path to ACDC data root")
    parser.add_argument("--output", default="cardio_data/processed/acdc_ef_extracted.csv",
                      help="Output CSV file for extracted EF data")
    
    args = parser.parse_args()
    
    print("=== ACDC EF Extraction ===")
    print(f"Data root: {args.data_root}")
    print(f"Metadata: {args.meta}")
    
    # Extract EF values
    df_ef = extract_acdc_ef(args.data_root)
    
    if len(df_ef) == 0:
        print("No EF values extracted!")
        return
    
    # Save extracted EF data
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_ef.to_csv(args.output, index=False)
    print(f"\nSaved extracted EF data to: {args.output}")
    
    # Statistics
    print(f"\n=== EF Statistics ===")
    print(f"Total patients: {len(df_ef)}")
    print(f"EF range: {df_ef['ef'].min():.1f} - {df_ef['ef'].max():.1f}%")
    print(f"EF mean ± std: {df_ef['ef'].mean():.1f} ± {df_ef['ef'].std():.1f}%")
    print("\nEF Categories:")
    print(df_ef['ef_category'].value_counts())
    print("\nCardiac Groups:")
    print(df_ef['group'].value_counts())
    
    # Update metadata
    if os.path.exists(args.meta):
        print(f"\n=== Updating Metadata ===")
        update_metadata_with_ef(df_ef, args.meta)
    else:
        print(f"Metadata file not found: {args.meta}")
        print("Run this script after creating the metadata file.")


if __name__ == "__main__":
    main()