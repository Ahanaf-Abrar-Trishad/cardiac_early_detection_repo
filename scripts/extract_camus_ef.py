#!/usr/bin/env python3
"""
Extract EF values from CAMUS Info files and update master metadata.
"""
import argparse
import pandas as pd
import json
from pathlib import Path

def extract_ef_from_info(info_path):
    """Extract EF value from CAMUS Info_*.cfg file."""
    try:
        with open(info_path, 'r') as f:
            for line in f:
                if line.startswith('EF:'):
                    return float(line.split(':')[1].strip())
    except:
        pass
    return None

def categorize_ef(ef_value):
    """Categorize EF into normal/mid/reduced based on clinical thresholds."""
    if ef_value is None:
        return None
    if ef_value >= 55:
        return "normal"
    elif ef_value >= 40:
        return "mid" 
    else:
        return "reduced"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="meta/master_metadata.csv", help="Master metadata CSV")
    ap.add_argument("--raw-camus", default="cardio_data/raw/camus", help="Raw CAMUS data directory")
    ap.add_argument("--output", help="Output CSV (default: update input file)")
    args = ap.parse_args()
    
    # Load metadata
    df = pd.read_csv(args.meta)
    
    # Process CAMUS entries
    for idx, row in df.iterrows():
        if row['dataset'] != 'camus':
            continue
            
        patient_id = row['patient_id']
        view = row.get('view', '4CH')  # Default to 4CH if view not specified
        
        # Look for Info file
        info_path = Path(args.raw_camus) / patient_id / f"Info_{view}.cfg"
        
        if info_path.exists():
            ef_value = extract_ef_from_info(info_path)
            if ef_value is not None:
                df.loc[idx, 'ef_lv'] = ef_value
                df.loc[idx, 'ef_bin'] = categorize_ef(ef_value)
                print(f"{patient_id} {view}: EF={ef_value:.1f}% -> {categorize_ef(ef_value)}")
    
    # Save updated metadata
    output_path = args.output if args.output else args.meta
    df.to_csv(output_path, index=False)
    print(f"\nUpdated metadata saved to: {output_path}")
    
    # Print summary
    camus_df = df[df['dataset'] == 'camus']
    ef_counts = camus_df['ef_bin'].value_counts()
    print(f"\nCAMUS EF label distribution:")
    for label, count in ef_counts.items():
        print(f"  {label}: {count}")

if __name__ == "__main__":
    main()