
#!/usr/bin/env python3
import argparse, json, numpy as np, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', default='meta/master_metadata.csv')
    ap.add_argument('--phase', default='ED', choices=['ED','ES','both'])
    ap.add_argument('--n', type=int, default=20, help='number of samples to scan')
    args = ap.parse_args()

    df = pd.read_csv(args.meta)
    df = df[(df['dataset']=='acdc')]
    if args.phase != 'both':
        df = df[df['phase']==args.phase]

    # expand JSON-like paths column
    import json
    def get_path(row, key):
        paths = json.loads(row['paths'])
        return paths.get(key, None)

    ed_key = 'nii_mask_ED'; es_key = 'nii_mask_ES'
    rows = df.sample(min(args.n, len(df)), random_state=42)

    uniq_counts = []
    for _, row in rows.iterrows():
        key = ed_key if row['phase']=='ED' else es_key
        try:
            p = json.loads(row['paths']).get(key, None)
        except Exception:
            p = None
        if not p:
            continue
        path = Path(p)
        if not path.exists():
            # allow relative paths from repo root
            alt = Path('cardiac_early_detection_repo-main') / p
            path = alt if alt.exists() else path
        if not path.exists():
            uniq_counts.append({'patient_id': row['patient_id'], 'phase': row['phase'], 'error': f'not found: {p}'})
            continue
        try:
            import nibabel as nib
            m = nib.load(str(path)).get_fdata().astype(np.uint8)
            u, c = np.unique(m, return_counts=True)
            uniq_counts.append({
                'patient_id': row['patient_id'],
                'phase': row['phase'],
                'labels': ','.join([str(int(x)) for x in u.tolist()]),
                'counts': ','.join([str(int(x)) for x in c.tolist()])
            })
        except Exception as e:
            uniq_counts.append({'patient_id': row['patient_id'], 'phase': row['phase'], 'error': str(e)})

    out = pd.DataFrame(uniq_counts)
    out.to_csv('label_scan_acdc.csv', index=False)
    print('Wrote label_scan_acdc.csv with', len(out), 'rows')

if __name__ == '__main__':
    main()
