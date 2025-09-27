#!/usr/bin/env python3
"""
Quick QC: EF histogram and a few overlay grids (stub).
"""
import argparse, pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="meta/master_metadata.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.meta)
    print(df.groupby(["dataset","split"])["ef_lv"].describe())
if __name__ == "__main__":
    main()
