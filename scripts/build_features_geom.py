#!/usr/bin/env python3
import itertools, json
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import (
    label as cc_label, binary_fill_holes, binary_closing,
    generate_binary_structure, distance_transform_edt
)

def best_mapping(pred, gt, n=4):
    cm = np.zeros((n,n), dtype=np.int64)
    for pl in range(n):
        for gl in range(n):
            cm[pl,gl] = np.sum((pred==pl) & (gt==gl))
    best_perm, best_sum = None, -1
    for perm in itertools.permutations(range(n)):
        s = sum(cm[pl, perm[pl]] for pl in range(n))
        if s > best_sum:
            best_sum, best_perm = s, np.array(perm)
    return best_perm

def keep_lcc(m):
    if m.sum()==0: return m
    st = generate_binary_structure(3,1)
    lab, n = cc_label(m, structure=st)
    if n<=1: return m
    cnt = np.bincount(lab.ravel()); cnt[0]=0
    return lab==(cnt.argmax())

def clean_labels(vol):
    out = np.zeros_like(vol, dtype=np.int16)
    for c in (1,2,3):
        m = (vol==c)
        m2 = np.zeros_like(m)
        for z in range(m.shape[0]):
            if m[z].any():
                mz = binary_fill_holes(m[z])
                mz = binary_closing(mz, structure=np.ones((3,3),bool))
                m2[z] = mz
        m = keep_lcc(m2)
        out[m] = c
    out[(out==1)&(out==2)] = 2
    out[(out>0)&(vol==3)] = 3
    return out

def voxel_sizes(p): return nib.load(p).header.get_zooms()[:3]

def thickness_stats(myo, spacing):
    edt = distance_transform_edt(myo, sampling=spacing)
    th = 2.0*edt[myo]
    if th.size==0: return dict(th_mean=np.nan, th_p95=np.nan, th_max=np.nan)
    return dict(
        th_mean=float(np.nanmean(th)),
        th_p95=float(np.nanpercentile(th,95)),
        th_max=float(np.nanmax(th)),
    )

def lv_axis_ratio(cav, spacing):
    idx = np.argwhere(cav)
    if idx.shape[0]<20: return dict(lv_axis23=np.nan, lv_axis13=np.nan)
    pts = idx * np.asarray(spacing)[None,:]
    X = pts - pts.mean(0)
    w,_ = np.linalg.eigh(X.T@X/(len(pts)-1))
    w = np.sort(w)[::-1]
    a1,a2,a3 = np.sqrt(np.maximum(w,1e-9))
    return dict(lv_axis23=float(a2/a1), lv_axis13=float(a3/a1))

def robust_ef(ed, es):
    if ed < 20: return np.nan
    x = 100.0*(ed-es)/max(ed,1e-6)
    return x if -5 <= x <= 90 else np.nan

def main():
    meta = pd.read_csv("meta/master_metadata.csv")
    path_map = {r.study_id: json.loads(r.paths) for _,r in meta[meta.dataset=="acdc"].iterrows()}
    ed = pd.read_csv("results/acdc_oof_index_ED.csv")
    es = pd.read_csv("results/acdc_oof_index_ES.csv").set_index("patient_id")

    rows=[]
    for _,r in ed.iterrows():
        pid=r["patient_id"]
        if pid not in es.index: continue
        sidE=r["study_id"]; sidS=es.loc[pid,"study_id"]
        pE = nib.load(r["pred_path"]); vE = pE.get_fdata().astype(np.int16)
        pS = nib.load(es.loc[pid,"pred_path"]); vS = pS.get_fdata().astype(np.int16)
        gE_path = path_map.get(sidE,{}).get("nii_mask_ED","")
        if not gE_path: continue
        gE = nib.load(gE_path).get_fdata().astype(np.int16)

        gS_path = path_map.get(sidS,{}).get("nii_mask_ES","")
        perm_ED = best_mapping(vE, gE, 4)
        if gS_path:
            gS = nib.load(gS_path).get_fdata().astype(np.int16)
            perm_ES = best_mapping(vS, gS, 4)
            if not np.array_equal(perm_ED, perm_ES):
                print(f"Warning: {pid} ED perm {perm_ED} != ES perm {perm_ES} — using ED perm")
        vE = perm_ED[vE]; vS = perm_ED[vS]
        vE = clean_labels(vE); vS = clean_labels(vS)

        spE = voxel_sizes(r["pred_path"]); spS = voxel_sizes(es.loc[pid,"pred_path"])
        vxE = np.prod(spE)/1000.0;        vxS = np.prod(spS)/1000.0

        RVEDV=(vE==1).sum()*vxE; RVESV=(vS==1).sum()*vxS
        LVEDV=(vE==3).sum()*vxE; LVESV=(vS==3).sum()*vxS
        MYO_ED=(vE==2).sum()*vxE; MYO_ES=(vS==2).sum()*vxS

        if (LVESV>LVEDV*1.03) and (RVESV>RVEDV*1.03):
            RVEDV,RVESV = RVESV,RVEDV
            LVEDV,LVESV = LVESV,LVEDV
            MYO_ED,MYO_ES = MYO_ES,MYO_ED
            vE, vS = vS, vE
            spE, spS = spS, spE

        thE = thickness_stats(vE==2, spE); thS = thickness_stats(vS==2, spS)
        shE = lv_axis_ratio(vE==3, spE)

        rows.append({
            "patient_id": pid,
            "RVEDV_ml": RVEDV, "RVESV_ml": RVESV, "RVEF_pct_robust": robust_ef(RVEDV,RVESV),
            "LVEDV_ml": LVEDV, "LVESV_ml": LVESV, "LVEF_pct_robust": robust_ef(LVEDV,LVESV),
            "MYO_ED_ml": MYO_ED, "MYO_ES_ml": MYO_ES,
            "MYO_ES_to_ED_ratio": (MYO_ES/MYO_ED) if MYO_ED>1e-3 else np.nan,
            "MYO_th_mean_ED_mm": thE["th_mean"], "MYO_th_p95_ED_mm": thE["th_p95"], "MYO_th_max_ED_mm": thE["th_max"],
            "MYO_th_mean_ES_mm": thS["th_mean"], "MYO_th_p95_ES_mm": thS["th_p95"], "MYO_th_max_ES_mm": thS["th_max"],
            "MYO_th_ratio_ES_ED": (thS["th_mean"]/thE["th_mean"]) if (thE["th_mean"] and not np.isnan(thE["th_mean"])) else np.nan,
            "LV_axis23_ED": shE["lv_axis23"], "LV_axis13_ED": shE["lv_axis13"],
            "study_id_ED": sidE, "study_id_ES": sidS
        })

    out = pd.DataFrame(rows).sort_values("patient_id")
    Path("results").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/acdc_oof_features_geom.csv", index=False)
    print("wrote results/acdc_oof_features_geom.csv")

if __name__ == "__main__":
    main()
