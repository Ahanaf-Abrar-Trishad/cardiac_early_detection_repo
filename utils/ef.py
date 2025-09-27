# utils/ef.py
import numpy as np

def volume_ml(mask_binary, pixdim_x, pixdim_y, slice_thickness):
    voxel_vol = pixdim_x * pixdim_y * slice_thickness / 1000.0  # mm^3 -> mL
    voxels = np.count_nonzero(mask_binary)
    return voxels * voxel_vol

def ef_from_cavity(ed_mask, es_mask, pixdim_x, pixdim_y, slice_thickness):
    edv = volume_ml(ed_mask, pixdim_x, pixdim_y, slice_thickness)
    esv = volume_ml(es_mask, pixdim_x, pixdim_y, slice_thickness)
    if edv <= 0: return np.nan, edv, esv
    ef = (edv - esv) / edv
    if ef < 0 or ef > 1:
        return np.nan, edv, esv
    return float(ef), edv, esv

def area_ef_from_masks(ed_mask_2d, es_mask_2d):
    ed = float(np.count_nonzero(ed_mask_2d))
    es = float(np.count_nonzero(es_mask_2d))
    if ed <= 0: return np.nan
    ef = (ed - es) / ed
    if ef < 0 or ef > 1: return np.nan
    return float(ef)
