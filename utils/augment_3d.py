
import numpy as np

def random_flip_3d(v, m, p=0.5, axes=(0,1,2)):
    """Randomly flip volume and mask along each axis with probability p.
    v: (Z,Y,X) float32, m: (Z,Y,X) int
    """
    for ax in axes:
        if np.random.rand() < p:
            v = np.flip(v, axis=ax).copy()
            m = np.flip(m, axis=ax).copy()
    return v, m

def random_rot90_3d(v, m, p=0.5):
    """Apply a random 90-degree rotation around a random pair of axes."""
    if np.random.rand() < p:
        # choose an axis pair among (Z,Y), (Z,X), (Y,X)
        pairs = [(0,1), (0,2), (1,2)]
        k = np.random.randint(1, 4)  # 90, 180, 270
        a1, a2 = pairs[np.random.randint(0, len(pairs))]
        v = np.rot90(v, k=k, axes=(a1, a2)).copy()
        m = np.rot90(m, k=k, axes=(a1, a2)).copy()
    return v, m

def random_intensity_jitter(v, p=0.5, brightness=0.1, contrast=0.1, gamma=0.1):
    """Apply simple intensity jitter to the volume only (mask unaffected).
    brightness: add offset in [-b, b] * std
    contrast: scale in [1-c, 1+c]
    gamma: exponent in [1-g, 1+g]
    """
    if np.random.rand() < p:
        std = float(v.std() + 1e-6)
        v = v + (np.random.uniform(-brightness, brightness) * std)
        v = v * (1.0 + np.random.uniform(-contrast, contrast))
        g = 1.0 + np.random.uniform(-gamma, gamma)
        v = np.sign(v) * (np.abs(v) ** g)
        # re-standardize
        v = (v - v.mean()) / (v.std() + 1e-6)
    return v

def default_aug_3d(v, m):
    """A default, lightweight augmentation: flips + occasional rot90 + intensity jitter."""
    v, m = random_flip_3d(v, m, p=0.5)
    v, m = random_rot90_3d(v, m, p=0.25)
    v = random_intensity_jitter(v, p=0.5, brightness=0.1, contrast=0.1, gamma=0.05)
    return v, m
