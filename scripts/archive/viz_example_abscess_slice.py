"""
Same isolation approach as the earlier centroid-slice diagnostic, but for
the ABSCESS/CASE0000000 sample that still looks binary when run through
evaluate_ed_official_test.py (submit_ed_official_test_eval.sh).

Loads the exact image/mask evaluate_ed_official_test.py uses via the same
load_nifti_canonical() GrounderDataset uses, finds the GT mask centroid slice
(same as evaluate_ed_official_test.py's _mask_centroid), and clips that raw-HU
slice to [-150, 250] directly (its display window) -- no normalized-space
math, no model involved.

If this looks fine: the bug is somewhere between raw load and the model-eval
visualization path (batching, trimming, etc in evaluate_ed_official_test.py).
If this still looks binary/washed out: something about this specific
image/mask pair or the ED_TEST_SET_resampled mask is the actual cause.
"""
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import load_nifti_canonical

IMAGE = "/path/to/data/inhouse_abdominal_ct/nifti_resampled/CASE0000000/2__ST.nii.gz"
MASK = "/path/to/data/inhouse_abdominal_ct/ED_TEST_SET_resampled/ABSCESS/CASE0000000/Struct_ABSCESS_2_118_0.nii.gz"

image = load_nifti_canonical(IMAGE)  # (D, H, W) raw HU, canonicalized
mask = load_nifti_canonical(MASK)    # (D, H, W)

coords = np.argwhere(mask > 0.5)
if len(coords) == 0:
    d = image.shape[0] // 2
    print("mask is empty; falling back to middle slice")
else:
    d, h, w = coords.mean(axis=0).astype(int)
    print(f"GT mask centroid slice: d={d} (image shape {image.shape}, mask shape {mask.shape})")

img_slice = image[d, :, :]
print(f"raw HU range at this slice: [{img_slice.min():.1f}, {img_slice.max():.1f}]")
pct = np.percentile(img_slice, [1, 5, 50, 95, 99])
print(f"percentiles: p1={pct[0]:.1f} p5={pct[1]:.1f} p50={pct[2]:.1f} p95={pct[3]:.1f} p99={pct[4]:.1f}")

img_clipped = np.clip(img_slice, -150, 250)
plt.imshow(img_clipped, cmap="gray")
plt.title(f"CASE0000000 slice d={d} (GT centroid), clipped [-150, 250] HU")
plt.savefig("image_example_abscess_clipped.png")
plt.close()
