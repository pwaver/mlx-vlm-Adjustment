# %% Configuration
"""
Merge missing angiogram datasets from WebknossosAngiogramsRevisedUInt8List.h5
into AngiogramsDistilledUInt8List.h5. Only copies datasets whose keys are
not already present in the destination file.
"""

import os
import h5py

H5_DIR = "/Volumes/X10Pro/AWIBuffer/Angiostore"
DST_FILE = os.path.join(H5_DIR, "AngiogramsDistilledUInt8List.h5")
SRC_FILE = os.path.join(H5_DIR, "WebknossosAngiogramsRevisedUInt8List.h5")

# %% Inspect and compute diff
with h5py.File(DST_FILE, "r") as dst, h5py.File(SRC_FILE, "r") as src:
    dst_keys = set(dst.keys())
    src_keys = set(src.keys())

common = sorted(dst_keys & src_keys)
keys_to_add = sorted(src_keys - dst_keys)

print(f"Destination: {len(dst_keys)} datasets")
print(f"Source:      {len(src_keys)} datasets")
print(f"Common:      {len(common)}")
print(f"To copy:     {len(keys_to_add)}")
for k in keys_to_add:
    print(f"  {k}")

# %% Copy missing datasets
with h5py.File(DST_FILE, "a") as dst, h5py.File(SRC_FILE, "r") as src:
    for i, key in enumerate(keys_to_add, 1):
        src.copy(key, dst)
        ds = dst[key]
        print(f"  [{i}/{len(keys_to_add)}] {key}: shape={ds.shape}, dtype={ds.dtype}")

print(f"\nDone. Copied {len(keys_to_add)} datasets.")

# %% Verify
with h5py.File(DST_FILE, "r") as f:
    final_keys = sorted(f.keys())
    print(f"Destination now has {len(final_keys)} datasets.")

# %%