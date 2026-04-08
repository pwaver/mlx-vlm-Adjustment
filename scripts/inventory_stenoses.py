# %% Imports and configuration
"""
Inventory coronary artery stenoses across all angiogram datasets.

Uses MedGemma 27B to scan a mid-sequence frame (peak contrast) of each
dataset in the HDF5 file and catalog any stenoses found, with vessel
name and location. Results are written incrementally to a markdown table.
"""

import os
import re

import h5py
from PIL import Image

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

MODEL_ID = "mlx-community/medgemma-27b-it-8bit"
MAX_TOKENS = 2048

H5_FILENAME = "AngiogramsDistilledUInt8List.h5"
H5_CANDIDATES = [
    f"/Volumes/X10Pro/AWIBuffer/Angiostore/{H5_FILENAME}",   # macOS
    f"/media/billb/WMDPP/Angiostore/{H5_FILENAME}",          # Ubuntu / WDMPP
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "stenosis_inventory.md")

STENOSIS_PROMPT = """\
Examine this coronary angiogram for stenoses. For each stenosis found, \
report the vessel name and the location within that vessel. \
Return ONLY a semicolon-separated list of findings, each formatted as: \
vessel name (location, severity). \
If no stenoses are visible, return exactly: none

Use these vessel abbreviations: \
LM (left main), LAD (left anterior descending), \
LCx (left circumflex), RCA (right coronary artery), \
D1/D2 (diagonal branches), OM1/OM2 (obtuse marginal branches), \
PDA (posterior descending artery), PLV (posterolateral branch), \
ramus (ramus intermedius).

Use these location terms: ostial, proximal, mid, distal.

Use these severity terms: mild (<50%), moderate (50-70%), \
severe (>70%), subtotal, total occlusion.

Example valid responses:
- LAD (mid, severe >70%); LCx (proximal, moderate 50-70%)
- RCA (mid, total occlusion); LAD (proximal, mild <50%)
- none"""


def clean_response(text: str) -> str:
    """Normalize model output."""
    text = text.strip().strip(".")
    text = re.sub(r"^[-*]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def write_results(path: str, results: dict, all_keys: list):
    """Write full markdown table, preserving key order from HDF5."""
    with open(path, "w") as f:
        f.write("# Stenosis Inventory\n\n")
        f.write("| Dataset | Stenoses |\n")
        f.write("|---------|----------|\n")
        for key in all_keys:
            if key in results:
                f.write(f"| {key} | {results[key]} |\n")


def parse_existing_results(path: str) -> dict:
    """Read the markdown table and return {dataset_key: stenosis_str}."""
    results = {}
    if not os.path.exists(path):
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|") or line.startswith("| Dataset") or line.startswith("|---"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                results[parts[1]] = parts[2]
    return results


# %% Resume: check for existing results
completed = parse_existing_results(OUTPUT_FILE)
print(f"Previously completed: {len(completed)} datasets")

# %% Load model (run once)
print(f"Loading model: {MODEL_ID}")
model, processor = load(MODEL_ID)
print("Model loaded.")

# %% Open HDF5 and build work list
h5_path = next((p for p in H5_CANDIDATES if os.path.exists(p)), None)
if h5_path is None:
    raise FileNotFoundError(
        f"Could not find {H5_FILENAME} at any candidate location:\n"
        + "\n".join(f"  - {p}" for p in H5_CANDIDATES)
    )

with h5py.File(h5_path, "r") as f:
    all_keys = sorted(f.keys())

total = len(all_keys)
pending_keys = [k for k in all_keys if k not in completed]
print(f"Total datasets: {total}, already done: {total - len(pending_keys)}, remaining: {len(pending_keys)}")

# %% Run inventory
with h5py.File(h5_path, "r") as f:
    for i, key in enumerate(pending_keys, 1):
        ds = f[key]
        global_index = all_keys.index(key) + 1
        n_frames = ds.shape[0]

        if n_frames < 3:
            result_text = "*skipped (fewer than 3 frames)*"
            print(f"[{global_index}/{total}] {key}: SKIPPED ({n_frames} frames)")
        else:
            # Use 40% of sequence length for peak contrast opacification
            frame_index = int(n_frames * 0.4)
            frame = ds[frame_index]
            image = Image.fromarray(frame)

            prompt = apply_chat_template(
                processor, config=model.config,
                prompt=STENOSIS_PROMPT, num_images=1,
            )
            output = generate(
                model, processor, prompt,
                image=[image], max_tokens=MAX_TOKENS,
            )
            result_text = clean_response(output.text)
            print(f"[{global_index}/{total}] {key} (frame {frame_index}/{n_frames}): {result_text}")

        completed[key] = result_text
        write_results(OUTPUT_FILE, completed, all_keys)

print(f"\nDone. {len(completed)} datasets cataloged.")

# %% Summary
print(f"Results written to: {OUTPUT_FILE}")
print(f"Total datasets: {total}")
skipped = sum(1 for v in completed.values() if "skipped" in v)
with_stenosis = sum(1 for v in completed.values() if v != "none" and "skipped" not in v)
without_stenosis = sum(1 for v in completed.values() if v == "none")
print(f"  With stenoses:    {with_stenosis}")
print(f"  No stenoses:      {without_stenosis}")
print(f"  Skipped:          {skipped}")

# %%
