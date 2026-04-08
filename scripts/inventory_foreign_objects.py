# %% Imports and configuration
"""
Inventory foreign objects across all coronary angiogram datasets.

Uses MedGemma 27B to scan frame 2 (pre-contrast) of each dataset in the
HDF5 file and catalog visible implanted devices. Results are written
incrementally to a markdown table for later parsing.
"""

import os
import re

import h5py
from PIL import Image

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

MODEL_ID = "mlx-community/medgemma-27b-it-8bit"
FRAME_INDEX = 2
MAX_TOKENS = 1028

H5_FILENAME = "AngiogramsDistilledUInt8List.h5"
H5_CANDIDATES = [
    f"/Volumes/X10Pro/AWIBuffer/Angiostore/{H5_FILENAME}",   # macOS
    f"/media/billb/WMDPP/Angiostore/{H5_FILENAME}",          # Ubuntu / WDMPP
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "foreign_objects_inventory.md")

INVENTORY_PROMPT = """\
List all implanted foreign objects or devices visible in this coronary angiogram. \
Return ONLY a comma-separated list of object names, nothing else. \
If no foreign objects are visible, return exactly: none

Objects to look for include: \
diagnostic catheter, guiding catheter, guidewire, coronary stent, \
prosthetic heart valve, mechanical heart valve, bioprosthetic heart valve, \
ICD leads, ICD generator, pacemaker leads, pacemaker generator, \
CCM device, CRT leads, LVAD cannula, IABP catheter, Impella catheter, \
ECMO cannulae, sternotomy wires, surgical clips, \
ECG leads, ECG electrodes, Swan-Ganz catheter, central venous catheter, \
chest tube, mediastinal drain, atrial septal occluder, ventricular septal occluder, \
PFO closure device, LAA occluder, aortic endograft, \
embolization coils, vascular closure device, pigtail catheter.

Example valid responses:
- guiding catheter, guidewire, sternotomy wires
- none
- pacemaker leads, pacemaker generator, guiding catheter"""


def clean_response(text: str) -> str:
    """Normalize model output to a clean comma-delimited list."""
    text = text.strip().strip(".")
    text = re.sub(r"^[-*]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def write_results(path: str, results: dict, all_keys: list):
    """Write full markdown table, preserving key order from HDF5."""
    with open(path, "w") as f:
        f.write("# Foreign Objects Inventory\n\n")
        f.write("| Dataset | Foreign Objects |\n")
        f.write("|---------|----------------|\n")
        for key in all_keys:
            if key in results:
                f.write(f"| {key} | {results[key]} |\n")


def parse_existing_results(path: str) -> dict:
    """Read the markdown table and return {dataset_key: foreign_objects_str}."""
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

        if ds.shape[0] < 3:
            result_text = "*skipped (fewer than 3 frames)*"
            print(f"[{global_index}/{total}] {key}: SKIPPED ({ds.shape[0]} frames)")
        else:
            frame = ds[FRAME_INDEX]
            image = Image.fromarray(frame)

            prompt = apply_chat_template(
                processor, config=model.config,
                prompt=INVENTORY_PROMPT, num_images=1,
            )
            output = generate(
                model, processor, prompt,
                image=[image], max_tokens=MAX_TOKENS,
            )
            result_text = clean_response(output.text)
            print(f"[{global_index}/{total}] {key}: {result_text}")

        completed[key] = result_text
        write_results(OUTPUT_FILE, completed, all_keys)

print(f"\nDone. {len(completed)} datasets cataloged.")

# %% Summary
print(f"Results written to: {OUTPUT_FILE}")
print(f"Total datasets: {total}")
skipped = sum(1 for v in completed.values() if "skipped" in v)
with_objects = sum(1 for v in completed.values() if v != "none" and "skipped" not in v)
without_objects = sum(1 for v in completed.values() if v == "none")
print(f"  With foreign objects: {with_objects}")
print(f"  No foreign objects:   {without_objects}")
print(f"  Skipped:              {skipped}")

# %%
