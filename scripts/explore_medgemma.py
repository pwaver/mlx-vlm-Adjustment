# %% Imports and configuration
"""
Explore MedGemma on coronary angiography frames.

Load a model once, then iterate: pick an angiogram, display a frame,
and run prompts. The angiogram/prompt cells can be re-evaluated
independently without reloading the model.
"""
1+1
import os

import h5py
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from IPython.display import display, Markdown

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template


def show_result(prompt_text, result):
    """Pretty-print a GenerationResult with soft-wrapped markdown."""
    tps = f"{result.generation_tps:.1f} tok/s" if result.generation_tps else ""
    stats = f"[{result.generation_tokens} tokens, {tps}]"
    display(Markdown(f"**Model:** {MODEL_ID}\n\n**Q:** {prompt_text}\n\n{result.text}\n\n---\n*{stats}*"))

MODEL_ID = "mlx-community/medgemma-27b-it-8bit"
# MODEL_ID = "mlx-community/medgemma-1.5-4b-it-bf16"
# MODEL_ID = "mlx-community/medgemma-4b-it-8bit"

CLINICAL_PROMPT = """\
You are assisting a cardiologist who manages patients with coronary artery \
disease but does not perform coronary angiography. Analyze this coronary \
angiogram and provide:
1. Vessel identification: Which coronary artery or arteries are visualized, \
and what projection does this appear to be?
2. Lesion assessment: Describe any stenoses, their approximate location \
(proximal, mid, distal), and estimated severity (mild <50%, moderate 50-70%, \
severe >70%, or occlusion).
3. Myocardial territory at risk: Based on any lesions identified, which \
myocardial segments and wall regions would be affected?
4. Clinical significance: Summarize the findings in terms of their \
implications for medical management, need for revascularization, or further \
workup."""

FOREIGN_OBJECTS_PROMPT = """\
Examine this coronary angiogram and identify all implanted foreign objects \
or devices visible in the image. For each object found, state what it is \
and where it appears. Objects to look for include but are not limited to: \
diagnostic or guiding catheter, microguide wire, coronary stent, \
prosthetic heart valve (mechanical or bioprosthetic), \
ICD (implantable cardioverter-defibrillator) leads and generator, \
pacemaker leads and generator, \
CCM (cardiac contractility modulation) device, \
CRT (cardiac resynchronization therapy) leads, \
LVAD (left ventricular assist device) cannula, \
IABP (intra-aortic balloon pump), \
Impella catheter, \
ECMO (extracorporeal membrane oxygenation) cannulae, \
sternotomy wires, surgical clips, \
ECG (electrocardiogram) leads and electrodes, \
Swan-Ganz (pulmonary artery) catheter, \
central venous catheter, \
chest tube, mediastinal drain, \
atrial or ventricular septal occluder device, \
PFO (patent foramen ovale) closure device, \
LAA (left atrial appendage) occluder, \
aortic endograft or stent graft, \
embolization coils, and vascular closure device. \
If no foreign objects are visible, state that explicitly."""

PROMPTS = [
    "Describe the vascular anatomy visible in this coronary angiogram.",
    "What abnormalities, if any, are present in this coronary angiogram?",
    "Describe the contrast opacification pattern in this image.",
    FOREIGN_OBJECTS_PROMPT,
    CLINICAL_PROMPT,
]

H5_FILENAME = "AngiogramsDistilledUInt8List.h5"
H5_CANDIDATES = [
    f"/Volumes/X10Pro/AWIBuffer/Angiostore/{H5_FILENAME}",   # macOS
    f"/media/billb/WMDPP/Angiostore/{H5_FILENAME}",          # Ubuntu / WDMPP
]

# %% Load MedGemma model (run once)
print(f"Loading model: {MODEL_ID}")
model, processor = load(MODEL_ID)
print("Model loaded.")

# %% Load angiogram frame from HDF5
# DATASET_KEY = "Angios_129_rev"
# DATASET_KEY = "Angios_188_rev"
# DATASET_KEY = "07_Case_CSF8W6GY_6"
# DATASET_KEY = "Angios_061_rev"
 DATASET_KEY = "Angios_096_rev"
# DATASET_KEY = "Napari_48_rev"
# DATASET_KEY = "Napari_36_rev"
# DATASET_KEY = "Napari_29_rev"

h5_path = next((p for p in H5_CANDIDATES if os.path.exists(p)), None)
if h5_path is None:
    raise FileNotFoundError(
        f"Could not find {H5_FILENAME} at any candidate location:\n"
        + "\n".join(f"  - {p}" for p in H5_CANDIDATES)
    )

print(f"Loading from {h5_path} [{DATASET_KEY}]")
with h5py.File(h5_path, "r") as f:
    available_keys = list(f.keys())
    print(f"Available keys ({len(available_keys)}): {available_keys[:10]}{'...' if len(available_keys) > 10 else ''}")
    frames = f[DATASET_KEY][:]  # uint8, (N, 512, 512)

print(f"Dataset '{DATASET_KEY}': shape={frames.shape}, dtype={frames.dtype}")

# %% Display frame
frame_index = 35
image = Image.fromarray(frames[frame_index])

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(frames[frame_index], cmap="gray")
ax.set_title(f"Frame {frame_index} — {DATASET_KEY}")
ax.axis("off")
plt.tight_layout()
plt.show()

print(f"Image size: {image.size}, mode: {image.mode}")

# %% Prompt 1 — Vascular anatomy
prompt_text = PROMPTS[0]
prompt = apply_chat_template(
    processor, config=model.config, prompt=prompt_text, num_images=1,
)
output = generate(model, processor, prompt, image=[image], max_tokens=512)
show_result(prompt_text, output)

# %% Prompt 2 — Abnormalities
prompt_text = PROMPTS[1]
prompt = apply_chat_template(
    processor, config=model.config, prompt=prompt_text, num_images=1,
)
output = generate(model, processor, prompt, image=[image], max_tokens=512)
show_result(prompt_text, output)

# %% Prompt 3 — Contrast opacification
prompt_text = PROMPTS[2]
prompt = apply_chat_template(
    processor, config=model.config, prompt=prompt_text, num_images=1,
)
output = generate(model, processor, prompt, image=[image], max_tokens=512)
show_result(prompt_text, output)

# %% Prompt 4 — Foreign objects inventory
prompt_text = PROMPTS[3]
prompt = apply_chat_template(
    processor, config=model.config, prompt=prompt_text, num_images=1,
)
output = generate(model, processor, prompt, image=[image], max_tokens=1024)
show_result(prompt_text, output)

# %% Prompt 5 — Clinical assessment (cardiologist/intensivist)
prompt_text = PROMPTS[4]
prompt = apply_chat_template(
    processor, config=model.config, prompt=prompt_text, num_images=1,
)
output = generate(model, processor, prompt, image=[image], max_tokens=2048)
show_result(prompt_text, output)

# %% Multi-frame inference (commented out — mnemonic for future exploration)
# MedGemma 1.5 supports multiple images per prompt (designed for CT/MRI slices
# and longitudinal X-ray comparison). This could be used to pass a temporal
# sequence of angiogram frames spanning the cardiac cycle, letting the model
# reason about contrast flow dynamics, filling patterns, or collateral vessels
# across time. Uncomment and adjust frame range to experiment.
#
# frames_pil = [Image.fromarray(frames[i]) for i in range(10, 20)]
# prompt_text = (
#     "These images are sequential frames from a coronary angiogram captured "
#     "during contrast injection. Describe how the contrast opacification "
#     "evolves across the sequence, identify the vessels visualized, and note "
#     "any regions of delayed filling or flow limitation."
# )
# prompt = apply_chat_template(
#     processor, config=model.config, prompt=prompt_text,
#     num_images=len(frames_pil),
# )
# output = generate(model, processor, prompt, image=frames_pil, max_tokens=2048)
# show_result(prompt_text, output)

# %%