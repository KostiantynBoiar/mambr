"""Kaggle GPU env check — paste as the first cell of the Kaggle notebook.

Confirms the heavy toolchain is ready BEFORE any feature extraction:
  1. CUDA GPU is visible,
  2. the frozen VideoMAE backbone loads,
  3. one dummy 16-frame clip embeds end-to-end.

Run on Kaggle with: Settings -> Accelerator: GPU, Internet: ON.
"""

import subprocess
import sys

# --- 1. Install the extras Kaggle's base image lacks ---
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers>=4.40",
        "av>=11",
        "librosa>=0.10",
        "soundfile>=0.12",
        "pyarrow>=14",
    ],
    check=True,
)

import torch  # noqa: E402
from transformers import VideoMAEModel  # noqa: E402

# --- 2. GPU gate ---
assert torch.cuda.is_available(), "No CUDA GPU — set Accelerator: GPU in Kaggle settings."
device = "cuda"
print("CUDA device:", torch.cuda.get_device_name(0))

# --- 3. Load frozen VideoMAE (nearest-available HF checkpoint; swap if authors share theirs) ---
VIDEO_CKPT = "MCG-NJU/videomae-base"  # K400-pretrained ViT-B; 16 frames, 224x224
model = VideoMAEModel.from_pretrained(VIDEO_CKPT).to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)  # FROZEN — never trained (CLAUDE.md §3)

# --- 4. Embed one dummy 16-frame clip ---
with torch.no_grad():
    pixel_values = torch.randn(1, 16, 3, 224, 224, device=device)  # [B, T, C, H, W]
    out = model(pixel_values)
    emb = out.last_hidden_state.mean(dim=1)  # mean-pool tokens -> [1, hidden]
print(
    "VideoMAE OK. Token seq:",
    tuple(out.last_hidden_state.shape),
    "-> pooled embedding:",
    tuple(emb.shape),
)
print("GATE PASSED: GPU + frozen VideoMAE + one-clip embed all work.")
