import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path("/Users/pranavthobhani/CV Project")
DATA_DIR = PROJECT_ROOT / "data" / "pilot"
OUT_DIR = PROJECT_ROOT / "demo_dryrun_outputs"
OUT_DIR.mkdir(exist_ok=True)

# Select representative real and fake face
real_img = cv2.cvtColor(cv2.imread(str(DATA_DIR / "real" / "real_0000.png")), cv2.COLOR_BGR2RGB)
fake_img = cv2.cvtColor(cv2.imread(str(DATA_DIR / "fake" / "fake_0000.png")), cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.8), dpi=300)

# Real Face
axes[0].imshow(real_img)
axes[0].set_title("AUTHENTIC REAL FACE\n(FFHQ Optical Capture)", fontsize=11, fontweight="bold", color="#065F46", pad=10)
axes[0].axis("off")
# Annotations/box around real
for spine in axes[0].spines.values():
    spine.set_edgecolor("#065F46")
    spine.set_linewidth(3)
axes[0].text(0.5, -0.08, "• Natural lens point-spread function\n• Uniform Poisson-Gaussian sensor noise\n• Physically continuous gradient decay",
             transform=axes[0].transAxes, fontsize=8.5, ha="center", va="top", color="#1E293B",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#ECFDF5", edgecolor="#A7F3D0", linewidth=1.5))

# Fake Face
axes[1].imshow(fake_img)
axes[1].set_title("AI-GENERATED SYNTHETIC FACE\n(StyleGAN Latent Synthesis)", fontsize=11, fontweight="bold", color="#991B1B", pad=10)
axes[1].axis("off")
for spine in axes[1].spines.values():
    spine.set_edgecolor("#991B1B")
    spine.set_linewidth(3)
axes[1].text(0.5, -0.08, "• Latent deconvolution upsampling grids\n• Overly planar skin + edge phase ringing\n• Fragile high-frequency spectrum",
             transform=axes[1].transAxes, fontsize=8.5, ha="center", va="top", color="#1E293B",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF2F2", edgecolor="#FECACA", linewidth=1.5))

plt.tight_layout()
out_path = OUT_DIR / "slide1_real_vs_fake_diagram.png"
fig.savefig(out_path, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved Slide 1 Real vs Fake comparison diagram to: {out_path}")
