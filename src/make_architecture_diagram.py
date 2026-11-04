import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT_DIR = Path("/Users/pranavthobhani/CV Project/demo_dryrun_outputs")
OUT_DIR.mkdir(exist_ok=True)

# Create 16:9 widescreen canvas
fig, ax = plt.subplots(figsize=(15.5, 7.8), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Color palette (Modern academic corporate slate / cyan / emerald / crimson)
C_BG = "#F8FAFC"         # ultra light background
C_DARK = "#0F172A"       # dark slate for main titles
C_CYAN = "#0E7490"       # cyan primary accent
C_BLUE = "#1E40AF"       # deep blue
C_INDIGO = "#4338CA"     # indigo
C_AMBER = "#D97706"      # amber accent
C_EMERALD = "#059669"    # emerald real
C_CRIMSON = "#DC2626"    # crimson fake
C_CARD_BG = "#FFFFFF"    # white card
C_BORDER = "#CBD5E1"     # light border

# Background fill
bg = patches.Rectangle((0, 0), 100, 100, facecolor=C_BG, zorder=0)
ax.add_patch(bg)

def draw_card(x, y, w, h, title, subtitle="", border_color=C_BORDER, header_color=C_CYAN, corner_radius=1.8):
    # Main card body
    card = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={corner_radius}",
                                  facecolor=C_CARD_BG, edgecolor=border_color, linewidth=1.8, zorder=2)
    ax.add_patch(card)
    
    # Header bar
    header_h = 4.2
    header = patches.FancyBboxPatch((x, y + h - header_h), w, header_h,
                                    boxstyle=f"round,pad=0,rounding_size={corner_radius}",
                                    facecolor=header_color, edgecolor=header_color, linewidth=1, zorder=3)
    ax.add_patch(header)
    # Square bottom of header
    header_sq = patches.Rectangle((x, y + h - header_h), w, header_h/2, facecolor=header_color, edgecolor=header_color, zorder=3)
    ax.add_patch(header_sq)
    
    # Header text
    ax.text(x + w/2, y + h - header_h/2, title, color="#FFFFFF", fontsize=10.5, fontweight="bold",
            ha="center", va="center", zorder=4)
    
    if subtitle:
        ax.text(x + w/2, y + h - header_h - 1.8, subtitle, color=C_DARK, fontsize=8.5, fontweight="bold",
                ha="center", va="center", zorder=4)

def draw_arrow_down(x, y_start, y_end, color="#64748B", text=""):
    ax.annotate("", xy=(x, y_end), xytext=(x, y_start),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.2, mutation_scale=16),
                zorder=5)
    if text:
        ax.text(x + 1.2, (y_start + y_end)/2, text, color=color, fontsize=8, fontweight="bold", va="center", zorder=6)

def draw_arrow_right(x_start, x_end, y, color="#64748B", text=""):
    ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.2, mutation_scale=16),
                zorder=5)
    if text:
        ax.text((x_start + x_end)/2, y + 1.2, text, color=color, fontsize=8, fontweight="bold", ha="center", zorder=6)

# ==============================================================================
# 1. INPUT STAGE (Top Left)
# ==============================================================================
draw_card(2, 63, 17, 30, "1. INPUT INGESTION", "Any Raw Face Image", border_color="#94A3B8", header_color="#334155")
input_items = [
    "• Single Face Portrait",
    "• Real or Synthetic",
    "• Any Image Format",
    "  (PNG, JPG, WEBP)",
    "• Varying Resolutions",
    "• Zero GPU Reqs"
]
for i, txt in enumerate(input_items):
    ax.text(3.5, 83.5 - i*3.5, txt, color="#334155", fontsize=8.2, va="center", zorder=4)

draw_arrow_right(19, 23, 78, color=C_CYAN, text="Detect & Crop")

# ==============================================================================
# 2. STANDARDIZATION STAGE (Top Center-Left)
# ==============================================================================
draw_card(23, 63, 20, 30, "2. STANDARDIZATION", "Face Normalization (~4.5ms)", border_color="#38BDF8", header_color=C_CYAN)
std_items = [
    "• OpenCV Haar Cascade",
    "  - Frontal (96.0% hits)",
    "  - Profile fallback (0.8%)",
    "  - Center fallback (3.2%)",
    "• +10% Bounding Margin",
    "• Standardized 256×256 px",
    "• Normalized Float Grayscale"
]
for i, txt in enumerate(std_items):
    fontw = "bold" if "256×256" in txt else "normal"
    ax.text(24.5, 83.5 - i*3.3, txt, color="#1E293B", fontsize=8, fontweight=fontw, va="center", zorder=4)

draw_arrow_right(43, 47, 78, color=C_BLUE, text="Feature Flow")

# ==============================================================================
# 3. MULTI-BRANCH FEATURE EXTRACTION (Right Column - 3 Branches)
# ==============================================================================
# Banner over the 3 branches
feature_banner = patches.FancyBboxPatch((47, 63), 51, 30, boxstyle="round,pad=0,rounding_size=1.8",
                                       facecolor="#EFF6FF", edgecolor="#60A5FA", linewidth=1.5, zorder=1)
ax.add_patch(feature_banner)
ax.text(72.5, 91.0, "3. MULTI-BRANCH HANDCRAFTED EXTRACTION (61-D Total, ~30.3 ms)",
        color=C_BLUE, fontsize=10.5, fontweight="bold", ha="center", zorder=4)

# Sub-branch 1: Static Invariants (Left sub-card)
draw_card(48.5, 65, 15, 23.5, "Branch A: Static (25D)", "Optical & Frequency Cues", border_color="#93C5FD", header_color=C_BLUE)
b1_items = [
    "• Sobel Gradients (5D):",
    "  Mean, std, kurt, skew",
    "• Structure Tensor (4D):",
    "  Coherence mean, p90",
    "• Sensor Residual (6D):",
    "  Var, kurt, lag-1 corr",
    "• 2D FFT Radial (10D):",
    "  Slope α≈2, peak score"
]
for i, txt in enumerate(b1_items):
    fontw = "bold" if "•" in txt else "normal"
    ax.text(49.5, 82.5 - i*2.5, txt, color="#1E293B", fontsize=7.2, fontweight=fontw, va="center", zorder=4)

# Sub-branch 2: Landmark Zones (Middle sub-card)
draw_card(65, 65, 15, 23.5, "Branch B: Zones (15D)", "5 Facial Anatomy Zones", border_color="#C084FC", header_color=C_INDIGO)
b2_items = [
    "• 5 Anatomical Zones:",
    "  - Forehead (planar skin)",
    "  - Eyes (iris geometry)",
    "  - Nose (depth gradient)",
    "  - Mouth (teeth/lips)",
    "  - Cheeks (highlights)",
    "• 3 Stats per Zone (15D):",
    "  Grad, residual, kurtosis"
]
for i, txt in enumerate(b2_items):
    fontw = "bold" if "•" in txt else "normal"
    ax.text(66, 82.5 - i*2.5, txt, color="#1E293B", fontsize=7.2, fontweight=fontw, va="center", zorder=4)

# Sub-branch 3: Perturbation Dynamics (Right sub-card)
draw_card(81.5, 65, 15, 23.5, "Branch C: Dynamics (21D)", "Stress Tests ΔΦ = Φ - Φ(T)", border_color="#F472B6", header_color="#BE185D")
b3_items = [
    "• 3 Controlled Tests T:",
    "  - T1: JPEG Q=70",
    "  - T2: 0.5× Downscale",
    "  - T3: Gaussian Blur 1.0",
    "• 7 Deltas per Test (21D):",
    "  Δcoh, Δres_var, Δgrad,",
    "  Δfft_slope, Δpeak_score",
    "• Exposes fragile artifacts"
]
for i, txt in enumerate(b3_items):
    fontw = "bold" if "•" in txt else "normal"
    ax.text(82.5, 82.5 - i*2.5, txt, color="#1E293B", fontsize=7.2, fontweight=fontw, va="center", zorder=4)

# Arrow from Feature Extraction down to Vector Assembly
draw_arrow_down(72.5, 63, 56, color=C_BLUE, text="Concatenate")

# ==============================================================================
# 4. COMPACT 61-D VECTOR & SCALING (Middle)
# ==============================================================================
draw_card(20, 42, 60, 14, "4. COMPACT 61-DIMENSIONAL FORENSIC VECTOR & Z-SCORE SCALING",
          "X = [ S_1...S_25 (Static)  |  Z_1...Z_15 (Facial Zones)  |  P_1...P_21 (Perturbations) ] ∈ ℝ⁶¹",
          border_color="#F59E0B", header_color=C_AMBER)

vec_boxes = [
    (22, 44, 18, 5.5, "Static Cues (25D)\nOptical Coherence & FFT", "#DBEAFE", "#1E40AF"),
    (41.5, 44, 18, 5.5, "Facial Zones (15D)\nRegional Skin & Anatomy", "#F3E8FF", "#6B21A8"),
    (61, 44, 17.5, 5.5, "Perturbation ΔΦ (21D)\nJPEG, Resize, Blur Shifts", "#FCE7F3", "#9D174D")
]
for bx, by, bw, bh, btxt, bfc, btc in vec_boxes:
    box = patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=1",
                                facecolor=bfc, edgecolor=btc, linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(bx + bw/2, by + bh/2, btxt, color=btc, fontsize=7.8, fontweight="bold", ha="center", va="center", zorder=4)

draw_arrow_down(50, 42, 35, color=C_AMBER, text="Normalized Input X_scaled")

# ==============================================================================
# 5. CLASSIFICATION INFERENCE LAYER (Bottom Left & Center)
# ==============================================================================
draw_card(4, 5, 48, 30, "5. MULTI-CLASSIFIER BENCHMARK & INFERENCE (< 0.05 ms on CPU)",
          "Optimal Separating Hyperplane with Probability Calibration", border_color="#34D399", header_color=C_EMERALD)

# Highlight Linear SVM (The Winner)
svm_box = patches.FancyBboxPatch((6, 17), 44, 11, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor="#ECFDF5", edgecolor=C_EMERALD, linewidth=2, zorder=3)
ax.add_patch(svm_box)
ax.text(7.5, 25.5, "★ PRIMARY MODEL: Regularized Linear SVM (Platt Calibrated)", color=C_EMERALD, fontsize=9.2, fontweight="bold", zorder=4)
ax.text(7.5, 22.5, "• Test Accuracy: 68.33%  |  ROC-AUC: 0.707  |  F1-Score: 0.698  |  Latency: 0.03 ms", color=C_DARK, fontsize=8.2, fontweight="bold", zorder=4)
ax.text(7.5, 19.5, "• Objective: Minimize 1/2||W||² + C∑ξ_i   --> Resists overfitting on sub-pixel texture noise", color="#475569", fontsize=7.8, zorder=4)

# Comparison baselines
ax.text(6, 13.5, "Ablation Classifiers Evaluated on Identical 61-D Vector:", color="#334155", fontsize=8.2, fontweight="bold", zorder=4)
ax.text(6, 10.5, "• Logistic Regression: 65.83% Accuracy, 0.709 ROC-AUC, 0.004 ms inference", color="#475569", fontsize=7.8, zorder=4)
ax.text(6, 7.5, "• Random Forest & HistGradientBoosting (GBDT): 59.2%–60.0% (tree over-splitting on noise)", color="#475569", fontsize=7.8, zorder=4)

draw_arrow_right(52, 56, 20, color=C_EMERALD, text="Binary Verdict")

# ==============================================================================
# 6. DECISION & VERDICT OUTPUT (Bottom Right)
# ==============================================================================
draw_card(56, 5, 42, 30, "6. FINAL FORENSIC VERDICT & VISUAL PROOF",
          "Interpretable Output & Heatmap Generation", border_color="#F87171", header_color="#991B1B")

# Real verdict card
real_badge = patches.FancyBboxPatch((58, 20), 18, 8, boxstyle="round,pad=0,rounding_size=1",
                                   facecolor="#ECFDF5", edgecolor=C_EMERALD, linewidth=1.5, zorder=3)
ax.add_patch(real_badge)
ax.text(67, 25.5, "AUTHENTIC REAL FACE", color=C_EMERALD, fontsize=8.5, fontweight="bold", ha="center", zorder=4)
ax.text(67, 22.5, "P(AI) < 0.50 (e.g. 31.2%)\nNatural Lens & Photon Shot Noise", color="#065F46", fontsize=7.2, ha="center", zorder=4)

# Fake verdict card
fake_badge = patches.FancyBboxPatch((78, 20), 18, 8, boxstyle="round,pad=0,rounding_size=1",
                                   facecolor="#FEF2F2", edgecolor=C_CRIMSON, linewidth=1.5, zorder=3)
ax.add_patch(fake_badge)
ax.text(87, 25.5, "AI-GENERATED FACE", color=C_CRIMSON, fontsize=8.5, fontweight="bold", ha="center", zorder=4)
ax.text(87, 22.5, "P(AI) ≥ 0.50 (e.g. 68.8%)\nDeconvolution Grid & Planar Skin", color="#991B1B", fontsize=7.2, ha="center", zorder=4)

# Key Metrics Summary
metrics_box = patches.FancyBboxPatch((58, 7), 38, 10.5, boxstyle="round,pad=0,rounding_size=1",
                                     facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=1.2, zorder=3)
ax.add_patch(metrics_box)
ax.text(59.5, 14.8, "SYSTEM EFFICIENCY & DEPLOYMENT METRICS:", color=C_DARK, fontsize=8, fontweight="bold", zorder=4)
ax.text(59.5, 12.2, "• Total Processing Time: ~35 ms / face on single CPU (Zero GPU needed)", color="#334155", fontsize=7.5, zorder=4)
ax.text(59.5, 9.8, "• Model RAM Footprint: 5.5 KB (250,000× lighter than CLIP ViT-L)", color="#334155", fontsize=7.5, zorder=4)
ax.text(59.5, 7.5, "• Interpretability: 4-Panel Optical Heatmap generated per image", color="#334155", fontsize=7.5, zorder=4)

plt.tight_layout()
out_file = OUT_DIR / "dynforensic_system_architecture_diagram.png"
fig.savefig(out_file, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved publication-grade architecture diagram to: {out_file}")
