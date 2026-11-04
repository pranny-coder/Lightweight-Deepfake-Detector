import sys
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

PPTX_PATH = "DynForensic_GBT_Proposal_v3_dryrun (1).pptx"
prs = pptx.Presentation(PPTX_PATH)

DARK_BLUE = RGBColor(0x0F, 0x17, 0x2A)   # #0f172a
CYAN_ACCENT = RGBColor(0x0E, 0x74, 0x90) # #0e7490
BODY_GRAY = RGBColor(0x33, 0x41, 0x55)   # #334155
MUTED_GRAY = RGBColor(0x64, 0x74, 0x8B)  # #64748b

def format_bullet_point(tf, title, body, size=Pt(13.5), space_after=Pt(10), is_first=False):
    p = tf.paragraphs[0] if is_first else tf.add_paragraph()
    p.space_after = space_after
    p.space_before = Pt(0)
    p.line_spacing = 1.15
    
    # Bullet marker & Bold Title
    r_bullet = p.add_run()
    r_bullet.text = "•  "
    r_bullet.font.name = "Calibri"
    r_bullet.font.size = size
    r_bullet.font.bold = True
    r_bullet.font.color.rgb = CYAN_ACCENT
    
    r_title = p.add_run()
    r_title.text = title + (" — " if title else "")
    r_title.font.name = "Calibri"
    r_title.font.size = size
    r_title.font.bold = True
    r_title.font.color.rgb = DARK_BLUE
    
    r_body = p.add_run()
    r_body.text = body
    r_body.font.name = "Calibri"
    r_body.font.size = size
    r_body.font.bold = False
    r_body.font.color.rgb = BODY_GRAY

# ==============================================================================
# SLIDE 1: Problem and Relevance
# ==============================================================================
slide1 = prs.slides[0]
tb_main = slide1.shapes[2]  # TextBox 5
tf1 = tb_main.text_frame
tf1.clear()
tf1.word_wrap = True

format_bullet_point(tf1, "Proliferation", "Generators (StyleGAN3, Stable Diffusion, Midjourney) produce photorealistic portraits.", is_first=True)
format_bullet_point(tf1, "Security Threat", "Critical risks for biometric spoofing, automated KYC bypass, and digital identity fraud.")
format_bullet_point(tf1, "Generalization Bottleneck", "Detectors trained on one generator suffer sharp accuracy drops on unseen architectures.")
format_bullet_point(tf1, "Compute Dilemma", "Heavy foundation models (CLIP ViT-L, 428M) & reverse diffusion (DIRE, 3–10s) prevent fast CPU deployment.")
format_bullet_point(tf1, "Research Question", "Can compact physical cues, face zones, and perturbation dynamics achieve viable CPU detection?", space_after=Pt(4))

# Slide 1: Forensic Divergence (TextBox 8)
tb_div = slide1.shapes[3]  # TextBox 8
tf1_div = tb_div.text_frame
tf1_div.clear()
tf1_div.word_wrap = True

p_hdr = tf1_div.paragraphs[0]
p_hdr.space_after = Pt(6)
r_h = p_hdr.add_run()
r_h.text = "The Forensic Divergence:"
r_h.font.name = "Calibri"
r_h.font.size = Pt(13)
r_h.font.bold = True
r_h.font.color.rgb = CYAN_ACCENT

format_bullet_point(tf1_div, "Authentic", "Physical optical imaging, continuous sensor shot noise & natural lens falloff.", size=Pt(11.5), space_after=Pt(4))
format_bullet_point(tf1_div, "Synthetic", "Latent synthesis leaving deconvolution artifacts & distinct perturbation shifts.", size=Pt(11.5), space_after=Pt(2))

# ==============================================================================
# SLIDE 2: Existing Work & Research Gap
# ==============================================================================
slide2 = prs.slides[1]

# Column 1: Face Forgery (Shape 3)
tf2_col1 = slide2.shapes[3].text_frame
tf2_col1.clear()
tf2_col1.word_wrap = True
format_bullet_point(tf2_col1, "1. Face Forgery Methods", "FaceForensics++, Face X-Ray, Celeb-DF", size=Pt(13), space_after=Pt(8), is_first=True)
format_bullet_point(tf2_col1, "Mechanism", "Detects manipulation boundaries, seams, and landmark warping.", size=Pt(11.5), space_after=Pt(6))
format_bullet_point(tf2_col1, "Limitation", "Targets spliced faces; entirely misses whole-face synthetic portraits.", size=Pt(11.5), space_after=Pt(4))

# Column 2: General AIGC (Shape 4)
tf2_col2 = slide2.shapes[4].text_frame
tf2_col2.clear()
tf2_col2.word_wrap = True
format_bullet_point(tf2_col2, "2. General AIGC Forensics", "Wang et al., FreqNet, DIRE, AEROBLADE", size=Pt(13), space_after=Pt(8), is_first=True)
format_bullet_point(tf2_col2, "Mechanism", "Global spectral peaks, spatial CNN artifacts, reverse diffusion error.", size=Pt(11.5), space_after=Pt(6))
format_bullet_point(tf2_col2, "Limitation", "Heavy compute (DIRE: 3–10s/img on A100); spatial CNNs overfit generator footprints.", size=Pt(11.5), space_after=Pt(4))

# Column 3: Foundation & Tabular (Shape 5)
tf2_col3 = slide2.shapes[5].text_frame
tf2_col3.clear()
tf2_col3.word_wrap = True
format_bullet_point(tf2_col3, "3. Foundation & Tabular", "CLIP ViT-L/14, Handcrafted LightGBM", size=Pt(13), space_after=Pt(8), is_first=True)
format_bullet_point(tf2_col3, "Mechanism", "Zero-shot foundation embeddings or GBDTs over basic pixel statistics.", size=Pt(11.5), space_after=Pt(6))
format_bullet_point(tf2_col3, "Limitation", "CLIP needs 428M parameters; tabular baselines omit facial geometry & perturbations.", size=Pt(11.5), space_after=Pt(4))

# Bottom Gap Banner (Shape 7)
tf2_gap = slide2.shapes[7].text_frame
tf2_gap.clear()
tf2_gap.word_wrap = True
p_gap = tf2_gap.paragraphs[0]
p_gap.space_after = Pt(0)
r_gb = p_gap.add_run()
r_gb.text = "Identified Research Gap: "
r_gb.font.name = "Calibri"
r_gb.font.size = Pt(12)
r_gb.font.bold = True
r_gb.font.color.rgb = CYAN_ACCENT

r_gt = p_gap.add_run()
r_gt.text = "Face-forgery tools miss whole-face synthesis; AIGC methods require heavy GPUs; tabular methods ignore facial anatomy. A dedicated, lightweight, landmark-aligned perturbation framework remains missing."
r_gt.font.name = "Calibri"
r_gt.font.size = Pt(12)
r_gt.font.bold = False
r_gt.font.color.rgb = BODY_GRAY

# ==============================================================================
# SLIDE 3: Objectives & Framework (Group 17 diagram preserved)
# ==============================================================================
slide3 = prs.slides[2]
tf3_obj = slide3.shapes[2].text_frame  # TextBox 5
tf3_obj.clear()
tf3_obj.word_wrap = True

p_obj_hdr = tf3_obj.paragraphs[0]
p_obj_hdr.space_after = Pt(10)
r_oh = p_obj_hdr.add_run()
r_oh.text = "Core Research Objectives"
r_oh.font.name = "Calibri"
r_oh.font.size = Pt(16)
r_oh.font.bold = True
r_oh.font.color.rgb = DARK_BLUE

format_bullet_point(tf3_obj, "O1 — Cross-Generator Generalization", "Measure transfer from known generators (StyleGAN2) to unseen architectures (Diffusion, Midjourney).", size=Pt(12.5), space_after=Pt(10))
format_bullet_point(tf3_obj, "O2 — Post-Processing Robustness", "Evaluate stability across lossy JPEG (Q=90, 70, 50), resizing (0.75x, 0.5x), and Gaussian blur.", size=Pt(12.5), space_after=Pt(10))
format_bullet_point(tf3_obj, "O3 — Multi-Classifier Ablation", "Benchmark linear baselines (LogReg, Linear SVM) vs tree ensembles (Random Forest, LightGBM).", size=Pt(12.5), space_after=Pt(10))
format_bullet_point(tf3_obj, "O4 — Explainability & Attribution", "Apply SHAP and landmark profiling to identify highest-impact facial zones and perturbation operators.", size=Pt(12.5), space_after=Pt(4))

# ==============================================================================
# SLIDE 4: Proposed Technical Approach (Picture 6, 7, 8 diagrams preserved)
# ==============================================================================
slide4 = prs.slides[3]
tb4_tech = slide4.shapes[3]  # TextBox 5
# Adjust height so it doesn't overlap Picture 8 at the bottom left
tb4_tech.height = Inches(3.3)
tf4 = tb4_tech.text_frame
tf4.clear()
tf4.word_wrap = True

p_tech_hdr = tf4.paragraphs[0]
p_tech_hdr.space_after = Pt(8)
r_th = p_tech_hdr.add_run()
r_th.text = "Forensic Differentiators & Formulation"
r_th.font.name = "Calibri"
r_th.font.size = Pt(15)
r_th.font.bold = True
r_th.font.color.rgb = DARK_BLUE

format_bullet_point(tf4, "A. Static Physical Forensics (~45-D)", "Sobel/Scharr moments, structure tensor coherence, residual variance, 2D FFT radial power decay (α ≈ 2.0).", size=Pt(11.5), space_after=Pt(8))
format_bullet_point(tf4, "B. Landmark Facial Zones (~15-D)", "Canonical landmarks isolate 5 zones (eyes, nose, mouth, cheeks, forehead) to localize regional artifacts.", size=Pt(11.5), space_after=Pt(8))
format_bullet_point(tf4, "C. Perturbation Dynamics (~40-D)", "Differential shifts ΔΦ = Φ(I) − Φ(T(I)) under JPEG Q70, 0.5x resize, blur. Fast CPU execution (<35ms).", size=Pt(11.5), space_after=Pt(4))

# ==============================================================================
# SLIDE 5: Evaluation & Success Criteria (Picture 7 diagram preserved)
# ==============================================================================
slide5 = prs.slides[4]
tb5_eval = slide5.shapes[2]  # TextBox 5
tb5_eval.width = Inches(6.8)
tf5 = tb5_eval.text_frame
tf5.clear()
tf5.word_wrap = True

p_eval_hdr = tf5.paragraphs[0]
p_eval_hdr.space_after = Pt(12)
r_eh = p_eval_hdr.add_run()
r_eh.text = "Rigorous Evaluation Protocol"
r_eh.font.name = "Calibri"
r_eh.font.size = Pt(17)
r_eh.font.bold = True
r_eh.font.color.rgb = DARK_BLUE

format_bullet_point(tf5, "Dataset Curation", "Authentic FFHQ portraits paired with StyleGAN2/3, Latent Diffusion, and Midjourney.", size=Pt(13), space_after=Pt(12))
format_bullet_point(tf5, "Unseen Generator Split", "Train strictly on known generators; test on completely held-out unseen architectures.", size=Pt(13), space_after=Pt(12))
format_bullet_point(tf5, "Robustness Testing", "Quantify degradation curves across JPEG (Q=90, 70, 50), resizing (0.75x, 0.5x), and blur.", size=Pt(13), space_after=Pt(12))
format_bullet_point(tf5, "Systematic Ablation", "Isolate contributions: Edge, Texture, Frequency, Regional Zones, Static, Static + ΔΦ.", size=Pt(13), space_after=Pt(12))
format_bullet_point(tf5, "Classifier Benchmarking", "Evaluate LogReg, Linear SVM, Random Forest, and LightGBM with probability calibration.", size=Pt(13), space_after=Pt(4))

prs.save(PPTX_PATH)
print("Updated presentation saved successfully to:", PPTX_PATH)
