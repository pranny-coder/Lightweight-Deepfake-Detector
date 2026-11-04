import os
import io
import cv2
import numpy as np
from scipy import stats
from PIL import Image

def compute_lbp_image(gray):
    """Basic 8-neighbor Local Binary Pattern map."""
    padded = np.pad(gray, 1, mode="edge").astype(np.int32)
    center = gray.astype(np.int32)
    lbp = np.zeros_like(gray, dtype=np.uint8)
    shifts = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    for bit, (dy, dx) in enumerate(shifts):
        neighbor = padded[1+dy:1+dy+gray.shape[0], 1+dx:1+dx+gray.shape[1]]
        lbp += ((neighbor >= center) << bit).astype(np.uint8)
    return lbp

def compute_entropy(data, bins=32):
    """Computes Shannon entropy of an array."""
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def compute_block_a_edge_features(gray_img):
    """
    Feature Block A - Edge / Gradient Statistics (10 features)
    Includes Sobel, Scharr, Canny, and orientation metrics.
    """
    img_float = gray_img.astype(np.float64)

    # Sobel Gradients
    gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_dir = np.arctan2(gy, gx)

    mean_grad = float(np.mean(grad_mag))
    std_grad = float(np.std(grad_mag))
    grad_kurtosis = float(stats.kurtosis(grad_mag.ravel()))

    # Scharr Gradients
    sx = cv2.Scharr(img_float, cv2.CV_64F, 1, 0)
    sy = cv2.Scharr(img_float, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(sx**2 + sy**2)
    scharr_grad_mean = float(np.mean(scharr_mag))
    scharr_grad_std = float(np.std(scharr_mag))

    # Canny Edge Detection & Otsu
    high_thresh, _ = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(gray_img, int(low_thresh), int(high_thresh))
    edge_mask = edges > 0
    num_edges = int(np.sum(edge_mask))
    edge_density = float(num_edges / gray_img.size)

    if num_edges > 0:
        edge_strength_var = float(np.var(grad_mag[edge_mask]))
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        neighbor_count = cv2.filter2D(edges.astype(np.uint8) // 255, -1, kernel)
        edge_neighbors = neighbor_count[edge_mask]
        edge_continuity = float(np.mean(edge_neighbors >= 2))
    else:
        edge_strength_var = 0.0
        edge_continuity = 0.0

    mean_grad_dir = float(np.mean(np.abs(grad_dir)))
    otsu_threshold = float(high_thresh)

    return {
        "edge_density": edge_density,
        "mean_grad_mag": mean_grad,
        "std_grad_mag": std_grad,
        "grad_kurtosis": grad_kurtosis,
        "scharr_grad_mean": scharr_grad_mean,
        "scharr_grad_std": scharr_grad_std,
        "edge_strength_var": edge_strength_var,
        "edge_continuity": edge_continuity,
        "mean_grad_dir": mean_grad_dir,
        "otsu_threshold": otsu_threshold
    }

def compute_block_b_texture_features(gray_img):
    """
    Feature Block B - Local Texture & Residual Statistics (12 features)
    Includes local variance, Gaussian residual, LBP, and autocorrelation.
    """
    img_float = gray_img.astype(np.float64) / 255.0

    # Local texture stats (5x5 window)
    mean_local = cv2.blur(img_float, (5, 5))
    sq_mean_local = cv2.blur(img_float**2, (5, 5))
    local_var = np.maximum(sq_mean_local - mean_local**2, 0)
    local_std = np.sqrt(local_var)

    local_tex_mean = float(np.mean(local_std))
    local_tex_var = float(np.var(local_std))
    local_tex_std = float(np.std(local_std))

    # Gaussian residual: R = I - G(I)
    blurred = cv2.GaussianBlur(img_float, (5, 5), sigmaX=1.0)
    residual_gauss = img_float - blurred
    residual_energy = float(np.mean(residual_gauss**2))
    residual_std = float(np.std(residual_gauss))
    high_pass_residual_var = float(np.var(residual_gauss))

    # Residual lag-1 autocorrelation
    h_corr = float(np.mean(residual_gauss[:, :-1] * residual_gauss[:, 1:]) / (residual_std**2 + 1e-8))
    v_corr = float(np.mean(residual_gauss[:-1, :] * residual_gauss[1:, :]) / (residual_std**2 + 1e-8))
    residual_lag1_corr = float(0.5 * (h_corr + v_corr))

    # Laplacian high-frequency residual
    laplacian = cv2.Laplacian(img_float, cv2.CV_64F, ksize=3)
    laplacian_energy = float(np.mean(laplacian**2))
    laplacian_kurtosis = float(stats.kurtosis(laplacian.ravel()))

    # Local Binary Patterns (LBP)
    lbp_map = compute_lbp_image(gray_img)
    lbp_mean = float(np.mean(lbp_map))
    lbp_std = float(np.std(lbp_map))
    lbp_entropy = compute_entropy(lbp_map.ravel(), bins=64)

    return {
        "local_tex_mean": local_tex_mean,
        "local_tex_var": local_tex_var,
        "local_tex_std": local_tex_std,
        "residual_energy": residual_energy,
        "residual_std": residual_std,
        "high_pass_residual_var": high_pass_residual_var,
        "residual_lag1_corr": residual_lag1_corr,
        "laplacian_energy": laplacian_energy,
        "laplacian_kurtosis": laplacian_kurtosis,
        "lbp_mean": lbp_mean,
        "lbp_std": lbp_std,
        "lbp_entropy": lbp_entropy
    }

def compute_block_c_frequency_features(gray_img):
    """
    Feature Block C - Frequency-Domain FFT Statistics (8 features)
    Includes radial energy bands, ratio, peak score, slope, and centroid.
    """
    img_float = gray_img.astype(np.float64) / 255.0
    h, w = img_float.shape

    F = np.fft.fftshift(np.fft.fft2(img_float))
    mag = np.abs(F)
    power = mag**2

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_r = np.sqrt(cx**2 + cy**2)

    r_low = 0.15 * max_r
    r_mid = 0.50 * max_r

    low_mask = dist < r_low
    mid_mask = (dist >= r_low) & (dist < r_mid)
    high_mask = dist >= r_mid

    total_power = np.sum(power) + 1e-12
    low_freq_energy = float(np.sum(power[low_mask]) / total_power)
    mid_freq_energy = float(np.sum(power[mid_mask]) / total_power)
    high_freq_energy = float(np.sum(power[high_mask]) / total_power)
    high_freq_ratio = float(high_freq_energy / (low_freq_energy + mid_freq_energy + high_freq_energy + 1e-12))

    high_vals = power[high_mask]
    spectral_peak_score = float(np.max(high_vals) / (np.mean(high_vals) + 1e-12)) if len(high_vals) > 0 else 0.0

    # Spectral centroid (radial center of mass)
    spectral_centroid = float(np.sum(dist * power) / (np.sum(power) * max_r + 1e-12))

    # Radial profile and spectral slope (log power vs log freq)
    radii = np.arange(1, int(max_r))
    radial_profile = []
    for r in radii:
        ring = (dist >= r - 0.5) & (dist < r + 0.5)
        if np.any(ring):
            radial_profile.append(np.mean(power[ring]))
        else:
            radial_profile.append(1e-12)
    
    radial_profile = np.maximum(radial_profile, 1e-12)
    log_freq = np.log(radii)
    log_pwr = np.log(radial_profile)
    slope, _, _, _, _ = stats.linregress(log_freq[2:], log_pwr[2:])
    spectral_slope = float(slope)

    spectral_energy_skew = float(stats.skew(radial_profile))

    return {
        "low_freq_energy": low_freq_energy,
        "mid_freq_energy": mid_freq_energy,
        "high_freq_energy": high_freq_energy,
        "high_freq_ratio": high_freq_ratio,
        "spectral_peak_score": spectral_peak_score,
        "spectral_slope": spectral_slope,
        "spectral_centroid": spectral_centroid,
        "spectral_energy_skew": spectral_energy_skew
    }

def compute_block_d_facial_zones(gray_img):
    """
    Feature Block D - Face Region / Landmark-Aligned Zones (24 features)
    Canonical zones on aligned 256x256 facial crop:
    1. Forehead
    2. Left Eye / Eyebrow
    3. Right Eye / Eyebrow
    4. Nose / Central Face
    5. Mouth / Chin
    6. Cheeks
    Computes: edge_density, mean_grad, residual_energy, lbp_entropy per zone.
    """
    zones = {
        "forehead": (slice(20, 72), slice(51, 205)),
        "left_eye": (slice(72, 123), slice(38, 123)),
        "right_eye": (slice(72, 123), slice(133, 218)),
        "nose": (slice(115, 174), slice(90, 166)),
        "mouth": (slice(174, 230), slice(64, 192)),
        "cheeks": [(slice(123, 184), slice(30, 90)), (slice(123, 184), slice(166, 226))]
    }

    zone_features = {}
    for zname, bbox in zones.items():
        if isinstance(bbox, list):
            # Combine regions (e.g. left and right cheek)
            patch1 = gray_img[bbox[0]]
            patch2 = gray_img[bbox[1]]
            patch = np.concatenate([patch1.ravel(), patch2.ravel()])
            patch_2d = patch1 # for spatial filters
        else:
            patch_2d = gray_img[bbox]
            patch = patch_2d.ravel()

        p_float = patch_2d.astype(np.float64) / 255.0

        # 1. Edge density in zone
        edges = cv2.Canny(patch_2d, 50, 150)
        ed = float(np.mean(edges > 0))

        # 2. Mean gradient magnitude in zone
        gx = cv2.Sobel(p_float, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(p_float, cv2.CV_64F, 0, 1, ksize=3)
        gm = float(np.mean(np.sqrt(gx**2 + gy**2)))

        # 3. Residual energy in zone
        blurred = cv2.GaussianBlur(p_float, (5, 5), sigmaX=1.0)
        res_energy = float(np.mean((p_float - blurred)**2))

        # 4. LBP entropy in zone
        lbp_zone = compute_lbp_image(patch_2d)
        lbp_ent = compute_entropy(lbp_zone.ravel(), bins=32)

        zone_features[f"zone_{zname}_edge_density"] = ed
        zone_features[f"zone_{zname}_mean_grad"] = gm
        zone_features[f"zone_{zname}_residual_energy"] = res_energy
        zone_features[f"zone_{zname}_lbp_entropy"] = lbp_ent

    return zone_features

def apply_perturbation(gray_img, pert_type):
    """
    Applies one of the 4 controlled perturbations to a 256x256 uint8 gray image:
    1. 'jpeg70': JPEG compression quality 70
    2. 'resize50': Downsample 50% then upscale back to 256x256
    3. 'blur1': Gaussian blur sigma = 1.0
    4. 'crop80': Mild 80% center crop resized back to 256x256
    """
    h, w = gray_img.shape
    if pert_type == "jpeg70":
        pil_img = Image.fromarray(gray_img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=70)
        buf.seek(0)
        return np.array(Image.open(buf), dtype=np.uint8)

    elif pert_type == "resize50":
        down = cv2.resize(gray_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        return cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)

    elif pert_type == "blur1":
        return cv2.GaussianBlur(gray_img, (5, 5), sigmaX=1.0)

    elif pert_type == "crop80":
        ch, cw = int(h * 0.8), int(w * 0.8)
        sy = (h - ch) // 2
        sx = (w - cw) // 2
        return cv2.resize(gray_img[sy:sy + ch, sx:sx + cw], (w, h), interpolation=cv2.INTER_LINEAR)

    else:
        raise ValueError(f"Unknown perturbation type: {pert_type}")

def extract_static_features(gray_img):
    """Extracts Blocks A, B, C, and D (Total = 10 + 12 + 8 + 24 = 54 static features)."""
    feats = {}
    feats.update(compute_block_a_edge_features(gray_img))
    feats.update(compute_block_b_texture_features(gray_img))
    feats.update(compute_block_c_frequency_features(gray_img))
    feats.update(compute_block_d_facial_zones(gray_img))
    return feats

def compute_block_e_perturbation_features(gray_img, base_features=None):
    """
    Feature Block E - Perturbation Response Statistics: DeltaPhi = Phi(I) - Phi(T(I))
    4 perturbations: JPEG70, Resize50, Blur1, Crop80
    Tracks 10 sensitive static features across edge, texture, frequency, and face zones,
    plus mean and std delta summary for each perturbation.
    Total = 4 * (10 + 1) = 44 or 46 delta features.
    """
    if base_features is None:
        base_features = extract_static_features(gray_img)

    perturbations = ["jpeg70", "resize50", "blur1", "crop80"]
    delta_feats = {}

    tracked_keys = [
        "edge_density",
        "mean_grad_mag",
        "residual_energy",
        "high_freq_ratio",
        "local_tex_mean",
        "spectral_peak_score",
        "lbp_entropy",
        "zone_forehead_mean_grad",
        "zone_left_eye_mean_grad",
        "zone_mouth_residual_energy"
    ]

    for p in perturbations:
        p_img = apply_perturbation(gray_img, p)
        p_feats = extract_static_features(p_img)
        
        deltas = []
        for k in tracked_keys:
            d_val = float(base_features[k] - p_feats[k])
            delta_feats[f"delta_{p}_{k}"] = d_val
            deltas.append(d_val)
        
        # Summary statistics
        delta_feats[f"delta_{p}_mean"] = float(np.mean(deltas))
        delta_feats[f"delta_{p}_std"] = float(np.std(deltas))

    # Total delta features: 4 * 12 = 48 features
    return delta_feats

def extract_all_features(gray_img):
    """
    Extracts all feature blocks (A, B, C, D, E).
    Total features: 54 static + 48 perturbation = 102 dimensions (exact ~100-D representation!).
    """
    static_feats = extract_static_features(gray_img)
    pert_feats = compute_block_e_perturbation_features(gray_img, base_features=static_feats)
    all_feats = {}
    all_feats.update(static_feats)
    all_feats.update(pert_feats)
    return all_feats
