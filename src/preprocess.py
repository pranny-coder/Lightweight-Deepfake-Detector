import os
import cv2
import numpy as np
from PIL import Image

# Initialize OpenCV Haar Cascade Face Detectors
HAAR_DIR = cv2.data.haarcascades
FRONTAL_CASCADE_PATH = os.path.join(HAAR_DIR, "haarcascade_frontalface_default.xml")
PROFILE_CASCADE_PATH = os.path.join(HAAR_DIR, "haarcascade_profileface.xml")

frontal_face_cascade = cv2.CascadeClassifier(FRONTAL_CASCADE_PATH)
profile_face_cascade = cv2.CascadeClassifier(PROFILE_CASCADE_PATH)

def detect_and_crop_face(img_rgb, margin_ratio=0.10):
    """
    Detects the main face in an RGB image, crops with margin, and resizes to 256x256.
    Returns:
        cropped_rgb: (256, 256, 3) uint8
        cropped_gray: (256, 256) uint8
        detection_status: 'primary', 'fallback_profile', or 'fallback_center'
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Primary detector: Frontal Face Haar Cascade
    faces = frontal_face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    status = "primary"
    bbox = None

    if len(faces) > 0:
        # Choose the largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        bbox = faces[0]
    else:
        # 2. Fallback detector: Profile Face
        profile_faces = profile_face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
        )
        if len(profile_faces) > 0:
            profile_faces = sorted(profile_faces, key=lambda f: f[2] * f[3], reverse=True)
            bbox = profile_faces[0]
            status = "fallback_profile"
        else:
            # 3. Fallback: Center crop (FFHQ / StyleGAN images are naturally centered faces)
            # Take central 80% square
            side = int(min(h, w) * 0.85)
            x = (w - side) // 2
            y = (h - side) // 2
            bbox = (x, y, side, side)
            status = "fallback_center"

    x, y, fw, fh = bbox

    # Add margin
    mx = int(fw * margin_ratio)
    my = int(fh * margin_ratio)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + fw + mx)
    y2 = min(h, y + fh + my)

    cropped = img_rgb[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = img_rgb

    # Resize to 256x256
    cropped_rgb_256 = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)
    cropped_gray_256 = cv2.cvtColor(cropped_rgb_256, cv2.COLOR_RGB2GRAY)

    return cropped_rgb_256, cropped_gray_256, status

def load_and_preprocess(image_path):
    """
    Loads image from disk, applies preprocessing, and returns preprocessed arrays + status.
    """
    pil_img = Image.open(image_path).convert("RGB")
    img_rgb = np.array(pil_img, dtype=np.uint8)
    return detect_and_crop_face(img_rgb)
