import os
import sys
from PIL import Image
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pilot")
REAL_DIR = os.path.join(DATA_DIR, "real")
FAKE_DIR = os.path.join(DATA_DIR, "fake")

TARGET_PER_CLASS = 200

def acquire_pilot_dataset():
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

    existing_real = [f for f in os.listdir(REAL_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    existing_fake = [f for f in os.listdir(FAKE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Existing images - Real: {len(existing_real)}, Fake: {len(existing_fake)}")

    if len(existing_real) >= TARGET_PER_CLASS and len(existing_fake) >= TARGET_PER_CLASS:
        print("Target dataset already acquired: 200 real and 200 fake face images.")
        return

    print("Connecting to Hugging Face dataset: TheKernel01/140k-Real-and-Fake-Faces (streaming mode)...")
    dataset = load_dataset("TheKernel01/140k-Real-and-Fake-Faces", split="train", streaming=True)

    real_count = len(existing_real)
    fake_count = len(existing_fake)

    # Label 0: 'real' (FFHQ), Label 1: 'fake' (StyleGAN)
    for idx, item in enumerate(dataset):
        if real_count >= TARGET_PER_CLASS and fake_count >= TARGET_PER_CLASS:
            break

        img = item.get("image")
        label = item.get("label")

        is_real = None
        if isinstance(label, int):
            is_real = (label == 0)
        elif isinstance(label, str):
            is_real = (label.lower() == "real")

        if is_real is None:
            continue

        if is_real and real_count < TARGET_PER_CLASS:
            save_path = os.path.join(REAL_DIR, f"real_{real_count:04d}.png")
            if isinstance(img, Image.Image):
                img.convert("RGB").save(save_path)
            else:
                Image.fromarray(img).convert("RGB").save(save_path)
            real_count += 1
        elif (not is_real) and fake_count < TARGET_PER_CLASS:
            save_path = os.path.join(FAKE_DIR, f"fake_{fake_count:04d}.png")
            if isinstance(img, Image.Image):
                img.convert("RGB").save(save_path)
            else:
                Image.fromarray(img).convert("RGB").save(save_path)
            fake_count += 1

        if (real_count + fake_count) % 40 == 0:
            print(f"Acquired: {real_count}/{TARGET_PER_CLASS} real, {fake_count}/{TARGET_PER_CLASS} fake")

    print(f"Dataset acquisition complete: {real_count} real faces, {fake_count} fake faces. Total: {real_count + fake_count} images.")

if __name__ == "__main__":
    acquire_pilot_dataset()
