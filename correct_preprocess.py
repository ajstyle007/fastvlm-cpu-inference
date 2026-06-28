# final_preprocess.py
import numpy as np
from PIL import Image

def expand2square(image, background_color=(0, 0, 0)):
    w, h = image.size
    if w == h:
        return image
    size = max(w, h)
    result = Image.new("RGB", (size, size), background_color)
    result.paste(image, ((size - w) // 2, (size - h) // 2))
    return result

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Exact replication of process_images() with image_aspect_ratio='pad':
    1. expand2square with black padding (image_mean=[0,0,0] → color=(0,0,0))
    2. CLIPImageProcessor: resize shortest_edge=1024, center_crop 1024x1024
       rescale /255, mean=0 std=1 (no-op normalization)
    """
    img = Image.open(image_path).convert("RGB")

    # Step 1: pad to square
    img = expand2square(img, (0, 0, 0))

    # Step 2: resize shortest_edge → 1024 using round() like CLIPImageProcessor
    w, h = img.size
    short = min(w, h)
    scale = 1024 / short
    new_w = round(w * scale)
    new_h = round(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # Step 3: center crop to 1024x1024
    w, h = img.size
    left = (w - 1024) // 2
    top  = (h - 1024) // 2
    img  = img.crop((left, top, left + 1024, top + 1024))

    # Step 4: rescale to [0,1] — mean=[0,0,0] std=[1,1,1] = no-op normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 1024, 1024)
    return arr


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 \
                 else "./llava-fastvithd_0.5b_stage3/GOT.jpg"

    arr = preprocess_image(image_path)
    print(f"Shape : {arr.shape}")
    print(f"Range : [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"Mean  : {arr.mean():.6f}")

    try:
        ref = np.load("correct_pixels.npy")
        diff = np.abs(ref - arr[0])
        print(f"Max  diff : {diff.max():.6f}")
        print(f"Mean diff : {diff.mean():.6f}")
        if diff.mean() < 0.005:
            print("✅ Close enough for inference!")
    except FileNotFoundError:
        pass