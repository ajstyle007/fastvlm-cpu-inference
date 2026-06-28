# debug_exact_preprocess.py
# Run on Windows with the original model loaded
import torch
import numpy as np
from PIL import Image
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, get_model_name_from_path

disable_torch_init()
model_path = "./llava-fastvithd_0.5b_stage3"
model_name  = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name, device="cuda"
)

# ── What is the image_processor actually? ─────────────────────────────────────
print(f"image_processor class : {type(image_processor).__name__}")
print(f"image_processor module: {type(image_processor).__module__}")
print()

# Print every attribute
for attr in dir(image_processor):
    if not attr.startswith("_"):
        try:
            val = getattr(image_processor, attr)
            if not callable(val):
                print(f"  {attr:35s} = {val}")
        except:
            pass

# ── Process the test image and inspect the tensor ────────────────────────────
image = Image.open("./llava-fastvithd_0.5b_stage3/GOT.jpg").convert("RGB")
print(f"\nOriginal image size: {image.size}")

image_tensor = process_images([image], image_processor, model.config)[0]
print(f"\nProcessed tensor shape : {image_tensor.shape}")
print(f"Processed tensor dtype : {image_tensor.dtype}")
print(f"Min  : {image_tensor.min():.6f}")
print(f"Max  : {image_tensor.max():.6f}")
print(f"Mean : {image_tensor.mean():.6f}")
print(f"Std  : {image_tensor.std():.6f}")

# Channel-wise stats
for c, name in enumerate(["R", "G", "B"]):
    ch = image_tensor[c]
    print(f"  {name}: min={ch.min():.4f} max={ch.max():.4f} mean={ch.mean():.4f}")

# ── Compare with our preprocessing ───────────────────────────────────────────
MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

img = image.copy()
w, h = img.size
size = max(w, h)
padded = Image.new("RGB", (size, size), (0, 0, 0))
padded.paste(img, ((size - w) // 2, (size - h) // 2))
resized = padded.resize((1024, 1024), Image.BICUBIC)
arr = np.array(resized, dtype=np.float32) / 255.0
arr = (arr - MEAN) / STD
arr = arr.transpose(2, 0, 1)

print(f"\nOur preprocessing shape: {arr.shape}")
print(f"Our Min  : {arr.min():.6f}")
print(f"Our Max  : {arr.max():.6f}")
print(f"Our Mean : {arr.mean():.6f}")

# Diff
ref = image_tensor.cpu().float().numpy()
diff = np.abs(ref - arr)
print(f"\nMax  diff vs original : {diff.max():.6f}")
print(f"Mean diff vs original : {diff.mean():.6f}")

if diff.max() < 0.01:
    print("✅ Preprocessing matches!")
else:
    print("❌ Preprocessing mismatch — need to fix")
    # Save the correct tensor for WSL
    np.save("correct_pixels.npy", ref)
    print("Saved correct_pixels.npy")