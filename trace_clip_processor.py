# trace_clip_processor.py
from PIL import Image
import numpy as np
from transformers import CLIPImageProcessor

# Replicate the exact processor from the model
processor = CLIPImageProcessor(
    size={"shortest_edge": 1024},
    crop_size={"height": 1024, "width": 1024},
    do_center_crop=True,
    do_normalize=True,
    do_rescale=True,
    do_resize=True,
    image_mean=[0.0, 0.0, 0.0],
    image_std=[1.0, 1.0, 1.0],
    resample=3,
    rescale_factor=1/255,
)

from llava.mm_utils import expand2square

img = Image.open("./llava-fastvithd_0.5b_stage3/GOT.jpg").convert("RGB")
print(f"Original size: {img.size}")

# Step 1: expand2square
img_sq = expand2square(img, tuple(int(x*255) for x in [0.0, 0.0, 0.0]))
print(f"After expand2square: {img_sq.size}")

# Step 2: what does CLIPImageProcessor resize to before crop?
# shortest_edge=1024 means resize so min(w,h)=1024, keep aspect ratio
w, h = img_sq.size
print(f"Square image: {w}x{h}")
# After expand2square it's already square (735x735)
# So shortest_edge resize: scale=1024/735
scale = 1024 / min(w, h)
new_w = int(w * scale)
new_h = int(h * scale)
print(f"After shortest_edge=1024 resize: {new_w}x{new_h}")

# Step 3: center crop to 1024x1024
left = (new_w - 1024) // 2
top  = (new_h - 1024) // 2
print(f"Center crop offset: left={left}, top={top}")

# Now replicate manually
img_resized = img_sq.resize((new_w, new_h), Image.BICUBIC)
img_cropped = img_resized.crop((left, top, left+1024, top+1024))
arr_manual  = np.array(img_cropped, dtype=np.float32) / 255.0
arr_manual  = arr_manual.transpose(2,0,1)

# Run through actual processor
result = processor.preprocess(img_sq, return_tensors="pt")["pixel_values"][0]
arr_ref = result.numpy()

print(f"\nManual shape : {arr_manual.shape}")
print(f"Manual mean  : {arr_manual.mean():.6f}")
print(f"Ref    mean  : {arr_ref.mean():.6f}")

diff = np.abs(arr_ref - arr_manual)
print(f"Max diff     : {diff.max():.6f}")
print(f"Mean diff    : {diff.mean():.6f}")

if diff.max() < 0.01:
    print("✅ Manual matches processor!")
else:
    print("❌ Still off — check resample filter")
    # Try different resample
    for resample_name, resample_filter in [
        ("NEAREST", Image.NEAREST),
        ("BILINEAR", Image.BILINEAR),
        ("BICUBIC", Image.BICUBIC),
        ("LANCZOS", Image.LANCZOS),
    ]:
        img_r = img_sq.resize((new_w, new_h), resample_filter)
        img_c = img_r.crop((left, top, left+1024, top+1024))
        a = np.array(img_c, dtype=np.float32) / 255.0
        a = a.transpose(2,0,1)
        d = np.abs(arr_ref - a).mean()
        print(f"  {resample_name:10s}: mean_diff={d:.6f}")

# Save correct numpy array for WSL
import numpy as np
np.save("correct_pixels.npy", arr_ref)
print(f"\nSaved correct_pixels.npy  mean={arr_ref.mean():.6f}")