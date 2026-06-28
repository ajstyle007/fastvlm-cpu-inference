# fastvlm_final.py
import numpy as np
import onnxruntime as ort
from PIL import Image
import struct, tempfile, os, sys, subprocess

ONNX_PATH = "./vision_encoder_fp32.onnx"
INFER_BIN = "./fastvlm_infer_v2"

def expand2square(image, background_color=(0, 0, 0)):
    w, h = image.size
    if w == h:
        return image
    size = max(w, h)
    result = Image.new("RGB", (size, size), background_color)
    result.paste(image, ((size - w) // 2, (size - h) // 2))
    return result

def preprocess_image(image_path: str) -> np.ndarray:
    img   = Image.open(image_path).convert("RGB")
    img   = expand2square(img, (0, 0, 0))
    w, h  = img.size
    scale = 1024 / min(w, h)
    img   = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h  = img.size
    img   = img.crop(((w-1024)//2, (h-1024)//2,
                       (w-1024)//2+1024, (h-1024)//2+1024))
    arr   = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis]   # (1,3,1024,1024)

def encode_image(image_path: str) -> np.ndarray:
    sess  = ort.InferenceSession(ONNX_PATH,
                providers=["CPUExecutionProvider"])
    pv    = preprocess_image(image_path)
    print(f"  pixel range : [{pv.min():.3f}, {pv.max():.3f}]  mean={pv.mean():.4f}")
    embd  = sess.run(["image_embeddings"], {"pixel_values": pv})[0]
    return embd[0]   # (256, 896)

def run(image_path: str, prompt: str):
    print("Encoding image with ONNX...")
    embd = encode_image(image_path)
    print(f"  embeddings  : {embd.shape}  [{embd.min():.3f}, {embd.max():.3f}]")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    n_tok, n_embd = embd.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", n_tok, n_embd))
        f.write(embd.astype(np.float32).tobytes())

    print("Running LLM...\n" + "="*60)
    try:
        subprocess.run([INFER_BIN,
                        "./fastvlm_qwen2_q4km.gguf",
                        path, prompt])
    finally:
        os.unlink(path)

if __name__ == "__main__":
    image  = sys.argv[1] if len(sys.argv) > 1 \
             else "./llava-fastvithd_0.5b_stage3/GOT.jpg"
    prompt = sys.argv[2] if len(sys.argv) > 2 \
             else "Describe what you see in this image in detail."
    run(image, prompt)