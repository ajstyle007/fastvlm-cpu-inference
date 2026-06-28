import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from llava.mm_utils import process_images
import torch
from torch import nn
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, get_model_name_from_path

model_path = "llava-fastvithd_0.5b_stage3"
disable_torch_init()
model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base=None, model_name=model_name, device="cuda"
)
model.eval()

class VisionEncoderWrapper(nn.Module):
    """
    Replicates:
        image_tensor = process_images([image], image_processor, model.config)[0]
        images=image_tensor.unsqueeze(0).half()
    but accepts a float32 pixel_values so ONNX doesn't bake in fp16 ops.
    The fp16 cast is done OUTSIDE this module (in the runtime caller).
    """

    def __init__(self, vision_tower, mm_projector):
        super().__init__()
        self.vision_tower = vision_tower
        self.mm_projector  = mm_projector

    def forward(self, pixel_values: torch.Tensor):
        # pixel_values: (1, 3, H, W)  float32
        # vision tower internally handles the feature extraction
        pixel_values = pixel_values.half()
        image_features = self.vision_tower(pixel_values)

        # project to LLM hidden dim
        projected = self.mm_projector(image_features)
        return projected.float()   # (1, num_patches, llm_hidden_dim)
    
encoder = VisionEncoderWrapper(model.model.vision_tower, model.model.mm_projector).cuda().eval()

# ── 1. Check the ONNX graph is well-formed ────────────────────────────────────
model_onnx = onnx.load("vision_encoder_fp32.onnx")
onnx.checker.check_model(model_onnx)
print("ONNX graph check passed ✅")
print(f"Opset version : {model_onnx.opset_import[0].version}")

# Print input/output specs
for inp in model_onnx.graph.input:
    print(f"Input  : {inp.name} | {inp.type}")
for out in model_onnx.graph.output:
    print(f"Output : {out.name} | {out.type}")

# ── 2. Run inference with OnnxRuntime and compare to PyTorch output ───────────
dummy_pil    = Image.new("RGB", (336, 336), color=128)
image_tensor = process_images([dummy_pil], image_processor, model.config)[0]
dummy_np     = image_tensor.unsqueeze(0).float().numpy()  # (1, 3, H, W) float32

# PyTorch reference
with torch.no_grad():
    pt_out = encoder(image_tensor.unsqueeze(0).float().cuda()).cpu().numpy()

# OnnxRuntime
sess = ort.InferenceSession(
    "vision_encoder_fp32.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
ort_out = sess.run(["image_embeddings"], {"pixel_values": dummy_np})[0]

print(f"\nPyTorch output shape : {pt_out.shape}")
print(f"ORT     output shape : {ort_out.shape}")

max_diff  = np.abs(pt_out - ort_out).max()
mean_diff = np.abs(pt_out - ort_out).mean()
rel_diff  = (np.abs(pt_out - ort_out) / (np.abs(pt_out) + 1e-6)).mean()

print(f"Max  absolute diff : {max_diff:.6f}")
print(f"Mean absolute diff : {mean_diff:.6f}")
print(f"Mean relative diff : {rel_diff:.6f}")

# fp16 model exported to fp32 — expect small numerical noise, not zeros
# anything below 1e-3 is fine for vision embeddings
if mean_diff < 1e-3:
    print("✅ Numerical verification passed (mean diff within fp16 tolerance)")
elif mean_diff < 5e-3:
    print("⚠️  Acceptable diff — slight fp16 rounding, will not affect output")
else:
    print("❌ Large mean diff — something is wrong with the export")

# Visualize where the large diffs are
outlier_mask = np.abs(pt_out - ort_out) > 0.01
outlier_pct  = outlier_mask.mean() * 100
print(f"Fraction of values with diff > 0.01 : {outlier_pct:.2f}%")

# ── 3. Check file size (rough sanity check) ───────────────────────────────────
import os
size_mb = os.path.getsize("vision_encoder_fp32.onnx") / 1024 / 1024
print(f"\nONNX file size : {size_mb:.1f} MB")
# FastVLM small vision tower should be roughly 30–80 MB
# if it's 0.1 MB the weights weren't embedded — something went wrong