import os
import torch
import torch.nn as nn
from PIL import Image
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

# ── Check actual model dtype before doing anything
vision_dtype = next(model.model.vision_tower.parameters()).dtype
proj_dtype   = next(model.model.mm_projector.parameters()).dtype
print(f"Vision tower dtype : {vision_dtype}")   # likely torch.float16
print(f"MM projector dtype : {proj_dtype}")     # likely torch.float16

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
    

# Build a dummy input the same way predict.py does 
# process_images returns a tensor already resized/normalized per image_processor

dummy_pil = Image.new("RGB", (336, 336), color=128)
image_tensor = process_images([dummy_pil], image_processor, model.config)[0]

# predict.py does .half() just before model.generate — we export in float32
# and cast at runtime so the ONNX graph stays portable

dummy_input  = image_tensor.unsqueeze(0).float().cuda()

print("Dummy input shape:", dummy_input.shape)   # should be (1, 3, 336, 336)
print("Dummy input dtype:", dummy_input.dtype)

# Export
encoder = VisionEncoderWrapper(model.model.vision_tower, model.model.mm_projector).cuda().eval()

# Verify output shape before exporting
with torch.no_grad():
    out = encoder(dummy_input)
    print("Encoder output shape:", out.shape)  # e.g. (1, 576, 3584)
    print(f"Output dtype : {out.dtype}")       # float32


torch.onnx.export(
    encoder,
    (dummy_input,),
    "vision_encoder_with_projector.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["pixel_values"],
    output_names=["image_embeddings"],
    dynamic_axes={
        "pixel_values":      {0: "batch_size"},
        "image_embeddings":  {0: "batch_size"},
    },
)
print("Export done.")


