from llava.model.builder import load_pretrained_model
import argparse
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import os

# model_path = "llava-fastvithd_0.5b_stage3"
# model_name = get_model_name_from_path(model_path)

# parser = argparse.ArgumentParser()
# parser.add_argument("--model-path", type=str, default="./llava-v1.5-13b")
# parser.add_argument("--model-base", type=str, default=None)
# parser.add_argument("--image-file", type=str, default=None, help="location of image file")
# parser.add_argument("--prompt", type=str, default="Describe the image.", help="Prompt for VLM.")
# parser.add_argument("--conv-mode", type=str, default="mistral")
# parser.add_argument("--temperature", type=float, default=0.2)
# parser.add_argument("--top_p", type=float, default=None)
# parser.add_argument("--num_beams", type=int, default=1)
# args, unknown = parser.parse_known_args()

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path, args.model_base, model_name, device="cuda"
# )

def get_model():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "llava-fastvithd_0.5b_stage3")

    print("model_path", model_path)

    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,device="cuda")
    
    return model, tokenizer

# print(model)

# for name, module in model.named_modules():
#     print(name, "->", type(module))


# from safetensors import safe_open

# with safe_open("llava-fastvithd_0.5b_stage3/model.safetensors", framework="pt", device="cpu") as f:
#     for k in f.keys():
#         print(k)


# import torch
# from safetensors.torch import save_file
# import os
# from pathlib import Path

# # Paths
# projector_path = "llava.projector"   # tera PyTorch projector file
# output_gguf = "llava-encoder.gguf"  # jo GGUF banana hai

# # Load projector (PyTorch)
# print(f"Loading projector from {projector_path}")
# checkpoint = torch.load(projector_path, map_location="cpu")

# # Make sure all tensors are float32
# for k in checkpoint.keys():
#     checkpoint[k] = checkpoint[k].float()

# # Save as GGUF
# # llama.cpp GGUF writer expects: keys = tensor_name, values = torch.Tensor
# # For simplicity, we just rename file with .gguf extension and save as safetensors
# # (llama.cpp GGUF loader can read basic tensors from safetensors as GGUF)
# from safetensors.torch import save_file
# save_file(checkpoint, output_gguf)

# print(f"GGUF projector saved to {output_gguf}")


# import torch
# # Replace with your actual file path
# state_dict = torch.load("llava.projector", map_location="cpu")
# for key, value in state_dict.items():
#     print(f"{key}: {value.shape}")