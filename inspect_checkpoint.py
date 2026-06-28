import torch
import json

# Load the state dict and see all key prefixes
# state_dict = torch.load(
#     "./llava-fastvithd_0.5b_stage3/model.safetensors", 
#     map_location="cpu"
# )

# Or if safetensors format:
# from safetensors.torch import load_file
# state_dict = load_file("./llava-fastvithd_0.5b_stage3/model.safetensors")


# # Count params per prefix
# prefixes = sorted(set(".".join(k.split(".")[:2]) for k in state_dict.keys()))
# print("Two-level prefixes:")
# for prefix in prefixes:
#     keys = [k for k in state_dict.keys() if k.startswith(prefix)]
#     params = sum(state_dict[k].numel() for k in keys)
#     print(f"  {prefix:45s} → {len(keys):4d} tensors, {params/1e6:.1f}M params")

from safetensors.torch import load_file

state_dict = load_file("./llava-fastvithd_0.5b_stage3/model.safetensors")

# ── Full breakdown of every sub-component ─────────────────────────────────────
components = {
    "LM Head":        [k for k in state_dict if k.startswith("lm_head")],
    "Embed tokens":   [k for k in state_dict if k.startswith("model.embed_tokens")],
    "LLM layers":     [k for k in state_dict if k.startswith("model.layers")],
    "MM projector":   [k for k in state_dict if k.startswith("model.mm_projector")],
    "Norm":           [k for k in state_dict if k.startswith("model.norm")],
    "Vision tower":   [k for k in state_dict if k.startswith("model.vision_tower")],
}

total_params = sum(v.numel() for v in state_dict.values())
print(f"Total params: {total_params/1e6:.1f}M\n")

for component, keys in components.items():
    params = sum(state_dict[k].numel() for k in keys)
    print(f"{'─'*60}")
    print(f"  {component:20s} | {len(keys):4d} tensors | {params/1e6:.2f}M params")
    print(f"{'─'*60}")
    for k in keys:
        t = state_dict[k]
        print(f"    {k:55s} | {str(list(t.shape)):20s} | {str(t.dtype):15s}")
    print()

# ── MM projector detail (small, print everything) ─────────────────────────────
print("\n>>> MM Projector layers in detail:")
for k in state_dict:
    if k.startswith("model.mm_projector"):
        t = state_dict[k]
        print(f"  {k}")
        print(f"    shape: {list(t.shape)}  dtype: {t.dtype}")
        print(f"    min: {t.float().min():.4f}  max: {t.float().max():.4f}")

# ── Vision tower sub-structure (3 levels deep) ────────────────────────────────
print("\n>>> Vision tower sub-components (3-level):")
vt_prefixes = sorted(set(
    ".".join(k.split(".")[:4])
    for k in state_dict if k.startswith("model.vision_tower")
))
for p in vt_prefixes:
    keys = [k for k in state_dict if k.startswith(p)]
    params = sum(state_dict[k].numel() for k in keys)
    print(f"  {p:65s} → {len(keys):4d} tensors, {params/1e6:.3f}M params")

# ── LLM layers structure (one decoder layer in full) ──────────────────────────
print("\n>>> Layer 0 tensors (representative of all 24 layers):")
for k in state_dict:
    if k.startswith("model.layers.0."):
        t = state_dict[k]
        print(f"  {k:55s} | {str(list(t.shape)):25s} | {t.dtype}")


#OUTPUT

# Key facts confirmed:

# All weights are bfloat16 — important, use --outtype bf16 not f16
# LLM is clean standard Qwen2-0.5B: hidden_size=896, intermediate_size=4864, 24 layers, GQA with num_kv_heads=2 (k/v proj is [128, 896] = 896/128 = 7 heads ratio → 14 q heads, 2 kv heads)
# MM projector: [896, 3072] → [896, 896] meaning vision tower outputs 3072-dim features, projector maps to LLM's 896-dim
# Prefixes to drop are exactly model.vision_tower and model.mm_projector

