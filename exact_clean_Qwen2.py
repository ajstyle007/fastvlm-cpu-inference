import os
import json
import shutil
from safetensors.torch import load_file, save_file

src_dir = "./llava-fastvithd_0.5b_stage3"
dst_dir = "./qwen2_llm_only"
os.makedirs(dst_dir, exist_ok=True)

# ── Extract LLM-only weights ───────────────────────────────────────────────────
state_dict = load_file(os.path.join(src_dir, "model.safetensors"))

SKIP_PREFIXES = ("model.vision_tower", "model.mm_projector")
llm_state_dict = {
    k: v for k, v in state_dict.items()
    if not any(k.startswith(p) for p in SKIP_PREFIXES)
}

print(f"Full checkpoint : {len(state_dict)} tensors")
print(f"LLM only        : {len(llm_state_dict)} tensors")
print(f"Dropped         : {len(state_dict) - len(llm_state_dict)} tensors")

save_file(llm_state_dict, os.path.join(dst_dir, "model.safetensors"))
print(f"Saved weights  → {dst_dir}/model.safetensors")

# ── Build clean Qwen2 config — every field sourced directly from config.json ──
qwen2_config = {
    "architectures":           ["Qwen2ForCausalLM"],   # ← only change
    "model_type":              "qwen2",                 # ← only change
    "hidden_act":              "silu",
    "hidden_size":             896,
    "intermediate_size":       4864,
    "max_position_embeddings": 32768,
    "max_window_layers":       24,
    "num_attention_heads":     14,
    "num_hidden_layers":       24,
    "num_key_value_heads":     2,
    "rms_norm_eps":            1e-06,
    "rope_theta":              1000000.0,
    "sliding_window":          32768,
    "tie_word_embeddings":     True,
    "torch_dtype":             "bfloat16",
    "transformers_version":    "4.39.3",
    "use_cache":               True,
    "use_sliding_window":      False,
    "vocab_size":              151936,
    "bos_token_id":            151643,
    "eos_token_id":            151645,
    "attention_dropout":       0.0,
    "initializer_range":       0.02,
}

with open(os.path.join(dst_dir, "config.json"), "w") as f:
    json.dump(qwen2_config, f, indent=2)
print(f"Saved config   → {dst_dir}/config.json")

# ── Copy all tokenizer files ───────────────────────────────────────────────────
for fname in [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
]:
    src = os.path.join(src_dir, fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(dst_dir, fname))
        print(f"Copied         → {fname}")
    else:
        print(f"Not found      → {fname} (skipping)")