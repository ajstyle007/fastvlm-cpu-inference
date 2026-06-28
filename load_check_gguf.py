# ── Quick load test before GGUF conversion ────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./qwen2_llm_only")
print(f"Vocab size : {tokenizer.vocab_size}")
print(f"BOS token  : {tokenizer.bos_token} ({tokenizer.bos_token_id})")
print(f"EOS token  : {tokenizer.eos_token} ({tokenizer.eos_token_id})")

print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    "./qwen2_llm_only",
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)
print(f"Parameters : {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Test generation
inputs = tokenizer("The capital of France is", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(f"Test output: {tokenizer.decode(out[0], skip_special_tokens=True)}")
print("Model verified ✅")