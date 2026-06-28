# FastVLM CPU Inference — Model Conversion Pipeline

Complete pipeline for converting FastVLM from PyTorch multimodal checkpoint to optimized ONNX + GGUF format for CPU-only inference.

<img width="1500" height="900" alt="Screenshot 2026-06-15 012608" src="https://github.com/user-attachments/assets/79721530-61db-4fda-b8f8-495d8e9aa149" />

## Overview

This repository contains the **model conversion scripts** to transform the FastVLM multimodal model into production-ready formats:

For live running models you can check below links:

- For 512 resolution: https://musk12-fastvlm-cpu-inference-demo.hf.space
- For 1024 resolution: https://musk12-fastvlm-fast-inference-on-cpu.hf.space

- **Vision Encoder**: PyTorch → ONNX (1024 & 512 resolution variants)
- **Language Model**: HuggingFace safetensors → GGUF Q4_K_M quantization
- **C Inference Server**: llama.cpp integration for persistent LLM serving

**Result**: ~1.4 second TTFT on CPU (from 1–1.5 minutes in initial naive approach).

## Repository Structure

```
fastvlm-cpu-inference/
├── vision_proj_to_onnx32.py          # Export vision encoder to ONNX
├── convert_onnx_float32.py           # Convert ONNX fp16→fp32 + merge external data
├── exact_clean_Qwen2.py              # Extract LLM-only weights
├── convert_hf_to_gguf.py             # HuggingFace→GGUF conversion (llama.cpp)
├── fastvlm_infer_v3.c                # Persistent LLM server (C source)
├── llama.cpp/                        # llama.cpp submodule (included)
└── README.md                         # This file
```

## Prerequisites

### System Requirements
- **GPU** (CUDA 11.x or higher) for model conversion
- **CPU** (for inference — 8GB RAM minimum)
- **Disk**: ~50GB free (for models + intermediate files)
- **OS**: Linux recommended (tested on Ubuntu 22.04); Windows/WSL2 also supported

### Python Environment
```bash
python 3.10+
```

### Python Dependencies
```bash
pip install torch torchvision transformers pillow safetensors onnx onnxruntime numpy
```

### Build Tools (for C binary)
```bash
gcc g++ cmake make
```

## Quick Start: Full Conversion Pipeline

### Step 1: Export Vision Encoder (PyTorch → ONNX)

```bash
# Requires GPU + the FastVLM multimodal checkpoint: llava-fastvithd_0.5b_stage3/

python vision_proj_to_onnx32.py
```

**Output**:
- `vision_encoder_with_projector.onnx` (fp16 with external data)
- `vision_encoder_with_projector.onnx.data` (weight tensors)

### Step 2: Merge ONNX External Data (Make Standalone File)

```bash
python - <<'PY'
import onnx
model = onnx.load("vision_encoder_with_projector.onnx")
onnx.save_model(model, "vision_projector_v1.onnx", save_as_external_data=False)
PY
```

**Output**: `vision_projector_v1.onnx` (single standalone file)

### Step 3: Convert FP16→FP32 (Optional, for Portability)

```bash
python convert_onnx_float32.py
```

This replaces fp16 initializers and Cast nodes with fp32, making the model portable across more runtime environments.

**Output**: `vision_encoder_fp32.onnx`

### Step 4: Extract LLM-Only Weights

```bash
# Requires: llava-fastvithd_0.5b_stage3/model.safetensors

python exact_clean_Qwen2.py
```

Creates a clean Qwen2-only checkpoint:
- `qwen2_llm_only/model.safetensors` (LLM weights only)
- `qwen2_llm_only/config.json` (Qwen2 config)
- `qwen2_llm_only/` tokenizer files (copied from source)

### Step 5: Convert to GGUF

```bash
# Requires: gguf-py installed (see llama.cpp/README.md)

python convert_hf_to_gguf.py qwen2_llm_only \
    --outfile fastvlm_qwen2_f16.gguf \
    --outtype f16
```

**Output**: `fastvlm_qwen2_f16.gguf` (high-precision GGUF, ~1.7 GB)

### Step 6: Quantize to Q4_K_M

```bash
# Build llama-quantize tool first:
cd llama.cpp/tools/quantize && cmake . && make

# Then quantize:
./llama.cpp/tools/quantize/llama-quantize \
    fastvlm_qwen2_f16.gguf \
    fastvlm_qwen2_q4km.gguf \
    Q4_K_M 8
```

**Output**: `fastvlm_qwen2_q4km.gguf` (quantized, ~463 MB)

### Step 7: Build Inference Server (C Binary)

```bash
# Assuming you've built llama.cpp libraries:
gcc fastvlm_infer_v3.c \
    -o fastvlm_server \
    -I./llama.cpp/include \
    -I./llama.cpp/ggml/include \
    -L/path/to/llama.cpp/build/src \
    -lllama -lggml -lggml-base \
    -lstdc++ -lm \
    -Wl,-rpath,"/path/to/llama.cpp/build/src"
```

**Output**: `fastvlm_server` (executable)

## File Descriptions

### Vision Encoder Conversion

| File | Purpose |
|------|---------|
| `vision_proj_to_onnx32.py` | Exports `model.vision_tower` + `model.mm_projector` from PyTorch → ONNX with float32 input, float32 output |
| `convert_onnx_float32.py` | Converts fp16 initializers→fp32, removes redundant Cast nodes, merges external data |

**Key points**:
- ONNX input: `pixel_values` shape `(batch, 3, 512/1024, 512/1024)`
- ONNX output: `image_embeddings` shape `(batch, 256, 896)` (image tokens)
- Both 512 and 1024 resolution variants supported

### Language Model Conversion

| File | Purpose |
|------|---------|
| `exact_clean_Qwen2.py` | Extracts LLM-only state dict, removes `model.vision_tower` & `model.mm_projector`, writes HuggingFace-compatible folder |
| `convert_hf_to_gguf.py` | Converts HuggingFace safetensors→GGUF (supports quantization) |

**Key points**:
- LLM: Qwen2-0.5B (24 layers, hidden_size=896, vocab=151936)
- Quantization: Q4_K_M provides ~4.9 bits/weight, excellent speed/quality tradeoff
- Output is llama.cpp compatible

### Inference Server

| File | Purpose |
|------|---------|
| `fastvlm_infer_v3.c` | Persistent C server: loads GGUF once, accepts embeddings+prompt via stdin, streams response via stdout |

**Key features**:
- Loads model once at startup
- Supports concurrent requests via sequence slots (KV cache management)
- Streams tokens as they're generated
- Signals completion with `---END---` sentinel

## Resolution Variants

### 1024 Resolution (Higher Quality)
- ONNX: `vision_encoder_fp32.onnx` (1024×1024 input)
- Embedding shape: (256, 896)
- Vision inference: ~4079–5071ms
- Use for detailed analysis tasks

### 512 Resolution (Faster)
- ONNX: `vision_projector_v1_standalone.onnx` (512×512 input)
- Embedding shape: (64, 896) — smaller embedding
- Vision inference: ~1290 ms to 1430ms
- Use for speed-critical applications

**Trade-off**: 512 resolution is 3–5× faster with minor quality loss on fine-grained details.

## Expected Outputs

After full pipeline:

```
fastvlm_qwen2_q4km.gguf          (463 MB)  ← LLM
vision_encoder_fp32.onnx         (~200 MB) ← Vision (1024 res)
vision_projector_v1_standalone.onnx (~50 MB)  ← Vision (512 res)
fastvlm_server                   (executable) ← C inference binary
```

## Performance Metrics

Measured on Intel CPU (6 cores):

| Component | 1024 Res | 512 Res |
|-----------|----------|---------|
| Vision preprocess | 10–20ms | 10–15ms |
| ONNX inference | 4079–5071ms | 1290-1430ms |
| LLM first token | 300–500ms | 300–500ms |
| **Total TTFT** | **5450-9637ms** | **1450-1640ms** |

TTFT = Time To First Token (includes image encoding, LLM generation start)

## Usage Examples

### Single-Shot Inference

```bash
python -c "
from fastvlm_inference import run
run('path/to/image.jpg', 'Describe this image in detail.')
"
```

### Persistent Server

```bash
# Terminal 1: Start API server
python final_working_model_files_new/stream_api.py

# Terminal 2: Use Gradio UI
python fast_inference_files/gradio_app.py
```

Then visit `http://localhost:7860`

## Troubleshooting

### ONNX External Data Issues
If you see `.onnx.data` files but need a single file:
```bash
python - <<'PY'
import onnx
model = onnx.load("model.onnx")
onnx.save_model(model, "model_standalone.onnx", save_as_external_data=False)
PY
```

### GGUF Conversion Fails
- Ensure safetensors folder has all required files: `model.safetensors`, `config.json`, `tokenizer.json`
- Check `convert_hf_to_gguf.py` has `NO_LOCAL_GGUF` not set
- Verify llama.cpp is built: `ls llama.cpp/build/src/*.so`

### C Binary Build Fails
- Verify llama.cpp is compiled: `cd llama.cpp && cmake . && make`
- Check include paths: `-I./llama.cpp/include -I./llama.cpp/ggml/include`
- Set LD_LIBRARY_PATH: `export LD_LIBRARY_PATH=/path/to/llama.cpp/build/src:$LD_LIBRARY_PATH`

## References

- **FastVLM**: https://huggingface.co/apple/FastVLM-0.5B
- **llama.cpp**: https://github.com/ggml-org/llama.cpp
- **ONNX**: https://onnx.ai/
- **GGUF Format**: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## Citation

If you use this pipeline in research or production:

```bibtex
@software{fastvlm_cpu_inference,
  title = {FastVLM CPU Inference Pipeline},
  author = {Ajay Kumar},
  url = {https://github.com/ajstyle007/fastvlm-cpu-inference},
  year = {2026}
}
```

## License

MIT License — See LICENSE file for details

---

**Need help?** Check the individual script docstrings or open an issue on GitHub.
