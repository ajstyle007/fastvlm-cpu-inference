import onnx
from onnx import numpy_helper
import numpy as np

def convert_onnx_fp16_to_fp32(input_path, output_path):
    model = onnx.load(input_path)
    print(f"Loaded model, opset: {model.opset_import[0].version}")

    converted = 0
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT16:
            # Convert fp16 initializer weights to fp32
            arr_fp16 = numpy_helper.to_array(initializer)
            arr_fp32 = arr_fp16.astype(np.float32)
            new_init = numpy_helper.from_array(arr_fp32, name=initializer.name)
            initializer.CopyFrom(new_init)
            converted += 1
    
    print(f"Converted {converted} fp16 weight tensors to fp32")

    # Also fix Cast nodes: fp32->fp16->fp32 chains become no-ops
    # Remove Cast nodes that cast to fp16 and back since weights are now fp32

    nodes_to_remove = []
    cast_outputs = {}  # map: cast_output_name -> cast_input_name (for fp16 casts)

    for node in model.graph.node:
        if node.op_type == "Cast":
            to_type = next(
                (attr.i for attr in node.attribute if attr.name == "to"), None
            )

            if to_type == onnx.TensorProto.FLOAT16:
            
            # This cast goes fp32->fp16, mark for removal
                cast_outputs[node.output[0]] = node.input[0]
                nodes_to_remove.append(node)
            elif to_type == onnx.TensorProto.FLOAT:
                # Check if input was produced by an fp16 cast we're removing
                if node.input[0] in cast_outputs:
                    # fp32->fp16->fp32 round trip — both casts become no-ops
                    cast_outputs[node.output[0]] = cast_outputs[node.input[0]]
                    nodes_to_remove.append(node)

    
    # Rewire edges: replace uses of removed cast outputs with original inputs
    for node in model.graph.node:
        if node in nodes_to_remove:
            continue
        for i, inp in enumerate(node.input):
            if inp in cast_outputs:
                node.input[i] = cast_outputs[inp]

    # Fix graph outputs too
    for out in model.graph.output:
        if out.name in cast_outputs:
            out.name = cast_outputs[out.name]

    for node in nodes_to_remove:
        model.graph.node.remove(node)

    print(f"Removed {len(nodes_to_remove)} redundant Cast nodes")

    # Update all value_info type annotations from fp16 -> fp32
    for value_info in list(model.graph.value_info):
        if value_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            value_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    onnx.save(model, output_path)
    print(f"Saved fp32 model to: {output_path}")

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"File size: {size_mb:.1f} MB")
    return model

# ── Run conversion ─────────────────────────────────────────────────────────────
import os
convert_onnx_fp16_to_fp32(
    "vision_encoder_with_projector.onnx",
    "vision_encoder_fp32.onnx"
)

# ── Verify the converted model loads and runs ──────────────────────────────────
import onnxruntime as ort
import numpy as np

onnx.checker.check_model(onnx.load("vision_encoder_fp32.onnx"))
print("Graph check passed ✅")

sess = ort.InferenceSession(
    "vision_encoder_fp32.onnx",
    providers=["CPUExecutionProvider"]   # CPU only, no CUDA needed
)

dummy_np = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
ort_out  = sess.run(["image_embeddings"], {"pixel_values": dummy_np})[0]

print(f"Output shape : {ort_out.shape}")   # (1, 256, 896)
print(f"Output dtype : {ort_out.dtype}")   # float32
print("ORT inference passed ✅")