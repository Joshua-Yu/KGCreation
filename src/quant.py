import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot

model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"  # Replace with your exact model path/ID
output_dir = "./qwen-30b-fp8-dynamic"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Configure FP8 (W8A8 Dynamic)
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],  # Keep output projection layer in uncompressed precision
)

print("Applying FP8 quantization...")
oneshot(
    model=model,
    recipe=recipe,
    output_dir=output_dir,
)

tokenizer.save_pretrained(output_dir)
print(f"Quantized model saved successfully to: {output_dir}")
