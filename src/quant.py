import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot

def quantize_fp8(model_id: str, output_dir: str):
    print(f"--> Target Model: {model_id}")
    print(f"--> Output Directory: {output_dir}")

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
        ignore=["lm_head"],  # Keep output head unquantized for accuracy
    )

    print("Applying FP8 quantization...")
    oneshot(
        model=model,
        recipe=recipe,
        output_dir=output_dir,
    )

    tokenizer.save_pretrained(output_dir)
    print(f"\n[Done] Quantized model and tokenizer saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model to FP8 Dynamic using llmcompressor")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Hugging Face repo ID or local path to unquantized model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_model_fp8",
        help="Path where quantized model will be saved",
    )
    args = parser.parse_args()

    quantize_fp8(args.model_id, args.output_dir)
