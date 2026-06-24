#!/usr/bin/env python3
"""
quantize_fp8.py — FP8 dynamic quantization of the verifier (Qwen/Qwen3-8B).

Runs in comp_venv. Applies FP8 dynamic quantization to all Linear layers,
leaves lm_head alone, and saves to a NEW directory (never overwriting the
original model). Recipe follows the llm-compressor FP8 example:
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]

Usage (from repo root, with comp_venv active):
    python quantize_fp8.py
    python quantize_fp8.py --model-id Qwen/Qwen3-8B --save-dir ./Qwen3-8B-FP8-Dynamic
"""
import argparse
import json
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def main() -> int:
    parser = argparse.ArgumentParser(description="FP8 dynamic quantization")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B",
                        help="Source model to quantize.")
    parser.add_argument("--save-dir", default="./Qwen3-8B-FP8-Dynamic",
                        help="Output directory for the quantized model.")
    args = parser.parse_args()

    if os.path.abspath(args.model_id) == os.path.abspath(args.save_dir):
        print("ERROR: --save-dir must differ from --model-id "
              "(never overwrite the original).", file=sys.stderr)
        return 1

    print(f"[quant] loading {args.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # FP8 dynamic: static per-channel weights, dynamic per-token activations.
    # No calibration data is needed for this scheme.
    print("[quant] applying FP8_DYNAMIC to Linear layers (lm_head ignored) ...")
    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"],
    )
    oneshot(model=model, recipe=recipe)

    print(f"[quant] saving to {args.save_dir} ...")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    # Sanity check: the saved config must contain a quantization section,
    # otherwise vLLM won't serve it as FP8 (homework verification hint).
    config_path = os.path.join(args.save_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    qcfg = cfg.get("quantization_config")
    if not qcfg:
        print("ERROR: no 'quantization_config' found in saved config.json.",
              file=sys.stderr)
        return 1

    print("[quant] OK — quantization section present:")
    print(json.dumps(qcfg, indent=2)[:1000])
    print(f"\n[quant] Done. Serve with:  vllm serve {args.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
