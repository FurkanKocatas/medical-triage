#!/usr/bin/env python3
"""Verify all 3 models load + do a zero-shot Turkish triage test."""
import os
import time
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Desktop/medical-triage/.hf_cache"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    ("L1", "HuggingFaceTB/SmolLM2-360M"),
    ("L2", "Qwen/Qwen2.5-1.5B"),
    ("L3", "google/gemma-4-E2B"),
]

TEST_PROMPT = (
    "Hasta semptomlari: Ani siddetli gogus agrisi ve nefes darligi, "
    "sol kolda uyusma. Triaj seviyesi nedir?\nCevap:"
)


def main():
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}\n")

    for tag, mid in MODELS:
        print(f"=== {tag} : {mid} ===")
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(
                mid, dtype=torch.bfloat16, device_map="cuda",
            )
            t_load = time.time() - t0
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            mem_mb = torch.cuda.memory_allocated() / 1e6

            inputs = tok(TEST_PROMPT, return_tensors="pt").to("cuda")
            t1 = time.time()
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            t_gen = time.time() - t1
            txt = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            print(f"  params={n_params:.1f}M load={t_load:.2f}s gen={t_gen:.2f}s mem={mem_mb:.0f}MB")
            print(f"  out: {txt[:200]!r}")

            del model, tok, out
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
