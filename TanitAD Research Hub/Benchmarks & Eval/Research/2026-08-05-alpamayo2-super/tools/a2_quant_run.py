"""Run Alpamayo 2 Super on a 46 GB A40 by 4-bit-quantising ONLY the VLM backbone.

⛔ WHY THIS IS NEEDED. NVIDIA's measured peak is 72,115 MiB on an H100 80 GB.
Our A40 has 46,068 MiB. The model does not fit as shipped, and NVIDIA state that
"other GPU architectures have not yet been validated".

⛔ WHAT IS QUANTISED, AND WHAT IS DELIBERATELY NOT. The checkpoint splits cleanly
into `vlm.*` (1,058 tensors, the 32B Qwen3-VL) and `expert.*` (717 tensors, the
2.3B diffusion action expert). Only the backbone's LANGUAGE layers go to NF4:

  quantised  : vlm.model.language_model.*            (~31B -> ~17.5 GB)
  kept BF16  : vlm.model.visual.*   (vision tower)
               vlm.lm_head
               expert.*             (action expert + action_in/out_proj)

The expert is the thing that emits the trajectory. Quantising a 2.3B diffusion
head to 4 bits to save ~3.5 GB, when the 31B backbone is where the memory
actually is, would corrupt the measured output to buy nothing. Same for the
vision tower: it is small and it is the only thing that sees.

⚠️ THIS IS NOT NVIDIA'S MODEL ANY MORE. NF4 is lossy and is not part of any
validated configuration. Every number produced here is labelled
`QUANTISED-4BIT-UNVALIDATED` and may not be compared against NVIDIA's published
figures. The honest use is a *self-consistent* comparison against our own arm on
our own windows, plus a same-input sanity check.

The load is otherwise NVIDIA's own: this patches exactly one line
(`inference_smoke.py:133`,
 `Alpamayo2Super.from_pretrained(model_id, dtype=bfloat16, device_map="cuda:0")`)
and reuses their entire pipeline — data loading, prompt construction, CoC
generation, expert sampling, minADE, visualisation.
"""
from __future__ import annotations

import os
import sys

import torch
from transformers import BitsAndBytesConfig

# ⛔ Skip-list is matched by SUBSTRING against module names by bitsandbytes, so
# "expert" also covers expert.action_in_proj / action_out_proj. Verified against
# the real module names read from model.safetensors.index.json.
SKIP = ["visual", "lm_head", "expert", "action_in_proj", "action_out_proj"]

BNB = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=SKIP,
)

from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super  # noqa: E402

_ORIG = Alpamayo2Super.from_pretrained.__func__


@classmethod
def _patched_from_pretrained(cls, *args, **kwargs):
    kwargs["quantization_config"] = BNB
    kwargs["device_map"] = {"": 0}
    kwargs.setdefault("dtype", torch.bfloat16)
    # flash-attn may not have built on this box; SDPA is the backend NVIDIA's own
    # memory profile was measured with, so it is not a downgrade.
    kwargs.setdefault("attn_implementation",
                      os.environ.get("A2_ATTN", "sdpa"))
    print(f"[quant] NF4 backbone · BF16 skip-list {SKIP} · "
          f"attn={kwargs['attn_implementation']}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    m = _ORIG(cls, *args, **kwargs)
    n_q = sum(1 for _ in m.modules() if type(_).__name__ == "Linear4bit")
    print(f"[quant] Linear4bit modules: {n_q}", flush=True)
    print(f"[quant] weights resident: "
          f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    return m


Alpamayo2Super.from_pretrained = _patched_from_pretrained

if __name__ == "__main__":
    from alpamayo2_super import inference_smoke
    try:
        inference_smoke.main()
    finally:
        if torch.cuda.is_available():
            print(f"[quant] PEAK device memory: "
                  f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
                  f"(A40 capacity 45.0 GiB)", flush=True)
