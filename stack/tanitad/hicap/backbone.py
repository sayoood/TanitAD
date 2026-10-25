"""Frozen VLM tap wrapper (paper sec:backbone, sec:backbone-choice). GPU / Hugging Face only; torch+transformers imported lazily.

STATUS: written against the model card and the framework source; NOT RUN (no GPU, Transformers `main` required for Cosmos3-Edge).
Edge hidden-state tuple is per BLOCK (57 entries): taps 28/42/56. Qwen3-VL-architecture models are per LAYER (L+1 entries).
Processor min_pixels must be <= 163,840 (256x640) or Cosmos-Reason2's sample defaults upscale a frame to ~286 tokens.
"""
from __future__ import annotations
import json
from dataclasses import dataclass

EDGE_TAPS = (28, 42, 56)                       # tuple indices (blocks)
EXPECTED_TOK_PER_FRAME = 160                   # 256x640, patch 16, 2x2 merge


def qwen3vl_taps(n_layers: int) -> tuple:
    return (n_layers // 2, (3 * n_layers) // 4, n_layers)


def pool_tokens(h, n_out: int = 128):
    """Adaptive average pooling over the token axis: [N,d] -> [n_out,d] (fixed, parameter-free)."""
    import torch
    return torch.nn.functional.adaptive_avg_pool1d(h.t().unsqueeze(0), n_out).squeeze(0).t()


@dataclass
class TapSpec:
    model_id: str = "nvidia/Cosmos3-Edge"
    taps: tuple = EDGE_TAPS
    unit: str = "block"
    n_pool: int = 128
    dtype: str = "bfloat16"


class FrozenTapBackbone:
    def __init__(self, spec: TapSpec = TapSpec(), device: str = "cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor     # Transformers `main` for Edge
        self.spec, self.device = spec, device
        self.processor = AutoProcessor.from_pretrained(spec.model_id)
        if hasattr(self.processor, "image_processor"):
            ip = self.processor.image_processor
            ip.min_pixels, ip.max_pixels = 256 * 640, 196_608                    # avoid the 286-token upscale trap
        self.model = AutoModelForImageTextToText.from_pretrained(spec.model_id, torch_dtype=getattr(torch, spec.dtype)).to(device).eval()
        for p in self.model.parameters(): p.requires_grad_(False)

    def manifest(self) -> dict:
        return dict(model_id=self.spec.model_id, taps=list(self.spec.taps), unit=self.spec.unit, n_pool=self.spec.n_pool,
                    tok_per_frame=EXPECTED_TOK_PER_FRAME)

    def extract(self, image, prompt: str = "Describe the driving scene and what the ego vehicle should do."):
        """image: PIL (256x640). Returns dict(vlm [3,n_pool,d], last [d], n_visual) -- the cache entry for one refresh."""
        import torch
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, use_cache=False)    # prefill only; no generate
        hs = out.hidden_states
        img_id = getattr(self.model.config, "image_token_id", None)
        vis = (inputs["input_ids"][0] == img_id) if img_id is not None else None
        n_vis = int(vis.sum()) if vis is not None else -1
        if n_vis != EXPECTED_TOK_PER_FRAME:
            raise RuntimeError(f"visual tokens {n_vis} != {EXPECTED_TOK_PER_FRAME}: check min_pixels / processor (fail loud)")
        taps = [pool_tokens(hs[t][0][vis].float(), self.spec.n_pool) for t in self.spec.taps]
        last = hs[self.spec.taps[-1]][0, -1].float()
        return dict(vlm=torch.stack(taps).half().cpu(), last=last.half().cpu(), n_visual=n_vis)
