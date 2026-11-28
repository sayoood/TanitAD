"""E-ENC-1 — BUILD THE FROZEN-EXTERNAL-ENCODER SWAP AND COUNT IT.

⛔ The brief's binding constraint: *"State the parameter delta, MEASURED BY
BUILDING IT — not estimated."* So nothing here is arithmetic on a spec sheet:
every number is `sum(p.numel())` over an object that was instantiated.

WHAT THE SWAP IS, AND WHAT IT HOLDS FIXED
  incumbent : frames[B,9,256,640] -> ViT5Encoder(768x12) -> tokens[B,640,768]
              -> SpatialGridReadout(640,768 -> 4x4x128) -> z_op[B,2048]
  E-ENC-1   : frames[B,9,256,640] -> 3 x (3ch sub-frame, resized 224x560,
              ImageNet-normalised) -> FROZEN facebook/dinov2-base -> drop CLS
              -> [B,640,768] each -> concat -> [B,640,2304]
              -> TRAINABLE Linear(2304->768) -> tokens[B,640,768]
              -> THE SAME SpatialGridReadout -> z_op[B,2048]

⭐ The adapter's OUTPUT WIDTH IS 768 ON PURPOSE. It makes the swap a
ONE-VARIABLE change: `n_tokens` 640, `d_model` 768, `grid_shape` (16,40) and
therefore `d_op` 2048 are all UNCHANGED, so `SpatialGridReadout`,
`OperativePredictor`, every uplink and every loss are shape-identical and the
readout can even be WARM-STARTED from the live checkpoint. An adapter that
emitted 2304 would have changed the readout, the state width and the predictor
in the same edit — the `--v2` conflation this programme refuses.

⚠️ THE 3-SUB-FRAME CONCATENATION IS INHERITED FROM THE ER10 DISCRIMINATOR, NOT
INVENTED HERE (`er10_dino_cache.py` docstring; `dino_meta.json`
`sub_frames_concatenated: 3`, `d_model_tokens: 2304`). It is also a DECLARED
ASYMMETRY against our encoder — see the prereg's §Falsifiers C-CONCAT.

⛔ NO TRAINING. Instantiation + a single forward shape check on CPU/GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor, nn

_REPO = Path(__file__).resolve().parents[5]
for _p in (_REPO / "stack",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# the shim: satisfies EXACTLY the contract V6Stack.__init__ reads off an encoder
#   .n_tokens  .grid_shape  .grid_h/.grid_w  .cfg  forward([B,C,H,W])->[B,N,D]
# (v6.py:3328-3332 reads n_tokens + grid_shape; encoder.py:98 states the contract)
# --------------------------------------------------------------------------- #
class ExternalTokenEncoder(nn.Module):
    """FROZEN external backbone + a trainable width adapter.

    ⛔ The backbone is frozen at construction (`requires_grad_(False)`) and the
    freeze is ASSERTED, not assumed — a "frozen" arm whose backbone quietly
    trains is the leak class this programme keeps re-finding.
    """

    def __init__(self, cfg: EncoderConfig, repo: str = "facebook/dinov2-base",
                 sub_frames: int = 3, ext_hw: tuple[int, int] = (224, 560),
                 real_weights: bool = True):
        super().__init__()
        self.cfg = cfg
        h, w = cfg.image_hw()
        self.grid_h, self.grid_w = h // cfg.patch_size, w // cfg.patch_size
        self.n_tokens = self.grid_h * self.grid_w
        self.sub_frames = int(sub_frames)
        self.ext_hw = tuple(int(v) for v in ext_hw)
        if cfg.in_channels != 3 * self.sub_frames:
            raise ValueError(
                f"in_channels {cfg.in_channels} != 3 x {self.sub_frames} "
                f"sub-frames — the D-015 stack is 3ch per sub-frame")

        import truststore
        truststore.inject_into_ssl()          # certifi fails behind this proxy
        from transformers import AutoConfig, AutoModel
        if real_weights:
            self.backbone = AutoModel.from_pretrained(repo)
        else:                                  # architecture-only (offline)
            self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(repo))
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        d_ext = int(self.backbone.config.hidden_size)
        self.patch_ext = int(self.backbone.config.patch_size)
        self.d_ext = d_ext
        # ⭐ 768 out, NOT 2304 — see module docstring
        self.adapter = nn.Linear(d_ext * self.sub_frames, cfg.d_model)
        self.register_buffer(
            "_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        # the grid the external backbone actually produces at ext_hw
        self.ext_grid = (self.ext_hw[0] // self.patch_ext,
                         self.ext_hw[1] // self.patch_ext)
        if self.ext_grid != (self.grid_h, self.grid_w):
            raise ValueError(
                f"external grid {self.ext_grid} != our grid "
                f"{(self.grid_h, self.grid_w)} at {self.ext_hw} / patch "
                f"{self.patch_ext} — a swap that changes the grid changes the "
                f"readout too, which is two variables")

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.grid_h, self.grid_w)

    @property
    def grid_hw(self) -> int:
        if self.grid_h != self.grid_w:
            raise ValueError(f"token grid is {self.grid_h}x{self.grid_w} "
                             f"(non-square) — use grid_shape")
        return self.grid_h

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-2:] != torch.Size(self.cfg.image_hw()):
            raise ValueError(f"input {tuple(x.shape[-2:])} != declared "
                             f"{self.cfg.image_hw()}")
        b = x.shape[0]
        outs = []
        for i in range(self.sub_frames):
            f = x[:, 3 * i:3 * i + 3]
            f = torch.nn.functional.interpolate(
                f, size=self.ext_hw, mode="bilinear",
                align_corners=False, antialias=True)
            f = (f - self._mean) / self._std
            with torch.no_grad():
                t = self.backbone(pixel_values=f).last_hidden_state
            outs.append(t[:, -self.n_tokens:])       # drop CLS/register prefix
        t = torch.cat(outs, dim=-1)                  # [B, 640, 3*d_ext]
        return self.adapter(t).reshape(b, self.n_tokens, self.cfg.d_model)


# --------------------------------------------------------------------------- #
def cfg_from_ckpt(vc: dict) -> V6Config:
    """Rebuild the LIVE V6Config from the checkpoint's own recorded dict.

    ⛔ `_derived` is NOT a field — it is what V6Config computes. Passing it back
    would be the "read a number from the wrong scope" family."""
    vc = dict(vc)
    vc.pop("_derived", None)
    enc = EncoderConfig(**vc.pop("encoder"))
    ro = ReadoutConfig(**vc.pop("readout"))
    pr = vc.pop("predictor")
    pr["horizons"] = tuple(pr["horizons"])
    pred = PredictorConfig(**pr)
    for k in ("op_band_s", "tac_band_s"):
        if k in vc:
            vc[k] = tuple(vc[k])
    return V6Config(encoder=enc, readout=ro, predictor=pred, **vc)


def group_counts(stack: V6Stack) -> dict:
    rep = {}
    for name, p in stack.named_parameters():
        g = stack.group_of(name)
        d = rep.setdefault(g, {"total": 0, "trainable": 0, "frozen": 0})
        d["total"] += p.numel()
        d["trainable" if p.requires_grad else "frozen"] += p.numel()
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="facebook/dinov2-base")
    ap.add_argument("--forward-check", action="store_true")
    a = ap.parse_args()

    out: dict = {
        "_evidence_class": "MEASURED (ours; counts + shapes AT INSTANTIATION)",
        "eval_tier": "N/A — a build measurement, not an eval",
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    meta = ck["_meta"]["config"]
    out["ckpt"] = {
        "path": a.ckpt, "step": int(ck["_meta"]["step"]),
        "stage": meta["args"]["stage"], "run": meta["run"],
        "recorded_param_report_total": int(meta["param_report"]["total"]),
        "recorded_per_group": meta["param_report"]["per_group"],
        "state_dict_numel": int(sum(v.numel() for v in ck["model"].values())),
        "state_dict_n_tensors": len(ck["model"]),
    }

    # ---- 1. the INCUMBENT, re-instantiated from the checkpoint's own config --
    cfg = cfg_from_ckpt(meta["v6_config"])
    inc = V6Stack(cfg)
    inc_rep = inc.param_report()
    out["incumbent"] = {
        "total": int(inc_rep["total"]),
        "per_group": {k: int(v) for k, v in inc_rep["per_group"].items()},
        "encoder_class": type(inc.encoder).__name__,
        "n_tokens": int(inc.encoder.n_tokens),
        "grid_shape": list(inc.encoder.grid_shape),
        "d_model": int(cfg.encoder.d_model),
        "d_op": int(cfg.d_op),
        "readout_out_dim": int(inc.readout.out_dim),
    }
    out["incumbent"]["matches_recorded"] = (
        int(inc_rep["total"]) == int(meta["param_report"]["total"]))

    # ---- 2. THE SWAP ------------------------------------------------------- #
    swap = V6Stack(cfg)                      # same config => same everything...
    ext = ExternalTokenEncoder(cfg.encoder, repo=a.repo)
    old_enc_n = sum(p.numel() for p in swap.encoder.parameters())
    swap.encoder = ext                       # ...then ONE module is replaced
    new_rep = swap.param_report()
    ext_backbone_n = sum(p.numel() for p in ext.backbone.parameters())
    ext_adapter_n = sum(p.numel() for p in ext.adapter.parameters())
    frozen_n = sum(p.numel() for p in ext.backbone.parameters() if not p.requires_grad)
    out["swap"] = {
        "external_repo": a.repo,
        "external_hidden": int(ext.d_ext),
        "external_patch": int(ext.patch_ext),
        "external_input_hw": list(ext.ext_hw),
        "external_grid": list(ext.ext_grid),
        "sub_frames_concatenated": int(ext.sub_frames),
        "backbone_params": int(ext_backbone_n),
        "backbone_params_frozen": int(frozen_n),
        "backbone_fully_frozen": bool(frozen_n == ext_backbone_n),
        "adapter_params": int(ext_adapter_n),
        "adapter_shape": [int(ext.adapter.in_features),
                          int(ext.adapter.out_features)],
        "encoder_group_after_swap": int(ext_backbone_n + ext_adapter_n),
        "total": int(new_rep["total"]),
        "per_group": {k: int(v) for k, v in new_rep["per_group"].items()},
    }
    out["delta"] = {
        "incumbent_encoder_params": int(old_enc_n),
        "swap_encoder_params": int(ext_backbone_n + ext_adapter_n),
        "encoder_group_delta": int(ext_backbone_n + ext_adapter_n - old_enc_n),
        "total_delta": int(new_rep["total"] - inc_rep["total"]),
        "total_incumbent": int(inc_rep["total"]),
        "total_swap": int(new_rep["total"]),
        "swap_TRAINABLE_encoder_params": int(ext_adapter_n),
        "trainable_delta_if_backbone_frozen": int(ext_adapter_n - old_enc_n),
        "budget": int(cfg.param_budget),
        "swap_within_budget": bool(new_rep["total"] <= cfg.param_budget),
        "capacity_note": (
            "the external backbone is FROZEN, so the arm's TRAINABLE capacity "
            "FALLS by the encoder minus the adapter — this arm cannot win on "
            "trainable capacity, which is the C6 confound running the "
            "favourable way for once"),
    }

    # ---- 3. per-group frozen/trainable under the S-W freeze ----------------- #
    from tanitad.models.v6 import apply_stage_freeze
    aud_inc = apply_stage_freeze(inc, "S-W")
    swap.encoder.backbone.requires_grad_(False)   # re-assert after the freeze
    aud_swap = apply_stage_freeze(swap, "S-W")
    swap.encoder.backbone.requires_grad_(False)
    out["s_w_freeze"] = {
        "incumbent_n_trainable": int(aud_inc["n_trainable"]),
        "swap_n_trainable_before_backbone_refreeze": int(aud_swap["n_trainable"]),
        "swap_n_trainable_after_backbone_refreeze": int(
            sum(p.numel() for p in swap.parameters() if p.requires_grad)),
        "⛔ WARNING": (
            "apply_stage_freeze sets requires_grad from the GROUP MAP, so it "
            "UN-FREEZES the external backbone (group 'encoder'). MEASURED "
            "here. A real E-ENC-1 run MUST re-assert the backbone freeze after "
            "apply_stage_freeze, or the 'frozen external encoder' arm trains "
            "86 M foreign params and is not the arm it claims to be."),
    }
    out["groups_after_swap"] = group_counts(swap)

    # ---- 4. a REAL forward, so 'it builds' is not the whole claim ----------- #
    if a.forward_check:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ext = ext.to(dev).eval()
        x = torch.randn(1, cfg.encoder.in_channels, *cfg.encoder.image_hw(),
                        device=dev)
        with torch.no_grad():
            tok = ext(x)
        ro = inc.readout.to(dev)
        with torch.no_grad():
            z = ro(tok)
        out["forward_check"] = {
            "device": dev,
            "in_shape": list(x.shape),
            "token_shape": list(tok.shape),
            "z_op_shape": list(z.shape),
            "z_op_matches_d_op": bool(z.shape[-1] == cfg.d_op),
            "cuda_max_mem_gb": (float(torch.cuda.max_memory_allocated()) / 1e9
                                if dev == "cuda" else None),
            "_note": "⛔ only torch.cuda.max_memory_allocated() is admissible",
        }

    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), "utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
