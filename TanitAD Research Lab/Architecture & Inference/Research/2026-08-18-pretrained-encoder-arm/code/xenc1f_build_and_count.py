"""E-XENC-1F — BUILD THE *TRAINABLE* PRETRAINED-ENCODER ARM AND MEASURE IT.

⛔ The brief's binding constraint: *"Parameter cost MEASURED BY BUILDING, never
estimated."* Every number this emits is `sum(p.numel())` / `p.grad` over an
object that was instantiated and, where stated, actually stepped.

WHAT THIS MEASURES THAT THE FROZEN ARM CANNOT
  E-XENC-1  (sibling, `2026-08-18-encoder-experiments`) : DINOv2 FROZEN,
            trainable width adapter only. Its own forward wraps the backbone in
            `torch.no_grad()`.
  E-XENC-1F (this document)                              : the SAME swap with
            the backbone TRAINABLE and the `no_grad` REMOVED.

  ⭐ The pair is a one-variable ablation on *"does our S-W objective PRESERVE or
  DESTROY pretrained geometry?"* — the question neither REF-A nor E-XENC-1 can
  answer, because in both of them the encoder never trains.

SECTIONS (priority order — each banks to JSON before the next runs)
  1. INVARIANT   default `V6Stack(V6Config())` must stay 87,893,449 / 405.
  2. INCUMBENT   rebuilt from the LIVE checkpoint's own recorded `v6_config`.
  3. SWAP        E-XENC-1 (frozen twin) and E-XENC-1F (trainable), param counts
                 under the REAL `apply_stage_freeze("S-W")`.
  4. GRAD-REACH  a real forward+backward; count backbone params that receive a
                 non-None and a NON-ZERO gradient. ⛔ `requires_grad=True` is
                 NOT evidence that a parameter trains — a `no_grad` in the
                 forward silences it while the freeze audit still calls it
                 trainable. This section is what tells the two apart.
  5. WARMSTART   can DINOv2-base weights be loaded into OUR `ViT5Encoder`
                 instead? Enumerate shape-compatible / incompatible / missing
                 keys. This costs the *other* way of doing "pretrained+trainable".
  6. MEMORY      `torch.cuda.max_memory_allocated()` only (⛔ the ONLY admissible
                 device-memory probe in this programme).

⛔ NO TRAINING. No checkpoint is written. Nothing touches the live run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn

_REPO = Path(__file__).resolve().parents[5]
for _p in (_REPO / "stack",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.encoder import ViT5Encoder  # noqa: E402
from tanitad.models.v6 import (  # noqa: E402
    STAGE_GROUPS, V6Config, V6Stack, apply_stage_freeze,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The pinned invariant. `load_resume` is hard `strict=True`, so a default-build
# change kills the live v6F S-W resume (v6.py:3069, :3174;
# tests/test_v6_agent_slots.py:176).
DEFAULT_PARAMS = 87_893_449
DEFAULT_KEYS = 405


class ExternalTokenEncoder(nn.Module):
    """External backbone + width adapter, with the freeze as an EXPLICIT knob.

    ⛔ `trainable_backbone` controls TWO things that must move together, and
    conflating them is the trap this class exists to make visible:
      (a) `requires_grad` on the backbone parameters, and
      (b) whether the forward runs under `torch.no_grad()`.
    Setting (a) without (b) produces a model whose freeze audit says "trainable"
    and whose parameters receive NO GRADIENT — the same wrong-scope family as
    reading `df` on a pod. §4 measures both.
    """

    def __init__(self, cfg: EncoderConfig, repo: str = "facebook/dinov2-base",
                 sub_frames: int = 3, ext_hw: tuple[int, int] = (224, 560),
                 trainable_backbone: bool = False):
        super().__init__()
        self.cfg = cfg
        h, w = cfg.image_hw()
        self.grid_h, self.grid_w = h // cfg.patch_size, w // cfg.patch_size
        self.n_tokens = self.grid_h * self.grid_w
        self.sub_frames = int(sub_frames)
        self.ext_hw = tuple(int(v) for v in ext_hw)
        self.trainable_backbone = bool(trainable_backbone)
        if cfg.in_channels != 3 * self.sub_frames:
            raise ValueError(
                f"in_channels {cfg.in_channels} != 3 x {self.sub_frames} "
                f"sub-frames — the D-015 stack is 3ch per sub-frame")

        import truststore
        truststore.inject_into_ssl()          # certifi fails behind this proxy
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(repo)
        self.backbone.requires_grad_(self.trainable_backbone)
        # ⭐ eval() on the FROZEN arm only. A trainable backbone must run in
        # train() or its dropout/LayerScale path differs from what it optimises.
        if not self.trainable_backbone:
            self.backbone.eval()
        self.d_ext = int(self.backbone.config.hidden_size)
        self.patch_ext = int(self.backbone.config.patch_size)
        self.adapter = nn.Linear(self.d_ext * self.sub_frames, cfg.d_model)
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
                             persistent=False)
        self.ext_grid = (self.ext_hw[0] // self.patch_ext,
                         self.ext_hw[1] // self.patch_ext)
        if self.ext_grid != (self.grid_h, self.grid_w):
            raise ValueError(
                f"external grid {self.ext_grid} != our grid "
                f"{(self.grid_h, self.grid_w)} — a swap that changes the grid "
                f"changes the readout too, which is two variables")

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
            if self.trainable_backbone:
                t = self.backbone(pixel_values=f).last_hidden_state
            else:
                with torch.no_grad():
                    t = self.backbone(pixel_values=f).last_hidden_state
            outs.append(t[:, -self.n_tokens:])       # drop CLS/register prefix
        t = torch.cat(outs, dim=-1)
        return self.adapter(t).reshape(b, self.n_tokens, self.cfg.d_model)


def cfg_from_ckpt(vc: dict) -> V6Config:
    """Rebuild the LIVE V6Config from the checkpoint's own recorded dict."""
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
    rep: dict = {}
    for name, p in stack.named_parameters():
        g = stack.group_of(name)
        d = rep.setdefault(g, {"total": 0, "trainable": 0, "frozen": 0})
        d["total"] += p.numel()
        d["trainable" if p.requires_grad else "frozen"] += p.numel()
    return rep


def bank(out: dict, path: Path) -> None:
    """Write incrementally — a killed run still yields every finished section."""
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="facebook/dinov2-base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-grad", action="store_true")
    a = ap.parse_args()

    outp = Path(a.out)
    dev = a.device if torch.cuda.is_available() else "cpu"
    out: dict = {
        "_evidence_class": "MEASURED (ours; counts/grads/shapes AT INSTANTIATION)",
        "eval_tier": "N/A - a build measurement, not an eval",
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used": dev,
        "device_name": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
        "STAGE_GROUPS_S_W": list(STAGE_GROUPS["S-W"]),
    }

    # ---- 1. THE INVARIANT ---------------------------------------------------
    t0 = time.time()
    dflt = V6Stack(V6Config())
    dflt_sd = dflt.state_dict()
    dflt_params = int(sum(p.numel() for p in dflt.parameters()))
    out["invariant_default_build"] = {
        "params": dflt_params,
        "state_dict_keys": len(dflt_sd),
        "expect_params": DEFAULT_PARAMS,
        "expect_keys": DEFAULT_KEYS,
        "params_ok": dflt_params == DEFAULT_PARAMS,
        "keys_ok": len(dflt_sd) == DEFAULT_KEYS,
        "_why": ("load_resume is hard strict=True; a default-build change kills "
                 "the live v6F S-W resume (v6.py:3069,:3174)"),
        "secs": round(time.time() - t0, 2),
    }
    del dflt, dflt_sd
    bank(out, outp)

    # ---- 2. THE INCUMBENT, from the checkpoint's own config ------------------
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    meta = ck["_meta"]["config"]
    out["ckpt"] = {
        "path": a.ckpt,
        "step": int(ck["_meta"]["step"]),
        "stage": meta["args"]["stage"],
        "run": meta["run"],
        "recorded_param_report_total": int(meta["param_report"]["total"]),
        "state_dict_n_tensors": len(ck["model"]),
    }
    cfg = cfg_from_ckpt(meta["v6_config"])
    inc = V6Stack(cfg)
    inc_rep = inc.param_report()
    inc_freeze = apply_stage_freeze(inc, "S-W")
    out["incumbent"] = {
        "total": int(inc_rep["total"]),
        "per_group": {k: int(v) for k, v in inc_rep["per_group"].items()},
        "encoder_class": type(inc.encoder).__name__,
        "n_tokens": int(inc.encoder.n_tokens),
        "grid_shape": list(inc.encoder.grid_shape),
        "d_model": int(cfg.encoder.d_model),
        "d_op": int(inc.readout.out_dim),
        "n_trainable_under_S_W": int(sum(p.numel() for p in inc.parameters()
                                         if p.requires_grad)),
        "freeze_audit": {k: {kk: int(vv) for kk, vv in v.items()}
                         for k, v in inc_freeze.items()
                         if isinstance(v, dict)
                         and set(v) >= {"trainable", "frozen"}},
        "matches_recorded": int(inc_rep["total"]) == int(meta["param_report"]["total"]),
    }
    del ck
    bank(out, outp)

    # ---- 3. THE TWO SWAPS ---------------------------------------------------
    variants: dict[str, dict] = {}
    built: dict[str, V6Stack] = {}
    for tag, train_bb in (("E-XENC-1_frozen", False), ("E-XENC-1F_trainable", True)):
        enc = ExternalTokenEncoder(cfg.encoder, repo=a.repo,
                                   trainable_backbone=train_bb)
        st = V6Stack(cfg)            # same config => same everything...
        st.encoder = enc             # ...then ONE module is replaced
        rep = st.param_report()
        bb = [p for n, p in st.named_parameters() if n.startswith("encoder.backbone.")]
        ad = [p for n, p in st.named_parameters() if n.startswith("encoder.adapter.")]
        pre = {
            "backbone_params": int(sum(p.numel() for p in bb)),
            "adapter_params": int(sum(p.numel() for p in ad)),
            "backbone_requires_grad_at_construction":
                bool(all(p.requires_grad for p in bb)),
        }
        fz = apply_stage_freeze(st, "S-W")
        post = {
            "n_trainable_after_apply_stage_freeze":
                int(sum(p.numel() for p in st.parameters() if p.requires_grad)),
            "backbone_requires_grad_after_freeze":
                bool(all(p.requires_grad for p in bb)),
        }
        # the runbook step the FROZEN arm needs, measured rather than assumed
        st.encoder.backbone.requires_grad_(False)
        post["n_trainable_if_backbone_refrozen"] = int(
            sum(p.numel() for p in st.parameters() if p.requires_grad))
        # restore the arm's intended state before anything downstream runs
        st.encoder.backbone.requires_grad_(train_bb)
        variants[tag] = {
            "trainable_backbone_flag": train_bb,
            "total": int(rep["total"]),
            "per_group": {k: int(v) for k, v in rep["per_group"].items()},
            "external_repo": a.repo,
            "external_hidden": int(enc.d_ext),
            "external_patch": int(enc.patch_ext),
            "external_input_hw": list(enc.ext_hw),
            "external_grid": list(enc.ext_grid),
            "sub_frames": int(enc.sub_frames),
            **pre, **post,
            "freeze_audit": {k: {kk: int(vv) for kk, vv in v.items()}
                             for k, v in fz.items()
                             if isinstance(v, dict)
                             and set(v) >= {"trainable", "frozen"}},
            "group_counts": {k: {kk: int(vv) for kk, vv in v.items()}
                             for k, v in group_counts(st).items()},
        }
        built[tag] = st
        out["variants"] = variants
        bank(out, outp)

    # deltas against the incumbent — the number the brief asks to be MEASURED
    out["deltas_vs_incumbent"] = {
        tag: {
            "d_total": variants[tag]["total"] - out["incumbent"]["total"],
            "d_total_pct": round(
                100.0 * (variants[tag]["total"] - out["incumbent"]["total"])
                / out["incumbent"]["total"], 4),
            "d_trainable_under_S_W":
                variants[tag]["n_trainable_after_apply_stage_freeze"]
                - out["incumbent"]["n_trainable_under_S_W"],
            "d_trainable_pct": round(
                100.0 * (variants[tag]["n_trainable_after_apply_stage_freeze"]
                         - out["incumbent"]["n_trainable_under_S_W"])
                / out["incumbent"]["n_trainable_under_S_W"], 4),
        } for tag in variants
    }
    bank(out, outp)

    # ---- 4. THE GRADIENT-REACH CONTROL --------------------------------------
    # ⛔ requires_grad is a DECLARATION. This measures ARRIVAL.
    if not a.skip_grad:
        grad_reach: dict = {}
        for tag, st in built.items():
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                st = st.to(dev)
                st.train()
                x = torch.randn(1, cfg.encoder.in_channels,
                                *cfg.encoder.image_hw(), device=dev)
                fwd_t0 = time.time()
                tok = st.encoder(x)
                z = st.readout(tok)
                fwd_peak = (torch.cuda.max_memory_allocated() / 2**30
                            if torch.cuda.is_available() else None)
                # a scalar that depends on every token channel
                loss = z.float().pow(2).mean()
                loss.backward()
                bwd_peak = (torch.cuda.max_memory_allocated() / 2**30
                            if torch.cuda.is_available() else None)
                bb = [(n, p) for n, p in st.named_parameters()
                      if n.startswith("encoder.backbone.")]
                ad = [(n, p) for n, p in st.named_parameters()
                      if n.startswith("encoder.adapter.")]
                n_bb_grad_notnone = sum(1 for _, p in bb if p.grad is not None)
                n_bb_grad_nonzero = sum(
                    1 for _, p in bb
                    if p.grad is not None and bool(p.grad.abs().sum() > 0))
                numel_bb_grad = int(sum(p.numel() for _, p in bb
                                        if p.grad is not None))
                bb_gnorm = float(torch.sqrt(sum(
                    (p.grad.float().pow(2).sum() for _, p in bb
                     if p.grad is not None),
                    torch.zeros((), device=dev)))) if n_bb_grad_notnone else 0.0
                ad_gnorm = float(torch.sqrt(sum(
                    (p.grad.float().pow(2).sum() for _, p in ad
                     if p.grad is not None),
                    torch.zeros((), device=dev)))) if ad else 0.0
                grad_reach[tag] = {
                    "z_op_shape": list(z.shape),
                    "tokens_shape": list(tok.shape),
                    "z_op_matches_d_op":
                        int(z.shape[-1]) == int(out["incumbent"]["d_op"]),
                    "n_backbone_tensors": len(bb),
                    "n_backbone_tensors_with_grad": n_bb_grad_notnone,
                    "n_backbone_tensors_with_NONZERO_grad": n_bb_grad_nonzero,
                    "numel_backbone_receiving_grad": numel_bb_grad,
                    "backbone_grad_l2": bb_gnorm,
                    "adapter_grad_l2": ad_gnorm,
                    "peak_mem_GiB_forward": (round(fwd_peak, 4)
                                             if fwd_peak is not None else None),
                    "peak_mem_GiB_fwd_plus_bwd": (round(bwd_peak, 4)
                                                  if bwd_peak is not None else None),
                    "fwd_bwd_secs_batch1": round(time.time() - fwd_t0, 3),
                    "_probe": ("torch.cuda.max_memory_allocated ONLY - the one "
                               "admissible device-memory probe (CLAUDE.md)"),
                }
            except Exception as exc:                     # bank the failure too
                grad_reach[tag] = {"ERROR": f"{type(exc).__name__}: {exc}"}
            finally:
                built[tag] = st.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            out["grad_reach"] = grad_reach
            bank(out, outp)

    del built
    bank(out, outp)

    # ---- 5. WARM-START FEASIBILITY INTO **OUR** ViT-5 ------------------------
    # The other way to do "pretrained + trainable": keep OUR architecture and
    # initialise it from DINOv2. Measured, not asserted.
    try:
        import truststore
        truststore.inject_into_ssl()
        from transformers import AutoModel
        dino = AutoModel.from_pretrained(a.repo)
        d_sd = dino.state_dict()
        ours = ViT5Encoder(cfg.encoder)
        o_sd = ours.state_dict()
        d_shapes = {k: tuple(v.shape) for k, v in d_sd.items()}
        o_shapes = {k: tuple(v.shape) for k, v in o_sd.items()}
        shared_names = sorted(set(d_shapes) & set(o_shapes))
        shape_match = [k for k in shared_names if d_shapes[k] == o_shapes[k]]
        d_by_shape: dict[tuple, int] = {}
        for v in d_shapes.values():
            d_by_shape[v] = d_by_shape.get(v, 0) + 1
        o_transferable = int(sum(
            v.numel() for k, v in o_sd.items()
            if d_by_shape.get(tuple(v.shape), 0) > 0))
        out["warmstart_into_our_vit5"] = {
            "_question": ("can DINOv2-base weights initialise OUR ViT5Encoder "
                          "instead of being bolted on as a foreign backbone?"),
            "dino_n_tensors": len(d_sd),
            "dino_numel": int(sum(v.numel() for v in d_sd.values())),
            "ours_n_tensors": len(o_sd),
            "ours_numel": int(sum(v.numel() for v in o_sd.values())),
            "n_names_in_common": len(shared_names),
            "n_names_in_common_with_matching_shape": len(shape_match),
            "ours_numel_with_ANY_shape_compatible_dino_tensor": o_transferable,
            "ours_numel_with_NO_shape_compatible_dino_tensor":
                int(sum(v.numel() for v in o_sd.values())) - o_transferable,
            "patch_embed_ours": list(o_shapes.get("patch.weight", ())) or None,
            "patch_embed_dino": next(
                (list(v) for k, v in d_shapes.items()
                 if k.endswith("patch_embeddings.projection.weight")), None),
            "our_keys_sample": sorted(o_shapes)[:12],
            "dino_keys_sample": sorted(d_shapes)[:12],
            "our_norm_type": "RMSNorm (weight only, no bias, no re-centering)",
            "dino_norm_type": "LayerNorm (weight + bias, re-centering)",
            "our_qkv_bias": any(k.endswith("attn.qkv.bias") for k in o_shapes),
            "dino_qkv_bias": any("attention" in k and k.endswith(".bias")
                                 for k in d_shapes),
            "our_has_qk_norm": any(".q_norm." in k for k in o_shapes),
            "dino_has_qk_norm": any(".q_norm." in k or "query_norm" in k
                                    for k in d_shapes),
        }
    except Exception as exc:
        out["warmstart_into_our_vit5"] = {"ERROR": f"{type(exc).__name__}: {exc}"}
    bank(out, outp)

    print(json.dumps({k: v for k, v in out.items()
                      if k in ("invariant_default_build", "deltas_vs_incumbent",
                               "grad_reach")}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
