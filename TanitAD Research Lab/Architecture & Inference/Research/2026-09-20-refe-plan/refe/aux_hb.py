#!/usr/bin/env python3
"""VDAC A5, arm HB: a LoRA-adapted COPY of P6's trunk that feeds ONLY the auxiliary drivable-area head (PREREG_VDAC_P6_A5_DRAFT section 2, READINESS.md H1-H7).

WHY A NEW MODULE.  model.py is in the launch gate's pinned closure and `scorer_finetune.main()` keeps `fwd` as a closure, so every HB piece that can be a plain function lives here and
scorer_finetune.py only CALLS it (one call site per hop; the trace table is BUILD4.md section 2).  Nothing here touches `model.py`.

THE DESIGN IN ONE PARAGRAPH.  `HBTrunk` holds `copy.deepcopy(model.backbone)` as a SIBLING (never a submodule of `model`, so it is not in `model.state_dict()` and `model_sft_*.pt` still loads
strictly in the planner).  Every base weight of the copy is frozen; only the rank-r Q/V LoRA tensors `A`, `B` (`model.LoRALinear`) train.  Its context re-runs the six lines of
`model.REFe.forward` that make `visual_ctx` (`flat = img.reshape(B*N, ...)`, `backbone(flat)`, `+ model._pos3d(img, N, calib)`, `model.scene_proj(...)`) on the copy.  The head reads that tensor;
the scorer arms keep reading P6's own `visual_ctx`, so there is NO autograd path from the scorer loss to the copy, and none from L_aux to P6's trunk (P6's parameters stay `requires_grad False`).

DEFAULTS.  Every default disables the mechanism: `--aux-drv-trunk-lora off`, `--aux-drv-lora-lr None`, `--aux-drv-cell 2.0`, `--aux-drv-bw 0.0`.  An arm that tests a lever NAMES it in its argv
(launch_gate.vdac_argv).

DELIBERATE REGRESSIONS (runtime switches, each must turn G21 / G12 / G19 RED; test_vdac_hb.py runs every one):
  REFE_AUX_MUTATE_HB_P6CTX        the head reads P6's ctx instead of the copy's           (the copy's LoRA gets no gradient)
  REFE_AUX_MUTATE_HB_SHARE        the "copy" IS P6's backbone (shared storage)            (P6's LoRA gets a gradient)
  REFE_AUX_MUTATE_HB_UNFREEZE_SP  P6's scene_proj is unfrozen                              (P6 gradient / state changes)
  REFE_AUX_MUTATE_HB_DETACH       the copy's output is detached                            (the copy's LoRA gets no gradient)
  REFE_AUX_MUTATE_HB_STALE_TAP    G12's tap row reads outs[1] (P6's visual_ctx) while HB is on
  REFE_AUX_MUTATE_EGO_TRUNK       the ego + command token is added to the copy's visual tokens BEFORE scene_proj (G19 must see it)
"""
from __future__ import annotations

import copy
import hashlib
import math
import os

import torch
import torch.nn as nn

FORMAT_V2 = "aux_drv/2"
ENV_P6CTX = "REFE_AUX_MUTATE_HB_P6CTX"
ENV_SHARE = "REFE_AUX_MUTATE_HB_SHARE"
ENV_UNFREEZE_SP = "REFE_AUX_MUTATE_HB_UNFREEZE_SP"
ENV_DETACH = "REFE_AUX_MUTATE_HB_DETACH"
ENV_STALE_TAP = "REFE_AUX_MUTATE_HB_STALE_TAP"
ENV_EGO_TRUNK = "REFE_AUX_MUTATE_EGO_TRUNK"
HB_MUTATE_ENVS = (ENV_P6CTX, ENV_SHARE, ENV_UNFREEZE_SP, ENV_DETACH, ENV_STALE_TAP, ENV_EGO_TRUNK)
VITL16_LORA_PARAMS = 3_145_728          # 24 blocks x {q, v} x 32 x (1,024 + 1,024): ARITHMETIC (model.py LoRALinear, BACKBONES["vitl16"]); the preflight prints the built count and G-DVB asserts it
P6_LORA_LR = 2e-4                       # P6's OWN LoRA learning rate = train.py's single AdamW group at --lr 2e-4 (p6_pod.sh:487, train.py:1462, 1469): see BUILD4.md section 1


class HBCkptRefused(RuntimeError):
    """an aux_drv/2 file that is not the complete HB state (a head-only file, a missing LoRA tensor or optimiser state, a wrong key set)"""


# ================================================================================================ CLI
def add_hb_args(ap) -> None:
    """HB's and HC's flags.  Kept OUT of aux_drivable_head.add_aux_args (that function also builds train.py's parser, whose G-DVB table must not grow)."""
    ap.add_argument("--aux-drv-trunk-lora", default="off", choices=("off", "on"),
                    help="VDAC A5 arm HB: the head reads a LoRA-adapted COPY of P6's trunk (default off = the head reads P6's visual_ctx).  Needs --aux-drv and --aux-drv-lora-lr")
    ap.add_argument("--aux-drv-lora-lr", type=float, default=None,
                    help="HB: the copy's LoRA peak learning rate (its OWN AdamW group; the scorer keeps --lr).  None = no LoRA optimiser; naming it without --aux-drv-trunk-lora on is refused")
    ap.add_argument("--aux-drv-cell", type=float, default=2.0,
                    help="HC: the head's cell in metres (2.0 = HA/HB, 1.0 = HC: 80 x 64 queries over the same extent).  Default 2.0 = the head the S1 arms built")
    ap.add_argument("--aux-drv-bw", type=float, default=0.0,
                    help="HC: the boundary-band weight b: the Huber term of a cell with |block-mean SDF| < --aux-drv-bw-band-m is multiplied by 1 + b (default 0.0 = the unweighted loss, bit for bit)")
    ap.add_argument("--aux-drv-bw-band-m", type=float, default=3.0, help="HC: the boundary band half-width in metres (the card's near-boundary definition, 3 m); read only when --aux-drv-bw > 0")


def hb_refusals(a) -> list:
    """Every reason an HB / HC command line must not start (the `epo_refusals` pattern).  [] = fine."""
    bad = []
    on = bool(getattr(a, "aux_drv", False))
    hb = getattr(a, "aux_drv_trunk_lora", "off") == "on"
    lr = getattr(a, "aux_drv_lora_lr", None)
    if not on:
        for nm, off_val in (("aux_drv_trunk_lora", "off"), ("aux_drv_lora_lr", None), ("aux_drv_cell", 2.0), ("aux_drv_bw", 0.0)):
            if getattr(a, nm, off_val) != off_val:
                bad.append(f"--{nm.replace('_', '-')} is set but --aux-drv is not: the lever would silently do nothing")
        return bad
    if lr is not None and not hb:
        bad.append("--aux-drv-lora-lr is set but --aux-drv-trunk-lora is not on: the lever would silently do nothing")
    if hb and not (lr is not None and lr > 0.0):
        bad.append("--aux-drv-trunk-lora on needs --aux-drv-lora-lr > 0 NAMED: the LoRA learning rate is a pre-registered value, not --lr")
    if hb and getattr(a, "aux_drv_feed", "off") == "on":
        bad.append("--aux-drv-trunk-lora on with --aux-drv-feed on: HB is an H1 arm (the scorer never reads the head or the copy)")
    if hb and getattr(a, "aux_drv_tap", "visual") != "visual":
        bad.append("--aux-drv-trunk-lora on needs --aux-drv-tap visual (the copy's ctx replaces visual_ctx)")
    if not (getattr(a, "aux_drv_cell", 2.0) > 0.0):
        bad.append("--aux-drv-cell must be > 0")
    if getattr(a, "aux_drv_bw", 0.0) < 0.0:
        bad.append("--aux-drv-bw must be >= 0")
    if getattr(a, "aux_drv_bw", 0.0) == 0.0 and getattr(a, "aux_drv_bw_band_m", 3.0) != 3.0:
        bad.append("--aux-drv-bw-band-m is set but --aux-drv-bw is 0: the lever would silently do nothing")
    return bad


# ================================================================================================ the copy
def is_lora_tensor(name: str) -> bool:
    return name.endswith(".A") or name.endswith(".B")


def lora_named(backbone) -> list:
    """[(name, parameter)] of every LoRALinear A / B in the backbone, in module order (LoRALinear is recognised by class name so the module has no import of model.py)."""
    out = []
    for mn, m in backbone.named_modules():
        if type(m).__name__ == "LoRALinear":
            for k in ("A", "B"):
                out.append((f"{mn}.{k}" if mn else k, getattr(m, k)))
    return out


def expected_lora_params(backbone) -> int:
    """ARITHMETIC from the modules themselves: sum over LoRALinear of rank x (in + out) -- independent of what is trainable (vitl16: 3,145,728)."""
    n = 0
    for m in backbone.modules():
        if type(m).__name__ == "LoRALinear":
            n += int(m.A.shape[0]) * (int(m.A.shape[1]) + int(m.B.shape[0]))
    return n


class HBTrunk(nn.Module):
    """The LoRA trunk copy.  A SIBLING of the model (build it, hold it, never assign it to a model attribute)."""

    def __init__(self, model):
        super().__init__()
        shared = os.environ.get(ENV_SHARE) == "1"                     # deliberate regression: no copy, the head's trunk IS P6's
        self.backbone = model.backbone if shared else copy.deepcopy(model.backbone)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for _n, p in lora_named(self.backbone):
            p.requires_grad_(True)
        if os.environ.get(ENV_UNFREEZE_SP) == "1":                    # deliberate regression: P6's shared scene_proj is no longer frozen
            for p in model.scene_proj.parameters():
                p.requires_grad_(True)

    def lora_parameters(self) -> list:
        return [p for _n, p in lora_named(self.backbone)]

    def lora_state(self) -> dict:
        return {n: p.detach().to("cpu", copy=True) for n, p in lora_named(self.backbone)}

    def ctx(self, model, img, calib=None, ego_term=None):
        """The copy's visual_ctx: model.REFe.forward's `visual_ctx = scene_proj(backbone(flat) + pos3d)` with the copy in place of model.backbone.  P6's `_pos3d` and `scene_proj` are executed
        (their parameters stay frozen; gradient flows through them to the copy).  `ego_term` (a zero-argument callable -> [B]) is read ONLY under REFE_AUX_MUTATE_EGO_TRUNK=1."""
        if img.dim() == 4:
            img = img.unsqueeze(1)
        B, N = img.shape[0], img.shape[1]
        flat = img.reshape(B * N, *img.shape[2:])
        tok = self.backbone(flat)
        visual = tok.reshape(B, N * tok.shape[1], tok.shape[2])
        visual = visual + model._pos3d(img, N, calib)
        if os.environ.get(ENV_EGO_TRUNK) == "1" and ego_term is not None:      # G19's trunk-side regression: the model's ego + command token enters the copy's tokens before scene_proj
            visual = visual + ego_term().reshape(-1, 1, 1).to(visual.dtype)
        out = model.scene_proj(visual)
        return out.detach() if os.environ.get(ENV_DETACH) == "1" else out

    def built_facts(self, model, scorer_names) -> dict:
        """BUILT facts read off the objects the run holds (G-DVB / G21 compare them with the declared flags): never an echo of the namespace."""
        bb = model.backbone
        p6_ptrs = {p.data_ptr() for p in bb.parameters()}
        trainable = [(n, p) for n, p in self.backbone.named_parameters() if p.requires_grad]
        sd_ptrs = {v.data_ptr() for v in model.state_dict().values() if hasattr(v, "data_ptr")}
        own_ids = {id(p) for p in self.parameters()}
        return {"trunk_copy_built": True, "n_lora_trainable": sum(int(p.numel()) for _n, p in trainable), "n_lora_tensors": len(trainable),
                "trainable_all_lora": all(is_lora_tensor(n) for n, _p in trainable), "lora_expected": expected_lora_params(self.backbone),
                "copy_storage_disjoint": not any(p.data_ptr() in p6_ptrs for p in self.backbone.parameters()),
                "copy_not_in_model_state_dict": not any(p.data_ptr() in sd_ptrs for p in self.backbone.parameters()),
                "copy_not_a_model_param": not any(id(p) in own_ids for p in model.parameters()),
                "p6_params_requires_grad": sum(1 for n, p in model.named_parameters() if p.requires_grad and n.split(".")[0] not in scorer_names)}


def select_actx(hb, model, img, calib, sctx, ego_term=None):
    """THE wiring line of HB (scorer_finetune.fwd calls it once): the tensor the head reads.  hb None -> P6's `sctx` (visual_ctx), the HA path, untouched.  Under REFE_AUX_MUTATE_HB_P6CTX the
    head reads P6's ctx instead of the copy's (the deliberate regression of 'built, tested, unreachable')."""
    if hb is None or os.environ.get(ENV_P6CTX) == "1":
        return sctx
    return hb.ctx(model, img, calib, ego_term)


def tap_fields(outs, aux_ctx, hb_on: bool, tap_is) -> dict:
    """G12's tap row.  `outs` = scene_proj's outputs in ONE forward (the hook list): HA = [scene_ctx, visual_ctx]; HB = [scene_ctx, visual_ctx, copy ctx].  `tap_is_visual_ctx` keeps its name and now
    means 'the head read the tensor it is DECLARED to read': P6's visual_ctx (outs[1]) under HA, the copy's ctx (outs[2]) under HB."""
    idx = 2 if hb_on else 1
    if hb_on and os.environ.get(ENV_STALE_TAP) == "1":
        idx = 1                                                       # the stale index (READINESS H7 / T8)
    f = {"scene_proj_calls": len(outs), "hb": bool(hb_on), "tap_index": idx,
         "tap_is_visual_ctx": bool(len(outs) > idx and tap_is(aux_ctx, outs[idx])), "tap_is_scene_ctx": bool(len(outs) >= 1 and tap_is(aux_ctx, outs[0]))}
    if hb_on:
        f["tap_is_p6_visual_ctx"] = bool(len(outs) >= 2 and tap_is(aux_ctx, outs[1]))
    return f


# ================================================================================================ optimiser + schedule
def build_lora_optimizer(hb, lr: float, weight_decay: float = 0.01):
    return torch.optim.AdamW(hb.lora_parameters(), lr=lr, weight_decay=weight_decay)


def lora_step(hb, optL, lr_now: float) -> None:
    """one update of the copy: its OWN grad-clip norm (1.0) and its OWN AdamW, so it can never change the scorer's or the head's update scale"""
    torch.nn.utils.clip_grad_norm_(hb.lora_parameters(), 1.0)
    for gr in optL.param_groups:
        gr["lr"] = lr_now
    optL.step()
    optL.zero_grad(set_to_none=True)


# ================================================================================================ aux_drv/2
def save_v2(path, *, aux, hb, optH, optL, update, meta, atomic_save) -> str:
    atomic_save({"format": FORMAT_V2, "head": aux.head.state_dict(), "feed": aux.feed.state_dict() if aux.feed is not None else None, "hb_lora": hb.lora_state(),
                 "opt": optH.state_dict(), "opt_lora": optL.state_dict(), "update": update, "meta": meta}, path)
    return path


def load_v2(ck, *, aux, hb, optH, optL) -> int:
    """STRICT load of an aux_drv/2 dict (or path) into the live objects -> the update count.  Refuses a head-only file (aux_drv/1), a different LoRA key set, a missing optimiser state."""
    if isinstance(ck, (str, os.PathLike)):
        ck = torch.load(ck, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or ck.get("format") != FORMAT_V2:
        raise HBCkptRefused(f"HB mode needs an {FORMAT_V2} file, got format {ck.get('format') if isinstance(ck, dict) else type(ck).__name__!r} (a head-only file cannot restore the copy)")
    for k in ("head", "hb_lora", "opt", "opt_lora", "update"):
        if ck.get(k) is None:
            raise HBCkptRefused(f"the {FORMAT_V2} file carries no {k!r}")
    want = [n for n, _p in lora_named(hb.backbone)]
    if sorted(ck["hb_lora"]) != sorted(want):
        raise HBCkptRefused(f"the file's LoRA tensors ({len(ck['hb_lora'])}) are not the copy's ({len(want)}): different key set")
    aux.head.load_state_dict(ck["head"], strict=True)
    if aux.feed is not None:
        aux.feed.load_state_dict(ck["feed"], strict=True)
    named = dict(lora_named(hb.backbone))
    with torch.no_grad():
        for n, v in ck["hb_lora"].items():
            if tuple(named[n].shape) != tuple(v.shape):
                raise HBCkptRefused(f"LoRA tensor {n}: shape {tuple(v.shape)} != {tuple(named[n].shape)}")
            named[n].copy_(v)
    optH.load_state_dict(ck["opt"])
    optL.load_state_dict(ck["opt_lora"])
    return int(ck["update"])


# ================================================================================================ G21
def sha256_state(model) -> str:
    """sha256 over every tensor of model.state_dict() (name, dtype, shape, bytes): 'P6 is bit-identical before / after'"""
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        if hasattr(v, "detach"):
            t = v.detach().cpu().contiguous()
            h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
            if t.numel():
                h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _read(ps):
    """[(None-count, min |g|, max |g|, finite?)] of a parameter list's gradients"""
    none = sum(1 for p in ps if p.grad is None)
    vals = [float(p.grad.detach().abs().max()) for p in ps if p.grad is not None]
    return {"none": none, "n": len(ps), "min_abs": (min(vals) if vals else None), "max_abs": (max(vals) if vals else None), "all_finite": all(math.isfinite(v) for v in vals)}


def g21_fields(*, model, hb, aux, scorer_names, fwd, aux_backward, batch, lv, step_probe=None) -> dict:
    """G21 (copy isolation), one dict of RAW FIELDS that vdac_probes.judge_g21 re-judges.  Run by the trainer's preflight on the REAL model and the REAL batch.
      structure   the copy is a deepcopy (storage disjoint, not in model.state_dict(), not a model parameter), its trainable tensors are exactly the LoRA A/B (n == the arithmetic), and
                  every P6 parameter outside the scorer is requires_grad False
      live        after the run's own aux_backward at the DECLARED weight every LoRA tensor holds a finite NON-ZERO gradient (none None), and P6's non-scorer parameters hold NO gradient
      control     the same backward at weight 0.0 reads EXACTLY 0.0 on every LoRA tensor (not None: combine() always multiplies)
      no path     autograd.grad(scorer loss, LoRA tensors) is None for every tensor
      step 0      the copy's ctx equals P6's visual_ctx bit for bit (the copy starts from P6's trained LoRA)
      state       two real head + LoRA updates (a scratch deepcopy of both) leave sha256(model.state_dict()) unchanged"""
    lp = hb.lora_parameters()
    f = {"mutate": [k for k in HB_MUTATE_ENVS if os.environ.get(k) == "1"], **hb.built_facts(model, scorer_names)}
    for p in lp:
        p.grad = None
    s0, sA, sB, tg = fwd(batch)
    gs = torch.autograd.grad(sB.sum(), lp, retain_graph=True, allow_unused=True)
    f["scorer_to_copy_grads_all_none"] = bool(all(g is None for g in gs))
    w_keep = aux.w_aux
    reads = {}
    for mode in ("declared", "w0"):
        for p in lp:
            p.grad = None
        for p in model.parameters():
            p.grad = None
        if mode == "w0":
            aux.w_aux = 0.0
        try:
            aux_backward(lv, 1.0, retain=True)
        finally:
            aux.w_aux = w_keep
        reads[mode] = _read(lp)
        reads[mode + "_p6"] = {"grad_not_none": sum(1 for n, p in model.named_parameters() if n.split(".")[0] not in scorer_names and p.grad is not None)}
    f["lora_grad_declared"], f["lora_grad_w0"] = reads["declared"], reads["w0"]
    f["p6_nonscorer_grad_not_none_declared"] = reads["declared_p6"]["grad_not_none"]
    f["w_aux"] = float(w_keep)
    for p in lp:
        p.grad = None
    for p in model.parameters():
        p.grad = None
    if step_probe is not None:
        f.update(step_probe())
    return f


def two_update_probe(*, model, hb, aux, scorer_names, img, calib, lv, dev, autocast, lr_head, lr_lora) -> dict:
    """G21's 'state' fact: two REAL updates (head + copy LoRA, on scratch deepcopies, so the run's own objects and optimiser states are untouched) leave sha256(model.state_dict()) unchanged.
    The scratch optimiser also holds every parameter of `model` outside the scorer that is trainable, so an unfrozen P6 tensor MOVES here and the sha reads different."""
    sha0 = sha256_state(model)
    H, Hb = copy.deepcopy(aux), copy.deepcopy(hb)
    extra = [p for n, p in model.named_parameters() if p.requires_grad and n.split(".")[0] not in scorer_names]
    oH, oL = torch.optim.AdamW(list(H.head.parameters()), lr=lr_head, weight_decay=0.01), build_lora_optimizer(Hb, lr_lora)
    oX = torch.optim.AdamW(extra, lr=lr_lora, weight_decay=0.01) if extra else None
    for _ in range(2):
        with autocast():
            lg, sd = H.predict(Hb.ctx(model, img, calib))
        H.combine(torch.zeros((), device=dev), H.loss(lg, sd, lv[0].to(dev), lv[1].to(dev))[0]).backward()
        torch.nn.utils.clip_grad_norm_(list(H.head.parameters()), 1.0)
        oH.step(); oH.zero_grad(set_to_none=True)
        lora_step(Hb, oL, lr_lora)
        if oX is not None:
            oX.step(); oX.zero_grad(set_to_none=True)
    sha1 = sha256_state(model)
    for p in model.parameters():
        p.grad = None
    return {"p6_sha256_before": sha0, "p6_sha256_after": sha1, "p6_state_unchanged_2_updates": sha0 == sha1,
            "lora_moved_in_2_updates": any(not torch.equal(a.detach(), b.detach()) for a, b in zip(hb.lora_parameters(), Hb.lora_parameters())),
            "adam_lora_params": len(Hb.lora_parameters()), "adam_lora_params_with_state": sum(1 for p in Hb.lora_parameters() if p in oL.state and "exp_avg" in oL.state[p])}


def step0_equal(model, hb, img, calib, sctx) -> dict:
    """the copy's ctx vs P6's visual_ctx for the same images (T4): equal bit for bit, and the largest absolute difference"""
    with torch.no_grad():
        c = hb.ctx(model, img, calib)
    same = tuple(c.shape) == tuple(sctx.shape) and bool(torch.equal(c.to(sctx.dtype), sctx))
    return {"step0_ctx_equal": same, "step0_ctx_max_abs_diff": float((c.float() - sctx.float()).abs().max()) if tuple(c.shape) == tuple(sctx.shape) else None}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HB memory instrument + hygiene (HB_MEMORY.md).  Neither changes a number: the first only READS the allocator's counters, the second only drops references whose values nothing reads again.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

MEM_EVENT = "preflight_mem"
MEM_FAIL_EVENT = "preflight_mem_fail"


def mem_fields(reset: bool = True) -> dict:
    """The allocator counters since the last call, in bytes: peak allocated, peak reserved, reserved now.  CUDA only; on CPU every value is None (a null, never a 0 that reads like a measurement).
    `reset` restarts the peak, so each gate reads its OWN peak."""
    if not torch.cuda.is_available():
        return {"max_allocated_bytes": None, "max_reserved_bytes": None, "reserved_bytes": None, "allocated_bytes": None}
    f = {"max_allocated_bytes": int(torch.cuda.max_memory_allocated()), "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
         "reserved_bytes": int(torch.cuda.memory_reserved()), "allocated_bytes": int(torch.cuda.memory_allocated())}
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return f


class MemGates(dict):
    """The preflight's `gates` dict.  Assigning a gate (which the trainer does when the gate FINISHES) appends one `preflight_mem` row to preflight.jsonl with that gate's own peak allocated / reserved
    bytes, then resets the peak.  A dict subclass, so `ok = all(v["ok"] for v in gates.values())`, `json.dump` and every consumer of the gates read exactly what they read from a plain dict.
    `peak_allocated` is the running maximum over every window, so the end-of-preflight `gpu_mem_gb` still reports the whole preflight's peak although the counter was reset between gates."""

    def __init__(self, log):
        super().__init__()
        self._log = log
        self.last = None
        self.peak_allocated = 0
        mem_fields(reset=True)                                                    # the first gate's window starts here

    def __setitem__(self, name, value):
        super().__setitem__(name, value)
        f = mem_fields(reset=True)
        if f["max_allocated_bytes"] is not None:
            self.peak_allocated = max(self.peak_allocated, f["max_allocated_bytes"])
        self.last = name
        self._log({"event": MEM_EVENT, "after_gate": name, **f})

    def fail_row(self, et, ev, tb) -> dict:
        """The row for an unhandled exception (an OOM): the last gate that FINISHED, the peak since it, and the innermost frames -- the failing gate is the one whose function is on the stack."""
        import traceback
        fr = [f"{os.path.basename(x.filename)}:{x.lineno}:{x.name}" for x in traceback.extract_tb(tb)][-6:]
        return {"event": MEM_FAIL_EVENT, "after_gate": self.last, "error": f"{getattr(et, '__name__', et)}: {str(ev)[:300]}", "frames": fr, **mem_fields(reset=False)}

    def install_fail_hook(self):
        import sys
        prev = sys.excepthook

        def hook(et, ev, tb):
            try:
                self._log(self.fail_row(et, ev, tb))
            except Exception:                                                       # noqa: BLE001 -- the instrument must never hide the original failure
                pass
            prev(et, ev, tb)
        sys.excepthook = hook
        return prev


def release_aux_graph(cap: dict) -> int:
    """Drop the head's PREVIOUS forward from the trainer's `cap` dict: `cap["aux"]` = (logit, sdf) and `cap["aux_ctx"]` = the tensor the head read.  Under HB these are the output of the LoRA trunk COPY's
    forward and keep its whole autograd graph (every ViT-L activation) alive until the NEXT fwd() overwrites them -- which happens only AFTER that next fwd() has built a second graph, so two copies
    of the trunk's activations coexist (the gate-to-gate doubling at G12 and at G21).  Every reader of these two keys (aux_backward, G12's tap row, G19's _vdac_run) reads them right after
    ITS OWN fwd(), so dropping the stale pair before a gate's fresh fwd() changes no value.  -> the number of keys dropped.  Also returns the allocator's free blocks to the driver."""
    n = 0
    for k in ("aux", "aux_ctx"):
        if cap.get(k) is not None:
            cap[k] = None
            n += 1
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n

