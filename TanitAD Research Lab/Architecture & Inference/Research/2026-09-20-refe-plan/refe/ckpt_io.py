"""Checkpoint I/O shared by the trainer and the planner -- one implementation, two consumers.

⛔⛔ WHY THIS FILE EXISTS (2026-09-23, R20). Until today `train.py` saved NOTHING: no checkpoint,
no final model, no resume. The planned run is ViT-L for ~60 days on ONE A40, so any pod restart
would have cost the whole run, and a run that FINISHED would have produced no model at all -- the
trained weights lived only in process memory. Every earlier run was an overfit control or a
smoke test, where that is invisible.

Two formats, because the model is 94 % FROZEN (ViT-L: 322.07 M total / 18.99 M trainable):

* **FULL** (`model_final.pt`) -- `{"model": state_dict, "meta": {...}}`. What `planner.py` has
  always loaded (strict). Self-contained; ~1.3 GB for ViT-L.
* **PARTIAL** (`ckpt_last.pt`, `snap_epochNNN.pt`) -- only what training changes (every
  `requires_grad` parameter) plus every buffer, ~76 MB for ViT-L. The frozen DINOv3 trunk is
  rebuilt from its published weights, and a SHA-256 FINGERPRINT of the frozen tensors travels
  with the file so a partial can never be silently completed with a DIFFERENT trunk.

⛔ A partial load is admitted only on a POSITIVE assertion: the keys it did not fill must be
EXACTLY the frozen parameters, it may carry no key the model lacks, and the fingerprint must
match. "It loaded without error" is not the test -- `strict=False` loads anything.
"""
from __future__ import annotations

import hashlib
import os

import torch

FORMAT_PARTIAL = "refe-partial-v1"
FORMAT_FULL = "refe-full-v1"


def trainable_names(model) -> set:
    return {n for n, p in model.named_parameters() if p.requires_grad}


def frozen_names(model) -> set:
    return {n for n, p in model.named_parameters() if not p.requires_grad}


def partial_state(model) -> dict:
    """Every trainable parameter and every persistent buffer, detached, on CPU."""
    keep = trainable_names(model) | {n for n, _ in model.named_buffers()}
    return {k: v.detach().to("cpu", copy=True)
            for k, v in model.state_dict().items() if k in keep}


def full_state(model) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def frozen_fingerprint(model) -> str:
    """SHA-256 over every FROZEN parameter's name and raw bytes, in name order.

    Byte-exact by construction (no reduction, so no summation-order dependence); ~1-2 s for the
    ViT-L trunk. Computed once per process -- frozen tensors do not change.
    """
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters(), key=lambda t: t[0]):
        if p.requires_grad:
            continue
        h.update(n.encode("utf-8"))
        t = p.detach().to("cpu").contiguous().reshape(-1)
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_partial(model, partial: dict, expect_frozen_sha256: str | None = None) -> int:
    """Apply a PARTIAL state to a model whose frozen trunk is already loaded. Returns n tensors.

    Raises ValueError unless the unfilled keys are exactly the frozen parameters, no key is
    unexpected, and (when given) the frozen fingerprint matches.
    """
    missing, unexpected = model.load_state_dict(partial, strict=False)
    frozen = frozen_names(model)
    bad = []
    if unexpected:
        bad.append(f"{len(unexpected)} key(s) the model does not have, e.g. {list(unexpected)[:3]}")
    extra_missing = sorted(set(missing) - frozen)
    if extra_missing:
        bad.append(f"{len(extra_missing)} TRAINED key(s) absent from the file, "
                   f"e.g. {extra_missing[:3]}")
    if expect_frozen_sha256 is not None:
        got = frozen_fingerprint(model)
        if got != expect_frozen_sha256:
            bad.append(f"frozen-trunk fingerprint {got[:12]} != recorded {expect_frozen_sha256[:12]}"
                       f" -- this partial was trained on a DIFFERENT trunk")
    if bad:
        raise ValueError("partial checkpoint refused: " + "; ".join(bad))
    return len(partial)


def atomic_save(obj, path) -> int:
    """torch.save to `<path>.tmp`, then an atomic rename. Returns the byte size written.

    A kill mid-write leaves the previous file intact instead of a truncated one.
    """
    path = str(path)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    n = os.path.getsize(tmp)
    if n <= 0:
        raise OSError(f"atomic_save: {tmp} is empty after torch.save")
    os.replace(tmp, path)
    return n


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_for_inference(model, path: str, map_location="cpu", backbone: str | None = None,
                       meta_out: dict | None = None) -> str:
    """Load EITHER format into `model` (built by REFe(cfg)). Returns the format name.

    FULL: strict load, exactly what planner.py always did. PARTIAL: load the DINOv3 trunk the
    snapshot names (falling back to the default backbone root), then `load_partial` with the
    recorded fingerprint -- so an epoch snapshot is evaluable without a 1.3 GB full copy.
    `meta_out`, if given, receives the file's meta (e.g. `per_sample_calib`, which the planner
    must honour: a model trained on per-sample rigs must be FED per-sample rigs).
    """
    sd = torch.load(path, map_location=map_location, weights_only=False)
    if meta_out is not None and isinstance(sd, dict):
        meta_out.update(sd.get("meta", {}) or {})
    fmt = sd.get("format") if isinstance(sd, dict) else None
    if fmt == FORMAT_PARTIAL or (isinstance(sd, dict) and "model_partial" in sd):
        import load_dinov3 as LD
        from model import BACKBONES
        meta = sd.get("meta", {}) or {}
        wdir = meta.get("weights_dir")
        if not wdir or not os.path.isdir(wdir):
            bb = backbone or meta.get("backbone")
            wdir = os.path.join(LD.BACKBONE_ROOT, BACKBONES[bb]["weights"])
        loaded, missing, unused = LD.map_into_backbone(model.backbone, LD.load_state(wdir))
        if not loaded or missing or unused:
            raise ValueError(f"trunk load for a partial snapshot failed: loaded={len(loaded)} "
                             f"missing={len(missing)} unused={len(unused)} from {wdir}")
        load_partial(model, sd["model_partial"], meta.get("frozen_sha256"))
        return FORMAT_PARTIAL
    model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    return FORMAT_FULL
